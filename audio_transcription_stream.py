
from __future__ import annotations

import audioop
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import queue
import shutil
import tempfile
import time
import wave
from typing import ClassVar

from pydantic import BaseModel, Field, field_validator

from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from vibe.core.types import ToolCallEvent, ToolResultEvent


_MODEL_CACHE: dict[tuple[str, str, str, str, str], object] = {}


@dataclass(frozen=True)
class _CaptureResult:
    speech_frames: list[bytes]
    all_frames: list[bytes]
    speech_start_frame: int
    speech_end_frame: int
    total_frames: int
    duration_sec: float
    partials: list["PartialTranscript"]


class AudioTranscriptionStreamArgs(BaseModel):
    capture_backend: str | None = Field(
        default=None,
        description="Capture backend: auto, sounddevice, or pyaudio.",
    )
    duration_sec: float | None = Field(
        default=None, description="Fixed duration when VAD is disabled."
    )
    max_duration_sec: float | None = Field(
        default=None, description="Hard stop for streaming capture."
    )
    sample_rate: int | None = Field(default=None, description="Sample rate in Hz.")
    channels: int | None = Field(default=None, description="Number of channels.")
    device: str | None = Field(
        default=None, description="Input device name or index."
    )
    vad: bool = Field(
        default=True, description="Enable voice activity detection."
    )
    vad_mode: int = Field(
        default=2, description="VAD aggressiveness (0-3)."
    )
    frame_ms: int = Field(
        default=30, description="Frame size for VAD/capture (10/20/30 ms)."
    )
    silence_timeout_ms: int = Field(
        default=800, description="Silence timeout once speech starts."
    )
    leading_silence_ms: int = Field(
        default=2000, description="Stop if speech never starts."
    )
    pre_speech_ms: int = Field(
        default=200, description="Pre-roll kept before speech start."
    )
    energy_threshold: float = Field(
        default=0.01, description="RMS threshold when VAD unavailable."
    )
    partial_interval_ms: int = Field(
        default=1000, description="Interval for partial transcription (0 disables)."
    )
    return_partials: bool = Field(
        default=True, description="Return partial transcription updates."
    )
    save_audio: bool = Field(
        default=False, description="Save captured audio to WAV."
    )
    output_dir: str | None = Field(
        default=None, description="Output directory for saved audio."
    )
    output_name: str | None = Field(
        default=None, description="Output filename for saved audio."
    )
    backend: str | None = Field(
        default=None, description="STT backend: auto, faster_whisper, or whisper."
    )
    model: str | None = Field(
        default=None, description="STT model name or path."
    )
    language: str | None = Field(
        default=None, description="Optional language code."
    )
    task: str | None = Field(
        default=None, description="Task: transcribe or translate."
    )
    beam_size: int | None = Field(
        default=None, description="Override beam size."
    )
    temperature: float | None = Field(
        default=None, description="Override decoding temperature."
    )
    vad_filter: bool | None = Field(
        default=None, description="Apply model VAD when supported."
    )
    word_timestamps: bool | None = Field(
        default=None, description="Return word timestamps when supported."
    )


class AudioSegment(BaseModel):
    start: float
    end: float
    text: str


class PartialTranscript(BaseModel):
    sequence: int
    timestamp_sec: float
    text: str
    final: bool = Field(default=False)


class AudioTranscriptionStreamResult(BaseModel):
    text: str
    language: str | None
    segments: list[AudioSegment] | None
    partials: list[PartialTranscript]
    audio_path: str | None
    duration_sec: float
    backend: str
    model: str


class AudioTranscriptionStreamConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    default_capture_backend: str = "auto"
    default_duration_sec: float = 6.0
    max_duration_sec: float = 30.0
    default_sample_rate: int = 16000
    default_channels: int = 1
    output_dir: Path = Field(default=Path.home() / ".vibe" / "audio")
    backend: str = Field(
        default="auto",
        description="Transcription backend: auto, faster_whisper, or whisper.",
    )
    model_name: str = Field(default="small", description="Model name or path.")
    device: str = Field(default="auto", description="cpu, cuda, or auto.")
    compute_type: str = Field(
        default="int8",
        description="Compute type for faster-whisper.",
    )
    download_root: Path | None = Field(
        default=None, description="Optional model cache directory."
    )
    beam_size: int = Field(default=5, description="Beam size for decoding.")
    temperature: float = Field(default=0.0, description="Decoding temperature.")
    vad_filter: bool = Field(
        default=False, description="Apply model VAD filter when supported."
    )
    word_timestamps: bool = Field(
        default=False, description="Return word timestamps when supported."
    )
    max_audio_bytes: int = Field(
        default=100_000_000, description="Maximum audio size in bytes."
    )

    @field_validator("output_dir", mode="before")
    @classmethod
    def set_default_output_dir(cls, v: Path | str) -> Path:
        if isinstance(v, Path):
            return v
        if not v or not str(v).strip():
            return Path.home() / ".vibe" / "audio"
        return Path(v)

    @field_validator("output_dir", mode="after")
    @classmethod
    def expand_output_dir(cls, v: Path) -> Path:
        return v.expanduser().resolve()


class AudioTranscriptionStreamState(BaseToolState):
    pass


class AudioTranscriptionStream(
    BaseTool[
        AudioTranscriptionStreamArgs,
        AudioTranscriptionStreamResult,
        AudioTranscriptionStreamConfig,
        AudioTranscriptionStreamState,
    ],
    ToolUIData[AudioTranscriptionStreamArgs, AudioTranscriptionStreamResult],
):
    description: ClassVar[str] = (
        "Stream microphone audio with VAD and emit partial/final transcription."
    )

    async def run(self, args: AudioTranscriptionStreamArgs) -> AudioTranscriptionStreamResult:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        self._ensure_ffmpeg()

        sample_rate = args.sample_rate or self.config.default_sample_rate
        channels = args.channels or self.config.default_channels
        if sample_rate <= 0:
            raise ToolError("sample_rate must be positive.")
        if channels <= 0:
            raise ToolError("channels must be positive.")

        capture_backend = self._resolve_capture_backend(args.capture_backend)
        frame_ms = int(args.frame_ms)
        if args.vad and frame_ms not in {10, 20, 30}:
            raise ToolError("frame_ms must be 10, 20, or 30 when VAD is enabled.")
        if frame_ms <= 0:
            raise ToolError("frame_ms must be positive.")

        duration_sec = (
            args.duration_sec
            if args.duration_sec is not None
            else self.config.default_duration_sec
        )
        max_duration_sec = (
            args.max_duration_sec
            if args.max_duration_sec is not None
            else self.config.max_duration_sec
        )
        if duration_sec <= 0:
            raise ToolError("duration_sec must be positive.")
        if max_duration_sec <= 0:
            raise ToolError("max_duration_sec must be positive.")

        if capture_backend == "sounddevice":
            capture = self._capture_sounddevice(
                sample_rate,
                channels,
                frame_ms,
                duration_sec,
                max_duration_sec,
                args,
            )
        else:
            capture = self._capture_pyaudio(
                sample_rate,
                channels,
                frame_ms,
                duration_sec,
                max_duration_sec,
                args,
            )

        if not capture.speech_frames:
            raise ToolError("No speech detected in captured audio.")

        audio_path = None
        if args.save_audio:
            audio_path = self._write_full_audio(
                capture.all_frames, sample_rate, channels, args
            )

        backend = self._resolve_backend(args.backend)
        model_name = args.model or self.config.model_name
        language = self._normalize_language(args.language)
        task = self._normalize_task(args.task)
        beam_size = (
            args.beam_size if args.beam_size is not None else self.config.beam_size
        )
        temperature = (
            args.temperature
            if args.temperature is not None
            else self.config.temperature
        )
        vad_filter = (
            args.vad_filter
            if args.vad_filter is not None
            else self.config.vad_filter
        )
        word_timestamps = (
            args.word_timestamps
            if args.word_timestamps is not None
            else self.config.word_timestamps
        )

        text, segments, detected_language = self._transcribe_bytes(
            capture.speech_frames,
            sample_rate,
            channels,
            backend,
            model_name,
            language,
            task,
            beam_size,
            temperature,
            vad_filter,
            word_timestamps,
        )

        speech_start_sec = capture.speech_start_frame / float(sample_rate)
        if segments is not None:
            segments = [
                AudioSegment(
                    start=segment.start + speech_start_sec,
                    end=segment.end + speech_start_sec,
                    text=segment.text,
                )
                for segment in segments
            ]

        partials = list(capture.partials)
        if args.return_partials:
            if text:
                if partials:
                    last = partials[-1]
                    if last.text.strip() == text.strip():
                        partials[-1] = PartialTranscript(
                            sequence=last.sequence,
                            timestamp_sec=last.timestamp_sec,
                            text=last.text,
                            final=True,
                        )
                    else:
                        partials.append(
                            PartialTranscript(
                                sequence=last.sequence + 1,
                                timestamp_sec=capture.duration_sec,
                                text=text,
                                final=True,
                            )
                        )
                else:
                    partials.append(
                        PartialTranscript(
                            sequence=1,
                            timestamp_sec=capture.duration_sec,
                            text=text,
                            final=True,
                        )
                    )
        else:
            partials = []

        return AudioTranscriptionStreamResult(
            text=text,
            language=language or detected_language,
            segments=segments,
            partials=partials,
            audio_path=str(audio_path) if audio_path else None,
            duration_sec=capture.duration_sec,
            backend=backend,
            model=model_name,
        )
    def _capture_sounddevice(
        self,
        sample_rate: int,
        channels: int,
        frame_ms: int,
        duration_sec: float,
        max_duration_sec: float,
        args: AudioTranscriptionStreamArgs,
    ) -> _CaptureResult:
        try:
            import numpy as np
            import sounddevice as sd
        except ModuleNotFoundError as exc:
            raise ToolError("sounddevice is not installed.") from exc

        blocksize = int(sample_rate * frame_ms / 1000)
        if blocksize <= 0:
            raise ToolError("Invalid blocksize for capture settings.")

        q: queue.Queue = queue.Queue()
        all_frames: list[bytes] = []
        speech_frames: list[bytes] = []
        pre_frames: list[bytes] = []
        pre_frames_samples = 0
        partials: list[PartialTranscript] = []
        last_partial_text = ""

        def callback(indata, frames_count, time_info, status):
            del frames_count, time_info, status
            q.put(indata.copy())

        vad = self._create_vad(args.vad, args.vad_mode)
        speech_started = not args.vad
        last_voice = time.monotonic()
        start = last_voice
        stop_at = start + (max_duration_sec if args.vad else duration_sec)
        leading_silence_sec = args.leading_silence_ms / 1000.0
        silence_timeout_sec = args.silence_timeout_ms / 1000.0
        pre_speech_frames = int(sample_rate * args.pre_speech_ms / 1000.0)
        total_frames = 0
        speech_start_frame = 0
        speech_end_frame = 0
        last_partial_time = start
        sequence = 0

        with sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="int16",
            blocksize=blocksize,
            device=args.device,
            callback=callback,
        ):
            while True:
                now = time.monotonic()
                if now >= stop_at:
                    break
                try:
                    chunk = q.get(timeout=0.5)
                except queue.Empty:
                    continue

                frame_count = int(chunk.shape[0])
                current_start_frame = total_frames
                total_frames += frame_count
                chunk_bytes = chunk.tobytes()
                all_frames.append(chunk_bytes)

                is_voice = self._is_voice_numpy(
                    chunk, sample_rate, vad, args.energy_threshold
                )

                if args.vad:
                    if is_voice:
                        if not speech_started:
                            speech_started = True
                            speech_start_frame = max(
                                0, current_start_frame - pre_frames_samples
                            )
                            speech_frames.extend(pre_frames)
                            pre_frames.clear()
                            pre_frames_samples = 0
                        last_voice = now
                        speech_end_frame = total_frames
                        speech_frames.append(chunk_bytes)
                    else:
                        if speech_started:
                            if (now - last_voice) >= silence_timeout_sec:
                                break
                        else:
                            pre_frames.append(chunk_bytes)
                            pre_frames_samples += frame_count
                            while (
                                pre_frames_samples > pre_speech_frames
                                and pre_frames
                            ):
                                dropped = pre_frames.pop(0)
                                pre_frames_samples -= int(len(dropped) / (2 * channels))
                            if (now - start) >= leading_silence_sec:
                                break
                else:
                    if not speech_started:
                        speech_started = True
                        speech_start_frame = 0
                    speech_end_frame = total_frames
                    speech_frames.append(chunk_bytes)

                if (
                    args.return_partials
                    and args.partial_interval_ms > 0
                    and speech_started
                    and (now - last_partial_time)
                    >= (args.partial_interval_ms / 1000.0)
                ):
                    partial_text = self._transcribe_partial(
                        speech_frames, sample_rate, channels, args
                    )
                    if partial_text and partial_text != last_partial_text:
                        sequence += 1
                        partials.append(
                            PartialTranscript(
                                sequence=sequence,
                                timestamp_sec=total_frames / float(sample_rate),
                                text=partial_text,
                                final=False,
                            )
                        )
                        last_partial_text = partial_text
                    last_partial_time = now

        if not all_frames:
            raise ToolError("No audio captured.")

        if not speech_started:
            speech_start_frame = 0
            speech_end_frame = total_frames

        duration = total_frames / float(sample_rate)
        return _CaptureResult(
            speech_frames=speech_frames,
            all_frames=all_frames,
            speech_start_frame=speech_start_frame,
            speech_end_frame=speech_end_frame or total_frames,
            total_frames=total_frames,
            duration_sec=duration,
            partials=partials,
        )

    def _capture_pyaudio(
        self,
        sample_rate: int,
        channels: int,
        frame_ms: int,
        duration_sec: float,
        max_duration_sec: float,
        args: AudioTranscriptionStreamArgs,
    ) -> _CaptureResult:
        try:
            import pyaudio
        except ModuleNotFoundError as exc:
            raise ToolError("pyaudio is not installed.") from exc

        blocksize = int(sample_rate * frame_ms / 1000)
        if blocksize <= 0:
            raise ToolError("Invalid blocksize for capture settings.")

        vad = self._create_vad(args.vad, args.vad_mode)
        speech_started = not args.vad
        last_voice = time.monotonic()
        start = last_voice
        stop_at = start + (max_duration_sec if args.vad else duration_sec)
        leading_silence_sec = args.leading_silence_ms / 1000.0
        silence_timeout_sec = args.silence_timeout_ms / 1000.0
        pre_speech_frames = int(sample_rate * args.pre_speech_ms / 1000.0)

        audio = pyaudio.PyAudio()
        all_frames: list[bytes] = []
        speech_frames: list[bytes] = []
        pre_frames: list[bytes] = []
        pre_frames_samples = 0
        partials: list[PartialTranscript] = []
        last_partial_text = ""
        total_frames = 0
        speech_start_frame = 0
        speech_end_frame = 0
        last_partial_time = start
        sequence = 0

        try:
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=blocksize,
                input_device_index=self._parse_device_index(args.device),
            )
        except Exception as exc:
            audio.terminate()
            raise ToolError(f"pyaudio open failed: {exc}") from exc

        try:
            while True:
                now = time.monotonic()
                if now >= stop_at:
                    break
                data = stream.read(blocksize, exception_on_overflow=False)
                all_frames.append(data)
                frame_count = int(len(data) / (2 * channels))
                current_start_frame = total_frames
                total_frames += frame_count

                is_voice = self._is_voice_bytes(
                    data, sample_rate, channels, vad, args.energy_threshold
                )

                if args.vad:
                    if is_voice:
                        if not speech_started:
                            speech_started = True
                            speech_start_frame = max(
                                0, current_start_frame - pre_frames_samples
                            )
                            speech_frames.extend(pre_frames)
                            pre_frames.clear()
                            pre_frames_samples = 0
                        last_voice = now
                        speech_end_frame = total_frames
                        speech_frames.append(data)
                    else:
                        if speech_started:
                            if (now - last_voice) >= silence_timeout_sec:
                                break
                        else:
                            pre_frames.append(data)
                            pre_frames_samples += frame_count
                            while (
                                pre_frames_samples > pre_speech_frames
                                and pre_frames
                            ):
                                dropped = pre_frames.pop(0)
                                pre_frames_samples -= int(len(dropped) / (2 * channels))
                            if (now - start) >= leading_silence_sec:
                                break
                else:
                    if not speech_started:
                        speech_started = True
                        speech_start_frame = 0
                    speech_end_frame = total_frames
                    speech_frames.append(data)

                if (
                    args.return_partials
                    and args.partial_interval_ms > 0
                    and speech_started
                    and (now - last_partial_time)
                    >= (args.partial_interval_ms / 1000.0)
                ):
                    partial_text = self._transcribe_partial(
                        speech_frames, sample_rate, channels, args
                    )
                    if partial_text and partial_text != last_partial_text:
                        sequence += 1
                        partials.append(
                            PartialTranscript(
                                sequence=sequence,
                                timestamp_sec=total_frames / float(sample_rate),
                                text=partial_text,
                                final=False,
                            )
                        )
                        last_partial_text = partial_text
                    last_partial_time = now
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

        if not all_frames:
            raise ToolError("No audio captured.")

        if not speech_started:
            speech_start_frame = 0
            speech_end_frame = total_frames

        duration = total_frames / float(sample_rate)
        return _CaptureResult(
            speech_frames=speech_frames,
            all_frames=all_frames,
            speech_start_frame=speech_start_frame,
            speech_end_frame=speech_end_frame or total_frames,
            total_frames=total_frames,
            duration_sec=duration,
            partials=partials,
        )
    def _transcribe_partial(
        self,
        frames: list[bytes],
        sample_rate: int,
        channels: int,
        args: AudioTranscriptionStreamArgs,
    ) -> str:
        backend = self._resolve_backend(args.backend)
        model_name = args.model or self.config.model_name
        language = self._normalize_language(args.language)
        task = self._normalize_task(args.task)
        beam_size = (
            args.beam_size if args.beam_size is not None else self.config.beam_size
        )
        temperature = (
            args.temperature
            if args.temperature is not None
            else self.config.temperature
        )
        vad_filter = (
            args.vad_filter
            if args.vad_filter is not None
            else self.config.vad_filter
        )
        return self._transcribe_bytes(
            frames,
            sample_rate,
            channels,
            backend,
            model_name,
            language,
            task,
            beam_size,
            temperature,
            vad_filter,
            False,
        )[0]

    def _write_full_audio(
        self,
        frames: list[bytes],
        sample_rate: int,
        channels: int,
        args: AudioTranscriptionStreamArgs,
    ) -> Path:
        output_dir = self.config.output_dir
        if args.output_dir:
            path = Path(args.output_dir).expanduser()
            if not path.is_absolute():
                path = self.config.effective_workdir / path
            output_dir = path
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.output_name:
            filename = args.output_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stream_capture_{timestamp}.wav"
        path = output_dir / filename
        if path.suffix.lower() != ".wav":
            path = path.with_suffix(".wav")
        path = path.resolve()

        self._write_wav_bytes(path, frames, sample_rate, channels)
        size = path.stat().st_size
        self._enforce_size_limit(path, size)
        return path

    def _write_wav_bytes(
        self, path: Path, frames: list[bytes], sample_rate: int, channels: int
    ) -> None:
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            for chunk in frames:
                wf.writeframes(chunk)

    def _transcribe_bytes(
        self,
        frames: list[bytes],
        sample_rate: int,
        channels: int,
        backend: str,
        model_name: str,
        language: str | None,
        task: str,
        beam_size: int,
        temperature: float,
        vad_filter: bool,
        word_timestamps: bool,
    ) -> tuple[str, list[AudioSegment], str | None]:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_path = Path(temp.name)
        temp.close()
        try:
            self._write_wav_bytes(temp_path, frames, sample_rate, channels)
            if backend == "faster_whisper":
                return self._transcribe_faster_whisper(
                    temp_path,
                    model_name,
                    language,
                    task,
                    beam_size,
                    temperature,
                    vad_filter,
                    word_timestamps,
                )
            return self._transcribe_whisper(
                temp_path,
                model_name,
                language,
                task,
                beam_size,
                temperature,
                word_timestamps,
            )
        finally:
            try:
                temp_path.unlink()
            except OSError:
                pass

    def _transcribe_faster_whisper(
        self,
        audio_path: Path,
        model_name: str,
        language: str | None,
        task: str,
        beam_size: int,
        temperature: float,
        vad_filter: bool,
        word_timestamps: bool,
    ) -> tuple[str, list[AudioSegment], str | None]:
        model = self._get_faster_whisper_model(model_name)
        options = {
            "language": language,
            "task": task,
            "beam_size": beam_size,
            "temperature": temperature,
        }
        if vad_filter:
            options["vad_filter"] = True
        if word_timestamps:
            options["word_timestamps"] = True
        options = {k: v for k, v in options.items() if v is not None}
        try:
            segments_iter, info = model.transcribe(str(audio_path), **options)
        except Exception as exc:
            raise ToolError(f"faster-whisper transcription failed: {exc}") from exc

        segments: list[AudioSegment] = []
        text_chunks: list[str] = []
        for segment in segments_iter:
            text_chunks.append(segment.text)
            segments.append(
                AudioSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=segment.text.strip(),
                )
            )
        text = "".join(text_chunks).strip()
        detected_language = getattr(info, "language", None)
        return text, segments, detected_language

    def _transcribe_whisper(
        self,
        audio_path: Path,
        model_name: str,
        language: str | None,
        task: str,
        beam_size: int,
        temperature: float,
        word_timestamps: bool,
    ) -> tuple[str, list[AudioSegment], str | None]:
        model = self._get_whisper_model(model_name)
        options = {
            "language": language,
            "task": task,
            "temperature": temperature,
            "beam_size": beam_size,
        }
        if word_timestamps:
            options["word_timestamps"] = True
        options = {k: v for k, v in options.items() if v is not None}
        if self.config.device.strip().lower() == "cpu":
            options["fp16"] = False
        try:
            result = model.transcribe(str(audio_path), **options)
        except Exception as exc:
            raise ToolError(f"whisper transcription failed: {exc}") from exc

        segments_raw = result.get("segments") or []
        segments = [
            AudioSegment(
                start=float(segment.get("start", 0.0)),
                end=float(segment.get("end", 0.0)),
                text=str(segment.get("text", "")).strip(),
            )
            for segment in segments_raw
        ]
        text = str(result.get("text", "")).strip()
        detected_language = result.get("language")
        return text, segments, detected_language

    def _get_faster_whisper_model(self, model_name: str) -> object:
        key = self._model_cache_key("faster_whisper", model_name)
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            from faster_whisper import WhisperModel
        except ModuleNotFoundError as exc:
            raise ToolError("faster-whisper is not installed.") from exc
        device = self._resolve_device_for_faster_whisper()
        download_root = self._resolve_download_root()
        try:
            model = WhisperModel(
                model_name,
                device=device,
                compute_type=self.config.compute_type,
                download_root=download_root,
            )
        except Exception as exc:
            raise ToolError(f"Failed to load faster-whisper model: {exc}") from exc
        _MODEL_CACHE[key] = model
        return model

    def _get_whisper_model(self, model_name: str) -> object:
        key = self._model_cache_key("whisper", model_name)
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            import whisper
        except ModuleNotFoundError as exc:
            raise ToolError("whisper is not installed.") from exc
        download_root = self._resolve_download_root()
        device = self._resolve_device_for_whisper()
        try:
            model = whisper.load_model(
                model_name, device=device, download_root=download_root
            )
        except Exception as exc:
            raise ToolError(f"Failed to load whisper model: {exc}") from exc
        _MODEL_CACHE[key] = model
        return model
    def _resolve_download_root(self) -> str | None:
        if self.config.download_root is None:
            return None
        return str(self.config.download_root.expanduser())

    def _resolve_backend(self, override: str | None) -> str:
        backend = (override or self.config.backend or "auto").strip().lower()
        if backend not in {"auto", "faster_whisper", "whisper"}:
            raise ToolError("backend must be auto, faster_whisper, or whisper.")
        if backend == "auto":
            if self._module_available("faster_whisper"):
                return "faster_whisper"
            if self._module_available("whisper"):
                return "whisper"
            raise ToolError(
                "No transcription backend found. Install faster-whisper or whisper."
            )
        if backend == "faster_whisper" and not self._module_available("faster_whisper"):
            raise ToolError("faster-whisper is not installed.")
        if backend == "whisper" and not self._module_available("whisper"):
            raise ToolError("whisper is not installed.")
        return backend

    def _resolve_capture_backend(self, override: str | None) -> str:
        backend = (override or self.config.default_capture_backend or "auto").strip().lower()
        if backend not in {"auto", "sounddevice", "pyaudio"}:
            raise ToolError("capture_backend must be auto, sounddevice, or pyaudio.")
        if backend == "auto":
            if self._module_available("sounddevice"):
                return "sounddevice"
            if self._module_available("pyaudio"):
                return "pyaudio"
            raise ToolError("No audio capture backend available.")
        if backend == "sounddevice" and not self._module_available("sounddevice"):
            raise ToolError("sounddevice is not installed.")
        if backend == "pyaudio" and not self._module_available("pyaudio"):
            raise ToolError("pyaudio is not installed.")
        return backend

    def _resolve_device_for_faster_whisper(self) -> str:
        device = self.config.device.strip().lower()
        if device == "auto":
            return "cuda" if self._cuda_available() else "cpu"
        if device in {"cpu", "cuda"}:
            return device
        raise ToolError("device must be cpu, cuda, or auto.")

    def _resolve_device_for_whisper(self) -> str | None:
        device = self.config.device.strip().lower()
        if device == "auto":
            return None
        if device in {"cpu", "cuda"}:
            return device
        raise ToolError("device must be cpu, cuda, or auto.")

    def _cuda_available(self) -> bool:
        try:
            import ctranslate2
        except ModuleNotFoundError:
            return False
        try:
            return ctranslate2.get_cuda_device_count() > 0
        except Exception:
            return False

    def _model_cache_key(
        self, backend: str, model_name: str
    ) -> tuple[str, str, str, str, str]:
        download_root = self._resolve_download_root() or ""
        device = self.config.device.strip().lower()
        compute_type = self.config.compute_type.strip().lower()
        return (backend, model_name, device, download_root, compute_type)

    def _normalize_language(self, language: str | None) -> str | None:
        if language is None:
            return None
        value = language.strip()
        return value or None

    def _normalize_task(self, task: str | None) -> str:
        value = (task or "transcribe").strip().lower()
        if value not in {"transcribe", "translate"}:
            raise ToolError("task must be transcribe or translate.")
        return value

    def _create_vad(self, enabled: bool, mode: int):
        if not enabled:
            return None
        try:
            import webrtcvad
        except ModuleNotFoundError:
            return None
        mode = max(0, min(3, int(mode)))
        return webrtcvad.Vad(mode)

    def _is_voice_numpy(self, chunk, sample_rate: int, vad, energy_threshold: float) -> bool:
        import numpy as np

        if chunk.size == 0:
            return False
        if chunk.ndim > 1:
            data = chunk.mean(axis=1)
        else:
            data = chunk
        data = data.astype(np.int16, copy=False)

        if vad is not None:
            frame_bytes = data.tobytes()
            return vad.is_speech(frame_bytes, sample_rate)

        rms = np.sqrt(np.mean((data.astype(np.float32) / 32768.0) ** 2))
        return rms >= energy_threshold

    def _is_voice_bytes(
        self,
        data: bytes,
        sample_rate: int,
        channels: int,
        vad,
        energy_threshold: float,
    ) -> bool:
        if not data:
            return False
        if channels > 1:
            try:
                data = audioop.tomono(data, 2, 1.0, 1.0)
            except audioop.error:
                pass
        if vad is not None:
            return vad.is_speech(data, sample_rate)
        rms = audioop.rms(data, 2)
        return (rms / 32768.0) >= energy_threshold

    def _parse_device_index(self, device: str | None) -> int | None:
        if device is None:
            return None
        try:
            return int(device)
        except ValueError:
            return None

    def _enforce_size_limit(self, path: Path, size: int) -> None:
        max_bytes = self.config.max_audio_bytes
        if max_bytes > 0 and size > max_bytes:
            try:
                path.unlink()
            except OSError:
                pass
            raise ToolError(
                f"Audio is {size} bytes, exceeds {max_bytes} bytes."
            )

    def _module_available(self, module: str) -> bool:
        try:
            __import__(module)
            return True
        except ModuleNotFoundError:
            return False
        except Exception:
            return False

    def _ensure_ffmpeg(self) -> None:
        if shutil.which("ffmpeg") is None:
            raise ToolError(
                "ffmpeg is required for transcription. Install ffmpeg and add it to PATH."
            )

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, AudioTranscriptionStreamArgs):
            return ToolCallDisplay(summary="audio_transcription_stream")
        return ToolCallDisplay(
            summary="audio_transcription_stream",
            details={
                "capture_backend": event.args.capture_backend,
                "backend": event.args.backend,
                "model": event.args.model,
                "vad": event.args.vad,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, AudioTranscriptionStreamResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=True,
            message="Streaming transcription complete",
            details={
                "text": event.result.text,
                "language": event.result.language,
                "segments": event.result.segments,
                "partials": event.result.partials,
                "audio_path": event.result.audio_path,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Streaming transcription"
