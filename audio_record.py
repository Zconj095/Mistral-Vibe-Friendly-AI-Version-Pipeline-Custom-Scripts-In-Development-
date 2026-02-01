from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import audioop
import queue
import shutil
import subprocess
import sys
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


@dataclass(frozen=True)
class RecordStats:
    duration_sec: float
    frame_count: int
    sample_rate: int
    channels: int
    backend: str


class AudioRecordArgs(BaseModel):
    path: str | None = Field(
        default=None, description="Optional output path for the recording."
    )
    duration_sec: float | None = Field(
        default=None,
        description="Fixed recording duration in seconds (defaults to config).",
    )
    max_duration_sec: float | None = Field(
        default=None,
        description="Hard stop in seconds when using VAD or no duration specified.",
    )
    sample_rate: int | None = Field(
        default=None, description="Sample rate in Hz."
    )
    channels: int | None = Field(default=None, description="Number of channels.")
    backend: str | None = Field(
        default=None,
        description="Recording backend: auto, sounddevice, pyaudio, or ffmpeg.",
    )
    vad: bool = Field(
        default=False,
        description="Enable voice activity detection when supported.",
    )
    vad_mode: int = Field(
        default=2,
        description="VAD aggressiveness (0-3) when webrtcvad is available.",
    )
    silence_timeout_ms: int = Field(
        default=800,
        description="Stop after this much silence once speech has started.",
    )
    leading_silence_ms: int = Field(
        default=2000,
        description="Stop if speech never starts within this window.",
    )
    energy_threshold: float = Field(
        default=0.01,
        description="Fallback RMS threshold when webrtcvad is unavailable.",
    )
    device: str | None = Field(
        default=None,
        description="Input device name or index for sounddevice/pyaudio.",
    )
    ffmpeg_device: str | None = Field(
        default=None,
        description="FFmpeg device name (platform specific).",
    )
    ffmpeg_format: str | None = Field(
        default=None,
        description="FFmpeg input format (dshow, avfoundation, alsa).",
    )


class AudioRecordResult(BaseModel):
    path: str
    bytes_written: int
    duration_sec: float
    sample_rate: int
    channels: int
    backend: str


class AudioRecordConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    output_dir: Path = Field(default=Path.home() / ".vibe" / "audio")
    default_backend: str = "auto"
    default_duration_sec: float = 6.0
    max_duration_sec: float = 30.0
    default_sample_rate: int = 16000
    default_channels: int = 1
    max_audio_bytes: int = 100_000_000
    ffmpeg_device: str | None = None
    ffmpeg_format: str | None = None

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


class AudioRecordState(BaseToolState):
    pass


class AudioRecord(
    BaseTool[AudioRecordArgs, AudioRecordResult, AudioRecordConfig, AudioRecordState],
    ToolUIData[AudioRecordArgs, AudioRecordResult],
):
    description: ClassVar[str] = (
        "Record audio from the microphone to a WAV file (offline)."
    )

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, AudioRecordArgs):
            return ToolCallDisplay(summary="audio_record")
        return ToolCallDisplay(
            summary="Recording audio",
            details={
                "path": event.args.path,
                "duration_sec": event.args.duration_sec,
                "backend": event.args.backend,
                "vad": event.args.vad,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if isinstance(event.result, AudioRecordResult):
            return ToolResultDisplay(
                success=True,
                message=f"Recorded audio to {Path(event.result.path).name}",
                details={
                    "path": event.result.path,
                    "bytes_written": event.result.bytes_written,
                    "duration_sec": event.result.duration_sec,
                    "sample_rate": event.result.sample_rate,
                    "channels": event.result.channels,
                    "backend": event.result.backend,
                },
            )
        return ToolResultDisplay(success=True, message="Audio recorded")

    @classmethod
    def get_status_text(cls) -> str:
        return "Recording audio"

    async def run(self, args: AudioRecordArgs) -> AudioRecordResult:
        output_path = self._resolve_output_path(args)
        backend = self._resolve_backend(args.backend)
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
        sample_rate = args.sample_rate or self.config.default_sample_rate
        channels = args.channels or self.config.default_channels

        if duration_sec <= 0:
            raise ToolError("duration_sec must be positive.")
        if max_duration_sec <= 0:
            raise ToolError("max_duration_sec must be positive.")
        if sample_rate <= 0:
            raise ToolError("sample_rate must be positive.")
        if channels <= 0:
            raise ToolError("channels must be positive.")

        if backend == "sounddevice":
            stats = self._record_sounddevice(
                output_path,
                duration_sec,
                max_duration_sec,
                sample_rate,
                channels,
                args,
            )
        elif backend == "pyaudio":
            stats = self._record_pyaudio(
                output_path,
                duration_sec,
                max_duration_sec,
                sample_rate,
                channels,
                args,
            )
        else:
            stats = self._record_ffmpeg(
                output_path,
                duration_sec,
                sample_rate,
                channels,
                args,
            )

        bytes_written = output_path.stat().st_size
        self._enforce_size_limit(output_path, bytes_written)

        return AudioRecordResult(
            path=str(output_path),
            bytes_written=bytes_written,
            duration_sec=stats.duration_sec,
            sample_rate=stats.sample_rate,
            channels=stats.channels,
            backend=stats.backend,
        )

    def _resolve_output_path(self, args: AudioRecordArgs) -> Path:
        if args.path:
            path = Path(args.path).expanduser()
            if not path.is_absolute():
                path = self.config.effective_workdir / path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audio_{timestamp}.wav"
            path = self.config.output_dir / filename

        if not path.suffix:
            path = path.with_suffix(".wav")
        elif path.suffix.lower() != ".wav":
            path = path.with_suffix(".wav")

        path.parent.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    def _resolve_backend(self, override: str | None) -> str:
        backend = (override or self.config.default_backend or "auto").strip().lower()
        if backend not in {"auto", "sounddevice", "pyaudio", "ffmpeg"}:
            raise ToolError("backend must be auto, sounddevice, pyaudio, or ffmpeg.")
        if backend == "auto":
            if self._module_available("sounddevice"):
                return "sounddevice"
            if self._module_available("pyaudio"):
                return "pyaudio"
            return "ffmpeg"
        if backend == "sounddevice" and not self._module_available("sounddevice"):
            raise ToolError("sounddevice is not installed.")
        if backend == "pyaudio" and not self._module_available("pyaudio"):
            raise ToolError("pyaudio is not installed.")
        return backend

    def _record_sounddevice(
        self,
        output_path: Path,
        duration_sec: float,
        max_duration_sec: float,
        sample_rate: int,
        channels: int,
        args: AudioRecordArgs,
    ) -> RecordStats:
        try:
            import numpy as np
            import sounddevice as sd
        except ModuleNotFoundError as exc:
            raise ToolError("sounddevice is not installed.") from exc

        frame_ms = 30
        blocksize = int(sample_rate * frame_ms / 1000)
        if blocksize <= 0:
            raise ToolError("Invalid blocksize for VAD settings.")

        q: queue.Queue = queue.Queue()
        frames: list[np.ndarray] = []

        def callback(indata, frames_count, time_info, status):
            del frames_count, time_info, status
            q.put(indata.copy())

        vad = self._create_vad(args.vad, args.vad_mode)
        speech_started = not args.vad
        last_voice = time.monotonic()
        start = last_voice
        stop_at = start + min(max_duration_sec, duration_sec if not args.vad else max_duration_sec)
        leading_silence_sec = args.leading_silence_ms / 1000.0
        silence_timeout_sec = args.silence_timeout_ms / 1000.0

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
                frames.append(chunk)

                if args.vad:
                    if self._is_voice_numpy(chunk, sample_rate, vad, args.energy_threshold):
                        speech_started = True
                        last_voice = now
                    elif speech_started and (now - last_voice) >= silence_timeout_sec:
                        break
                    elif not speech_started and (now - start) >= leading_silence_sec:
                        break

        if not frames:
            raise ToolError("No audio captured.")

        frame_count = sum(chunk.shape[0] for chunk in frames)
        duration = frame_count / float(sample_rate)
        self._write_wav_numpy(output_path, frames, sample_rate, channels)
        return RecordStats(
            duration_sec=duration,
            frame_count=frame_count,
            sample_rate=sample_rate,
            channels=channels,
            backend="sounddevice",
        )

    def _record_pyaudio(
        self,
        output_path: Path,
        duration_sec: float,
        max_duration_sec: float,
        sample_rate: int,
        channels: int,
        args: AudioRecordArgs,
    ) -> RecordStats:
        try:
            import pyaudio
        except ModuleNotFoundError as exc:
            raise ToolError("pyaudio is not installed.") from exc

        frame_ms = 30
        blocksize = int(sample_rate * frame_ms / 1000)
        if blocksize <= 0:
            raise ToolError("Invalid blocksize for VAD settings.")

        vad = self._create_vad(args.vad, args.vad_mode)
        speech_started = not args.vad
        last_voice = time.monotonic()
        start = last_voice
        stop_at = start + min(max_duration_sec, duration_sec if not args.vad else max_duration_sec)
        leading_silence_sec = args.leading_silence_ms / 1000.0
        silence_timeout_sec = args.silence_timeout_ms / 1000.0

        audio = pyaudio.PyAudio()
        frames: list[bytes] = []
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
                frames.append(data)

                if args.vad:
                    if self._is_voice_bytes(
                        data,
                        sample_rate,
                        channels,
                        vad,
                        args.energy_threshold,
                    ):
                        speech_started = True
                        last_voice = now
                    elif speech_started and (now - last_voice) >= silence_timeout_sec:
                        break
                    elif not speech_started and (now - start) >= leading_silence_sec:
                        break
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

        if not frames:
            raise ToolError("No audio captured.")

        frame_count = int(len(frames) * blocksize)
        duration = frame_count / float(sample_rate)
        self._write_wav_bytes(output_path, frames, sample_rate, channels)
        return RecordStats(
            duration_sec=duration,
            frame_count=frame_count,
            sample_rate=sample_rate,
            channels=channels,
            backend="pyaudio",
        )

    def _record_ffmpeg(
        self,
        output_path: Path,
        duration_sec: float,
        sample_rate: int,
        channels: int,
        args: AudioRecordArgs,
    ) -> RecordStats:
        if shutil.which("ffmpeg") is None:
            raise ToolError("ffmpeg is required for ffmpeg backend.")

        fmt = args.ffmpeg_format or self.config.ffmpeg_format or self._default_ffmpeg_format()
        device = args.ffmpeg_device or self.config.ffmpeg_device or "default"
        if fmt == "dshow":
            input_spec = f"audio={device}"
        else:
            input_spec = device

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            fmt,
            "-i",
            input_spec,
            "-t",
            f"{duration_sec:.3f}",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            str(output_path),
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            message = stderr or "ffmpeg recording failed."
            raise ToolError(message) from exc

        frame_count, duration = self._wav_stats(output_path)
        return RecordStats(
            duration_sec=duration,
            frame_count=frame_count,
            sample_rate=sample_rate,
            channels=channels,
            backend="ffmpeg",
        )

    def _default_ffmpeg_format(self) -> str:
        if sys.platform.startswith("win"):
            return "dshow"
        if sys.platform == "darwin":
            return "avfoundation"
        return "alsa"

    def _write_wav_numpy(
        self,
        path: Path,
        frames,
        sample_rate: int,
        channels: int,
    ) -> None:
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            for chunk in frames:
                wf.writeframes(chunk.tobytes())

    def _write_wav_bytes(
        self,
        path: Path,
        frames: list[bytes],
        sample_rate: int,
        channels: int,
    ) -> None:
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            for chunk in frames:
                wf.writeframes(chunk)

    def _wav_stats(self, path: Path) -> tuple[int, float]:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
        duration = frames / float(rate) if rate else 0.0
        return frames, duration

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
