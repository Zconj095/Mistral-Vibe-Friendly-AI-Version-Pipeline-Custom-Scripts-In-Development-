from __future__ import annotations

import base64
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, Field

from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData

if TYPE_CHECKING:
    from vibe.core.types import ToolCallEvent, ToolResultEvent


DEFAULT_MODEL_NAME = "small"
DEFAULT_MAX_AUDIO_BYTES = 100_000_000

_MODEL_CACHE: dict[tuple[str, str, str, str, str], object] = {}


class AudioTranscriptionConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    backend: str = Field(
        default="auto",
        description="Transcription backend: auto, faster_whisper, or whisper.",
    )
    model_name: str = Field(
        default=DEFAULT_MODEL_NAME,
        description="Local model name or path.",
    )
    device: str = Field(
        default="cpu",
        description="Device for inference (cpu or cuda).",
    )
    compute_type: str = Field(
        default="int8",
        description="Compute type for faster-whisper (for example: int8, int8_float16).",
    )
    download_root: Path | None = Field(
        default=None,
        description="Optional model cache directory.",
    )
    ffmpeg_path: Path | None = Field(
        default=None,
        description="Optional path to ffmpeg (file or directory).",
    )
    beam_size: int = Field(default=5, description="Beam size for decoding.")
    temperature: float = Field(default=0.0, description="Decoding temperature.")
    vad_filter: bool = Field(
        default=False, description="Apply VAD filtering when supported."
    )
    word_timestamps: bool = Field(
        default=False, description="Request word timestamps when supported."
    )
    max_audio_bytes: int = Field(
        default=DEFAULT_MAX_AUDIO_BYTES,
        description="Maximum allowed audio size in bytes.",
    )


class AudioTranscriptionState(BaseToolState):
    pass


class AudioTranscriptionArgs(BaseModel):
    path: str | None = Field(
        default=None, description="Path to a local audio file."
    )
    audio_base64: str | None = Field(
        default=None,
        description="Base64-encoded audio data (optionally data:...;base64,...).",
    )
    model: str | None = Field(
        default=None, description="Override the configured model name or path."
    )
    language: str | None = Field(
        default=None, description="Optional language code for transcription."
    )
    timestamp_granularities: list[str] | None = Field(
        default=None,
        description="Timestamp granularities to return (only 'segment' supported).",
    )
    task: str | None = Field(
        default=None, description="Task: transcribe or translate."
    )
    beam_size: int | None = Field(default=None, description="Override beam size.")
    temperature: float | None = Field(
        default=None, description="Override decoding temperature."
    )
    vad_filter: bool | None = Field(
        default=None, description="Override VAD filter setting."
    )
    word_timestamps: bool | None = Field(
        default=None, description="Override word timestamp setting."
    )
    backend: str | None = Field(
        default=None, description="Override backend selection."
    )


class AudioSegment(BaseModel):
    start: float
    end: float
    text: str


class AudioTranscriptionResult(BaseModel):
    text: str
    language: str | None
    segments: list[AudioSegment] | None
    backend: str
    model: str


class AudioTranscription(
    BaseTool[
        AudioTranscriptionArgs,
        AudioTranscriptionResult,
        AudioTranscriptionConfig,
        AudioTranscriptionState,
    ],
    ToolUIData[AudioTranscriptionArgs, AudioTranscriptionResult],
):
    description: ClassVar[str] = (
        "Transcribe local audio using an offline speech model (faster-whisper/whisper)."
    )

    async def run(self, args: AudioTranscriptionArgs) -> AudioTranscriptionResult:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        audio_path, cleanup = self._resolve_audio_input(args)
        try:
            self._ensure_ffmpeg()
            backend = self._resolve_backend(args.backend)
            model_name = args.model or self.config.model_name
            language = self._normalize_language(args.language)
            timestamp_granularities = self._normalize_timestamp_granularities(
                args.timestamp_granularities
            )
            if timestamp_granularities and language:
                raise ToolError(
                    "timestamp_granularities is not compatible with language."
                )

            task = self._normalize_task(args.task)
            beam_size = (
                args.beam_size
                if args.beam_size is not None
                else self.config.beam_size
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

            if beam_size <= 0:
                raise ToolError("beam_size must be a positive integer.")
            if temperature < 0:
                raise ToolError("temperature cannot be negative.")

            if backend == "faster_whisper":
                text, segments, detected_language = self._transcribe_faster_whisper(
                    audio_path,
                    model_name,
                    language,
                    task,
                    beam_size,
                    temperature,
                    vad_filter,
                    word_timestamps,
                )
            else:
                text, segments, detected_language = self._transcribe_whisper(
                    audio_path,
                    model_name,
                    language,
                    task,
                    beam_size,
                    temperature,
                    word_timestamps,
                )

            if not timestamp_granularities:
                segments = None

            return AudioTranscriptionResult(
                text=text,
                language=language or detected_language,
                segments=segments,
                backend=backend,
                model=model_name,
            )
        finally:
            cleanup()

    def _resolve_audio_input(
        self, args: AudioTranscriptionArgs
    ) -> tuple[Path, callable]:
        if args.path and args.audio_base64:
            raise ToolError("Provide either path or audio_base64, not both.")

        if args.path:
            path = Path(args.path).expanduser()
            if not path.is_absolute():
                path = self.config.effective_workdir / path
            try:
                resolved = path.resolve()
            except OSError as exc:
                raise ToolError(f"Invalid path: {exc}") from exc

            if not resolved.exists():
                raise ToolError(f"Audio file not found at: {resolved}")
            if resolved.is_dir():
                raise ToolError(f"Path is a directory, not a file: {resolved}")
            self._enforce_size_limit(resolved.stat().st_size)
            return resolved, lambda: None

        if args.audio_base64:
            raw = self._decode_audio_base64(args.audio_base64)
            self._enforce_size_limit(len(raw))
            temp_dir = tempfile.TemporaryDirectory()
            path = Path(temp_dir.name) / "input_audio"
            path.write_bytes(raw)
            return path, temp_dir.cleanup

        raise ToolError("Provide either path or audio_base64.")

    def _decode_audio_base64(self, data: str) -> bytes:
        payload = data.strip()
        if payload.startswith("data:"):
            parts = payload.split(",", 1)
            payload = parts[1] if len(parts) > 1 else ""
        try:
            return base64.b64decode(payload, validate=False)
        except Exception as exc:
            raise ToolError(f"Invalid base64 audio data: {exc}") from exc

    def _enforce_size_limit(self, size: int) -> None:
        if self.config.max_audio_bytes <= 0:
            return
        if size > self.config.max_audio_bytes:
            raise ToolError(
                f"Audio exceeds max_audio_bytes ({self.config.max_audio_bytes})."
            )

    def _ensure_ffmpeg(self) -> None:
        ffmpeg_path = self._resolve_ffmpeg_path()
        if ffmpeg_path is not None:
            if not ffmpeg_path.exists():
                raise ToolError(f"ffmpeg not found at: {ffmpeg_path}")
            os.environ["PATH"] = (
                f"{ffmpeg_path.parent}{os.pathsep}{os.environ.get('PATH', '')}"
            )
        if shutil.which("ffmpeg") is None:
            raise ToolError(
                "ffmpeg is required for audio decoding. Install ffmpeg and add it to PATH."
            )

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

    def _normalize_timestamp_granularities(
        self, value: list[str] | None
    ) -> list[str] | None:
        if not value:
            return None
        normalized = [item.strip().lower() for item in value if item.strip()]
        if not normalized:
            return None
        if any(item != "segment" for item in normalized):
            raise ToolError("Only 'segment' timestamp granularity is supported.")
        return normalized

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

    def _resolve_ffmpeg_path(self) -> Path | None:
        if self.config.ffmpeg_path is None:
            return None
        path = self.config.ffmpeg_path.expanduser()
        if path.is_dir():
            exe_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
            path = path / exe_name
        return path

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

    def _module_available(self, module: str) -> bool:
        try:
            __import__(module)
            return True
        except ModuleNotFoundError:
            return False
        except Exception:
            return False

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, AudioTranscriptionArgs):
            return ToolCallDisplay(summary="audio_transcription")

        source = event.args.path or ("base64" if event.args.audio_base64 else "unknown")
        return ToolCallDisplay(
            summary=f"audio_transcription: {source}",
            details={
                "path": event.args.path,
                "model": event.args.model,
                "language": event.args.language,
                "timestamp_granularities": event.args.timestamp_granularities,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, AudioTranscriptionResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        segment_count = len(event.result.segments or [])
        message = "Transcription complete"
        if event.result.segments is not None:
            message += f" ({segment_count} segments)"

        return ToolResultDisplay(
            success=True,
            message=message,
            details={
                "text": event.result.text,
                "language": event.result.language,
                "segments": event.result.segments,
                "backend": event.result.backend,
                "model": event.result.model,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Transcribing audio"
