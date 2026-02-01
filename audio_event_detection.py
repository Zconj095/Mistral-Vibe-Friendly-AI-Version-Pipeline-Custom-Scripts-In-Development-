
from __future__ import annotations

import audioop
import base64
import csv
import shutil
import tempfile
import wave
from dataclasses import dataclass
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


DEFAULT_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class _AudioStats:
    duration_sec: float
    rms: float
    peak: float
    zcr: float
    spectral_centroid: float


class AudioEventDetectionArgs(BaseModel):
    path: str | None = Field(
        default=None, description="Path to a local audio file."
    )
    audio_base64: str | None = Field(
        default=None,
        description="Base64-encoded audio data (optionally data:...;base64,...).",
    )
    backend: str | None = Field(
        default=None, description="Backend: auto, basic, or yamnet."
    )
    yamnet_model_path: str | None = Field(
        default=None, description="Local YAMNet model path."
    )
    yamnet_class_map_path: str | None = Field(
        default=None, description="Local YAMNet class map CSV path."
    )
    sample_rate: int | None = Field(
        default=None, description="Target sample rate (Hz)."
    )
    max_audio_bytes: int | None = Field(
        default=None, description="Maximum audio bytes to accept."
    )
    max_duration_sec: float | None = Field(
        default=None, description="Optional max duration to analyze."
    )
    top_k: int = Field(
        default=5, description="Number of events to return for YAMNet."
    )
    min_score: float | None = Field(
        default=None, description="Minimum score for YAMNet events."
    )
    return_features: bool = Field(
        default=True, description="Include audio feature statistics."
    )


class AudioEvent(BaseModel):
    label: str
    score: float | None = None


class AudioEventDetectionResult(BaseModel):
    backend: str
    duration_sec: float
    sample_rate: int
    events: list[AudioEvent]
    features: dict | None
    warnings: list[str]


class AudioEventDetectionConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    backend: str = Field(
        default="auto", description="auto, basic, or yamnet."
    )
    sample_rate: int = Field(
        default=DEFAULT_SAMPLE_RATE, description="Target sample rate."
    )
    max_audio_bytes: int = Field(
        default=50_000_000, description="Maximum audio bytes to accept."
    )
    max_duration_sec: float = Field(
        default=120.0, description="Maximum duration to analyze."
    )
    yamnet_model_path: Path | None = Field(
        default=None, description="Local YAMNet model path."
    )
    yamnet_class_map_path: Path | None = Field(
        default=None, description="Local YAMNet class map CSV path."
    )
    silence_rms_threshold: float = Field(
        default=0.01, description="RMS threshold for silence detection."
    )
    noise_zcr_threshold: float = Field(
        default=0.2, description="ZCR threshold for noise detection."
    )
    noise_centroid_threshold: float = Field(
        default=3000.0, description="Spectral centroid threshold for noise detection."
    )


class AudioEventDetectionState(BaseToolState):
    pass


class AudioEventDetection(
    BaseTool[
        AudioEventDetectionArgs,
        AudioEventDetectionResult,
        AudioEventDetectionConfig,
        AudioEventDetectionState,
    ],
    ToolUIData[AudioEventDetectionArgs, AudioEventDetectionResult],
):
    description: ClassVar[str] = (
        "Detect basic audio events or run YAMNet classification offline."
    )

    async def run(self, args: AudioEventDetectionArgs) -> AudioEventDetectionResult:
        warnings: list[str] = []

        backend = (args.backend or self.config.backend).strip().lower()
        if backend not in {"auto", "basic", "yamnet"}:
            raise ToolError("backend must be auto, basic, or yamnet.")

        if args.path and args.audio_base64:
            raise ToolError("Provide path or audio_base64, not both.")
        if not args.path and not args.audio_base64:
            raise ToolError("Provide path or audio_base64.")

        target_sr = args.sample_rate or self.config.sample_rate
        max_audio_bytes = (
            args.max_audio_bytes
            if args.max_audio_bytes is not None
            else self.config.max_audio_bytes
        )
        max_duration = (
            args.max_duration_sec
            if args.max_duration_sec is not None
            else self.config.max_duration_sec
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = self._prepare_audio_source(args, max_audio_bytes, tmp_dir)
            wav_path = self._ensure_wav(source_path, target_sr, tmp_dir)
            waveform, sample_rate = self._load_waveform(wav_path, target_sr, max_duration)

        stats = self._compute_stats(waveform, sample_rate)

        if backend == "auto":
            if self._yamnet_available(args):
                backend = "yamnet"
            else:
                backend = "basic"
                warnings.append("YAMNet not available; falling back to basic events.")

        if backend == "yamnet":
            events = self._detect_yamnet(waveform, sample_rate, args, warnings)
        else:
            events = self._detect_basic(stats)

        features = None
        if args.return_features:
            features = {
                "rms": stats.rms,
                "peak": stats.peak,
                "zcr": stats.zcr,
                "spectral_centroid": stats.spectral_centroid,
            }

        return AudioEventDetectionResult(
            backend=backend,
            duration_sec=stats.duration_sec,
            sample_rate=sample_rate,
            events=events,
            features=features,
            warnings=warnings,
        )

    def _prepare_audio_source(
        self,
        args: AudioEventDetectionArgs,
        max_audio_bytes: int,
        tmp_dir: str,
    ) -> Path:
        if args.path:
            path = Path(args.path).expanduser()
            if not path.is_absolute():
                path = self.config.effective_workdir / path
            path = path.resolve()
            if not path.exists():
                raise ToolError(f"Audio path not found: {path}")
            if path.is_dir():
                raise ToolError(f"Audio path is a directory: {path}")
            if max_audio_bytes > 0 and path.stat().st_size > max_audio_bytes:
                raise ToolError("Audio file exceeds max_audio_bytes.")
            return path

        audio_b64 = args.audio_base64 or ""
        prefix = "base64,"
        if prefix in audio_b64:
            audio_b64 = audio_b64.split(prefix, 1)[1]
        try:
            raw = base64.b64decode(audio_b64, validate=True)
        except Exception as exc:
            raise ToolError(f"Invalid base64 audio: {exc}") from exc
        if max_audio_bytes > 0 and len(raw) > max_audio_bytes:
            raise ToolError("Audio data exceeds max_audio_bytes.")

        temp_path = Path(tmp_dir) / "input_audio"
        temp_path.write_bytes(raw)
        return temp_path

    def _ensure_wav(self, path: Path, target_sr: int, tmp_dir: str) -> Path:
        if path.suffix.lower() == ".wav":
            return path
        if shutil.which("ffmpeg") is None:
            raise ToolError("ffmpeg is required to decode non-wav audio.")

        wav_path = Path(tmp_dir) / "decoded.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            str(target_sr),
            "-f",
            "wav",
            str(wav_path),
        ]
        try:
            import subprocess

            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise ToolError(stderr or "ffmpeg decode failed.") from exc
        return wav_path

    def _load_waveform(
        self, path: Path, target_sr: int, max_duration: float
    ) -> tuple[list[float], int]:
        try:
            with wave.open(str(path), "rb") as wf:
                channels = wf.getnchannels()
                sample_rate = wf.getframerate()
                sampwidth = wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())
        except wave.Error as exc:
            raise ToolError(f"Failed to read WAV: {exc}") from exc

        if sampwidth != 2:
            frames = audioop.lin2lin(frames, sampwidth, 2)
        if channels > 1:
            frames = audioop.tomono(frames, 2, 0.5, 0.5)
            channels = 1

        if sample_rate != target_sr:
            frames, _state = audioop.ratecv(frames, 2, channels, sample_rate, target_sr, None)
            sample_rate = target_sr

        if max_duration > 0:
            max_samples = int(sample_rate * max_duration)
            frames = frames[: max_samples * 2]

        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise ToolError("numpy is required for audio analysis.") from exc

        data = np.frombuffer(frames, dtype=np.int16)
        if data.size == 0:
            raise ToolError("No audio samples loaded.")
        waveform = (data.astype(np.float32) / 32768.0).tolist()
        return waveform, sample_rate

    def _compute_stats(self, waveform: list[float], sample_rate: int) -> _AudioStats:
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise ToolError("numpy is required for audio analysis.") from exc

        data = np.array(waveform, dtype=np.float32)
        duration = float(data.size) / float(sample_rate)
        rms = float(np.sqrt(np.mean(data ** 2)))
        peak = float(np.max(np.abs(data)))
        zcr = float(np.mean(np.abs(np.diff(np.sign(data))))) / 2.0 if data.size > 1 else 0.0
        if data.size > 0:
            spectrum = np.abs(np.fft.rfft(data))
            freqs = np.fft.rfftfreq(data.size, d=1.0 / sample_rate)
            denom = float(np.sum(spectrum))
            centroid = float(np.sum(freqs * spectrum) / denom) if denom > 0 else 0.0
        else:
            centroid = 0.0

        return _AudioStats(
            duration_sec=duration,
            rms=rms,
            peak=peak,
            zcr=zcr,
            spectral_centroid=centroid,
        )

    def _detect_basic(self, stats: _AudioStats) -> list[AudioEvent]:
        if stats.rms < self.config.silence_rms_threshold:
            return [AudioEvent(label="silence", score=None)]
        if stats.zcr >= self.config.noise_zcr_threshold and stats.spectral_centroid >= self.config.noise_centroid_threshold:
            return [AudioEvent(label="noise", score=None)]
        return [AudioEvent(label="speech_like", score=None)]

    def _yamnet_available(self, args: AudioEventDetectionArgs) -> bool:
        model_path = args.yamnet_model_path or self.config.yamnet_model_path
        if not model_path:
            return False
        path = Path(model_path).expanduser()
        if not path.exists():
            return False
        try:
            import tensorflow as _tf  # noqa: F401
        except ModuleNotFoundError:
            return False
        return True

    def _detect_yamnet(
        self,
        waveform: list[float],
        sample_rate: int,
        args: AudioEventDetectionArgs,
        warnings: list[str],
    ) -> list[AudioEvent]:
        model_path = args.yamnet_model_path or self.config.yamnet_model_path
        if not model_path:
            raise ToolError("yamnet_model_path is required for yamnet backend.")
        path = Path(model_path).expanduser()
        if not path.exists():
            raise ToolError(f"yamnet_model_path not found: {path}")

        try:
            import numpy as np
            import tensorflow as tf
        except ModuleNotFoundError as exc:
            raise ToolError("tensorflow and numpy are required for yamnet backend.") from exc

        model = self._load_yamnet_model(path)
        audio = np.array(waveform, dtype=np.float32)
        if sample_rate != DEFAULT_SAMPLE_RATE:
            warnings.append("YAMNet expects 16kHz audio; results may be degraded.")

        scores = self._run_yamnet(model, audio, tf)
        if scores is None or scores.size == 0:
            raise ToolError("YAMNet produced no scores.")

        if scores.ndim == 1:
            avg_scores = scores
        else:
            avg_scores = scores.mean(axis=0)

        labels = self._load_yamnet_labels(args)
        top_k = max(1, int(args.top_k))
        min_score = args.min_score if args.min_score is not None else 0.0
        ranked = sorted(enumerate(avg_scores.tolist()), key=lambda x: x[1], reverse=True)
        events: list[AudioEvent] = []
        for index, score in ranked[:top_k]:
            if score < min_score:
                continue
            label = labels[index] if index < len(labels) else f"class_{index}"
            events.append(AudioEvent(label=label, score=float(score)))
        return events

    def _load_yamnet_model(self, path: Path):
        try:
            import tensorflow as tf
        except ModuleNotFoundError as exc:
            raise ToolError("tensorflow is required for yamnet backend.") from exc

        if path.is_dir():
            try:
                return tf.saved_model.load(str(path))
            except Exception as exc:
                raise ToolError(f"Failed to load SavedModel: {exc}") from exc
        try:
            return tf.keras.models.load_model(str(path))
        except Exception as exc:
            raise ToolError(f"Failed to load model: {exc}") from exc

    def _run_yamnet(self, model, audio, tf):
        try:
            scores = model(audio)
        except Exception:
            try:
                signature = model.signatures.get("serving_default")
            except Exception:
                signature = None
            if signature is None:
                return None
            tensor = tf.convert_to_tensor(audio)
            try:
                scores = signature(tensor)
            except Exception:
                scores = signature(waveform=tensor)
        if isinstance(scores, dict):
            if "scores" in scores:
                scores = scores["scores"]
            else:
                scores = list(scores.values())[0]
        if isinstance(scores, (list, tuple)):
            scores = scores[0]
        try:
            return scores.numpy()
        except Exception:
            return scores

    def _load_yamnet_labels(self, args: AudioEventDetectionArgs) -> list[str]:
        label_path = args.yamnet_class_map_path or self.config.yamnet_class_map_path
        if not label_path:
            return []
        path = Path(label_path).expanduser()
        if not path.exists():
            return []

        labels: list[str] = []
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    key = "display_name"
                    if key not in reader.fieldnames:
                        key = reader.fieldnames[-1]
                    for row in reader:
                        value = (row.get(key) or "").strip()
                        labels.append(value)
                    return labels
        except Exception:
            labels = []

        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if not row:
                        continue
                    labels.append(row[-1].strip())
        except Exception:
            return []

        return labels

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, AudioEventDetectionArgs):
            return ToolCallDisplay(summary="audio_event_detection")
        return ToolCallDisplay(
            summary="audio_event_detection",
            details={
                "backend": event.args.backend,
                "path": event.args.path,
                "top_k": event.args.top_k,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, AudioEventDetectionResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = f"Detected {len(event.result.events)} event(s)"
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=event.result.warnings,
            details={
                "backend": event.result.backend,
                "events": event.result.events,
                "duration_sec": event.result.duration_sec,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Detecting audio events"
