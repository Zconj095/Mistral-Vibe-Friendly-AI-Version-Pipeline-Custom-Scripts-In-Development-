from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import base64
import shutil
import subprocess
import sys
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
class SynthesisStats:
    backend: str
    path: Path


class AudioSynthesisArgs(BaseModel):
    text: str = Field(description="Text to synthesize.")
    path: str | None = Field(
        default=None, description="Optional output path for the audio."
    )
    backend: str | None = Field(
        default=None,
        description="Backend: auto, piper, pyttsx3, or sapi.",
    )
    voice: str | None = Field(
        default=None, description="Voice name for pyttsx3 or SAPI."
    )
    rate: int | None = Field(
        default=None, description="Speech rate (backend specific)."
    )
    volume: int | None = Field(
        default=None, description="Volume 0-100 for SAPI."
    )
    model_path: str | None = Field(
        default=None, description="Piper model path."
    )
    speaker: str | None = Field(
        default=None, description="Piper speaker name or id."
    )


class AudioSynthesisResult(BaseModel):
    path: str
    backend: str
    bytes_written: int


class AudioSynthesisConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    output_dir: Path = Field(default=Path.home() / ".vibe" / "audio")
    default_backend: str = "auto"
    piper_bin: str = "piper"
    piper_model: Path | None = None
    piper_config: Path | None = None

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


class AudioSynthesisState(BaseToolState):
    pass


class AudioSynthesis(
    BaseTool[
        AudioSynthesisArgs,
        AudioSynthesisResult,
        AudioSynthesisConfig,
        AudioSynthesisState,
    ],
    ToolUIData[AudioSynthesisArgs, AudioSynthesisResult],
):
    description: ClassVar[str] = (
        "Synthesize speech from text using offline TTS backends."
    )

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, AudioSynthesisArgs):
            return ToolCallDisplay(summary="audio_synthesis")
        return ToolCallDisplay(
            summary="Synthesizing audio",
            details={
                "path": event.args.path,
                "backend": event.args.backend,
                "voice": event.args.voice,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if isinstance(event.result, AudioSynthesisResult):
            return ToolResultDisplay(
                success=True,
                message=f"Synthesized {Path(event.result.path).name}",
                details={
                    "path": event.result.path,
                    "backend": event.result.backend,
                    "bytes_written": event.result.bytes_written,
                },
            )
        return ToolResultDisplay(success=True, message="Audio synthesized")

    @classmethod
    def get_status_text(cls) -> str:
        return "Synthesizing speech"

    async def run(self, args: AudioSynthesisArgs) -> AudioSynthesisResult:
        if not args.text.strip():
            raise ToolError("Text cannot be empty.")

        output_path = self._resolve_output_path(args)
        backend = self._resolve_backend(args)

        stats = self._synthesize_text(args, output_path, backend)
        bytes_written = output_path.stat().st_size

        return AudioSynthesisResult(
            path=str(output_path),
            backend=stats.backend,
            bytes_written=bytes_written,
        )

    def _resolve_output_path(self, args: AudioSynthesisArgs) -> Path:
        if args.path:
            path = Path(args.path).expanduser()
            if not path.is_absolute():
                path = self.config.effective_workdir / path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tts_{timestamp}.wav"
            path = self.config.output_dir / filename

        if not path.suffix:
            path = path.with_suffix(".wav")
        elif path.suffix.lower() != ".wav":
            path = path.with_suffix(".wav")

        path.parent.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    def _resolve_backend(self, args: AudioSynthesisArgs) -> str:
        backend = (args.backend or self.config.default_backend or "auto").strip().lower()
        if backend not in {"auto", "piper", "pyttsx3", "sapi"}:
            raise ToolError("backend must be auto, piper, pyttsx3, or sapi.")
        if backend == "auto":
            if self._piper_available(args.model_path):
                return "piper"
            if self._module_available("pyttsx3"):
                return "pyttsx3"
            if sys.platform.startswith("win"):
                return "sapi"
            raise ToolError("No TTS backend available.")
        if backend == "piper" and not self._piper_available(args.model_path):
            raise ToolError("piper is not available or model path is missing.")
        if backend == "pyttsx3" and not self._module_available("pyttsx3"):
            raise ToolError("pyttsx3 is not installed.")
        if backend == "sapi" and not sys.platform.startswith("win"):
            raise ToolError("sapi backend is only available on Windows.")
        return backend

    def _synthesize_text(
        self,
        args: AudioSynthesisArgs,
        output_path: Path,
        backend: str,
    ) -> SynthesisStats:
        if backend == "piper":
            self._synthesize_piper(args, output_path)
        elif backend == "pyttsx3":
            self._synthesize_pyttsx3(args, output_path)
        else:
            self._synthesize_sapi(args, output_path)

        if not output_path.exists():
            raise ToolError("TTS backend did not create an output file.")
        return SynthesisStats(backend=backend, path=output_path)

    def _synthesize_piper(self, args: AudioSynthesisArgs, output_path: Path) -> None:
        model_path: Path | None
        if args.model_path:
            model_path = Path(args.model_path).expanduser()
        else:
            model_path = self.config.piper_model
        if model_path is None:
            raise ToolError("Piper model path is required.")
        if not model_path.exists():
            raise ToolError(f"Piper model not found: {model_path}")

        cmd = [
            self.config.piper_bin,
            "--model",
            str(model_path),
            "--output_file",
            str(output_path),
        ]
        if self.config.piper_config:
            cmd.extend(["--config", str(self.config.piper_config)])
        if args.speaker:
            cmd.extend(["--speaker", str(args.speaker)])

        try:
            subprocess.run(
                cmd,
                input=args.text,
                check=True,
                text=True,
                capture_output=True,
            )
        except FileNotFoundError as exc:
            raise ToolError("piper binary not found.") from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise ToolError(stderr or "piper synthesis failed.") from exc

    def _synthesize_pyttsx3(self, args: AudioSynthesisArgs, output_path: Path) -> None:
        try:
            import pyttsx3
        except ModuleNotFoundError as exc:
            raise ToolError("pyttsx3 is not installed.") from exc

        engine = pyttsx3.init()
        if args.voice:
            engine.setProperty("voice", args.voice)
        if args.rate is not None:
            engine.setProperty("rate", int(args.rate))
        if args.volume is not None:
            engine.setProperty("volume", max(0.0, min(1.0, args.volume / 100.0)))
        engine.save_to_file(args.text, str(output_path))
        engine.runAndWait()

    def _synthesize_sapi(self, args: AudioSynthesisArgs, output_path: Path) -> None:
        text_bytes = args.text.encode("utf-8")
        text_b64 = base64.b64encode(text_bytes).decode("ascii")
        voice = args.voice or ""
        rate = args.rate if args.rate is not None else 0
        volume = args.volume if args.volume is not None else 100
        script = f"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$text = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{text_b64}'))
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
if ('{voice}' -ne '') {{ $synth.SelectVoice('{voice}') }}
$synth.Rate = {int(rate)}
$synth.Volume = {int(volume)}
$synth.SetOutputToWaveFile('{str(output_path)}')
$synth.Speak($text)
$synth.SetOutputToNull()
"""
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise ToolError(stderr or "SAPI synthesis failed.") from exc

    def _piper_available(self, model_path: str | None) -> bool:
        if shutil.which(self.config.piper_bin) is None:
            return False
        if model_path:
            return Path(model_path).expanduser().exists()
        default_model = self.config.piper_model
        return default_model is not None and default_model.exists()

    def _module_available(self, module: str) -> bool:
        try:
            __import__(module)
            return True
        except ModuleNotFoundError:
            return False
        except Exception:
            return False
