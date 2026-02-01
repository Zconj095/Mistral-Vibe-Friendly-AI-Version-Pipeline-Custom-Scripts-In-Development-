from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import shutil
import subprocess
import sys
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


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class AudioTtsStreamArgs(BaseModel):
    text: str = Field(description="Text to synthesize.")
    backend: str | None = Field(
        default=None,
        description="Backend: auto, piper, pyttsx3, or sapi.",
    )
    model_path: str | None = Field(
        default=None, description="Piper model path."
    )
    speaker: str | None = Field(
        default=None, description="Piper speaker name or id."
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
    chunk_mode: str = Field(
        default="sentence",
        description="Chunk mode: sentence, line, word, or fixed.",
    )
    max_chunk_chars: int = Field(
        default=220, description="Maximum characters per chunk."
    )
    min_chunk_chars: int = Field(
        default=40, description="Minimum characters per chunk."
    )
    output_dir: str | None = Field(
        default=None, description="Output directory for audio files."
    )
    output_prefix: str | None = Field(
        default=None, description="Prefix for output file names."
    )
    combine: bool = Field(
        default=False, description="Combine chunks into a single WAV file."
    )
    play: bool = Field(
        default=False, description="Play chunks as they are produced."
    )
    playback_backend: str | None = Field(
        default=None,
        description="Playback backend: auto, ffplay, winsound, or sounddevice.",
    )
    blocking: bool = Field(
        default=True, description="Wait for playback to finish."
    )


class AudioTtsStreamResult(BaseModel):
    backend: str
    chunk_paths: list[str]
    combined_path: str | None
    chunk_count: int
    bytes_written: int
    warnings: list[str]


class AudioTtsStreamConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    output_dir: Path = Field(default=Path.home() / ".vibe" / "audio")
    default_backend: str = "auto"
    piper_bin: str = "piper"
    piper_model: Path | None = None
    piper_config: Path | None = None
    default_playback_backend: str = "auto"
    max_total_bytes: int = 200_000_000

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


class AudioTtsStreamState(BaseToolState):
    pass


class AudioTtsStream(
    BaseTool[
        AudioTtsStreamArgs,
        AudioTtsStreamResult,
        AudioTtsStreamConfig,
        AudioTtsStreamState,
    ],
    ToolUIData[AudioTtsStreamArgs, AudioTtsStreamResult],
):
    description: ClassVar[str] = (
        "Stream TTS by chunking text and synthesizing WAV segments."
    )

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, AudioTtsStreamArgs):
            return ToolCallDisplay(summary="audio_tts_stream")
        return ToolCallDisplay(
            summary="Streaming TTS",
            details={
                "backend": event.args.backend,
                "chunk_mode": event.args.chunk_mode,
                "play": event.args.play,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if isinstance(event.result, AudioTtsStreamResult):
            return ToolResultDisplay(
                success=True,
                message=f"Generated {event.result.chunk_count} chunk(s)",
                warnings=event.result.warnings,
                details={
                    "backend": event.result.backend,
                    "chunk_paths": event.result.chunk_paths,
                    "combined_path": event.result.combined_path,
                },
            )
        return ToolResultDisplay(success=True, message="Streaming TTS complete")

    @classmethod
    def get_status_text(cls) -> str:
        return "Streaming TTS"

    async def run(self, args: AudioTtsStreamArgs) -> AudioTtsStreamResult:
        if not args.text.strip():
            raise ToolError("Text cannot be empty.")

        output_dir = self._resolve_output_dir(args.output_dir)
        backend = self._resolve_backend(args)

        chunks = self._split_text(
            args.text,
            args.chunk_mode.strip().lower(),
            args.max_chunk_chars,
            args.min_chunk_chars,
        )
        if not chunks:
            raise ToolError("No chunks produced from text.")

        prefix = self._resolve_prefix(args.output_prefix)
        chunk_paths: list[Path] = []
        warnings: list[str] = []
        total_bytes = 0

        playback_backend = None
        if args.play:
            playback_backend = self._resolve_playback_backend(args.playback_backend)

        for idx, chunk in enumerate(chunks, start=1):
            path = output_dir / f"{prefix}_{idx:03d}.wav"
            self._synthesize_chunk(chunk, path, backend, args)
            chunk_paths.append(path)
            total_bytes += path.stat().st_size
            if self.config.max_total_bytes > 0 and total_bytes > self.config.max_total_bytes:
                warnings.append("Max total output bytes reached; stopping early.")
                break
            if args.play and playback_backend:
                self._play(path, playback_backend, args.blocking)

        if not chunk_paths:
            raise ToolError("No audio chunks were produced.")

        combined_path = None
        if args.combine:
            combined_path = self._combine_chunks(chunk_paths, output_dir, prefix, warnings)

        return AudioTtsStreamResult(
            backend=backend,
            chunk_paths=[str(path) for path in chunk_paths],
            combined_path=str(combined_path) if combined_path else None,
            chunk_count=len(chunk_paths),
            bytes_written=total_bytes,
            warnings=warnings,
        )

    def _resolve_output_dir(self, override: str | None) -> Path:
        if override:
            path = Path(override).expanduser()
            if not path.is_absolute():
                path = self.config.effective_workdir / path
        else:
            path = self.config.output_dir
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    def _resolve_prefix(self, prefix: str | None) -> str:
        if prefix:
            return re.sub(r"[^A-Za-z0-9_-]+", "_", prefix).strip("_") or "tts_stream"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"tts_stream_{timestamp}"

    def _resolve_backend(self, args: AudioTtsStreamArgs) -> str:
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

    def _resolve_playback_backend(self, override: str | None) -> str:
        backend = (override or self.config.default_playback_backend or "auto").strip().lower()
        if backend not in {"auto", "ffplay", "winsound", "sounddevice"}:
            raise ToolError("playback_backend must be auto, ffplay, winsound, or sounddevice.")
        if backend == "auto":
            if shutil.which("ffplay"):
                return "ffplay"
            if sys.platform.startswith("win"):
                return "winsound"
            if self._module_available("sounddevice"):
                return "sounddevice"
            raise ToolError("No playback backend available.")
        if backend == "ffplay" and shutil.which("ffplay") is None:
            raise ToolError("ffplay is not installed.")
        if backend == "winsound" and not sys.platform.startswith("win"):
            raise ToolError("winsound backend is only available on Windows.")
        if backend == "sounddevice" and not self._module_available("sounddevice"):
            raise ToolError("sounddevice is not installed.")
        return backend

    def _split_text(
        self,
        text: str,
        mode: str,
        max_chars: int,
        min_chars: int,
    ) -> list[str]:
        content = text.strip()
        if not content:
            return []

        if max_chars <= 0:
            max_chars = len(content)

        if mode == "sentence":
            tokens = SENTENCE_SPLIT_RE.split(content)
            separator = " "
        elif mode == "line":
            tokens = content.splitlines()
            separator = "\n"
        elif mode == "word":
            tokens = content.split()
            separator = " "
        elif mode == "fixed":
            tokens = [content]
            separator = ""
        else:
            raise ToolError("chunk_mode must be sentence, line, word, or fixed.")

        chunks: list[str] = []
        current = ""
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            parts = self._split_long_token(token, max_chars)
            for part in parts:
                if not current:
                    current = part
                    continue
                candidate = f"{current}{separator}{part}" if separator else f"{current}{part}"
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    chunks.append(current)
                    current = part
        if current:
            chunks.append(current)

        if min_chars > 0:
            merged: list[str] = []
            for chunk in chunks:
                if (
                    merged
                    and len(chunk) < min_chars
                    and (len(merged[-1]) + 1 + len(chunk) <= max_chars)
                ):
                    merged[-1] = f"{merged[-1]} {chunk}"
                else:
                    merged.append(chunk)
            chunks = merged

        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def _split_long_token(self, token: str, max_chars: int) -> list[str]:
        if len(token) <= max_chars:
            return [token]
        parts = []
        for i in range(0, len(token), max_chars):
            parts.append(token[i : i + max_chars])
        return parts

    def _synthesize_chunk(
        self,
        text: str,
        output_path: Path,
        backend: str,
        args: AudioTtsStreamArgs,
    ) -> None:
        if backend == "piper":
            self._synthesize_piper(text, output_path, args)
        elif backend == "pyttsx3":
            self._synthesize_pyttsx3(text, output_path, args)
        else:
            self._synthesize_sapi(text, output_path, args)

        if not output_path.exists():
            raise ToolError("TTS backend did not create an output file.")

    def _synthesize_piper(
        self, text: str, output_path: Path, args: AudioTtsStreamArgs
    ) -> None:
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
                input=text,
                check=True,
                text=True,
                capture_output=True,
            )
        except FileNotFoundError as exc:
            raise ToolError("piper binary not found.") from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise ToolError(stderr or "piper synthesis failed.") from exc

    def _synthesize_pyttsx3(
        self, text: str, output_path: Path, args: AudioTtsStreamArgs
    ) -> None:
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
        engine.save_to_file(text, str(output_path))
        engine.runAndWait()

    def _synthesize_sapi(
        self, text: str, output_path: Path, args: AudioTtsStreamArgs
    ) -> None:
        text_bytes = text.encode("utf-8")
        text_b64 = text_bytes.hex()
        voice = args.voice or ""
        rate = args.rate if args.rate is not None else 0
        volume = args.volume if args.volume is not None else 100
        script = f"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$bytes = for ($i = 0; $i -lt '{text_b64}'.Length; $i += 2) {{ [Convert]::ToByte('{text_b64}'.Substring($i,2),16) }}
$text = [System.Text.Encoding]::UTF8.GetString([byte[]]$bytes)
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

    def _combine_chunks(
        self,
        chunk_paths: list[Path],
        output_dir: Path,
        prefix: str,
        warnings: list[str],
    ) -> Path | None:
        combined_path = output_dir / f"{prefix}_combined.wav"
        params = None
        format_sig = None
        frames: list[bytes] = []
        for path in chunk_paths:
            try:
                with wave.open(str(path), "rb") as wf:
                    current_params = wf.getparams()
                    current_sig = (current_params.nchannels, current_params.sampwidth, current_params.framerate, current_params.comptype, current_params.compname)
                    if params is None:
                        params = current_params
                        format_sig = current_sig
                    else:
                        if current_sig != format_sig:
                            warnings.append("Chunk format mismatch; skipping combine.")
                            return None
                    frames.append(wf.readframes(wf.getnframes()))
            except OSError:
                warnings.append(f"Unable to read WAV chunk: {path}")
                return None
        if params is None:
            return None
        with wave.open(str(combined_path), "wb") as wf:
            wf.setparams(params)
            for data in frames:
                wf.writeframes(data)
        return combined_path

    def _play(self, path: Path, backend: str, blocking: bool) -> None:
        if backend == "ffplay":
            cmd = [
                "ffplay",
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "error",
                str(path),
            ]
            if blocking:
                subprocess.run(cmd, check=False, capture_output=True, text=True)
            else:
                subprocess.Popen(cmd)
            return
        if backend == "winsound":
            import winsound

            flags = winsound.SND_FILENAME
            if not blocking:
                flags |= winsound.SND_ASYNC
            winsound.PlaySound(str(path), flags)
            return
        if backend == "sounddevice":
            try:
                import numpy as np
                import sounddevice as sd
            except ModuleNotFoundError as exc:
                raise ToolError("sounddevice is not installed.") from exc
            with wave.open(str(path), "rb") as wf:
                channels = wf.getnchannels()
                sample_rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
            data = np.frombuffer(frames, dtype=np.int16)
            if channels > 1:
                data = data.reshape(-1, channels)
            sd.play(data, sample_rate)
            if blocking:
                sd.wait()
            return
        raise ToolError("Unknown playback backend.")

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
