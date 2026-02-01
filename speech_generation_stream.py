
from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
import sys
import wave
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar
import urllib.error
import urllib.request

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
WHITESPACE_RE = re.compile(r"\s+")


class SpeechGenerationStreamArgs(BaseModel):
    prompt: str | None = Field(default=None, description="Prompt to generate speech.")
    messages: list[dict] | None = Field(
        default=None, description="Chat messages in OpenAI format."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt."
    )
    llm_api_base: str | None = Field(
        default=None, description="OpenAI-compatible API base URL."
    )
    llm_model: str | None = Field(
        default=None, description="LLM model name."
    )
    llm_temperature: float = Field(default=0.2, description="LLM temperature.")
    llm_max_tokens: int = Field(default=400, description="LLM max tokens.")
    llm_stream: bool = Field(default=True, description="Stream LLM tokens.")
    tts_backend: str | None = Field(
        default=None, description="TTS backend: auto, piper, pyttsx3, or sapi."
    )
    tts_model_path: str | None = Field(
        default=None, description="Piper model path."
    )
    tts_speaker: str | None = Field(
        default=None, description="Piper speaker name or id."
    )
    tts_voice: str | None = Field(
        default=None, description="Voice name for pyttsx3 or SAPI."
    )
    tts_rate: int | None = Field(
        default=None, description="Speech rate (backend specific)."
    )
    tts_volume: int | None = Field(
        default=None, description="Volume 0-100 for SAPI."
    )
    chunk_mode: str = Field(
        default="sentence", description="sentence, line, word, or fixed."
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
    return_audio_base64: bool = Field(
        default=False, description="Return base64 audio when possible."
    )
    max_audio_base64_bytes: int | None = Field(
        default=None, description="Max audio bytes for base64."
    )
    max_total_bytes: int | None = Field(
        default=None, description="Max total bytes of audio output."
    )


class SpeechGenerationStreamResult(BaseModel):
    text: str
    backend: str
    chunk_paths: list[str]
    combined_path: str | None
    chunk_count: int
    bytes_written: int
    audio_base64: str | None
    llm_model: str
    warnings: list[str]


class SpeechGenerationStreamConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    output_dir: Path = Field(default=Path.home() / ".vibe" / "audio")
    llm_api_base: str = "http://127.0.0.1:11434/v1"
    llm_model: str = "gpt-oss:latest"
    default_backend: str = "auto"
    piper_bin: str = "piper"
    piper_model: Path | None = None
    piper_config: Path | None = None
    default_playback_backend: str = "auto"
    max_total_bytes: int = 200_000_000
    max_audio_base64_bytes: int = 5_000_000

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


class SpeechGenerationStreamState(BaseToolState):
    pass


class SpeechGenerationStream(
    BaseTool[
        SpeechGenerationStreamArgs,
        SpeechGenerationStreamResult,
        SpeechGenerationStreamConfig,
        SpeechGenerationStreamState,
    ],
    ToolUIData[SpeechGenerationStreamArgs, SpeechGenerationStreamResult],
):
    description: ClassVar[str] = (
        "Generate speech by streaming LLM text into offline TTS backends."
    )

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, SpeechGenerationStreamArgs):
            return ToolCallDisplay(summary="speech_generation_stream")
        return ToolCallDisplay(
            summary="speech_generation_stream",
            details={
                "llm_model": event.args.llm_model,
                "chunk_mode": event.args.chunk_mode,
                "play": event.args.play,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, SpeechGenerationStreamResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = f"Generated {event.result.chunk_count} chunk(s)"
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=event.result.warnings,
            details={
                "backend": event.result.backend,
                "chunk_paths": event.result.chunk_paths,
                "combined_path": event.result.combined_path,
                "llm_model": event.result.llm_model,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Generating speech"

    async def run(self, args: SpeechGenerationStreamArgs) -> SpeechGenerationStreamResult:
        messages = self._build_messages(args)
        output_dir = self._resolve_output_dir(args.output_dir)
        prefix = self._resolve_prefix(args.output_prefix)
        backend = self._resolve_backend(args)
        playback_backend = None
        if args.play:
            playback_backend = self._resolve_playback_backend(args.playback_backend)
        if args.max_chunk_chars <= 0:
            raise ToolError("max_chunk_chars must be a positive integer.")
        if args.min_chunk_chars < 0:
            raise ToolError("min_chunk_chars must be >= 0.")
        if args.min_chunk_chars > args.max_chunk_chars:
            raise ToolError("min_chunk_chars must be <= max_chunk_chars.")

        max_total_bytes = (
            args.max_total_bytes
            if args.max_total_bytes is not None
            else self.config.max_total_bytes
        )
        max_audio_base64_bytes = (
            args.max_audio_base64_bytes
            if args.max_audio_base64_bytes is not None
            else self.config.max_audio_base64_bytes
        )

        warnings: list[str] = []
        chunk_paths: list[Path] = []
        bytes_written = 0
        full_text = ""
        stopped_early = False

        if args.llm_stream:
            buffer = ""
            for piece in self._stream_llm(messages, args):
                if not piece:
                    continue
                full_text += piece
                buffer += piece
                buffer, bytes_written, stop = self._drain_buffer(
                    buffer,
                    args,
                    output_dir,
                    prefix,
                    backend,
                    chunk_paths,
                    playback_backend,
                    bytes_written,
                    max_total_bytes,
                    warnings,
                    force=False,
                )
                if stop:
                    stopped_early = True
                    warnings.append("Max total output bytes reached; stopping early.")
                    break

            if not stopped_early:
                buffer, bytes_written, _stop = self._drain_buffer(
                    buffer,
                    args,
                    output_dir,
                    prefix,
                    backend,
                    chunk_paths,
                    playback_backend,
                    bytes_written,
                    max_total_bytes,
                    warnings,
                    force=True,
                )
        else:
            full_text = self._call_llm(messages, args)
            chunks = self._split_text(
                full_text,
                args.chunk_mode.strip().lower(),
                args.max_chunk_chars,
                args.min_chunk_chars,
            )
            for chunk in chunks:
                path = output_dir / f"{prefix}_{len(chunk_paths) + 1:03d}.wav"
                self._synthesize_chunk(chunk, path, backend, args)
                chunk_paths.append(path)
                bytes_written += path.stat().st_size
                if args.play and playback_backend:
                    self._play(path, playback_backend, args.blocking)
                if max_total_bytes > 0 and bytes_written > max_total_bytes:
                    warnings.append("Max total output bytes reached; stopping early.")
                    break

        combined_path = None
        if args.combine and chunk_paths:
            combined_path = self._combine_chunks(
                chunk_paths, output_dir, prefix, warnings
            )

        audio_base64 = None
        if args.return_audio_base64:
            target_path = combined_path or (chunk_paths[-1] if chunk_paths else None)
            if target_path is None:
                warnings.append("No audio available for base64 output.")
            else:
                try:
                    audio_base64 = self._encode_audio_base64(
                        target_path, max_audio_base64_bytes
                    )
                except ToolError as exc:
                    warnings.append(str(exc))

        return SpeechGenerationStreamResult(
            text=full_text.strip(),
            backend=backend,
            chunk_paths=[str(path) for path in chunk_paths],
            combined_path=str(combined_path) if combined_path else None,
            chunk_count=len(chunk_paths),
            bytes_written=bytes_written,
            audio_base64=audio_base64,
            llm_model=args.llm_model or self.config.llm_model,
            warnings=warnings,
        )

    def _build_messages(self, args: SpeechGenerationStreamArgs) -> list[dict[str, str]]:
        if args.prompt and args.messages:
            raise ToolError("Provide prompt or messages, not both.")
        if not args.prompt and not args.messages:
            raise ToolError("Provide prompt or messages.")

        if args.messages is not None:
            messages = [dict(item) for item in args.messages]
        else:
            messages = [{"role": "user", "content": args.prompt or ""}]

        if args.system_prompt:
            messages = [{"role": "system", "content": args.system_prompt}] + messages
        return messages

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
            return re.sub(r"[^A-Za-z0-9_-]+", "_", prefix).strip("_") or "speech"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"speech_{timestamp}"

    def _resolve_backend(self, args: SpeechGenerationStreamArgs) -> str:
        backend = (args.tts_backend or self.config.default_backend or "auto").strip().lower()
        if backend not in {"auto", "piper", "pyttsx3", "sapi"}:
            raise ToolError("tts_backend must be auto, piper, pyttsx3, or sapi.")
        if backend == "auto":
            if self._piper_available(args.tts_model_path):
                return "piper"
            if self._module_available("pyttsx3"):
                return "pyttsx3"
            if sys.platform.startswith("win"):
                return "sapi"
            raise ToolError("No TTS backend available.")
        if backend == "piper" and not self._piper_available(args.tts_model_path):
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
    def _call_llm(self, messages: list[dict[str, str]], args: SpeechGenerationStreamArgs) -> str:
        api_base = (args.llm_api_base or self.config.llm_api_base).rstrip("/")
        url = api_base + "/chat/completions"
        payload = {
            "model": args.llm_model or self.config.llm_model,
            "messages": messages,
            "temperature": args.llm_temperature,
            "max_tokens": args.llm_max_tokens,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=600)
        except urllib.error.URLError as exc:
            raise ToolError(f"LLM request failed: {exc}") from exc

        body = resp.read().decode("utf-8")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ToolError(f"LLM response parse failed: {exc}") from exc
        return parsed["choices"][0]["message"].get("content", "").strip()

    def _stream_llm(
        self, messages: list[dict[str, str]], args: SpeechGenerationStreamArgs
    ) -> Iterable[str]:
        api_base = (args.llm_api_base or self.config.llm_api_base).rstrip("/")
        url = api_base + "/chat/completions"
        payload = {
            "model": args.llm_model or self.config.llm_model,
            "messages": messages,
            "temperature": args.llm_temperature,
            "max_tokens": args.llm_max_tokens,
            "stream": True,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=600)
        except urllib.error.URLError as exc:
            raise ToolError(f"LLM request failed: {exc}") from exc

        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            if line.startswith("data:"):
                line = line[len("data:") :].strip()
            if line == "[DONE]":
                break
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            choice = chunk.get("choices", [{}])[0]
            delta = choice.get("delta") or choice.get("message") or {}
            content = delta.get("content")
            if content:
                sys.stdout.write(content)
                sys.stdout.flush()
                yield content
        sys.stdout.write("\n")
        sys.stdout.flush()

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

    def _pop_chunk(
        self,
        buffer: str,
        mode: str,
        max_chars: int,
        min_chars: int,
        force: bool,
    ) -> tuple[str | None, str]:
        if not buffer:
            return None, buffer

        if max_chars <= 0:
            max_chars = len(buffer)

        if mode not in {"sentence", "line", "word", "fixed"}:
            raise ToolError("chunk_mode must be sentence, line, word, or fixed.")

        if mode == "fixed":
            if len(buffer) < max_chars and not force:
                return None, buffer
            cut = min(len(buffer), max_chars)
            return buffer[:cut].strip(), buffer[cut:]

        if len(buffer) < min_chars and not force:
            return None, buffer

        boundary = -1
        if mode == "sentence":
            matches = list(SENTENCE_SPLIT_RE.finditer(buffer))
            if matches:
                boundary = matches[-1].end()
        elif mode == "line":
            boundary = buffer.rfind("\n")
            if boundary != -1:
                boundary += 1
        elif mode == "word":
            matches = list(WHITESPACE_RE.finditer(buffer))
            if matches:
                boundary = matches[-1].end()

        if boundary == -1:
            if force:
                cut = min(len(buffer), max_chars)
                return buffer[:cut].strip(), buffer[cut:]
            if len(buffer) >= max_chars:
                cut = max_chars
                return buffer[:cut].strip(), buffer[cut:]
            return None, buffer

        if boundary > max_chars:
            boundary = max_chars
        if boundary < min_chars and not force:
            return None, buffer

        return buffer[:boundary].strip(), buffer[boundary:]

    def _drain_buffer(
        self,
        buffer: str,
        args: SpeechGenerationStreamArgs,
        output_dir: Path,
        prefix: str,
        backend: str,
        chunk_paths: list[Path],
        playback_backend: str | None,
        bytes_written: int,
        max_total_bytes: int,
        warnings: list[str],
        force: bool,
    ) -> tuple[str, int, bool]:
        stop = False
        mode = args.chunk_mode.strip().lower()
        while True:
            chunk, buffer = self._pop_chunk(
                buffer,
                mode,
                args.max_chunk_chars,
                args.min_chunk_chars,
                force,
            )
            if not chunk:
                break
            path = output_dir / f"{prefix}_{len(chunk_paths) + 1:03d}.wav"
            self._synthesize_chunk(chunk, path, backend, args)
            chunk_paths.append(path)
            bytes_written += path.stat().st_size
            if playback_backend:
                self._play(path, playback_backend, args.blocking)
            if max_total_bytes > 0 and bytes_written > max_total_bytes:
                stop = True
                break
        return buffer, bytes_written, stop
    def _synthesize_chunk(
        self,
        text: str,
        output_path: Path,
        backend: str,
        args: SpeechGenerationStreamArgs,
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
        self, text: str, output_path: Path, args: SpeechGenerationStreamArgs
    ) -> None:
        if args.tts_model_path:
            model_path = Path(args.tts_model_path).expanduser()
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
        if args.tts_speaker:
            cmd.extend(["--speaker", str(args.tts_speaker)])

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
        self, text: str, output_path: Path, args: SpeechGenerationStreamArgs
    ) -> None:
        try:
            import pyttsx3
        except ModuleNotFoundError as exc:
            raise ToolError("pyttsx3 is not installed.") from exc

        engine = pyttsx3.init()
        if args.tts_voice:
            engine.setProperty("voice", args.tts_voice)
        if args.tts_rate is not None:
            engine.setProperty("rate", int(args.tts_rate))
        if args.tts_volume is not None:
            engine.setProperty("volume", max(0.0, min(1.0, args.tts_volume / 100.0)))
        engine.save_to_file(text, str(output_path))
        engine.runAndWait()

    def _synthesize_sapi(
        self, text: str, output_path: Path, args: SpeechGenerationStreamArgs
    ) -> None:
        text_bytes = text.encode("utf-8")
        text_b64 = text_bytes.hex()
        voice = args.tts_voice or ""
        rate = args.tts_rate if args.tts_rate is not None else 0
        volume = args.tts_volume if args.tts_volume is not None else 100
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

    def _encode_audio_base64(self, path: Path, max_bytes: int) -> str:
        data = path.read_bytes()
        if max_bytes > 0 and len(data) > max_bytes:
            raise ToolError("Audio too large for base64 output.")
        return base64.b64encode(data).decode("ascii")

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
