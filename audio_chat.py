from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import base64
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
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
class _TranscribedPart:
    message_index: int
    part_index: int
    text: str
    source: str


class AudioChatArgs(BaseModel):
    messages: list[dict] = Field(
        description="Chat messages. Content can include text or input_audio parts."
    )
    llm_api_base: str | None = Field(
        default=None, description="OpenAI-compatible API base URL."
    )
    llm_model: str | None = Field(
        default=None, description="LLM model name."
    )
    llm_temperature: float = Field(default=0.2, description="LLM temperature.")
    llm_max_tokens: int = Field(default=400, description="LLM max tokens.")
    llm_stream: bool = Field(default=False, description="Stream LLM tokens.")
    stt_backend: str = Field(
        default="auto",
        description="STT backend: auto, faster_whisper, or whisper.",
    )
    stt_model: str = Field(
        default="small", description="STT model name or path."
    )
    stt_device: str = Field(
        default="auto", description="STT device: auto, cpu, or cuda."
    )
    stt_compute_type: str = Field(
        default="int8", description="Compute type for faster-whisper."
    )
    tts_backend: str = Field(
        default="auto", description="TTS backend: auto, piper, pyttsx3, or sapi."
    )
    tts_model_path: str | None = Field(
        default=None, description="Piper model path."
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
    return_audio: bool = Field(
        default=False, description="Synthesize audio response."
    )
    return_audio_base64: bool = Field(
        default=False, description="Include base64 audio if size allows."
    )
    output_dir: str | None = Field(
        default=None, description="Output directory for audio files."
    )
    output_name: str | None = Field(
        default=None, description="Output audio filename (wav)."
    )


class AudioChatResult(BaseModel):
    text: str
    audio_path: str | None
    audio_base64: str | None
    transcripts: list[dict]
    llm_model: str


class AudioChatConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    output_dir: Path = Field(default=Path.home() / ".vibe" / "audio")
    llm_api_base: str = "http://127.0.0.1:11434/v1"
    llm_model: str = "gpt-oss:latest"
    tts_piper_bin: str = "piper"
    tts_piper_model: Path | None = None
    max_audio_bytes: int = 100_000_000
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


class AudioChatState(BaseToolState):
    pass


_STT_MODEL_CACHE: dict[tuple[str, str, str, str], object] = {}


class AudioChat(
    BaseTool[AudioChatArgs, AudioChatResult, AudioChatConfig, AudioChatState],
    ToolUIData[AudioChatArgs, AudioChatResult],
):
    description: ClassVar[str] = (
        "Chat with mixed text/audio inputs by chaining STT -> LLM -> optional TTS."
    )

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, AudioChatArgs):
            return ToolCallDisplay(summary="audio_chat")
        return ToolCallDisplay(
            summary="audio_chat",
            details={
                "message_count": len(event.args.messages),
                "llm_model": event.args.llm_model,
                "return_audio": event.args.return_audio,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if isinstance(event.result, AudioChatResult):
            return ToolResultDisplay(
                success=True,
                message="Audio chat complete",
                details={
                    "text": event.result.text,
                    "audio_path": event.result.audio_path,
                    "audio_base64": event.result.audio_base64,
                    "transcripts": event.result.transcripts,
                },
            )
        return ToolResultDisplay(success=True, message="Audio chat complete")

    @classmethod
    def get_status_text(cls) -> str:
        return "Running audio chat"

    async def run(self, args: AudioChatArgs) -> AudioChatResult:
        if not args.messages:
            raise ToolError("messages cannot be empty.")

        output_dir = self._resolve_output_dir(args.output_dir)
        processed_messages, transcripts = self._process_messages(args)

        response_text = self._call_llm(processed_messages, args)
        response_text = response_text.strip()

        audio_path = None
        audio_base64 = None
        if args.return_audio or args.return_audio_base64:
            audio_path = self._synthesize_tts(response_text, output_dir, args)
            if args.return_audio_base64:
                audio_base64 = self._encode_audio_base64(audio_path)

        transcript_payloads = [
            {
                "message_index": item.message_index,
                "part_index": item.part_index,
                "text": item.text,
                "source": item.source,
            }
            for item in transcripts
        ]

        return AudioChatResult(
            text=response_text,
            audio_path=str(audio_path) if audio_path else None,
            audio_base64=audio_base64,
            transcripts=transcript_payloads,
            llm_model=args.llm_model or self.config.llm_model,
        )

    def _resolve_output_dir(self, value: str | None) -> Path:
        if value:
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = self.config.effective_workdir / path
        else:
            path = self.config.output_dir
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    def _process_messages(
        self, args: AudioChatArgs
    ) -> tuple[list[dict[str, str]], list[_TranscribedPart]]:
        processed: list[dict[str, str]] = []
        transcripts: list[_TranscribedPart] = []

        for message_index, message in enumerate(args.messages):
            if not isinstance(message, dict):
                raise ToolError("Each message must be an object.")
            role = str(message.get("role", "")).strip()
            if not role:
                raise ToolError("Each message must include a role.")
            content = message.get("content")
            if isinstance(content, str):
                processed.append({"role": role, "content": content})
                continue
            if not isinstance(content, list):
                raise ToolError("Message content must be a string or a list.")

            parts: list[str] = []
            for part_index, part in enumerate(content):
                if not isinstance(part, dict):
                    raise ToolError("Content parts must be objects.")
                part_type = str(part.get("type", "")).strip()
                if part_type == "text":
                    text = str(part.get("text", "")).strip()
                    if text:
                        parts.append(text)
                    continue
                if part_type == "input_audio":
                    input_audio = part.get("input_audio") or {}
                    transcript = self._transcribe_audio_input(
                        input_audio, args, message_index, part_index
                    )
                    if transcript:
                        parts.append(transcript.text)
                        transcripts.append(transcript)
                    continue
                raise ToolError(f"Unsupported content type: {part_type}")

            merged = "\n".join([p for p in parts if p])
            processed.append({"role": role, "content": merged})

        return processed, transcripts

    def _transcribe_audio_input(
        self,
        input_audio: dict,
        args: AudioChatArgs,
        message_index: int,
        part_index: int,
    ) -> _TranscribedPart:
        path, cleanup = self._resolve_audio_input(input_audio)
        try:
            self._ensure_ffmpeg()
            text = self._transcribe_audio(path, args)
            source = str(path)
            return _TranscribedPart(
                message_index=message_index,
                part_index=part_index,
                text=text,
                source=source,
            )
        finally:
            cleanup()

    def _resolve_audio_input(self, input_audio: dict) -> tuple[Path, callable]:
        data = input_audio.get("data")
        path = input_audio.get("path")
        if data and path:
            raise ToolError("Provide either data or path for input_audio.")
        if path:
            resolved = Path(path).expanduser()
            if not resolved.is_absolute():
                resolved = self.config.effective_workdir / resolved
            resolved = resolved.resolve()
            if not resolved.exists():
                raise ToolError(f"Audio file not found: {resolved}")
            if resolved.is_dir():
                raise ToolError(f"Audio path is a directory: {resolved}")
            return resolved, lambda: None

        if data:
            payload = str(data).strip()
            if payload.startswith("data:"):
                payload = payload.split(",", 1)[1] if "," in payload else ""
            try:
                raw = base64.b64decode(payload, validate=False)
            except Exception as exc:
                raise ToolError(f"Invalid base64 audio data: {exc}") from exc
            if (
                self.config.max_audio_bytes > 0
                and len(raw) > self.config.max_audio_bytes
            ):
                raise ToolError("Audio payload exceeds max_audio_bytes.")
            temp_dir = tempfile.TemporaryDirectory()
            tmp_path = Path(temp_dir.name) / "input_audio"
            tmp_path.write_bytes(raw)
            return tmp_path, temp_dir.cleanup

        raise ToolError("input_audio requires data or path.")

    def _ensure_ffmpeg(self) -> None:
        if shutil.which("ffmpeg") is None:
            raise ToolError("ffmpeg is required for audio decoding.")

    def _transcribe_audio(self, path: Path, args: AudioChatArgs) -> str:
        backend = self._resolve_stt_backend(args.stt_backend)
        if backend == "faster_whisper":
            return self._transcribe_faster_whisper(path, args)
        return self._transcribe_whisper(path, args)

    def _resolve_stt_backend(self, value: str) -> str:
        backend = value.strip().lower()
        if backend not in {"auto", "faster_whisper", "whisper"}:
            raise ToolError("stt_backend must be auto, faster_whisper, or whisper.")
        if backend == "auto":
            if self._module_available("faster_whisper"):
                return "faster_whisper"
            if self._module_available("whisper"):
                return "whisper"
            raise ToolError("No STT backend found.")
        if backend == "faster_whisper" and not self._module_available("faster_whisper"):
            raise ToolError("faster-whisper is not installed.")
        if backend == "whisper" and not self._module_available("whisper"):
            raise ToolError("whisper is not installed.")
        return backend

    def _transcribe_faster_whisper(self, path: Path, args: AudioChatArgs) -> str:
        model = self._get_faster_whisper_model(args)
        try:
            segments, _info = model.transcribe(
                str(path),
                beam_size=5,
                language=None,
                task="transcribe",
            )
        except Exception as exc:
            raise ToolError(f"faster-whisper failed: {exc}") from exc

        parts = [segment.text for segment in segments]
        return "".join(parts).strip()

    def _transcribe_whisper(self, path: Path, args: AudioChatArgs) -> str:
        model = self._get_whisper_model(args)
        try:
            result = model.transcribe(str(path))
        except Exception as exc:
            raise ToolError(f"whisper failed: {exc}") from exc
        return str(result.get("text", "")).strip()

    def _get_faster_whisper_model(self, args: AudioChatArgs) -> object:
        key = (args.stt_model, args.stt_device, args.stt_compute_type, "faster_whisper")
        cached = _STT_MODEL_CACHE.get(key)
        if cached is not None:
            return cached

        try:
            from faster_whisper import WhisperModel
        except ModuleNotFoundError as exc:
            raise ToolError("faster-whisper is not installed.") from exc

        device = self._resolve_device(args.stt_device)
        model = WhisperModel(
            args.stt_model,
            device=device,
            compute_type=args.stt_compute_type,
        )
        _STT_MODEL_CACHE[key] = model
        return model

    def _get_whisper_model(self, args: AudioChatArgs) -> object:
        key = (args.stt_model, args.stt_device, "whisper", "")
        cached = _STT_MODEL_CACHE.get(key)
        if cached is not None:
            return cached

        try:
            import whisper
        except ModuleNotFoundError as exc:
            raise ToolError("whisper is not installed.") from exc

        device = None if args.stt_device == "auto" else args.stt_device
        model = whisper.load_model(args.stt_model, device=device)
        _STT_MODEL_CACHE[key] = model
        return model

    def _resolve_device(self, device: str) -> str:
        device = device.strip().lower()
        if device == "auto":
            return "cuda" if self._cuda_available() else "cpu"
        if device in {"cpu", "cuda"}:
            return device
        raise ToolError("stt_device must be auto, cpu, or cuda.")

    def _cuda_available(self) -> bool:
        try:
            import ctranslate2
        except ModuleNotFoundError:
            return False
        try:
            return ctranslate2.get_cuda_device_count() > 0
        except Exception:
            return False

    def _call_llm(self, messages: list[dict[str, str]], args: AudioChatArgs) -> str:
        api_base = (args.llm_api_base or self.config.llm_api_base).rstrip("/")
        url = api_base + "/chat/completions"
        payload = {
            "model": args.llm_model or self.config.llm_model,
            "messages": messages,
            "temperature": args.llm_temperature,
            "max_tokens": args.llm_max_tokens,
            "stream": bool(args.llm_stream),
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

        if not args.llm_stream:
            body = resp.read().decode("utf-8")
            parsed = json.loads(body)
            return parsed["choices"][0]["message"].get("content", "").strip()

        return self._read_streaming_response(resp)

    def _read_streaming_response(self, resp) -> str:
        parts: list[str] = []
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
                parts.append(content)
                sys.stdout.write(content)
                sys.stdout.flush()
        if parts:
            sys.stdout.write("\n")
            sys.stdout.flush()
        return "".join(parts).strip()

    def _synthesize_tts(
        self, text: str, output_dir: Path, args: AudioChatArgs
    ) -> Path:
        if not text:
            raise ToolError("Cannot synthesize empty response.")
        output_path = self._resolve_output_path(output_dir, args.output_name)
        backend = args.tts_backend.strip().lower()
        if backend == "auto":
            if self._piper_available(args.tts_model_path):
                backend = "piper"
            elif self._module_available("pyttsx3"):
                backend = "pyttsx3"
            elif sys.platform.startswith("win"):
                backend = "sapi"
            else:
                raise ToolError("No TTS backend available.")

        if backend == "piper":
            self._tts_piper(text, output_path, args)
        elif backend == "pyttsx3":
            self._tts_pyttsx3(text, output_path, args)
        elif backend == "sapi":
            self._tts_sapi(text, output_path, args)
        else:
            raise ToolError("tts_backend must be auto, piper, pyttsx3, or sapi.")
        return output_path

    def _resolve_output_path(self, output_dir: Path, name: str | None) -> Path:
        if name:
            filename = name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audio_chat_{timestamp}.wav"
        path = output_dir / filename
        if path.suffix.lower() != ".wav":
            path = path.with_suffix(".wav")
        return path

    def _tts_piper(self, text: str, output_path: Path, args: AudioChatArgs) -> None:
        model_path = args.tts_model_path or str(self.config.tts_piper_model or "")
        if not model_path:
            raise ToolError("Piper model path is required for piper backend.")
        if shutil.which(self.config.tts_piper_bin) is None:
            raise ToolError("piper binary not found.")

        cmd = [
            self.config.tts_piper_bin,
            "--model",
            model_path,
            "--output_file",
            str(output_path),
        ]
        try:
            subprocess.run(
                cmd,
                input=text,
                check=True,
                text=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise ToolError(stderr or "piper synthesis failed.") from exc

    def _tts_pyttsx3(self, text: str, output_path: Path, args: AudioChatArgs) -> None:
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

    def _tts_sapi(self, text: str, output_path: Path, args: AudioChatArgs) -> None:
        text_b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
        voice = args.tts_voice or ""
        rate = args.tts_rate if args.tts_rate is not None else 0
        volume = args.tts_volume if args.tts_volume is not None else 100
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

    def _encode_audio_base64(self, path: Path) -> str:
        data = path.read_bytes()
        max_bytes = self.config.max_audio_base64_bytes
        if max_bytes > 0 and len(data) > max_bytes:
            raise ToolError("Audio too large for base64 output.")
        return base64.b64encode(data).decode("ascii")

    def _module_available(self, module: str) -> bool:
        try:
            __import__(module)
            return True
        except ModuleNotFoundError:
            return False
        except Exception:
            return False

    def _piper_available(self, model_path: str | None) -> bool:
        if shutil.which(self.config.tts_piper_bin) is None:
            return False
        if model_path:
            return Path(model_path).expanduser().exists()
        if self.config.tts_piper_model:
            return self.config.tts_piper_model.exists()
        return False
