from __future__ import annotations

from datetime import datetime
from pathlib import Path
import base64
import json
import queue
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
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


class VoiceAgentSessionArgs(BaseModel):
    input_mode: str = Field(
        default="audio",
        description="Input mode: audio or text.",
    )
    max_turns: int = Field(default=8, description="Maximum turns to run.")
    output_dir: str | None = Field(
        default=None, description="Output directory for logs and audio."
    )
    system_prompt: str | None = Field(
        default=None, description="System prompt for the agent."
    )
    llm_api_base: str | None = Field(
        default=None, description="OpenAI-compatible API base URL."
    )
    llm_model: str | None = Field(
        default=None, description="LLM model name (Ollama/OpenAI style)."
    )
    llm_temperature: float = Field(default=0.2, description="LLM temperature.")
    llm_max_tokens: int = Field(default=400, description="LLM max tokens.")
    llm_stream: bool = Field(
        default=True, description="Stream tokens from the LLM."
    )
    record_duration_sec: float = Field(
        default=6.0, description="Audio recording duration in seconds."
    )
    record_max_duration_sec: float = Field(
        default=30.0, description="Hard stop for audio recording."
    )
    record_backend: str = Field(
        default="auto",
        description="Recording backend: auto, sounddevice, or ffmpeg.",
    )
    record_vad: bool = Field(
        default=True, description="Enable VAD during recording when possible."
    )
    record_vad_mode: int = Field(
        default=2, description="VAD aggressiveness (0-3)."
    )
    silence_timeout_ms: int = Field(
        default=800, description="Silence timeout for VAD stop."
    )
    sample_rate: int = Field(default=16000, description="Sample rate in Hz.")
    channels: int = Field(default=1, description="Number of channels.")
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
    play_audio: bool = Field(
        default=True, description="Play synthesized speech."
    )


class VoiceAgentSessionResult(BaseModel):
    output_dir: str
    transcript_path: str
    turns: int


class VoiceAgentSessionConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    output_dir: Path = Field(default=Path.home() / ".vibe" / "voice_sessions")
    llm_api_base: str = "http://127.0.0.1:11434/v1"
    llm_model: str = "gpt-oss:latest"
    tts_piper_bin: str = "piper"
    tts_piper_model: Path | None = None

    @field_validator("output_dir", mode="before")
    @classmethod
    def set_default_output_dir(cls, v: Path | str) -> Path:
        if isinstance(v, Path):
            return v
        if not v or not str(v).strip():
            return Path.home() / ".vibe" / "voice_sessions"
        return Path(v)

    @field_validator("output_dir", mode="after")
    @classmethod
    def expand_output_dir(cls, v: Path) -> Path:
        return v.expanduser().resolve()


class VoiceAgentSessionState(BaseToolState):
    pass


_STT_MODEL_CACHE: dict[tuple[str, str, str, str], object] = {}


class VoiceAgentSession(
    BaseTool[
        VoiceAgentSessionArgs,
        VoiceAgentSessionResult,
        VoiceAgentSessionConfig,
        VoiceAgentSessionState,
    ],
    ToolUIData[VoiceAgentSessionArgs, VoiceAgentSessionResult],
):
    description: ClassVar[str] = (
        "Run an offline chained voice agent session (record -> STT -> LLM -> TTS)."
    )

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, VoiceAgentSessionArgs):
            return ToolCallDisplay(summary="voice_agent_session")
        return ToolCallDisplay(
            summary="Starting voice agent session",
            details={
                "input_mode": event.args.input_mode,
                "llm_model": event.args.llm_model,
                "stt_model": event.args.stt_model,
                "tts_backend": event.args.tts_backend,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if isinstance(event.result, VoiceAgentSessionResult):
            return ToolResultDisplay(
                success=True,
                message="Voice agent session complete",
                details={
                    "output_dir": event.result.output_dir,
                    "transcript_path": event.result.transcript_path,
                    "turns": event.result.turns,
                },
            )
        return ToolResultDisplay(success=True, message="Voice agent session complete")

    @classmethod
    def get_status_text(cls) -> str:
        return "Running voice agent session"

    async def run(self, args: VoiceAgentSessionArgs) -> VoiceAgentSessionResult:
        if args.max_turns <= 0:
            raise ToolError("max_turns must be positive.")

        output_dir = self._resolve_output_dir(args)
        transcript_path = output_dir / "transcript.jsonl"

        system_prompt = args.system_prompt or self._default_system_prompt()
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

        turn = 0
        while turn < args.max_turns:
            user_text = self._collect_user_input(args, output_dir, turn)
            if not user_text:
                continue
            if user_text.lower().strip() in {"exit", "quit", "stop"}:
                break

            messages.append({"role": "user", "content": user_text})
            self._append_transcript(transcript_path, "user", user_text)

            assistant_text = self._call_llm(messages, args)
            if not assistant_text:
                assistant_text = "Sorry, I did not catch that."

            messages.append({"role": "assistant", "content": assistant_text})
            self._append_transcript(transcript_path, "assistant", assistant_text)

            audio_path = output_dir / f"tts_{turn:02d}.wav"
            self._synthesize_tts(assistant_text, audio_path, args)
            if args.play_audio:
                self._play_audio(audio_path)

            turn += 1

        return VoiceAgentSessionResult(
            output_dir=str(output_dir),
            transcript_path=str(transcript_path),
            turns=turn,
        )

    def _resolve_output_dir(self, args: VoiceAgentSessionArgs) -> Path:
        if args.output_dir:
            path = Path(args.output_dir).expanduser()
            if not path.is_absolute():
                path = self.config.effective_workdir / path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.config.output_dir / f"session_{timestamp}"
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    def _default_system_prompt(self) -> str:
        return (
            "You are a helpful voice assistant. Keep responses concise and clear. "
            "If you need the user to repeat something, ask directly."
        )

    def _collect_user_input(
        self, args: VoiceAgentSessionArgs, output_dir: Path, turn: int
    ) -> str:
        mode = args.input_mode.strip().lower()
        if mode == "text":
            try:
                return input("User: ").strip()
            except EOFError:
                return ""
        if mode != "audio":
            raise ToolError("input_mode must be audio or text.")

        print("\nPress Enter to start recording, then speak.")
        try:
            input()
        except EOFError:
            return ""

        audio_path = output_dir / f"user_{turn:02d}.wav"
        self._record_audio(audio_path, args)
        return self._transcribe_audio(audio_path, args)

    def _record_audio(self, path: Path, args: VoiceAgentSessionArgs) -> None:
        backend = args.record_backend.strip().lower()
        if backend == "auto":
            backend = "sounddevice" if self._module_available("sounddevice") else "ffmpeg"
        if backend == "sounddevice":
            self._record_sounddevice(path, args)
        elif backend == "ffmpeg":
            self._record_ffmpeg(path, args)
        else:
            raise ToolError("record_backend must be auto, sounddevice, or ffmpeg.")

    def _record_sounddevice(self, path: Path, args: VoiceAgentSessionArgs) -> None:
        try:
            import numpy as np
            import sounddevice as sd
        except ModuleNotFoundError as exc:
            raise ToolError("sounddevice is not installed.") from exc

        sample_rate = args.sample_rate
        channels = args.channels
        frame_ms = 30
        blocksize = int(sample_rate * frame_ms / 1000)
        if blocksize <= 0:
            raise ToolError("Invalid blocksize for recording.")

        q: queue.Queue = queue.Queue()
        frames: list[np.ndarray] = []

        def callback(indata, frames_count, time_info, status):
            del frames_count, time_info, status
            q.put(indata.copy())

        vad = self._create_vad(args.record_vad, args.record_vad_mode)
        speech_started = not args.record_vad
        last_voice = time.monotonic()
        start = last_voice
        if args.record_vad:
            stop_at = start + args.record_max_duration_sec
        else:
            stop_at = start + min(args.record_duration_sec, args.record_max_duration_sec)
        silence_timeout_sec = args.silence_timeout_ms / 1000.0

        with sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="int16",
            blocksize=blocksize,
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
                if args.record_vad:
                    if self._is_voice_numpy(chunk, sample_rate, vad):
                        speech_started = True
                        last_voice = now
                    elif speech_started and (now - last_voice) >= silence_timeout_sec:
                        break

        if not frames:
            raise ToolError("No audio captured.")

        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            for chunk in frames:
                wf.writeframes(chunk.tobytes())

    def _record_ffmpeg(self, path: Path, args: VoiceAgentSessionArgs) -> None:
        if shutil.which("ffmpeg") is None:
            raise ToolError("ffmpeg is required for ffmpeg recording.")

        if sys.platform.startswith("win"):
            fmt = "dshow"
            input_spec = "audio=default"
        elif sys.platform == "darwin":
            fmt = "avfoundation"
            input_spec = ":0"
        else:
            fmt = "alsa"
            input_spec = "default"
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            fmt,
            "-i",
            input_spec,
            "-t",
            f"{args.record_duration_sec:.3f}",
            "-ac",
            str(args.channels),
            "-ar",
            str(args.sample_rate),
            str(path),
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
            raise ToolError(stderr or "ffmpeg recording failed.") from exc

    def _transcribe_audio(self, path: Path, args: VoiceAgentSessionArgs) -> str:
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

    def _transcribe_faster_whisper(
        self, path: Path, args: VoiceAgentSessionArgs
    ) -> str:
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

        parts = []
        for segment in segments:
            parts.append(segment.text)
        return "".join(parts).strip()

    def _transcribe_whisper(self, path: Path, args: VoiceAgentSessionArgs) -> str:
        model = self._get_whisper_model(args)
        try:
            result = model.transcribe(str(path))
        except Exception as exc:
            raise ToolError(f"whisper failed: {exc}") from exc
        return str(result.get("text", "")).strip()

    def _get_faster_whisper_model(self, args: VoiceAgentSessionArgs) -> object:
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

    def _get_whisper_model(self, args: VoiceAgentSessionArgs) -> object:
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

    def _call_llm(self, messages: list[dict[str, str]], args: VoiceAgentSessionArgs) -> str:
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
        self, text: str, output_path: Path, args: VoiceAgentSessionArgs
    ) -> None:
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

    def _tts_piper(self, text: str, output_path: Path, args: VoiceAgentSessionArgs) -> None:
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

    def _tts_pyttsx3(self, text: str, output_path: Path, args: VoiceAgentSessionArgs) -> None:
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

    def _tts_sapi(self, text: str, output_path: Path, args: VoiceAgentSessionArgs) -> None:
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

    def _play_audio(self, path: Path) -> None:
        if shutil.which("ffplay"):
            cmd = [
                "ffplay",
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "error",
                str(path),
            ]
            subprocess.Popen(cmd)
            return
        if sys.platform.startswith("win") and path.suffix.lower() == ".wav":
            import winsound

            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            return
        raise ToolError("No audio playback backend available.")

    def _append_transcript(self, path: Path, role: str, text: str) -> None:
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "role": role,
            "text": text,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _create_vad(self, enabled: bool, mode: int):
        if not enabled:
            return None
        try:
            import webrtcvad
        except ModuleNotFoundError:
            return None
        mode = max(0, min(3, int(mode)))
        return webrtcvad.Vad(mode)

    def _is_voice_numpy(self, chunk, sample_rate: int, vad) -> bool:
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
        return rms >= 0.01

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
