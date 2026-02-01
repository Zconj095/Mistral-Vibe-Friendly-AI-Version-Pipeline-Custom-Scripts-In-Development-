from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import wave
from typing import ClassVar

from pydantic import BaseModel, Field

from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from vibe.core.types import ToolCallEvent, ToolResultEvent


class AudioPlaybackArgs(BaseModel):
    path: str = Field(description="Path to an audio file to play.")
    backend: str | None = Field(
        default=None,
        description="Playback backend: auto, ffplay, winsound, or sounddevice.",
    )
    blocking: bool = Field(
        default=True, description="Wait for playback to finish."
    )


class AudioPlaybackResult(BaseModel):
    path: str
    backend: str
    duration_sec: float | None


class AudioPlaybackConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    default_backend: str = "auto"


class AudioPlaybackState(BaseToolState):
    pass


class AudioPlayback(
    BaseTool[
        AudioPlaybackArgs,
        AudioPlaybackResult,
        AudioPlaybackConfig,
        AudioPlaybackState,
    ],
    ToolUIData[AudioPlaybackArgs, AudioPlaybackResult],
):
    description: ClassVar[str] = "Play a local audio file."

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, AudioPlaybackArgs):
            return ToolCallDisplay(summary="audio_playback")
        return ToolCallDisplay(
            summary=f"Playing {Path(event.args.path).name}",
            details={
                "path": event.args.path,
                "backend": event.args.backend,
                "blocking": event.args.blocking,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if isinstance(event.result, AudioPlaybackResult):
            return ToolResultDisplay(
                success=True,
                message=f"Played {Path(event.result.path).name}",
                details={
                    "path": event.result.path,
                    "backend": event.result.backend,
                    "duration_sec": event.result.duration_sec,
                },
            )
        return ToolResultDisplay(success=True, message="Audio played")

    @classmethod
    def get_status_text(cls) -> str:
        return "Playing audio"

    async def run(self, args: AudioPlaybackArgs) -> AudioPlaybackResult:
        path = Path(args.path).expanduser()
        if not path.is_absolute():
            path = self.config.effective_workdir / path
        path = path.resolve()
        if not path.exists():
            raise ToolError(f"Audio file not found: {path}")
        if path.is_dir():
            raise ToolError(f"Path is a directory: {path}")

        backend = self._resolve_backend(args.backend, path)
        duration = self._duration_sec(path)

        if backend == "ffplay":
            self._play_ffplay(path, args.blocking)
        elif backend == "winsound":
            self._play_winsound(path, args.blocking)
        else:
            self._play_sounddevice(path, args.blocking)

        return AudioPlaybackResult(
            path=str(path),
            backend=backend,
            duration_sec=duration,
        )

    def _resolve_backend(self, override: str | None, path: Path) -> str:
        backend = (override or self.config.default_backend or "auto").strip().lower()
        if backend not in {"auto", "ffplay", "winsound", "sounddevice"}:
            raise ToolError("backend must be auto, ffplay, winsound, or sounddevice.")
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
        if backend == "winsound" and path.suffix.lower() != ".wav":
            raise ToolError("winsound backend only supports WAV files.")
        return backend

    def _play_ffplay(self, path: Path, blocking: bool) -> None:
        cmd = [
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "error",
            str(path),
        ]
        if blocking:
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or "").strip()
                raise ToolError(stderr or "ffplay failed.") from exc
        else:
            subprocess.Popen(cmd)

    def _play_winsound(self, path: Path, blocking: bool) -> None:
        import winsound

        flags = winsound.SND_FILENAME
        if not blocking:
            flags |= winsound.SND_ASYNC
        winsound.PlaySound(str(path), flags)

    def _play_sounddevice(self, path: Path, blocking: bool) -> None:
        try:
            import numpy as np
            import sounddevice as sd
        except ModuleNotFoundError as exc:
            raise ToolError("sounddevice is not installed.") from exc

        if path.suffix.lower() != ".wav":
            raise ToolError("sounddevice backend only supports WAV files.")

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

    def _duration_sec(self, path: Path) -> float | None:
        if path.suffix.lower() != ".wav":
            return None
        try:
            with wave.open(str(path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
        except OSError:
            return None
        if rate <= 0:
            return None
        return frames / float(rate)

    def _module_available(self, module: str) -> bool:
        try:
            __import__(module)
            return True
        except ModuleNotFoundError:
            return False
        except Exception:
            return False
