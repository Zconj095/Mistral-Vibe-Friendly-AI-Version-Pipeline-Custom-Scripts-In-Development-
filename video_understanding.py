
from __future__ import annotations

import base64
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Iterable
import urllib.error
import urllib.request

from pydantic import BaseModel, Field

try:
    from actions_lib import validate_args
except ModuleNotFoundError:  # Fallback when tools directory is not on sys.path.
    _actions_path = Path(__file__).with_name("actions_lib.py")
    _spec = importlib.util.spec_from_file_location("actions_lib", _actions_path)
    if not _spec or not _spec.loader:
        raise
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    validate_args = _module.validate_args

from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from vibe.core.types import ToolCallEvent, ToolResultEvent

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_sec": {"type": "number"},
                    "end_sec": {"type": "number"},
                    "description": {"type": "string"},
                },
                "required": ["description"],
                "additionalProperties": False,
            },
        },
        "answer": {"type": "string"},
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class _Frame:
    index: int
    timestamp_sec: float
    path: Path


class VideoUnderstandingArgs(BaseModel):
    path: str = Field(description="Path to a local video file.")
    question: str | None = Field(default=None, description="Optional question.")
    frame_prompt: str | None = Field(
        default=None, description="Prompt for frame captions."
    )
    frame_rate: float | None = Field(
        default=None, description="Frames per second to sample."
    )
    frame_interval_sec: float | None = Field(
        default=None, description="Seconds between sampled frames."
    )
    start_time_sec: float | None = Field(
        default=None, description="Start time in seconds."
    )
    end_time_sec: float | None = Field(
        default=None, description="End time in seconds."
    )
    max_frames: int | None = Field(
        default=None, description="Maximum number of frames to sample."
    )
    scale_width: int | None = Field(
        default=None, description="Scale frames to this width (preserve aspect)."
    )
    output_dir: str | None = Field(
        default=None, description="Output directory for extracted frames."
    )
    keep_frames: bool = Field(
        default=True, description="Keep extracted frames on disk."
    )
    include_frames: bool = Field(
        default=True, description="Include frame captions in the output."
    )
    group_duration_sec: float | None = Field(
        default=None, description="Seconds per timeline group."
    )
    max_captions_per_group: int | None = Field(
        default=None, description="Max captions per timeline group."
    )
    audio_transcribe: bool = Field(
        default=False, description="Transcribe audio track with STT."
    )
    stt_backend: str = Field(
        default="auto", description="STT backend: auto, faster_whisper, or whisper."
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
    vision_backend: str | None = Field(
        default=None, description="Vision backend: ollama."
    )
    vision_model: str | None = Field(
        default=None, description="Vision model name."
    )
    vision_temperature: float | None = Field(
        default=None, description="Vision temperature."
    )
    vision_max_tokens: int | None = Field(
        default=None, description="Vision max tokens."
    )
    vision_num_gpu: int | None = Field(
        default=None, description="Vision num_gpu option (0 forces CPU)."
    )
    llm_api_base: str | None = Field(
        default=None, description="OpenAI-compatible API base URL."
    )
    llm_model: str | None = Field(default=None, description="LLM model name.")
    llm_temperature: float = Field(default=0.2, description="LLM temperature.")
    llm_max_tokens: int = Field(default=600, description="LLM max tokens.")
    llm_stream: bool = Field(default=False, description="Stream LLM output.")
    max_retries: int | None = Field(
        default=None, description="Max retries for JSON output."
    )


class FrameCaption(BaseModel):
    index: int
    timestamp_sec: float
    image_path: str | None
    caption: str


class TimelineSegment(BaseModel):
    start_sec: float
    end_sec: float
    captions: list[str]


class VideoUnderstandingResult(BaseModel):
    video_path: str
    duration_sec: float | None
    frame_count: int
    frames: list[FrameCaption] | None
    timeline: list[TimelineSegment]
    audio_transcript: str | None
    summary: dict | None
    warnings: list[str]
    errors: list[str]


class VideoUnderstandingConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    output_dir: Path = Field(default=Path.home() / ".vibe" / "video_frames")
    ollama_url: str = Field(
        default="http://127.0.0.1:11434",
        description="Base URL for the local Ollama server.",
    )
    vision_model: str = Field(
        default="llava", description="Default vision model name."
    )
    vision_temperature: float = Field(default=0.2, description="Vision temperature.")
    vision_max_tokens: int = Field(
        default=256, description="Vision max tokens."
    )
    llm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL.",
    )
    llm_model: str = Field(
        default="gpt-oss:latest", description="Default LLM model name."
    )
    frame_rate: float = Field(default=1.0, description="Default frame rate.")
    max_frames: int = Field(default=60, description="Default max frames.")
    group_duration_sec: float = Field(
        default=10.0, description="Default seconds per group."
    )
    max_captions_per_group: int = Field(
        default=12, description="Default captions per group."
    )
    max_retries: int = Field(default=2, description="Max JSON retries.")
    ffmpeg_bin: str | None = Field(
        default=None, description="Override ffmpeg path."
    )
    ffprobe_bin: str | None = Field(
        default=None, description="Override ffprobe path."
    )


class VideoUnderstandingState(BaseToolState):
    pass


_STT_MODEL_CACHE: dict[tuple[str, str, str, str], object] = {}


class VideoUnderstanding(
    BaseTool[
        VideoUnderstandingArgs,
        VideoUnderstandingResult,
        VideoUnderstandingConfig,
        VideoUnderstandingState,
    ],
    ToolUIData[VideoUnderstandingArgs, VideoUnderstandingResult],
):
    description: ClassVar[str] = (
        "Understand video by sampling frames, captioning them, and summarizing the timeline."
    )

    async def run(self, args: VideoUnderstandingArgs) -> VideoUnderstandingResult:
        warnings: list[str] = []
        errors: list[str] = []

        video_path = self._resolve_video_path(args.path)
        ffmpeg = self._resolve_ffmpeg()
        ffprobe = self._resolve_ffprobe()
        duration = self._probe_duration(ffprobe, video_path, warnings)

        fps = self._resolve_fps(args)
        start_time = max(0.0, float(args.start_time_sec or 0.0))
        end_time = self._resolve_end_time(args, duration)

        output_dir, temp_dir = self._prepare_output_dir(args)
        max_frames = (
            args.max_frames if args.max_frames is not None else self.config.max_frames
        )
        frame_files = self._extract_frames(
            ffmpeg,
            video_path,
            output_dir,
            fps,
            start_time,
            end_time,
            max_frames,
            args.scale_width,
            warnings,
        )
        frames = self._build_frames(frame_files, fps, start_time)

        frame_prompt = args.frame_prompt or self._default_frame_prompt(args.question)
        captions = self._caption_frames(frames, frame_prompt, args, warnings)

        audio_transcript = None
        if args.audio_transcribe:
            audio_transcript = self._transcribe_audio(
                ffmpeg, video_path, start_time, end_time, args, warnings
            )

        timeline = self._build_timeline(captions, args)
        summary = None
        if captions:
            summary, summary_errors = self._summarize_video(
                captions, timeline, audio_transcript, args
            )
            errors.extend(summary_errors)

        frame_results = None
        if args.include_frames:
            if args.keep_frames:
                frame_results = captions
            else:
                frame_results = [
                    FrameCaption(
                        index=item.index,
                        timestamp_sec=item.timestamp_sec,
                        image_path=None,
                        caption=item.caption,
                    )
                    for item in captions
                ]

        if temp_dir and not args.keep_frames:
            shutil.rmtree(temp_dir, ignore_errors=True)

        return VideoUnderstandingResult(
            video_path=str(video_path),
            duration_sec=duration,
            frame_count=len(frames),
            frames=frame_results,
            timeline=timeline,
            audio_transcript=audio_transcript,
            summary=summary,
            warnings=warnings,
            errors=errors,
        )
    def _resolve_video_path(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.config.effective_workdir / path
        path = path.resolve()
        if not path.exists():
            raise ToolError(f"Video not found: {path}")
        if path.is_dir():
            raise ToolError(f"Video path is a directory: {path}")
        return path

    def _resolve_ffmpeg(self) -> str:
        if self.config.ffmpeg_bin:
            return self.config.ffmpeg_bin
        found = shutil.which("ffmpeg")
        if found:
            return found
        return self._bundled_ffmpeg("ffmpeg")

    def _resolve_ffprobe(self) -> str:
        if self.config.ffprobe_bin:
            return self.config.ffprobe_bin
        found = shutil.which("ffprobe")
        if found:
            return found
        return self._bundled_ffmpeg("ffprobe")

    def _bundled_ffmpeg(self, name: str) -> str:
        suffix = ".exe" if sys.platform.startswith("win") else ""
        candidate = Path(__file__).with_name("ffmpeg") / "ffmpeg-8.0.1-essentials_build" / "bin" / f"{name}{suffix}"
        if candidate.exists():
            return str(candidate)
        raise ToolError(f"{name} not found in PATH or bundled tools.")

    def _probe_duration(self, ffprobe: str, path: Path, warnings: list[str]) -> float | None:
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            value = result.stdout.strip()
            return float(value) if value else None
        except Exception as exc:
            warnings.append(f"ffprobe failed: {exc}")
            return None

    def _resolve_fps(self, args: VideoUnderstandingArgs) -> float:
        if args.frame_rate is not None and args.frame_interval_sec is not None:
            raise ToolError("Provide frame_rate or frame_interval_sec, not both.")
        if args.frame_interval_sec is not None:
            interval = float(args.frame_interval_sec)
            if interval <= 0:
                raise ToolError("frame_interval_sec must be positive.")
            return 1.0 / interval
        fps = float(args.frame_rate if args.frame_rate is not None else self.config.frame_rate)
        if fps <= 0:
            raise ToolError("frame_rate must be positive.")
        return fps

    def _resolve_end_time(
        self, args: VideoUnderstandingArgs, duration: float | None
    ) -> float | None:
        if args.end_time_sec is not None:
            return float(args.end_time_sec)
        return duration

    def _prepare_output_dir(
        self, args: VideoUnderstandingArgs
    ) -> tuple[Path, str | None]:
        if args.output_dir:
            path = Path(args.output_dir).expanduser()
            if not path.is_absolute():
                path = self.config.effective_workdir / path
            path = path.resolve()
            path.mkdir(parents=True, exist_ok=True)
            return path, None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.config.output_dir / f"video_{timestamp}"
        path.mkdir(parents=True, exist_ok=True)
        return path, str(path)

    def _extract_frames(
        self,
        ffmpeg: str,
        video_path: Path,
        output_dir: Path,
        fps: float,
        start_time: float,
        end_time: float | None,
        max_frames: int,
        scale_width: int | None,
        warnings: list[str],
    ) -> list[Path]:
        if max_frames <= 0:
            raise ToolError("max_frames must be positive.")
        vf = [f"fps={fps:.6f}"]
        if scale_width is not None and scale_width > 0:
            vf.append(f"scale={int(scale_width)}:-1")
        filter_arg = ",".join(vf)

        output_pattern = output_dir / "frame_%06d.jpg"
        cmd = [ffmpeg, "-y", "-ss", f"{start_time:.3f}", "-i", str(video_path)]
        if end_time is not None:
            duration = max(0.0, end_time - start_time)
            cmd.extend(["-t", f"{duration:.3f}"])
        cmd.extend(["-vf", filter_arg, "-frames:v", str(max_frames), str(output_pattern)])

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise ToolError(stderr or "ffmpeg frame extraction failed.") from exc

        files = sorted(output_dir.glob("frame_*.jpg"))
        if not files:
            warnings.append("No frames extracted.")
        return files

    def _build_frames(self, files: list[Path], fps: float, start_time: float) -> list[_Frame]:
        frames: list[_Frame] = []
        for idx, path in enumerate(files, start=1):
            timestamp = start_time + (idx - 1) / fps
            frames.append(_Frame(index=idx, timestamp_sec=timestamp, path=path))
        return frames

    def _default_frame_prompt(self, question: str | None) -> str:
        if question:
            return (
                "Describe this video frame with focus on the question:\n"
                f"{question}\n"
                "Use 1-2 concise sentences."
            )
        return "Describe this video frame in 1-2 concise sentences."

    def _caption_frames(
        self,
        frames: list[_Frame],
        prompt: str,
        args: VideoUnderstandingArgs,
        warnings: list[str],
    ) -> list[FrameCaption]:
        captions: list[FrameCaption] = []
        for frame in frames:
            try:
                caption = self._call_vision(frame.path, prompt, args)
            except ToolError as exc:
                warnings.append(str(exc))
                caption = ""
            captions.append(
                FrameCaption(
                    index=frame.index,
                    timestamp_sec=frame.timestamp_sec,
                    image_path=str(frame.path),
                    caption=caption,
                )
            )
        return captions

    def _call_vision(self, image_path: Path, prompt: str, args: VideoUnderstandingArgs) -> str:
        backend = (args.vision_backend or "ollama").strip().lower()
        if backend != "ollama":
            raise ToolError("Only the ollama vision backend is supported.")

        model = args.vision_model or self.config.vision_model
        temperature = (
            args.vision_temperature
            if args.vision_temperature is not None
            else self.config.vision_temperature
        )
        max_tokens = (
            args.vision_max_tokens
            if args.vision_max_tokens is not None
            else self.config.vision_max_tokens
        )
        num_gpu = args.vision_num_gpu

        image_bytes = image_path.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        payload: dict[str, object] = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt, "images": [image_b64]}
            ],
            "stream": False,
        }
        options: dict[str, object] = {}
        if temperature is not None:
            options["temperature"] = float(temperature)
        if max_tokens is not None:
            options["num_predict"] = int(max_tokens)
        if num_gpu is not None:
            options["num_gpu"] = int(num_gpu)
        if options:
            payload["options"] = options

        data = json.dumps(payload).encode("utf-8")
        url = self.config.ollama_url.rstrip("/") + "/api/chat"
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                response = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise ToolError(f"Ollama vision call failed: {exc}") from exc

        if isinstance(response, dict) and response.get("error"):
            raise ToolError(f"Ollama error: {response.get('error')}")

        message = response.get("message") if isinstance(response, dict) else None
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"].strip()

        raise ToolError("Unexpected response from Ollama vision endpoint.")

    def _transcribe_audio(
        self,
        ffmpeg: str,
        video_path: Path,
        start_time: float,
        end_time: float | None,
        args: VideoUnderstandingArgs,
        warnings: list[str],
    ) -> str | None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "audio.wav"
            cmd = [ffmpeg, "-y", "-ss", f"{start_time:.3f}", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000"]
            if end_time is not None:
                duration = max(0.0, end_time - start_time)
                cmd.extend(["-t", f"{duration:.3f}"])
            cmd.append(str(audio_path))
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or "").strip()
                warnings.append(stderr or "ffmpeg audio extraction failed.")
                return None

            return self._run_stt(audio_path, args)

    def _run_stt(self, audio_path: Path, args: VideoUnderstandingArgs) -> str:
        backend = self._resolve_stt_backend(args.stt_backend)
        if backend == "faster_whisper":
            return self._transcribe_faster_whisper(audio_path, args)
        return self._transcribe_whisper(audio_path, args)

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
        self, path: Path, args: VideoUnderstandingArgs
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

    def _transcribe_whisper(self, path: Path, args: VideoUnderstandingArgs) -> str:
        model = self._get_whisper_model(args)
        try:
            result = model.transcribe(str(path))
        except Exception as exc:
            raise ToolError(f"whisper failed: {exc}") from exc
        return str(result.get("text", "")).strip()

    def _get_faster_whisper_model(self, args: VideoUnderstandingArgs) -> object:
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

    def _get_whisper_model(self, args: VideoUnderstandingArgs) -> object:
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

    def _build_timeline(
        self, captions: list[FrameCaption], args: VideoUnderstandingArgs
    ) -> list[TimelineSegment]:
        group_duration = (
            args.group_duration_sec
            if args.group_duration_sec is not None
            else self.config.group_duration_sec
        )
        max_captions = (
            args.max_captions_per_group
            if args.max_captions_per_group is not None
            else self.config.max_captions_per_group
        )
        if group_duration <= 0:
            group_duration = 0.0

        groups: dict[int, list[FrameCaption]] = {}
        for caption in captions:
            if group_duration == 0.0:
                group_id = caption.index
            else:
                group_id = int(caption.timestamp_sec // group_duration)
            groups.setdefault(group_id, []).append(caption)

        timeline: list[TimelineSegment] = []
        for group_id in sorted(groups.keys()):
            group = groups[group_id]
            if group_duration == 0.0:
                start_sec = group[0].timestamp_sec
                end_sec = group[0].timestamp_sec
            else:
                start_sec = group_id * group_duration
                end_sec = start_sec + group_duration
            captions_text = [item.caption for item in group if item.caption]
            if max_captions > 0:
                captions_text = captions_text[:max_captions]
            timeline.append(
                TimelineSegment(
                    start_sec=float(start_sec),
                    end_sec=float(end_sec),
                    captions=captions_text,
                )
            )
        return timeline

    def _summarize_video(
        self,
        captions: list[FrameCaption],
        timeline: list[TimelineSegment],
        audio_transcript: str | None,
        args: VideoUnderstandingArgs,
    ) -> tuple[dict | None, list[str]]:
        errors: list[str] = []
        lines: list[str] = []
        for segment in timeline:
            if not segment.captions:
                continue
            captions_text = "; ".join(segment.captions)
            lines.append(
                f"{segment.start_sec:.1f}-{segment.end_sec:.1f}s: {captions_text}"
            )
        if audio_transcript:
            lines.append("Audio transcript:")
            lines.append(audio_transcript)
        if args.question:
            lines.append("Question:")
            lines.append(args.question)

        context = "\n".join(lines).strip()
        if not context:
            return None, ["No captions available for summary."]

        payload, call_errors = self._call_llm_json(context, args)
        errors.extend(call_errors)
        if not isinstance(payload, dict):
            return None, errors
        return payload, errors

    def _call_llm_json(
        self, context: str, args: VideoUnderstandingArgs
    ) -> tuple[dict | None, list[str]]:
        errors: list[str] = []
        max_retries = (
            args.max_retries if args.max_retries is not None else self.config.max_retries
        )
        system_prompt = (
            "Reply ONLY with valid JSON that conforms to the JSON schema below. "
            "Do not include extra keys, comments, or markdown.\n"
            f"JSON schema:\n{json.dumps(SUMMARY_SCHEMA, ensure_ascii=True)}"
        )
        user_prompt = (
            "Summarize the video timeline below. Produce a concise summary, and include "
            "time-bounded events when possible. If a question is provided, answer it in the 'answer' field.\n\n"
            f"Timeline:\n{context}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(max_retries + 1):
            raw = self._call_llm(messages, args)
            parsed, parse_error = self._parse_json(raw)
            if parse_error:
                errors.append(parse_error)
                if attempt < max_retries:
                    messages = self._append_retry(messages, raw, parse_error)
                    continue
                return None, errors

            validation_errors = validate_args(SUMMARY_SCHEMA, parsed)
            if validation_errors:
                errors.extend(validation_errors)
                if attempt < max_retries:
                    messages = self._append_retry(
                        messages, raw, "; ".join(validation_errors)
                    )
                    continue
                return None, errors

            return parsed, errors
        return None, errors

    def _append_retry(
        self, messages: list[dict[str, str]], raw: str, error: str
    ) -> list[dict[str, str]]:
        retry_prompt = (
            "Your previous output was invalid.\n"
            f"Error: {error}\n"
            "Reply again with ONLY valid JSON that matches the schema."
        )
        return messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": retry_prompt},
        ]

    def _parse_json(self, raw: str) -> tuple[Any | None, str | None]:
        text = raw.strip()
        if not text:
            return None, "Empty output"
        try:
            return json.loads(text), None
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            start = min(
                [i for i in (text.find("{"), text.find("[")) if i != -1],
                default=-1,
            )
            if start == -1:
                return None, "No JSON object or array found"
            try:
                obj, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError as exc:
                return None, f"JSON parse error: {exc}"
            return obj, None

    def _call_llm(self, messages: list[dict[str, str]], args: VideoUnderstandingArgs) -> str:
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
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError as exc:
                raise ToolError(f"LLM response parse failed: {exc}") from exc
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
        if not isinstance(event.args, VideoUnderstandingArgs):
            return ToolCallDisplay(summary="video_understanding")
        return ToolCallDisplay(
            summary="video_understanding",
            details={
                "path": event.args.path,
                "question": event.args.question,
                "frame_rate": event.args.frame_rate,
                "audio_transcribe": event.args.audio_transcribe,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, VideoUnderstandingResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = f"Processed {event.result.frame_count} frame(s)"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "frame_count": event.result.frame_count,
                "duration_sec": event.result.duration_sec,
                "summary": event.result.summary,
                "audio_transcript": event.result.audio_transcript,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Understanding video"
