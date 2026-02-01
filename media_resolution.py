from __future__ import annotations

import base64
import json
import mimetypes
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from urllib import request

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


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".wmv", ".m4v"}
TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".rst",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
    ".tsv",
    ".html",
    ".xml",
    ".ini",
    ".cfg",
    ".log",
    ".css",
    ".js",
    ".ts",
    ".py",
}


@dataclass(frozen=True)
class _MediaInfo:
    media_type: str
    mime_type: str | None
    metadata: dict[str, Any]


class MediaResolutionConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    max_files: int = Field(default=8, description="Maximum files per call.")
    max_text_chars: int = Field(
        default=5000, description="Max characters for extracted text."
    )
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
    frame_rate: float = Field(default=1.0, description="Default frame rate.")
    max_frames: int = Field(default=24, description="Default max frames.")
    output_dir: Path = Field(
        default=Path.home() / ".vibe" / "media_frames",
        description="Default output directory for frames.",
    )
    group_duration_sec: float = Field(
        default=10.0, description="Default seconds per timeline group."
    )
    max_captions_per_group: int = Field(
        default=12, description="Default captions per timeline group."
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
    ffmpeg_bin: str | None = Field(
        default=None, description="Override ffmpeg path."
    )
    ffprobe_bin: str | None = Field(
        default=None, description="Override ffprobe path."
    )


class MediaResolutionState(BaseToolState):
    pass


class MediaResolutionArgs(BaseModel):
    path: str | None = Field(default=None, description="Single file path.")
    paths: list[str] | None = Field(default=None, description="File paths.")
    max_files: int | None = Field(default=None, description="Max files per call.")
    max_text_chars: int | None = Field(
        default=None, description="Override max characters for text."
    )
    include_bundle_text: bool = Field(
        default=True, description="Include a bundle text summary."
    )
    include_frames: bool = Field(
        default=False, description="Include per-frame captions."
    )
    keep_frames: bool = Field(
        default=False, description="Keep extracted frames on disk."
    )
    output_dir: str | None = Field(
        default=None, description="Output directory for frames."
    )
    image_caption: bool = Field(
        default=True, description="Generate captions for images."
    )
    video_caption: bool = Field(
        default=True, description="Caption sampled video frames."
    )
    audio_transcribe: bool = Field(
        default=True, description="Transcribe audio files."
    )
    video_audio_transcribe: bool = Field(
        default=False, description="Transcribe audio tracks from videos."
    )
    frame_rate: float | None = Field(
        default=None, description="Frames per second to sample."
    )
    frame_interval_sec: float | None = Field(
        default=None, description="Seconds between sampled frames."
    )
    max_frames: int | None = Field(
        default=None, description="Max frames to sample."
    )
    scale_width: int | None = Field(
        default=None, description="Scale frames to this width."
    )
    group_duration_sec: float | None = Field(
        default=None, description="Seconds per timeline group."
    )
    max_captions_per_group: int | None = Field(
        default=None, description="Max captions per timeline group."
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
        default=None, description="Vision num_gpu option."
    )
    image_prompt: str | None = Field(
        default=None, description="Prompt for image captions."
    )
    frame_prompt: str | None = Field(
        default=None, description="Prompt for video frame captions."
    )
    stt_backend: str | None = Field(
        default=None, description="Override STT backend."
    )
    stt_model: str | None = Field(
        default=None, description="Override STT model."
    )
    stt_device: str | None = Field(
        default=None, description="Override STT device."
    )
    stt_compute_type: str | None = Field(
        default=None, description="Override STT compute type."
    )
    ffmpeg_bin: str | None = Field(
        default=None, description="Override ffmpeg path."
    )
    ffprobe_bin: str | None = Field(
        default=None, description="Override ffprobe path."
    )


_STT_MODEL_CACHE: dict[tuple[str, str, str, str], object] = {}


class FrameCaption(BaseModel):
    index: int
    timestamp_sec: float
    image_path: str | None
    caption: str


class TimelineSegment(BaseModel):
    start_sec: float
    end_sec: float
    captions: list[str]


class ResolvedMediaItem(BaseModel):
    path: str
    media_type: str
    mime_type: str | None
    size_bytes: int | None
    metadata: dict[str, Any] | None
    text: str | None
    transcript: str | None
    frames: list[FrameCaption] | None
    timeline: list[TimelineSegment] | None
    warnings: list[str]
    errors: list[str]


class MediaResolutionResult(BaseModel):
    items: list[ResolvedMediaItem]
    bundle_text: str | None
    warnings: list[str]
    errors: list[str]


class MediaResolution(
    BaseTool[
        MediaResolutionArgs,
        MediaResolutionResult,
        MediaResolutionConfig,
        MediaResolutionState,
    ],
    ToolUIData[MediaResolutionArgs, MediaResolutionResult],
):
    description: ClassVar[str] = (
        "Resolve media files into normalized text, transcripts, and captions."
    )

    async def run(self, args: MediaResolutionArgs) -> MediaResolutionResult:
        warnings: list[str] = []
        errors: list[str] = []

        paths = self._resolve_paths(args)
        if not paths:
            raise ToolError("No valid paths provided.")

        items: list[ResolvedMediaItem] = []
        for path in paths:
            item = self._resolve_media_item(path, args)
            items.append(item)
            if item.warnings:
                warnings.extend(item.warnings)
            if item.errors:
                errors.extend(item.errors)

        bundle_text = None
        if args.include_bundle_text:
            bundle_text = self._build_bundle_text(items, args)

        return MediaResolutionResult(
            items=items,
            bundle_text=bundle_text,
            warnings=warnings,
            errors=errors,
        )

    def _resolve_paths(self, args: MediaResolutionArgs) -> list[Path]:
        if args.path and args.paths:
            raise ToolError("Provide path or paths, not both.")

        raw_paths = []
        if args.path:
            raw_paths = [args.path]
        elif args.paths:
            raw_paths = list(args.paths)
        else:
            raise ToolError("path or paths is required.")

        max_files = (
            args.max_files if args.max_files is not None else self.config.max_files
        )
        if max_files <= 0:
            raise ToolError("max_files must be positive.")

        resolved: list[Path] = []
        for raw in raw_paths:
            if len(resolved) >= max_files:
                break
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = self.config.effective_workdir / path
            path = path.resolve()
            if not path.exists():
                raise ToolError(f"Path not found: {path}")
            if path.is_dir():
                raise ToolError(f"Path is a directory: {path}")
            resolved.append(path)
        return resolved

    def _resolve_media_item(
        self, path: Path, args: MediaResolutionArgs
    ) -> ResolvedMediaItem:
        item_warnings: list[str] = []
        item_errors: list[str] = []
        info = self._detect_media(path, args, item_warnings)
        metadata = dict(info.metadata)
        text = None
        transcript = None
        frames = None
        timeline = None

        try:
            if info.media_type == "image":
                metadata.update(self._image_metadata(path, item_warnings))
                text = self._resolve_image(path, args)
            elif info.media_type == "audio":
                metadata.update(self._audio_metadata(path, args, item_warnings))
                transcript = self._resolve_audio(path, args, item_warnings)
            elif info.media_type == "video":
                metadata.update(self._video_metadata(path, args, item_warnings))
                text, transcript, frames, timeline = self._resolve_video(
                    path, args, item_warnings
                )
            elif info.media_type == "pdf":
                text = self._resolve_pdf(path, args, metadata, item_warnings)
            elif info.media_type == "text":
                text = self._resolve_text(path, args, metadata, item_warnings)
            else:
                item_warnings.append("Unsupported media type; skipped processing.")
        except ToolError as exc:
            item_errors.append(str(exc))

        max_chars = args.max_text_chars or self.config.max_text_chars
        if text:
            text = self._truncate_text(text, max_chars)
        if transcript:
            transcript = self._truncate_text(transcript, max_chars)

        return ResolvedMediaItem(
            path=str(path),
            media_type=info.media_type,
            mime_type=info.mime_type,
            size_bytes=metadata.get("size_bytes"),
            metadata=metadata,
            text=text,
            transcript=transcript,
            frames=frames,
            timeline=timeline,
            warnings=item_warnings,
            errors=item_errors,
        )

    def _detect_media(
        self, path: Path, args: MediaResolutionArgs, warnings: list[str]
    ) -> _MediaInfo:
        mime_type, _ = mimetypes.guess_type(str(path))
        ext = path.suffix.lower()
        metadata: dict[str, Any] = {"size_bytes": path.stat().st_size}

        if ext == ".pdf":
            return _MediaInfo("pdf", mime_type or "application/pdf", metadata)
        if ext in IMAGE_EXTENSIONS or (mime_type or "").startswith("image/"):
            return _MediaInfo("image", mime_type, metadata)
        if ext in AUDIO_EXTENSIONS or (mime_type or "").startswith("audio/"):
            return _MediaInfo("audio", mime_type, metadata)
        if ext in VIDEO_EXTENSIONS or (mime_type or "").startswith("video/"):
            return _MediaInfo("video", mime_type, metadata)
        if ext in TEXT_EXTENSIONS or (mime_type or "").startswith("text/"):
            return _MediaInfo("text", mime_type, metadata)

        ffprobe = self._resolve_ffprobe(args)
        if ffprobe:
            probe = self._probe_media(ffprobe, path, warnings)
            streams = probe.get("streams", [])
            format_data = probe.get("format", {})
            metadata.update(self._format_metadata(format_data))
            media_type = self._media_type_from_streams(streams)
            if media_type:
                return _MediaInfo(media_type, mime_type, metadata)

        return _MediaInfo("unknown", mime_type, metadata)

    def _resolve_ffmpeg(self, args: MediaResolutionArgs) -> str | None:
        if args.ffmpeg_bin:
            return args.ffmpeg_bin
        if self.config.ffmpeg_bin:
            return self.config.ffmpeg_bin
        found = shutil.which("ffmpeg")
        if found:
            return found
        return self._bundled_ffmpeg("ffmpeg")

    def _resolve_ffprobe(self, args: MediaResolutionArgs) -> str | None:
        if args.ffprobe_bin:
            return args.ffprobe_bin
        if self.config.ffprobe_bin:
            return self.config.ffprobe_bin
        found = shutil.which("ffprobe")
        if found:
            return found
        return self._bundled_ffmpeg("ffprobe")

    def _bundled_ffmpeg(self, name: str) -> str | None:
        suffix = ".exe" if sys.platform.startswith("win") else ""
        candidate = (
            Path(__file__).with_name("ffmpeg")
            / "ffmpeg-8.0.1-essentials_build"
            / "bin"
            / f"{name}{suffix}"
        )
        if candidate.exists():
            return str(candidate)
        return None

    def _probe_media(
        self, ffprobe: str, path: Path, warnings: list[str]
    ) -> dict[str, Any]:
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            return json.loads(result.stdout or "{}")
        except Exception as exc:
            warnings.append(f"ffprobe failed: {exc}")
            return {}

    def _media_type_from_streams(self, streams: list[dict[str, Any]]) -> str | None:
        has_video = any(stream.get("codec_type") == "video" for stream in streams)
        has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
        if has_video:
            return "video"
        if has_audio:
            return "audio"
        return None

    def _format_metadata(self, fmt: dict[str, Any]) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        duration = fmt.get("duration")
        if duration is not None:
            try:
                metadata["duration_sec"] = float(duration)
            except (TypeError, ValueError):
                pass
        size = fmt.get("size")
        if size is not None:
            try:
                metadata["size_bytes"] = int(size)
            except (TypeError, ValueError):
                pass
        bit_rate = fmt.get("bit_rate")
        if bit_rate is not None:
            try:
                metadata["bit_rate"] = int(bit_rate)
            except (TypeError, ValueError):
                pass
        return metadata

    def _resolve_image(self, path: Path, args: MediaResolutionArgs) -> str | None:
        if not args.image_caption:
            return None
        prompt = args.image_prompt or "Describe this image in 1-2 concise sentences."
        return self._call_vision(path, prompt, args)

    def _image_metadata(self, path: Path, warnings: list[str]) -> dict[str, Any]:
        try:
            from PIL import Image
        except ModuleNotFoundError:
            return {}

        try:
            with Image.open(path) as img:
                return {
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "mode": img.mode,
                }
        except Exception as exc:
            warnings.append(f"PIL failed: {exc}")
            return {}

    def _call_vision(
        self, image_path: Path, prompt: str, args: MediaResolutionArgs
    ) -> str:
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
        req = request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=120) as resp:
                response = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise ToolError(f"Ollama vision call failed: {exc}") from exc

        if isinstance(response, dict) and response.get("error"):
            raise ToolError(f"Ollama error: {response.get('error')}")

        message = response.get("message") if isinstance(response, dict) else None
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"].strip()

        raise ToolError("Unexpected response from Ollama vision endpoint.")

    def _audio_metadata(
        self, path: Path, args: MediaResolutionArgs, warnings: list[str]
    ) -> dict[str, Any]:
        ffprobe = self._resolve_ffprobe(args)
        if not ffprobe:
            return {}
        probe = self._probe_media(ffprobe, path, warnings)
        metadata = self._format_metadata(probe.get("format", {}))
        for stream in probe.get("streams", []):
            if stream.get("codec_type") != "audio":
                continue
            metadata["codec"] = stream.get("codec_name")
            metadata["sample_rate"] = self._parse_int(stream.get("sample_rate"))
            metadata["channels"] = self._parse_int(stream.get("channels"))
            metadata["channel_layout"] = stream.get("channel_layout")
            if "duration" in stream:
                metadata["duration_sec"] = self._parse_float(stream.get("duration"))
            break
        return {k: v for k, v in metadata.items() if v is not None}

    def _video_metadata(
        self, path: Path, args: MediaResolutionArgs, warnings: list[str]
    ) -> dict[str, Any]:
        ffprobe = self._resolve_ffprobe(args)
        if not ffprobe:
            return {}
        probe = self._probe_media(ffprobe, path, warnings)
        metadata = self._format_metadata(probe.get("format", {}))
        for stream in probe.get("streams", []):
            if stream.get("codec_type") != "video":
                continue
            metadata["codec"] = stream.get("codec_name")
            metadata["width"] = self._parse_int(stream.get("width"))
            metadata["height"] = self._parse_int(stream.get("height"))
            metadata["frame_rate"] = self._parse_fraction(
                stream.get("avg_frame_rate") or stream.get("r_frame_rate")
            )
            if "duration" in stream:
                metadata["duration_sec"] = self._parse_float(stream.get("duration"))
            break
        return {k: v for k, v in metadata.items() if v is not None}

    def _resolve_audio(
        self, path: Path, args: MediaResolutionArgs, warnings: list[str]
    ) -> str | None:
        if not args.audio_transcribe:
            return None
        try:
            return self._run_stt(path, args)
        except ToolError as exc:
            warnings.append(str(exc))
            return None

    def _resolve_video(
        self,
        path: Path,
        args: MediaResolutionArgs,
        warnings: list[str],
    ) -> tuple[str | None, str | None, list[FrameCaption] | None, list[TimelineSegment] | None]:
        ffmpeg = self._resolve_ffmpeg(args)
        if not ffmpeg:
            raise ToolError("ffmpeg is required to process videos.")

        fps = self._resolve_fps(args)
        max_frames = args.max_frames if args.max_frames is not None else self.config.max_frames
        if max_frames <= 0:
            raise ToolError("max_frames must be positive.")

        frame_captions = None
        timeline = None
        text = None
        temp_dir = None

        if args.video_caption:
            output_dir, temp_dir = self._prepare_output_dir(args)
            frame_files = self._extract_frames(
                ffmpeg,
                path,
                output_dir,
                fps,
                max_frames,
                args.scale_width,
                warnings,
            )
            frames = self._build_frames(frame_files, fps)
            if frames:
                prompt = (
                    args.frame_prompt
                    or "Describe this video frame in 1-2 concise sentences."
                )
                captions = self._caption_frames(frames, prompt, args, warnings)
                timeline = self._build_timeline(captions, args)
                text = self._timeline_text(timeline)
                if args.include_frames:
                    frame_captions = captions

        transcript = None
        if args.video_audio_transcribe:
            transcript = self._transcribe_video_audio(ffmpeg, path, args, warnings)

        if temp_dir and not args.keep_frames:
            shutil.rmtree(temp_dir, ignore_errors=True)

        return text, transcript, frame_captions, timeline

    def _resolve_pdf(
        self,
        path: Path,
        args: MediaResolutionArgs,
        metadata: dict[str, Any],
        warnings: list[str],
    ) -> str | None:
        text = self._extract_pdf_text(path, warnings, metadata)
        if not text:
            warnings.append("No text extracted from PDF.")
            return None
        metadata["char_count"] = len(text)
        metadata["line_count"] = text.count("\n") + 1 if text else 0
        return text

    def _resolve_text(
        self,
        path: Path,
        args: MediaResolutionArgs,
        metadata: dict[str, Any],
        warnings: list[str],
    ) -> str | None:
        try:
            content = path.read_text("utf-8", errors="ignore")
        except OSError as exc:
            warnings.append(f"Failed to read text: {exc}")
            return None
        metadata["char_count"] = len(content)
        metadata["line_count"] = content.count("\n") + 1 if content else 0
        return content.strip() or None

    def _truncate_text(self, text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return text
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "..."

    def _build_bundle_text(
        self, items: list[ResolvedMediaItem], args: MediaResolutionArgs
    ) -> str:
        parts: list[str] = []
        max_chars = args.max_text_chars or self.config.max_text_chars
        for item in items:
            header = f"[{item.media_type.upper()}] {item.path}"
            parts.append(header)
            if item.text:
                parts.append(self._truncate_text(item.text, max_chars))
            if item.transcript:
                parts.append("Transcript:")
                parts.append(self._truncate_text(item.transcript, max_chars))
            if item.timeline:
                parts.append("Timeline:")
                parts.extend(self._timeline_lines(item.timeline))
        return "\n".join(parts).strip()

    def _resolve_fps(self, args: MediaResolutionArgs) -> float:
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

    def _prepare_output_dir(
        self, args: MediaResolutionArgs
    ) -> tuple[Path, str | None]:
        if args.output_dir:
            path = Path(args.output_dir).expanduser()
            if not path.is_absolute():
                path = self.config.effective_workdir / path
            path = path.resolve()
            path.mkdir(parents=True, exist_ok=True)
            return path, None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.config.output_dir / f"media_{timestamp}"
        path.mkdir(parents=True, exist_ok=True)
        return path, str(path)

    def _extract_frames(
        self,
        ffmpeg: str,
        video_path: Path,
        output_dir: Path,
        fps: float,
        max_frames: int,
        scale_width: int | None,
        warnings: list[str],
    ) -> list[Path]:
        vf = [f"fps={fps:.6f}"]
        if scale_width is not None and scale_width > 0:
            vf.append(f"scale={int(scale_width)}:-1")
        filter_arg = ",".join(vf)

        output_pattern = output_dir / "frame_%06d.jpg"
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-vf",
            filter_arg,
            "-frames:v",
            str(max_frames),
            str(output_pattern),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise ToolError(stderr or "ffmpeg frame extraction failed.") from exc

        files = sorted(output_dir.glob("frame_*.jpg"))
        if not files:
            warnings.append("No frames extracted.")
        return files

    def _build_frames(self, files: list[Path], fps: float) -> list[FrameCaption]:
        frames: list[FrameCaption] = []
        for idx, path in enumerate(files, start=1):
            timestamp = (idx - 1) / fps
            frames.append(
                FrameCaption(
                    index=idx,
                    timestamp_sec=timestamp,
                    image_path=str(path),
                    caption="",
                )
            )
        return frames

    def _caption_frames(
        self,
        frames: list[FrameCaption],
        prompt: str,
        args: MediaResolutionArgs,
        warnings: list[str],
    ) -> list[FrameCaption]:
        captions: list[FrameCaption] = []
        for frame in frames:
            if not frame.image_path:
                continue
            try:
                caption = self._call_vision(Path(frame.image_path), prompt, args)
            except ToolError as exc:
                warnings.append(str(exc))
                caption = ""
            captions.append(
                FrameCaption(
                    index=frame.index,
                    timestamp_sec=frame.timestamp_sec,
                    image_path=frame.image_path if args.keep_frames else None,
                    caption=caption,
                )
            )
        return captions

    def _build_timeline(
        self, captions: list[FrameCaption], args: MediaResolutionArgs
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

    def _timeline_text(self, timeline: list[TimelineSegment]) -> str:
        return "\n".join(self._timeline_lines(timeline))

    def _timeline_lines(self, timeline: list[TimelineSegment]) -> list[str]:
        lines: list[str] = []
        for segment in timeline:
            if not segment.captions:
                continue
            captions_text = "; ".join(segment.captions)
            lines.append(
                f"{segment.start_sec:.1f}-{segment.end_sec:.1f}s: {captions_text}"
            )
        return lines

    def _transcribe_video_audio(
        self,
        ffmpeg: str,
        video_path: Path,
        args: MediaResolutionArgs,
        warnings: list[str],
    ) -> str | None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "audio.wav"
            cmd = [
                ffmpeg,
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(audio_path),
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or "").strip()
                warnings.append(stderr or "ffmpeg audio extraction failed.")
                return None

            try:
                return self._run_stt(audio_path, args)
            except ToolError as exc:
                warnings.append(str(exc))
                return None

    def _extract_pdf_text(
        self,
        path: Path,
        warnings: list[str],
        metadata: dict[str, Any],
    ) -> str | None:
        text = self._extract_text_with_pypdf(path, metadata)
        if text is not None:
            return text

        text = self._extract_text_with_pdftotext(path, warnings, metadata)
        if text is not None:
            return text

        warnings.append(
            "No PDF text extractor available. Install 'pypdf' or add 'pdftotext' to PATH."
        )
        return None

    def _extract_text_with_pypdf(
        self, path: Path, metadata: dict[str, Any]
    ) -> str | None:
        try:
            import pypdf
        except ModuleNotFoundError:
            return None

        try:
            reader = pypdf.PdfReader(str(path))
        except Exception as exc:
            raise ToolError(f"Failed to open PDF: {exc}") from exc

        pages = reader.pages
        metadata["page_count"] = len(pages)
        chunks: list[str] = []
        for page_index in range(len(pages)):
            page_text = pages[page_index].extract_text() or ""
            chunks.append(page_text)
        return "\n".join(chunks)

    def _extract_text_with_pdftotext(
        self,
        path: Path,
        warnings: list[str],
        metadata: dict[str, Any],
    ) -> str | None:
        if not shutil.which("pdftotext"):
            return None

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                out_path = Path(tmp_dir) / "out.txt"
                cmd = ["pdftotext", "-layout", str(path), str(out_path)]
                proc = subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if proc.stderr:
                    stderr = proc.stderr.strip()
                    if stderr:
                        warnings.append(f"pdftotext error: {stderr}")
                return out_path.read_text("utf-8", errors="ignore")
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else "pdftotext failed."
            warnings.append(stderr)
            return None
        except OSError as exc:
            warnings.append(f"pdftotext failed: {exc}")
            return None

    def _parse_int(self, value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _parse_float(self, value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _parse_fraction(self, value: Any) -> float | None:
        if not isinstance(value, str):
            return self._parse_float(value)
        if "/" in value:
            parts = value.split("/", 1)
            try:
                num = float(parts[0])
                denom = float(parts[1])
                if denom == 0:
                    return None
                return num / denom
            except (TypeError, ValueError):
                return None
        return self._parse_float(value)

    def _run_stt(self, audio_path: Path, args: MediaResolutionArgs) -> str:
        backend = self._resolve_stt_backend(args)
        if backend == "faster_whisper":
            return self._transcribe_faster_whisper(audio_path, args)
        return self._transcribe_whisper(audio_path, args)

    def _resolve_stt_backend(self, args: MediaResolutionArgs) -> str:
        backend = (args.stt_backend or self.config.stt_backend).strip().lower()
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
        self, path: Path, args: MediaResolutionArgs
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

    def _transcribe_whisper(self, path: Path, args: MediaResolutionArgs) -> str:
        model = self._get_whisper_model(args)
        try:
            result = model.transcribe(str(path))
        except Exception as exc:
            raise ToolError(f"whisper failed: {exc}") from exc
        return str(result.get("text", "")).strip()

    def _get_faster_whisper_model(self, args: MediaResolutionArgs) -> object:
        model_name = args.stt_model or self.config.stt_model
        device = args.stt_device or self.config.stt_device
        compute_type = args.stt_compute_type or self.config.stt_compute_type
        key = (model_name, device, compute_type, "faster_whisper")
        cached = _STT_MODEL_CACHE.get(key)
        if cached is not None:
            return cached

        try:
            from faster_whisper import WhisperModel
        except ModuleNotFoundError as exc:
            raise ToolError("faster-whisper is not installed.") from exc

        resolved_device = self._resolve_device(device)
        model = WhisperModel(
            model_name,
            device=resolved_device,
            compute_type=compute_type,
        )
        _STT_MODEL_CACHE[key] = model
        return model

    def _get_whisper_model(self, args: MediaResolutionArgs) -> object:
        model_name = args.stt_model or self.config.stt_model
        device = args.stt_device or self.config.stt_device
        key = (model_name, device, "whisper", "")
        cached = _STT_MODEL_CACHE.get(key)
        if cached is not None:
            return cached

        try:
            import whisper
        except ModuleNotFoundError as exc:
            raise ToolError("whisper is not installed.") from exc

        resolved_device = None if device == "auto" else device
        model = whisper.load_model(model_name, device=resolved_device)
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
        if not isinstance(event.args, MediaResolutionArgs):
            return ToolCallDisplay(summary="media_resolution")
        return ToolCallDisplay(
            summary="media_resolution",
            details={
                "paths": event.args.paths,
                "image_caption": event.args.image_caption,
                "audio_transcribe": event.args.audio_transcribe,
                "video_caption": event.args.video_caption,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, MediaResolutionResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = f"Resolved {len(event.result.items)} item(s)"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "items": event.result.items,
                "bundle_text": event.result.bundle_text,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Resolving media"
