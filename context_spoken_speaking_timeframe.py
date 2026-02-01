from __future__ import annotations

import re
from datetime import datetime, timedelta
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


SENTENCE_RE = re.compile(r"[^.!?]+[.!?]*", re.S)
WORD_RE = re.compile(r"[A-Za-z0-9_']+")


class ContextSpokenSpeakingTimeframeConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_source_bytes: int = Field(
        default=5_000_000, description="Maximum bytes per content."
    )
    max_segments: int = Field(default=200, description="Maximum segments to return.")
    preview_chars: int = Field(default=200, description="Preview snippet length.")
    default_segment_by: str = Field(
        default="sentences", description="sentences, lines, or paragraphs."
    )
    min_segment_chars: int = Field(default=12, description="Minimum segment length.")
    default_pace: str = Field(default="steady", description="slow, steady, fast.")
    default_style: str = Field(default="neutral", description="Speech style label.")
    base_interval_ms: int = Field(default=350, description="Interval between segments.")
    min_interval_ms: int = Field(default=120, description="Minimum interval.")
    max_interval_ms: int = Field(default=1000, description="Maximum interval.")
    default_timezone: str = Field(
        default="local", description="Timezone label for timestamps."
    )


class ContextSpokenSpeakingTimeframeState(BaseToolState):
    pass


class ContextSpokenSpeakingTimeframeArgs(BaseModel):
    content: str | None = Field(default=None, description="Content to speak.")
    path: str | None = Field(default=None, description="Path to content.")
    segment_by: str | None = Field(
        default=None, description="sentences, lines, or paragraphs."
    )
    max_segments: int | None = Field(
        default=None, description="Override max segments."
    )
    start_time: str | None = Field(
        default=None,
        description="Start time (ISO 8601 or HH:MM:SS). Defaults to now.",
    )
    pace: str | None = Field(
        default=None, description="slow, steady, fast."
    )
    style: str | None = Field(default=None, description="Speech style label.")
    interval_ms: int | None = Field(
        default=None, description="Override interval between segments."
    )
    timezone_label: str | None = Field(
        default=None, description="Timezone label for output."
    )


class SpeakingTimeframeSegment(BaseModel):
    index: int
    scheduled_time: str
    text: str
    word_count: int
    pace: str
    style: str
    interval_ms: int
    cue: str
    preview: str


class ContextSpokenSpeakingTimeframeResult(BaseModel):
    segments: list[SpeakingTimeframeSegment]
    segment_count: int
    start_time: str
    end_time: str
    pace: str
    style: str
    interval_ms: int
    timezone_label: str
    speech_opening: str
    speech_closing: str
    warnings: list[str]


class ContextSpokenSpeakingTimeframe(
    BaseTool[
        ContextSpokenSpeakingTimeframeArgs,
        ContextSpokenSpeakingTimeframeResult,
        ContextSpokenSpeakingTimeframeConfig,
        ContextSpokenSpeakingTimeframeState,
    ],
    ToolUIData[
        ContextSpokenSpeakingTimeframeArgs, ContextSpokenSpeakingTimeframeResult
    ],
):
    description: ClassVar[str] = (
        "Plan when to speak and how to speak across a timeframe."
    )

    async def run(
        self, args: ContextSpokenSpeakingTimeframeArgs
    ) -> ContextSpokenSpeakingTimeframeResult:
        content = self._load_content(args)
        segment_by = (args.segment_by or self.config.default_segment_by).strip().lower()
        if segment_by not in {"sentences", "lines", "paragraphs"}:
            raise ToolError("segment_by must be sentences, lines, or paragraphs.")

        max_segments = (
            args.max_segments if args.max_segments is not None else self.config.max_segments
        )
        if max_segments <= 0:
            raise ToolError("max_segments must be positive.")

        pace = (args.pace or self.config.default_pace).strip().lower()
        if pace not in {"slow", "steady", "fast"}:
            raise ToolError("pace must be slow, steady, or fast.")

        style = args.style or self.config.default_style
        interval_ms = args.interval_ms if args.interval_ms is not None else self.config.base_interval_ms
        if interval_ms < 0:
            raise ToolError("interval_ms must be non-negative.")

        interval_ms = self._clamp_interval(interval_ms)
        timezone_label = args.timezone_label or self.config.default_timezone

        start_dt = self._parse_start_time(args.start_time)
        segments_raw = self._split_segments(content, segment_by)

        segments: list[SpeakingTimeframeSegment] = []
        warnings: list[str] = []
        current_time = start_dt

        for raw in segments_raw:
            if len(segments) >= max_segments:
                warnings.append("Segment limit reached; output truncated.")
                break
            if len(raw) < self.config.min_segment_chars:
                continue
            cue = self._build_cue(pace, style, interval_ms)
            segments.append(
                SpeakingTimeframeSegment(
                    index=len(segments) + 1,
                    scheduled_time=current_time.isoformat(),
                    text=raw.strip(),
                    word_count=len(WORD_RE.findall(raw)),
                    pace=pace,
                    style=style,
                    interval_ms=interval_ms,
                    cue=cue,
                    preview=self._preview(raw),
                )
            )
            current_time = current_time + timedelta(milliseconds=interval_ms)

        if not segments:
            raise ToolError("No segments generated.")

        speech_opening = f"Start speaking at {segments[0].scheduled_time} ({timezone_label})."
        speech_closing = "Keep pacing and style consistent until the final segment."

        return ContextSpokenSpeakingTimeframeResult(
            segments=segments,
            segment_count=len(segments),
            start_time=segments[0].scheduled_time,
            end_time=segments[-1].scheduled_time,
            pace=pace,
            style=style,
            interval_ms=interval_ms,
            timezone_label=timezone_label,
            speech_opening=speech_opening,
            speech_closing=speech_closing,
            warnings=warnings,
        )

    def _load_content(self, args: ContextSpokenSpeakingTimeframeArgs) -> str:
        if args.content and args.path:
            raise ToolError("Provide content or path, not both.")
        if args.content is not None:
            data = args.content.encode("utf-8")
            if len(data) > self.config.max_source_bytes:
                raise ToolError(
                    f"content exceeds max_source_bytes ({len(data)} > {self.config.max_source_bytes})."
                )
            return args.content
        if not args.path:
            raise ToolError("Provide content or path.")
        path = self._resolve_path(args.path)
        size = path.stat().st_size
        if size > self.config.max_source_bytes:
            raise ToolError(
                f"{path} exceeds max_source_bytes ({size} > {self.config.max_source_bytes})."
            )
        return path.read_text("utf-8", errors="ignore")

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.config.effective_workdir / path
        path = path.resolve()
        if not path.exists():
            raise ToolError(f"Path not found: {path}")
        if path.is_dir():
            raise ToolError(f"Path is a directory: {path}")
        return path

    def _split_segments(self, text: str, mode: str) -> list[str]:
        if mode == "lines":
            return [line for line in text.splitlines() if line.strip()]
        if mode == "paragraphs":
            return [chunk for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
        return [seg.strip() for seg in SENTENCE_RE.findall(text) if seg.strip()]

    def _parse_start_time(self, value: str | None) -> datetime:
        if not value:
            return datetime.now()
        value = value.strip()
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
        try:
            parts = value.split(":")
            if len(parts) == 3:
                hours, minutes, seconds = [int(part) for part in parts]
                now = datetime.now()
                return now.replace(hour=hours, minute=minutes, second=seconds, microsecond=0)
        except ValueError as exc:
            raise ToolError(f"Invalid start_time: {value}") from exc
        raise ToolError(f"Invalid start_time: {value}")

    def _build_cue(self, pace: str, style: str, interval_ms: int) -> str:
        return f"Speak {pace} with {style} style every {interval_ms}ms."

    def _clamp_interval(self, interval_ms: int) -> int:
        return max(self.config.min_interval_ms, min(self.config.max_interval_ms, interval_ms))

    def _preview(self, text: str) -> str:
        max_chars = self.config.preview_chars
        if max_chars <= 0:
            return ""
        return text if len(text) <= max_chars else text[:max_chars]

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextSpokenSpeakingTimeframeArgs):
            return ToolCallDisplay(summary="context_spoken_speaking_timeframe")
        return ToolCallDisplay(
            summary="context_spoken_speaking_timeframe",
            details={
                "path": event.args.path,
                "pace": event.args.pace,
                "style": event.args.style,
                "start_time": event.args.start_time,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpokenSpeakingTimeframeResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = f"Prepared {event.result.segment_count} speaking timeframe(s)"
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=event.result.warnings,
            details={
                "segment_count": event.result.segment_count,
                "start_time": event.result.start_time,
                "end_time": event.result.end_time,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Planning speaking timeframe"
