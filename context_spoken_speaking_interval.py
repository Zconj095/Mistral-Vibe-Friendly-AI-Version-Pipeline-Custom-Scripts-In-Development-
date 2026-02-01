from __future__ import annotations

import re
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


class ContextSpokenSpeakingIntervalConfig(BaseToolConfig):
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
    interval_mode: str = Field(
        default="punctuation", description="punctuation, uniform, adaptive."
    )
    base_interval_ms: int = Field(default=350, description="Base interval in ms.")
    short_interval_ms: int = Field(default=200, description="Short interval in ms.")
    long_interval_ms: int = Field(default=500, description="Long interval in ms.")
    min_interval_ms: int = Field(default=120, description="Minimum interval.")
    max_interval_ms: int = Field(default=1000, description="Maximum interval.")
    long_punct: list[str] = Field(
        default_factory=lambda: [".", "!", "?"],
        description="Punctuation triggering long intervals.",
    )
    short_punct: list[str] = Field(
        default_factory=lambda: [",", ";", ":"],
        description="Punctuation triggering short intervals.",
    )


class ContextSpokenSpeakingIntervalState(BaseToolState):
    pass


class ContextSpokenSpeakingIntervalArgs(BaseModel):
    content: str | None = Field(default=None, description="Content to analyze.")
    path: str | None = Field(default=None, description="Path to content.")
    segment_by: str | None = Field(
        default=None, description="sentences, lines, or paragraphs."
    )
    max_segments: int | None = Field(
        default=None, description="Override max segments."
    )
    interval_mode: str | None = Field(
        default=None, description="punctuation, uniform, adaptive."
    )
    base_interval_ms: int | None = Field(
        default=None, description="Override base interval."
    )
    short_interval_ms: int | None = Field(
        default=None, description="Override short interval."
    )
    long_interval_ms: int | None = Field(
        default=None, description="Override long interval."
    )
    min_interval_ms: int | None = Field(
        default=None, description="Override minimum interval."
    )
    max_interval_ms: int | None = Field(
        default=None, description="Override maximum interval."
    )


class SpeakingIntervalSegment(BaseModel):
    index: int
    text: str
    word_count: int
    interval_ms: int
    interval_type: str
    reason: str
    preview: str


class ContextSpokenSpeakingIntervalResult(BaseModel):
    segments: list[SpeakingIntervalSegment]
    segment_count: int
    total_interval_ms: int
    average_interval_ms: float
    speech_opening: str
    speech_closing: str
    warnings: list[str]


class ContextSpokenSpeakingInterval(
    BaseTool[
        ContextSpokenSpeakingIntervalArgs,
        ContextSpokenSpeakingIntervalResult,
        ContextSpokenSpeakingIntervalConfig,
        ContextSpokenSpeakingIntervalState,
    ],
    ToolUIData[
        ContextSpokenSpeakingIntervalArgs, ContextSpokenSpeakingIntervalResult
    ],
):
    description: ClassVar[str] = (
        "Plan speaking intervals between segments."
    )

    async def run(
        self, args: ContextSpokenSpeakingIntervalArgs
    ) -> ContextSpokenSpeakingIntervalResult:
        content = self._load_content(args)
        segment_by = (args.segment_by or self.config.default_segment_by).strip().lower()
        if segment_by not in {"sentences", "lines", "paragraphs"}:
            raise ToolError("segment_by must be sentences, lines, or paragraphs.")

        max_segments = (
            args.max_segments if args.max_segments is not None else self.config.max_segments
        )
        if max_segments <= 0:
            raise ToolError("max_segments must be positive.")

        mode = (args.interval_mode or self.config.interval_mode).strip().lower()
        if mode not in {"punctuation", "uniform", "adaptive"}:
            raise ToolError("interval_mode must be punctuation, uniform, or adaptive.")

        base_interval = (
            args.base_interval_ms if args.base_interval_ms is not None else self.config.base_interval_ms
        )
        short_interval = (
            args.short_interval_ms if args.short_interval_ms is not None else self.config.short_interval_ms
        )
        long_interval = (
            args.long_interval_ms if args.long_interval_ms is not None else self.config.long_interval_ms
        )
        min_interval = (
            args.min_interval_ms if args.min_interval_ms is not None else self.config.min_interval_ms
        )
        max_interval = (
            args.max_interval_ms if args.max_interval_ms is not None else self.config.max_interval_ms
        )
        for value in (base_interval, short_interval, long_interval, min_interval, max_interval):
            if value < 0:
                raise ToolError("interval values must be non-negative.")

        segments_raw = self._split_segments(content, segment_by)
        segments: list[SpeakingIntervalSegment] = []
        warnings: list[str] = []
        total_interval = 0

        for raw in segments_raw:
            if len(segments) >= max_segments:
                warnings.append("Segment limit reached; output truncated.")
                break
            if len(raw) < self.config.min_segment_chars:
                continue
            interval_ms, interval_type, reason = self._interval_for_segment(
                raw,
                mode,
                base_interval,
                short_interval,
                long_interval,
                min_interval,
                max_interval,
            )
            total_interval += interval_ms
            segments.append(
                SpeakingIntervalSegment(
                    index=len(segments) + 1,
                    text=raw.strip(),
                    word_count=len(WORD_RE.findall(raw)),
                    interval_ms=interval_ms,
                    interval_type=interval_type,
                    reason=reason,
                    preview=self._preview(raw),
                )
            )

        if not segments:
            raise ToolError("No segments generated.")

        average_interval = total_interval / len(segments)
        speech_opening = f"Use {mode} speaking intervals with {base_interval}ms base."
        speech_closing = "Keep intervals consistent for steady pacing."

        return ContextSpokenSpeakingIntervalResult(
            segments=segments,
            segment_count=len(segments),
            total_interval_ms=total_interval,
            average_interval_ms=average_interval,
            speech_opening=speech_opening,
            speech_closing=speech_closing,
            warnings=warnings,
        )

    def _load_content(self, args: ContextSpokenSpeakingIntervalArgs) -> str:
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

    def _interval_for_segment(
        self,
        text: str,
        mode: str,
        base_interval: int,
        short_interval: int,
        long_interval: int,
        min_interval: int,
        max_interval: int,
    ) -> tuple[int, str, str]:
        stripped = text.strip()
        if mode == "uniform":
            interval = base_interval
            return self._clamp_interval(interval, min_interval, max_interval), "uniform", "uniform mode"

        if mode == "adaptive":
            word_count = len(WORD_RE.findall(stripped))
            interval = base_interval
            if word_count <= 6:
                interval = max(min_interval, base_interval - 80)
                return interval, "short", "short segment"
            if word_count >= 20:
                interval = min(max_interval, base_interval + 120)
                return interval, "long", "long segment"
            return self._clamp_interval(interval, min_interval, max_interval), "base", "adaptive base"

        last_char = stripped[-1] if stripped else ""
        if last_char in self.config.long_punct:
            interval = long_interval
            return self._clamp_interval(interval, min_interval, max_interval), "long", f"ending {last_char}"
        if last_char in self.config.short_punct:
            interval = short_interval
            return self._clamp_interval(interval, min_interval, max_interval), "short", f"ending {last_char}"
        return self._clamp_interval(base_interval, min_interval, max_interval), "base", "default pacing"

    def _clamp_interval(self, value: int, min_interval: int, max_interval: int) -> int:
        return max(min_interval, min(max_interval, value))

    def _preview(self, text: str) -> str:
        max_chars = self.config.preview_chars
        if max_chars <= 0:
            return ""
        return text if len(text) <= max_chars else text[:max_chars]

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextSpokenSpeakingIntervalArgs):
            return ToolCallDisplay(summary="context_spoken_speaking_interval")
        return ToolCallDisplay(
            summary="context_spoken_speaking_interval",
            details={
                "path": event.args.path,
                "interval_mode": event.args.interval_mode,
                "segment_by": event.args.segment_by,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpokenSpeakingIntervalResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = (
            f"Prepared {event.result.segment_count} speaking interval(s)"
        )
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=event.result.warnings,
            details={
                "segment_count": event.result.segment_count,
                "average_interval_ms": event.result.average_interval_ms,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Planning speaking intervals"
