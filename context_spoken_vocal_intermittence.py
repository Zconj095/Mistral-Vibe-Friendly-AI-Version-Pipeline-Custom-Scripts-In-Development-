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
PAUSE_MARKER_RE = re.compile(
    r"\\[(pause|silence|break)(?:\\s+(?P<ms>\\d+))?\\]", re.IGNORECASE
)


class ContextSpokenVocalIntermittenceConfig(BaseToolConfig):
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
    short_pause_ms: int = Field(default=200, description="Short pause duration.")
    long_pause_ms: int = Field(default=600, description="Long pause duration.")
    long_punct: list[str] = Field(
        default_factory=lambda: [".", "!", "?"],
        description="Punctuation that triggers long pauses.",
    )
    short_punct: list[str] = Field(
        default_factory=lambda: [",", ";", ":"],
        description="Punctuation that triggers short pauses.",
    )
    intermittence_threshold: float = Field(
        default=0.3, description="Ratio threshold for high intermittence."
    )


class ContextSpokenVocalIntermittenceState(BaseToolState):
    pass


class ContextSpokenVocalIntermittenceArgs(BaseModel):
    content: str | None = Field(default=None, description="Content to analyze.")
    path: str | None = Field(default=None, description="Path to content.")
    segment_by: str | None = Field(
        default=None, description="sentences, lines, or paragraphs."
    )
    max_segments: int | None = Field(
        default=None, description="Override max segments."
    )
    short_pause_ms: int | None = Field(
        default=None, description="Override short pause duration."
    )
    long_pause_ms: int | None = Field(
        default=None, description="Override long pause duration."
    )


class IntermittenceSegment(BaseModel):
    index: int
    text: str
    word_count: int
    pause_ms: int
    pause_type: str
    reason: str
    preview: str


class ContextSpokenVocalIntermittenceResult(BaseModel):
    segments: list[IntermittenceSegment]
    segment_count: int
    intermittence_ratio: float
    intermittence_label: str
    total_pause_ms: int
    average_pause_ms: float
    speech_opening: str
    speech_closing: str
    warnings: list[str]


class ContextSpokenVocalIntermittence(
    BaseTool[
        ContextSpokenVocalIntermittenceArgs,
        ContextSpokenVocalIntermittenceResult,
        ContextSpokenVocalIntermittenceConfig,
        ContextSpokenVocalIntermittenceState,
    ],
    ToolUIData[
        ContextSpokenVocalIntermittenceArgs, ContextSpokenVocalIntermittenceResult
    ],
):
    description: ClassVar[str] = "Measure vocal intermittence and pause cadence."

    async def run(
        self, args: ContextSpokenVocalIntermittenceArgs
    ) -> ContextSpokenVocalIntermittenceResult:
        content = self._load_content(args)
        segment_by = (args.segment_by or self.config.default_segment_by).strip().lower()
        if segment_by not in {"sentences", "lines", "paragraphs"}:
            raise ToolError("segment_by must be sentences, lines, or paragraphs.")

        max_segments = (
            args.max_segments if args.max_segments is not None else self.config.max_segments
        )
        if max_segments <= 0:
            raise ToolError("max_segments must be positive.")

        short_pause_ms = (
            args.short_pause_ms if args.short_pause_ms is not None else self.config.short_pause_ms
        )
        long_pause_ms = (
            args.long_pause_ms if args.long_pause_ms is not None else self.config.long_pause_ms
        )
        if short_pause_ms < 0 or long_pause_ms < 0:
            raise ToolError("pause durations must be non-negative.")

        segments_raw = self._split_segments(content, segment_by)
        segments: list[IntermittenceSegment] = []
        warnings: list[str] = []
        total_pause_ms = 0
        pause_segments = 0

        for raw in segments_raw:
            if len(segments) >= max_segments:
                warnings.append("Segment limit reached; output truncated.")
                break
            if len(raw) < self.config.min_segment_chars:
                continue
            pause_ms, pause_type, reason = self._pause_for_segment(
                raw, short_pause_ms, long_pause_ms
            )
            if pause_ms > 0:
                pause_segments += 1
            total_pause_ms += pause_ms
            segments.append(
                IntermittenceSegment(
                    index=len(segments) + 1,
                    text=raw.strip(),
                    word_count=len(WORD_RE.findall(raw)),
                    pause_ms=pause_ms,
                    pause_type=pause_type,
                    reason=reason,
                    preview=self._preview(raw),
                )
            )

        if not segments:
            raise ToolError("No segments generated.")

        intermittence_ratio = pause_segments / len(segments)
        label = self._label(intermittence_ratio)
        average_pause_ms = total_pause_ms / len(segments)

        speech_opening = "Use intentional pauses to manage intermittence."
        speech_closing = "Balance silence with continuous speech for clarity."

        return ContextSpokenVocalIntermittenceResult(
            segments=segments,
            segment_count=len(segments),
            intermittence_ratio=intermittence_ratio,
            intermittence_label=label,
            total_pause_ms=total_pause_ms,
            average_pause_ms=average_pause_ms,
            speech_opening=speech_opening,
            speech_closing=speech_closing,
            warnings=warnings,
        )

    def _load_content(self, args: ContextSpokenVocalIntermittenceArgs) -> str:
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

    def _pause_for_segment(
        self, text: str, short_pause_ms: int, long_pause_ms: int
    ) -> tuple[int, str, str]:
        stripped = text.strip()
        marker = PAUSE_MARKER_RE.search(stripped)
        if marker:
            value = marker.group("ms")
            pause_ms = int(value) if value else long_pause_ms
            return pause_ms, "explicit", "pause marker"
        if stripped:
            last_char = stripped[-1]
            if last_char in self.config.long_punct:
                return long_pause_ms, "long", f"ending {last_char}"
            if last_char in self.config.short_punct:
                return short_pause_ms, "short", f"ending {last_char}"
        return 0, "none", "continuous"

    def _label(self, ratio: float) -> str:
        if ratio >= self.config.intermittence_threshold:
            return "high"
        if ratio >= self.config.intermittence_threshold / 2:
            return "moderate"
        return "low"

    def _preview(self, text: str) -> str:
        max_chars = self.config.preview_chars
        if max_chars <= 0:
            return ""
        return text if len(text) <= max_chars else text[:max_chars]

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextSpokenVocalIntermittenceArgs):
            return ToolCallDisplay(summary="context_spoken_vocal_intermittence")
        return ToolCallDisplay(
            summary="context_spoken_vocal_intermittence",
            details={
                "path": event.args.path,
                "segment_by": event.args.segment_by,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpokenVocalIntermittenceResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = (
            f"Intermittence {event.result.intermittence_label} "
            f"({event.result.intermittence_ratio:.2f})"
        )
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=event.result.warnings,
            details={
                "segment_count": event.result.segment_count,
                "intermittence_ratio": event.result.intermittence_ratio,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Analyzing vocal intermittence"
