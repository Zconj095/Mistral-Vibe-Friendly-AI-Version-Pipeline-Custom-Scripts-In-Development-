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


class ContextSpokenBreathPausesConfig(BaseToolConfig):
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
    short_pause_ms: int = Field(default=250, description="Short pause duration.")
    long_pause_ms: int = Field(default=700, description="Long pause duration.")
    breath_pause_ms: int = Field(
        default=500, description="Breath pause duration."
    )
    breath_interval_words: int = Field(
        default=18, description="Words between breath pauses."
    )
    breath_interval_sentences: int = Field(
        default=2, description="Sentences between breath pauses."
    )
    long_pause_punct: list[str] = Field(
        default_factory=lambda: [".", "!", "?"],
        description="Punctuation triggering long pauses.",
    )
    short_pause_punct: list[str] = Field(
        default_factory=lambda: [",", ";", ":"],
        description="Punctuation triggering short pauses.",
    )


class ContextSpokenBreathPausesState(BaseToolState):
    pass


class ContextSpokenBreathPausesArgs(BaseModel):
    content: str | None = Field(default=None, description="Content to analyze.")
    path: str | None = Field(default=None, description="Path to content.")
    segment_by: str | None = Field(
        default=None, description="sentences, lines, or paragraphs."
    )
    max_segments: int | None = Field(
        default=None, description="Override max segments."
    )
    short_pause_ms: int | None = Field(default=None, description="Override short pause.")
    long_pause_ms: int | None = Field(default=None, description="Override long pause.")
    breath_pause_ms: int | None = Field(
        default=None, description="Override breath pause."
    )
    breath_interval_words: int | None = Field(
        default=None, description="Override words between breaths."
    )
    breath_interval_sentences: int | None = Field(
        default=None, description="Override sentences between breaths."
    )
    include_breaths: bool = Field(
        default=True, description="Include breath pauses."
    )


class BreathPauseSegment(BaseModel):
    index: int
    text: str
    word_count: int
    pause_ms: int
    pause_type: str
    breath_after: bool
    reason: str
    preview: str


class ContextSpokenBreathPausesResult(BaseModel):
    segments: list[BreathPauseSegment]
    segment_count: int
    total_pause_ms: int
    short_pause_count: int
    long_pause_count: int
    breath_pause_count: int
    warnings: list[str]
    speech_opening: str
    speech_closing: str


class ContextSpokenBreathPauses(
    BaseTool[
        ContextSpokenBreathPausesArgs,
        ContextSpokenBreathPausesResult,
        ContextSpokenBreathPausesConfig,
        ContextSpokenBreathPausesState,
    ],
    ToolUIData[
        ContextSpokenBreathPausesArgs, ContextSpokenBreathPausesResult
    ],
):
    description: ClassVar[str] = (
        "Plan short and long pauses with intermittent breath breaks."
    )

    async def run(
        self, args: ContextSpokenBreathPausesArgs
    ) -> ContextSpokenBreathPausesResult:
        content = self._load_content(args)
        segment_by = (args.segment_by or self.config.default_segment_by).strip().lower()
        if segment_by not in {"sentences", "lines", "paragraphs"}:
            raise ToolError("segment_by must be sentences, lines, or paragraphs.")

        max_segments = (
            args.max_segments if args.max_segments is not None else self.config.max_segments
        )
        if max_segments <= 0:
            raise ToolError("max_segments must be positive.")

        short_pause_ms = args.short_pause_ms if args.short_pause_ms is not None else self.config.short_pause_ms
        long_pause_ms = args.long_pause_ms if args.long_pause_ms is not None else self.config.long_pause_ms
        breath_pause_ms = (
            args.breath_pause_ms if args.breath_pause_ms is not None else self.config.breath_pause_ms
        )
        breath_interval_words = (
            args.breath_interval_words
            if args.breath_interval_words is not None
            else self.config.breath_interval_words
        )
        breath_interval_sentences = (
            args.breath_interval_sentences
            if args.breath_interval_sentences is not None
            else self.config.breath_interval_sentences
        )
        if short_pause_ms < 0 or long_pause_ms < 0 or breath_pause_ms < 0:
            raise ToolError("pause durations must be non-negative.")
        if breath_interval_words <= 0 or breath_interval_sentences <= 0:
            raise ToolError("breath intervals must be positive.")

        segments_raw = self._split_segments(content, segment_by)
        segments: list[BreathPauseSegment] = []
        warnings: list[str] = []
        total_pause_ms = 0
        short_pause_count = 0
        long_pause_count = 0
        breath_pause_count = 0

        words_since_breath = 0
        sentences_since_breath = 0

        for raw in segments_raw:
            if len(segments) >= max_segments:
                warnings.append("Segment limit reached; output truncated.")
                break
            if len(raw) < self.config.min_segment_chars:
                continue
            word_count = len(WORD_RE.findall(raw))
            pause_ms, pause_type, reason = self._pause_for_segment(
                raw, short_pause_ms, long_pause_ms
            )
            breath_after = False

            if args.include_breaths:
                words_since_breath += word_count
                sentences_since_breath += 1
                if (
                    words_since_breath >= breath_interval_words
                    or sentences_since_breath >= breath_interval_sentences
                ):
                    breath_after = True
                    breath_pause_count += 1
                    words_since_breath = 0
                    sentences_since_breath = 0
                    pause_type = "breath"
                    pause_ms = max(pause_ms, breath_pause_ms)
                    reason = "breath interval reached"

            if pause_type == "short":
                short_pause_count += 1
            elif pause_type == "long":
                long_pause_count += 1

            total_pause_ms += pause_ms

            segments.append(
                BreathPauseSegment(
                    index=len(segments) + 1,
                    text=raw.strip(),
                    word_count=word_count,
                    pause_ms=pause_ms,
                    pause_type=pause_type,
                    breath_after=breath_after,
                    reason=reason,
                    preview=self._preview(raw),
                )
            )

        if not segments:
            raise ToolError("No segments generated.")

        return ContextSpokenBreathPausesResult(
            segments=segments,
            segment_count=len(segments),
            total_pause_ms=total_pause_ms,
            short_pause_count=short_pause_count,
            long_pause_count=long_pause_count,
            breath_pause_count=breath_pause_count,
            warnings=warnings,
            speech_opening="Use short pauses for pacing and long pauses for emphasis.",
            speech_closing="Keep breath pauses consistent to avoid rushed delivery.",
        )

    def _load_content(self, args: ContextSpokenBreathPausesArgs) -> str:
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
        if not stripped:
            return short_pause_ms, "short", "empty segment"
        last_char = stripped[-1]
        if last_char in self.config.long_pause_punct:
            return long_pause_ms, "long", f"ending punctuation {last_char}"
        if last_char in self.config.short_pause_punct:
            return short_pause_ms, "short", f"ending punctuation {last_char}"
        return short_pause_ms, "short", "default pacing"

    def _preview(self, text: str) -> str:
        max_chars = self.config.preview_chars
        if max_chars <= 0:
            return ""
        return text if len(text) <= max_chars else text[:max_chars]

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextSpokenBreathPausesArgs):
            return ToolCallDisplay(summary="context_spoken_breath_pauses")
        return ToolCallDisplay(
            summary="context_spoken_breath_pauses",
            details={
                "path": event.args.path,
                "segment_by": event.args.segment_by,
                "include_breaths": event.args.include_breaths,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpokenBreathPausesResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = (
            f"Planned {event.result.segment_count} pause segment(s) "
            f"with {event.result.breath_pause_count} breath break(s)"
        )
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=event.result.warnings,
            details={
                "segment_count": event.result.segment_count,
                "total_pause_ms": event.result.total_pause_ms,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Planning spoken breath pauses"
