from __future__ import annotations

from pathlib import Path
import re
from typing import ClassVar, TYPE_CHECKING

from pydantic import BaseModel, Field

from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    ToolCallDisplay,
    ToolError,
    ToolPermission,
    ToolResultDisplay,
    ToolUIData,
)

if TYPE_CHECKING:
    from vibe.core.types import ToolCallEvent, ToolResultEvent


WORD_RE = re.compile(r"[A-Za-z0-9_']+")


class ContextSpeechPatternSegmentationConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_items: int = Field(default=40, description="Maximum items to evaluate.")
    max_source_bytes: int = Field(default=3_000_000, description="Maximum bytes per item.")
    max_total_bytes: int = Field(default=20_000_000, description="Max bytes across items.")
    preview_chars: int = Field(default=400, description="Preview snippet length.")
    max_segments: int = Field(default=300, description="Maximum segments per item.")
    max_segment_chars: int = Field(default=800, description="Max characters per segment.")
    short_pause_ms: int = Field(default=250, description="Pause for commas/short breaks.")
    medium_pause_ms: int = Field(default=450, description="Pause for semicolons/colons.")
    long_pause_ms: int = Field(default=700, description="Pause for sentence boundaries.")
    newline_pause_ms: int = Field(default=650, description="Pause for newlines.")
    short_word_threshold: int = Field(default=6, description="Words for short bursts.")
    long_word_threshold: int = Field(default=25, description="Words for long runs.")
    default_wpm: int = Field(default=160, description="Words per minute for estimates.")


class ContextSpeechPatternSegmentationState(BaseToolState):
    pass


class SpeechPatternItem(BaseModel):
    id: str | None = Field(default=None)
    name: str | None = Field(default=None)
    content: str | None = Field(default=None)
    path: str | None = Field(default=None)
    source: str | None = Field(default=None)


class SpeechPatternSegment(BaseModel):
    index: int
    start: int
    end: int
    text: str
    char_count: int
    word_count: int
    boundary: str
    base_pattern: str
    size_class: str
    pause_ms: int
    estimated_duration_ms: int
    tags: list[str]


class SpeechPatternSummary(BaseModel):
    total_segments: int
    total_words: int
    total_chars: int
    estimated_total_duration_ms: int
    pattern_counts: dict[str, int]
    size_counts: dict[str, int]


class SpeechPatternInsight(BaseModel):
    index: int
    id: str | None
    name: str | None
    source_path: str | None
    preview: str
    segments: list[SpeechPatternSegment]
    summary: SpeechPatternSummary


class ContextSpeechPatternSegmentationResult(BaseModel):
    items: list[SpeechPatternInsight]
    item_count: int
    truncated: bool
    warnings: list[str]
    errors: list[str]


class ContextSpeechPatternSegmentation(
    BaseTool[
        SpeechPatternItem,
        ContextSpeechPatternSegmentationResult,
        ContextSpeechPatternSegmentationConfig,
        ContextSpeechPatternSegmentationState,
    ],
    ToolUIData[SpeechPatternItem, ContextSpeechPatternSegmentationResult],
):
    description: ClassVar[str] = "Segment speech patterns from text."

    async def run(
        self, args: SpeechPatternItem | list[SpeechPatternItem]
    ) -> ContextSpeechPatternSegmentationResult:
        items = args if isinstance(args, list) else [args]
        if not items:
            raise ToolError("items is required.")

        errors: list[str] = []
        warnings: list[str] = []
        insights: list[SpeechPatternInsight] = []
        total_bytes = 0
        truncated = False

        for idx, item in enumerate(items, start=1):
            try:
                content, source_path, size_bytes = self._load_item(item)
                if size_bytes is None:
                    raise ToolError("Item has no content.")
                if total_bytes + size_bytes > self.config.max_total_bytes:
                    truncated = True
                    warnings.append("Budget exceeded; stopping evaluation.")
                    break
                total_bytes += size_bytes

                segments = self._segment_text(content)
                summary = self._summarize_segments(segments)

                insights.append(
                    SpeechPatternInsight(
                        index=len(insights) + 1,
                        id=item.id,
                        name=item.name,
                        source_path=source_path,
                        preview=self._preview(content),
                        segments=segments[: self.config.max_segments],
                        summary=summary,
                    )
                )
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not insights:
            raise ToolError("No insights generated.")

        return ContextSpeechPatternSegmentationResult(
            items=insights,
            item_count=len(insights),
            truncated=truncated,
            warnings=warnings,
            errors=errors,
        )

    def _segment_text(self, text: str) -> list[SpeechPatternSegment]:
        segments: list[SpeechPatternSegment] = []
        boundary_chars = {".", "!", "?", ",", ";", ":", "\n"}
        start = 0

        def add_segment(raw_start: int, raw_end: int, boundary: str) -> None:
            slice_text = text[raw_start:raw_end]
            if not slice_text:
                return
            lstrip = len(slice_text) - len(slice_text.lstrip())
            rstrip = len(slice_text) - len(slice_text.rstrip())
            seg_start = raw_start + lstrip
            seg_end = raw_end - rstrip
            if seg_end <= seg_start:
                return
            seg_text = text[seg_start:seg_end]
            for part_start, part_end, part_text in self._split_long_segment(
                seg_text, seg_start
            ):
                pause_ms, base_pattern = self._format_boundary(boundary)
                word_count = self._word_count(part_text)
                size_class = self._size_class(word_count)
                tags = self._tags(base_pattern, size_class, word_count)
                duration_ms = self._estimate_duration_ms(word_count, pause_ms)
                segments.append(
                    SpeechPatternSegment(
                        index=len(segments) + 1,
                        start=part_start,
                        end=part_end,
                        text=part_text,
                        char_count=len(part_text),
                        word_count=word_count,
                        boundary=boundary,
                        base_pattern=base_pattern,
                        size_class=size_class,
                        pause_ms=pause_ms,
                        estimated_duration_ms=duration_ms,
                        tags=tags,
                    )
                )

        for idx, char in enumerate(text):
            if char in boundary_chars:
                end = idx + 1
                add_segment(start, end, char)
                start = end

        if start < len(text):
            add_segment(start, len(text), "")

        return segments[: self.config.max_segments]

    def _split_long_segment(
        self, seg_text: str, base_start: int
    ) -> list[tuple[int, int, str]]:
        max_chars = self.config.max_segment_chars
        if max_chars <= 0 or len(seg_text) <= max_chars:
            return [(base_start, base_start + len(seg_text), seg_text)]

        parts: list[tuple[int, int, str]] = []
        offset = 0
        while offset < len(seg_text):
            remaining = seg_text[offset:]
            if len(remaining) <= max_chars:
                parts.append((base_start + offset, base_start + offset + len(remaining), remaining))
                break
            cut = remaining.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            chunk = remaining[:cut]
            lstrip = len(chunk) - len(chunk.lstrip())
            rstrip = len(chunk) - len(chunk.rstrip())
            part_start = base_start + offset + lstrip
            part_end = base_start + offset + cut - rstrip
            part_text = seg_text[offset + lstrip : offset + cut - rstrip]
            if part_text:
                parts.append((part_start, part_end, part_text))
            offset += cut
        return parts

    def _format_boundary(self, boundary: str) -> tuple[int, str]:
        if boundary == "?":
            return self.config.long_pause_ms, "question"
        if boundary == "!":
            return self.config.long_pause_ms, "exclaim"
        if boundary in {".", "\n"}:
            pause = self.config.newline_pause_ms if boundary == "\n" else self.config.long_pause_ms
            return pause, "statement"
        if boundary in {";", ":"}:
            return self.config.medium_pause_ms, "clause"
        if boundary == ",":
            return self.config.short_pause_ms, "phrase"
        return self.config.short_pause_ms, "fragment"

    def _size_class(self, word_count: int) -> str:
        if word_count <= self.config.short_word_threshold:
            return "short_burst"
        if word_count >= self.config.long_word_threshold:
            return "long_run"
        return "normal"

    def _tags(self, base_pattern: str, size_class: str, word_count: int) -> list[str]:
        tags = [base_pattern, size_class]
        if word_count <= self.config.short_word_threshold:
            tags.append("staccato")
        elif word_count >= self.config.long_word_threshold:
            tags.append("flowing")
        else:
            tags.append("steady")
        return tags

    def _estimate_duration_ms(self, word_count: int, pause_ms: int) -> int:
        wpm = max(self.config.default_wpm, 1)
        speaking_ms = int((word_count / wpm) * 60_000)
        return speaking_ms + pause_ms

    def _word_count(self, text: str) -> int:
        return len(WORD_RE.findall(text))

    def _summarize_segments(self, segments: list[SpeechPatternSegment]) -> SpeechPatternSummary:
        pattern_counts: dict[str, int] = {}
        size_counts: dict[str, int] = {}
        total_words = 0
        total_chars = 0
        total_duration = 0

        for segment in segments:
            pattern_counts[segment.base_pattern] = pattern_counts.get(segment.base_pattern, 0) + 1
            size_counts[segment.size_class] = size_counts.get(segment.size_class, 0) + 1
            total_words += segment.word_count
            total_chars += segment.char_count
            total_duration += segment.estimated_duration_ms

        return SpeechPatternSummary(
            total_segments=len(segments),
            total_words=total_words,
            total_chars=total_chars,
            estimated_total_duration_ms=total_duration,
            pattern_counts=pattern_counts,
            size_counts=size_counts,
        )

    def _load_item(self, item: SpeechPatternItem) -> tuple[str, str | None, int | None]:
        if item.content and item.path:
            raise ToolError("Provide content or path, not both.")
        if item.path:
            path = Path(item.path).expanduser()
            if not path.is_absolute():
                path = self.config.effective_workdir / path
            path = path.resolve()
            if not path.exists():
                raise ToolError(f"Path not found: {path}")
            if path.is_dir():
                raise ToolError(f"Path is a directory: {path}")
            size = path.stat().st_size
            if size > self.config.max_source_bytes:
                raise ToolError(
                    f"{path} exceeds max_source_bytes ({size} > {self.config.max_source_bytes})."
                )
            return path.read_text("utf-8", errors="ignore"), str(path), size
        if item.content is not None:
            size = len(item.content.encode("utf-8"))
            if size > self.config.max_source_bytes:
                raise ToolError(
                    f"content exceeds max_source_bytes ({size} > {self.config.max_source_bytes})."
                )
            return item.content, None, size
        return "", None, 0

    def _preview(self, text: str) -> str:
        max_chars = self.config.preview_chars
        if max_chars <= 0:
            return ""
        return text if len(text) <= max_chars else text[:max_chars]

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if isinstance(event.args, SpeechPatternItem):
            return ToolCallDisplay(
                summary="context_speech_pattern_segmentation",
                details={"id": event.args.id, "name": event.args.name},
            )
        return ToolCallDisplay(summary="context_speech_pattern_segmentation")

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpeechPatternSegmentationResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=(
                f"Segmented {event.result.item_count} item(s) into "
                f"{event.result.items[0].summary.total_segments if event.result.items else 0} patterns"
            ),
            warnings=event.result.warnings,
            details={
                "item_count": event.result.item_count,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Segmenting speech patterns"
