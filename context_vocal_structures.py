from __future__ import annotations

from pathlib import Path
from typing import ClassVar, TYPE_CHECKING

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


class ContextVocalStructuresConfig(BaseToolConfig):
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


class ContextVocalStructuresState(BaseToolState):
    pass


class VocalItem(BaseModel):
    id: str | None = Field(default=None)
    name: str | None = Field(default=None)
    content: str | None = Field(default=None)
    path: str | None = Field(default=None)
    source: str | None = Field(default=None)



class ContextVocalStructuresArgs(BaseModel):
    items: list[VocalItem] = Field(description="Items to evaluate.")

class VocalSegment(BaseModel):
    index: int
    start: int
    end: int
    text: str
    char_count: int
    word_count: int
    boundary: str
    segment_type: str
    pause_ms: int
    intonation: str


class VocalStructureInsight(BaseModel):
    index: int
    id: str | None
    name: str | None
    source_path: str | None
    preview: str
    segments: list[VocalSegment]
    segment_count: int
    total_words: int
    total_chars: int


class ContextVocalStructuresResult(BaseModel):
    items: list[VocalStructureInsight]
    item_count: int
    total_segments: int
    truncated: bool
    warnings: list[str]
    errors: list[str]


class ContextVocalStructures(
    BaseTool[
        ContextVocalStructuresArgs,
        ContextVocalStructuresResult,
        ContextVocalStructuresConfig,
        ContextVocalStructuresState,
    ],
    ToolUIData[ContextVocalStructuresArgs, ContextVocalStructuresResult],
):
    description: ClassVar[str] = (
        "Form vocal structures by segmenting text into speech-friendly units."
    )

    async def run(self, args: ContextVocalStructuresArgs) -> ContextVocalStructuresResult:
        items = list(args.items)
        if not items:
            raise ToolError("items is required.")

        errors: list[str] = []
        warnings: list[str] = []
        insights: list[VocalStructureInsight] = []
        total_bytes = 0
        truncated = False
        total_segments = 0

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
                total_segments += len(segments)
                total_words = sum(seg.word_count for seg in segments)
                total_chars = sum(seg.char_count for seg in segments)

                insights.append(
                    VocalStructureInsight(
                        index=len(insights) + 1,
                        id=item.id,
                        name=item.name,
                        source_path=source_path,
                        preview=self._preview(content),
                        segments=segments[: self.config.max_segments],
                        segment_count=len(segments),
                        total_words=total_words,
                        total_chars=total_chars,
                    )
                )
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not insights:
            raise ToolError("No insights generated.")

        return ContextVocalStructuresResult(
            items=insights,
            item_count=len(insights),
            total_segments=total_segments,
            truncated=truncated,
            warnings=warnings,
            errors=errors,
        )

    def _segment_text(self, text: str) -> list[VocalSegment]:
        segments: list[VocalSegment] = []
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
                pause_ms, intonation, segment_type = self._format_boundary(boundary)
                if part_text != seg_text:
                    pause_ms = self.config.short_pause_ms
                    intonation = "neutral"
                    segment_type = "fragment"
                segments.append(
                    VocalSegment(
                        index=len(segments) + 1,
                        start=part_start,
                        end=part_end,
                        text=part_text,
                        char_count=len(part_text),
                        word_count=self._word_count(part_text),
                        boundary=boundary,
                        segment_type=segment_type,
                        pause_ms=pause_ms,
                        intonation=intonation,
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

    def _format_boundary(self, boundary: str) -> tuple[int, str, str]:
        if boundary == "?":
            return self.config.long_pause_ms, "rising", "sentence"
        if boundary == "!":
            return self.config.long_pause_ms, "exclaim", "sentence"
        if boundary in {".", "\n"}:
            pause = self.config.newline_pause_ms if boundary == "\n" else self.config.long_pause_ms
            return pause, "falling", "sentence"
        if boundary in {";", ":"}:
            return self.config.medium_pause_ms, "continuation", "phrase"
        if boundary == ",":
            return self.config.short_pause_ms, "continuation", "phrase"
        return self.config.short_pause_ms, "neutral", "fragment"

    def _word_count(self, text: str) -> int:
        return len([part for part in text.strip().split() if part])

    def _load_item(self, item: VocalItem) -> tuple[str, str | None, int | None]:
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
        if not isinstance(event.args, ContextVocalStructuresArgs):
            return ToolCallDisplay(summary="context_vocal_structures")
        if not event.args.items:
            return ToolCallDisplay(summary="context_vocal_structures")
        first = event.args.items[0]
        return ToolCallDisplay(
            summary="context_vocal_structures",
            details={
                "item_count": len(event.args.items),
                "id": first.id,
                "name": first.name,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextVocalStructuresResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=(
                f"Analyzed {event.result.item_count} item(s) with "
                f"{event.result.total_segments} vocal segments"
            ),
            warnings=event.result.warnings,
            details={
                "item_count": event.result.item_count,
                "total_segments": event.result.total_segments,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Forming vocal structures"
