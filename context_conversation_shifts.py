from __future__ import annotations

import re
from dataclasses import dataclass
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


WORD_RE = re.compile(r"[A-Za-z0-9_]+")
SPEAKER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _.'-]{0,40}):\s*(.*)$")


@dataclass
class _SegmentTokenStats:
    tokens: list[str]
    unique: set[str]


class ContextConversationShiftsConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_items: int = Field(default=40, description="Maximum items to evaluate.")
    max_source_bytes: int = Field(default=3_000_000, description="Maximum bytes per item.")
    max_total_bytes: int = Field(default=20_000_000, description="Max bytes across items.")
    preview_chars: int = Field(default=400, description="Preview snippet length.")
    max_segments: int = Field(default=200, description="Maximum segments per item.")
    min_segment_chars: int = Field(default=20, description="Minimum segment length.")
    min_tokens_for_similarity: int = Field(default=6, description="Min tokens for similarity.")
    similarity_threshold: float = Field(default=0.2, description="Topic shift Jaccard threshold.")
    max_events: int = Field(default=200, description="Maximum shift events per item.")
    topic_markers: list[str] = Field(
        default_factory=lambda: [
            "anyway",
            "by the way",
            "on another note",
            "on a different note",
            "switching topics",
            "moving on",
            "new topic",
            "different topic",
            "back to",
            "returning to",
            "let's talk about",
        ],
        description="Markers that often signal topic shifts.",
    )


class ContextConversationShiftsState(BaseToolState):
    pass


class ConversationItem(BaseModel):
    id: str | None = Field(default=None)
    name: str | None = Field(default=None)
    content: str | None = Field(default=None)
    path: str | None = Field(default=None)
    source: str | None = Field(default=None)



class ContextConversationShiftsArgs(BaseModel):
    items: list[ConversationItem] = Field(description="Items to evaluate.")

class ConversationSegment(BaseModel):
    index: int
    speaker: str | None
    text: str
    start: int
    end: int
    token_count: int


class ShiftEvent(BaseModel):
    index: int
    shift_types: list[str]
    reasons: list[str]
    from_segment: int
    to_segment: int
    speaker_from: str | None
    speaker_to: str | None
    similarity: float | None
    shift_score: float


class ConversationShiftInsight(BaseModel):
    index: int
    id: str | None
    name: str | None
    source_path: str | None
    preview: str
    segments: list[ConversationSegment]
    shifts: list[ShiftEvent]
    segment_count: int
    shift_count: int


class ContextConversationShiftsResult(BaseModel):
    items: list[ConversationShiftInsight]
    item_count: int
    total_shifts: int
    truncated: bool
    warnings: list[str]
    errors: list[str]


class ContextConversationShifts(
    BaseTool[
        ContextConversationShiftsArgs,
        ContextConversationShiftsResult,
        ContextConversationShiftsConfig,
        ContextConversationShiftsState,
    ],
    ToolUIData[ContextConversationShiftsArgs, ContextConversationShiftsResult],
):
    description: ClassVar[str] = (
        "Interpret conversation shifts (speaker or topic changes) in speech-like text."
    )

    async def run(self, args: ContextConversationShiftsArgs) -> ContextConversationShiftsResult:
        items = list(args.items)
        if not items:
            raise ToolError("items is required.")

        errors: list[str] = []
        warnings: list[str] = []
        insights: list[ConversationShiftInsight] = []
        total_bytes = 0
        truncated = False
        total_shifts = 0

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

                segments = self._segment_conversation(content)
                shifts = self._detect_shifts(segments, content)
                total_shifts += len(shifts)

                insights.append(
                    ConversationShiftInsight(
                        index=len(insights) + 1,
                        id=item.id,
                        name=item.name,
                        source_path=source_path,
                        preview=self._preview(content),
                        segments=segments[: self.config.max_segments],
                        shifts=shifts[: self.config.max_events],
                        segment_count=len(segments),
                        shift_count=len(shifts),
                    )
                )
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not insights:
            raise ToolError("No insights generated.")

        return ContextConversationShiftsResult(
            items=insights,
            item_count=len(insights),
            total_shifts=total_shifts,
            truncated=truncated,
            warnings=warnings,
            errors=errors,
        )

    def _segment_conversation(self, text: str) -> list[ConversationSegment]:
        segments: list[ConversationSegment] = []
        lines = text.splitlines()
        current_lines: list[str] = []
        current_speaker: str | None = None
        seg_start = 0
        cursor = 0

        def flush(end_offset: int) -> None:
            nonlocal current_lines, current_speaker, seg_start
            if not current_lines:
                return
            seg_text = "\n".join(current_lines).strip()
            if not seg_text:
                current_lines = []
                return
            if len(seg_text) < self.config.min_segment_chars and segments:
                segments[-1].text = f"{segments[-1].text}\n{seg_text}".strip()
                segments[-1].end = end_offset
                segments[-1].token_count = self._token_count(segments[-1].text)
            else:
                segments.append(
                    ConversationSegment(
                        index=len(segments) + 1,
                        speaker=current_speaker,
                        text=seg_text,
                        start=seg_start,
                        end=end_offset,
                        token_count=self._token_count(seg_text),
                    )
                )
            current_lines = []
            current_speaker = None

        for line in lines:
            line_len = len(line) + 1
            speaker_match = SPEAKER_RE.match(line)
            if speaker_match:
                flush(cursor)
                current_speaker = speaker_match.group(1).strip()
                current_lines.append(speaker_match.group(2))
                seg_start = cursor
            elif not line.strip():
                flush(cursor)
            else:
                if not current_lines:
                    seg_start = cursor
                current_lines.append(line)
            cursor += line_len

        flush(cursor)
        return segments[: self.config.max_segments]

    def _detect_shifts(
        self, segments: list[ConversationSegment], text: str
    ) -> list[ShiftEvent]:
        events: list[ShiftEvent] = []
        topic_markers = [marker.lower() for marker in self.config.topic_markers]
        token_stats = [self._token_stats(seg.text) for seg in segments]

        for idx in range(1, len(segments)):
            prev = segments[idx - 1]
            curr = segments[idx]
            reasons: list[str] = []
            types: list[str] = []
            score = 0.0
            similarity: float | None = None

            if prev.speaker and curr.speaker and prev.speaker != curr.speaker:
                types.append("speaker_change")
                reasons.append(f"speaker change {prev.speaker} -> {curr.speaker}")
                score += 0.4

            curr_lower = curr.text.lower()
            marker_hit = None
            for marker in topic_markers:
                if marker in curr_lower:
                    marker_hit = marker
                    break
            if marker_hit:
                types.append("marker_shift")
                reasons.append(f"topic marker: {marker_hit}")
                score += 0.3

            similarity = self._jaccard_similarity(token_stats[idx - 1], token_stats[idx])
            if similarity is not None and similarity < self.config.similarity_threshold:
                types.append("topic_shift")
                reasons.append(f"low similarity {similarity:.2f}")
                score += 0.3

            if types:
                events.append(
                    ShiftEvent(
                        index=len(events) + 1,
                        shift_types=types,
                        reasons=reasons,
                        from_segment=prev.index,
                        to_segment=curr.index,
                        speaker_from=prev.speaker,
                        speaker_to=curr.speaker,
                        similarity=similarity,
                        shift_score=round(min(score, 1.0), 3),
                    )
                )
        return events[: self.config.max_events]

    def _token_stats(self, text: str) -> _SegmentTokenStats:
        tokens = [token.lower() for token in WORD_RE.findall(text)]
        return _SegmentTokenStats(tokens=tokens, unique=set(tokens))

    def _jaccard_similarity(
        self, left: _SegmentTokenStats, right: _SegmentTokenStats
    ) -> float | None:
        if len(left.tokens) < self.config.min_tokens_for_similarity:
            return None
        if len(right.tokens) < self.config.min_tokens_for_similarity:
            return None
        if not left.unique or not right.unique:
            return None
        union = left.unique | right.unique
        if not union:
            return None
        return len(left.unique & right.unique) / len(union)

    def _token_count(self, text: str) -> int:
        return len(WORD_RE.findall(text))

    def _load_item(self, item: ConversationItem) -> tuple[str, str | None, int | None]:
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
        if not isinstance(event.args, ContextConversationShiftsArgs):
            return ToolCallDisplay(summary="context_conversation_shifts")
        if not event.args.items:
            return ToolCallDisplay(summary="context_conversation_shifts")
        first = event.args.items[0]
        return ToolCallDisplay(
            summary="context_conversation_shifts",
            details={
                "item_count": len(event.args.items),
                "id": first.id,
                "name": first.name,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextConversationShiftsResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=(
                f"Analyzed {event.result.item_count} item(s) with "
                f"{event.result.total_shifts} shift(s)"
            ),
            warnings=event.result.warnings,
            details={
                "item_count": event.result.item_count,
                "total_shifts": event.result.total_shifts,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Detecting conversation shifts"
