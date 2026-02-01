from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
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


TIME_RANGE_RE = re.compile(
    r"(?P<start>(?:\d{1,2}:)?\d{1,2}:\d{2})\s*[-–]\s*(?P<end>(?:\d{1,2}:)?\d{1,2}:\d{2})"
)
TIME_SINGLE_RE = re.compile(r"\b(?P<time>(?:\d{1,2}:)?\d{1,2}:\d{2})\b")
WORD_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class _Marker:
    start: int
    end: int
    start_time: int | None
    end_time: int | None


class ContextSpokenTimelineConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_items: int = Field(default=40, description="Maximum items to evaluate.")
    max_source_bytes: int = Field(default=3_000_000, description="Maximum bytes per item.")
    max_total_bytes: int = Field(default=20_000_000, description="Max bytes across items.")
    preview_chars: int = Field(default=400, description="Preview snippet length.")
    max_segments: int = Field(default=200, description="Maximum segments per item.")
    min_segment_chars: int = Field(default=20, description="Minimum segment length.")
    link_window_segments: int = Field(default=5, description="Segments to look back for links.")
    min_token_length: int = Field(default=3, description="Minimum token length.")
    min_shared_tokens: int = Field(default=2, description="Minimum shared tokens to link.")
    max_shared_tokens: int = Field(default=8, description="Maximum shared tokens to include.")
    min_similarity: float = Field(default=0.08, description="Minimum similarity score.")
    max_links_per_segment: int = Field(default=3, description="Maximum links per segment.")


class ContextSpokenTimelineState(BaseToolState):
    pass


class SpokenTimelineItem(BaseModel):
    id: str | None = Field(default=None)
    name: str | None = Field(default=None)
    content: str | None = Field(default=None)
    path: str | None = Field(default=None)
    source: str | None = Field(default=None)



class ContextSpokenTimelineArgs(BaseModel):
    items: list[SpokenTimelineItem] = Field(description="Items to evaluate.")

class SpokenInterval(BaseModel):
    index: int
    start_time: int | None
    end_time: int | None
    text: str
    token_count: int
    word_count: int
    start: int
    end: int


class IntervalLink(BaseModel):
    from_index: int
    to_index: int
    shared_tokens: list[str]
    similarity: float
    time_delta_sec: int | None


class SpokenTimelineInsight(BaseModel):
    index: int
    id: str | None
    name: str | None
    source_path: str | None
    preview: str
    intervals: list[SpokenInterval]
    links: list[IntervalLink]
    interval_count: int
    link_count: int


class ContextSpokenTimelineResult(BaseModel):
    items: list[SpokenTimelineInsight]
    item_count: int
    total_links: int
    truncated: bool
    warnings: list[str]
    errors: list[str]


class ContextSpokenTimeline(
    BaseTool[
        ContextSpokenTimelineArgs,
        ContextSpokenTimelineResult,
        ContextSpokenTimelineConfig,
        ContextSpokenTimelineState,
    ],
    ToolUIData[ContextSpokenTimelineArgs, ContextSpokenTimelineResult],
):
    description: ClassVar[str] = (
        "Reference information across spoken intervals and timeframes."
    )

    async def run(self, args: ContextSpokenTimelineArgs) -> ContextSpokenTimelineResult:
        items = list(args.items)
        if not items:
            raise ToolError("items is required.")

        errors: list[str] = []
        warnings: list[str] = []
        insights: list[SpokenTimelineInsight] = []
        total_bytes = 0
        truncated = False
        total_links = 0

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

                intervals = self._build_intervals(content)
                links = self._link_intervals(intervals)
                total_links += len(links)

                insights.append(
                    SpokenTimelineInsight(
                        index=len(insights) + 1,
                        id=item.id,
                        name=item.name,
                        source_path=source_path,
                        preview=self._preview(content),
                        intervals=intervals[: self.config.max_segments],
                        links=links,
                        interval_count=len(intervals),
                        link_count=len(links),
                    )
                )
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not insights:
            raise ToolError("No insights generated.")

        return ContextSpokenTimelineResult(
            items=insights,
            item_count=len(insights),
            total_links=total_links,
            truncated=truncated,
            warnings=warnings,
            errors=errors,
        )

    def _build_intervals(self, text: str) -> list[SpokenInterval]:
        markers = self._extract_markers(text)
        if not markers:
            return self._split_paragraphs(text)

        intervals: list[SpokenInterval] = []
        for idx, marker in enumerate(markers):
            seg_start = marker.end
            seg_end = markers[idx + 1].start if idx + 1 < len(markers) else len(text)
            seg_text = text[seg_start:seg_end].strip()
            if len(seg_text) < self.config.min_segment_chars:
                continue
            token_count = self._token_count(seg_text)
            intervals.append(
                SpokenInterval(
                    index=len(intervals) + 1,
                    start_time=marker.start_time,
                    end_time=marker.end_time or (markers[idx + 1].start_time if idx + 1 < len(markers) else None),
                    text=seg_text,
                    token_count=token_count,
                    word_count=token_count,
                    start=seg_start,
                    end=seg_end,
                )
            )

        return intervals[: self.config.max_segments]

    def _split_paragraphs(self, text: str) -> list[SpokenInterval]:
        chunks = [chunk for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
        intervals: list[SpokenInterval] = []
        cursor = 0
        for chunk in chunks:
            start = text.find(chunk, cursor)
            end = start + len(chunk)
            cursor = end
            seg_text = chunk.strip()
            if len(seg_text) < self.config.min_segment_chars:
                continue
            token_count = self._token_count(seg_text)
            intervals.append(
                SpokenInterval(
                    index=len(intervals) + 1,
                    start_time=None,
                    end_time=None,
                    text=seg_text,
                    token_count=token_count,
                    word_count=token_count,
                    start=start,
                    end=end,
                )
            )
        return intervals[: self.config.max_segments]

    def _extract_markers(self, text: str) -> list[_Marker]:
        markers: list[_Marker] = []
        for match in TIME_RANGE_RE.finditer(text):
            start_time = self._parse_time(match.group("start"))
            end_time = self._parse_time(match.group("end"))
            markers.append(
                _Marker(
                    start=match.start(),
                    end=match.end(),
                    start_time=start_time,
                    end_time=end_time,
                )
            )

        for match in TIME_SINGLE_RE.finditer(text):
            if any(m.start <= match.start() < m.end for m in markers):
                continue
            time_value = self._parse_time(match.group("time"))
            markers.append(
                _Marker(
                    start=match.start(),
                    end=match.end(),
                    start_time=time_value,
                    end_time=None,
                )
            )

        return sorted(markers, key=lambda marker: marker.start)

    def _parse_time(self, text: str) -> int | None:
        parts = [int(part) for part in text.split(":") if part.isdigit()]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        return None

    def _token_count(self, text: str) -> int:
        return len(WORD_RE.findall(text))

    def _token_set(self, text: str) -> set[str]:
        return {
            token.lower()
            for token in WORD_RE.findall(text)
            if len(token) >= self.config.min_token_length
        }

    def _link_intervals(self, intervals: list[SpokenInterval]) -> list[IntervalLink]:
        links: list[IntervalLink] = []
        token_sets = [self._token_set(interval.text) for interval in intervals]
        for idx, interval in enumerate(intervals):
            candidates: list[IntervalLink] = []
            start = max(0, idx - self.config.link_window_segments)
            for prev_idx in range(start, idx):
                shared = token_sets[idx] & token_sets[prev_idx]
                if len(shared) < self.config.min_shared_tokens:
                    continue
                union = token_sets[idx] | token_sets[prev_idx]
                if not union:
                    continue
                similarity = len(shared) / len(union)
                if similarity < self.config.min_similarity:
                    continue
                delta = None
                if interval.start_time is not None and intervals[prev_idx].start_time is not None:
                    delta = interval.start_time - intervals[prev_idx].start_time
                candidates.append(
                    IntervalLink(
                        from_index=interval.index,
                        to_index=intervals[prev_idx].index,
                        shared_tokens=sorted(shared)[: self.config.max_shared_tokens],
                        similarity=round(similarity, 3),
                        time_delta_sec=delta,
                    )
                )
            candidates.sort(key=lambda link: link.similarity, reverse=True)
            links.extend(candidates[: self.config.max_links_per_segment])
        return links

    def _load_item(self, item: SpokenTimelineItem) -> tuple[str, str | None, int | None]:
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
        if not isinstance(event.args, ContextSpokenTimelineArgs):
            return ToolCallDisplay(summary="context_spoken_timeline")
        if not event.args.items:
            return ToolCallDisplay(summary="context_spoken_timeline")
        first = event.args.items[0]
        return ToolCallDisplay(
            summary="context_spoken_timeline",
            details={
                "item_count": len(event.args.items),
                "id": first.id,
                "name": first.name,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpokenTimelineResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=(
                f"Analyzed {event.result.item_count} item(s) with "
                f"{event.result.total_links} timeline link(s)"
            ),
            warnings=event.result.warnings,
            details={
                "item_count": event.result.item_count,
                "total_links": event.result.total_links,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Linking spoken timelines"
