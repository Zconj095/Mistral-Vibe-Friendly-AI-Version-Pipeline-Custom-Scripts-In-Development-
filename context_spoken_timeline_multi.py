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


class ContextSpokenTimelineMultiConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_items: int = Field(default=40, description="Maximum timelines to evaluate.")
    max_source_bytes: int = Field(default=3_000_000, description="Maximum bytes per item.")
    max_total_bytes: int = Field(default=20_000_000, description="Max bytes across items.")
    preview_chars: int = Field(default=400, description="Preview snippet length.")
    max_segments: int = Field(default=200, description="Maximum intervals per timeline.")
    max_total_intervals: int = Field(default=2000, description="Maximum total intervals.")
    min_segment_chars: int = Field(default=20, description="Minimum segment length.")
    min_token_length: int = Field(default=3, description="Minimum token length.")
    min_shared_tokens: int = Field(default=2, description="Minimum shared tokens to link.")
    max_shared_tokens: int = Field(default=8, description="Maximum shared tokens.")
    min_similarity: float = Field(default=0.08, description="Minimum similarity score.")
    max_links_per_interval: int = Field(default=4, description="Maximum links per interval.")
    max_cross_links_total: int = Field(default=1000, description="Maximum cross links.")
    time_tolerance_sec: int = Field(
        default=300, description="Max time delta for cross timeline alignment."
    )
    allow_untimed_match: bool = Field(
        default=True, description="Allow linking when times are missing."
    )


class ContextSpokenTimelineMultiState(BaseToolState):
    pass


class SpokenTimelineMultiItem(BaseModel):
    id: str | None = Field(default=None)
    name: str | None = Field(default=None)
    timeline_id: str | None = Field(default=None)
    content: str | None = Field(default=None)
    path: str | None = Field(default=None)
    source: str | None = Field(default=None)



class ContextSpokenTimelineMultiArgs(BaseModel):
    items: list[SpokenTimelineMultiItem] = Field(description="Items to evaluate.")

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
    from_timeline: str
    to_timeline: str
    from_index: int
    to_index: int
    shared_tokens: list[str]
    similarity: float
    time_delta_sec: int | None


class SpokenTimelineMultiInsight(BaseModel):
    index: int
    id: str | None
    name: str | None
    timeline_id: str
    source_path: str | None
    preview: str
    intervals: list[SpokenInterval]
    interval_count: int


class ContextSpokenTimelineMultiResult(BaseModel):
    items: list[SpokenTimelineMultiInsight]
    cross_links: list[IntervalLink]
    item_count: int
    total_links: int
    truncated: bool
    warnings: list[str]
    errors: list[str]


class ContextSpokenTimelineMulti(
    BaseTool[
        ContextSpokenTimelineMultiArgs,
        ContextSpokenTimelineMultiResult,
        ContextSpokenTimelineMultiConfig,
        ContextSpokenTimelineMultiState,
    ],
    ToolUIData[ContextSpokenTimelineMultiArgs, ContextSpokenTimelineMultiResult],
):
    description: ClassVar[str] = (
        "Process spoken conversations across multiple timelines and timeframes."
    )

    async def run(self, args: ContextSpokenTimelineMultiArgs) -> ContextSpokenTimelineMultiResult:
        items = list(args.items)
        if not items:
            raise ToolError("items is required.")

        errors: list[str] = []
        warnings: list[str] = []
        insights: list[SpokenTimelineMultiInsight] = []
        total_bytes = 0
        truncated = False

        all_intervals: list[tuple[str, int, SpokenInterval, set[str]]] = []
        total_intervals = 0

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
                if total_intervals + len(intervals) > self.config.max_total_intervals:
                    truncated = True
                    warnings.append("Interval budget exceeded; stopping evaluation.")
                    break
                timeline_id = self._timeline_id(item, idx)
                for interval in intervals:
                    all_intervals.append(
                        (
                            timeline_id,
                            interval.index,
                            interval,
                            self._token_set(interval.text),
                        )
                    )
                total_intervals += len(intervals)

                insights.append(
                    SpokenTimelineMultiInsight(
                        index=len(insights) + 1,
                        id=item.id,
                        name=item.name,
                        timeline_id=timeline_id,
                        source_path=source_path,
                        preview=self._preview(content),
                        intervals=intervals[: self.config.max_segments],
                        interval_count=len(intervals),
                    )
                )
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not insights:
            raise ToolError("No insights generated.")

        cross_links = self._link_across_timelines(all_intervals)
        return ContextSpokenTimelineMultiResult(
            items=insights,
            cross_links=cross_links,
            item_count=len(insights),
            total_links=len(cross_links),
            truncated=truncated,
            warnings=warnings,
            errors=errors,
        )

    def _timeline_id(self, item: SpokenTimelineMultiItem, idx: int) -> str:
        if item.timeline_id:
            return item.timeline_id
        if item.id:
            return item.id
        if item.name:
            return item.name
        return f"timeline_{idx}"

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
                    end_time=marker.end_time
                    or (
                        markers[idx + 1].start_time if idx + 1 < len(markers) else None
                    ),
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

    def _time_ok(self, left: SpokenInterval, right: SpokenInterval) -> bool:
        if left.start_time is None or right.start_time is None:
            return self.config.allow_untimed_match
        if self.config.time_tolerance_sec <= 0:
            return True
        return abs(left.start_time - right.start_time) <= self.config.time_tolerance_sec

    def _link_across_timelines(
        self, intervals: list[tuple[str, int, SpokenInterval, set[str]]]
    ) -> list[IntervalLink]:
        links: list[IntervalLink] = []
        link_counts: dict[int, int] = {}
        for idx, (timeline_id, _, interval, tokens) in enumerate(intervals):
            candidates: list[IntervalLink] = []
            for jdx, (other_timeline, _, other_interval, other_tokens) in enumerate(
                intervals
            ):
                if idx == jdx:
                    continue
                if timeline_id == other_timeline:
                    continue
                if not self._time_ok(interval, other_interval):
                    continue
                shared = tokens & other_tokens
                if len(shared) < self.config.min_shared_tokens:
                    continue
                union = tokens | other_tokens
                if not union:
                    continue
                similarity = len(shared) / len(union)
                if similarity < self.config.min_similarity:
                    continue
                delta = None
                if interval.start_time is not None and other_interval.start_time is not None:
                    delta = interval.start_time - other_interval.start_time
                candidates.append(
                    IntervalLink(
                        from_timeline=timeline_id,
                        to_timeline=other_timeline,
                        from_index=interval.index,
                        to_index=other_interval.index,
                        shared_tokens=sorted(shared)[: self.config.max_shared_tokens],
                        similarity=round(similarity, 3),
                        time_delta_sec=delta,
                    )
                )

            candidates.sort(key=lambda link: link.similarity, reverse=True)
            for link in candidates[: self.config.max_links_per_interval]:
                link_counts[idx] = link_counts.get(idx, 0) + 1
                links.append(link)
                if len(links) >= self.config.max_cross_links_total:
                    return links
        return links

    def _load_item(
        self, item: SpokenTimelineMultiItem
    ) -> tuple[str, str | None, int | None]:
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
        if not isinstance(event.args, ContextSpokenTimelineMultiArgs):
            return ToolCallDisplay(summary="context_spoken_timeline_multi")
        if not event.args.items:
            return ToolCallDisplay(summary="context_spoken_timeline_multi")
        first = event.args.items[0]
        return ToolCallDisplay(
            summary="context_spoken_timeline_multi",
            details={
                "item_count": len(event.args.items),
                "id": first.id,
                "name": first.name,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpokenTimelineMultiResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=(
                f"Analyzed {event.result.item_count} timeline(s) with "
                f"{event.result.total_links} cross links"
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
        return "Linking multiple spoken timelines"
