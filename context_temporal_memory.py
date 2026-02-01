from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import TYPE_CHECKING, ClassVar, Iterable

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


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}

TIME_UNIT_NS = {
    "ns": 1,
    "nanosecond": 1,
    "nanoseconds": 1,
    "us": 1_000,
    "microsecond": 1_000,
    "microseconds": 1_000,
    "ms": 1_000_000,
    "millisecond": 1_000_000,
    "milliseconds": 1_000_000,
    "s": 1_000_000_000,
    "sec": 1_000_000_000,
    "secs": 1_000_000_000,
    "second": 1_000_000_000,
    "seconds": 1_000_000_000,
    "m": 60 * 1_000_000_000,
    "min": 60 * 1_000_000_000,
    "mins": 60 * 1_000_000_000,
    "minute": 60 * 1_000_000_000,
    "minutes": 60 * 1_000_000_000,
    "h": 3600 * 1_000_000_000,
    "hr": 3600 * 1_000_000_000,
    "hour": 3600 * 1_000_000_000,
    "hours": 3600 * 1_000_000_000,
    "d": 86400 * 1_000_000_000,
    "day": 86400 * 1_000_000_000,
    "days": 86400 * 1_000_000_000,
    "w": 7 * 86400 * 1_000_000_000,
    "week": 7 * 86400 * 1_000_000_000,
    "weeks": 7 * 86400 * 1_000_000_000,
    "mo": 30 * 86400 * 1_000_000_000,
    "month": 30 * 86400 * 1_000_000_000,
    "months": 30 * 86400 * 1_000_000_000,
    "y": 365 * 86400 * 1_000_000_000,
    "yr": 365 * 86400 * 1_000_000_000,
    "year": 365 * 86400 * 1_000_000_000,
    "years": 365 * 86400 * 1_000_000_000,
    "decade": 10 * 365 * 86400 * 1_000_000_000,
    "decades": 10 * 365 * 86400 * 1_000_000_000,
}


@dataclass
class _TemporalRecord:
    item_index: int
    item_id: str
    timestamp_ns: int
    source: str | None
    tokens: set[str]
    token_counts: dict[str, int]
    preview: str


class ContextTemporalMemoryConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_items: int = Field(default=200, description="Maximum items to process.")
    max_source_bytes: int = Field(
        default=5_000_000, description="Maximum size per item (bytes)."
    )
    max_total_bytes: int = Field(
        default=20_000_000, description="Maximum total bytes across items."
    )
    preview_chars: int = Field(default=240, description="Preview length per item.")
    default_time_unit: str = Field(
        default="s", description="Default time unit for numeric timestamps."
    )
    short_term_window: str = Field(
        default="1h", description="Short term window duration (e.g. 30m, 1h)."
    )
    long_term_window: str = Field(
        default="30d", description="Long term window duration (e.g. 7d, 1y)."
    )
    bucket_size: str = Field(
        default="1d", description="Bucket size for timeline summaries."
    )
    min_word_length: int = Field(default=3, description="Minimum token length.")
    max_keywords: int = Field(default=20, description="Maximum keywords per summary.")
    max_shared_terms: int = Field(default=50, description="Maximum shared terms.")
    max_trend_terms: int = Field(default=50, description="Maximum trend terms.")
    max_links_per_item: int = Field(
        default=5, description="Maximum links per short-term item."
    )
    max_links_total: int = Field(
        default=200, description="Maximum total links."
    )
    min_similarity: float = Field(
        default=0.1, description="Minimum similarity for cross-window links."
    )
    max_items_per_window: int = Field(
        default=50, description="Maximum item summaries per window."
    )


class ContextTemporalMemoryState(BaseToolState):
    pass


class TemporalItem(BaseModel):
    id: str | None = Field(default=None, description="Optional item id.")
    content: str | None = Field(default=None, description="Inline text content.")
    path: str | None = Field(default=None, description="Path to a text file.")
    timestamp: float | int | str = Field(description="Timestamp value or ISO string.")
    time_unit: str | None = Field(default=None, description="Unit for numeric timestamps.")
    source: str | None = Field(default=None, description="Source description.")


class ContextTemporalMemoryArgs(BaseModel):
    items: list[TemporalItem] = Field(description="Temporal text items.")
    short_term_window: str | float | int | None = Field(
        default=None, description="Override short term window duration."
    )
    long_term_window: str | float | int | None = Field(
        default=None, description="Override long term window duration."
    )
    bucket_size: str | float | int | None = Field(
        default=None, description="Override timeline bucket size."
    )
    default_time_unit: str | None = Field(
        default=None, description="Override default time unit."
    )
    max_items: int | None = Field(default=None, description="Override max_items.")
    max_source_bytes: int | None = Field(
        default=None, description="Override max_source_bytes."
    )
    max_total_bytes: int | None = Field(
        default=None, description="Override max_total_bytes."
    )
    preview_chars: int | None = Field(default=None, description="Override preview_chars.")
    min_word_length: int | None = Field(
        default=None, description="Override min_word_length."
    )
    max_keywords: int | None = Field(
        default=None, description="Override max_keywords."
    )
    max_shared_terms: int | None = Field(
        default=None, description="Override max_shared_terms."
    )
    max_trend_terms: int | None = Field(
        default=None, description="Override max_trend_terms."
    )
    max_links_per_item: int | None = Field(
        default=None, description="Override max_links_per_item."
    )
    max_links_total: int | None = Field(
        default=None, description="Override max_links_total."
    )
    min_similarity: float | None = Field(
        default=None, description="Override min_similarity."
    )
    max_items_per_window: int | None = Field(
        default=None, description="Override max_items_per_window."
    )


class TemporalWindowSummary(BaseModel):
    name: str
    start_ns: int
    end_ns: int
    item_count: int
    keywords: list[str]


class TemporalBucket(BaseModel):
    index: int
    start_ns: int
    end_ns: int
    item_count: int
    keywords: list[str]


class TemporalItemSummary(BaseModel):
    item_id: str
    timestamp_ns: int
    source: str | None
    preview: str
    keywords: list[str]


class TemporalLink(BaseModel):
    short_id: str
    short_time_ns: int
    long_id: str
    long_time_ns: int
    similarity: float
    shared_terms: list[str]


class ContextTemporalMemoryResult(BaseModel):
    short_term: TemporalWindowSummary
    long_term: TemporalWindowSummary
    short_term_items: list[TemporalItemSummary]
    long_term_items: list[TemporalItemSummary]
    bridge_terms: list[str]
    new_terms: list[str]
    fading_terms: list[str]
    buckets: list[TemporalBucket]
    links: list[TemporalLink]
    item_count: int
    bucket_count: int
    link_count: int
    truncated: bool
    errors: list[str]


class ContextTemporalMemory(
    BaseTool[
        ContextTemporalMemoryArgs,
        ContextTemporalMemoryResult,
        ContextTemporalMemoryConfig,
        ContextTemporalMemoryState,
    ],
    ToolUIData[ContextTemporalMemoryArgs, ContextTemporalMemoryResult],
):
    description: ClassVar[str] = (
        "Reason across short-term and long-term temporal text memory."
    )

    async def run(
        self, args: ContextTemporalMemoryArgs
    ) -> ContextTemporalMemoryResult:
        if not args.items:
            raise ToolError("items is required.")

        max_items = args.max_items if args.max_items is not None else self.config.max_items
        if max_items <= 0:
            raise ToolError("max_items must be a positive integer.")
        if len(args.items) > max_items:
            raise ToolError(f"items exceeds max_items ({len(args.items)} > {max_items}).")

        max_source_bytes = (
            args.max_source_bytes
            if args.max_source_bytes is not None
            else self.config.max_source_bytes
        )
        max_total_bytes = (
            args.max_total_bytes
            if args.max_total_bytes is not None
            else self.config.max_total_bytes
        )
        preview_chars = (
            args.preview_chars if args.preview_chars is not None else self.config.preview_chars
        )
        min_word_length = (
            args.min_word_length
            if args.min_word_length is not None
            else self.config.min_word_length
        )
        max_keywords = (
            args.max_keywords if args.max_keywords is not None else self.config.max_keywords
        )
        max_shared_terms = (
            args.max_shared_terms
            if args.max_shared_terms is not None
            else self.config.max_shared_terms
        )
        max_trend_terms = (
            args.max_trend_terms
            if args.max_trend_terms is not None
            else self.config.max_trend_terms
        )
        max_links_per_item = (
            args.max_links_per_item
            if args.max_links_per_item is not None
            else self.config.max_links_per_item
        )
        max_links_total = (
            args.max_links_total
            if args.max_links_total is not None
            else self.config.max_links_total
        )
        min_similarity = (
            args.min_similarity
            if args.min_similarity is not None
            else self.config.min_similarity
        )
        max_items_per_window = (
            args.max_items_per_window
            if args.max_items_per_window is not None
            else self.config.max_items_per_window
        )
        default_time_unit = (
            args.default_time_unit
            if args.default_time_unit is not None
            else self.config.default_time_unit
        )

        if max_source_bytes <= 0:
            raise ToolError("max_source_bytes must be a positive integer.")
        if max_total_bytes <= 0:
            raise ToolError("max_total_bytes must be a positive integer.")
        if preview_chars < 0:
            raise ToolError("preview_chars must be >= 0.")
        if min_word_length <= 0:
            raise ToolError("min_word_length must be a positive integer.")
        if max_keywords < 0:
            raise ToolError("max_keywords must be >= 0.")
        if max_shared_terms < 0:
            raise ToolError("max_shared_terms must be >= 0.")
        if max_trend_terms < 0:
            raise ToolError("max_trend_terms must be >= 0.")
        if max_links_per_item < 0:
            raise ToolError("max_links_per_item must be >= 0.")
        if max_links_total < 0:
            raise ToolError("max_links_total must be >= 0.")
        if min_similarity < 0:
            raise ToolError("min_similarity must be >= 0.")
        if max_items_per_window < 0:
            raise ToolError("max_items_per_window must be >= 0.")

        short_window_ns = self._parse_duration(
            args.short_term_window or self.config.short_term_window, default_time_unit
        )
        long_window_ns = self._parse_duration(
            args.long_term_window or self.config.long_term_window, default_time_unit
        )
        bucket_size_ns = self._parse_duration(
            args.bucket_size or self.config.bucket_size, default_time_unit
        )

        if short_window_ns <= 0:
            raise ToolError("short_term_window must be a positive duration.")
        if long_window_ns <= 0:
            raise ToolError("long_term_window must be a positive duration.")
        if bucket_size_ns <= 0:
            raise ToolError("bucket_size must be a positive duration.")

        records: list[_TemporalRecord] = []
        errors: list[str] = []
        truncated = False
        total_bytes = 0

        for idx, item in enumerate(args.items, start=1):
            try:
                content, size_bytes = self._load_item_content(item, max_source_bytes)
                if content is None:
                    raise ToolError("Item has no content to analyze.")
                if size_bytes is not None:
                    if total_bytes + size_bytes > max_total_bytes:
                        truncated = True
                        break
                    total_bytes += size_bytes

                timestamp_ns = self._parse_timestamp(
                    item.timestamp, item.time_unit or default_time_unit
                )
                tokens, token_counts = self._extract_tokens(
                    content, min_word_length
                )
                item_id = item.id or f"item{idx}"
                preview = self._preview_text(content, preview_chars)
                records.append(
                    _TemporalRecord(
                        item_index=idx,
                        item_id=item_id,
                        timestamp_ns=timestamp_ns,
                        source=item.source,
                        tokens=tokens,
                        token_counts=token_counts,
                        preview=preview,
                    )
                )
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not records:
            raise ToolError("No valid items were processed.")

        records.sort(key=lambda item: item.timestamp_ns)
        min_time = records[0].timestamp_ns
        max_time = records[-1].timestamp_ns
        short_start = max_time - short_window_ns
        long_start = max_time - long_window_ns

        short_items = [item for item in records if item.timestamp_ns >= short_start]
        long_items = [item for item in records if item.timestamp_ns >= long_start]
        long_history_items = [item for item in records if long_start <= item.timestamp_ns < short_start]

        short_counts = self._merge_token_counts(short_items)
        long_counts = self._merge_token_counts(long_items)

        short_keywords = self._select_keywords(short_counts, max_keywords)
        long_keywords = self._select_keywords(long_counts, max_keywords)

        short_terms = set(short_counts.keys())
        long_terms = set(long_counts.keys())
        bridge_terms = sorted(short_terms & long_terms)
        if max_shared_terms > 0 and len(bridge_terms) > max_shared_terms:
            bridge_terms = bridge_terms[:max_shared_terms]

        new_terms = sorted(short_terms - long_terms)
        fading_terms = sorted(long_terms - short_terms)
        if max_trend_terms > 0 and len(new_terms) > max_trend_terms:
            new_terms = new_terms[:max_trend_terms]
        if max_trend_terms > 0 and len(fading_terms) > max_trend_terms:
            fading_terms = fading_terms[:max_trend_terms]

        buckets = self._build_buckets(
            records, min_time, max_time, bucket_size_ns, max_keywords
        )

        short_term_summary = TemporalWindowSummary(
            name="short_term",
            start_ns=short_start,
            end_ns=max_time,
            item_count=len(short_items),
            keywords=short_keywords,
        )
        long_term_summary = TemporalWindowSummary(
            name="long_term",
            start_ns=long_start,
            end_ns=max_time,
            item_count=len(long_items),
            keywords=long_keywords,
        )

        short_term_items = self._summarize_items(
            short_items, max_items_per_window, max_keywords
        )
        long_term_items = self._summarize_items(
            long_items, max_items_per_window, max_keywords
        )

        links = self._build_links(
            short_items,
            long_history_items,
            max_links_per_item,
            max_links_total,
            min_similarity,
            max_shared_terms,
        )
        if len(links) >= max_links_total:
            truncated = True

        return ContextTemporalMemoryResult(
            short_term=short_term_summary,
            long_term=long_term_summary,
            short_term_items=short_term_items,
            long_term_items=long_term_items,
            bridge_terms=bridge_terms,
            new_terms=new_terms,
            fading_terms=fading_terms,
            buckets=buckets,
            links=links,
            item_count=len(records),
            bucket_count=len(buckets),
            link_count=len(links),
            truncated=truncated,
            errors=errors,
        )

    def _load_item_content(
        self, item: TemporalItem, max_source_bytes: int
    ) -> tuple[str | None, int | None]:
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
                raise ToolError(f"Path is a directory, not a file: {path}")
            size = path.stat().st_size
            if size > max_source_bytes:
                raise ToolError(
                    f"{path} exceeds max_source_bytes ({size} > {max_source_bytes})."
                )
            return path.read_text("utf-8", errors="ignore"), size
        if item.content is not None:
            size = len(item.content.encode("utf-8"))
            if size > max_source_bytes:
                raise ToolError(
                    f"content exceeds max_source_bytes ({size} > {max_source_bytes})."
                )
            return item.content, size
        return None, None

    def _preview_text(self, text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def _extract_tokens(
        self, text: str, min_word_length: int
    ) -> tuple[set[str], dict[str, int]]:
        token_counts: dict[str, int] = {}
        for token in TOKEN_RE.findall(text):
            lowered = token.lower()
            if len(lowered) < min_word_length:
                continue
            if lowered in STOPWORDS:
                continue
            token_counts[lowered] = token_counts.get(lowered, 0) + 1
        return set(token_counts.keys()), token_counts

    def _merge_token_counts(
        self, items: Iterable[_TemporalRecord]
    ) -> dict[str, int]:
        merged: dict[str, int] = {}
        for item in items:
            for token, count in item.token_counts.items():
                merged[token] = merged.get(token, 0) + count
        return merged

    def _select_keywords(self, token_counts: dict[str, int], limit: int) -> list[str]:
        if limit <= 0:
            return []
        ordered = sorted(token_counts.items(), key=lambda item: (-item[1], item[0]))
        return [token for token, _ in ordered[:limit]]

    def _summarize_items(
        self, items: list[_TemporalRecord], limit: int, max_keywords: int
    ) -> list[TemporalItemSummary]:
        summaries: list[TemporalItemSummary] = []
        for item in items[:limit] if limit > 0 else []:
            keywords = self._select_keywords(item.token_counts, max_keywords)
            summaries.append(
                TemporalItemSummary(
                    item_id=item.item_id,
                    timestamp_ns=item.timestamp_ns,
                    source=item.source,
                    preview=item.preview,
                    keywords=keywords,
                )
            )
        return summaries

    def _build_buckets(
        self,
        items: list[_TemporalRecord],
        min_time: int,
        max_time: int,
        bucket_size_ns: int,
        max_keywords: int,
    ) -> list[TemporalBucket]:
        if bucket_size_ns <= 0:
            return []
        bucket_count = ((max_time - min_time) // bucket_size_ns) + 1
        buckets: list[TemporalBucket] = []
        bucket_tokens: list[dict[str, int]] = [
            {} for _ in range(int(bucket_count))
        ]
        bucket_sizes = [0 for _ in range(int(bucket_count))]

        for item in items:
            index = (item.timestamp_ns - min_time) // bucket_size_ns
            idx = int(max(index, 0))
            if idx >= len(bucket_tokens):
                continue
            bucket_sizes[idx] += 1
            for token, count in item.token_counts.items():
                bucket_tokens[idx][token] = bucket_tokens[idx].get(token, 0) + count

        for idx in range(int(bucket_count)):
            start_ns = min_time + (idx * bucket_size_ns)
            end_ns = start_ns + bucket_size_ns
            keywords = self._select_keywords(bucket_tokens[idx], max_keywords)
            buckets.append(
                TemporalBucket(
                    index=idx,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    item_count=bucket_sizes[idx],
                    keywords=keywords,
                )
            )

        return buckets

    def _build_links(
        self,
        short_items: list[_TemporalRecord],
        long_items: list[_TemporalRecord],
        max_links_per_item: int,
        max_links_total: int,
        min_similarity: float,
        max_shared_terms: int,
    ) -> list[TemporalLink]:
        links: list[TemporalLink] = []
        if not short_items or not long_items:
            return links

        for short_item in short_items:
            if max_links_total > 0 and len(links) >= max_links_total:
                break
            scored: list[tuple[float, _TemporalRecord, list[str]]] = []
            for long_item in long_items:
                shared = short_item.tokens & long_item.tokens
                if not shared:
                    continue
                union = short_item.tokens | long_item.tokens
                similarity = len(shared) / len(union) if union else 0.0
                if similarity < min_similarity:
                    continue
                shared_terms = sorted(shared)
                if max_shared_terms > 0 and len(shared_terms) > max_shared_terms:
                    shared_terms = shared_terms[:max_shared_terms]
                scored.append((similarity, long_item, shared_terms))

            scored.sort(key=lambda item: (-item[0], item[1].timestamp_ns))
            for similarity, long_item, shared_terms in scored[:max_links_per_item]:
                if max_links_total > 0 and len(links) >= max_links_total:
                    break
                links.append(
                    TemporalLink(
                        short_id=short_item.item_id,
                        short_time_ns=short_item.timestamp_ns,
                        long_id=long_item.item_id,
                        long_time_ns=long_item.timestamp_ns,
                        similarity=round(similarity, 6),
                        shared_terms=shared_terms,
                    )
                )

        return links

    def _parse_duration(self, value: str | float | int, default_unit: str) -> int:
        if isinstance(value, (int, float)):
            unit_ns = self._resolve_time_unit(default_unit)
            return int(round(float(value) * unit_ns))
        if not isinstance(value, str):
            raise ToolError("Duration value must be a string or number.")
        text = value.strip()
        if not text:
            raise ToolError("Duration value cannot be empty.")
        if self._is_numeric(text):
            unit_ns = self._resolve_time_unit(default_unit)
            return int(round(float(text) * unit_ns))
        match = re.match(r"^\s*([-+]?[0-9]*\.?[0-9]+)\s*([A-Za-z]+)\s*$", text)
        if not match:
            raise ToolError(f"Invalid duration format: {value}")
        amount = float(match.group(1))
        unit = match.group(2).lower()
        unit_ns = self._resolve_time_unit(unit)
        return int(round(amount * unit_ns))

    def _parse_timestamp(self, value: float | int | str, time_unit: str) -> int:
        if isinstance(value, (int, float)):
            unit_ns = self._resolve_time_unit(time_unit)
            return int(round(float(value) * unit_ns))
        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise ToolError("timestamp cannot be empty.")
            if self._is_numeric(text):
                unit_ns = self._resolve_time_unit(time_unit)
                return int(round(float(text) * unit_ns))
            try:
                if text.endswith("Z"):
                    text = text[:-1] + "+00:00"
                dt = datetime.fromisoformat(text)
            except ValueError as exc:
                raise ToolError(f"Invalid timestamp string: {value}") from exc
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1_000_000_000)
        raise ToolError("timestamp must be a number or string.")

    def _resolve_time_unit(self, unit: str | None) -> int:
        key = (unit or "").strip().lower()
        if not key:
            raise ToolError("time_unit cannot be empty.")
        if key not in TIME_UNIT_NS:
            raise ToolError(f"Unsupported time unit: {unit}")
        return TIME_UNIT_NS[key]

    def _is_numeric(self, value: str) -> bool:
        return bool(re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value))

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextTemporalMemoryArgs):
            return ToolCallDisplay(summary="context_temporal_memory")

        summary = f"context_temporal_memory: {len(event.args.items)} item(s)"
        return ToolCallDisplay(
            summary=summary,
            details={
                "item_count": len(event.args.items),
                "short_term_window": event.args.short_term_window,
                "long_term_window": event.args.long_term_window,
                "bucket_size": event.args.bucket_size,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextTemporalMemoryResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = (
            f"Processed {event.result.item_count} item(s) with "
            f"{event.result.link_count} link(s)"
        )
        warnings = event.result.errors[:]
        if event.result.truncated:
            warnings.append("Output truncated by size or limits")

        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=warnings,
            details={
                "item_count": event.result.item_count,
                "bucket_count": event.result.bucket_count,
                "link_count": event.result.link_count,
                "truncated": event.result.truncated,
                "errors": event.result.errors,
                "short_term": event.result.short_term,
                "long_term": event.result.long_term,
                "bridge_terms": event.result.bridge_terms,
                "new_terms": event.result.new_terms,
                "fading_terms": event.result.fading_terms,
                "buckets": event.result.buckets,
                "links": event.result.links,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Reasoning across temporal memory"
