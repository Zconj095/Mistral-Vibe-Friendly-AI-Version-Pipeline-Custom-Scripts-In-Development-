from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
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


WORD_RE = re.compile(r"[A-Za-z0-9_']+")
SPEAKER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _.'-]{0,40}):\s*(.*)$")


def _tokenize(text: str, min_len: int) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text) if len(token) >= min_len]


class ContextSpokenWordCorrelationConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_items: int = Field(default=40, description="Maximum items to evaluate.")
    max_source_bytes: int = Field(default=3_000_000, description="Maximum bytes per item.")
    max_total_bytes: int = Field(default=20_000_000, description="Max bytes across items.")
    preview_chars: int = Field(default=400, description="Preview snippet length.")
    max_segments: int = Field(default=200, description="Maximum segments per item.")
    min_segment_chars: int = Field(default=20, description="Minimum segment length.")
    max_segment_chars: int = Field(default=400, description="Max characters per segment.")
    min_token_length: int = Field(default=3, description="Minimum token length.")
    max_tokens_per_segment: int = Field(
        default=80, description="Maximum unique tokens per segment."
    )
    max_words: int = Field(default=2000, description="Maximum words to track.")
    max_word_stats: int = Field(default=200, description="Maximum word stats to return.")
    max_pairs: int = Field(default=500, description="Maximum word pairs to return.")
    min_cooccurrence: int = Field(default=1, description="Minimum pair co-occurrence.")
    min_pair_score: float = Field(default=0.05, description="Minimum pair score.")
    max_links_per_word: int = Field(
        default=6, description="Maximum correlated words per word."
    )
    max_segments_per_word: int = Field(
        default=20, description="Maximum segment indices per word."
    )
    max_segment_keywords: int = Field(
        default=8, description="Keywords to display per segment."
    )
    max_bridge_words: int = Field(
        default=20, description="Maximum bridge words to return."
    )


class ContextSpokenWordCorrelationState(BaseToolState):
    pass


class SpokenWordCorrelationItem(BaseModel):
    id: str | None = Field(default=None)
    name: str | None = Field(default=None)
    content: str | None = Field(default=None)
    path: str | None = Field(default=None)
    source: str | None = Field(default=None)



class ContextSpokenWordCorrelationArgs(BaseModel):
    items: list[SpokenWordCorrelationItem] = Field(description="Items to evaluate.")

class SpokenSegmentSummary(BaseModel):
    index: int
    speaker: str | None
    text: str
    start: int
    end: int
    token_count: int
    unique_tokens: int
    keywords: list[str]


class WordLink(BaseModel):
    word: str
    cooccurrence: int
    score: float


class WordStat(BaseModel):
    word: str
    total_count: int
    segment_count: int
    segment_indices: list[int]
    top_links: list[WordLink]


class WordPair(BaseModel):
    word_a: str
    word_b: str
    cooccurrence: int
    jaccard: float
    overlap: float
    score: float


class WordBridge(BaseModel):
    word: str
    segment_count: int
    total_count: int


class SpokenWordCorrelationInsight(BaseModel):
    index: int
    id: str | None
    name: str | None
    source_path: str | None
    preview: str
    segments: list[SpokenSegmentSummary]
    word_stats: list[WordStat]
    word_pairs: list[WordPair]
    bridge_words: list[WordBridge]
    segment_count: int
    word_count: int
    pair_count: int


class ContextSpokenWordCorrelationResult(BaseModel):
    items: list[SpokenWordCorrelationInsight]
    item_count: int
    total_pairs: int
    truncated: bool
    warnings: list[str]
    errors: list[str]


class ContextSpokenWordCorrelation(
    BaseTool[
        ContextSpokenWordCorrelationArgs,
        ContextSpokenWordCorrelationResult,
        ContextSpokenWordCorrelationConfig,
        ContextSpokenWordCorrelationState,
    ],
    ToolUIData[ContextSpokenWordCorrelationArgs, ContextSpokenWordCorrelationResult],
):
    description: ClassVar[str] = (
        "Correlate words across spoken dialogue and the words it speaks."
    )

    async def run(self, args: ContextSpokenWordCorrelationArgs) -> ContextSpokenWordCorrelationResult:
        items = list(args.items)
        if not items:
            raise ToolError("items is required.")

        errors: list[str] = []
        warnings: list[str] = []
        insights: list[SpokenWordCorrelationInsight] = []
        total_bytes = 0
        truncated = False
        total_pairs = 0

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

                segments, segment_tokens = self._segment_conversation(content)
                word_counts, word_segments = self._accumulate_word_stats(
                    segment_tokens
                )
                top_words = self._select_top_words(word_counts)
                pair_counts = self._count_pairs(segment_tokens, top_words)
                word_pairs = self._build_word_pairs(
                    pair_counts, word_segments
                )
                word_stats = self._build_word_stats(
                    word_counts, word_segments, word_pairs
                )
                bridge_words = self._build_bridge_words(word_counts, word_segments)

                total_pairs += len(word_pairs)

                insights.append(
                    SpokenWordCorrelationInsight(
                        index=len(insights) + 1,
                        id=item.id,
                        name=item.name,
                        source_path=source_path,
                        preview=self._preview(content),
                        segments=segments[: self.config.max_segments],
                        word_stats=word_stats,
                        word_pairs=word_pairs,
                        bridge_words=bridge_words,
                        segment_count=len(segments),
                        word_count=len(word_counts),
                        pair_count=len(word_pairs),
                    )
                )
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not insights:
            raise ToolError("No insights generated.")

        return ContextSpokenWordCorrelationResult(
            items=insights,
            item_count=len(insights),
            total_pairs=total_pairs,
            truncated=truncated,
            warnings=warnings,
            errors=errors,
        )

    def _segment_conversation(
        self, text: str
    ) -> tuple[list[SpokenSegmentSummary], list[Counter[str]]]:
        segments: list[SpokenSegmentSummary] = []
        segment_tokens: list[Counter[str]] = []
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
                previous = segments[-1]
                previous.text = f"{previous.text}\n{seg_text}".strip()
                previous.end = end_offset
                tokens = _tokenize(previous.text, self.config.min_token_length)
                token_counts = Counter(tokens)
                previous.token_count = len(tokens)
                previous.unique_tokens = len(token_counts)
                previous.keywords = self._segment_keywords(token_counts)
                segment_tokens[-1] = token_counts
            else:
                tokens = _tokenize(seg_text, self.config.min_token_length)
                token_counts = Counter(tokens)
                segments.append(
                    SpokenSegmentSummary(
                        index=len(segments) + 1,
                        speaker=current_speaker,
                        text=self._truncate_segment_text(seg_text),
                        start=seg_start,
                        end=end_offset,
                        token_count=len(tokens),
                        unique_tokens=len(token_counts),
                        keywords=self._segment_keywords(token_counts),
                    )
                )
                segment_tokens.append(token_counts)
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
        return segments[: self.config.max_segments], segment_tokens[: self.config.max_segments]

    def _segment_keywords(self, token_counts: Counter[str]) -> list[str]:
        if self.config.max_segment_keywords <= 0:
            return []
        return [token for token, _ in token_counts.most_common(self.config.max_segment_keywords)]

    def _truncate_segment_text(self, text: str) -> str:
        max_chars = self.config.max_segment_chars
        if max_chars <= 0:
            return text
        return text if len(text) <= max_chars else text[:max_chars]

    def _accumulate_word_stats(
        self, segment_tokens: list[Counter[str]]
    ) -> tuple[Counter[str], dict[str, set[int]]]:
        word_counts: Counter[str] = Counter()
        word_segments: dict[str, set[int]] = defaultdict(set)
        for idx, token_counts in enumerate(segment_tokens, start=1):
            word_counts.update(token_counts)
            for token in token_counts:
                word_segments[token].add(idx)
        return word_counts, word_segments

    def _select_top_words(self, word_counts: Counter[str]) -> set[str]:
        if self.config.max_words <= 0 or len(word_counts) <= self.config.max_words:
            return set(word_counts.keys())
        return set(token for token, _ in word_counts.most_common(self.config.max_words))

    def _count_pairs(
        self, segment_tokens: list[Counter[str]], top_words: set[str]
    ) -> Counter[tuple[str, str]]:
        pair_counts: Counter[tuple[str, str]] = Counter()
        max_tokens = self.config.max_tokens_per_segment
        for token_counts in segment_tokens:
            tokens = [token for token in token_counts if token in top_words]
            if max_tokens > 0 and len(tokens) > max_tokens:
                tokens = [token for token, _ in token_counts.most_common(max_tokens) if token in top_words]
            tokens = sorted(set(tokens))
            for i, left in enumerate(tokens):
                for right in tokens[i + 1 :]:
                    pair_counts[(left, right)] += 1
        return pair_counts

    def _build_word_pairs(
        self,
        pair_counts: Counter[tuple[str, str]],
        word_segments: dict[str, set[int]],
    ) -> list[WordPair]:
        pairs: list[WordPair] = []
        for (left, right), count in pair_counts.items():
            if count < self.config.min_cooccurrence:
                continue
            left_segments = word_segments.get(left, set())
            right_segments = word_segments.get(right, set())
            if not left_segments or not right_segments:
                continue
            union = left_segments | right_segments
            if not union:
                continue
            jaccard = count / len(union)
            overlap = count / max(1, min(len(left_segments), len(right_segments)))
            score = (jaccard + overlap) / 2.0
            if score < self.config.min_pair_score:
                continue
            pairs.append(
                WordPair(
                    word_a=left,
                    word_b=right,
                    cooccurrence=count,
                    jaccard=round(jaccard, 3),
                    overlap=round(overlap, 3),
                    score=round(score, 3),
                )
            )
        pairs.sort(key=lambda pair: (pair.score, pair.cooccurrence), reverse=True)
        return pairs[: self.config.max_pairs]

    def _build_word_stats(
        self,
        word_counts: Counter[str],
        word_segments: dict[str, set[int]],
        word_pairs: list[WordPair],
    ) -> list[WordStat]:
        link_map: dict[str, list[WordLink]] = defaultdict(list)
        for pair in word_pairs:
            link = WordLink(
                word=pair.word_b, cooccurrence=pair.cooccurrence, score=pair.score
            )
            link_map[pair.word_a].append(link)
            link = WordLink(
                word=pair.word_a, cooccurrence=pair.cooccurrence, score=pair.score
            )
            link_map[pair.word_b].append(link)

        stats: list[WordStat] = []
        for word, total_count in word_counts.items():
            segments = sorted(word_segments.get(word, set()))
            segment_count = len(segments)
            top_links = sorted(
                link_map.get(word, []), key=lambda item: item.score, reverse=True
            )[: self.config.max_links_per_word]
            if self.config.max_segments_per_word > 0:
                segments = segments[: self.config.max_segments_per_word]
            stats.append(
                WordStat(
                    word=word,
                    total_count=total_count,
                    segment_count=segment_count,
                    segment_indices=segments,
                    top_links=top_links,
                )
            )
        stats.sort(
            key=lambda entry: (entry.segment_count, entry.total_count), reverse=True
        )
        return stats[: self.config.max_word_stats]

    def _build_bridge_words(
        self,
        word_counts: Counter[str],
        word_segments: dict[str, set[int]],
    ) -> list[WordBridge]:
        bridges: list[WordBridge] = []
        for word, segments in word_segments.items():
            if len(segments) <= 1:
                continue
            bridges.append(
                WordBridge(
                    word=word,
                    segment_count=len(segments),
                    total_count=word_counts.get(word, 0),
                )
            )
        bridges.sort(
            key=lambda entry: (entry.segment_count, entry.total_count), reverse=True
        )
        return bridges[: self.config.max_bridge_words]

    def _load_item(
        self, item: SpokenWordCorrelationItem
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
        if not isinstance(event.args, ContextSpokenWordCorrelationArgs):
            return ToolCallDisplay(summary="context_spoken_word_correlation")
        if not event.args.items:
            return ToolCallDisplay(summary="context_spoken_word_correlation")
        first = event.args.items[0]
        return ToolCallDisplay(
            summary="context_spoken_word_correlation",
            details={
                "item_count": len(event.args.items),
                "id": first.id,
                "name": first.name,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpokenWordCorrelationResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=(
                f"Analyzed {event.result.item_count} item(s) with "
                f"{event.result.total_pairs} word pair(s)"
            ),
            warnings=event.result.warnings,
            details={
                "item_count": event.result.item_count,
                "total_pairs": event.result.total_pairs,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Correlating spoken words"
