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


class ContextSpokenWordContextMeaningConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_items: int = Field(default=40, description="Maximum items to evaluate.")
    max_source_bytes: int = Field(default=3_000_000, description="Maximum bytes per item.")
    max_total_bytes: int = Field(default=20_000_000, description="Max bytes across items.")
    preview_chars: int = Field(default=400, description="Preview snippet length.")
    max_segments: int = Field(default=200, description="Maximum segments per item.")
    min_segment_chars: int = Field(default=20, description="Minimum segment length.")
    min_token_length: int = Field(default=3, description="Minimum token length.")
    max_keywords_per_segment: int = Field(default=12, description="Max keywords per segment.")
    max_words: int = Field(default=2000, description="Maximum unique tokens to track.")
    max_word_stats: int = Field(default=200, description="Maximum word stats to return.")
    max_segments_per_word: int = Field(default=20, description="Max segments per word.")
    max_context_terms: int = Field(default=12, description="Max context terms per word.")
    max_related_words: int = Field(default=8, description="Max related words per word.")
    min_pair_count: int = Field(default=1, description="Minimum pair count.")
    min_pair_score: float = Field(default=0.05, description="Minimum pair score.")
    max_pairs: int = Field(default=500, description="Maximum word pairs to return.")


class ContextSpokenWordContextMeaningState(BaseToolState):
    pass


class SpokenWordContextMeaningItem(BaseModel):
    id: str | None = Field(default=None)
    name: str | None = Field(default=None)
    content: str | None = Field(default=None)
    path: str | None = Field(default=None)
    source: str | None = Field(default=None)



class ContextSpokenWordContextMeaningArgs(BaseModel):
    items: list[SpokenWordContextMeaningItem] = Field(description="Items to evaluate.")

class SpokenSegmentSummary(BaseModel):
    index: int
    speaker: str | None
    text: str
    start: int
    end: int
    token_count: int
    keywords: list[str]


class RelatedWord(BaseModel):
    word: str
    cooccurrence: int
    score: float


class WordContextStat(BaseModel):
    word: str
    total_count: int
    segment_count: int
    segment_indices: list[int]
    context_terms: list[str]
    related_words: list[RelatedWord]


class WordPair(BaseModel):
    word_a: str
    word_b: str
    cooccurrence: int
    jaccard: float
    overlap: float
    score: float


class SpokenWordContextMeaningInsight(BaseModel):
    index: int
    id: str | None
    name: str | None
    source_path: str | None
    preview: str
    segments: list[SpokenSegmentSummary]
    word_stats: list[WordContextStat]
    word_pairs: list[WordPair]
    segment_count: int
    word_count: int
    pair_count: int


class ContextSpokenWordContextMeaningResult(BaseModel):
    items: list[SpokenWordContextMeaningInsight]
    item_count: int
    total_pairs: int
    truncated: bool
    warnings: list[str]
    errors: list[str]


class ContextSpokenWordContextMeaning(
    BaseTool[
        ContextSpokenWordContextMeaningArgs,
        ContextSpokenWordContextMeaningResult,
        ContextSpokenWordContextMeaningConfig,
        ContextSpokenWordContextMeaningState,
    ],
    ToolUIData[ContextSpokenWordContextMeaningArgs, ContextSpokenWordContextMeaningResult],
):
    description: ClassVar[str] = (
        "Map spoken words to contextual meaning across dialogue segments."
    )

    async def run(self, args: ContextSpokenWordContextMeaningArgs) -> ContextSpokenWordContextMeaningResult:
        items = list(args.items)
        if not items:
            raise ToolError("items is required.")

        errors: list[str] = []
        warnings: list[str] = []
        insights: list[SpokenWordContextMeaningInsight] = []
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

                segments, segment_tokens, segment_keywords = self._segment_conversation(
                    content
                )
                word_counts, word_segments = self._accumulate_word_stats(segment_tokens)
                top_words = self._select_top_words(word_counts)
                pair_counts = self._count_pairs(segment_tokens, top_words)
                word_pairs = self._build_pairs(pair_counts, word_segments)
                word_stats = self._build_word_stats(
                    word_counts, word_segments, segment_keywords, word_pairs
                )

                total_pairs += len(word_pairs)

                insights.append(
                    SpokenWordContextMeaningInsight(
                        index=len(insights) + 1,
                        id=item.id,
                        name=item.name,
                        source_path=source_path,
                        preview=self._preview(content),
                        segments=segments[: self.config.max_segments],
                        word_stats=word_stats,
                        word_pairs=word_pairs,
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

        return ContextSpokenWordContextMeaningResult(
            items=insights,
            item_count=len(insights),
            total_pairs=total_pairs,
            truncated=truncated,
            warnings=warnings,
            errors=errors,
        )

    def _segment_conversation(
        self, text: str
    ) -> tuple[list[SpokenSegmentSummary], list[list[str]], list[list[tuple[str, float]]]]:
        segments: list[SpokenSegmentSummary] = []
        segment_tokens: list[list[str]] = []
        segment_keywords: list[list[tuple[str, float]]] = []

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
                tokens = _tokenize(segments[-1].text, self.config.min_token_length)
                segment_tokens[-1] = tokens
            else:
                tokens = _tokenize(seg_text, self.config.min_token_length)
                segments.append(
                    SpokenSegmentSummary(
                        index=len(segments) + 1,
                        speaker=current_speaker,
                        text=seg_text,
                        start=seg_start,
                        end=end_offset,
                        token_count=len(tokens),
                        keywords=[],
                    )
                )
                segment_tokens.append(tokens)
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

        df_counts: dict[str, int] = {}
        for tokens in segment_tokens:
            for token in set(tokens):
                df_counts[token] = df_counts.get(token, 0) + 1

        total_segments = max(len(segment_tokens), 1)
        for idx, tokens in enumerate(segment_tokens):
            tf_counts: dict[str, int] = {}
            for token in tokens:
                tf_counts[token] = tf_counts.get(token, 0) + 1
            scored = []
            for token, tf in tf_counts.items():
                df = df_counts.get(token, 1)
                idf = math.log((total_segments + 1) / df) + 1.0
                scored.append((token, tf * idf))
            scored.sort(key=lambda item: item[1], reverse=True)
            top_scored = scored[: self.config.max_keywords_per_segment]
            segment_keywords.append(top_scored)
            if idx < len(segments):
                segments[idx].keywords = [token for token, _ in top_scored]

        return segments[: self.config.max_segments], segment_tokens[: self.config.max_segments], segment_keywords[: self.config.max_segments]

    def _accumulate_word_stats(
        self, segment_tokens: list[list[str]]
    ) -> tuple[Counter[str], dict[str, set[int]]]:
        word_counts: Counter[str] = Counter()
        word_segments: dict[str, set[int]] = defaultdict(set)
        for idx, tokens in enumerate(segment_tokens, start=1):
            word_counts.update(tokens)
            for token in set(tokens):
                word_segments[token].add(idx)
        return word_counts, word_segments

    def _select_top_words(self, word_counts: Counter[str]) -> set[str]:
        if self.config.max_words <= 0 or len(word_counts) <= self.config.max_words:
            return set(word_counts.keys())
        return set(token for token, _ in word_counts.most_common(self.config.max_words))

    def _count_pairs(
        self, segment_tokens: list[list[str]], top_words: set[str]
    ) -> Counter[tuple[str, str]]:
        pair_counts: Counter[tuple[str, str]] = Counter()
        for tokens in segment_tokens:
            unique_tokens = sorted({token for token in tokens if token in top_words})
            for i, left in enumerate(unique_tokens):
                for right in unique_tokens[i + 1 :]:
                    pair_counts[(left, right)] += 1
        return pair_counts

    def _build_pairs(
        self,
        pair_counts: Counter[tuple[str, str]],
        word_segments: dict[str, set[int]],
    ) -> list[WordPair]:
        pairs: list[WordPair] = []
        for (left, right), count in pair_counts.items():
            if count < self.config.min_pair_count:
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
        pairs.sort(key=lambda item: (item.score, item.cooccurrence), reverse=True)
        return pairs[: self.config.max_pairs]

    def _build_word_stats(
        self,
        word_counts: Counter[str],
        word_segments: dict[str, set[int]],
        segment_keywords: list[list[tuple[str, float]]],
        word_pairs: list[WordPair],
    ) -> list[WordContextStat]:
        related_map: dict[str, list[RelatedWord]] = defaultdict(list)
        for pair in word_pairs:
            related_map[pair.word_a].append(
                RelatedWord(
                    word=pair.word_b,
                    cooccurrence=pair.cooccurrence,
                    score=pair.score,
                )
            )
            related_map[pair.word_b].append(
                RelatedWord(
                    word=pair.word_a,
                    cooccurrence=pair.cooccurrence,
                    score=pair.score,
                )
            )

        context_map: dict[str, Counter[str]] = defaultdict(Counter)
        for idx, keywords in enumerate(segment_keywords, start=1):
            for token in word_segments:
                if idx not in word_segments[token]:
                    continue
                for kw, score in keywords:
                    if kw == token:
                        continue
                    context_map[token][kw] += score

        stats: list[WordContextStat] = []
        for word, total_count in word_counts.most_common(self.config.max_word_stats):
            segments = sorted(word_segments.get(word, set()))
            if self.config.max_segments_per_word > 0:
                segments = segments[: self.config.max_segments_per_word]
            context_terms = [
                term
                for term, _ in context_map[word].most_common(self.config.max_context_terms)
            ]
            related_words = sorted(
                related_map.get(word, []), key=lambda item: item.score, reverse=True
            )[: self.config.max_related_words]
            stats.append(
                WordContextStat(
                    word=word,
                    total_count=total_count,
                    segment_count=len(word_segments.get(word, set())),
                    segment_indices=segments,
                    context_terms=context_terms,
                    related_words=related_words,
                )
            )
        return stats

    def _load_item(
        self, item: SpokenWordContextMeaningItem
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
        if not isinstance(event.args, ContextSpokenWordContextMeaningArgs):
            return ToolCallDisplay(summary="context_spoken_word_context_meaning")
        if not event.args.items:
            return ToolCallDisplay(summary="context_spoken_word_context_meaning")
        first = event.args.items[0]
        return ToolCallDisplay(
            summary="context_spoken_word_context_meaning",
            details={
                "item_count": len(event.args.items),
                "id": first.id,
                "name": first.name,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpokenWordContextMeaningResult):
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
        return "Mapping spoken word context"
