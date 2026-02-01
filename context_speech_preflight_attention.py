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


def _tokenize(text: str, min_len: int) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text) if len(token) >= min_len]


class ContextSpeechPreflightAttentionConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_items: int = Field(default=40, description="Maximum items to evaluate.")
    max_source_bytes: int = Field(default=3_000_000, description="Maximum bytes per item.")
    max_total_bytes: int = Field(default=20_000_000, description="Max bytes across items.")
    preview_chars: int = Field(default=400, description="Preview snippet length.")
    max_sentences: int = Field(default=400, description="Maximum sentences per item.")
    min_sentence_chars: int = Field(default=20, description="Minimum sentence length.")
    max_sentence_chars: int = Field(default=400, description="Maximum chars per sentence.")
    min_token_length: int = Field(default=3, description="Minimum token length.")
    max_tokens_per_sentence: int = Field(default=80, description="Max tokens per sentence.")
    max_keywords_per_sentence: int = Field(default=8, description="Max keywords per sentence.")
    window_size: int = Field(default=4, description="Token window for correlations.")
    max_tokens: int = Field(default=2000, description="Maximum unique tokens to track.")
    max_pairs: int = Field(default=500, description="Maximum token pairs to return.")
    max_links_per_token: int = Field(default=6, description="Top links per token.")
    max_positions_per_token: int = Field(default=30, description="Max positions per token.")
    min_pair_score: float = Field(default=0.05, description="Minimum pair score.")
    include_bigrams: bool = Field(default=True, description="Include bigram stats.")


class ContextSpeechPreflightAttentionState(BaseToolState):
    pass


class SpeechPreflightItem(BaseModel):
    id: str | None = Field(default=None)
    name: str | None = Field(default=None)
    content: str | None = Field(default=None)
    path: str | None = Field(default=None)
    source: str | None = Field(default=None)



class ContextSpeechPreflightAttentionArgs(BaseModel):
    items: list[SpeechPreflightItem] = Field(description="Items to evaluate.")

class PlannedSentence(BaseModel):
    index: int
    text: str
    start: int
    end: int
    token_count: int
    unique_tokens: int
    tokens: list[str]
    keywords: list[str]


class TokenLink(BaseModel):
    token: str
    cooccurrence: int
    score: float


class TokenAttention(BaseModel):
    token: str
    count: int
    positions: list[int]
    top_links: list[TokenLink]


class TokenPair(BaseModel):
    token_a: str
    token_b: str
    cooccurrence: int
    avg_distance: float
    score: float


class BigramStat(BaseModel):
    bigram: str
    count: int


class SpeechPreflightInsight(BaseModel):
    index: int
    id: str | None
    name: str | None
    source_path: str | None
    preview: str
    sentences: list[PlannedSentence]
    token_attention: list[TokenAttention]
    token_pairs: list[TokenPair]
    bigrams: list[BigramStat]
    sentence_count: int
    token_count: int
    unique_token_count: int


class ContextSpeechPreflightAttentionResult(BaseModel):
    items: list[SpeechPreflightInsight]
    item_count: int
    truncated: bool
    warnings: list[str]
    errors: list[str]


class ContextSpeechPreflightAttention(
    BaseTool[
        ContextSpeechPreflightAttentionArgs,
        ContextSpeechPreflightAttentionResult,
        ContextSpeechPreflightAttentionConfig,
        ContextSpeechPreflightAttentionState,
    ],
    ToolUIData[ContextSpeechPreflightAttentionArgs, ContextSpeechPreflightAttentionResult],
):
    description: ClassVar[str] = (
        "Preflight attention on planned speech to correlate words before speaking."
    )

    async def run(self, args: ContextSpeechPreflightAttentionArgs) -> ContextSpeechPreflightAttentionResult:
        items = list(args.items)
        if not items:
            raise ToolError("items is required.")

        errors: list[str] = []
        warnings: list[str] = []
        insights: list[SpeechPreflightInsight] = []
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

                sentences, sentence_tokens = self._segment_sentences(content)
                (
                    token_counts,
                    token_positions,
                    token_order,
                    token_total,
                ) = self._tokenize_all(sentence_tokens)
                top_tokens = self._select_top_tokens(token_counts)
                pair_counts, pair_distances = self._count_pairs(
                    token_order, top_tokens
                )
                token_pairs = self._build_pairs(
                    pair_counts, pair_distances, token_counts
                )
                token_attention = self._build_attention(
                    token_counts, token_positions, token_pairs
                )
                bigrams = self._build_bigrams(token_order)

                insights.append(
                    SpeechPreflightInsight(
                        index=len(insights) + 1,
                        id=item.id,
                        name=item.name,
                        source_path=source_path,
                        preview=self._preview(content),
                        sentences=sentences[: self.config.max_sentences],
                        token_attention=token_attention,
                        token_pairs=token_pairs,
                        bigrams=bigrams,
                        sentence_count=len(sentences),
                        token_count=token_total,
                        unique_token_count=len(token_counts),
                    )
                )
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not insights:
            raise ToolError("No insights generated.")

        return ContextSpeechPreflightAttentionResult(
            items=insights,
            item_count=len(insights),
            truncated=truncated,
            warnings=warnings,
            errors=errors,
        )

    def _segment_sentences(
        self, text: str
    ) -> tuple[list[PlannedSentence], list[list[str]]]:
        sentences: list[PlannedSentence] = []
        sentence_tokens: list[list[str]] = []
        boundaries = {".", "!", "?", "\n"}
        start = 0

        def add_sentence(raw_start: int, raw_end: int) -> None:
            slice_text = text[raw_start:raw_end]
            if not slice_text:
                return
            seg_text = slice_text.strip()
            if not seg_text:
                return
            if len(seg_text) < self.config.min_sentence_chars and sentences:
                previous = sentences[-1]
                previous.text = f"{previous.text} {seg_text}".strip()
                previous.end = raw_end
                tokens = _tokenize(previous.text, self.config.min_token_length)
                previous.token_count = len(tokens)
                previous.unique_tokens = len(set(tokens))
                previous.tokens = self._limit_tokens(tokens)
                previous.keywords = self._sentence_keywords(tokens)
                sentence_tokens[-1] = tokens
                return
            tokens = _tokenize(seg_text, self.config.min_token_length)
            sentences.append(
                PlannedSentence(
                    index=len(sentences) + 1,
                    text=self._truncate_sentence(seg_text),
                    start=raw_start,
                    end=raw_end,
                    token_count=len(tokens),
                    unique_tokens=len(set(tokens)),
                    tokens=self._limit_tokens(tokens),
                    keywords=self._sentence_keywords(tokens),
                )
            )
            sentence_tokens.append(tokens)

        for idx, char in enumerate(text):
            if char in boundaries:
                end = idx + 1
                add_sentence(start, end)
                start = end

        if start < len(text):
            add_sentence(start, len(text))

        return sentences[: self.config.max_sentences], sentence_tokens[: self.config.max_sentences]

    def _sentence_keywords(self, tokens: list[str]) -> list[str]:
        if self.config.max_keywords_per_sentence <= 0:
            return []
        counts = Counter(tokens)
        return [token for token, _ in counts.most_common(self.config.max_keywords_per_sentence)]

    def _limit_tokens(self, tokens: list[str]) -> list[str]:
        max_tokens = self.config.max_tokens_per_sentence
        if max_tokens <= 0:
            return tokens
        return tokens[:max_tokens]

    def _truncate_sentence(self, text: str) -> str:
        max_chars = self.config.max_sentence_chars
        if max_chars <= 0:
            return text
        return text if len(text) <= max_chars else text[:max_chars]

    def _tokenize_all(
        self, sentence_tokens: list[list[str]]
    ) -> tuple[Counter[str], dict[str, list[int]], list[str], int]:
        token_counts: Counter[str] = Counter()
        token_positions: dict[str, list[int]] = defaultdict(list)
        token_order: list[str] = []
        position = 0

        for tokens in sentence_tokens:
            token_counts.update(tokens)
            token_order.extend(tokens)
            for token in tokens:
                if len(token_positions[token]) < self.config.max_positions_per_token:
                    token_positions[token].append(position)
                position += 1

        return token_counts, token_positions, token_order, len(token_order)

    def _select_top_tokens(self, token_counts: Counter[str]) -> set[str]:
        if self.config.max_tokens <= 0 or len(token_counts) <= self.config.max_tokens:
            return set(token_counts.keys())
        return set(token for token, _ in token_counts.most_common(self.config.max_tokens))

    def _count_pairs(
        self, token_order: list[str], top_tokens: set[str]
    ) -> tuple[Counter[tuple[str, str]], Counter[tuple[str, str]]]:
        pair_counts: Counter[tuple[str, str]] = Counter()
        pair_distances: Counter[tuple[str, str]] = Counter()
        window = max(self.config.window_size, 1)
        total_tokens = len(token_order)
        for idx, token in enumerate(token_order):
            if token not in top_tokens:
                continue
            end = min(total_tokens, idx + window + 1)
            for next_idx in range(idx + 1, end):
                other = token_order[next_idx]
                if other not in top_tokens:
                    continue
                left, right = sorted((token, other))
                pair_counts[(left, right)] += 1
                pair_distances[(left, right)] += next_idx - idx
        return pair_counts, pair_distances

    def _build_pairs(
        self,
        pair_counts: Counter[tuple[str, str]],
        pair_distances: Counter[tuple[str, str]],
        token_counts: Counter[str],
    ) -> list[TokenPair]:
        pairs: list[TokenPair] = []
        for (left, right), count in pair_counts.items():
            if count <= 0:
                continue
            denom = max(1, min(token_counts.get(left, 0), token_counts.get(right, 0)))
            score = count / denom
            if score < self.config.min_pair_score:
                continue
            avg_distance = pair_distances[(left, right)] / count
            pairs.append(
                TokenPair(
                    token_a=left,
                    token_b=right,
                    cooccurrence=count,
                    avg_distance=round(avg_distance, 2),
                    score=round(score, 3),
                )
            )
        pairs.sort(key=lambda item: (item.score, item.cooccurrence), reverse=True)
        return pairs[: self.config.max_pairs]

    def _build_attention(
        self,
        token_counts: Counter[str],
        token_positions: dict[str, list[int]],
        token_pairs: list[TokenPair],
    ) -> list[TokenAttention]:
        link_map: dict[str, list[TokenLink]] = defaultdict(list)
        for pair in token_pairs:
            link_map[pair.token_a].append(
                TokenLink(
                    token=pair.token_b,
                    cooccurrence=pair.cooccurrence,
                    score=pair.score,
                )
            )
            link_map[pair.token_b].append(
                TokenLink(
                    token=pair.token_a,
                    cooccurrence=pair.cooccurrence,
                    score=pair.score,
                )
            )

        attention: list[TokenAttention] = []
        for token, count in token_counts.items():
            positions = token_positions.get(token, [])
            links = sorted(
                link_map.get(token, []), key=lambda item: item.score, reverse=True
            )[: self.config.max_links_per_token]
            attention.append(
                TokenAttention(
                    token=token,
                    count=count,
                    positions=positions,
                    top_links=links,
                )
            )
        attention.sort(key=lambda item: item.count, reverse=True)
        return attention[: self.config.max_tokens]

    def _build_bigrams(self, token_order: list[str]) -> list[BigramStat]:
        if not self.config.include_bigrams:
            return []
        counts: Counter[str] = Counter()
        for idx in range(len(token_order) - 1):
            counts[f"{token_order[idx]} {token_order[idx + 1]}"] += 1
        return [
            BigramStat(bigram=bigram, count=count)
            for bigram, count in counts.most_common(self.config.max_pairs)
        ]

    def _load_item(
        self, item: SpeechPreflightItem
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
        if not isinstance(event.args, ContextSpeechPreflightAttentionArgs):
            return ToolCallDisplay(summary="context_speech_preflight_attention")
        if not event.args.items:
            return ToolCallDisplay(summary="context_speech_preflight_attention")
        first = event.args.items[0]
        return ToolCallDisplay(
            summary="context_speech_preflight_attention",
            details={
                "item_count": len(event.args.items),
                "id": first.id,
                "name": first.name,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpeechPreflightAttentionResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=(
                f"Analyzed {event.result.item_count} item(s) with "
                f"{len(event.result.items[0].sentences) if event.result.items else 0} sentences"
            ),
            warnings=event.result.warnings,
            details={
                "item_count": event.result.item_count,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Preflighting speech attention"
