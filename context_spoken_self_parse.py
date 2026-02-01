from __future__ import annotations

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


class ContextSpokenSelfParseConfig(BaseToolConfig):
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
    max_word_stats: int = Field(default=200, description="Max word stats to return.")
    max_word_positions: int = Field(default=30, description="Max positions per word.")
    include_tokens: bool = Field(default=True, description="Include token lists per sentence.")


class ContextSpokenSelfParseState(BaseToolState):
    pass


class SpokenSelfParseItem(BaseModel):
    id: str | None = Field(default=None)
    name: str | None = Field(default=None)
    content: str | None = Field(default=None)
    path: str | None = Field(default=None)
    source: str | None = Field(default=None)



class ContextSpokenSelfParseArgs(BaseModel):
    items: list[SpokenSelfParseItem] = Field(description="Items to evaluate.")

class ParsedSentence(BaseModel):
    index: int
    text: str
    start: int
    end: int
    token_count: int
    unique_tokens: int
    tokens: list[str]
    keywords: list[str]


class WordStat(BaseModel):
    word: str
    count: int
    sentence_indices: list[int]
    token_positions: list[int]


class SpokenSelfParseInsight(BaseModel):
    index: int
    id: str | None
    name: str | None
    source_path: str | None
    preview: str
    sentences: list[ParsedSentence]
    word_stats: list[WordStat]
    sentence_count: int
    token_count: int
    unique_token_count: int


class ContextSpokenSelfParseResult(BaseModel):
    items: list[SpokenSelfParseInsight]
    item_count: int
    truncated: bool
    warnings: list[str]
    errors: list[str]


class ContextSpokenSelfParse(
    BaseTool[
        ContextSpokenSelfParseArgs,
        ContextSpokenSelfParseResult,
        ContextSpokenSelfParseConfig,
        ContextSpokenSelfParseState,
    ],
    ToolUIData[ContextSpokenSelfParseArgs, ContextSpokenSelfParseResult],
):
    description: ClassVar[str] = (
        "Parse text and words in spoken output into sentences and word stats."
    )

    async def run(self, args: ContextSpokenSelfParseArgs) -> ContextSpokenSelfParseResult:
        items = list(args.items)
        if not items:
            raise ToolError("items is required.")

        errors: list[str] = []
        warnings: list[str] = []
        insights: list[SpokenSelfParseInsight] = []
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
                word_counts, word_sentences, word_positions, token_total = (
                    self._build_word_stats(sentence_tokens)
                )
                word_stats = self._format_word_stats(
                    word_counts, word_sentences, word_positions
                )

                insights.append(
                    SpokenSelfParseInsight(
                        index=len(insights) + 1,
                        id=item.id,
                        name=item.name,
                        source_path=source_path,
                        preview=self._preview(content),
                        sentences=sentences[: self.config.max_sentences],
                        word_stats=word_stats,
                        sentence_count=len(sentences),
                        token_count=token_total,
                        unique_token_count=len(word_counts),
                    )
                )
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not insights:
            raise ToolError("No insights generated.")

        return ContextSpokenSelfParseResult(
            items=insights,
            item_count=len(insights),
            truncated=truncated,
            warnings=warnings,
            errors=errors,
        )

    def _segment_sentences(
        self, text: str
    ) -> tuple[list[ParsedSentence], list[list[str]]]:
        sentences: list[ParsedSentence] = []
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
                ParsedSentence(
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
        if not self.config.include_tokens:
            return []
        max_tokens = self.config.max_tokens_per_sentence
        if max_tokens <= 0:
            return tokens
        return tokens[:max_tokens]

    def _truncate_sentence(self, text: str) -> str:
        max_chars = self.config.max_sentence_chars
        if max_chars <= 0:
            return text
        return text if len(text) <= max_chars else text[:max_chars]

    def _build_word_stats(
        self, sentence_tokens: list[list[str]]
    ) -> tuple[Counter[str], dict[str, set[int]], dict[str, list[int]], int]:
        word_counts: Counter[str] = Counter()
        word_sentences: dict[str, set[int]] = defaultdict(set)
        word_positions: dict[str, list[int]] = defaultdict(list)
        token_total = 0
        position = 0

        for sentence_index, tokens in enumerate(sentence_tokens, start=1):
            token_total += len(tokens)
            word_counts.update(tokens)
            for token in set(tokens):
                word_sentences[token].add(sentence_index)
            for token in tokens:
                if len(word_positions[token]) < self.config.max_word_positions:
                    word_positions[token].append(position)
                position += 1

        return word_counts, word_sentences, word_positions, token_total

    def _format_word_stats(
        self,
        word_counts: Counter[str],
        word_sentences: dict[str, set[int]],
        word_positions: dict[str, list[int]],
    ) -> list[WordStat]:
        stats: list[WordStat] = []
        for word, count in word_counts.most_common(self.config.max_word_stats):
            stats.append(
                WordStat(
                    word=word,
                    count=count,
                    sentence_indices=sorted(word_sentences.get(word, set())),
                    token_positions=word_positions.get(word, []),
                )
            )
        return stats

    def _load_item(
        self, item: SpokenSelfParseItem
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
        if not isinstance(event.args, ContextSpokenSelfParseArgs):
            return ToolCallDisplay(summary="context_spoken_self_parse")
        if not event.args.items:
            return ToolCallDisplay(summary="context_spoken_self_parse")
        first = event.args.items[0]
        return ToolCallDisplay(
            summary="context_spoken_self_parse",
            details={
                "item_count": len(event.args.items),
                "id": first.id,
                "name": first.name,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpokenSelfParseResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=(
                f"Parsed {event.result.item_count} item(s) with "
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
        return "Parsing spoken text"
