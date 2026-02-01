from __future__ import annotations

import re
from collections import Counter
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
SPEAKER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _.'-]{0,40})\s*:\s*(.*)$")
SENTENCE_RE = re.compile(r"[.!?]+")
PUNCT_KEYS = (".", ",", "?", "!", ":", ";", "-")


class ContextSelfSpeechPatternsConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_items: int = Field(default=40, description="Maximum dialogues to evaluate.")
    max_source_bytes: int = Field(default=3_000_000, description="Maximum bytes per item.")
    max_total_bytes: int = Field(default=20_000_000, description="Max bytes across items.")
    preview_chars: int = Field(default=400, description="Preview snippet length.")
    max_segments: int = Field(default=200, description="Maximum segments per dialogue.")
    min_segment_chars: int = Field(default=20, description="Minimum segment length.")
    min_token_length: int = Field(default=3, description="Minimum token length.")
    max_keywords_per_segment: int = Field(default=10, description="Max keywords per segment.")
    max_word_stats: int = Field(default=200, description="Max word stats returned.")
    max_segment_indices_per_word: int = Field(
        default=30, description="Max segment indices per word."
    )
    min_shared_tokens: int = Field(default=2, description="Minimum shared tokens.")
    min_similarity: float = Field(default=0.1, description="Minimum link similarity.")
    max_links_per_segment: int = Field(default=4, description="Max links per segment.")
    max_links_total: int = Field(default=500, description="Max links total.")
    token_weight: float = Field(default=0.7, description="Token similarity weight.")
    pattern_weight: float = Field(default=0.3, description="Pattern similarity weight.")
    include_unlabeled_as_self: bool = Field(
        default=True, description="Treat unlabeled segments as self."
    )
    self_speaker_labels: list[str] = Field(
        default_factory=lambda: [
            "assistant",
            "ai",
            "model",
            "bot",
            "mistral",
            "vibe",
            "system",
        ],
        description="Speaker labels treated as self.",
    )


class ContextSelfSpeechPatternsState(BaseToolState):
    pass


class SelfSpeechItem(BaseModel):
    id: str | None = Field(default=None)
    name: str | None = Field(default=None)
    dialogue_id: str | None = Field(default=None)
    content: str | None = Field(default=None)
    path: str | None = Field(default=None)
    source: str | None = Field(default=None)


class ContextSelfSpeechPatternsArgs(BaseModel):
    items: list[SelfSpeechItem] = Field(description="Dialogues to evaluate.")


class SelfSpeechSegment(BaseModel):
    index: int
    speaker: str | None
    text: str
    start: int
    end: int
    token_count: int
    unique_tokens: int
    avg_token_length: float
    avg_sentence_length: float
    uppercase_ratio: float
    punctuation_per_100: dict[str, float]
    keywords: list[str]


class SelfSpeechLink(BaseModel):
    from_index: int
    to_index: int
    shared_tokens: list[str]
    token_similarity: float
    pattern_similarity: float
    combined_score: float


class SelfWordStat(BaseModel):
    word: str
    total_count: int
    segment_count: int
    segment_indices: list[int]


class SelfSpeechProfile(BaseModel):
    segment_count: int
    total_tokens: int
    unique_tokens: int
    vocab_ratio: float
    avg_token_length: float
    avg_sentence_length: float
    uppercase_ratio: float
    punctuation_per_100: dict[str, float]
    top_keywords: list[str]


class SelfSpeechInsight(BaseModel):
    index: int
    id: str | None
    name: str | None
    dialogue_id: str
    source_path: str | None
    preview: str
    segments: list[SelfSpeechSegment]
    profile: SelfSpeechProfile
    word_stats: list[SelfWordStat]
    links: list[SelfSpeechLink]
    segment_count: int
    link_count: int


class ContextSelfSpeechPatternsResult(BaseModel):
    items: list[SelfSpeechInsight]
    item_count: int
    total_links: int
    truncated: bool
    warnings: list[str]
    errors: list[str]


class ContextSelfSpeechPatterns(
    BaseTool[
        ContextSelfSpeechPatternsArgs,
        ContextSelfSpeechPatternsResult,
        ContextSelfSpeechPatternsConfig,
        ContextSelfSpeechPatternsState,
    ],
    ToolUIData[ContextSelfSpeechPatternsArgs, ContextSelfSpeechPatternsResult],
):
    description: ClassVar[str] = (
        "Recognize self speech patterns and correlate words across spoken intervals."
    )

    async def run(
        self, args: ContextSelfSpeechPatternsArgs
    ) -> ContextSelfSpeechPatternsResult:
        items = list(args.items)
        if not items:
            raise ToolError("items is required.")

        errors: list[str] = []
        warnings: list[str] = []
        total_bytes = 0
        truncated = False
        insights: list[SelfSpeechInsight] = []
        total_links = 0

        if len(items) > self.config.max_items:
            warnings.append("Item limit reached; truncating input list.")
            items = items[: self.config.max_items]

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

                segments = self._segment_dialogue(content)
                if not segments:
                    raise ToolError("No segments found.")
                if len(segments) > self.config.max_segments:
                    segments = segments[: self.config.max_segments]
                    warnings.append("Segment limit reached; truncating dialogue.")

                segments = self._select_self_segments(segments)
                if not segments:
                    raise ToolError("No self segments matched.")

                profile = self._build_profile(segments)
                word_stats = self._build_word_stats(segments)
                links = self._link_segments(segments)
                total_links += len(links)

                insights.append(
                    SelfSpeechInsight(
                        index=len(insights) + 1,
                        id=item.id,
                        name=item.name,
                        dialogue_id=self._dialogue_id(item, idx),
                        source_path=source_path,
                        preview=self._preview(content),
                        segments=segments,
                        profile=profile,
                        word_stats=word_stats,
                        links=links,
                        segment_count=len(segments),
                        link_count=len(links),
                    )
                )
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not insights:
            raise ToolError("No insights generated.")

        return ContextSelfSpeechPatternsResult(
            items=insights,
            item_count=len(insights),
            total_links=total_links,
            truncated=truncated,
            warnings=warnings,
            errors=errors,
        )

    def _dialogue_id(self, item: SelfSpeechItem, idx: int) -> str:
        if item.dialogue_id:
            return item.dialogue_id
        if item.id:
            return item.id
        if item.name:
            return item.name
        return f"dialogue_{idx}"

    def _segment_dialogue(self, text: str) -> list[SelfSpeechSegment]:
        segments: list[SelfSpeechSegment] = []
        buffer: list[str] = []
        current_speaker: str | None = None
        segment_start: int | None = None
        segment_end: int | None = None
        pos = 0

        def flush() -> None:
            nonlocal buffer, segment_start, segment_end, current_speaker
            if not buffer:
                return
            combined = " ".join(buffer).strip()
            if len(combined) >= self.config.min_segment_chars:
                segments.append(
                    self._build_segment(
                        combined,
                        current_speaker,
                        segment_start or 0,
                        segment_end or 0,
                    )
                )
            buffer = []
            segment_start = None
            segment_end = None

        for raw_line in text.splitlines(True):
            line = raw_line.rstrip("\r\n")
            line_start = pos
            pos += len(raw_line)
            segment_end = pos
            match = SPEAKER_RE.match(line)
            if match:
                flush()
                current_speaker = match.group(1).strip()
                line_text = match.group(2).strip()
                segment_start = line_start
                if line_text:
                    buffer.append(line_text)
                continue
            if not line.strip():
                flush()
                continue
            if segment_start is None:
                segment_start = line_start
            buffer.append(line.strip())

        flush()
        if segments:
            return segments

        chunks = [chunk for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
        for chunk in chunks:
            seg_text = chunk.strip()
            if len(seg_text) < self.config.min_segment_chars:
                continue
            segments.append(self._build_segment(seg_text, None, 0, 0))

        return segments[: self.config.max_segments]

    def _build_segment(
        self, text: str, speaker: str | None, start: int, end: int
    ) -> SelfSpeechSegment:
        tokens = self._tokenize(text)
        token_count = len(tokens)
        unique_tokens = len(set(tokens))
        avg_token_length = self._avg_token_length(tokens)
        avg_sentence_length = self._avg_sentence_length(text)
        uppercase_ratio = self._uppercase_ratio(tokens)
        punctuation = self._punctuation_rates(text, token_count)
        keywords = self._top_keywords(tokens, self.config.max_keywords_per_segment)

        return SelfSpeechSegment(
            index=0,
            speaker=speaker,
            text=text,
            start=start,
            end=end,
            token_count=token_count,
            unique_tokens=unique_tokens,
            avg_token_length=round(avg_token_length, 3),
            avg_sentence_length=round(avg_sentence_length, 3),
            uppercase_ratio=round(uppercase_ratio, 4),
            punctuation_per_100=punctuation,
            keywords=keywords,
        )

    def _select_self_segments(
        self, segments: list[SelfSpeechSegment]
    ) -> list[SelfSpeechSegment]:
        label_set = {label.lower() for label in self.config.self_speaker_labels}
        has_speakers = any(segment.speaker for segment in segments)
        matched = [
            segment
            for segment in segments
            if segment.speaker and segment.speaker.strip().lower() in label_set
        ]
        if matched:
            for idx, segment in enumerate(matched, start=1):
                segment.index = idx
            return matched
        if not has_speakers and self.config.include_unlabeled_as_self:
            for idx, segment in enumerate(segments, start=1):
                segment.index = idx
            return segments
        unlabeled = [
            segment for segment in segments if segment.speaker is None
        ]
        if unlabeled and self.config.include_unlabeled_as_self:
            for idx, segment in enumerate(unlabeled, start=1):
                segment.index = idx
            return unlabeled
        return []

    def _build_profile(self, segments: list[SelfSpeechSegment]) -> SelfSpeechProfile:
        token_counts = Counter()
        sentence_lengths: list[int] = []
        token_lengths: list[int] = []
        uppercase_tokens = 0
        punct_counts = Counter({key: 0 for key in PUNCT_KEYS})
        total_tokens = 0

        for segment in segments:
            tokens = self._tokenize(segment.text)
            token_counts.update(tokens)
            total_tokens += len(tokens)
            token_lengths.extend(len(token) for token in tokens)
            uppercase_tokens += sum(
                1 for token in tokens if token.isupper() and len(token) > 1
            )
            sentence_lengths.extend(self._sentence_lengths(segment.text))
            for key in PUNCT_KEYS:
                punct_counts[key] += segment.text.count(key)

        unique_tokens = len(token_counts)
        vocab_ratio = unique_tokens / total_tokens if total_tokens else 0.0
        avg_token_length = (
            sum(token_lengths) / len(token_lengths) if token_lengths else 0.0
        )
        avg_sentence_length = (
            sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0.0
        )
        uppercase_ratio = uppercase_tokens / total_tokens if total_tokens else 0.0
        punctuation = self._punctuation_rates_by_counts(punct_counts, total_tokens)
        top_keywords = [
            word for word, _ in token_counts.most_common(self.config.max_keywords_per_segment)
        ]

        return SelfSpeechProfile(
            segment_count=len(segments),
            total_tokens=total_tokens,
            unique_tokens=unique_tokens,
            vocab_ratio=round(vocab_ratio, 4),
            avg_token_length=round(avg_token_length, 3),
            avg_sentence_length=round(avg_sentence_length, 3),
            uppercase_ratio=round(uppercase_ratio, 4),
            punctuation_per_100=punctuation,
            top_keywords=top_keywords,
        )

    def _build_word_stats(self, segments: list[SelfSpeechSegment]) -> list[SelfWordStat]:
        word_counts = Counter()
        word_segments: dict[str, set[int]] = {}
        for segment in segments:
            tokens = self._tokenize(segment.text)
            word_counts.update(tokens)
            for token in set(tokens):
                word_segments.setdefault(token, set()).add(segment.index)

        stats: list[SelfWordStat] = []
        for word, count in word_counts.most_common(self.config.max_word_stats):
            indices = sorted(word_segments.get(word, set()))
            stats.append(
                SelfWordStat(
                    word=word,
                    total_count=count,
                    segment_count=len(indices),
                    segment_indices=indices[: self.config.max_segment_indices_per_word],
                )
            )
        return stats

    def _link_segments(self, segments: list[SelfSpeechSegment]) -> list[SelfSpeechLink]:
        token_sets = [set(self._tokenize(segment.text)) for segment in segments]
        links: list[SelfSpeechLink] = []
        for idx, segment in enumerate(segments):
            candidates: list[SelfSpeechLink] = []
            for jdx, other in enumerate(segments):
                if idx == jdx:
                    continue
                shared = token_sets[idx] & token_sets[jdx]
                if len(shared) < self.config.min_shared_tokens:
                    continue
                union = token_sets[idx] | token_sets[jdx]
                if not union:
                    continue
                token_similarity = len(shared) / len(union)
                if token_similarity < self.config.min_similarity:
                    continue
                pattern_similarity = self._pattern_similarity(segment, other)
                combined = (
                    token_similarity * self.config.token_weight
                    + pattern_similarity * self.config.pattern_weight
                )
                if combined < self.config.min_similarity:
                    continue
                candidates.append(
                    SelfSpeechLink(
                        from_index=segment.index,
                        to_index=other.index,
                        shared_tokens=sorted(shared)[: self.config.max_keywords_per_segment],
                        token_similarity=round(token_similarity, 3),
                        pattern_similarity=round(pattern_similarity, 3),
                        combined_score=round(combined, 3),
                    )
                )
            candidates.sort(key=lambda link: link.combined_score, reverse=True)
            for link in candidates[: self.config.max_links_per_segment]:
                links.append(link)
                if len(links) >= self.config.max_links_total:
                    return links
        return links

    def _pattern_similarity(self, left: SelfSpeechSegment, right: SelfSpeechSegment) -> float:
        scores = [
            self._ratio_similarity(left.avg_token_length, right.avg_token_length),
            self._ratio_similarity(left.avg_sentence_length, right.avg_sentence_length),
            self._ratio_similarity(left.uppercase_ratio, right.uppercase_ratio),
        ]
        for key in PUNCT_KEYS:
            scores.append(
                self._ratio_similarity(
                    left.punctuation_per_100.get(key, 0.0),
                    right.punctuation_per_100.get(key, 0.0),
                )
            )
        return sum(scores) / len(scores) if scores else 0.0

    def _ratio_similarity(self, left: float, right: float) -> float:
        if left == right:
            return 1.0
        denom = max(left, right, 1e-6)
        return 1.0 - min(1.0, abs(left - right) / denom)

    def _punctuation_rates(self, text: str, token_count: int) -> dict[str, float]:
        counts = Counter({key: 0 for key in PUNCT_KEYS})
        for key in PUNCT_KEYS:
            counts[key] = text.count(key)
        return self._punctuation_rates_by_counts(counts, token_count)

    def _punctuation_rates_by_counts(
        self, counts: Counter[str], token_count: int
    ) -> dict[str, float]:
        scale = 100.0 / max(token_count, 1)
        return {key: round(counts[key] * scale, 3) for key in PUNCT_KEYS}

    def _sentence_lengths(self, text: str) -> list[int]:
        lengths: list[int] = []
        for sentence in SENTENCE_RE.split(text):
            tokens = self._tokenize(sentence)
            if tokens:
                lengths.append(len(tokens))
        return lengths

    def _avg_sentence_length(self, text: str) -> float:
        lengths = self._sentence_lengths(text)
        if not lengths:
            return 0.0
        return sum(lengths) / len(lengths)

    def _avg_token_length(self, tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        return sum(len(token) for token in tokens) / len(tokens)

    def _uppercase_ratio(self, tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        count = sum(1 for token in tokens if token.isupper() and len(token) > 1)
        return count / len(tokens)

    def _top_keywords(self, tokens: list[str], max_items: int) -> list[str]:
        return [word for word, _ in Counter(tokens).most_common(max_items)]

    def _tokenize(self, text: str) -> list[str]:
        return [
            token.lower()
            for token in WORD_RE.findall(text)
            if len(token) >= self.config.min_token_length
        ]

    def _load_item(self, item: SelfSpeechItem) -> tuple[str, str | None, int | None]:
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
        if not isinstance(event.args, ContextSelfSpeechPatternsArgs):
            return ToolCallDisplay(summary="context_self_speech_patterns")
        if not event.args.items:
            return ToolCallDisplay(summary="context_self_speech_patterns")
        first = event.args.items[0]
        return ToolCallDisplay(
            summary="context_self_speech_patterns",
            details={
                "item_count": len(event.args.items),
                "id": first.id,
                "name": first.name,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSelfSpeechPatternsResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=(
                f"Analyzed {event.result.item_count} dialogue(s) with "
                f"{event.result.total_links} self links"
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
        return "Correlating self speech patterns"
