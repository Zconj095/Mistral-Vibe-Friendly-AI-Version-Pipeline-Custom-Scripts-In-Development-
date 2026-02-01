from __future__ import annotations

import importlib.util
import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

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


class ContextSpeechDynamicMemoryConfig(BaseToolConfig):
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
    max_keywords: int = Field(default=20, description="Maximum keywords per item.")
    max_links_per_item: int = Field(default=5, description="Max links per item.")
    max_links_total: int = Field(default=200, description="Maximum total links.")
    min_similarity: float = Field(default=0.1, description="Minimum similarity.")
    max_items_per_window: int = Field(
        default=50, description="Maximum item summaries per window."
    )
    max_question_keywords: int = Field(
        default=12, description="Maximum keywords extracted from question."
    )
    max_answer_segments: int = Field(
        default=6, description="Maximum answer segments for questions."
    )
    max_memory_segments: int = Field(
        default=20, description="Maximum memory segments."
    )
    max_link_segments: int = Field(
        default=8, description="Maximum link segments."
    )
    max_speech_segments: int = Field(
        default=40, description="Maximum total speech segments."
    )


class ContextSpeechDynamicMemoryState(BaseToolState):
    pass


class MemoryItem(BaseModel):
    id: str | None = Field(default=None, description="Optional item id.")
    content: str | None = Field(default=None, description="Inline text content.")
    path: str | None = Field(default=None, description="Path to a text file.")
    timestamp: float | int | str = Field(description="Timestamp value or ISO string.")
    time_unit: str | None = Field(default=None, description="Unit for numeric timestamps.")
    source: str | None = Field(default=None, description="Source description.")


class ContextSpeechDynamicMemoryArgs(BaseModel):
    items: list[MemoryItem] = Field(description="Memory items to analyze.")
    question: str | None = Field(
        default=None, description="Question to answer using memory."
    )
    default_time_unit: str | None = Field(
        default=None, description="Override default time unit."
    )
    short_term_window: str | float | int | None = Field(
        default=None, description="Override short term window duration."
    )
    long_term_window: str | float | int | None = Field(
        default=None, description="Override long term window duration."
    )
    bucket_size: str | float | int | None = Field(
        default=None, description="Override timeline bucket size."
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
    max_question_keywords: int | None = Field(
        default=None, description="Override max_question_keywords."
    )
    max_answer_segments: int | None = Field(
        default=None, description="Override max_answer_segments."
    )
    max_memory_segments: int | None = Field(
        default=None, description="Override max_memory_segments."
    )
    max_link_segments: int | None = Field(
        default=None, description="Override max_link_segments."
    )
    max_speech_segments: int | None = Field(
        default=None, description="Override max_speech_segments."
    )
    include_opening: bool = Field(
        default=True, description="Include speech opening."
    )
    include_closing: bool = Field(
        default=True, description="Include speech closing."
    )


class MemoryItemSummary(BaseModel):
    item_id: str
    window: str
    timestamp_ns: int
    source: str | None
    preview: str
    keywords: list[str]


class MemoryLink(BaseModel):
    short_id: str
    long_id: str
    similarity: float
    shared_terms: list[str]


class QuestionMatch(BaseModel):
    item_id: str
    window: str
    overlap_keywords: list[str]
    score: float


class SpeechSegment(BaseModel):
    index: int
    kind: str
    item_ids: list[str]
    cue: str


class ContextSpeechDynamicMemoryResult(BaseModel):
    question: str | None
    question_keywords: list[str]
    question_matches: list[QuestionMatch]
    short_term_items: list[MemoryItemSummary]
    long_term_items: list[MemoryItemSummary]
    bridge_terms: list[str]
    new_terms: list[str]
    fading_terms: list[str]
    links: list[MemoryLink]
    speech_opening: str
    speech_segments: list[SpeechSegment]
    speech_closing: str
    item_count: int
    truncated: bool
    warnings: list[str]


class ContextSpeechDynamicMemory(
    BaseTool[
        ContextSpeechDynamicMemoryArgs,
        ContextSpeechDynamicMemoryResult,
        ContextSpeechDynamicMemoryConfig,
        ContextSpeechDynamicMemoryState,
    ],
    ToolUIData[ContextSpeechDynamicMemoryArgs, ContextSpeechDynamicMemoryResult],
):
    description: ClassVar[str] = (
        "Reason across dynamic memory windows and prepare speech cues."
    )

    async def run(
        self, args: ContextSpeechDynamicMemoryArgs
    ) -> ContextSpeechDynamicMemoryResult:
        if not args.items:
            raise ToolError("items is required.")

        temporal_module = self._load_temporal_module()

        config = temporal_module.ContextTemporalMemoryConfig(
            max_items=self.config.max_items,
            max_source_bytes=self.config.max_source_bytes,
            max_total_bytes=self.config.max_total_bytes,
            preview_chars=self.config.preview_chars,
            default_time_unit=self.config.default_time_unit,
            short_term_window=self.config.short_term_window,
            long_term_window=self.config.long_term_window,
            bucket_size=self.config.bucket_size,
            min_word_length=self.config.min_word_length,
            max_keywords=self.config.max_keywords,
            max_links_per_item=self.config.max_links_per_item,
            max_links_total=self.config.max_links_total,
            min_similarity=self.config.min_similarity,
            max_items_per_window=self.config.max_items_per_window,
            workdir=self.config.effective_workdir,
        )
        tool = temporal_module.ContextTemporalMemory.from_config(config)

        temporal_items = [
            temporal_module.TemporalItem(**item.model_dump())
            for item in args.items
        ]

        temporal_args = temporal_module.ContextTemporalMemoryArgs(
            items=temporal_items,
            short_term_window=args.short_term_window,
            long_term_window=args.long_term_window,
            bucket_size=args.bucket_size,
            default_time_unit=args.default_time_unit,
            max_items=args.max_items,
            max_source_bytes=args.max_source_bytes,
            max_total_bytes=args.max_total_bytes,
            preview_chars=args.preview_chars,
            min_word_length=args.min_word_length,
            max_keywords=args.max_keywords,
            max_links_per_item=args.max_links_per_item,
            max_links_total=args.max_links_total,
            min_similarity=args.min_similarity,
            max_items_per_window=args.max_items_per_window,
        )

        result = await tool.run(temporal_args)
        warnings = list(result.errors)
        if result.truncated:
            warnings.append("Temporal memory output truncated by limits.")

        short_items = [
            MemoryItemSummary(
                item_id=item.item_id,
                window="short_term",
                timestamp_ns=item.timestamp_ns,
                source=item.source,
                preview=item.preview,
                keywords=item.keywords,
            )
            for item in result.short_term_items
        ]
        long_items = [
            MemoryItemSummary(
                item_id=item.item_id,
                window="long_term",
                timestamp_ns=item.timestamp_ns,
                source=item.source,
                preview=item.preview,
                keywords=item.keywords,
            )
            for item in result.long_term_items
        ]

        question = (args.question or "").strip()
        question_keywords = self._question_keywords(
            question,
            args.min_word_length or self.config.min_word_length,
            args.max_question_keywords or self.config.max_question_keywords,
        )
        question_terms = set(question_keywords)

        question_matches: list[QuestionMatch] = []
        if question_terms:
            for item in short_items + long_items:
                overlap = sorted(question_terms & set(item.keywords))
                if not overlap:
                    continue
                score = len(overlap) / len(question_terms)
                question_matches.append(
                    QuestionMatch(
                        item_id=item.item_id,
                        window=item.window,
                        overlap_keywords=overlap[: self.config.max_keywords],
                        score=round(score, 6),
                    )
                )
            question_matches.sort(key=lambda match: (-match.score, match.item_id))

        links = [
            MemoryLink(
                short_id=link.short_id,
                long_id=link.long_id,
                similarity=link.similarity,
                shared_terms=link.shared_terms,
            )
            for link in result.links
        ]

        speech_opening = self._speech_opening(
            args,
            len(short_items),
            len(long_items),
            question,
        )

        speech_segments, segments_truncated = self._speech_segments(
            short_items,
            long_items,
            links,
            question,
            question_keywords,
            question_matches,
            args.max_answer_segments or self.config.max_answer_segments,
            args.max_memory_segments or self.config.max_memory_segments,
            args.max_link_segments or self.config.max_link_segments,
            args.max_speech_segments or self.config.max_speech_segments,
        )
        if segments_truncated:
            warnings.append("Speech segments truncated by limits.")

        speech_closing = self._speech_closing(args, question)

        return ContextSpeechDynamicMemoryResult(
            question=question or None,
            question_keywords=question_keywords,
            question_matches=question_matches,
            short_term_items=short_items,
            long_term_items=long_items,
            bridge_terms=result.bridge_terms,
            new_terms=result.new_terms,
            fading_terms=result.fading_terms,
            links=links,
            speech_opening=speech_opening,
            speech_segments=speech_segments,
            speech_closing=speech_closing,
            item_count=result.item_count,
            truncated=result.truncated,
            warnings=warnings,
        )

    @staticmethod
    def _load_temporal_module() -> Any:
        module_path = Path(__file__).with_name("context_temporal_memory.py")
        if not module_path.exists():
            raise ToolError("context_temporal_memory.py not found.")
        spec = importlib.util.spec_from_file_location(
            "vibe.tools.context_temporal_memory", module_path
        )
        if spec is None or spec.loader is None:
            raise ToolError("Failed to load context_temporal_memory module.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _question_keywords(
        self, question: str, min_word_length: int, max_keywords: int
    ) -> list[str]:
        if not question or max_keywords <= 0:
            return []
        tokens = [
            token.lower()
            for token in WORD_RE.findall(question)
            if len(token) >= min_word_length and token.lower() not in STOPWORDS
        ]
        counts = Counter(tokens)
        return [token for token, _ in counts.most_common(max_keywords)]

    def _speech_opening(
        self,
        args: ContextSpeechDynamicMemoryArgs,
        short_count: int,
        long_count: int,
        question: str,
    ) -> str:
        if not args.include_opening:
            return ""
        parts = [
            f"Begin reasoning across memory: {short_count} short-term, {long_count} long-term."
        ]
        if question:
            parts.append(f"Question focus: {question}.")
        return " ".join(parts)

    def _speech_segments(
        self,
        short_items: list[MemoryItemSummary],
        long_items: list[MemoryItemSummary],
        links: list[MemoryLink],
        question: str,
        question_keywords: list[str],
        question_matches: list[QuestionMatch],
        max_answer_segments: int,
        max_memory_segments: int,
        max_link_segments: int,
        max_speech_segments: int,
    ) -> tuple[list[SpeechSegment], bool]:
        segments: list[SpeechSegment] = []

        if question and question_matches:
            for match in question_matches[:max_answer_segments]:
                overlap = ", ".join(match.overlap_keywords[:6])
                cue_parts = [f"Answer using memory {match.item_id} ({match.window})."]
                if overlap:
                    cue_parts.append(f"Relevant terms: {overlap}.")
                segments.append(
                    SpeechSegment(
                        index=len(segments) + 1,
                        kind="question",
                        item_ids=[match.item_id],
                        cue=" ".join(cue_parts).strip(),
                    )
                )
        if question and question_keywords:
            cue = f"Anchor the answer with: {', '.join(question_keywords[:6])}."
            segments.append(
                SpeechSegment(
                    index=len(segments) + 1,
                    kind="question_focus",
                    item_ids=[],
                    cue=cue,
                )
            )

        for item in (short_items + long_items)[:max_memory_segments]:
            keywords = ", ".join(item.keywords[:6])
            cue_parts = [f"Recall {item.window} memory {item.item_id}."]
            if item.source:
                cue_parts.append(f"Source: {item.source}.")
            if keywords:
                cue_parts.append(f"Key terms: {keywords}.")
            if item.preview:
                cue_parts.append(f"Context: {item.preview}")
            segments.append(
                SpeechSegment(
                    index=len(segments) + 1,
                    kind="memory",
                    item_ids=[item.item_id],
                    cue=" ".join(cue_parts).strip(),
                )
            )

        for link in links[:max_link_segments]:
            cue = (
                f"Bridge {link.short_id} and {link.long_id} "
                f"on {', '.join(link.shared_terms[:6])}."
            )
            segments.append(
                SpeechSegment(
                    index=len(segments) + 1,
                    kind="link",
                    item_ids=[link.short_id, link.long_id],
                    cue=cue,
                )
            )

        truncated = False
        if max_speech_segments and len(segments) > max_speech_segments:
            segments = segments[:max_speech_segments]
            truncated = True

        for idx, segment in enumerate(segments, start=1):
            segment.index = idx
        return segments, truncated

    def _speech_closing(self, args: ContextSpeechDynamicMemoryArgs, question: str) -> str:
        if not args.include_closing:
            return ""
        if question:
            return "Close by confirming the answer uses both short and long-term memory."
        return "Close by summarizing the most relevant memory links."

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextSpeechDynamicMemoryArgs):
            return ToolCallDisplay(summary="context_speech_dynamic_memory")
        return ToolCallDisplay(
            summary="context_speech_dynamic_memory",
            details={
                "item_count": len(event.args.items),
                "question": event.args.question,
                "short_term_window": event.args.short_term_window,
                "long_term_window": event.args.long_term_window,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpeechDynamicMemoryResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = (
            f"Prepared speech cues for {event.result.item_count} memory item(s)"
        )
        warnings = event.result.warnings[:]
        if event.result.truncated:
            warnings.append("Output truncated by size or limits")
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=warnings,
            details={
                "item_count": event.result.item_count,
                "question": event.result.question,
                "question_match_count": len(event.result.question_matches),
                "short_term_count": len(event.result.short_term_items),
                "long_term_count": len(event.result.long_term_items),
                "link_count": len(event.result.links),
                "truncated": event.result.truncated,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Reasoning across dynamic memory for speech"
