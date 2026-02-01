from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

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


SENTENCE_RE = re.compile(r"[^.!?]+[.!?]*", re.S)
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


class ContextSpokenFluencyConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_source_bytes: int = Field(
        default=5_000_000, description="Maximum bytes per content."
    )
    max_segments: int = Field(default=200, description="Maximum segments to return.")
    preview_chars: int = Field(default=240, description="Preview snippet length.")
    default_segment_by: str = Field(
        default="sentences", description="sentences, lines, or paragraphs."
    )
    min_segment_chars: int = Field(default=12, description="Minimum segment length.")
    min_token_length: int = Field(default=2, description="Minimum token length.")
    max_keywords: int = Field(default=6, description="Max repeated terms to return.")
    filler_words: list[str] = Field(
        default_factory=lambda: [
            "um",
            "uh",
            "like",
            "you",
            "know",
            "i",
            "mean",
            "sort",
            "kind",
            "okay",
            "right",
            "well",
        ],
        description="Single-word filler tokens.",
    )
    filler_phrases: list[str] = Field(
        default_factory=lambda: [
            "you know",
            "i mean",
            "sort of",
            "kind of",
        ],
        description="Multi-word filler phrases.",
    )
    short_sentence_target: int = Field(
        default=8, description="Target average sentence length."
    )


class ContextSpokenFluencyState(BaseToolState):
    pass


class ContextSpokenFluencyArgs(BaseModel):
    content: str | None = Field(default=None, description="Content to analyze.")
    path: str | None = Field(default=None, description="Path to content.")
    segment_by: str | None = Field(
        default=None, description="sentences, lines, or paragraphs."
    )
    max_segments: int | None = Field(
        default=None, description="Override max segments."
    )
    include_recommendations: bool = Field(
        default=True, description="Include fluency recommendations."
    )


class FluencySegment(BaseModel):
    index: int
    text: str
    word_count: int
    sentence_count: int
    average_sentence_length: float
    filler_count: int
    filler_rate: float
    repetition_rate: float
    repeated_terms: list[str]
    fluency_score: float
    fluency_label: str
    cue: str
    recommendations: list[str]


class ContextSpokenFluencyResult(BaseModel):
    segments: list[FluencySegment]
    segment_count: int
    total_words: int
    overall_filler_rate: float
    overall_repetition_rate: float
    overall_fluency_score: float
    fluency_label: str
    overall_recommendations: list[str]
    speech_opening: str
    speech_closing: str
    warnings: list[str]


class ContextSpokenFluency(
    BaseTool[
        ContextSpokenFluencyArgs,
        ContextSpokenFluencyResult,
        ContextSpokenFluencyConfig,
        ContextSpokenFluencyState,
    ],
    ToolUIData[ContextSpokenFluencyArgs, ContextSpokenFluencyResult],
):
    description: ClassVar[str] = "Evaluate spoken fluency and provide cues."

    async def run(self, args: ContextSpokenFluencyArgs) -> ContextSpokenFluencyResult:
        content = self._load_content(args)
        segment_by = (args.segment_by or self.config.default_segment_by).strip().lower()
        if segment_by not in {"sentences", "lines", "paragraphs"}:
            raise ToolError("segment_by must be sentences, lines, or paragraphs.")

        max_segments = (
            args.max_segments if args.max_segments is not None else self.config.max_segments
        )
        if max_segments <= 0:
            raise ToolError("max_segments must be positive.")

        segments_raw = self._split_segments(content, segment_by)
        segments: list[FluencySegment] = []
        warnings: list[str] = []

        total_words = 0
        total_fillers = 0
        total_repeated = 0
        total_tokens = 0

        for idx, text in enumerate(segments_raw, start=1):
            if len(segments) >= max_segments:
                warnings.append("Segment limit reached; output truncated.")
                break
            if len(text) < self.config.min_segment_chars:
                continue
            segment = self._analyze_segment(text, args.include_recommendations, idx)
            segments.append(segment)
            total_words += segment.word_count
            total_fillers += segment.filler_count
            total_repeated += int(segment.repetition_rate * max(1, segment.word_count))
            total_tokens += segment.word_count

        if not segments:
            raise ToolError("No segments generated.")

        overall_filler_rate = total_fillers / max(1, total_tokens)
        overall_repetition_rate = total_repeated / max(1, total_tokens)
        overall_score = self._fluency_score(
            overall_filler_rate, overall_repetition_rate, self.config.short_sentence_target
        )
        overall_label = self._fluency_label(overall_score)
        overall_recommendations = self._overall_recommendations(
            overall_filler_rate, overall_repetition_rate
        )

        return ContextSpokenFluencyResult(
            segments=segments,
            segment_count=len(segments),
            total_words=total_words,
            overall_filler_rate=overall_filler_rate,
            overall_repetition_rate=overall_repetition_rate,
            overall_fluency_score=overall_score,
            fluency_label=overall_label,
            overall_recommendations=overall_recommendations,
            speech_opening="Start with clear pacing and minimal filler words.",
            speech_closing="Close with a concise summary and smooth transition.",
            warnings=warnings,
        )

    def _load_content(self, args: ContextSpokenFluencyArgs) -> str:
        if args.content and args.path:
            raise ToolError("Provide content or path, not both.")
        if args.content is not None:
            data = args.content.encode("utf-8")
            if len(data) > self.config.max_source_bytes:
                raise ToolError(
                    f"content exceeds max_source_bytes ({len(data)} > {self.config.max_source_bytes})."
                )
            return args.content
        if not args.path:
            raise ToolError("Provide content or path.")
        path = self._resolve_path(args.path)
        size = path.stat().st_size
        if size > self.config.max_source_bytes:
            raise ToolError(
                f"{path} exceeds max_source_bytes ({size} > {self.config.max_source_bytes})."
            )
        return path.read_text("utf-8", errors="ignore")

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.config.effective_workdir / path
        path = path.resolve()
        if not path.exists():
            raise ToolError(f"Path not found: {path}")
        if path.is_dir():
            raise ToolError(f"Path is a directory: {path}")
        return path

    def _split_segments(self, text: str, mode: str) -> list[str]:
        if mode == "lines":
            return [line for line in text.splitlines() if line.strip()]
        if mode == "paragraphs":
            return [chunk for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
        return [seg.strip() for seg in SENTENCE_RE.findall(text) if seg.strip()]

    def _analyze_segment(
        self, text: str, include_recommendations: bool, index: int
    ) -> FluencySegment:
        tokens = [
            token.lower()
            for token in WORD_RE.findall(text)
            if len(token) >= self.config.min_token_length
        ]
        word_count = len(tokens)
        sentence_count = max(1, len([seg for seg in SENTENCE_RE.findall(text) if seg.strip()]))
        avg_sentence_len = word_count / sentence_count if sentence_count else float(word_count)

        filler_count, filler_terms = self._count_fillers(text, tokens)
        filler_rate = filler_count / max(1, word_count)

        repetition_rate, repeated_terms = self._repetition_rate(tokens)

        score = self._fluency_score(filler_rate, repetition_rate, avg_sentence_len)
        label = self._fluency_label(score)

        cue = self._build_cue(label, filler_terms, repeated_terms)
        recommendations = []
        if include_recommendations:
            recommendations = self._segment_recommendations(
                filler_rate, repetition_rate, avg_sentence_len, filler_terms, repeated_terms
            )

        return FluencySegment(
            index=index,
            text=text.strip(),
            word_count=word_count,
            sentence_count=sentence_count,
            average_sentence_length=avg_sentence_len,
            filler_count=filler_count,
            filler_rate=filler_rate,
            repetition_rate=repetition_rate,
            repeated_terms=repeated_terms,
            fluency_score=score,
            fluency_label=label,
            cue=cue,
            recommendations=recommendations,
        )

    def _count_fillers(self, text: str, tokens: list[str]) -> tuple[int, list[str]]:
        filler_words = {word.lower() for word in self.config.filler_words}
        count = sum(1 for token in tokens if token in filler_words)
        found_terms = set(token for token in tokens if token in filler_words)

        lowered = text.lower()
        for phrase in self.config.filler_phrases:
            phrase_key = phrase.lower()
            phrase_count = lowered.count(phrase_key)
            if phrase_count:
                count += phrase_count
                found_terms.add(phrase_key)

        return count, sorted(found_terms)

    def _repetition_rate(self, tokens: list[str]) -> tuple[float, list[str]]:
        filtered = [token for token in tokens if token not in STOPWORDS]
        if not filtered:
            return 0.0, []
        counts = Counter(filtered)
        repeated = [word for word, count in counts.items() if count > 1]
        repeated_count = sum(count - 1 for count in counts.values() if count > 1)
        repetition_rate = repeated_count / max(1, len(filtered))
        repeated.sort(key=lambda word: counts[word], reverse=True)
        return repetition_rate, repeated[: self.config.max_keywords]

    def _fluency_score(
        self, filler_rate: float, repetition_rate: float, avg_sentence_len: float
    ) -> float:
        short_target = max(1, self.config.short_sentence_target)
        short_penalty = max(0.0, (short_target - avg_sentence_len) / short_target) * 0.2
        score = 1.0 - filler_rate * 0.6 - repetition_rate * 0.4 - short_penalty
        return max(0.0, min(1.0, score))

    def _fluency_label(self, score: float) -> str:
        if score >= 0.8:
            return "high"
        if score >= 0.6:
            return "moderate"
        return "low"

    def _build_cue(
        self, label: str, filler_terms: list[str], repeated_terms: list[str]
    ) -> str:
        parts = [f"Fluency {label}"]
        if filler_terms:
            parts.append(f"reduce fillers: {', '.join(filler_terms[:3])}")
        if repeated_terms:
            parts.append(f"avoid repeats: {', '.join(repeated_terms[:3])}")
        return "; ".join(parts) + "."

    def _segment_recommendations(
        self,
        filler_rate: float,
        repetition_rate: float,
        avg_sentence_len: float,
        filler_terms: list[str],
        repeated_terms: list[str],
    ) -> list[str]:
        recommendations: list[str] = []
        if filler_rate > 0.04:
            recommendations.append("Reduce filler words to smooth pacing.")
        if filler_terms:
            recommendations.append(f"Watch filler usage: {', '.join(filler_terms[:4])}.")
        if repetition_rate > 0.08:
            recommendations.append("Vary word choice to avoid repetition.")
        if repeated_terms:
            recommendations.append(f"Rotate terms: {', '.join(repeated_terms[:4])}.")
        if avg_sentence_len < self.config.short_sentence_target:
            recommendations.append("Blend short and medium sentences for flow.")
        return recommendations

    def _overall_recommendations(
        self, filler_rate: float, repetition_rate: float
    ) -> list[str]:
        recs: list[str] = []
        if filler_rate > 0.04:
            recs.append("Reduce filler usage across the full response.")
        if repetition_rate > 0.08:
            recs.append("Increase vocabulary variety to improve fluency.")
        if not recs:
            recs.append("Maintain steady pacing and consistent word choice.")
        return recs

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextSpokenFluencyArgs):
            return ToolCallDisplay(summary="context_spoken_fluency")
        return ToolCallDisplay(
            summary="context_spoken_fluency",
            details={
                "path": event.args.path,
                "segment_by": event.args.segment_by,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpokenFluencyResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = (
            f"Fluency {event.result.fluency_label} "
            f"({event.result.overall_fluency_score:.2f})"
        )
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=event.result.warnings,
            details={
                "segment_count": event.result.segment_count,
                "overall_fluency_score": event.result.overall_fluency_score,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Evaluating spoken fluency"
