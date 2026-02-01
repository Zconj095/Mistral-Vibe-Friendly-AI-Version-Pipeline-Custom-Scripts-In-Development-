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


LINE_TS_RE = re.compile(
    r"^\s*(?:\\[|\\()?(?P<ts>(?:\\d{1,2}:)?\\d{1,2}:\\d{2}(?:\\.\\d{1,3})?)"
    r"(?:\\]|\\))?\\s*"
)
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


class ContextSpeechFrequencyConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_source_bytes: int = Field(
        default=50_000_000, description="Maximum bytes per transcript."
    )
    preview_chars: int = Field(default=200, description="Preview snippet length.")
    default_unit: str = Field(
        default="word", description="word, bigram, trigram, character."
    )
    min_token_length: int = Field(default=2, description="Minimum token length.")
    max_terms: int = Field(default=50, description="Maximum terms to return.")
    include_stopwords: bool = Field(
        default=False, description="Include stopwords in frequency counts."
    )


class ContextSpeechFrequencyState(BaseToolState):
    pass


class ContextSpeechFrequencyArgs(BaseModel):
    content: str | None = Field(default=None, description="Transcript content.")
    path: str | None = Field(default=None, description="Path to transcript.")
    unit: str | None = Field(default=None, description="word, bigram, trigram, character.")
    top_n: int | None = Field(default=None, description="Override max terms.")
    duration_seconds: float | None = Field(
        default=None, description="Total duration in seconds."
    )
    include_stopwords: bool | None = Field(
        default=None, description="Override include_stopwords."
    )
    segment_by_timestamps: bool = Field(
        default=True, description="Use timestamps to segment if present."
    )


class TermFrequency(BaseModel):
    term: str
    count: int
    per_minute: float | None


class SegmentFrequency(BaseModel):
    index: int
    start_time: float
    end_time: float | None
    duration_seconds: float | None
    total_terms: int
    terms_per_minute: float | None
    top_terms: list[TermFrequency]
    preview: str


class ContextSpeechFrequencyResult(BaseModel):
    total_terms: int
    duration_seconds: float | None
    terms_per_minute: float | None
    unit: str
    top_terms: list[TermFrequency]
    segments: list[SegmentFrequency]
    warnings: list[str]


class ContextSpeechFrequency(
    BaseTool[
        ContextSpeechFrequencyArgs,
        ContextSpeechFrequencyResult,
        ContextSpeechFrequencyConfig,
        ContextSpeechFrequencyState,
    ],
    ToolUIData[ContextSpeechFrequencyArgs, ContextSpeechFrequencyResult],
):
    description: ClassVar[str] = "Compute speech frequency statistics from text."

    async def run(
        self, args: ContextSpeechFrequencyArgs
    ) -> ContextSpeechFrequencyResult:
        content = self._load_content(args)
        unit = (args.unit or self.config.default_unit).strip().lower()
        if unit not in {"word", "bigram", "trigram", "character"}:
            raise ToolError("unit must be word, bigram, trigram, or character.")

        top_n = args.top_n if args.top_n is not None else self.config.max_terms
        if top_n <= 0:
            raise ToolError("top_n must be positive.")

        include_stopwords = (
            args.include_stopwords
            if args.include_stopwords is not None
            else self.config.include_stopwords
        )

        segments = self._build_segments(
            content, unit, include_stopwords, args.segment_by_timestamps, top_n
        )
        total_terms = sum(seg.total_terms for seg in segments)

        duration_seconds = self._resolve_duration(segments, args.duration_seconds)
        warnings: list[str] = []

        terms_per_minute = None
        if duration_seconds:
            if duration_seconds <= 0:
                raise ToolError("duration_seconds must be positive.")
            terms_per_minute = (total_terms / duration_seconds) * 60.0
        else:
            warnings.append("Duration not provided or inferred; per-minute rates unavailable.")

        overall_counts = Counter()
        for seg in segments:
            for term in seg.top_terms:
                overall_counts[term.term] += term.count

        top_terms = self._top_terms(overall_counts, top_n, terms_per_minute, total_terms)

        return ContextSpeechFrequencyResult(
            total_terms=total_terms,
            duration_seconds=duration_seconds,
            terms_per_minute=terms_per_minute,
            unit=unit,
            top_terms=top_terms,
            segments=segments,
            warnings=warnings,
        )

    def _load_content(self, args: ContextSpeechFrequencyArgs) -> str:
        if args.content and args.path:
            raise ToolError("Provide content or path, not both.")
        if args.content is None and args.path is None:
            raise ToolError("Provide content or path.")

        if args.content is not None:
            data = args.content.encode("utf-8")
            if len(data) > self.config.max_source_bytes:
                raise ToolError(
                    f"content exceeds max_source_bytes ({len(data)} > {self.config.max_source_bytes})."
                )
            return args.content

        path = self._resolve_path(args.path or "")
        size = path.stat().st_size
        if size > self.config.max_source_bytes:
            raise ToolError(
                f"{path} exceeds max_source_bytes ({size} > {self.config.max_source_bytes})."
            )
        return path.read_text("utf-8", errors="ignore")

    def _resolve_path(self, raw_path: str) -> Path:
        if not raw_path.strip():
            raise ToolError("Path cannot be empty.")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.config.effective_workdir / path
        path = path.resolve()
        if not path.exists():
            raise ToolError(f"Path not found: {path}")
        if path.is_dir():
            raise ToolError(f"Path is a directory: {path}")
        return path

    def _build_segments(
        self,
        content: str,
        unit: str,
        include_stopwords: bool,
        use_timestamps: bool,
        top_n: int,
    ) -> list[SegmentFrequency]:
        lines = content.splitlines()
        segments: list[SegmentFrequency] = []
        current_ts: float | None = None
        buffer: list[str] = []

        def flush(next_ts: float | None) -> None:
            nonlocal current_ts, buffer
            if current_ts is None and not buffer:
                return
            start = current_ts or 0.0
            text = "\n".join(buffer).strip()
            segments.append(
                self._segment_from_text(
                    len(segments) + 1,
                    start,
                    next_ts,
                    text,
                    unit,
                    include_stopwords,
                    top_n,
                )
            )

        if use_timestamps:
            for line in lines:
                match = LINE_TS_RE.match(line)
                if match:
                    ts = self._parse_timestamp(match.group("ts"))
                    flush(ts)
                    current_ts = ts
                    buffer = [line[match.end() :]]
                else:
                    buffer.append(line)
            flush(None)

        if not segments:
            segments.append(
                self._segment_from_text(
                    1,
                    0.0,
                    None,
                    content.strip(),
                    unit,
                    include_stopwords,
                    top_n,
                )
            )

        return segments

    def _segment_from_text(
        self,
        index: int,
        start_time: float,
        end_time: float | None,
        text: str,
        unit: str,
        include_stopwords: bool,
        top_n: int,
    ) -> SegmentFrequency:
        terms = self._extract_terms(text, unit, include_stopwords)
        counts = Counter(terms)
        total_terms = sum(counts.values())
        duration_seconds = None
        terms_per_minute = None
        if end_time is not None and end_time > start_time:
            duration_seconds = end_time - start_time
            terms_per_minute = (total_terms / duration_seconds) * 60.0
        top_terms = self._top_terms(counts, top_n, terms_per_minute, total_terms)
        return SegmentFrequency(
            index=index,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration_seconds,
            total_terms=total_terms,
            terms_per_minute=terms_per_minute,
            top_terms=top_terms,
            preview=self._preview(text),
        )

    def _extract_terms(
        self, text: str, unit: str, include_stopwords: bool
    ) -> list[str]:
        if unit == "character":
            return [ch for ch in text if not ch.isspace()]

        tokens = [
            token.lower()
            for token in WORD_RE.findall(text)
            if len(token) >= self.config.min_token_length
        ]
        if not include_stopwords:
            tokens = [tok for tok in tokens if tok not in STOPWORDS]

        if unit == "word":
            return tokens
        if unit == "bigram":
            return [" ".join(tokens[i : i + 2]) for i in range(len(tokens) - 1)]
        if unit == "trigram":
            return [" ".join(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
        return tokens

    def _top_terms(
        self,
        counts: Counter[str],
        top_n: int,
        terms_per_minute: float | None,
        total_terms: int,
    ) -> list[TermFrequency]:
        top_terms: list[TermFrequency] = []
        for term, count in counts.most_common(top_n):
            per_minute = None
            if terms_per_minute is not None and total_terms > 0:
                per_minute = terms_per_minute * (count / total_terms)
            top_terms.append(
                TermFrequency(term=term, count=count, per_minute=per_minute)
            )
        return top_terms

    def _resolve_duration(
        self, segments: list[SegmentFrequency], duration_seconds: float | None
    ) -> float | None:
        if duration_seconds is not None:
            return duration_seconds
        if len(segments) >= 2:
            last = segments[-1]
            if last.end_time:
                return last.end_time
        return None

    def _parse_timestamp(self, raw: str) -> float:
        parts = raw.split(":")
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        raise ToolError(f"Invalid timestamp: {raw}")

    def _preview(self, text: str) -> str:
        max_chars = self.config.preview_chars
        if max_chars <= 0:
            return ""
        return text if len(text) <= max_chars else text[:max_chars]

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextSpeechFrequencyArgs):
            return ToolCallDisplay(summary="context_speech_frequency")
        return ToolCallDisplay(
            summary="context_speech_frequency",
            details={
                "path": event.args.path,
                "unit": event.args.unit,
                "duration_seconds": event.args.duration_seconds,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpeechFrequencyResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = f"Computed {event.result.unit} frequency"
        if event.result.terms_per_minute:
            message = f"{message} at {event.result.terms_per_minute:.1f} per minute"
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=event.result.warnings,
            details={
                "total_terms": event.result.total_terms,
                "unit": event.result.unit,
                "segments": event.result.segments,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Calculating speech frequency"
