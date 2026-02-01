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

REQUEST_MARKERS = {"please", "could you", "would you", "can you"}
DIRECTIVE_MARKERS = {"must", "should", "need", "required"}
COLLAB_MARKERS = {"let's", "lets"}


class ContextSpokenCommunicationConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_source_bytes: int = Field(
        default=5_000_000, description="Maximum bytes per content."
    )
    max_segments: int = Field(default=200, description="Maximum segments to return.")
    preview_chars: int = Field(default=200, description="Preview snippet length.")
    default_segment_by: str = Field(
        default="sentences", description="sentences, lines, or paragraphs."
    )
    min_segment_chars: int = Field(default=12, description="Minimum segment length.")
    max_keywords: int = Field(default=6, description="Max emphasis words per segment.")
    min_token_length: int = Field(default=2, description="Minimum token length.")


class ContextSpokenCommunicationState(BaseToolState):
    pass


class ContextSpokenCommunicationArgs(BaseModel):
    content: str | None = Field(default=None, description="Content to speak.")
    path: str | None = Field(default=None, description="Path to content.")
    speaker: str | None = Field(default=None, description="Speaker name.")
    audience: str | None = Field(default=None, description="Audience name.")
    style: str | None = Field(default=None, description="Spoken style to apply.")
    intent: str | None = Field(default=None, description="Overall intent.")
    segment_by: str | None = Field(
        default=None, description="sentences, lines, or paragraphs."
    )
    max_segments: int | None = Field(
        default=None, description="Override max segments."
    )
    include_opening: bool = Field(
        default=True, description="Include opening cue."
    )
    include_closing: bool = Field(
        default=True, description="Include closing cue."
    )


class SpokenCommunicationSegment(BaseModel):
    index: int
    speaker: str
    text: str
    word_count: int
    intent: str
    tone: str
    pace: str
    emphasis_words: list[str]
    cue: str


class ContextSpokenCommunicationResult(BaseModel):
    segments: list[SpokenCommunicationSegment]
    segment_count: int
    speech_opening: str
    speech_closing: str
    summary_keywords: list[str]
    warnings: list[str]


class ContextSpokenCommunication(
    BaseTool[
        ContextSpokenCommunicationArgs,
        ContextSpokenCommunicationResult,
        ContextSpokenCommunicationConfig,
        ContextSpokenCommunicationState,
    ],
    ToolUIData[ContextSpokenCommunicationArgs, ContextSpokenCommunicationResult],
):
    description: ClassVar[str] = (
        "Transform text into a spoken communication plan with cues."
    )

    async def run(
        self, args: ContextSpokenCommunicationArgs
    ) -> ContextSpokenCommunicationResult:
        content = self._load_content(args)
        segment_by = (args.segment_by or self.config.default_segment_by).strip().lower()
        if segment_by not in {"sentences", "lines", "paragraphs"}:
            raise ToolError("segment_by must be sentences, lines, or paragraphs.")

        max_segments = args.max_segments if args.max_segments is not None else self.config.max_segments
        if max_segments <= 0:
            raise ToolError("max_segments must be positive.")

        speaker = args.speaker or "Speaker"
        audience = args.audience or "listener"
        style = args.style or ""
        default_intent = args.intent or ""

        segments_raw = self._split_segments(content, segment_by)
        segments: list[SpokenCommunicationSegment] = []
        warnings: list[str] = []
        token_counts: Counter[str] = Counter()

        for idx, text in enumerate(segments_raw, start=1):
            if len(segments) >= max_segments:
                warnings.append("Segment limit reached; output truncated.")
                break
            if len(text) < self.config.min_segment_chars:
                continue
            intent = self._detect_intent(text, default_intent)
            tone = "neutral"
            pace = self._pace_from_text(text)
            emphasis = self._emphasis_words(text)
            token_counts.update(emphasis)
            cue = self._build_cue(speaker, intent, tone, pace, emphasis, style)
            segments.append(
                SpokenCommunicationSegment(
                    index=len(segments) + 1,
                    speaker=speaker,
                    text=text.strip(),
                    word_count=len(WORD_RE.findall(text)),
                    intent=intent,
                    tone=tone,
                    pace=pace,
                    emphasis_words=emphasis,
                    cue=cue,
                )
            )

        if not segments:
            raise ToolError("No segments generated.")

        summary_keywords = [word for word, _ in token_counts.most_common(self.config.max_keywords)]
        speech_opening = self._speech_opening(args, speaker, audience, style)
        speech_closing = self._speech_closing(args)

        return ContextSpokenCommunicationResult(
            segments=segments,
            segment_count=len(segments),
            speech_opening=speech_opening,
            speech_closing=speech_closing,
            summary_keywords=summary_keywords,
            warnings=warnings,
        )

    def _load_content(self, args: ContextSpokenCommunicationArgs) -> str:
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
            return [chunk for chunk in re.split(r"\n\\s*\n", text) if chunk.strip()]
        segments = [seg.strip() for seg in SENTENCE_RE.findall(text) if seg.strip()]
        return segments

    def _detect_intent(self, text: str, default_intent: str) -> str:
        lower = text.lower().strip()
        if lower.endswith("?"):
            return "question"
        if any(marker in lower for marker in REQUEST_MARKERS):
            return "request"
        if any(marker in lower for marker in DIRECTIVE_MARKERS):
            return "directive"
        if any(marker in lower for marker in COLLAB_MARKERS):
            return "collaborative"
        return default_intent or "statement"

    def _pace_from_text(self, text: str) -> str:
        word_count = len(WORD_RE.findall(text))
        if word_count <= 8:
            return "quick"
        if word_count <= 20:
            return "steady"
        return "measured"

    def _emphasis_words(self, text: str) -> list[str]:
        tokens = [
            token.lower()
            for token in WORD_RE.findall(text)
            if len(token) >= self.config.min_token_length
        ]
        filtered = [tok for tok in tokens if tok not in STOPWORDS]
        return [word for word, _ in Counter(filtered).most_common(self.config.max_keywords)]

    def _build_cue(
        self,
        speaker: str,
        intent: str,
        tone: str,
        pace: str,
        emphasis: list[str],
        style: str,
    ) -> str:
        parts = [f"{speaker} {intent}", f"tone {tone}", f"pace {pace}"]
        if style:
            parts.append(f"style {style}")
        if emphasis:
            parts.append(f"emphasize {', '.join(emphasis)}")
        return "; ".join(parts) + "."

    def _speech_opening(
        self,
        args: ContextSpokenCommunicationArgs,
        speaker: str,
        audience: str,
        style: str,
    ) -> str:
        if not args.include_opening:
            return ""
        base = f"{speaker} addresses {audience}."
        if style:
            base += f" Use {style}."
        if args.intent:
            base += f" Intent: {args.intent}."
        return base

    def _speech_closing(self, args: ContextSpokenCommunicationArgs) -> str:
        if not args.include_closing:
            return ""
        return "Close by confirming understanding and next steps."

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextSpokenCommunicationArgs):
            return ToolCallDisplay(summary="context_spoken_communication")
        return ToolCallDisplay(
            summary="context_spoken_communication",
            details={
                "path": event.args.path,
                "speaker": event.args.speaker,
                "segment_by": event.args.segment_by,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpokenCommunicationResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = f"Prepared {event.result.segment_count} spoken segment(s)"
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=event.result.warnings,
            details={"segment_count": event.result.segment_count},
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Preparing spoken communication plan"
