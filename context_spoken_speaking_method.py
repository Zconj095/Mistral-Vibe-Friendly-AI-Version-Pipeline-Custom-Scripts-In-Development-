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


class SpeakingMethod(BaseModel):
    name: str
    description: str
    cue: str


class ContextSpokenSpeakingMethodConfig(BaseToolConfig):
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
    max_methods: int = Field(default=6, description="Maximum methods to return.")
    min_token_length: int = Field(default=2, description="Minimum token length.")
    default_methods: list[SpeakingMethod] = Field(
        default_factory=lambda: [
            SpeakingMethod(
                name="direct",
                description="Deliver the main point first, then details.",
                cue="Lead with the conclusion, then support it.",
            ),
            SpeakingMethod(
                name="structured",
                description="Use clear sections with signposted transitions.",
                cue="State the section, then summarize before moving on.",
            ),
            SpeakingMethod(
                name="dialogic",
                description="Use question-and-answer pacing in the delivery.",
                cue="Pose a question, answer it, then confirm the takeaway.",
            ),
            SpeakingMethod(
                name="narrative",
                description="Explain in a short story-like flow.",
                cue="Describe the situation, action, and result in order.",
            ),
            SpeakingMethod(
                name="instructional",
                description="Give steps in sequence.",
                cue="Number the steps and pause briefly between them.",
            ),
            SpeakingMethod(
                name="summarize_confirm",
                description="Summarize, then confirm next steps.",
                cue="Recap key points, then ask for confirmation.",
            ),
        ],
        description="Default speaking methods to choose from.",
    )


class ContextSpokenSpeakingMethodState(BaseToolState):
    pass


class ContextSpokenSpeakingMethodArgs(BaseModel):
    content: str | None = Field(default=None, description="Content to analyze.")
    path: str | None = Field(default=None, description="Path to content.")
    segment_by: str | None = Field(
        default=None, description="sentences, lines, or paragraphs."
    )
    max_segments: int | None = Field(
        default=None, description="Override max segments."
    )
    methods: list[str] | None = Field(
        default=None, description="Speaking methods to apply."
    )
    method_sequence: list[str] | None = Field(
        default=None, description="Sequence of methods to cycle per segment."
    )
    max_methods: int | None = Field(
        default=None, description="Override max methods."
    )
    include_analysis: bool = Field(
        default=True, description="Include content-driven method hints."
    )


class SpeakingMethodSegment(BaseModel):
    index: int
    text: str
    method: str
    cue: str
    word_count: int
    preview: str


class SpeakingMethodResult(BaseModel):
    methods: list[SpeakingMethod]
    segments: list[SpeakingMethodSegment]
    segment_count: int
    method_sequence: list[str]
    method_prompt: str
    summary_keywords: list[str]
    analysis_notes: list[str]
    speech_opening: str
    speech_closing: str
    warnings: list[str]


class ContextSpokenSpeakingMethod(
    BaseTool[
        ContextSpokenSpeakingMethodArgs,
        SpeakingMethodResult,
        ContextSpokenSpeakingMethodConfig,
        ContextSpokenSpeakingMethodState,
    ],
    ToolUIData[ContextSpokenSpeakingMethodArgs, SpeakingMethodResult],
):
    description: ClassVar[str] = "Generate speaking method cues and prompts."

    async def run(self, args: ContextSpokenSpeakingMethodArgs) -> SpeakingMethodResult:
        warnings: list[str] = []
        analysis_notes: list[str] = []

        max_methods = args.max_methods if args.max_methods is not None else self.config.max_methods
        if max_methods <= 0:
            raise ToolError("max_methods must be positive.")

        methods = list(self.config.default_methods)
        method_lookup = {method.name.lower(): method for method in methods}
        summary_keywords: list[str] = []

        content = None
        if args.content or args.path:
            content = self._load_content(args)
        if args.include_analysis and content:
            summary_keywords = self._top_keywords(content, max_items=6)
            analysis_notes = self._analysis_notes(content)
        elif args.include_analysis and not content:
            warnings.append("No content provided; using default speaking methods.")

        if args.methods:
            methods = self._select_methods(methods, args.methods, warnings)
        elif content and args.include_analysis:
            methods = self._select_by_content(methods, content)

        methods = methods[:max_methods]
        method_sequence = self._resolve_method_sequence(
            args.method_sequence, method_lookup, warnings
        )
        if not method_sequence:
            method_sequence = [method.name for method in methods]

        segments, segment_count = self._build_segments(
            content,
            args,
            method_lookup,
            method_sequence,
            warnings,
        )
        method_prompt = self._method_prompt(methods)
        speech_opening = self._speech_opening(methods)
        speech_closing = "Maintain the chosen speaking method throughout the response."

        return SpeakingMethodResult(
            methods=methods,
            segments=segments,
            segment_count=segment_count,
            method_sequence=method_sequence,
            method_prompt=method_prompt,
            summary_keywords=summary_keywords,
            analysis_notes=analysis_notes,
            speech_opening=speech_opening,
            speech_closing=speech_closing,
            warnings=warnings,
        )

    def _load_content(self, args: ContextSpokenSpeakingMethodArgs) -> str:
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

    def _top_keywords(self, text: str, max_items: int) -> list[str]:
        tokens = [
            token.lower()
            for token in WORD_RE.findall(text)
            if len(token) >= self.config.min_token_length
            and token.lower() not in STOPWORDS
        ]
        return [word for word, _ in Counter(tokens).most_common(max_items)]

    def _analysis_notes(self, text: str) -> list[str]:
        sentences = [seg for seg in SENTENCE_RE.findall(text) if seg.strip()]
        notes = []
        if sentences:
            avg_len = sum(len(WORD_RE.findall(s)) for s in sentences) / len(sentences)
            notes.append(f"Average sentence length: {avg_len:.1f} words.")
        question_count = text.count("?")
        if question_count:
            notes.append(f"Questions detected: {question_count}.")
        return notes

    def _select_methods(
        self, methods: list[SpeakingMethod], requested: list[str], warnings: list[str]
    ) -> list[SpeakingMethod]:
        lookup = {method.name.lower(): method for method in methods}
        selected: list[SpeakingMethod] = []
        for name in requested:
            key = name.strip().lower()
            if key in lookup:
                selected.append(lookup[key])
            else:
                warnings.append(f"Unknown speaking method: {name}")
        return selected if selected else methods

    def _select_by_content(self, methods: list[SpeakingMethod], text: str) -> list[SpeakingMethod]:
        lowered = text.lower()
        if "step" in lowered or "first" in lowered:
            return [m for m in methods if m.name == "instructional"] + methods
        if "why" in lowered or "how" in lowered:
            return [m for m in methods if m.name == "dialogic"] + methods
        return methods

    def _resolve_method_sequence(
        self,
        sequence: list[str] | None,
        lookup: dict[str, SpeakingMethod],
        warnings: list[str],
    ) -> list[str]:
        if not sequence:
            return []
        resolved: list[str] = []
        for name in sequence:
            key = name.strip().lower()
            if key in lookup:
                resolved.append(lookup[key].name)
            else:
                warnings.append(f"Unknown speaking method in sequence: {name}")
        return resolved

    def _build_segments(
        self,
        content: str | None,
        args: ContextSpokenSpeakingMethodArgs,
        lookup: dict[str, SpeakingMethod],
        method_sequence: list[str],
        warnings: list[str],
    ) -> tuple[list[SpeakingMethodSegment], int]:
        if not content:
            return [], 0
        segment_by = (args.segment_by or self.config.default_segment_by).strip().lower()
        if segment_by not in {"sentences", "lines", "paragraphs"}:
            raise ToolError("segment_by must be sentences, lines, or paragraphs.")
        max_segments = (
            args.max_segments
            if args.max_segments is not None
            else self.config.max_segments
        )
        if max_segments <= 0:
            raise ToolError("max_segments must be positive.")
        segments_raw = self._split_segments(content, segment_by)
        segments: list[SpeakingMethodSegment] = []
        if not method_sequence:
            warnings.append("No method sequence provided; no segments generated.")
            return segments, 0

        for idx, text in enumerate(segments_raw, start=1):
            if len(segments) >= max_segments:
                warnings.append("Segment limit reached; output truncated.")
                break
            if len(text) < self.config.min_segment_chars:
                continue
            method_name = method_sequence[(idx - 1) % len(method_sequence)]
            method = lookup.get(method_name.lower())
            cue = method.cue if method else f"Use {method_name} method."
            segments.append(
                SpeakingMethodSegment(
                    index=len(segments) + 1,
                    text=text.strip(),
                    method=method_name,
                    cue=cue,
                    word_count=len(WORD_RE.findall(text)),
                    preview=self._preview(text),
                )
            )
        return segments, len(segments)

    def _method_prompt(self, methods: list[SpeakingMethod]) -> str:
        lines = ["Speaking method cues:"]
        for method in methods:
            lines.append(f"- {method.name}: {method.cue}")
        return "\n".join(lines)

    def _speech_opening(self, methods: list[SpeakingMethod]) -> str:
        names = ", ".join(method.name for method in methods[:3])
        return f"Start speaking with methods: {names}."

    def _split_segments(self, text: str, mode: str) -> list[str]:
        if mode == "lines":
            return [line for line in text.splitlines() if line.strip()]
        if mode == "paragraphs":
            return [chunk for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
        return [seg.strip() for seg in SENTENCE_RE.findall(text) if seg.strip()]

    def _preview(self, text: str) -> str:
        max_chars = self.config.preview_chars
        if max_chars <= 0:
            return ""
        return text if len(text) <= max_chars else text[:max_chars]

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextSpokenSpeakingMethodArgs):
            return ToolCallDisplay(summary="context_spoken_speaking_method")
        return ToolCallDisplay(
            summary="context_spoken_speaking_method",
            details={
                "path": event.args.path,
                "methods": event.args.methods,
                "method_sequence": event.args.method_sequence,
                "segment_by": event.args.segment_by,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, SpeakingMethodResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = (
            f"Prepared {len(event.result.methods)} speaking method(s) "
            f"with {event.result.segment_count} segment(s)"
        )
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=event.result.warnings,
            details={"methods": [method.name for method in event.result.methods]},
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Preparing speaking method cues"
