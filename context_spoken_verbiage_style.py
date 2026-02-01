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


class StyleSource(BaseModel):
    path: str
    category: str = Field(default="Speech Styles")
    encoding: str = Field(default="utf-8")


class ContextSpokenVerbiageStyleConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    style_sources: list[StyleSource] = Field(
        default_factory=lambda: [
            StyleSource(
                path=r"C:\Users\Zack\.vibe\definitions\speech_styles_data.py",
                category="Speech Styles",
            )
        ],
        description="Sources that define spoken verbiage styles.",
    )
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
    max_keywords: int = Field(default=8, description="Max keywords per segment.")
    max_styles: int = Field(default=5, description="Max styles to apply.")


class ContextSpokenVerbiageStyleState(BaseToolState):
    pass


class ContextSpokenVerbiageStyleArgs(BaseModel):
    content: str | None = Field(default=None, description="Content to style.")
    path: str | None = Field(default=None, description="Path to content.")
    styles: list[str] | None = Field(default=None, description="Style names to apply.")
    categories: list[str] | None = Field(
        default=None, description="Style categories to include."
    )
    segment_by: str | None = Field(
        default=None, description="sentences, lines, or paragraphs."
    )
    max_segments: int | None = Field(
        default=None, description="Override max segments."
    )
    max_styles: int | None = Field(default=None, description="Override max styles.")


class VerbiageStyleDefinition(BaseModel):
    term: str
    normalized_term: str
    description: str
    category: str
    source_path: str
    occurrences: int | None = None


class VerbiageStyleProfile(BaseModel):
    term: str
    category: str
    description: str
    cue: str


class VerbiageSegment(BaseModel):
    index: int
    text: str
    style: str
    keywords: list[str]
    cue: str


class ContextSpokenVerbiageStyleResult(BaseModel):
    styles: list[VerbiageStyleProfile]
    segments: list[VerbiageSegment]
    segment_count: int
    verbiage_prompt: str
    speech_opening: str
    speech_closing: str
    warnings: list[str]


class ContextSpokenVerbiageStyle(
    BaseTool[
        ContextSpokenVerbiageStyleArgs,
        ContextSpokenVerbiageStyleResult,
        ContextSpokenVerbiageStyleConfig,
        ContextSpokenVerbiageStyleState,
    ],
    ToolUIData[
        ContextSpokenVerbiageStyleArgs, ContextSpokenVerbiageStyleResult
    ],
):
    description: ClassVar[str] = (
        "Build spoken verbiage style cues for text segments."
    )

    async def run(
        self, args: ContextSpokenVerbiageStyleArgs
    ) -> ContextSpokenVerbiageStyleResult:
        content = self._load_content(args)
        segment_by = (args.segment_by or self.config.default_segment_by).strip().lower()
        if segment_by not in {"sentences", "lines", "paragraphs"}:
            raise ToolError("segment_by must be sentences, lines, or paragraphs.")

        max_segments = (
            args.max_segments if args.max_segments is not None else self.config.max_segments
        )
        if max_segments <= 0:
            raise ToolError("max_segments must be positive.")

        max_styles = (
            args.max_styles if args.max_styles is not None else self.config.max_styles
        )
        if max_styles <= 0:
            raise ToolError("max_styles must be positive.")

        definitions, warnings = self._load_definitions()
        if not definitions:
            raise ToolError("No spoken verbiage styles available.")

        selected = self._select_styles(args, content, definitions, max_styles, warnings)
        if not selected:
            raise ToolError("No spoken verbiage styles selected.")

        style_profiles = [
            VerbiageStyleProfile(
                term=item.term,
                category=item.category,
                description=item.description,
                cue=self._build_style_cue(item),
            )
            for item in selected
        ]

        segments_raw = self._split_segments(content, segment_by)
        segments: list[VerbiageSegment] = []
        style_cycle = [style.term for style in style_profiles]
        style_count = len(style_cycle)
        if style_count == 0:
            raise ToolError("No styles available for segmentation.")

        for idx, text in enumerate(segments_raw, start=1):
            if len(segments) >= max_segments:
                warnings.append("Segment limit reached; output truncated.")
                break
            if len(text) < self.config.min_segment_chars:
                continue
            style_name = style_cycle[(idx - 1) % style_count]
            keywords = self._extract_keywords(text, self.config.max_keywords)
            cue = self._build_segment_cue(style_name, keywords)
            segments.append(
                VerbiageSegment(
                    index=len(segments) + 1,
                    text=text.strip(),
                    style=style_name,
                    keywords=keywords,
                    cue=cue,
                )
            )

        verbiage_prompt = self._build_verbiage_prompt(style_profiles)
        speech_opening = self._speech_opening(style_profiles)
        speech_closing = self._speech_closing()

        return ContextSpokenVerbiageStyleResult(
            styles=style_profiles,
            segments=segments,
            segment_count=len(segments),
            verbiage_prompt=verbiage_prompt,
            speech_opening=speech_opening,
            speech_closing=speech_closing,
            warnings=warnings,
        )

    def _load_content(self, args: ContextSpokenVerbiageStyleArgs) -> str:
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

    def _extract_keywords(self, text: str, max_items: int) -> list[str]:
        tokens = []
        for token in WORD_RE.findall(text):
            lower = token.lower()
            if len(lower) < self.config.min_token_length:
                continue
            if lower in STOPWORDS:
                continue
            tokens.append(lower)
        return [word for word, _ in Counter(tokens).most_common(max_items)]

    def _select_styles(
        self,
        args: ContextSpokenVerbiageStyleArgs,
        content: str,
        definitions: list[VerbiageStyleDefinition],
        max_styles: int,
        warnings: list[str],
    ) -> list[VerbiageStyleDefinition]:
        if args.styles:
            return self._match_requested_styles(args.styles, definitions, warnings)

        if args.categories:
            categories = {self._normalize_term(cat).lower() for cat in args.categories}
            filtered = [d for d in definitions if d.category.lower() in categories]
            return filtered[:max_styles]

        matches = self._match_content(content, definitions)
        if matches:
            matches.sort(key=lambda item: item.occurrences or 0, reverse=True)
            return matches[:max_styles]

        warnings.append("No matching styles found; using defaults.")
        return definitions[:max_styles]

    def _match_requested_styles(
        self,
        styles: list[str],
        definitions: list[VerbiageStyleDefinition],
        warnings: list[str],
    ) -> list[VerbiageStyleDefinition]:
        normalized = {self._normalize_term(style).lower() for style in styles}
        lookup = {item.normalized_term.lower(): item for item in definitions}
        selected: list[VerbiageStyleDefinition] = []
        for term in normalized:
            if term in lookup:
                selected.append(lookup[term])
                continue
            alt = term.replace("-", " ")
            if alt in lookup:
                selected.append(lookup[alt])
            else:
                warnings.append(f"Unknown spoken verbiage style: {term}")
        return selected

    def _match_content(
        self, content: str, definitions: list[VerbiageStyleDefinition]
    ) -> list[VerbiageStyleDefinition]:
        lower = content.lower()
        space_normalized = lower.replace("-", " ")
        matches: list[VerbiageStyleDefinition] = []
        for definition in definitions:
            key = definition.normalized_term.lower()
            alt = key.replace("-", " ")
            occurrences = space_normalized.count(alt)
            if occurrences <= 0:
                continue
            definition = definition.model_copy()
            definition.occurrences = occurrences
            matches.append(definition)
        return matches

    def _build_style_cue(self, definition: VerbiageStyleDefinition) -> str:
        return f"Use {definition.term} verbiage: {definition.description}"

    def _build_segment_cue(self, style: str, keywords: list[str]) -> str:
        if keywords:
            return f"{style} verbiage; emphasize {', '.join(keywords[:5])}."
        return f"{style} verbiage; keep phrasing consistent."

    def _build_verbiage_prompt(self, styles: list[VerbiageStyleProfile]) -> str:
        lines = ["Spoken verbiage style:"]
        for profile in styles:
            lines.append(f"- {profile.term}: {profile.description}")
        lines.append("Keep word choice aligned to these styles.")
        return "\n".join(lines)

    def _speech_opening(self, styles: list[VerbiageStyleProfile]) -> str:
        names = ", ".join(style.term for style in styles)
        return f"Begin speaking with {names} verbiage."

    def _speech_closing(self) -> str:
        return "Close by maintaining the same verbiage style."

    def _load_definitions(self) -> tuple[list[VerbiageStyleDefinition], list[str]]:
        definitions: list[VerbiageStyleDefinition] = []
        warnings: list[str] = []
        seen: set[tuple[str, str]] = set()
        for src in self.config.style_sources:
            try:
                path = Path(src.path).expanduser()
                if not path.is_absolute():
                    path = (self.config.effective_workdir / path).resolve()
                if not path.exists():
                    warnings.append(f"Verbiage style definitions missing: {path}")
                    continue
                parsed = self._parse_python_definitions(path, src)
                for item in parsed:
                    key = (item.normalized_term, item.category)
                    if key in seen:
                        continue
                    seen.add(key)
                    definitions.append(item)
            except Exception as exc:
                warnings.append(f"failed to read {src.path}: {exc}")
        return definitions, warnings

    def _parse_python_definitions(
        self, path: Path, source: StyleSource
    ) -> list[VerbiageStyleDefinition]:
        try:
            import importlib.util
        except Exception as exc:
            raise ToolError(f"Unable to import definitions module: {exc}") from exc

        spec = importlib.util.spec_from_file_location("speech_styles_data", path)
        if spec is None or spec.loader is None:
            raise ToolError(f"Unable to load definitions module: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        raw_defs = getattr(module, "DEFINITIONS", None)
        if not isinstance(raw_defs, list):
            raise ToolError(f"DEFINITIONS missing or invalid in: {path}")

        entries: list[VerbiageStyleDefinition] = []
        for raw in raw_defs:
            if not isinstance(raw, dict):
                continue
            term = self._normalize_term(str(raw.get("term", "")))
            description = self._normalize_description(str(raw.get("description", "")))
            category = str(raw.get("category", source.category))
            if not term:
                continue
            entries.append(
                VerbiageStyleDefinition(
                    term=term,
                    normalized_term=term.lower(),
                    description=description or term,
                    category=category,
                    source_path=str(path),
                )
            )
        return entries

    def _normalize_term(self, term: str) -> str:
        return " ".join(term.strip().split())

    def _normalize_description(self, description: str) -> str:
        return " ".join(description.strip().split())

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextSpokenVerbiageStyleArgs):
            return ToolCallDisplay(summary="context_spoken_verbiage_style")
        return ToolCallDisplay(
            summary="context_spoken_verbiage_style",
            details={
                "path": event.args.path,
                "styles": event.args.styles,
                "categories": event.args.categories,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpokenVerbiageStyleResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = f"Prepared {event.result.segment_count} verbiage segment(s)"
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=event.result.warnings,
            details={
                "segment_count": event.result.segment_count,
                "styles": [style.term for style in event.result.styles],
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Preparing spoken verbiage style"
