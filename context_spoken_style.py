from __future__ import annotations

import re
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


class StyleSource(BaseModel):
    path: str
    category: str = Field(default="Speech Styles")
    encoding: str = Field(default="utf-8")


class ContextSpokenStyleConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    style_sources: list[StyleSource] = Field(
        default_factory=lambda: [
            StyleSource(
                path=r"C:\Users\Zack\.vibe\definitions\speech_styles_data.py",
                category="Speech Styles",
            )
        ],
        description="Sources that define spoken style terms and descriptions.",
    )
    max_segments: int = Field(default=200, description="Maximum segments to return.")
    max_styles: int = Field(default=8, description="Maximum styles to return.")
    max_source_bytes: int = Field(default=3_000_000, description="Max bytes per source.")
    preview_chars: int = Field(default=240, description="Preview snippet length.")
    default_segment_by: str = Field(
        default="sentences", description="sentences, lines, or paragraphs."
    )
    min_segment_chars: int = Field(default=12, description="Minimum segment length.")


class ContextSpokenStyleState(BaseToolState):
    pass


class ContextSpokenStyleArgs(BaseModel):
    styles: list[str] | None = Field(default=None, description="Style names to apply.")
    categories: list[str] | None = Field(
        default=None, description="Style categories to include."
    )
    content: str | None = Field(default=None, description="Optional text to match styles.")
    path: str | None = Field(default=None, description="Path to text to match styles.")
    segment_by: str | None = Field(
        default=None, description="sentences, lines, or paragraphs."
    )
    max_segments: int | None = Field(
        default=None, description="Override max segments."
    )
    style_sequence: list[str] | None = Field(
        default=None, description="Sequence of styles to cycle per segment."
    )
    max_styles: int | None = Field(
        default=None, description="Override maximum styles."
    )


class SpokenStyleDefinition(BaseModel):
    term: str
    normalized_term: str
    description: str
    category: str
    source_path: str
    occurrences: int | None = None


class SpokenStyleProfile(BaseModel):
    term: str
    category: str
    description: str
    cue: str
    occurrences: int | None


class SpokenStyleSegment(BaseModel):
    index: int
    text: str
    style: str
    cue: str
    word_count: int
    preview: str


class ContextSpokenStyleResult(BaseModel):
    styles: list[SpokenStyleProfile]
    segments: list[SpokenStyleSegment]
    segment_count: int
    style_sequence: list[str]
    style_prompt: str
    speech_opening: str
    speech_closing: str
    warnings: list[str]


class ContextSpokenStyle(
    BaseTool[
        ContextSpokenStyleArgs,
        ContextSpokenStyleResult,
        ContextSpokenStyleConfig,
        ContextSpokenStyleState,
    ],
    ToolUIData[ContextSpokenStyleArgs, ContextSpokenStyleResult],
):
    description: ClassVar[str] = "Build a spoken style prompt from style definitions."

    async def run(self, args: ContextSpokenStyleArgs) -> ContextSpokenStyleResult:
        definitions, warnings = self._load_definitions()
        if not definitions:
            raise ToolError("No spoken style definitions available.")

        max_styles = args.max_styles if args.max_styles is not None else self.config.max_styles
        if max_styles <= 0:
            raise ToolError("max_styles must be positive.")

        selected = self._select_styles(args, definitions, max_styles, warnings)
        if not selected:
            raise ToolError("No spoken styles selected.")

        style_lookup = {item.normalized_term.lower(): item for item in definitions}
        profiles = [
            SpokenStyleProfile(
                term=item.term,
                category=item.category,
                description=item.description,
                cue=self._build_cue(item),
                occurrences=getattr(item, "occurrences", None),
            )
            for item in selected
        ]

        style_sequence = self._resolve_style_sequence(
            args.style_sequence, profiles, style_lookup, warnings
        )
        if not style_sequence:
            style_sequence = [profile.term for profile in profiles]

        segments = self._build_segments(args, style_sequence, style_lookup, warnings)

        style_prompt = self._build_style_prompt(profiles)
        speech_opening = self._speech_opening(profiles)
        speech_closing = self._speech_closing()

        return ContextSpokenStyleResult(
            styles=profiles,
            segments=segments,
            segment_count=len(segments),
            style_sequence=style_sequence,
            style_prompt=style_prompt,
            speech_opening=speech_opening,
            speech_closing=speech_closing,
            warnings=warnings,
        )

    def _select_styles(
        self,
        args: ContextSpokenStyleArgs,
        definitions: list[SpokenStyleDefinition],
        max_styles: int,
        warnings: list[str],
    ) -> list[SpokenStyleDefinition]:
        if args.styles:
            return self._match_requested_styles(args.styles, definitions, warnings)

        if args.categories:
            categories = {self._normalize_term(cat).lower() for cat in args.categories}
            filtered = [d for d in definitions if d.category.lower() in categories]
            return filtered[:max_styles]

        if args.content or args.path:
            content = self._load_content(args)
            matches = self._match_content(content, definitions)
            matches.sort(key=lambda item: item.occurrences or 0, reverse=True)
            return matches[:max_styles]

        warnings.append("No styles specified; defaulting to the first available styles.")
        return definitions[:max_styles]

    def _match_requested_styles(
        self,
        styles: list[str],
        definitions: list[SpokenStyleDefinition],
        warnings: list[str],
    ) -> list[SpokenStyleDefinition]:
        normalized = {self._normalize_term(style).lower() for style in styles}
        lookup = {item.normalized_term.lower(): item for item in definitions}
        selected: list[SpokenStyleDefinition] = []
        for term in normalized:
            if term in lookup:
                selected.append(lookup[term])
                continue
            alt = term.replace("-", " ")
            if alt in lookup:
                selected.append(lookup[alt])
            else:
                warnings.append(f"Unknown spoken style: {term}")
        return selected

    def _match_content(
        self, content: str, definitions: list[SpokenStyleDefinition]
    ) -> list[SpokenStyleDefinition]:
        lower = content.lower()
        space_normalized = lower.replace("-", " ")
        matches: list[SpokenStyleDefinition] = []
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

    def _resolve_style_sequence(
        self,
        sequence: list[str] | None,
        profiles: list[SpokenStyleProfile],
        lookup: dict[str, SpokenStyleDefinition],
        warnings: list[str],
    ) -> list[str]:
        if not sequence:
            return []
        resolved: list[str] = []
        for name in sequence:
            key = self._normalize_term(name).lower()
            if key in lookup:
                resolved.append(lookup[key].term)
                continue
            alt = key.replace("-", " ")
            if alt in lookup:
                resolved.append(lookup[alt].term)
            else:
                warnings.append(f"Unknown spoken style in sequence: {name}")
        if not resolved:
            resolved = [profile.term for profile in profiles]
        return resolved

    def _build_segments(
        self,
        args: ContextSpokenStyleArgs,
        style_sequence: list[str],
        lookup: dict[str, SpokenStyleDefinition],
        warnings: list[str],
    ) -> list[SpokenStyleSegment]:
        if not args.content and not args.path:
            return []
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
        segments: list[SpokenStyleSegment] = []
        if not style_sequence:
            warnings.append("No style sequence provided; no segments generated.")
            return segments

        for idx, text in enumerate(segments_raw, start=1):
            if len(segments) >= max_segments:
                warnings.append("Segment limit reached; output truncated.")
                break
            if len(text) < self.config.min_segment_chars:
                continue
            style_name = style_sequence[(idx - 1) % len(style_sequence)]
            definition = lookup.get(style_name.lower())
            cue = (
                f"Use {style_name}: {definition.description}"
                if definition
                else f"Use {style_name} style."
            )
            segments.append(
                SpokenStyleSegment(
                    index=len(segments) + 1,
                    text=text.strip(),
                    style=style_name,
                    cue=cue,
                    word_count=len(re.findall(r"[A-Za-z0-9_']+", text)),
                    preview=self._preview(text),
                )
            )
        return segments

    def _build_cue(self, definition: SpokenStyleDefinition) -> str:
        return f"Use {definition.term}: {definition.description}"

    def _build_style_prompt(self, profiles: list[SpokenStyleProfile]) -> str:
        lines = ["Spoken style targets:"]
        for profile in profiles:
            lines.append(f"- {profile.term}: {profile.description}")
        lines.append("Apply these cues to tone, pacing, and word choice.")
        return "\n".join(lines)

    def _speech_opening(self, profiles: list[SpokenStyleProfile]) -> str:
        names = ", ".join(profile.term for profile in profiles)
        return f"Begin speaking with the following styles: {names}."

    def _speech_closing(self) -> str:
        return "Close by keeping the chosen spoken style consistent."

    def _split_segments(self, text: str, mode: str) -> list[str]:
        if mode == "lines":
            return [line for line in text.splitlines() if line.strip()]
        if mode == "paragraphs":
            return [chunk for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
        return [seg.strip() for seg in re.findall(r"[^.!?]+[.!?]*", text) if seg.strip()]

    def _preview(self, text: str) -> str:
        max_chars = self.config.preview_chars
        if max_chars <= 0:
            return ""
        return text if len(text) <= max_chars else text[:max_chars]

    def _load_content(self, args: ContextSpokenStyleArgs) -> str:
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

    def _load_definitions(self) -> tuple[list[SpokenStyleDefinition], list[str]]:
        definitions: list[SpokenStyleDefinition] = []
        warnings: list[str] = []
        seen: set[tuple[str, str]] = set()
        for src in self.config.style_sources:
            try:
                path = Path(src.path).expanduser()
                if not path.is_absolute():
                    path = (self.config.effective_workdir / path).resolve()
                if not path.exists():
                    warnings.append(f"Spoken style definitions missing: {path}")
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
    ) -> list[SpokenStyleDefinition]:
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

        entries: list[SpokenStyleDefinition] = []
        for raw in raw_defs:
            if not isinstance(raw, dict):
                continue
            term = self._normalize_term(str(raw.get("term", "")))
            description = self._normalize_description(str(raw.get("description", "")))
            category = str(raw.get("category", source.category))
            if not term:
                continue
            entries.append(
                SpokenStyleDefinition(
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
        if not isinstance(event.args, ContextSpokenStyleArgs):
            return ToolCallDisplay(summary="context_spoken_style")
        return ToolCallDisplay(
            summary="context_spoken_style",
            details={
                "styles": event.args.styles,
                "categories": event.args.categories,
                "path": event.args.path,
                "style_sequence": event.args.style_sequence,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpokenStyleResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = (
            f"Prepared {len(event.result.styles)} spoken style(s) "
            f"with {event.result.segment_count} segment(s)"
        )
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=event.result.warnings,
            details={"styles": [style.term for style in event.result.styles]},
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Preparing spoken style prompt"
