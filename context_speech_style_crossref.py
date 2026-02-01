from __future__ import annotations

from pathlib import Path
import re
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


WORD_RE = re.compile(r"[A-Za-z0-9_]+")
SPEAKER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _.'-]{0,40}):\s*(.*)$")


class SpeechStyleSource(BaseModel):
    path: str
    category: str = Field(default="Speech Styles")
    encoding: str = Field(default="utf-8")


class ContextSpeechStyleCrossrefConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    style_sources: list[SpeechStyleSource] = Field(
        default_factory=lambda: [
            SpeechStyleSource(
                path=r"C:\Users\Zack\.vibe\definitions\speech_styles_data.py",
                category="Speech Styles",
            )
        ],
        description="Sources that define speech emphasis styles and tags.",
    )
    max_items: int = Field(default=40, description="Maximum items to evaluate.")
    max_source_bytes: int = Field(default=3_000_000, description="Maximum bytes per item.")
    max_total_bytes: int = Field(default=20_000_000, description="Max bytes across items.")
    preview_chars: int = Field(default=400, description="Preview snippet length.")
    max_segments: int = Field(default=200, description="Maximum segments per item.")
    min_segment_chars: int = Field(default=20, description="Minimum segment length.")
    max_styles_per_segment: int = Field(default=20, description="Limit styles per segment.")
    min_shared_styles: int = Field(default=1, description="Minimum shared styles for linking.")
    link_window_segments: int = Field(
        default=50,
        description="Segments to look back for cross references (0 = all).",
    )
    max_crossrefs_per_segment: int = Field(
        default=3, description="Maximum cross references per segment."
    )
    max_crossrefs_total: int = Field(default=500, description="Maximum cross references.")


class ContextSpeechStyleCrossrefState(BaseToolState):
    pass


class SpeechStyleItem(BaseModel):
    id: str | None = Field(default=None)
    name: str | None = Field(default=None)
    content: str | None = Field(default=None)
    path: str | None = Field(default=None)
    source: str | None = Field(default=None)



class ContextSpeechStyleCrossrefArgs(BaseModel):
    items: list[SpeechStyleItem] = Field(description="Items to evaluate.")

class SpeechStyleDefinition(BaseModel):
    term: str
    normalized_term: str
    description: str
    category: str
    source_path: str


class SpeechStyleMatch(BaseModel):
    term: str
    category: str


class SpeechStyleSegment(BaseModel):
    index: int
    speaker: str | None
    text: str
    start: int
    end: int
    styles: list[SpeechStyleMatch]


class StyleCrossref(BaseModel):
    term: str
    category: str
    segment_indices: list[int]
    count: int


class SegmentCrossref(BaseModel):
    from_index: int
    to_index: int
    shared_styles: list[str]
    shared_count: int


class SpeechStyleCrossrefInsight(BaseModel):
    index: int
    id: str | None
    name: str | None
    source_path: str | None
    preview: str
    segments: list[SpeechStyleSegment]
    style_crossrefs: list[StyleCrossref]
    segment_crossrefs: list[SegmentCrossref]
    segment_count: int
    crossref_count: int


class ContextSpeechStyleCrossrefResult(BaseModel):
    items: list[SpeechStyleCrossrefInsight]
    item_count: int
    total_crossrefs: int
    truncated: bool
    warnings: list[str]
    errors: list[str]


class ContextSpeechStyleCrossref(
    BaseTool[
        ContextSpeechStyleCrossrefArgs,
        ContextSpeechStyleCrossrefResult,
        ContextSpeechStyleCrossrefConfig,
        ContextSpeechStyleCrossrefState,
    ],
    ToolUIData[ContextSpeechStyleCrossrefArgs, ContextSpeechStyleCrossrefResult],
):
    description: ClassVar[str] = (
        "Cross-reference speaking styles across a conversation."
    )

    async def run(self, args: ContextSpeechStyleCrossrefArgs) -> ContextSpeechStyleCrossrefResult:
        items = list(args.items)
        if not items:
            raise ToolError("items is required.")

        definitions, def_warnings = self._load_definitions()
        if not definitions:
            raise ToolError("No speech style definitions available.")

        errors: list[str] = []
        warnings: list[str] = def_warnings.copy()
        insights: list[SpeechStyleCrossrefInsight] = []
        total_bytes = 0
        truncated = False
        total_crossrefs = 0

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

                segments = self._segment_conversation(content)
                segments = self._attach_styles(segments, definitions)
                style_crossrefs = self._build_style_crossrefs(segments)
                segment_crossrefs = self._build_segment_crossrefs(segments)
                total_crossrefs += len(segment_crossrefs)

                insights.append(
                    SpeechStyleCrossrefInsight(
                        index=len(insights) + 1,
                        id=item.id,
                        name=item.name,
                        source_path=source_path,
                        preview=self._preview(content),
                        segments=segments[: self.config.max_segments],
                        style_crossrefs=style_crossrefs,
                        segment_crossrefs=segment_crossrefs,
                        segment_count=len(segments),
                        crossref_count=len(segment_crossrefs),
                    )
                )
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not insights:
            raise ToolError("No insights generated.")

        return ContextSpeechStyleCrossrefResult(
            items=insights,
            item_count=len(insights),
            total_crossrefs=total_crossrefs,
            truncated=truncated,
            warnings=warnings,
            errors=errors,
        )

    def _segment_conversation(self, text: str) -> list[SpeechStyleSegment]:
        segments: list[SpeechStyleSegment] = []
        lines = text.splitlines()
        current_lines: list[str] = []
        current_speaker: str | None = None
        seg_start = 0
        cursor = 0

        def flush(end_offset: int) -> None:
            nonlocal current_lines, current_speaker, seg_start
            if not current_lines:
                return
            seg_text = "\n".join(current_lines).strip()
            if not seg_text:
                current_lines = []
                return
            if len(seg_text) < self.config.min_segment_chars and segments:
                segments[-1].text = f"{segments[-1].text}\n{seg_text}".strip()
                segments[-1].end = end_offset
            else:
                segments.append(
                    SpeechStyleSegment(
                        index=len(segments) + 1,
                        speaker=current_speaker,
                        text=seg_text,
                        start=seg_start,
                        end=end_offset,
                        styles=[],
                    )
                )
            current_lines = []
            current_speaker = None

        for line in lines:
            line_len = len(line) + 1
            speaker_match = SPEAKER_RE.match(line)
            if speaker_match:
                flush(cursor)
                current_speaker = speaker_match.group(1).strip()
                current_lines.append(speaker_match.group(2))
                seg_start = cursor
            elif not line.strip():
                flush(cursor)
            else:
                if not current_lines:
                    seg_start = cursor
                current_lines.append(line)
            cursor += line_len

        flush(cursor)
        return segments[: self.config.max_segments]

    def _attach_styles(
        self, segments: list[SpeechStyleSegment], definitions: list[SpeechStyleDefinition]
    ) -> list[SpeechStyleSegment]:
        for segment in segments:
            styles = self._match_styles(segment.text, definitions)
            segment.styles = styles[: self.config.max_styles_per_segment]
        return segments

    def _match_styles(
        self, text: str, definitions: list[SpeechStyleDefinition]
    ) -> list[SpeechStyleMatch]:
        lower = text.lower()
        alt_text = lower.replace("-", " ")
        matches: dict[tuple[str, str], SpeechStyleMatch] = {}
        for definition in definitions:
            key = definition.normalized_term
            if not key:
                continue
            if key in lower or key.replace("-", " ") in alt_text or key.replace(" ", "-") in lower:
                match_key = (definition.term, definition.category)
                matches[match_key] = SpeechStyleMatch(
                    term=definition.term,
                    category=definition.category,
                )
        return list(matches.values())

    def _build_style_crossrefs(
        self, segments: list[SpeechStyleSegment]
    ) -> list[StyleCrossref]:
        crossrefs: dict[tuple[str, str], list[int]] = {}
        for segment in segments:
            for style in segment.styles:
                key = (style.term, style.category)
                crossrefs.setdefault(key, []).append(segment.index)
        output: list[StyleCrossref] = []
        for (term, category), indices in crossrefs.items():
            output.append(
                StyleCrossref(
                    term=term,
                    category=category,
                    segment_indices=indices,
                    count=len(indices),
                )
            )
        output.sort(key=lambda ref: (-ref.count, ref.term))
        return output

    def _build_segment_crossrefs(
        self, segments: list[SpeechStyleSegment]
    ) -> list[SegmentCrossref]:
        crossrefs: list[SegmentCrossref] = []
        style_sets = [
            {style.term for style in segment.styles} for segment in segments
        ]
        for idx, segment in enumerate(segments):
            candidates: list[SegmentCrossref] = []
            start = 0
            if self.config.link_window_segments > 0:
                start = max(0, idx - self.config.link_window_segments)
            for prev_idx in range(start, idx):
                shared = style_sets[idx] & style_sets[prev_idx]
                if len(shared) < self.config.min_shared_styles:
                    continue
                candidates.append(
                    SegmentCrossref(
                        from_index=segment.index,
                        to_index=segments[prev_idx].index,
                        shared_styles=sorted(shared),
                        shared_count=len(shared),
                    )
                )
            candidates.sort(key=lambda c: (-c.shared_count, c.to_index))
            crossrefs.extend(candidates[: self.config.max_crossrefs_per_segment])
            if len(crossrefs) >= self.config.max_crossrefs_total:
                break
        return crossrefs[: self.config.max_crossrefs_total]

    def _load_definitions(self) -> tuple[list[SpeechStyleDefinition], list[str]]:
        definitions: list[SpeechStyleDefinition] = []
        warnings: list[str] = []
        seen: set[tuple[str, str]] = set()
        for src in self.config.style_sources:
            try:
                path = Path(src.path).expanduser()
                if not path.is_absolute():
                    path = (self.config.effective_workdir / path).resolve()
                if not path.exists():
                    warnings.append(f"Speech style definitions missing: {path}")
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
        self, path: Path, source: SpeechStyleSource
    ) -> list[SpeechStyleDefinition]:
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

        entries: list[SpeechStyleDefinition] = []
        for raw in raw_defs:
            if not isinstance(raw, dict):
                continue
            term = self._normalize_term(str(raw.get("term", "")))
            description = self._normalize_description(str(raw.get("description", "")))
            category = str(raw.get("category", source.category))
            if not term:
                continue
            entries.append(
                SpeechStyleDefinition(
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

    def _load_item(self, item: SpeechStyleItem) -> tuple[str, str | None, int | None]:
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
        if not isinstance(event.args, ContextSpeechStyleCrossrefArgs):
            return ToolCallDisplay(summary="context_speech_style_crossref")
        if not event.args.items:
            return ToolCallDisplay(summary="context_speech_style_crossref")
        first = event.args.items[0]
        return ToolCallDisplay(
            summary="context_speech_style_crossref",
            details={
                "item_count": len(event.args.items),
                "id": first.id,
                "name": first.name,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpeechStyleCrossrefResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=(
                f"Analyzed {event.result.item_count} item(s) with "
                f"{event.result.total_crossrefs} cross-references"
            ),
            warnings=event.result.warnings,
            details={
                "item_count": event.result.item_count,
                "total_crossrefs": event.result.total_crossrefs,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Cross-referencing speaking styles"
