from __future__ import annotations

import re
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


SPEAKER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _.'-]{0,40}):\s*(.*)$")


def _normalize(text: str) -> str:
    lowered = text.lower().replace("-", " ")
    return " ".join(lowered.split())


class SpeechStyleSource(BaseModel):
    path: str
    category: str = Field(default="Speech Styles")
    encoding: str = Field(default="utf-8")


class ContextSpeakingStyleAccuracyConfig(BaseToolConfig):
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
    max_styles_per_segment: int = Field(default=5, description="Top styles per segment.")
    default_include_categories: list[str] = Field(
        default_factory=lambda: [
            "Emphasis Styles",
            "Styles of Emphasis Speaking",
            "Speaking Classification",
            "Vocal Types",
            "Vocal Development",
        ],
        description="Default categories used for accuracy scoring.",
    )
    default_exclude_categories: list[str] = Field(
        default_factory=list,
        description="Default categories to exclude.",
    )
    min_confidence: float = Field(
        default=0.2, description="Minimum confidence to treat a style as dominant."
    )


class ContextSpeakingStyleAccuracyState(BaseToolState):
    pass


class SpeakingStyleItem(BaseModel):
    id: str | None = Field(default=None)
    name: str | None = Field(default=None)
    content: str | None = Field(default=None)
    path: str | None = Field(default=None)
    source: str | None = Field(default=None)
    include_categories: list[str] | None = Field(default=None)
    exclude_categories: list[str] | None = Field(default=None)



class ContextSpeakingStyleAccuracyArgs(BaseModel):
    items: list[SpeakingStyleItem] = Field(description="Items to evaluate.")

class SpeechStyleDefinition(BaseModel):
    term: str
    normalized_term: str
    description: str
    category: str
    source_path: str


class StyleScore(BaseModel):
    term: str
    category: str
    occurrences: int
    score: float
    confidence: float


class StyleSegment(BaseModel):
    index: int
    speaker: str | None
    text: str
    start: int
    end: int
    top_style: str | None
    top_confidence: float
    matches: list[StyleScore]


class StyleTransition(BaseModel):
    from_index: int
    to_index: int
    from_style: str | None
    to_style: str | None
    change_type: str


class StyleSummary(BaseModel):
    term: str
    category: str
    segment_indices: list[int]
    count: int
    avg_confidence: float


class SpeakingStyleAccuracyInsight(BaseModel):
    index: int
    id: str | None
    name: str | None
    source_path: str | None
    preview: str
    segments: list[StyleSegment]
    transitions: list[StyleTransition]
    summary: list[StyleSummary]
    dominant_style: str | None
    dominant_coverage: float
    segment_count: int


class ContextSpeakingStyleAccuracyResult(BaseModel):
    items: list[SpeakingStyleAccuracyInsight]
    item_count: int
    truncated: bool
    warnings: list[str]
    errors: list[str]


class ContextSpeakingStyleAccuracy(
    BaseTool[
        ContextSpeakingStyleAccuracyArgs,
        ContextSpeakingStyleAccuracyResult,
        ContextSpeakingStyleAccuracyConfig,
        ContextSpeakingStyleAccuracyState,
    ],
    ToolUIData[ContextSpeakingStyleAccuracyArgs, ContextSpeakingStyleAccuracyResult],
):
    description: ClassVar[str] = (
        "Correlate the most accurate speaking style per segment in a conversation."
    )

    async def run(self, args: ContextSpeakingStyleAccuracyArgs) -> ContextSpeakingStyleAccuracyResult:
        items = list(args.items)
        if not items:
            raise ToolError("items is required.")

        definitions, def_warnings = self._load_definitions()
        if not definitions:
            raise ToolError("No speech style definitions available.")

        errors: list[str] = []
        warnings: list[str] = def_warnings.copy()
        insights: list[SpeakingStyleAccuracyInsight] = []
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

                segments = self._segment_conversation(content)
                segment_scores = self._score_segments(segments, definitions, item)
                transitions = self._build_transitions(segment_scores)
                summary, dominant_style, coverage = self._build_summary(segment_scores)

                insights.append(
                    SpeakingStyleAccuracyInsight(
                        index=len(insights) + 1,
                        id=item.id,
                        name=item.name,
                        source_path=source_path,
                        preview=self._preview(content),
                        segments=segment_scores[: self.config.max_segments],
                        transitions=transitions,
                        summary=summary,
                        dominant_style=dominant_style,
                        dominant_coverage=coverage,
                        segment_count=len(segment_scores),
                    )
                )
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not insights:
            raise ToolError("No insights generated.")

        return ContextSpeakingStyleAccuracyResult(
            items=insights,
            item_count=len(insights),
            truncated=truncated,
            warnings=warnings,
            errors=errors,
        )

    def _segment_conversation(self, text: str) -> list[StyleSegment]:
        segments: list[StyleSegment] = []
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
                    StyleSegment(
                        index=len(segments) + 1,
                        speaker=current_speaker,
                        text=seg_text,
                        start=seg_start,
                        end=end_offset,
                        top_style=None,
                        top_confidence=0.0,
                        matches=[],
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

    def _score_segments(
        self,
        segments: list[StyleSegment],
        definitions: list[SpeechStyleDefinition],
        item: SpeakingStyleItem,
    ) -> list[StyleSegment]:
        include = item.include_categories or self.config.default_include_categories
        exclude = item.exclude_categories or self.config.default_exclude_categories
        include_lower = {cat.lower() for cat in include} if include else set()
        exclude_lower = {cat.lower() for cat in exclude} if exclude else set()

        for segment in segments:
            segment_norm = _normalize(segment.text)
            match_scores: list[StyleScore] = []
            total_score = 0.0
            for definition in definitions:
                category_lower = definition.category.lower()
                if include_lower and category_lower not in include_lower:
                    continue
                if exclude_lower and category_lower in exclude_lower:
                    continue
                term_norm = definition.normalized_term
                if not term_norm:
                    continue
                occurrences = segment_norm.count(term_norm)
                if occurrences <= 0:
                    continue
                token_len = max(1, len(term_norm.split()))
                weight = min(3.0, 0.6 + 0.25 * token_len)
                score = occurrences * weight
                total_score += score
                match_scores.append(
                    StyleScore(
                        term=definition.term,
                        category=definition.category,
                        occurrences=occurrences,
                        score=score,
                        confidence=0.0,
                    )
                )
            match_scores.sort(key=lambda m: m.score, reverse=True)
            if total_score > 0:
                for match in match_scores:
                    match.confidence = round(match.score / total_score, 3)

            segment.matches = match_scores[: self.config.max_styles_per_segment]
            if segment.matches and segment.matches[0].confidence >= self.config.min_confidence:
                segment.top_style = segment.matches[0].term
                segment.top_confidence = segment.matches[0].confidence
            else:
                segment.top_style = None
                segment.top_confidence = 0.0
        return segments

    def _build_transitions(self, segments: list[StyleSegment]) -> list[StyleTransition]:
        transitions: list[StyleTransition] = []
        for idx in range(1, len(segments)):
            prev = segments[idx - 1]
            curr = segments[idx]
            if prev.top_style == curr.top_style:
                change_type = "stable"
            elif prev.top_style is None or curr.top_style is None:
                change_type = "uncertain"
            else:
                change_type = "style_change"
            transitions.append(
                StyleTransition(
                    from_index=prev.index,
                    to_index=curr.index,
                    from_style=prev.top_style,
                    to_style=curr.top_style,
                    change_type=change_type,
                )
            )
        return transitions

    def _build_summary(
        self, segments: list[StyleSegment]
    ) -> tuple[list[StyleSummary], str | None, float]:
        summary_map: dict[tuple[str, str], list[tuple[int, float]]] = {}
        for segment in segments:
            if segment.top_style is None:
                continue
            if not segment.matches:
                continue
            top_match = segment.matches[0]
            key = (top_match.term, top_match.category)
            summary_map.setdefault(key, []).append((segment.index, top_match.confidence))

        summary: list[StyleSummary] = []
        dominant_style = None
        dominant_count = 0
        for (term, category), entries in summary_map.items():
            indices = [entry[0] for entry in entries]
            avg_conf = sum(entry[1] for entry in entries) / len(entries)
            summary.append(
                StyleSummary(
                    term=term,
                    category=category,
                    segment_indices=indices,
                    count=len(indices),
                    avg_confidence=round(avg_conf, 3),
                )
            )
            if len(indices) > dominant_count:
                dominant_count = len(indices)
                dominant_style = term

        summary.sort(key=lambda entry: (-entry.count, entry.term))
        coverage = (dominant_count / len(segments)) if segments else 0.0
        return summary, dominant_style, round(coverage, 3)

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
                    normalized_term=_normalize(term),
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

    def _load_item(self, item: SpeakingStyleItem) -> tuple[str, str | None, int | None]:
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
        if not isinstance(event.args, ContextSpeakingStyleAccuracyArgs):
            return ToolCallDisplay(summary="context_speaking_style_accuracy")
        if not event.args.items:
            return ToolCallDisplay(summary="context_speaking_style_accuracy")
        first = event.args.items[0]
        return ToolCallDisplay(
            summary="context_speaking_style_accuracy",
            details={
                "item_count": len(event.args.items),
                "id": first.id,
                "name": first.name,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpeakingStyleAccuracyResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=(
                f"Analyzed {event.result.item_count} item(s) with "
                f"{len(event.result.items[0].segments) if event.result.items else 0} segments"
            ),
            warnings=event.result.warnings,
            details={
                "item_count": event.result.item_count,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Scoring speaking style accuracy"
