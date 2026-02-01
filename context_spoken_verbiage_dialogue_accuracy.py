from __future__ import annotations

import re
from collections import Counter, defaultdict
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


class ContextSpokenVerbiageDialogueAccuracyConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    style_sources: list[SpeechStyleSource] = Field(
        default_factory=lambda: [
            SpeechStyleSource(
                path=r"C:\Users\Zack\.vibe\definitions\speech_styles_data.py",
                category="Speech Styles",
            )
        ],
        description="Sources that define verbiage styles and tags.",
    )
    max_items: int = Field(default=40, description="Maximum items to evaluate.")
    max_source_bytes: int = Field(default=3_000_000, description="Maximum bytes per item.")
    max_total_bytes: int = Field(default=20_000_000, description="Max bytes across items.")
    preview_chars: int = Field(default=400, description="Preview snippet length.")
    max_segments: int = Field(default=200, description="Maximum segments per item.")
    min_segment_chars: int = Field(default=20, description="Minimum segment length.")
    max_styles_per_segment: int = Field(default=6, description="Top styles per segment.")
    min_confidence: float = Field(
        default=0.2, description="Minimum confidence to treat a style as matched."
    )
    default_include_categories: list[str] = Field(
        default_factory=lambda: [
            "Emphasis Styles",
            "Styles of Emphasis Speaking",
            "Speaking Classification",
        ],
        description="Default categories used for verbiage accuracy scoring.",
    )
    default_exclude_categories: list[str] = Field(
        default_factory=list,
        description="Default categories to exclude.",
    )


class ContextSpokenVerbiageDialogueAccuracyState(BaseToolState):
    pass


class VerbiageDialogueItem(BaseModel):
    id: str | None = Field(default=None)
    name: str | None = Field(default=None)
    content: str | None = Field(default=None)
    path: str | None = Field(default=None)
    source: str | None = Field(default=None)
    expected_styles: list[str] | None = Field(
        default=None, description="Expected verbiage styles for the dialogue."
    )
    expected_style_map: dict[str, list[str]] | None = Field(
        default=None, description="Expected styles per speaker."
    )
    include_categories: list[str] | None = Field(default=None)
    exclude_categories: list[str] | None = Field(default=None)



class ContextSpokenVerbiageDialogueAccuracyArgs(BaseModel):
    items: list[VerbiageDialogueItem] = Field(description="Items to evaluate.")


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


class VerbiageSegment(BaseModel):
    index: int
    speaker: str | None
    text: str
    start: int
    end: int
    top_style: str | None
    top_confidence: float
    expected_styles: list[str]
    matched_expected: list[str]
    accuracy: float | None
    accuracy_label: str
    matches: list[StyleScore]


class SpeakerAccuracy(BaseModel):
    speaker: str
    segment_indices: list[int]
    expected_styles: list[str]
    matched_styles: list[str]
    average_accuracy: float


class VerbiageDialogueInsight(BaseModel):
    index: int
    id: str | None
    name: str | None
    source_path: str | None
    preview: str
    segments: list[VerbiageSegment]
    speaker_accuracy: list[SpeakerAccuracy]
    dominant_styles: list[str]
    segment_count: int
    average_accuracy: float | None


class ContextSpokenVerbiageDialogueAccuracyResult(BaseModel):
    items: list[VerbiageDialogueInsight]
    item_count: int
    truncated: bool
    warnings: list[str]
    errors: list[str]


class ContextSpokenVerbiageDialogueAccuracy(
    BaseTool[
        ContextSpokenVerbiageDialogueAccuracyArgs,
        ContextSpokenVerbiageDialogueAccuracyResult,
        ContextSpokenVerbiageDialogueAccuracyConfig,
        ContextSpokenVerbiageDialogueAccuracyState,
    ],
    ToolUIData[
        ContextSpokenVerbiageDialogueAccuracyArgs, ContextSpokenVerbiageDialogueAccuracyResult
    ],
):
    description: ClassVar[str] = (
        "Recognize verbiage dialogue styles and score accuracy against expectations."
    )

    async def run(
        self, args: ContextSpokenVerbiageDialogueAccuracyArgs
    ) -> ContextSpokenVerbiageDialogueAccuracyResult:
        items = list(args.items)
        if not items:
            raise ToolError("items is required.")

        definitions, def_warnings = self._load_definitions()
        if not definitions:
            raise ToolError("No verbiage style definitions available.")

        errors: list[str] = []
        warnings: list[str] = def_warnings.copy()
        insights: list[VerbiageDialogueInsight] = []
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
                segments = self._score_segments(segments, definitions, item, warnings)
                speaker_accuracy = self._build_speaker_accuracy(segments, item)
                dominant_styles = self._dominant_styles(segments)
                average_accuracy = self._average_accuracy(segments)

                insights.append(
                    VerbiageDialogueInsight(
                        index=len(insights) + 1,
                        id=item.id,
                        name=item.name,
                        source_path=source_path,
                        preview=self._preview(content),
                        segments=segments[: self.config.max_segments],
                        speaker_accuracy=speaker_accuracy,
                        dominant_styles=dominant_styles,
                        segment_count=len(segments),
                        average_accuracy=average_accuracy,
                    )
                )
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not insights:
            raise ToolError("No insights generated.")

        return ContextSpokenVerbiageDialogueAccuracyResult(
            items=insights,
            item_count=len(insights),
            truncated=truncated,
            warnings=warnings,
            errors=errors,
        )

    def _segment_conversation(self, text: str) -> list[VerbiageSegment]:
        segments: list[VerbiageSegment] = []
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
                    VerbiageSegment(
                        index=len(segments) + 1,
                        speaker=current_speaker,
                        text=seg_text,
                        start=seg_start,
                        end=end_offset,
                        top_style=None,
                        top_confidence=0.0,
                        expected_styles=[],
                        matched_expected=[],
                        accuracy=None,
                        accuracy_label="no_expected",
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
        segments: list[VerbiageSegment],
        definitions: list[SpeechStyleDefinition],
        item: VerbiageDialogueItem,
        warnings: list[str],
    ) -> list[VerbiageSegment]:
        include = item.include_categories or self.config.default_include_categories
        exclude = item.exclude_categories or self.config.default_exclude_categories
        include_lower = {cat.lower() for cat in include} if include else set()
        exclude_lower = {cat.lower() for cat in exclude} if exclude else set()

        expected_map = {
            (speaker or "").lower(): styles
            for speaker, styles in (item.expected_style_map or {}).items()
        }
        expected_global = item.expected_styles or []

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

            expected = expected_global
            speaker_key = (segment.speaker or "").lower()
            if speaker_key in expected_map:
                expected = expected_map[speaker_key]
            expected_resolved, expected_warnings = self._resolve_expected_styles(
                expected, definitions
            )
            if expected_warnings:
                warnings.extend(expected_warnings)
            segment.expected_styles = expected_resolved

            matched_expected = [
                style
                for style in expected_resolved
                if any(
                    match.term == style
                    and match.confidence >= self.config.min_confidence
                    for match in segment.matches
                )
            ]
            segment.matched_expected = matched_expected

            if expected_resolved:
                segment.accuracy = round(
                    len(matched_expected) / len(expected_resolved), 3
                )
                if segment.accuracy >= 0.99:
                    segment.accuracy_label = "matched"
                elif segment.accuracy > 0.0:
                    segment.accuracy_label = "partial"
                else:
                    segment.accuracy_label = "missing"
            else:
                segment.accuracy = None
                segment.accuracy_label = "no_expected"

        return segments

    def _resolve_expected_styles(
        self,
        expected: list[str],
        definitions: list[SpeechStyleDefinition],
    ) -> tuple[list[str], list[str]]:
        warnings: list[str] = []
        if not expected:
            return [], warnings
        lookup = {definition.normalized_term: definition.term for definition in definitions}
        resolved: list[str] = []
        for name in expected:
            normalized = _normalize(name)
            term = lookup.get(normalized)
            if term:
                resolved.append(term)
                continue
            warnings.append(f"Unknown expected verbiage style: {name}")
        return resolved, warnings

    def _build_speaker_accuracy(
        self,
        segments: list[VerbiageSegment],
        item: VerbiageDialogueItem,
    ) -> list[SpeakerAccuracy]:
        speaker_map: dict[str, list[VerbiageSegment]] = defaultdict(list)
        for segment in segments:
            if segment.speaker:
                speaker_map[segment.speaker].append(segment)

        summaries: list[SpeakerAccuracy] = []
        for speaker, speaker_segments in speaker_map.items():
            accuracies = [seg.accuracy for seg in speaker_segments if seg.accuracy is not None]
            avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
            expected_styles = []
            if item.expected_style_map and speaker in item.expected_style_map:
                expected_styles = item.expected_style_map[speaker]
            elif item.expected_styles:
                expected_styles = item.expected_styles

            matched = []
            for seg in speaker_segments:
                matched.extend(seg.matched_expected)
            matched_sorted = sorted(set(matched))

            summaries.append(
                SpeakerAccuracy(
                    speaker=speaker,
                    segment_indices=[seg.index for seg in speaker_segments],
                    expected_styles=expected_styles,
                    matched_styles=matched_sorted,
                    average_accuracy=round(avg_accuracy, 3),
                )
            )
        return summaries

    def _dominant_styles(self, segments: list[VerbiageSegment]) -> list[str]:
        counts: Counter[str] = Counter()
        for segment in segments:
            if segment.top_style:
                counts[segment.top_style] += 1
        return [style for style, _ in counts.most_common(5)]

    def _average_accuracy(self, segments: list[VerbiageSegment]) -> float | None:
        values = [seg.accuracy for seg in segments if seg.accuracy is not None]
        if not values:
            return None
        return round(sum(values) / len(values), 3)

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

    def _load_item(
        self, item: VerbiageDialogueItem
    ) -> tuple[str, str | None, int | None]:
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
        if not isinstance(event.args, ContextSpokenVerbiageDialogueAccuracyArgs):
            return ToolCallDisplay(summary="context_spoken_verbiage_dialogue_accuracy")
        if not event.args.items:
            return ToolCallDisplay(summary="context_spoken_verbiage_dialogue_accuracy")
        first = event.args.items[0]
        return ToolCallDisplay(
            summary="context_spoken_verbiage_dialogue_accuracy",
            details={
                "item_count": len(event.args.items),
                "id": first.id,
                "name": first.name,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpokenVerbiageDialogueAccuracyResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=(
                f"Analyzed {event.result.item_count} item(s) for verbiage dialogue accuracy"
            ),
            warnings=event.result.warnings,
            details={
                "item_count": event.result.item_count,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Scoring verbiage dialogue accuracy"
