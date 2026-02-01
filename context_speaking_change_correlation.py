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


WORD_RE = re.compile(r"[A-Za-z0-9_]+")
SPEAKER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _.'-]{0,40}):\s*(.*)$")


class SpeechStyleSource(BaseModel):
    path: str
    category: str = Field(default="Speech Styles")
    encoding: str = Field(default="utf-8")


class ContextSpeakingChangeCorrelationConfig(BaseToolConfig):
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
    min_style_change: int = Field(default=1, description="Minimum style changes to flag.")
    token_similarity_threshold: float = Field(
        default=0.25, description="Token similarity threshold for shifts."
    )
    style_similarity_threshold: float = Field(
        default=0.2, description="Style similarity threshold for shifts."
    )
    max_events: int = Field(default=200, description="Maximum change events per item.")
    max_shared_styles: int = Field(default=8, description="Maximum shared styles to return.")


class ContextSpeakingChangeCorrelationState(BaseToolState):
    pass


class SpeakingChangeItem(BaseModel):
    id: str | None = Field(default=None)
    name: str | None = Field(default=None)
    content: str | None = Field(default=None)
    path: str | None = Field(default=None)
    source: str | None = Field(default=None)



class ContextSpeakingChangeCorrelationArgs(BaseModel):
    items: list[SpeakingChangeItem] = Field(description="Items to evaluate.")

class SpeechStyleDefinition(BaseModel):
    term: str
    normalized_term: str
    description: str
    category: str
    source_path: str


class SpeakingSegment(BaseModel):
    index: int
    speaker: str | None
    text: str
    start: int
    end: int
    token_count: int
    styles: list[str]


class SpeakingChangeEvent(BaseModel):
    index: int
    from_index: int
    to_index: int
    speaker_from: str | None
    speaker_to: str | None
    style_added: list[str]
    style_removed: list[str]
    shared_styles: list[str]
    style_similarity: float
    token_similarity: float
    change_score: float
    change_types: list[str]


class SpeakingChangeInsight(BaseModel):
    index: int
    id: str | None
    name: str | None
    source_path: str | None
    preview: str
    segments: list[SpeakingSegment]
    events: list[SpeakingChangeEvent]
    segment_count: int
    event_count: int


class ContextSpeakingChangeCorrelationResult(BaseModel):
    items: list[SpeakingChangeInsight]
    item_count: int
    total_events: int
    truncated: bool
    warnings: list[str]
    errors: list[str]


class ContextSpeakingChangeCorrelation(
    BaseTool[
        ContextSpeakingChangeCorrelationArgs,
        ContextSpeakingChangeCorrelationResult,
        ContextSpeakingChangeCorrelationConfig,
        ContextSpeakingChangeCorrelationState,
    ],
    ToolUIData[ContextSpeakingChangeCorrelationArgs, ContextSpeakingChangeCorrelationResult],
):
    description: ClassVar[str] = (
        "Correlate speaking changes across spoken words in a conversation."
    )

    async def run(self, args: ContextSpeakingChangeCorrelationArgs) -> ContextSpeakingChangeCorrelationResult:
        items = list(args.items)
        if not items:
            raise ToolError("items is required.")

        definitions, def_warnings = self._load_definitions()
        if not definitions:
            raise ToolError("No speech style definitions available.")

        errors: list[str] = []
        warnings: list[str] = def_warnings.copy()
        insights: list[SpeakingChangeInsight] = []
        total_bytes = 0
        truncated = False
        total_events = 0

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
                events = self._detect_changes(segments)
                total_events += len(events)

                insights.append(
                    SpeakingChangeInsight(
                        index=len(insights) + 1,
                        id=item.id,
                        name=item.name,
                        source_path=source_path,
                        preview=self._preview(content),
                        segments=segments[: self.config.max_segments],
                        events=events[: self.config.max_events],
                        segment_count=len(segments),
                        event_count=len(events),
                    )
                )
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not insights:
            raise ToolError("No insights generated.")

        return ContextSpeakingChangeCorrelationResult(
            items=insights,
            item_count=len(insights),
            total_events=total_events,
            truncated=truncated,
            warnings=warnings,
            errors=errors,
        )

    def _segment_conversation(self, text: str) -> list[SpeakingSegment]:
        segments: list[SpeakingSegment] = []
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
                segments[-1].token_count = self._token_count(segments[-1].text)
            else:
                segments.append(
                    SpeakingSegment(
                        index=len(segments) + 1,
                        speaker=current_speaker,
                        text=seg_text,
                        start=seg_start,
                        end=end_offset,
                        token_count=self._token_count(seg_text),
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
        self, segments: list[SpeakingSegment], definitions: list[SpeechStyleDefinition]
    ) -> list[SpeakingSegment]:
        for segment in segments:
            styles = self._match_styles(segment.text, definitions)
            segment.styles = styles[: self.config.max_styles_per_segment]
        return segments

    def _match_styles(
        self, text: str, definitions: list[SpeechStyleDefinition]
    ) -> list[str]:
        lower = text.lower()
        alt_text = lower.replace("-", " ")
        matches: set[str] = set()
        for definition in definitions:
            key = definition.normalized_term
            if not key:
                continue
            if key in lower or key.replace("-", " ") in alt_text or key.replace(" ", "-") in lower:
                matches.add(definition.term)
        return sorted(matches)

    def _detect_changes(self, segments: list[SpeakingSegment]) -> list[SpeakingChangeEvent]:
        events: list[SpeakingChangeEvent] = []
        token_sets = [self._token_set(seg.text) for seg in segments]
        style_sets = [set(seg.styles) for seg in segments]
        for idx in range(1, len(segments)):
            prev = segments[idx - 1]
            curr = segments[idx]
            added = sorted(style_sets[idx] - style_sets[idx - 1])
            removed = sorted(style_sets[idx - 1] - style_sets[idx])
            shared = sorted(style_sets[idx] & style_sets[idx - 1])[: self.config.max_shared_styles]
            style_similarity = self._jaccard(style_sets[idx - 1], style_sets[idx])
            token_similarity = self._jaccard(token_sets[idx - 1], token_sets[idx])
            change_types: list[str] = []
            change_score = 0.0

            if prev.speaker and curr.speaker and prev.speaker != curr.speaker:
                change_types.append("speaker_change")
                change_score += 0.2

            if len(added) + len(removed) >= self.config.min_style_change:
                change_types.append("style_change")
                change_score += 0.5

            if token_similarity < self.config.token_similarity_threshold:
                change_types.append("topic_shift")
                change_score += 0.2

            if style_similarity < self.config.style_similarity_threshold:
                change_types.append("style_shift")
                change_score += 0.1

            if change_types:
                events.append(
                    SpeakingChangeEvent(
                        index=len(events) + 1,
                        from_index=prev.index,
                        to_index=curr.index,
                        speaker_from=prev.speaker,
                        speaker_to=curr.speaker,
                        style_added=added,
                        style_removed=removed,
                        shared_styles=shared,
                        style_similarity=round(style_similarity, 3),
                        token_similarity=round(token_similarity, 3),
                        change_score=round(min(change_score, 1.0), 3),
                        change_types=change_types,
                    )
                )
        return events[: self.config.max_events]

    def _jaccard(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)

    def _token_set(self, text: str) -> set[str]:
        return {
            token.lower()
            for token in WORD_RE.findall(text)
            if len(token) >= 3
        }

    def _token_count(self, text: str) -> int:
        return len(WORD_RE.findall(text))

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

    def _load_item(self, item: SpeakingChangeItem) -> tuple[str, str | None, int | None]:
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
        if not isinstance(event.args, ContextSpeakingChangeCorrelationArgs):
            return ToolCallDisplay(summary="context_speaking_change_correlation")
        if not event.args.items:
            return ToolCallDisplay(summary="context_speaking_change_correlation")
        first = event.args.items[0]
        return ToolCallDisplay(
            summary="context_speaking_change_correlation",
            details={
                "item_count": len(event.args.items),
                "id": first.id,
                "name": first.name,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpeakingChangeCorrelationResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=(
                f"Analyzed {event.result.item_count} item(s) with "
                f"{event.result.total_events} speaking change event(s)"
            ),
            warnings=event.result.warnings,
            details={
                "item_count": event.result.item_count,
                "total_events": event.result.total_events,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Correlating speaking changes"
