from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import ClassVar, Literal, TYPE_CHECKING

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


DEFINE_EQUAL_PATTERN = re.compile(r"([A-Za-z][A-Za-z0-9_ ]*?)\s*=\s*\"(.*?)\"", re.DOTALL)


def _default_definition_sources() -> list["DefinitionSource"]:
    return [
        DefinitionSource(
            path=r"C:\Users\Zachary Confer-Bair\Documents\Obsidian\EMOTIONS, MOODS AND FEELINGS\MY CUSTOM DEFINED POSITIVE EMOTIONS.md",
            parser="equals",
            category="Positive Emotions",
        ),
        DefinitionSource(
            path=r"C:\Users\Zachary Confer-Bair\Documents\Obsidian\EMOTIONS, MOODS AND FEELINGS\MY CUSTOM DEFINED MOODS.md",
            parser="equals",
            category="Moods",
        ),
        DefinitionSource(
            path=r"C:\Users\Zachary Confer-Bair\Documents\Obsidian\EMOTIONS, MOODS AND FEELINGS\MY CUSTOM DEFINED FEELINGS.md",
            parser="equals",
            category="Feelings",
        ),
        DefinitionSource(
            path=r"C:\Users\Zachary Confer-Bair\Documents\Obsidian\PSYCHOLOGY & of Play + Cognitive Science\Human Natures.md",
            parser="colon",
            category="Human Natures",
        ),
        DefinitionSource(
            path=r"C:\Users\Zachary Confer-Bair\Documents\Obsidian\Zacks Morals\ASUNAAI PREREQUISITES.md",
            parser="line",
            category="ASUNAAI Prerequisites",
        ),
    ]


@dataclass
class _MatchSnippet:
    start: int
    length: int
    text: str


def _snippet(text: str, start: int, length: int, radius: int = 60) -> str:
    begin = max(0, start - radius)
    end = min(len(text), start + length + radius)
    return text[begin:end].strip()


class DefinitionSource(BaseModel):
    path: str
    parser: Literal["equals", "colon", "line", "python"] = Field(default="line")
    category: str | None = Field(default=None)
    encoding: str = Field(default="utf-8")


class ContextEmotionProfilesConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    definition_sources: list[DefinitionSource] = Field(
        default_factory=_default_definition_sources,
        description="Sources that define custom emotions, moods, feelings, and principles.",
    )
    max_items: int = Field(default=40, description="Maximum items to evaluate.")
    max_source_bytes: int = Field(default=3_000_000, description="Maximum bytes per item.")
    max_total_bytes: int = Field(default=20_000_000, description="Max bytes across items.")
    preview_chars: int = Field(default=400, description="Preview snippet length.")
    max_matches_per_item: int = Field(default=6, description="Limit matches per item.")
    max_snippets: int = Field(default=1, description="Snippets per match.")
    match_radius: int = Field(default=80, description="Radius for snippet window.")


class ContextEmotionProfilesState(BaseToolState):
    pass


class EmotionDefinition(BaseModel):
    term: str
    normalized_term: str
    description: str
    category: str | None
    source_path: str


class EmotionMatch(BaseModel):
    term: str
    category: str | None
    description: str
    occurrences: int
    snippet: str


class EmotionItem(BaseModel):
    id: str | None = Field(default=None)
    name: str | None = Field(default=None)
    content: str | None = Field(default=None)
    path: str | None = Field(default=None)
    source: str | None = Field(default=None)



class ContextEmotionProfilesArgs(BaseModel):
    items: list[EmotionItem] = Field(description="Items to evaluate.")

class ItemEmotionInsight(BaseModel):
    index: int
    id: str | None
    name: str | None
    source_path: str | None
    preview: str
    matches: list[EmotionMatch]
    total_matches: int


class ContextEmotionProfilesResult(BaseModel):
    items: list[ItemEmotionInsight]
    item_count: int
    total_matches: int
    definitions_loaded: int
    truncated: bool
    warnings: list[str]
    errors: list[str]


class ContextEmotionProfiles(
    BaseTool[
        ContextEmotionProfilesArgs,
        ContextEmotionProfilesResult,
        ContextEmotionProfilesConfig,
        ContextEmotionProfilesState,
    ],
    ToolUIData[ContextEmotionProfilesArgs, ContextEmotionProfilesResult],
):
    description: ClassVar[str] = (
        "Load your custom emotion/mood/feeling definitions and score text for matches."
    )

    async def run(self, args: ContextEmotionProfilesArgs) -> ContextEmotionProfilesResult:
        items = list(args.items)
        if not items:
            raise ToolError("items is required.")

        definitions, def_warnings = self._load_definitions()
        if not definitions:
            raise ToolError("No definitions available.")

        errors: list[str] = []
        warnings: list[str] = def_warnings.copy()
        insights: list[ItemEmotionInsight] = []
        total_bytes = 0
        truncated = False
        total_matches = 0

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

                matches = self._match_definitions(content, definitions)
                total_matches += len(matches)
                insights.append(
                    ItemEmotionInsight(
                        index=len(insights) + 1,
                        id=item.id,
                        name=item.name,
                        source_path=source_path,
                        preview=self._preview(content),
                        matches=matches[: self.config.max_matches_per_item],
                        total_matches=len(matches),
                    )
                )
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not insights:
            raise ToolError("No insights generated.")

        return ContextEmotionProfilesResult(
            items=insights,
            item_count=len(insights),
            total_matches=total_matches,
            definitions_loaded=len(definitions),
            truncated=truncated,
            warnings=warnings,
            errors=errors,
        )

    def _match_definitions(
        self, content: str, definitions: list[EmotionDefinition]
    ) -> list[EmotionMatch]:
        lower = content.lower()
        matches: list[EmotionMatch] = []
        for definition in definitions:
            key = definition.normalized_term
            if not key:
                continue
            idx = lower.find(key)
            if idx < 0:
                continue
            occurrences = lower.count(key)
            snippet = _snippet(content, idx, len(definition.term), radius=self.config.match_radius)
            matches.append(
                EmotionMatch(
                    term=definition.term,
                    category=definition.category,
                    description=definition.description,
                    occurrences=occurrences,
                    snippet=snippet if snippet else definition.description,
                )
            )
        return matches

    def _load_item(self, item: EmotionItem) -> tuple[str, str | None, int | None]:
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

    def _load_definitions(self) -> tuple[list[EmotionDefinition], list[str]]:
        definitions: list[EmotionDefinition] = []
        warnings: list[str] = []
        seen: set[tuple[str, str | None]] = set()
        for src in self.config.definition_sources:
            try:
                path = Path(src.path).expanduser()
                if not path.is_absolute():
                    path = (self.config.effective_workdir / path).resolve()
                if not path.exists():
                    warnings.append(f"Definition file missing: {path}")
                    continue
                text = path.read_text(encoding=src.encoding, errors="ignore")
                parsed = self._parse_definitions(text, path, src)
                for item in parsed:
                    key = (item.normalized_term, item.category)
                    if key in seen:
                        continue
                    seen.add(key)
                    definitions.append(item)
            except Exception as exc:
                warnings.append(f"failed to read {src.path}: {exc}")
        return definitions, warnings

    def _parse_definitions(
        self, text: str, source_path: Path, source: DefinitionSource
    ) -> list[EmotionDefinition]:
        entries: list[EmotionDefinition] = []
        if source.parser == "python":
            entries = self._parse_python_definitions(source_path, source)
        elif source.parser == "equals":
            entries = self._parse_equals(text, source, source_path)
        elif source.parser == "colon":
            entries = self._parse_colon(text, source, source_path)
        else:
            entries = self._parse_line(text, source, source_path)
        return entries

    def _parse_python_definitions(
        self, path: Path, source: DefinitionSource
    ) -> list[EmotionDefinition]:
        try:
            import importlib.util
        except Exception as exc:
            raise ToolError(f"Unable to import definitions module: {exc}") from exc

        spec = importlib.util.spec_from_file_location("emotion_profiles_data", path)
        if spec is None or spec.loader is None:
            raise ToolError(f"Unable to load definitions module: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        raw_defs = getattr(module, "DEFINITIONS", None)
        if not isinstance(raw_defs, list):
            raise ToolError(f"DEFINITIONS missing or invalid in: {path}")

        entries: list[EmotionDefinition] = []
        for raw in raw_defs:
            if not isinstance(raw, dict):
                continue
            term = self._normalize_term(str(raw.get("term", "")))
            description = self._normalize_description(str(raw.get("description", "")))
            category = raw.get("category") or source.category
            if not term:
                continue
            entries.append(
                EmotionDefinition(
                    term=term,
                    normalized_term=term.lower(),
                    description=description or term,
                    category=category,
                    source_path=str(path),
                )
            )
        return entries

    def _parse_equals(
        self, text: str, source: DefinitionSource, path: Path
    ) -> list[EmotionDefinition]:
        entries: list[EmotionDefinition] = []
        for match in DEFINE_EQUAL_PATTERN.finditer(text):
            term = self._normalize_term(match.group(1))
            description = self._normalize_description(match.group(2))
            if not term:
                continue
            entries.append(
                EmotionDefinition(
                    term=term,
                    normalized_term=term.lower(),
                    description=description,
                    category=source.category,
                    source_path=str(path),
                )
            )
        return entries

    def _parse_colon(
        self, text: str, source: DefinitionSource, path: Path
    ) -> list[EmotionDefinition]:
        entries: list[EmotionDefinition] = []
        for line in text.splitlines():
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            term = self._normalize_term(parts[0])
            description = self._normalize_description(parts[1])
            if not term:
                continue
            entries.append(
                EmotionDefinition(
                    term=term,
                    normalized_term=term.lower(),
                    description=description,
                    category=source.category,
                    source_path=str(path),
                )
            )
        return entries

    def _parse_line(
        self, text: str, source: DefinitionSource, path: Path
    ) -> list[EmotionDefinition]:
        entries: list[EmotionDefinition] = []
        for line in text.splitlines():
            clean = line.strip()
            if not clean:
                continue
            norm = self._normalize_term(clean)
            if not norm:
                continue
            entries.append(
                EmotionDefinition(
                    term=clean,
                    normalized_term=norm.lower(),
                    description=norm,
                    category=source.category,
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
        if not isinstance(event.args, ContextEmotionProfilesArgs):
            return ToolCallDisplay(summary="context_emotion_profiles")
        if not event.args.items:
            return ToolCallDisplay(summary="context_emotion_profiles")
        first = event.args.items[0]
        return ToolCallDisplay(
            summary="context_emotion_profiles",
            details={
                "item_count": len(event.args.items),
                "id": first.id,
                "name": first.name,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextEmotionProfilesResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=(
                f"Analyzed {event.result.item_count} item(s) with "
                f"{event.result.total_matches} emotion matches"
            ),
            warnings=event.result.warnings,
            details={
                "item_count": event.result.item_count,
                "total_matches": event.result.total_matches,
                "definitions_loaded": event.result.definitions_loaded,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Reading emotional definitions"
