from __future__ import annotations

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


def _snippet(text: str, start: int, length: int, radius: int = 60) -> str:
    begin = max(0, start - radius)
    end = min(len(text), start + length + radius)
    return text[begin:end].strip()


class SpeechStyleSource(BaseModel):
    path: str
    category: str = Field(default="Speech Styles")
    encoding: str = Field(default="utf-8")


class ContextSpeechStylesConfig(BaseToolConfig):
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
    max_matches_per_item: int = Field(default=12, description="Limit matches per item.")
    match_radius: int = Field(default=80, description="Radius for snippet window.")


class ContextSpeechStylesState(BaseToolState):
    pass


class SpeechStyleItem(BaseModel):
    id: str | None = Field(default=None)
    name: str | None = Field(default=None)
    content: str | None = Field(default=None)
    path: str | None = Field(default=None)
    source: str | None = Field(default=None)



class ContextSpeechStylesArgs(BaseModel):
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
    description: str
    occurrences: int
    snippet: str


class SpeechStyleInsight(BaseModel):
    index: int
    id: str | None
    name: str | None
    source_path: str | None
    preview: str
    matches: list[SpeechStyleMatch]
    total_matches: int


class ContextSpeechStylesResult(BaseModel):
    items: list[SpeechStyleInsight]
    item_count: int
    total_matches: int
    definitions_loaded: int
    truncated: bool
    warnings: list[str]
    errors: list[str]


class ContextSpeechStyles(
    BaseTool[
        ContextSpeechStylesArgs,
        ContextSpeechStylesResult,
        ContextSpeechStylesConfig,
        ContextSpeechStylesState,
    ],
    ToolUIData[ContextSpeechStylesArgs, ContextSpeechStylesResult],
):
    description: ClassVar[str] = (
        "Load speech emphasis styles and tags, then match them in text."
    )

    async def run(self, args: ContextSpeechStylesArgs) -> ContextSpeechStylesResult:
        items = list(args.items)
        if not items:
            raise ToolError("items is required.")

        definitions, def_warnings = self._load_definitions()
        if not definitions:
            raise ToolError("No speech style definitions available.")

        errors: list[str] = []
        warnings: list[str] = def_warnings.copy()
        insights: list[SpeechStyleInsight] = []
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
                    SpeechStyleInsight(
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

        return ContextSpeechStylesResult(
            items=insights,
            item_count=len(insights),
            total_matches=total_matches,
            definitions_loaded=len(definitions),
            truncated=truncated,
            warnings=warnings,
            errors=errors,
        )

    def _match_definitions(
        self, content: str, definitions: list[SpeechStyleDefinition]
    ) -> list[SpeechStyleMatch]:
        lower = content.lower()
        space_normalized = lower.replace("-", " ")
        matches: list[SpeechStyleMatch] = []
        for definition in definitions:
            key = definition.normalized_term
            if not key:
                continue
            idx = lower.find(key)
            if idx < 0:
                alt = key.replace("-", " ")
                idx = space_normalized.find(alt)
                if idx < 0:
                    continue
                snippet = _snippet(content, idx, len(definition.term), radius=self.config.match_radius)
                occurrences = space_normalized.count(alt)
            else:
                snippet = _snippet(content, idx, len(definition.term), radius=self.config.match_radius)
                occurrences = lower.count(key)
            matches.append(
                SpeechStyleMatch(
                    term=definition.term,
                    category=definition.category,
                    description=definition.description,
                    occurrences=occurrences,
                    snippet=snippet or definition.description,
                )
            )
        return matches

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

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextSpeechStylesArgs):
            return ToolCallDisplay(summary="context_speech_styles")
        if not event.args.items:
            return ToolCallDisplay(summary="context_speech_styles")
        first = event.args.items[0]
        return ToolCallDisplay(
            summary="context_speech_styles",
            details={
                "item_count": len(event.args.items),
                "id": first.id,
                "name": first.name,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpeechStylesResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=(
                f"Analyzed {event.result.item_count} item(s) with "
                f"{event.result.total_matches} speech style matches"
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
        return "Reading speech styles"
