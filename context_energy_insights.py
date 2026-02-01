
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
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


WORD_RE = re.compile(r"[A-Za-z0-9_]+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

CATEGORY_KEYWORDS = {
    "natural_energy": [
        "natural energy",
        "sunlight",
        "seasonal winds",
        "tidal",
        "ocean",
        "earth heat",
        "grounded flow",
        "unmediated",
    ],
    "artificial_energy": [
        "artificial energy",
        "engineered",
        "power grid",
        "laboratory",
        "machine",
        "controlled loop",
        "refined fuel",
        "technology",
    ],
    "seasonal_energy": [
        "spring",
        "summer",
        "autumn",
        "winter",
        "equinox",
        "solstice",
        "seasonal energy",
        "long-period",
    ],
    "land_weather": [
        "mountain",
        "coastal",
        "desert",
        "plateau",
        "local wind",
        "rainfall",
        "geography",
        "weather pattern",
    ],
    "artificial_life_ceremony": [
        "church",
        "ceremony",
        "ritual",
        "communal",
        "harmon",
        "reverence",
        "tradition",
    ],
    "artificial_life_medical": [
        "hospital",
        "incubator",
        "oxygen",
        "monitored",
        "clinical",
        "instrument",
        "medical",
        "sterile",
    ],
    "willed_energy": [
        "consent",
        "reciprocity",
        "harmony",
        "steady",
        "cooperative",
        "mutual",
        "agreement",
    ],
    "forced_energy": [
        "dominance",
        "pressure",
        "resistance",
        "imbalance",
        "imposition",
        "compressed",
        "strain",
    ],
}

ENERGY_TYPE_KEYWORDS = {
    "borrowed": ["borrowed", "transient", "lending", "loan"],
    "gathered": ["gathered", "inwards", "funnel", "collect"],
    "earned": ["earned", "effort", "work", "success", "achievement"],
    "obtained": ["obtained", "from source", "specific source", "taken from"],
    "economical": ["economical", "environment-based", "thrifty", "balance"],
    "stored": ["stored", "reservoir", "reserve", "past accumulation"],
    "required": ["required", "necessary", "essential", "foundational"],
}

CATEGORY_LABELS = {
    "natural_energy": "Natural Energy",
    "artificial_energy": "Artificial Energy",
    "seasonal_energy": "Seasonal Energy",
    "land_weather": "Land & Weather",
    "artificial_life_ceremony": "Artificial Life (Ceremony)",
    "artificial_life_medical": "Artificial Life (Medical)",
    "willed_energy": "Willed Energy",
    "forced_energy": "Forced Energy",
}

ENERGY_TYPE_LABELS = {
    "borrowed": "Borrowed Energy",
    "gathered": "Gathered Energy",
    "earned": "Earned Energy",
    "obtained": "Obtained Energy",
    "economical": "Economical Energy",
    "stored": "Stored Energy",
    "required": "Required Energy",
}


@dataclass
class _CategoryMatch:
    label: str
    keywords: list[str]


CATEGORY_MATCHES = [
    _CategoryMatch(label=CATEGORY_LABELS[key], keywords=keywords)
    for key, keywords in CATEGORY_KEYWORDS.items()
]

TYPE_MATCHES = [
    _CategoryMatch(label=ENERGY_TYPE_LABELS[key], keywords=keywords)
    for key, keywords in ENERGY_TYPE_KEYWORDS.items()
]


class ContextEnergyInsightsConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_items: int = Field(default=50, description="Maximum items to scan.")
    max_source_bytes: int = Field(
        default=3_000_000, description="Maximum bytes per document."
    )
    max_total_bytes: int = Field(
        default=10_000_000, description="Maximum total bytes across inputs."
    )
    preview_chars: int = Field(default=400, description="Preview snippet length.")
    max_snippets: int = Field(
        default=3, description="Max snippets stored per category.")

class ContextEnergyInsightsState(BaseToolState):
    pass


class EnergyItem(BaseModel):
    id: str | None = Field(default=None, description="Optional item id.")
    name: str | None = Field(default=None, description="Optional name.")
    content: str | None = Field(default=None, description="Inline content.")
    path: str | None = Field(default=None, description="File path to read.")
    source: str | None = Field(default=None, description="Source descriptor.")


class ContextEnergyInsightsArgs(BaseModel):
    items: list[EnergyItem] = Field(description="Document items to analyze.")


class EnergyCategoryMatch(BaseModel):
    label: str
    active: bool
    matched_terms: list[str]
    snippets: list[str]


class EnergyInsight(BaseModel):
    index: int
    id: str | None
    name: str | None
    source_path: str | None
    source: str | None
    natural_energy: EnergyCategoryMatch
    artificial_energy: EnergyCategoryMatch
    seasonal_energy: EnergyCategoryMatch
    land_weather: EnergyCategoryMatch
    artificial_life_ceremony: EnergyCategoryMatch
    artificial_life_medical: EnergyCategoryMatch
    willed_energy: EnergyCategoryMatch
    forced_energy: EnergyCategoryMatch
    energy_types: dict[str, EnergyCategoryMatch]
    preview: str
    truncated: bool


class ContextEnergyInsightsResult(BaseModel):
    items: list[EnergyInsight]
    item_count: int
    truncated: bool
    errors: list[str]
    warnings: list[str]


class ContextEnergyInsights(
    BaseTool[
        ContextEnergyInsightsArgs,
        ContextEnergyInsightsResult,
        ContextEnergyInsightsConfig,
        ContextEnergyInsightsState,
    ],
    ToolUIData[ContextEnergyInsightsArgs, ContextEnergyInsightsResult],
):
    description: ClassVar[str] = (
        "Tag natural vs artificial energies, seasons, and life energies from documents."
    )

    async def run(
        self, args: ContextEnergyInsightsArgs
    ) -> ContextEnergyInsightsResult:
        if not args.items:
            raise ToolError("items is required.")

        if len(args.items) > self.config.max_items:
            raise ToolError(
                f"items exceeds max_items ({len(args.items)} > {self.config.max_items})."
            )

        errors: list[str] = []
        warnings: list[str] = []
        insights: list[EnergyInsight] = []
        total_bytes = 0
        truncated = False

        for idx, item in enumerate(args.items, start=1):
            try:
                content, source_path, size_bytes = self._load_item(item)
                if size_bytes is None:
                    raise ToolError("Item has no content.")
                if total_bytes + size_bytes > self.config.max_total_bytes:
                    truncated = True
                    warnings.append("Total byte budget exceeded; stopping early.")
                    break
                total_bytes += size_bytes

                matches = self._classify_content(content)
                insights.append(
                    EnergyInsight(
                        index=len(insights) + 1,
                        id=item.id,
                        name=item.name,
                        source_path=source_path,
                        source=item.source,
                        natural_energy=matches["natural_energy"],
                        artificial_energy=matches["artificial_energy"],
                        seasonal_energy=matches["seasonal_energy"],
                        land_weather=matches["land_weather"],
                        artificial_life_ceremony=matches["artificial_life_ceremony"],
                        artificial_life_medical=matches["artificial_life_medical"],
                        willed_energy=matches["willed_energy"],
                        forced_energy=matches["forced_energy"],
                        energy_types={
                            key: matches[key] for key in ENERGY_TYPE_KEYWORDS.keys()
                        },
                        preview=self._preview_text(content),
                        truncated=False,
                    )
                )
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not insights:
            raise ToolError("No valid insights were produced.")

        return ContextEnergyInsightsResult(
            items=insights,
            item_count=len(insights),
            truncated=truncated,
            errors=errors,
            warnings=warnings,
        )

    def _load_item(
        self, item: EnergyItem
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

    def _preview_text(self, text: str) -> str:
        max_chars = self.config.preview_chars
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def _classify_content(self, content: str) -> dict[str, EnergyCategoryMatch]:
        text_lower = content.lower()
        matches: dict[str, EnergyCategoryMatch] = {}
        for key, keywords in CATEGORY_KEYWORDS.items():
            label = CATEGORY_LABELS.get(key, key.replace("_", " ").title())
            matches[key] = self._analyze_category(content, text_lower, keywords, label)
        for key, keywords in ENERGY_TYPE_KEYWORDS.items():
            label = ENERGY_TYPE_LABELS.get(key, key.replace("_", " ").title())
            matches[key] = self._analyze_category(content, text_lower, keywords, label)
        return matches

    def _analyze_category(
        self, content: str, lower: str, keywords: list[str], label: str
    ) -> EnergyCategoryMatch:
        matched_terms: list[str] = []
        snippets: list[str] = []
        for keyword in keywords:
            index = lower.find(keyword)
            if index < 0:
                continue
            matched_terms.append(keyword)
            snippet = self._snippet_text(content, index, len(keyword))
            if snippet and snippet not in snippets:
                snippets.append(snippet)
                if len(snippets) >= self.config.max_snippets:
                    break
        return EnergyCategoryMatch(
            label=label,
            active=bool(matched_terms),
            matched_terms=matched_terms,
            snippets=snippets,
        )

    def _snippet_text(self, content: str, index: int, length: int) -> str:
        if not content:
            return ""
        radius = 80
        start = max(0, index - radius)
        end = min(len(content), index + length + radius)
        snippet = content[start:end].strip()
        if len(snippet) > self.config.preview_chars:
            return snippet[: self.config.preview_chars]
        return snippet

    @classmethod
    def get_call_display(cls, event: 'ToolCallEvent') -> ToolCallDisplay:
        if not hasattr(event, 'args') or not isinstance(event.args, ContextEnergyInsightsArgs):
            return ToolCallDisplay(summary="context_energy_insights")
        return ToolCallDisplay(
            summary="context_energy_insights",
            details={"item_count": len(event.args.items)},
        )

    @classmethod
    def get_result_display(cls, event: 'ToolResultEvent') -> ToolResultDisplay:
        if not isinstance(event.result, ContextEnergyInsightsResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        warnings = event.result.warnings[:]
        if event.result.truncated:
            warnings.append("Output truncated by limits")
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=f"Tagged {event.result.item_count} item(s).",
            warnings=warnings,
            details={
                "item_count": event.result.item_count,
                "errors": event.result.errors,
                "warnings": event.result.warnings,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Analyzing energy contexts"
