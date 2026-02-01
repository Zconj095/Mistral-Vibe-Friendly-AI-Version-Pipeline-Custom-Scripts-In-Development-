
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
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

CATEGORY_SECTIONS = {
    "Natural Particle Understanding": [
        ("Natural Energies", ["natural energies", "naturally flowing"]),
        (
            "Naturally Defined Energies",
            ["naturally defined energies", "naturally defined"],
        ),
    ],
    "Energy Related": [
        ("Energy Classification", ["energy classification", "classification"]),
        (
            "Energy Genre Classification",
            ["energy genre classification", "energy genre"],
        ),
        ("Energy Grouping", ["energy grouping", "grouping"]),
    ],
    "Classes of Energy": [
        ("Artificial Energy", ["artificial energy", "engineered energy"]),
        ("Natural Energy", ["natural energy", "unmediated energy"]),
    ],
    "Types of Energy": [
        ("Willed Energy", ["willed energy", "consent energy"]),
        ("Pressured Energy", ["pressured energy", "pressure energy"]),
        ("Forced Energy", ["forced energy", "force energy"]),
    ],
    "Classes of Vocals": [
        ("Artificial Vocals", ["artificial vocals", "constructed voice"]),
        ("Natural Vocals", ["natural vocals", "organic voice"]),
    ],
    "Classes of Breathing Patterns": [
        (
            "Artificial Breathing Patterns",
            ["artificial breathing", "engineered breath"],
        ),
        (
            "Natural Breathing Patterns",
            ["natural breathing", "organic breath"],
        ),
    ],
    "Chemistry Types of Focus": [
        ("Biochemistry", ["biochemistry"]),
        ("Neurochemistry", ["neurochemistry"]),
        ("Geochemistry", ["geochemistry"]),
    ],
    "Types of Vocals": [
        ("Willed Vocals", ["willed vocals", "guided vocals"]),
        ("Pressured Vocals", ["pressured vocals", "coercive vocals"]),
        ("Forced Vocals", ["forced vocals", "commanded vocals"]),
    ],
}

ENERGY_ORIGIN_DEFINITIONS = [
    ("Natural Energy Origin", ["sunlight", "seasons", "wind", "ocean", "earth"]),
    (
        "Artificial Energy Origin",
        ["machine", "technology", "laboratory", "grid", "circuit"],
    ),
]

ENERGY_STAT_TERMS = [
    "energy",
    "flow",
    "spark",
    "pulse",
    "force",
    "heat",
    "light",
    "power",
    "current",
    "charge",
]

STORED_ENERGY_TERMS = [
    "stored",
    "reservoir",
    "reserve",
    "cache",
    "memory",
    "archive",
]

CATEGORY_LABEL_LOOKUP: dict[str, str] = {}
for section, entries in CATEGORY_SECTIONS.items():
    for label, _ in entries:
        CATEGORY_LABEL_LOOKUP[label] = f"{section} - {label}"
for label, _ in ENERGY_ORIGIN_DEFINITIONS:
    CATEGORY_LABEL_LOOKUP[label] = f"Energy Origin - {label}"

COMMON_ENERGY_KEYWORDS = set()
for _, entries in CATEGORY_SECTIONS.items():
    for _, keywords in entries:
        COMMON_ENERGY_KEYWORDS.update(keywords)
for _, keywords in ENERGY_ORIGIN_DEFINITIONS:
    COMMON_ENERGY_KEYWORDS.update(keywords)
COMMON_ENERGY_KEYWORDS.update(ENERGY_STAT_TERMS)
COMMON_ENERGY_KEYWORDS.update(STORED_ENERGY_TERMS)
@dataclass
class _CategoryHit:
    label: str
    keywords: list[str]


class EnergyStatisticsConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_items: int = Field(default=50, description="Maximum items to analyze.")
    max_source_bytes: int = Field(
        default=3_000_000, description="Maximum bytes per text item."
    )
    max_total_bytes: int = Field(
        default=12_000_000, description="Maximum total bytes across inputs."
    )
    preview_chars: int = Field(default=400, description="Preview snippet length.")
    max_category_snippets: int = Field(
        default=2, description="Max snippets stored per category match."
    )
    max_energy_words: int = Field(
        default=20, description="Maximum energy words tracked per item."
    )
    min_energy_density: float = Field(
        default=0.0, description="Minimum energy density when flagging stats."
    )
class EnergyStatisticsState(BaseToolState):
    pass


class EnergySpeechItem(BaseModel):
    id: str | None = Field(default=None, description="Optional item id.")
    name: str | None = Field(default=None, description="Optional item name.")
    content: str | None = Field(default=None, description="Inline text content.")
    path: str | None = Field(default=None, description="Path to a text file.")
    source: str | None = Field(default=None, description="Source description.")


class EnergyStatisticsArgs(BaseModel):
    items: list[EnergySpeechItem] = Field(description="Document items to analyze.")
class EnergyCategoryMatch(BaseModel):
    label: str
    active: bool
    matched_terms: list[str]
    snippets: list[str]


class EnergyWordStat(BaseModel):
    word: str
    count: int
    categories: list[str]


class EnergySpeechSummary(BaseModel):
    index: int
    id: str | None
    name: str | None
    source_path: str | None
    source: str | None
    preview: str
    word_count: int
    char_count: int
    sentence_count: int
    energy_word_count: int
    stored_word_count: int
    energy_density: float
    stored_energy_density: float
    category_matches: list[EnergyCategoryMatch]
    word_stats: list[EnergyWordStat]


class EnergyStatisticsResult(BaseModel):
    items: list[EnergySpeechSummary]
    item_count: int
    category_totals: list[EnergyCategoryMatch]
    energy_word_totals: list[EnergyWordStat]
    speech_metrics: dict[str, float]
    truncated: bool
    errors: list[str]
    warnings: list[str]
class ContextEnergyStatistics(
    BaseTool[
        EnergyStatisticsArgs,
        EnergyStatisticsResult,
        EnergyStatisticsConfig,
        EnergyStatisticsState,
    ],
    ToolUIData[EnergyStatisticsArgs, EnergyStatisticsResult],
):
    description: ClassVar[str] = (
        "Capture energy origins, speech stats, and common-sense classifications."
    )

    async def run(
        self, args: EnergyStatisticsArgs
    ) -> EnergyStatisticsResult:
        if not args.items:
            raise ToolError("items is required.")

        if len(args.items) > self.config.max_items:
            raise ToolError(
                f"items exceeds max_items ({len(args.items)} > {self.config.max_items})."
            )

        total_bytes = 0
        errors: list[str] = []
        warnings: list[str] = []
        summaries: list[EnergySpeechSummary] = []
        aggregated_word_counts: dict[str, int] = {}
        category_hits_totals: dict[str, EnergyCategoryMatch] = {}
        truncated = False

        for idx, item in enumerate(args.items, start=1):
            try:
                content, source_path, size_bytes = self._load_item(item)
                if size_bytes is None:
                    raise ToolError("Item has no content.")
                if total_bytes + size_bytes > self.config.max_total_bytes:
                    truncated = True
                    warnings.append("Byte budget exhausted; stopping early.")
                    break
                total_bytes += size_bytes

                summary = self._analyze_content(content, idx, item)
                summaries.append(summary)

                for word_stat in summary.word_stats:
                    aggregated_word_counts[word_stat.word] = (
                        aggregated_word_counts.get(word_stat.word, 0) + word_stat.count
                    )

                for match in summary.category_matches:
                    existing = category_hits_totals.get(match.label)
                    if not existing:
                        category_hits_totals[match.label] = EnergyCategoryMatch(
                            label=match.label,
                            active=match.active,
                            matched_terms=match.matched_terms[:],
                            snippets=match.snippets[:],
                        )
                    else:
                        existing.active = existing.active or match.active
                        existing.matched_terms.extend(match.matched_terms)
                        existing.snippets.extend(match.snippets)
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not summaries:
            raise ToolError("No valid items processed.")

        energy_word_totals = self._build_aggregated_word_stats(aggregated_word_counts)
        speech_metrics = self._build_speech_metrics(summaries)
        category_totals = list(category_hits_totals.values())

        return EnergyStatisticsResult(
            items=summaries,
            item_count=len(summaries),
            category_totals=category_totals,
            energy_word_totals=energy_word_totals,
            speech_metrics=speech_metrics,
            truncated=truncated,
            errors=errors,
            warnings=warnings,
        )
    def _analyze_content(
        self, content: str, index: int, item: EnergySpeechItem
    ) -> EnergySpeechSummary:
        words = WORD_RE.findall(content)
        sentences = [part for part in SENTENCE_RE.split(content) if part.strip()]
        word_count = len(words)
        char_count = len(content)
        sentence_count = len(sentences)

        category_matches: list[EnergyCategoryMatch] = []
        lower = content.lower()
        for section, entries in CATEGORY_SECTIONS.items():
            for label, keywords in entries:
                category_matches.append(
                    self._match_category(label, keywords, lower, content)
                )
        for label, keywords in ENERGY_ORIGIN_DEFINITIONS:
            category_matches.append(self._match_category(label, keywords, lower, content))

        word_freq: dict[str, int] = {}
        for word in words:
            key = word.lower()
            word_freq[key] = word_freq.get(key, 0) + 1

        energy_word_count = sum(
            count for word, count in word_freq.items() if word in COMMON_ENERGY_KEYWORDS
        )
        stored_word_count = sum(
            count for word, count in word_freq.items() if word in STORED_ENERGY_TERMS
        )
        energy_density = energy_word_count / max(word_count, 1)
        stored_energy_density = stored_word_count / max(word_count, 1)

        word_stats = self._build_word_stats(word_freq)

        return EnergySpeechSummary(
            index=index,
            id=item.id,
            name=item.name,
            source_path=item.path,
            source=item.source,
            preview=content[: self.config.preview_chars],
            word_count=word_count,
            char_count=char_count,
            sentence_count=sentence_count,
            energy_word_count=energy_word_count,
            stored_word_count=stored_word_count,
            energy_density=round(energy_density, 4),
            stored_energy_density=round(stored_energy_density, 4),
            category_matches=category_matches,
            word_stats=word_stats,
        )
    def _match_category(
        self, label: str, keywords: list[str], lower: str, content: str
    ) -> EnergyCategoryMatch:
        matched_terms: list[str] = []
        snippets: list[str] = []
        for keyword in keywords:
            idx = lower.find(keyword)
            if idx < 0:
                continue
            matched_terms.append(keyword)
            snippet = _snippet_text(content, idx, len(keyword))
            if snippet and len(snippets) < self.config.max_category_snippets:
                snippets.append(snippet)
        return EnergyCategoryMatch(
            label=CATEGORY_LABEL_LOOKUP.get(label, label),
            active=bool(matched_terms),
            matched_terms=matched_terms,
            snippets=snippets,
        )

    def _build_word_stats(self, word_freq: dict[str, int]) -> list[EnergyWordStat]:
        entries = [
            (word, count)
            for word, count in word_freq.items()
            if word in COMMON_ENERGY_KEYWORDS
        ]
        entries.sort(key=lambda item: (-item[1], item[0]))
        stats: list[EnergyWordStat] = []
        for word, count in entries[: self.config.max_energy_words]:
            categories = [
                CATEGORY_LABEL_LOOKUP.get(label, label)
                for section, entries in CATEGORY_SECTIONS.items()
                for label, keywords in entries
                if word in map(str.lower, keywords)
            ]
            destinations = [
                CATEGORY_LABEL_LOOKUP.get(label, label)
                for label, keywords in ENERGY_ORIGIN_DEFINITIONS
                if word in map(str.lower, keywords)
            ]
            stats.append(
                EnergyWordStat(
                    word=word,
                    count=count,
                    categories=list(dict.fromkeys(categories + destinations)),
                )
            )
        return stats

    def _build_aggregated_word_stats(
        self, aggregated_word_counts: dict[str, int]
    ) -> list[EnergyWordStat]:
        entries = [
            (word, count)
            for word, count in aggregated_word_counts.items()
            if word in COMMON_ENERGY_KEYWORDS
        ]
        entries.sort(key=lambda item: (-item[1], item[0]))
        return [EnergyWordStat(word=word, count=count, categories=[]) for word, count in entries]

    def _build_speech_metrics(
        self, summaries: list[EnergySpeechSummary]
    ) -> dict[str, float]:
        total_words = sum(summary.word_count for summary in summaries)
        total_chars = sum(summary.char_count for summary in summaries)
        total_sentences = sum(summary.sentence_count for summary in summaries)
        avg_energy_density = (
            sum(summary.energy_density for summary in summaries) / len(summaries)
        )
        avg_stored_density = (
            sum(summary.stored_energy_density for summary in summaries) / len(summaries)
        )
        return {
            "total_words": float(total_words),
            "total_chars": float(total_chars),
            "total_sentences": float(total_sentences),
            "avg_energy_density": round(avg_energy_density, 4),
            "avg_stored_energy_density": round(avg_stored_density, 4),
        }
    def _load_item(
        self, item: EnergySpeechItem
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

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, EnergyStatisticsArgs):
            return ToolCallDisplay(summary="context_energy_statistics")
        return ToolCallDisplay(
            summary="context_energy_statistics",
            details={"item_count": len(event.args.items), "max_items": cls().config.max_items},
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, EnergyStatisticsResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        warnings = event.result.warnings[:]
        if event.result.truncated:
            warnings.append("Output truncated by limits")
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=(
                f"Analyzed {event.result.item_count} item(s), "
                f"{len(event.result.energy_word_totals)} energy words"
            ),
            warnings=warnings,
            details={
                "item_count": event.result.item_count,
                "truncated": event.result.truncated,
                "errors": event.result.errors,
                "warnings": event.result.warnings,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Calculating energy statistics"
