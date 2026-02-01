from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
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


WORD_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

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

SUPPORTED_FORMATS = {"auto", "json", "jsonl", "csv", "tsv", "wordlist", "text"}
SUPPORTED_CHUNK_MODES = {"words", "paragraphs", "sentences"}
DEFAULT_DOMAIN = "general"

PRIMARY_DOMAINS = {
    "it",
    "game_design",
    "hardware",
    "medical",
}

DOMAIN_SYNONYMS = {
    "it": "it",
    "informationtechnology": "it",
    "information_technology": "it",
    "infotech": "it",
    "software": "it",
    "systems": "it",
    "sysadmin": "it",
    "networking": "it",
    "game": "game_design",
    "gamedesign": "game_design",
    "game_design": "game_design",
    "gamedev": "game_design",
    "game_development": "game_design",
    "hardware": "hardware",
    "electronics": "hardware",
    "embedded": "hardware",
    "firmware": "hardware",
    "medical": "medical",
    "medicine": "medical",
    "clinical": "medical",
    "healthcare": "medical",
    "biomedical": "medical",
}

DOMAIN_MARKERS = {
    "it": {
        "server",
        "network",
        "database",
        "api",
        "protocol",
        "packet",
        "latency",
        "cloud",
        "deploy",
        "runtime",
        "cache",
        "kernel",
        "encryption",
        "backend",
    },
    "game_design": {
        "mechanic",
        "level",
        "quest",
        "npc",
        "balance",
        "gameplay",
        "sandbox",
        "ui",
        "player",
        "enemy",
        "loot",
        "combat",
        "progression",
        "tutorial",
    },
    "hardware": {
        "circuit",
        "pcb",
        "voltage",
        "current",
        "transistor",
        "sensor",
        "firmware",
        "driver",
        "gpio",
        "solder",
        "microcontroller",
        "power",
    },
    "medical": {
        "patient",
        "diagnosis",
        "symptom",
        "clinical",
        "therapy",
        "medication",
        "pathology",
        "surgery",
        "treatment",
        "radiology",
        "cardiac",
        "oncology",
    },
}


@dataclass
class _DictEntry:
    word: str
    definition: str | None
    dict_id: str
    dict_name: str | None


class ContextDictionaryDomainsConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_items: int = Field(default=200, description="Maximum items to process.")
    max_source_bytes: int = Field(
        default=5_000_000, description="Maximum size per item (bytes)."
    )
    max_total_bytes: int = Field(
        default=20_000_000, description="Maximum total bytes across items."
    )
    max_dictionary_bytes: int = Field(
        default=5_000_000, description="Maximum size per dictionary source (bytes)."
    )
    max_dictionary_entries: int = Field(
        default=200_000, description="Maximum entries per dictionary source."
    )
    max_chunks: int = Field(
        default=500, description="Maximum chunks returned across all items."
    )
    preview_chars: int = Field(
        default=400, description="Preview length per chunk."
    )
    default_chunk_mode: str = Field(
        default="words", description="Default chunking mode: words, paragraphs, sentences."
    )
    default_chunk_size: int = Field(
        default=200, description="Default chunk size for words/paragraphs/sentences."
    )
    default_chunk_overlap: int = Field(
        default=20, description="Default word overlap for word chunking."
    )
    min_word_length: int = Field(
        default=2, description="Minimum word length to include."
    )
    max_words_per_chunk: int = Field(
        default=2000, description="Maximum unique words per chunk to analyze."
    )
    max_word_sources: int = Field(
        default=100, description="Maximum word source entries per chunk."
    )
    max_missing_words: int = Field(
        default=50, description="Maximum missing words per chunk."
    )
    max_shared_words: int = Field(
        default=50, description="Maximum shared words per chunk."
    )
    max_sample_words: int = Field(
        default=20, description="Maximum sample words per dictionary usage entry."
    )
    max_definition_checks: int = Field(
        default=50, description="Maximum definition checks per chunk."
    )
    max_definition_chars: int = Field(
        default=200, description="Maximum characters stored per definition snippet."
    )
    definition_similarity: float = Field(
        default=0.2, description="Similarity threshold for definition validation."
    )
    max_dictionary_overlaps: int = Field(
        default=200, description="Maximum dictionary overlap entries returned."
    )
    max_domain_overlaps: int = Field(
        default=200, description="Maximum domain overlap entries returned."
    )
    max_domain_recommendations: int = Field(
        default=5, description="Maximum domain recommendations per chunk."
    )
    max_bridge_words: int = Field(
        default=50, description="Maximum bridge words per chunk."
    )
    domain_marker_boost: float = Field(
        default=0.1, description="Boost for domain marker matches."
    )
    bridge_boost: float = Field(
        default=0.05, description="Boost for cross-domain bridge words."
    )


class ContextDictionaryDomainsState(BaseToolState):
    pass

class DictionaryEntry(BaseModel):
    word: str = Field(description="Dictionary word.")
    definition: str | None = Field(default=None, description="Optional definition text.")


class DictionaryDomainSource(BaseModel):
    id: str | None = Field(default=None, description="Optional dictionary id.")
    name: str | None = Field(default=None, description="Dictionary name.")
    path: str | None = Field(default=None, description="Path to a dictionary file.")
    content: str | None = Field(default=None, description="Inline dictionary content.")
    entries: list[DictionaryEntry] | dict[str, str] | list[dict] | None = Field(
        default=None, description="Inline dictionary entries."
    )
    format: str | None = Field(
        default="auto",
        description="auto, json, jsonl, csv, tsv, wordlist, text.",
    )
    separator: str | None = Field(
        default=None, description="Separator for text format entries."
    )
    source: str | None = Field(default=None, description="Source description.")
    domain: str | None = Field(default=None, description="Primary domain tag.")
    domains: list[str] | None = Field(default=None, description="Additional domain tags.")
    subdomain: str | None = Field(default=None, description="Subdomain tag.")
    subdomains: list[str] | None = Field(default=None, description="Subdomain tags.")
    industry: str | None = Field(default=None, description="Industry tag.")
    specialty: str | None = Field(default=None, description="Specialty tag.")
    discipline: str | None = Field(default=None, description="Discipline tag.")
    field: str | None = Field(default=None, description="Field tag.")


class TextItem(BaseModel):
    id: str | None = Field(default=None, description="Optional item id.")
    name: str | None = Field(default=None, description="Optional item name.")
    content: str | None = Field(default=None, description="Inline text content.")
    path: str | None = Field(default=None, description="Path to a text file.")
    source: str | None = Field(default=None, description="Source description.")
    chunk_mode: str | None = Field(
        default=None, description="words, paragraphs, sentences."
    )
    chunk_size: int | None = Field(default=None, description="Override chunk size.")
    chunk_overlap: int | None = Field(
        default=None, description="Override word overlap for word chunking."
    )


class ContextDictionaryDomainsArgs(BaseModel):
    dictionaries: list[DictionaryDomainSource] = Field(
        description="Dictionary sources to index."
    )
    items: list[TextItem] = Field(description="Text items to analyze.")
    chunk_mode: str | None = Field(
        default=None, description="Override chunk mode for items."
    )
    chunk_size: int | None = Field(
        default=None, description="Override chunk size for items."
    )
    chunk_overlap: int | None = Field(
        default=None, description="Override word overlap for word chunking."
    )
    max_items: int | None = Field(default=None, description="Override max_items.")
    max_source_bytes: int | None = Field(
        default=None, description="Override max_source_bytes."
    )
    max_total_bytes: int | None = Field(
        default=None, description="Override max_total_bytes."
    )
    max_dictionary_bytes: int | None = Field(
        default=None, description="Override max_dictionary_bytes."
    )
    max_dictionary_entries: int | None = Field(
        default=None, description="Override max_dictionary_entries."
    )
    max_chunks: int | None = Field(default=None, description="Override max_chunks.")
    preview_chars: int | None = Field(
        default=None, description="Override preview_chars."
    )
    min_word_length: int | None = Field(
        default=None, description="Override min_word_length."
    )
    max_words_per_chunk: int | None = Field(
        default=None, description="Override max_words_per_chunk."
    )
    max_word_sources: int | None = Field(
        default=None, description="Override max_word_sources."
    )
    max_missing_words: int | None = Field(
        default=None, description="Override max_missing_words."
    )
    max_shared_words: int | None = Field(
        default=None, description="Override max_shared_words."
    )
    max_sample_words: int | None = Field(
        default=None, description="Override max_sample_words."
    )
    max_definition_checks: int | None = Field(
        default=None, description="Override max_definition_checks."
    )
    max_definition_chars: int | None = Field(
        default=None, description="Override max_definition_chars."
    )
    definition_similarity: float | None = Field(
        default=None, description="Override definition_similarity."
    )
    max_dictionary_overlaps: int | None = Field(
        default=None, description="Override max_dictionary_overlaps."
    )
    max_domain_overlaps: int | None = Field(
        default=None, description="Override max_domain_overlaps."
    )
    max_domain_recommendations: int | None = Field(
        default=None, description="Override max_domain_recommendations."
    )
    max_bridge_words: int | None = Field(
        default=None, description="Override max_bridge_words."
    )
    domain_marker_boost: float | None = Field(
        default=None, description="Override domain_marker_boost."
    )
    bridge_boost: float | None = Field(
        default=None, description="Override bridge_boost."
    )


class DictionarySummary(BaseModel):
    dictionary_id: str
    name: str | None
    entry_count: int
    word_count: int
    definition_count: int
    source: str | None
    domain_tags: list[str]


class DictionaryUsage(BaseModel):
    dictionary_id: str
    name: str | None
    matched_words: int
    coverage: float
    sample_words: list[str]


class DomainUsage(BaseModel):
    tag: str
    matched_words: int
    coverage: float
    dictionary_ids: list[str]
    sample_words: list[str]


class DomainRecommendation(BaseModel):
    tag: str
    score: float
    coverage: float
    reasons: list[str]


class DomainBridgeWord(BaseModel):
    word: str
    domain_tags: list[str]
    dictionary_ids: list[str]


class WordSource(BaseModel):
    word: str
    dictionary_ids: list[str]
    domain_tags: list[str]


class ContextSignals(BaseModel):
    it_markers: int
    game_design_markers: int
    hardware_markers: int
    medical_markers: int


class DefinitionCheck(BaseModel):
    word: str
    dictionary_ids: list[str]
    average_similarity: float
    status: str
    notes: list[str]
    definitions: dict[str, str]


class ChunkAnalysis(BaseModel):
    item_index: int
    item_id: str | None
    chunk_index: int
    preview: str
    word_count: int
    unique_words: int
    dictionary_usages: list[DictionaryUsage]
    domain_usages: list[DomainUsage]
    domain_recommendations: list[DomainRecommendation]
    bridge_words: list[DomainBridgeWord]
    shared_words: list[str]
    missing_words: list[str]
    word_sources: list[WordSource]
    context_signals: ContextSignals
    definition_checks: list[DefinitionCheck]
    truncated: bool


class DictionaryOverlap(BaseModel):
    left_id: str
    right_id: str
    shared_words: int
    similarity: float


class DomainOverlap(BaseModel):
    left_tag: str
    right_tag: str
    shared_words: int
    similarity: float


class ContextDictionaryDomainsResult(BaseModel):
    dictionaries: list[DictionarySummary]
    chunks: list[ChunkAnalysis]
    dictionary_overlaps: list[DictionaryOverlap]
    domain_overlaps: list[DomainOverlap]
    dictionary_count: int
    chunk_count: int
    overlap_count: int
    domain_overlap_count: int
    truncated: bool
    errors: list[str]

class ContextDictionaryDomains(
    BaseTool[
        ContextDictionaryDomainsArgs,
        ContextDictionaryDomainsResult,
        ContextDictionaryDomainsConfig,
        ContextDictionaryDomainsState,
    ],
    ToolUIData[ContextDictionaryDomainsArgs, ContextDictionaryDomainsResult],
):
    description: ClassVar[str] = (
        "Chunk and reason across domain-specific English dictionaries."
    )

    async def run(
        self, args: ContextDictionaryDomainsArgs
    ) -> ContextDictionaryDomainsResult:
        if not args.dictionaries:
            raise ToolError("dictionaries is required.")
        if not args.items:
            raise ToolError("items is required.")

        max_items = args.max_items if args.max_items is not None else self.config.max_items
        if max_items <= 0:
            raise ToolError("max_items must be a positive integer.")
        if len(args.items) > max_items:
            raise ToolError(f"items exceeds max_items ({len(args.items)} > {max_items}).")

        max_source_bytes = (
            args.max_source_bytes
            if args.max_source_bytes is not None
            else self.config.max_source_bytes
        )
        max_total_bytes = (
            args.max_total_bytes
            if args.max_total_bytes is not None
            else self.config.max_total_bytes
        )
        max_dictionary_bytes = (
            args.max_dictionary_bytes
            if args.max_dictionary_bytes is not None
            else self.config.max_dictionary_bytes
        )
        max_dictionary_entries = (
            args.max_dictionary_entries
            if args.max_dictionary_entries is not None
            else self.config.max_dictionary_entries
        )
        max_chunks = args.max_chunks if args.max_chunks is not None else self.config.max_chunks
        preview_chars = (
            args.preview_chars if args.preview_chars is not None else self.config.preview_chars
        )
        min_word_length = (
            args.min_word_length
            if args.min_word_length is not None
            else self.config.min_word_length
        )
        max_words_per_chunk = (
            args.max_words_per_chunk
            if args.max_words_per_chunk is not None
            else self.config.max_words_per_chunk
        )
        max_word_sources = (
            args.max_word_sources
            if args.max_word_sources is not None
            else self.config.max_word_sources
        )
        max_missing_words = (
            args.max_missing_words
            if args.max_missing_words is not None
            else self.config.max_missing_words
        )
        max_shared_words = (
            args.max_shared_words
            if args.max_shared_words is not None
            else self.config.max_shared_words
        )
        max_sample_words = (
            args.max_sample_words
            if args.max_sample_words is not None
            else self.config.max_sample_words
        )
        max_definition_checks = (
            args.max_definition_checks
            if args.max_definition_checks is not None
            else self.config.max_definition_checks
        )
        max_definition_chars = (
            args.max_definition_chars
            if args.max_definition_chars is not None
            else self.config.max_definition_chars
        )
        definition_similarity = (
            args.definition_similarity
            if args.definition_similarity is not None
            else self.config.definition_similarity
        )
        max_dictionary_overlaps = (
            args.max_dictionary_overlaps
            if args.max_dictionary_overlaps is not None
            else self.config.max_dictionary_overlaps
        )
        max_domain_overlaps = (
            args.max_domain_overlaps
            if args.max_domain_overlaps is not None
            else self.config.max_domain_overlaps
        )
        max_domain_recommendations = (
            args.max_domain_recommendations
            if args.max_domain_recommendations is not None
            else self.config.max_domain_recommendations
        )
        max_bridge_words = (
            args.max_bridge_words
            if args.max_bridge_words is not None
            else self.config.max_bridge_words
        )
        domain_marker_boost = (
            args.domain_marker_boost
            if args.domain_marker_boost is not None
            else self.config.domain_marker_boost
        )
        bridge_boost = (
            args.bridge_boost
            if args.bridge_boost is not None
            else self.config.bridge_boost
        )

        if max_source_bytes <= 0:
            raise ToolError("max_source_bytes must be a positive integer.")
        if max_total_bytes <= 0:
            raise ToolError("max_total_bytes must be a positive integer.")
        if max_dictionary_bytes <= 0:
            raise ToolError("max_dictionary_bytes must be a positive integer.")
        if max_dictionary_entries <= 0:
            raise ToolError("max_dictionary_entries must be a positive integer.")
        if max_chunks <= 0:
            raise ToolError("max_chunks must be a positive integer.")
        if preview_chars < 0:
            raise ToolError("preview_chars must be >= 0.")
        if min_word_length <= 0:
            raise ToolError("min_word_length must be a positive integer.")
        if max_words_per_chunk <= 0:
            raise ToolError("max_words_per_chunk must be a positive integer.")
        if max_word_sources < 0:
            raise ToolError("max_word_sources must be >= 0.")
        if max_missing_words < 0:
            raise ToolError("max_missing_words must be >= 0.")
        if max_shared_words < 0:
            raise ToolError("max_shared_words must be >= 0.")
        if max_sample_words < 0:
            raise ToolError("max_sample_words must be >= 0.")
        if max_definition_checks < 0:
            raise ToolError("max_definition_checks must be >= 0.")
        if max_definition_chars < 0:
            raise ToolError("max_definition_chars must be >= 0.")
        if definition_similarity < 0:
            raise ToolError("definition_similarity must be >= 0.")
        if max_dictionary_overlaps < 0:
            raise ToolError("max_dictionary_overlaps must be >= 0.")
        if max_domain_overlaps < 0:
            raise ToolError("max_domain_overlaps must be >= 0.")
        if max_domain_recommendations < 0:
            raise ToolError("max_domain_recommendations must be >= 0.")
        if max_bridge_words < 0:
            raise ToolError("max_bridge_words must be >= 0.")
        if domain_marker_boost < 0:
            raise ToolError("domain_marker_boost must be >= 0.")
        if bridge_boost < 0:
            raise ToolError("bridge_boost must be >= 0.")

        dictionaries: list[DictionarySummary] = []
        dict_word_sets: dict[str, set[str]] = {}
        dict_names: dict[str, str | None] = {}
        dict_entries: dict[str, list[_DictEntry]] = {}
        dict_domain_tags: dict[str, set[str]] = {}
        errors: list[str] = []
        truncated = False

        used_ids: set[str] = set()
        for idx, source in enumerate(args.dictionaries, start=1):
            try:
                dict_id = self._resolve_dict_id(source, idx, used_ids)
                dict_name = source.name
                tags = self._build_domain_tags(source)
                if not tags:
                    tags = {DEFAULT_DOMAIN}
                entries, entry_count, definition_count, was_truncated = (
                    self._load_dictionary_entries(
                        source,
                        dict_id,
                        dict_name,
                        max_dictionary_bytes,
                        max_dictionary_entries,
                        min_word_length,
                    )
                )
                if was_truncated:
                    truncated = True
                if not entries:
                    raise ToolError("dictionary contains no valid entries.")

                dict_word_sets[dict_id] = {entry.word for entry in entries}
                dict_names[dict_id] = dict_name
                dict_entries[dict_id] = entries
                dict_domain_tags[dict_id] = tags

                dictionaries.append(
                    DictionarySummary(
                        dictionary_id=dict_id,
                        name=dict_name,
                        entry_count=entry_count,
                        word_count=len(dict_word_sets[dict_id]),
                        definition_count=definition_count,
                        source=source.source,
                        domain_tags=sorted(tags),
                    )
                )
            except ToolError as exc:
                errors.append(f"dictionary[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"dictionary[{idx}]: {exc}")

        if not dictionaries:
            raise ToolError("No valid dictionaries loaded.")

        word_index = self._build_word_index(dict_entries)

        chunks: list[ChunkAnalysis] = []
        total_bytes = 0
        chunk_count = 0

        chunk_mode_default = (args.chunk_mode or self.config.default_chunk_mode).strip().lower()
        if chunk_mode_default not in SUPPORTED_CHUNK_MODES:
            raise ToolError("chunk_mode must be words, paragraphs, or sentences.")

        for item_idx, item in enumerate(args.items, start=1):
            try:
                content, source_path, size_bytes = self._load_item_content(
                    item, max_source_bytes
                )
                if content is None:
                    raise ToolError("Item has no content to analyze.")
                if size_bytes is not None:
                    if total_bytes + size_bytes > max_total_bytes:
                        truncated = True
                        break
                    total_bytes += size_bytes

                chunk_mode = (
                    (item.chunk_mode or args.chunk_mode or self.config.default_chunk_mode)
                    .strip()
                    .lower()
                )
                if chunk_mode not in SUPPORTED_CHUNK_MODES:
                    raise ToolError("chunk_mode must be words, paragraphs, or sentences.")

                chunk_size = (
                    item.chunk_size
                    if item.chunk_size is not None
                    else (args.chunk_size if args.chunk_size is not None else self.config.default_chunk_size)
                )
                if chunk_size <= 0:
                    raise ToolError("chunk_size must be a positive integer.")

                chunk_overlap = (
                    item.chunk_overlap
                    if item.chunk_overlap is not None
                    else (
                        args.chunk_overlap
                        if args.chunk_overlap is not None
                        else self.config.default_chunk_overlap
                    )
                )
                if chunk_overlap < 0:
                    raise ToolError("chunk_overlap must be >= 0.")

                chunk_texts = self._chunk_content(
                    content, chunk_mode, chunk_size, chunk_overlap
                )
                for local_idx, chunk_text in enumerate(chunk_texts, start=1):
                    if chunk_count >= max_chunks:
                        truncated = True
                        break
                    analysis, was_truncated = self._analyze_chunk(
                        chunk_text,
                        word_index,
                        dict_word_sets,
                        dict_names,
                        dict_domain_tags,
                        max_words_per_chunk,
                        min_word_length,
                        max_word_sources,
                        max_missing_words,
                        max_shared_words,
                        max_sample_words,
                        max_definition_checks,
                        max_definition_chars,
                        definition_similarity,
                        preview_chars,
                        max_domain_recommendations,
                        max_bridge_words,
                        domain_marker_boost,
                        bridge_boost,
                    )
                    if was_truncated:
                        truncated = True
                    analysis.item_index = item_idx
                    analysis.item_id = item.id
                    analysis.chunk_index = local_idx
                    chunks.append(analysis)
                    chunk_count += 1
                if chunk_count >= max_chunks:
                    break
            except ToolError as exc:
                errors.append(f"item[{item_idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{item_idx}]: {exc}")

        overlaps, overlap_truncated = self._build_dictionary_overlaps(
            dict_word_sets, max_dictionary_overlaps
        )
        if overlap_truncated:
            truncated = True

        domain_overlaps, domain_truncated = self._build_domain_overlaps(
            dict_word_sets, dict_domain_tags, max_domain_overlaps
        )
        if domain_truncated:
            truncated = True

        return ContextDictionaryDomainsResult(
            dictionaries=dictionaries,
            chunks=chunks,
            dictionary_overlaps=overlaps,
            domain_overlaps=domain_overlaps,
            dictionary_count=len(dictionaries),
            chunk_count=len(chunks),
            overlap_count=len(overlaps),
            domain_overlap_count=len(domain_overlaps),
            truncated=truncated,
            errors=errors,
        )

    def _resolve_dict_id(
        self, source: DictionaryDomainSource, index: int, used_ids: set[str]
    ) -> str:
        raw = source.id or source.name or f"dict{index}"
        normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw.strip()).lower()
        if not normalized:
            normalized = f"dict{index}"
        candidate = normalized
        counter = 2
        while candidate in used_ids:
            candidate = f"{normalized}_{counter}"
            counter += 1
        used_ids.add(candidate)
        return candidate

    def _build_domain_tags(self, source: DictionaryDomainSource) -> set[str]:
        raw_tags: list[str] = []
        for value in [source.domain, source.industry, source.discipline, source.field]:
            if value:
                raw_tags.extend(self._split_tags(value))
        for value in source.domains or []:
            raw_tags.extend(self._split_tags(value))

        sub_tags: list[str] = []
        for value in [source.subdomain, source.specialty]:
            if value:
                sub_tags.extend(self._split_tags(value))
        for value in source.subdomains or []:
            sub_tags.extend(self._split_tags(value))

        normalized: set[str] = set()
        for tag in raw_tags:
            normalized_tag = self._normalize_tag(tag)
            if normalized_tag:
                normalized.add(normalized_tag)

        expanded: set[str] = set(normalized)
        for tag in list(normalized):
            if tag.startswith("medical_"):
                expanded.add("medical")
            if tag.startswith("hardware_"):
                expanded.add("hardware")
            if tag.startswith("game_design_"):
                expanded.add("game_design")
            if tag.startswith("it_"):
                expanded.add("it")

        base_domains = {tag for tag in expanded if tag in PRIMARY_DOMAINS}
        if not base_domains:
            base_domains = {tag for tag in expanded if tag.startswith(tuple(PRIMARY_DOMAINS))}

        for sub_tag in sub_tags:
            normalized_sub = self._normalize_tag(sub_tag)
            if not normalized_sub:
                continue
            if base_domains:
                for base in base_domains:
                    if normalized_sub == base or normalized_sub.startswith(f"{base}_"):
                        expanded.add(normalized_sub)
                    else:
                        expanded.add(f"{base}_{normalized_sub}")
            else:
                expanded.add(normalized_sub)

        return expanded

    def _split_tags(self, raw: str) -> list[str]:
        parts = re.split(r"[;,/|>]+", raw)
        return [part.strip() for part in parts if part.strip()]

    def _normalize_tag(self, raw: str) -> str | None:
        cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "", raw.strip().lower())
        if not cleaned:
            return None
        mapped = DOMAIN_SYNONYMS.get(cleaned, cleaned)
        mapped = mapped.replace("-", "_")
        return mapped

    def _load_dictionary_entries(
        self,
        source: DictionaryDomainSource,
        dict_id: str,
        dict_name: str | None,
        max_dictionary_bytes: int,
        max_dictionary_entries: int,
        min_word_length: int,
    ) -> tuple[list[_DictEntry], int, int, bool]:
        raw_entries: list[DictionaryEntry] = []
        if source.entries is not None:
            raw_entries = self._coerce_entries(source.entries)
        else:
            content = self._load_dictionary_content(source, max_dictionary_bytes)
            fmt = (source.format or "auto").strip().lower()
            if fmt not in SUPPORTED_FORMATS:
                raise ToolError("format must be auto, json, jsonl, csv, tsv, wordlist, or text.")
            raw_entries = self._parse_dictionary_content(content, fmt, source.separator)

        entry_count = 0
        truncated = False
        word_map: dict[str, str | None] = {}

        for entry in raw_entries:
            entry_count += 1
            if entry_count > max_dictionary_entries:
                truncated = True
                break
            word = self._normalize_word(entry.word, min_word_length)
            if not word:
                continue
            definition = self._normalize_definition(entry.definition)
            if word in word_map:
                if not word_map[word] and definition:
                    word_map[word] = definition
                continue
            word_map[word] = definition

        entries: list[_DictEntry] = []
        definition_count = 0
        for word, definition in word_map.items():
            if definition:
                definition_count += 1
            entries.append(
                _DictEntry(
                    word=word,
                    definition=definition,
                    dict_id=dict_id,
                    dict_name=dict_name,
                )
            )

        return entries, min(entry_count, max_dictionary_entries), definition_count, truncated

    def _load_dictionary_content(
        self, source: DictionaryDomainSource, max_dictionary_bytes: int
    ) -> str:
        if source.content and source.path:
            raise ToolError("Provide content or path for dictionaries, not both.")
        if source.content is None and source.path is None:
            raise ToolError("Dictionary source must include content, path, or entries.")

        if source.content is not None:
            size = len(source.content.encode("utf-8"))
            if size > max_dictionary_bytes:
                raise ToolError(
                    f"Dictionary content exceeds max_dictionary_bytes ({size} > {max_dictionary_bytes})."
                )
            return source.content

        path = Path(source.path or "").expanduser()
        if not path.is_absolute():
            path = self.config.effective_workdir / path
        path = path.resolve()
        if not path.exists():
            raise ToolError(f"Dictionary file not found: {path}")
        if path.is_dir():
            raise ToolError(f"Dictionary path is a directory: {path}")
        size = path.stat().st_size
        if size > max_dictionary_bytes:
            raise ToolError(
                f"Dictionary file exceeds max_dictionary_bytes ({size} > {max_dictionary_bytes})."
            )
        return path.read_text("utf-8", errors="ignore")

    def _coerce_entries(
        self, entries: list[DictionaryEntry] | dict[str, str] | list[dict]
    ) -> list[DictionaryEntry]:
        if isinstance(entries, dict):
            output: list[DictionaryEntry] = []
            for word, defn in entries.items():
                definition = str(defn) if defn is not None else None
                output.append(DictionaryEntry(word=str(word), definition=definition))
            return output

        output: list[DictionaryEntry] = []
        for item in entries:
            if isinstance(item, DictionaryEntry):
                output.append(item)
                continue
            if isinstance(item, dict):
                word = item.get("word") or item.get("term") or item.get("entry")
                if word is None:
                    continue
                definition = item.get("definition") or item.get("meaning") or item.get("def")
                output.append(
                    DictionaryEntry(
                        word=str(word),
                        definition=str(definition) if definition is not None else None,
                    )
                )
                continue
            if isinstance(item, str):
                output.append(DictionaryEntry(word=item, definition=None))
        return output

    def _parse_dictionary_content(
        self, content: str, fmt: str, separator: str | None
    ) -> list[DictionaryEntry]:
        detected = fmt
        if fmt == "auto":
            detected = self._detect_format(content)
        if detected == "json":
            try:
                return self._parse_json_entries(content)
            except Exception:
                return self._parse_jsonl_entries(content)
        if detected == "jsonl":
            return self._parse_jsonl_entries(content)
        if detected == "csv":
            return self._parse_csv_entries(content, ",")
        if detected == "tsv":
            return self._parse_csv_entries(content, "\t")
        if detected == "wordlist":
            return self._parse_wordlist_entries(content)
        return self._parse_text_entries(content, separator)

    def _detect_format(self, content: str) -> str:
        stripped = content.lstrip()
        if stripped.startswith("{") or stripped.startswith("["):
            return "json"
        lines = [line for line in content.splitlines() if line.strip()]
        if not lines:
            return "wordlist"
        if any("\t" in line for line in lines):
            return "tsv"
        comma_lines = sum(1 for line in lines if line.count(",") >= 1)
        if comma_lines >= max(2, len(lines) // 4):
            return "csv"
        if any(":" in line for line in lines) or any(" - " in line for line in lines):
            return "text"
        return "wordlist"

    def _parse_json_entries(self, content: str) -> list[DictionaryEntry]:
        data = json.loads(content)
        if isinstance(data, dict):
            return [
                DictionaryEntry(
                    word=str(word),
                    definition=self._stringify_definition(defn),
                )
                for word, defn in data.items()
            ]
        if isinstance(data, list):
            return self._coerce_entries(data)
        raise ToolError("JSON dictionary must be an object or list.")

    def _parse_jsonl_entries(self, content: str) -> list[DictionaryEntry]:
        entries: list[DictionaryEntry] = []
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            data = json.loads(stripped)
            if isinstance(data, dict):
                word = data.get("word") or data.get("term") or data.get("entry")
                if word is None:
                    continue
                definition = data.get("definition") or data.get("meaning") or data.get("def")
                entries.append(
                    DictionaryEntry(
                        word=str(word),
                        definition=self._stringify_definition(definition),
                    )
                )
            elif isinstance(data, str):
                entries.append(DictionaryEntry(word=data, definition=None))
        return entries

    def _parse_csv_entries(self, content: str, delimiter: str) -> list[DictionaryEntry]:
        entries: list[DictionaryEntry] = []
        rows = list(csv.reader(content.splitlines(), delimiter=delimiter))
        if not rows:
            return entries

        header = [cell.strip().lower() for cell in rows[0]]
        start_idx = 0
        word_idx = 0
        definition_idx = 1 if len(rows[0]) > 1 else None
        if any(token in header for token in {"word", "term", "entry"}):
            start_idx = 1
            for idx, token in enumerate(header):
                if token in {"word", "term", "entry"}:
                    word_idx = idx
                if token in {"definition", "meaning", "def"}:
                    definition_idx = idx

        for row in rows[start_idx:]:
            if not row:
                continue
            if word_idx >= len(row):
                continue
            word = row[word_idx].strip()
            if not word:
                continue
            definition = None
            if definition_idx is not None and definition_idx < len(row):
                definition = row[definition_idx].strip() or None
            entries.append(DictionaryEntry(word=word, definition=definition))

        return entries

    def _parse_wordlist_entries(self, content: str) -> list[DictionaryEntry]:
        entries: list[DictionaryEntry] = []
        for line in content.splitlines():
            word = line.strip()
            if not word or word.startswith("#"):
                continue
            entries.append(DictionaryEntry(word=word, definition=None))
        return entries

    def _parse_text_entries(
        self, content: str, separator: str | None
    ) -> list[DictionaryEntry]:
        entries: list[DictionaryEntry] = []
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            word = stripped
            definition = None
            split_token = None
            if separator:
                split_token = separator
            else:
                if ":" in stripped:
                    split_token = ":"
                elif " - " in stripped:
                    split_token = " - "
                elif "-" in stripped:
                    split_token = "-"
            if split_token and split_token in stripped:
                parts = stripped.split(split_token, 1)
                word = parts[0].strip()
                definition = parts[1].strip() if len(parts) > 1 else None
            entries.append(DictionaryEntry(word=word, definition=definition))
        return entries

    def _stringify_definition(self, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, list):
            items = [str(item) for item in value if item is not None]
            return "; ".join(items) if items else None
        return str(value)

    def _normalize_word(self, word: str, min_word_length: int) -> str | None:
        if not word:
            return None
        tokens = WORD_RE.findall(word.lower())
        if not tokens:
            return None
        normalized = tokens[0]
        if len(normalized) < min_word_length:
            return None
        return normalized

    def _normalize_definition(self, definition: str | None) -> str | None:
        if definition is None:
            return None
        text = definition.strip()
        return text if text else None

    def _load_item_content(
        self, item: TextItem, max_source_bytes: int
    ) -> tuple[str | None, str | None, int | None]:
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
                raise ToolError(f"Path is a directory, not a file: {path}")
            size = path.stat().st_size
            if size > max_source_bytes:
                raise ToolError(
                    f"{path} exceeds max_source_bytes ({size} > {max_source_bytes})."
                )
            return path.read_text("utf-8", errors="ignore"), str(path), size

        if item.content is not None:
            size = len(item.content.encode("utf-8"))
            if size > max_source_bytes:
                raise ToolError(
                    f"content exceeds max_source_bytes ({size} > {max_source_bytes})."
                )
            return item.content, None, size

        return None, None, None

    def _chunk_content(
        self, content: str, mode: str, size: int, overlap: int
    ) -> list[str]:
        if mode == "paragraphs":
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", content) if p.strip()]
            return self._group_chunks(paragraphs, size, "\n\n")
        if mode == "sentences":
            sentences = [s.strip() for s in SENTENCE_RE.split(content) if s.strip()]
            return self._group_chunks(sentences, size, " ")

        words = WORD_RE.findall(content)
        if not words:
            return []
        cleaned = [word.lower() for word in words]
        if overlap >= size:
            raise ToolError("chunk_overlap must be smaller than chunk_size for word chunking.")
        step = max(size - overlap, 1)
        chunks: list[str] = []
        for start in range(0, len(cleaned), step):
            subset = cleaned[start : start + size]
            if not subset:
                continue
            chunks.append(" ".join(subset))
        return chunks

    def _group_chunks(self, items: list[str], size: int, joiner: str) -> list[str]:
        if not items:
            return []
        if size <= 0:
            raise ToolError("chunk_size must be a positive integer.")
        chunks: list[str] = []
        for start in range(0, len(items), size):
            subset = items[start : start + size]
            if subset:
                chunks.append(joiner.join(subset))
        return chunks

    def _extract_words(self, text: str, min_word_length: int) -> list[str]:
        return [
            token.lower()
            for token in WORD_RE.findall(text)
            if len(token) >= min_word_length
        ]

    def _preview_text(self, text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def _build_word_index(
        self, dict_entries: dict[str, list[_DictEntry]]
    ) -> dict[str, list[_DictEntry]]:
        index: dict[str, list[_DictEntry]] = {}
        for entries in dict_entries.values():
            for entry in entries:
                index.setdefault(entry.word, []).append(entry)
        return index

    def _analyze_chunk(
        self,
        chunk_text: str,
        word_index: dict[str, list[_DictEntry]],
        dict_word_sets: dict[str, set[str]],
        dict_names: dict[str, str | None],
        dict_domain_tags: dict[str, set[str]],
        max_words_per_chunk: int,
        min_word_length: int,
        max_word_sources: int,
        max_missing_words: int,
        max_shared_words: int,
        max_sample_words: int,
        max_definition_checks: int,
        max_definition_chars: int,
        definition_similarity: float,
        preview_chars: int,
        max_domain_recommendations: int,
        max_bridge_words: int,
        domain_marker_boost: float,
        bridge_boost: float,
    ) -> tuple[ChunkAnalysis, bool]:
        words = self._extract_words(chunk_text, min_word_length)
        word_count = len(words)
        if not words:
            return (
                ChunkAnalysis(
                    item_index=0,
                    item_id=None,
                    chunk_index=0,
                    preview=self._preview_text(chunk_text, preview_chars),
                    word_count=0,
                    unique_words=0,
                    dictionary_usages=[],
                    domain_usages=[],
                    domain_recommendations=[],
                    bridge_words=[],
                    shared_words=[],
                    missing_words=[],
                    word_sources=[],
                    context_signals=ContextSignals(
                        it_markers=0,
                        game_design_markers=0,
                        hardware_markers=0,
                        medical_markers=0,
                    ),
                    definition_checks=[],
                    truncated=False,
                ),
                False,
            )

        truncated = False
        word_freq: dict[str, int] = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        unique_words = list(word_freq.keys())
        if len(unique_words) > max_words_per_chunk:
            truncated = True
            ordered = sorted(word_freq.items(), key=lambda item: (-item[1], item[0]))
            unique_words = [word for word, _ in ordered[:max_words_per_chunk]]

        word_set = set(unique_words)
        union_words: set[str] = set()
        for value in dict_word_sets.values():
            union_words |= value

        dictionary_usages: list[DictionaryUsage] = []
        for dict_id, words_set in dict_word_sets.items():
            matched = sorted(word_set & words_set)
            matched_count = len(matched)
            coverage = matched_count / len(word_set) if word_set else 0.0
            sample = matched[:max_sample_words] if max_sample_words > 0 else []
            dictionary_usages.append(
                DictionaryUsage(
                    dictionary_id=dict_id,
                    name=dict_names.get(dict_id),
                    matched_words=matched_count,
                    coverage=round(coverage, 6),
                    sample_words=sample,
                )
            )

        word_sources: list[WordSource] = []
        shared_candidates: list[tuple[int, str]] = []
        domain_word_map: dict[str, set[str]] = {}
        domain_dicts: dict[str, set[str]] = {}
        bridge_candidates: list[DomainBridgeWord] = []
        bridge_counts: dict[str, int] = {}
        available_tags: set[str] = set()
        for tags in dict_domain_tags.values():
            if tags:
                available_tags.update(tags)
        if not available_tags:
            available_tags.add(DEFAULT_DOMAIN)

        for word in word_set:
            entries = word_index.get(word)
            if not entries:
                continue
            dict_ids = sorted({entry.dict_id for entry in entries})
            if len(dict_ids) > 1:
                shared_candidates.append((len(dict_ids), word))

            tags: set[str] = set()
            for dict_id in dict_ids:
                dict_tags = dict_domain_tags.get(dict_id, set())
                if not dict_tags:
                    dict_tags = {DEFAULT_DOMAIN}
                tags.update(dict_tags)
                for tag in dict_tags:
                    domain_word_map.setdefault(tag, set()).add(word)
                    domain_dicts.setdefault(tag, set()).add(dict_id)
                    available_tags.add(tag)

            word_sources.append(
                WordSource(
                    word=word,
                    dictionary_ids=dict_ids,
                    domain_tags=sorted(tags),
                )
            )

            if len(tags) > 1:
                bridge_candidates.append(
                    DomainBridgeWord(
                        word=word,
                        domain_tags=sorted(tags),
                        dictionary_ids=dict_ids,
                    )
                )
                for tag in tags:
                    bridge_counts[tag] = bridge_counts.get(tag, 0) + 1

        shared_candidates.sort(key=lambda item: (-item[0], item[1]))
        shared_words = [word for _, word in shared_candidates]
        if max_shared_words > 0 and len(shared_words) > max_shared_words:
            truncated = True
            shared_words = shared_words[:max_shared_words]

        word_sources.sort(key=lambda item: (-len(item.dictionary_ids), item.word))
        if max_word_sources > 0 and len(word_sources) > max_word_sources:
            truncated = True
            word_sources = word_sources[:max_word_sources]

        missing_words = sorted(word_set - union_words)
        if max_missing_words > 0 and len(missing_words) > max_missing_words:
            truncated = True
            missing_words = missing_words[:max_missing_words]

        bridge_candidates.sort(key=lambda item: (-len(item.domain_tags), item.word))
        if max_bridge_words > 0 and len(bridge_candidates) > max_bridge_words:
            truncated = True
            bridge_candidates = bridge_candidates[:max_bridge_words]

        definition_checks = self._build_definition_checks(
            shared_words,
            word_index,
            definition_similarity,
            max_definition_checks,
            max_definition_chars,
        )
        if max_definition_checks > 0 and len(definition_checks) > max_definition_checks:
            truncated = True
            definition_checks = definition_checks[:max_definition_checks]

        domain_usages = self._build_domain_usages(
            domain_word_map,
            domain_dicts,
            len(word_set),
            max_sample_words,
        )
        signals = self._build_context_signals(word_freq)
        recommendations = self._recommend_domains(
            domain_usages,
            signals,
            available_tags,
            bridge_counts,
            max_domain_recommendations,
            domain_marker_boost,
            bridge_boost,
        )

        preview = self._preview_text(chunk_text, preview_chars)
        return (
            ChunkAnalysis(
                item_index=0,
                item_id=None,
                chunk_index=0,
                preview=preview,
                word_count=word_count,
                unique_words=len(word_set),
                dictionary_usages=dictionary_usages,
                domain_usages=domain_usages,
                domain_recommendations=recommendations,
                bridge_words=bridge_candidates,
                shared_words=shared_words,
                missing_words=missing_words,
                word_sources=word_sources,
                context_signals=signals,
                definition_checks=definition_checks,
                truncated=truncated,
            ),
            truncated,
        )

    def _build_domain_usages(
        self,
        domain_word_map: dict[str, set[str]],
        domain_dicts: dict[str, set[str]],
        total_words: int,
        max_sample_words: int,
    ) -> list[DomainUsage]:
        usages: list[DomainUsage] = []
        for tag, words in domain_word_map.items():
            matched = sorted(words)
            coverage = len(words) / total_words if total_words else 0.0
            sample = matched[:max_sample_words] if max_sample_words > 0 else []
            usages.append(
                DomainUsage(
                    tag=tag,
                    matched_words=len(words),
                    coverage=round(coverage, 6),
                    dictionary_ids=sorted(domain_dicts.get(tag, set())),
                    sample_words=sample,
                )
            )
        usages.sort(key=lambda item: (-item.coverage, item.tag))
        return usages

    def _build_context_signals(self, word_freq: dict[str, int]) -> ContextSignals:
        it_markers = sum(word_freq.get(word, 0) for word in DOMAIN_MARKERS["it"])
        game_markers = sum(word_freq.get(word, 0) for word in DOMAIN_MARKERS["game_design"])
        hardware_markers = sum(word_freq.get(word, 0) for word in DOMAIN_MARKERS["hardware"])
        medical_markers = sum(word_freq.get(word, 0) for word in DOMAIN_MARKERS["medical"])

        return ContextSignals(
            it_markers=it_markers,
            game_design_markers=game_markers,
            hardware_markers=hardware_markers,
            medical_markers=medical_markers,
        )

    def _base_domain(self, tag: str) -> str | None:
        if tag in PRIMARY_DOMAINS:
            return tag
        for base in PRIMARY_DOMAINS:
            if tag.startswith(f"{base}_"):
                return base
        return None

    def _recommend_domains(
        self,
        domain_usages: list[DomainUsage],
        signals: ContextSignals,
        available_tags: set[str],
        bridge_counts: dict[str, int],
        max_domain_recommendations: int,
        domain_marker_boost: float,
        bridge_boost: float,
    ) -> list[DomainRecommendation]:
        scores: dict[str, float] = {tag: 0.0 for tag in available_tags}
        reasons: dict[str, list[str]] = {tag: [] for tag in available_tags}
        for usage in domain_usages:
            scores[usage.tag] = usage.coverage
            reasons[usage.tag] = [f"coverage:{usage.coverage:.3f}"]

        marker_counts = {
            "it": signals.it_markers,
            "game_design": signals.game_design_markers,
            "hardware": signals.hardware_markers,
            "medical": signals.medical_markers,
        }
        for tag in available_tags:
            base = self._base_domain(tag)
            if base and marker_counts.get(base, 0) > 0:
                scores[tag] += domain_marker_boost
                reasons[tag].append(f"{base} markers")

        for tag, count in bridge_counts.items():
            if count <= 0:
                continue
            if tag not in scores:
                scores[tag] = 0.0
                reasons[tag] = []
            scores[tag] += bridge_boost
            reasons[tag].append("bridge terms")

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        recommendations: list[DomainRecommendation] = []
        for tag, score in ranked:
            usage = next((u for u in domain_usages if u.tag == tag), None)
            coverage = usage.coverage if usage else 0.0
            recommendations.append(
                DomainRecommendation(
                    tag=tag,
                    score=round(score, 6),
                    coverage=coverage,
                    reasons=reasons.get(tag, []),
                )
            )
            if max_domain_recommendations > 0 and len(recommendations) >= max_domain_recommendations:
                break

        return recommendations

    def _build_definition_checks(
        self,
        shared_words: list[str],
        word_index: dict[str, list[_DictEntry]],
        definition_similarity: float,
        max_definition_checks: int,
        max_definition_chars: int,
    ) -> list[DefinitionCheck]:
        checks: list[DefinitionCheck] = []
        for word in shared_words:
            if max_definition_checks > 0 and len(checks) >= max_definition_checks:
                break
            entries = word_index.get(word, [])
            definitions: dict[str, str] = {}
            missing: list[str] = []
            for entry in entries:
                if entry.definition:
                    definitions[entry.dict_id] = self._trim_definition(
                        entry.definition, max_definition_chars
                    )
                else:
                    missing.append(entry.dict_id)

            notes: list[str] = []
            if missing:
                notes.append("missing definitions for: " + ", ".join(sorted(set(missing))))

            if len(definitions) < 2:
                checks.append(
                    DefinitionCheck(
                        word=word,
                        dictionary_ids=sorted({entry.dict_id for entry in entries}),
                        average_similarity=0.0,
                        status="insufficient",
                        notes=notes,
                        definitions=definitions,
                    )
                )
                continue

            avg_similarity = self._definition_similarity(list(definitions.values()))
            status = "consistent" if avg_similarity >= definition_similarity else "divergent"
            if status == "divergent":
                notes.append("low definition overlap")

            checks.append(
                DefinitionCheck(
                    word=word,
                    dictionary_ids=sorted(definitions.keys()),
                    average_similarity=round(avg_similarity, 6),
                    status=status,
                    notes=notes,
                    definitions=definitions,
                )
            )

        return checks

    def _trim_definition(self, definition: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        text = definition.strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def _definition_similarity(self, definitions: list[str]) -> float:
        if len(definitions) < 2:
            return 0.0
        sets = [self._definition_tokens(defn) for defn in definitions]
        pairs = 0
        total = 0.0
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                left = sets[i]
                right = sets[j]
                if not left or not right:
                    continue
                overlap = len(left & right)
                union = len(left | right)
                similarity = overlap / union if union else 0.0
                total += similarity
                pairs += 1
        if pairs == 0:
            return 0.0
        return total / pairs

    def _definition_tokens(self, text: str) -> set[str]:
        tokens = {
            token.lower()
            for token in WORD_RE.findall(text)
            if len(token) >= 3 and token.lower() not in STOPWORDS
        }
        return tokens

    def _build_dictionary_overlaps(
        self,
        dict_word_sets: dict[str, set[str]],
        max_dictionary_overlaps: int,
    ) -> tuple[list[DictionaryOverlap], bool]:
        dict_ids = sorted(dict_word_sets.keys())
        overlaps: list[DictionaryOverlap] = []
        for i in range(len(dict_ids)):
            for j in range(i + 1, len(dict_ids)):
                left_id = dict_ids[i]
                right_id = dict_ids[j]
                left_words = dict_word_sets[left_id]
                right_words = dict_word_sets[right_id]
                shared = len(left_words & right_words)
                if shared == 0:
                    continue
                union = len(left_words | right_words)
                similarity = shared / union if union else 0.0
                overlaps.append(
                    DictionaryOverlap(
                        left_id=left_id,
                        right_id=right_id,
                        shared_words=shared,
                        similarity=round(similarity, 6),
                    )
                )

        overlaps.sort(key=lambda item: (-item.similarity, -item.shared_words, item.left_id))
        truncated = False
        if max_dictionary_overlaps > 0 and len(overlaps) > max_dictionary_overlaps:
            truncated = True
            overlaps = overlaps[:max_dictionary_overlaps]
        return overlaps, truncated

    def _build_domain_overlaps(
        self,
        dict_word_sets: dict[str, set[str]],
        dict_domain_tags: dict[str, set[str]],
        max_domain_overlaps: int,
    ) -> tuple[list[DomainOverlap], bool]:
        domain_word_sets: dict[str, set[str]] = {}
        for dict_id, tags in dict_domain_tags.items():
            words = dict_word_sets.get(dict_id, set())
            if not tags:
                domain_word_sets.setdefault(DEFAULT_DOMAIN, set()).update(words)
                continue
            for tag in tags:
                domain_word_sets.setdefault(tag, set()).update(words)

        tags = sorted(domain_word_sets.keys())
        overlaps: list[DomainOverlap] = []
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                left_tag = tags[i]
                right_tag = tags[j]
                left_words = domain_word_sets[left_tag]
                right_words = domain_word_sets[right_tag]
                shared = len(left_words & right_words)
                if shared == 0:
                    continue
                union = len(left_words | right_words)
                similarity = shared / union if union else 0.0
                overlaps.append(
                    DomainOverlap(
                        left_tag=left_tag,
                        right_tag=right_tag,
                        shared_words=shared,
                        similarity=round(similarity, 6),
                    )
                )

        overlaps.sort(key=lambda item: (-item.similarity, -item.shared_words, item.left_tag))
        truncated = False
        if max_domain_overlaps > 0 and len(overlaps) > max_domain_overlaps:
            truncated = True
            overlaps = overlaps[:max_domain_overlaps]
        return overlaps, truncated

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextDictionaryDomainsArgs):
            return ToolCallDisplay(summary="context_dictionary_domains")

        summary = f"context_dictionary_domains: {len(event.args.items)} item(s)"
        return ToolCallDisplay(
            summary=summary,
            details={
                "item_count": len(event.args.items),
                "dictionary_count": len(event.args.dictionaries),
                "chunk_mode": event.args.chunk_mode,
                "chunk_size": event.args.chunk_size,
                "definition_similarity": event.args.definition_similarity,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextDictionaryDomainsResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = (
            f"Processed {event.result.chunk_count} chunk(s) across "
            f"{event.result.dictionary_count} dictionaries"
        )
        warnings = event.result.errors[:]
        if event.result.truncated:
            warnings.append("Output truncated by size or limits")

        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=warnings,
            details={
                "dictionary_count": event.result.dictionary_count,
                "chunk_count": event.result.chunk_count,
                "overlap_count": event.result.overlap_count,
                "domain_overlap_count": event.result.domain_overlap_count,
                "truncated": event.result.truncated,
                "errors": event.result.errors,
                "dictionaries": event.result.dictionaries,
                "chunks": event.result.chunks,
                "dictionary_overlaps": event.result.dictionary_overlaps,
                "domain_overlaps": event.result.domain_overlaps,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Analyzing domain dictionaries"
