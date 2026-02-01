
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from array import array
from pathlib import Path
import csv
import json
import math
import re
import sqlite3
from typing import TYPE_CHECKING, ClassVar
from urllib import request

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
SUPPORTED_ACTIONS = {
    "add",
    "analyze",
    "list",
    "remove",
    "reset",
    "update",
}


@dataclass
class _StoredDictionary:
    dict_id: str
    name: str | None
    tags: list[str]
    markers: set[str]
    priority: int
    source: str | None
    created_at: str
    updated_at: str
    entries: dict[str, str | None]
    entry_count: int
    definition_count: int
    usage_count: int
    total_hits: int
    last_used_at: str | None
    persistent: bool = True


class ContextDictionaryMemoryConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    profile_path: Path = Field(
        default=Path.home() / ".vibe" / "memory" / "custom_dictionary_memory.json",
        description="Path to the dictionary memory profile.",
    )
    default_dictionaries: list[dict] = Field(
        default_factory=list,
        description="Dictionary sources always loaded for analyze.",
    )
    default_dictionary_priority: int = Field(
        default=0, description="Default priority for dictionaries without explicit priority."
    )
    modern_dictionary_tags: list[str] = Field(
        default_factory=lambda: ["modern", "modern_english", "standard"],
        description="Tags that mark a dictionary as modern/standard.",
    )
    modern_dictionary_priority: int = Field(
        default=-1, description="Priority for modern dictionaries when not specified."
    )
    ollama_url: str = Field(
        default="http://127.0.0.1:11434",
        description="Base URL for the Ollama/GPT-OSS server.",
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        description="Embedding model to use with Ollama/GPT-OSS.",
    )
    max_embedding_chars: int = Field(
        default=8000, description="Maximum characters embedded per document."
    )
    max_vector_docs: int = Field(
        default=200, description="Maximum documents loaded from vector stores."
    )
    max_vector_chunks_per_doc: int = Field(
        default=200, description="Maximum chunks per document from vector stores."
    )
    max_neighbors: int = Field(
        default=5, description="Maximum embedding neighbors per document."
    )
    min_similarity: float = Field(
        default=0.1, description="Minimum cosine similarity for neighbors."
    )
    max_dictionaries: int = Field(
        default=500, description="Maximum dictionaries stored in the profile."
    )
    max_dictionary_bytes: int = Field(
        default=5_000_000, description="Maximum size per dictionary source (bytes)."
    )
    max_dictionary_entries: int = Field(
        default=200_000, description="Maximum entries per dictionary source."
    )
    max_profile_bytes: int = Field(
        default=50_000_000, description="Maximum serialized profile size (bytes)."
    )
    max_items: int = Field(default=200, description="Maximum items to process.")
    max_source_bytes: int = Field(
        default=5_000_000, description="Maximum size per item (bytes)."
    )
    max_total_bytes: int = Field(
        default=20_000_000, description="Maximum total bytes across items."
    )
    max_chunks: int = Field(
        default=500, description="Maximum chunks returned across all items."
    )
    preview_chars: int = Field(default=400, description="Preview length per chunk.")
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
    max_definition_chars: int = Field(
        default=200, description="Maximum characters stored per definition snippet."
    )
    definition_min_token_length: int = Field(
        default=3, description="Minimum token length for definition matching."
    )
    definition_weight: float = Field(
        default=1.0, description="Weight for definition-context matching."
    )
    priority_weight: float = Field(
        default=0.2, description="Score boost per dictionary priority step."
    )
    size_penalty: float = Field(
        default=1.0, description="Penalty exponent for dictionary size normalization."
    )
    marker_boost: float = Field(
        default=0.1, description="Score boost per marker hit."
    )
    usage_boost: float = Field(
        default=0.01, description="Score boost per log(usage_count + 1)."
    )


class ContextDictionaryMemoryState(BaseToolState):
    pass


class DictionaryEntry(BaseModel):
    word: str = Field(description="Dictionary word.")
    definition: str | None = Field(default=None, description="Optional definition text.")


class DictionarySource(BaseModel):
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
    tags: list[str] | None = Field(
        default=None, description="Optional tags describing the dictionary."
    )
    markers: list[str] | None = Field(
        default=None, description="Optional marker words for scoring."
    )
    priority: int | None = Field(
        default=None, description="Optional priority for overlap resolution."
    )
    source: str | None = Field(default=None, description="Source description.")


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


class VectorStore(BaseModel):
    path: str = Field(description="Path to a vector DB (sqlite) or manifest file.")
    store_type: str | None = Field(
        default="auto", description="auto, text, or pdf."
    )
    table: str | None = Field(
        default=None, description="Override table name for chunks."
    )
    label: str | None = Field(default=None, description="Optional store label.")
    max_docs: int | None = Field(
        default=None, description="Override max docs for this store."
    )
    max_chunks_per_doc: int | None = Field(
        default=None, description="Override max chunks per document."
    )


class ContextDictionaryMemoryArgs(BaseModel):
    action: str | None = Field(
        default="analyze", description="add, update, remove, list, analyze, reset."
    )
    dictionaries: list[DictionarySource] | None = Field(
        default=None, description="Dictionary sources to ingest."
    )
    dictionary_ids: list[str] | None = Field(
        default=None, description="Dictionary ids to use or remove."
    )
    items: list[TextItem] | None = Field(
        default=None, description="Text items to analyze."
    )
    vector_stores: list[VectorStore] | None = Field(
        default=None, description="Vector stores containing text documents."
    )
    use_embeddings: bool | None = Field(
        default=None, description="Use embeddings to link documents."
    )
    profile_path: str | None = Field(
        default=None, description="Override profile path."
    )
    persist: bool | None = Field(
        default=None, description="Persist profile changes."
    )
    persist_dictionaries: bool | None = Field(
        default=None, description="Persist dictionaries provided with analyze."
    )
    replace_existing: bool | None = Field(
        default=None, description="Replace existing dictionaries on conflict."
    )
    update_stats: bool | None = Field(
        default=None, description="Update usage stats after analyze."
    )
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
    max_profile_bytes: int | None = Field(
        default=None, description="Override max_profile_bytes."
    )
    max_vector_docs: int | None = Field(
        default=None, description="Override max_vector_docs."
    )
    max_vector_chunks_per_doc: int | None = Field(
        default=None, description="Override max_vector_chunks_per_doc."
    )
    max_neighbors: int | None = Field(
        default=None, description="Override max_neighbors."
    )
    min_similarity: float | None = Field(
        default=None, description="Override min_similarity."
    )
    embedding_model: str | None = Field(
        default=None, description="Override embedding model."
    )
    ollama_url: str | None = Field(
        default=None, description="Override Ollama/GPT-OSS URL."
    )
    max_embedding_chars: int | None = Field(
        default=None, description="Override max_embedding_chars."
    )
    max_dictionaries: int | None = Field(
        default=None, description="Override max_dictionaries."
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
    max_definition_chars: int | None = Field(
        default=None, description="Override max_definition_chars."
    )
    definition_min_token_length: int | None = Field(
        default=None, description="Override definition_min_token_length."
    )
    definition_weight: float | None = Field(
        default=None, description="Override definition_weight."
    )
    priority_weight: float | None = Field(
        default=None, description="Override priority_weight."
    )
    size_penalty: float | None = Field(
        default=None, description="Override size_penalty."
    )
    marker_boost: float | None = Field(
        default=None, description="Override marker_boost."
    )
    usage_boost: float | None = Field(
        default=None, description="Override usage_boost."
    )

class DictionaryProfileSummary(BaseModel):
    dictionary_id: str
    name: str | None
    tags: list[str]
    priority: int
    entry_count: int
    definition_count: int
    usage_count: int
    total_hits: int
    last_used_at: str | None
    source: str | None
    persistent: bool


class DictionaryScore(BaseModel):
    dictionary_id: str
    name: str | None
    matched_words: int
    coverage: float
    score: float
    sample_words: list[str]


class WordSource(BaseModel):
    word: str
    dictionary_ids: list[str]


class AmbiguousTerm(BaseModel):
    word: str
    dictionary_ids: list[str]
    resolved_dictionary_id: str | None
    resolved_reason: str | None
    definitions: dict[str, str]
    definition_scores: dict[str, float]


class DictionaryUsageSummary(BaseModel):
    dictionary_id: str
    name: str | None
    matched_words: int
    chunks: int
    coverage: float


class DictionaryDocumentUsage(BaseModel):
    dictionary_id: str
    name: str | None
    documents: int
    matched_words: int


class DocumentNeighbor(BaseModel):
    document_id: str
    similarity: float


class DocumentUsage(BaseModel):
    document_id: str
    name: str | None
    source_path: str | None
    chunk_count: int
    dictionary_usage: list[DictionaryUsageSummary]
    neighbors: list[DocumentNeighbor]


class ChunkAnalysis(BaseModel):
    item_index: int
    item_id: str | None
    chunk_index: int
    preview: str
    word_count: int
    unique_words: int
    dictionary_scores: list[DictionaryScore]
    shared_words: list[str]
    missing_words: list[str]
    word_sources: list[WordSource]
    ambiguous_terms: list[AmbiguousTerm]
    truncated: bool


class ContextDictionaryMemoryResult(BaseModel):
    action: str
    dictionaries: list[DictionaryProfileSummary]
    chunks: list[ChunkAnalysis]
    dictionary_usage: list[DictionaryUsageSummary]
    dictionary_documents: list[DictionaryDocumentUsage]
    document_usage: list[DocumentUsage]
    dictionary_count: int
    chunk_count: int
    truncated: bool
    updated: bool
    embedding_used: bool
    errors: list[str]
    warnings: list[str]


class ContextDictionaryMemory(
    BaseTool[
        ContextDictionaryMemoryArgs,
        ContextDictionaryMemoryResult,
        ContextDictionaryMemoryConfig,
        ContextDictionaryMemoryState,
    ],
    ToolUIData[ContextDictionaryMemoryArgs, ContextDictionaryMemoryResult],
):
    description: ClassVar[str] = (
        "Persist custom dictionaries over time and analyze text against them."
    )

    async def run(
        self, args: ContextDictionaryMemoryArgs
    ) -> ContextDictionaryMemoryResult:
        action = (args.action or "analyze").strip().lower()
        if action == "ingest":
            action = "add"
        if action == "delete":
            action = "remove"
        if action not in SUPPORTED_ACTIONS:
            raise ToolError(
                "action must be add, update, remove, list, analyze, or reset."
            )

        max_items = args.max_items if args.max_items is not None else self.config.max_items
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
        max_profile_bytes = (
            args.max_profile_bytes
            if args.max_profile_bytes is not None
            else self.config.max_profile_bytes
        )
        max_dictionaries = (
            args.max_dictionaries
            if args.max_dictionaries is not None
            else self.config.max_dictionaries
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
        max_definition_chars = (
            args.max_definition_chars
            if args.max_definition_chars is not None
            else self.config.max_definition_chars
        )
        definition_min_token_length = (
            args.definition_min_token_length
            if args.definition_min_token_length is not None
            else self.config.definition_min_token_length
        )
        definition_weight = (
            args.definition_weight
            if args.definition_weight is not None
            else self.config.definition_weight
        )
        priority_weight = (
            args.priority_weight
            if args.priority_weight is not None
            else self.config.priority_weight
        )
        size_penalty = (
            args.size_penalty if args.size_penalty is not None else self.config.size_penalty
        )
        marker_boost = (
            args.marker_boost
            if args.marker_boost is not None
            else self.config.marker_boost
        )
        usage_boost = (
            args.usage_boost if args.usage_boost is not None else self.config.usage_boost
        )

        if max_items <= 0:
            raise ToolError("max_items must be a positive integer.")
        if max_dictionary_bytes <= 0:
            raise ToolError("max_dictionary_bytes must be a positive integer.")
        if max_dictionary_entries <= 0:
            raise ToolError("max_dictionary_entries must be a positive integer.")
        if max_profile_bytes <= 0:
            raise ToolError("max_profile_bytes must be a positive integer.")
        if max_dictionaries <= 0:
            raise ToolError("max_dictionaries must be a positive integer.")
        if max_chunks <= 0:
            raise ToolError("max_chunks must be a positive integer.")
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
        if max_definition_chars < 0:
            raise ToolError("max_definition_chars must be >= 0.")
        if definition_min_token_length <= 0:
            raise ToolError("definition_min_token_length must be > 0.")
        if definition_weight < 0:
            raise ToolError("definition_weight must be >= 0.")
        if priority_weight < 0:
            raise ToolError("priority_weight must be >= 0.")
        if size_penalty <= 0:
            raise ToolError("size_penalty must be > 0.")

        profile_path = (
            Path(args.profile_path).expanduser()
            if args.profile_path
            else self.config.profile_path
        )

        persist = True if args.persist is None else bool(args.persist)
        replace_existing = True if args.replace_existing is None else bool(args.replace_existing)
        update_stats = True if args.update_stats is None else bool(args.update_stats)
        persist_dictionaries = (
            False
            if args.persist_dictionaries is None
            else bool(args.persist_dictionaries)
        )

        errors: list[str] = []
        warnings: list[str] = []
        stored = self._load_profile(profile_path, errors)

        updated = False
        if action in {"add", "update"}:
            if not args.dictionaries:
                raise ToolError("dictionaries is required for add/update.")
            updated = self._ingest_dictionaries(
                args.dictionaries,
                stored,
                replace_existing,
                max_dictionary_bytes,
                max_dictionary_entries,
                min_word_length,
                max_dictionaries,
                warnings,
                errors,
            )
        elif action == "remove":
            if not args.dictionary_ids:
                raise ToolError("dictionary_ids is required for remove.")
            removed = 0
            for dict_id in args.dictionary_ids:
                if dict_id in stored:
                    stored.pop(dict_id)
                    removed += 1
                else:
                    warnings.append(f"dictionary not found: {dict_id}")
            updated = removed > 0
        elif action == "reset":
            if stored:
                stored.clear()
                updated = True
        elif action == "analyze":
            if not args.items and not args.vector_stores:
                raise ToolError("items or vector_stores is required for analyze.")

            items: list[TextItem] = list(args.items or [])
            vector_embeddings: dict[str, list[float]] = {}
            vector_sources: dict[str, str] = {}
            use_embeddings = (
                bool(args.vector_stores)
                if args.use_embeddings is None
                else bool(args.use_embeddings)
            )
            embedding_model = args.embedding_model or self.config.embedding_model
            ollama_url = args.ollama_url or self.config.ollama_url
            max_embedding_chars = (
                args.max_embedding_chars
                if args.max_embedding_chars is not None
                else self.config.max_embedding_chars
            )
            max_neighbors = (
                args.max_neighbors
                if args.max_neighbors is not None
                else self.config.max_neighbors
            )
            min_similarity = (
                args.min_similarity
                if args.min_similarity is not None
                else self.config.min_similarity
            )
            if args.vector_stores:
                store_items, store_embeddings, store_sources = self._load_vector_documents(
                    args.vector_stores,
                    args.max_vector_docs if args.max_vector_docs is not None else self.config.max_vector_docs,
                    args.max_vector_chunks_per_doc
                    if args.max_vector_chunks_per_doc is not None
                    else self.config.max_vector_chunks_per_doc,
                    max_source_bytes,
                    max_total_bytes,
                    use_embeddings,
                    errors,
                    warnings,
                )
                items.extend(store_items)
                vector_embeddings.update(store_embeddings)
                vector_sources.update(store_sources)

            if not items:
                raise ToolError("No items available for analyze.")
            if len(items) > max_items:
                raise ToolError(f"items exceeds max_items ({len(items)} > {max_items}).")

            ephemeral: dict[str, _StoredDictionary] = {}
            used_ids = set(stored)
            if args.dictionaries:
                if persist_dictionaries:
                    updated = self._ingest_dictionaries(
                        args.dictionaries,
                        stored,
                        replace_existing,
                        max_dictionary_bytes,
                        max_dictionary_entries,
                        min_word_length,
                        max_dictionaries,
                        warnings,
                        errors,
                    )
                    used_ids = set(stored)
                else:
                    ephemeral = self._build_ephemeral_dictionaries(
                        args.dictionaries,
                        max_dictionary_bytes,
                        max_dictionary_entries,
                        min_word_length,
                        warnings,
                        errors,
                        used_ids=used_ids,
                    )

            default_sources = self._coerce_config_dictionaries(
                self.config.default_dictionaries, warnings, errors
            )
            default_ephemeral: dict[str, _StoredDictionary] = {}
            if default_sources:
                default_ephemeral = self._build_ephemeral_dictionaries(
                    default_sources,
                    max_dictionary_bytes,
                    max_dictionary_entries,
                    min_word_length,
                    warnings,
                    errors,
                    used_ids=used_ids,
                    skip_existing=True,
                )

            active = {**stored, **ephemeral, **default_ephemeral}
            if args.dictionary_ids:
                requested = [dict_id for dict_id in args.dictionary_ids if dict_id in active]
                if not requested:
                    raise ToolError("dictionary_ids did not match any loaded dictionaries.")
                active = {dict_id: active[dict_id] for dict_id in requested}

            if not active:
                raise ToolError("no dictionaries available for analyze.")

            chunk_mode_default = (args.chunk_mode or self.config.default_chunk_mode).strip().lower()
            if chunk_mode_default not in SUPPORTED_CHUNK_MODES:
                raise ToolError("chunk_mode must be words, paragraphs, or sentences.")

            total_bytes = 0
            chunks: list[ChunkAnalysis] = []
            chunk_count = 0
            truncated = False
            total_unique_words = 0
            doc_text_samples: dict[str, str] = {}
            doc_sources: dict[str, str | None] = {}
            doc_names: dict[str, str | None] = {}
            global_hits: dict[str, int] = {dict_id: 0 for dict_id in active}
            global_chunks: dict[str, int] = {dict_id: 0 for dict_id in active}

            dict_word_sets = {dict_id: set(d.entries.keys()) for dict_id, d in active.items()}
            word_index = self._build_word_index(dict_word_sets)
            union_words = set()
            for words in dict_word_sets.values():
                union_words |= words

            for item_index, item in enumerate(items, start=1):
                if chunk_count >= max_chunks:
                    truncated = True
                    break
                content, resolved_path, size = self._load_item_content(item, max_source_bytes)
                if content is None:
                    errors.append(f"item[{item_index}] content empty")
                    continue
                if size:
                    total_bytes += size
                if total_bytes > max_total_bytes:
                    errors.append("max_total_bytes exceeded.")
                    truncated = True
                    break

                doc_id = item.id or resolved_path or item.name or f"item_{item_index}"
                doc_sources.setdefault(doc_id, resolved_path or vector_sources.get(doc_id))
                doc_names.setdefault(doc_id, item.name)
                if use_embeddings and doc_id not in vector_embeddings:
                    if doc_id not in doc_text_samples:
                        doc_text_samples[doc_id] = content[:max_embedding_chars]

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
                    analysis, was_truncated, matched_counts = self._analyze_chunk(
                        chunk_text,
                        word_index,
                        dict_word_sets,
                        active,
                        union_words,
                        max_words_per_chunk,
                        min_word_length,
                        max_word_sources,
                        max_missing_words,
                        max_shared_words,
                        max_sample_words,
                        max_definition_chars,
                        definition_min_token_length,
                        definition_weight,
                        priority_weight,
                        size_penalty,
                        marker_boost,
                        usage_boost,
                        preview_chars,
                    )
                    analysis.item_index = item_index
                    analysis.item_id = doc_id
                    analysis.chunk_index = local_idx
                    chunks.append(analysis)
                    chunk_count += 1
                    total_unique_words += analysis.unique_words
                    for dict_id, matched in matched_counts.items():
                        if matched <= 0:
                            continue
                        global_hits[dict_id] = global_hits.get(dict_id, 0) + matched
                        global_chunks[dict_id] = global_chunks.get(dict_id, 0) + 1
                    if was_truncated:
                        truncated = True
                if chunk_count >= max_chunks:
                    truncated = True
                    break

            if update_stats:
                now = self._now_iso()
                for dict_id, hits in global_hits.items():
                    if hits <= 0 or dict_id not in stored:
                        continue
                    entry = stored[dict_id]
                    entry.usage_count += 1
                    entry.total_hits += hits
                    entry.last_used_at = now
                    entry.updated_at = now
                    updated = True

            usage_summary: list[DictionaryUsageSummary] = []
            for dict_id, entry in active.items():
                matched = global_hits.get(dict_id, 0)
                chunks_hit = global_chunks.get(dict_id, 0)
                coverage = matched / total_unique_words if total_unique_words else 0.0
                usage_summary.append(
                    DictionaryUsageSummary(
                        dictionary_id=dict_id,
                        name=entry.name,
                        matched_words=matched,
                        chunks=chunks_hit,
                        coverage=round(coverage, 6),
                    )
                )
            usage_summary.sort(key=lambda item: (-item.matched_words, item.dictionary_id))

            doc_stats: dict[str, dict[str, object]] = {}
            for chunk in chunks:
                doc_id = chunk.item_id or f"item_{chunk.item_index}"
                stats = doc_stats.setdefault(
                    doc_id,
                    {
                        "unique_words": 0,
                        "chunk_count": 0,
                        "dict_hits": {},
                        "dict_chunks": {},
                    },
                )
                stats["unique_words"] = int(stats["unique_words"]) + chunk.unique_words
                stats["chunk_count"] = int(stats["chunk_count"]) + 1
                dict_hits = stats["dict_hits"]
                dict_chunks = stats["dict_chunks"]
                if isinstance(dict_hits, dict) and isinstance(dict_chunks, dict):
                    for score in chunk.dictionary_scores:
                        if score.matched_words <= 0:
                            continue
                        dict_hits[score.dictionary_id] = (
                            int(dict_hits.get(score.dictionary_id, 0)) + score.matched_words
                        )
                        dict_chunks[score.dictionary_id] = (
                            int(dict_chunks.get(score.dictionary_id, 0)) + 1
                        )

            doc_embeddings = dict(vector_embeddings)
            embedding_used = False
            if use_embeddings:
                for doc_id, text in doc_text_samples.items():
                    if doc_id in doc_embeddings:
                        continue
                    if max_embedding_chars <= 0:
                        break
                    try:
                        doc_embeddings[doc_id] = self._embed_text(
                            ollama_url, embedding_model, text[:max_embedding_chars]
                        )
                    except ToolError as exc:
                        warnings.append(str(exc))
                embedding_used = bool(doc_embeddings)

            neighbors_map: dict[str, list[DocumentNeighbor]] = {}
            if embedding_used and max_neighbors > 0:
                neighbors_map = self._build_neighbors(
                    doc_embeddings, max_neighbors, min_similarity
                )

            document_usage: list[DocumentUsage] = []
            for doc_id, stats in doc_stats.items():
                dict_hits = stats.get("dict_hits", {})
                dict_chunks = stats.get("dict_chunks", {})
                dict_usage: list[DictionaryUsageSummary] = []
                total_unique = int(stats.get("unique_words", 0))
                for dict_id, entry in active.items():
                    matched = int(dict_hits.get(dict_id, 0)) if isinstance(dict_hits, dict) else 0
                    chunks_hit = int(dict_chunks.get(dict_id, 0)) if isinstance(dict_chunks, dict) else 0
                    if matched <= 0:
                        continue
                    coverage = matched / total_unique if total_unique else 0.0
                    dict_usage.append(
                        DictionaryUsageSummary(
                            dictionary_id=dict_id,
                            name=entry.name,
                            matched_words=matched,
                            chunks=chunks_hit,
                            coverage=round(coverage, 6),
                        )
                    )
                dict_usage.sort(key=lambda item: (-item.matched_words, item.dictionary_id))
                document_usage.append(
                    DocumentUsage(
                        document_id=doc_id,
                        name=doc_names.get(doc_id),
                        source_path=doc_sources.get(doc_id),
                        chunk_count=int(stats.get("chunk_count", 0)),
                        dictionary_usage=dict_usage,
                        neighbors=neighbors_map.get(doc_id, []),
                    )
                )
            document_usage.sort(key=lambda item: item.document_id)

            dictionary_documents: list[DictionaryDocumentUsage] = []
            for dict_id, entry in active.items():
                doc_count = 0
                matched_total = 0
                for stats in doc_stats.values():
                    dict_hits = stats.get("dict_hits", {})
                    if not isinstance(dict_hits, dict):
                        continue
                    hits = int(dict_hits.get(dict_id, 0))
                    if hits > 0:
                        doc_count += 1
                        matched_total += hits
                dictionary_documents.append(
                    DictionaryDocumentUsage(
                        dictionary_id=dict_id,
                        name=entry.name,
                        documents=doc_count,
                        matched_words=matched_total,
                    )
                )
            dictionary_documents.sort(
                key=lambda item: (-item.documents, -item.matched_words, item.dictionary_id)
            )

            if updated and persist:
                self._save_profile(profile_path, stored, max_profile_bytes, errors)
            elif updated and not persist:
                warnings.append("persist=false; profile changes not saved.")

            return ContextDictionaryMemoryResult(
                action="analyze",
                dictionaries=self._summarize_dictionaries(active),
                chunks=chunks,
                dictionary_usage=usage_summary,
                dictionary_documents=dictionary_documents,
                document_usage=document_usage,
                dictionary_count=len(active),
                chunk_count=len(chunks),
                truncated=truncated,
                updated=updated and persist,
                embedding_used=embedding_used,
                errors=errors,
                warnings=warnings,
            )

        if updated and persist:
            self._save_profile(profile_path, stored, max_profile_bytes, errors)
        elif updated and not persist:
            warnings.append("persist=false; profile changes not saved.")

        summaries = self._summarize_dictionaries(stored)
        return ContextDictionaryMemoryResult(
            action=action,
            dictionaries=summaries,
            chunks=[],
            dictionary_usage=[],
            dictionary_documents=[],
            document_usage=[],
            dictionary_count=len(summaries),
            chunk_count=0,
            truncated=False,
            updated=updated and persist,
            embedding_used=False,
            errors=errors,
            warnings=warnings,
        )

    def _ingest_dictionaries(
        self,
        sources: list[DictionarySource],
        stored: dict[str, _StoredDictionary],
        replace_existing: bool,
        max_dictionary_bytes: int,
        max_dictionary_entries: int,
        min_word_length: int,
        max_dictionaries: int,
        warnings: list[str],
        errors: list[str],
    ) -> bool:
        updated = False
        used_ids = set(stored.keys())
        for idx, source in enumerate(sources, start=1):
            try:
                entry, truncated = self._build_dictionary(
                    source,
                    idx,
                    used_ids,
                    max_dictionary_bytes,
                    max_dictionary_entries,
                    min_word_length,
                )
                if entry.dict_id in stored and not replace_existing:
                    warnings.append(f"dictionary exists, skipped: {entry.dict_id}")
                    continue

                if entry.dict_id in stored:
                    existing = stored[entry.dict_id]
                    entry.created_at = existing.created_at
                    entry.usage_count = existing.usage_count
                    entry.total_hits = existing.total_hits
                    entry.last_used_at = existing.last_used_at

                stored[entry.dict_id] = entry
                used_ids.add(entry.dict_id)
                updated = True
                if truncated:
                    warnings.append(
                        f"dictionary {entry.dict_id} truncated to max_dictionary_entries"
                    )
            except Exception as exc:
                errors.append(f"dictionary[{idx}]: {exc}")

            if len(stored) > max_dictionaries:
                raise ToolError(
                    f"max_dictionaries exceeded ({len(stored)} > {max_dictionaries})."
                )

        return updated

    def _build_ephemeral_dictionaries(
        self,
        sources: list[DictionarySource],
        max_dictionary_bytes: int,
        max_dictionary_entries: int,
        min_word_length: int,
        warnings: list[str],
        errors: list[str],
        used_ids: set[str] | None = None,
        skip_existing: bool = False,
    ) -> dict[str, _StoredDictionary]:
        ephemeral: dict[str, _StoredDictionary] = {}
        if used_ids is None:
            used_ids = set()
        for idx, source in enumerate(sources, start=1):
            try:
                candidate_id = self._candidate_dict_id(source, idx)
                if skip_existing and candidate_id in used_ids:
                    warnings.append(f"dictionary exists, skipped: {candidate_id}")
                    continue
                entry, truncated = self._build_dictionary(
                    source,
                    idx,
                    used_ids,
                    max_dictionary_bytes,
                    max_dictionary_entries,
                    min_word_length,
                )
                entry.persistent = False
                ephemeral[entry.dict_id] = entry
                used_ids.add(entry.dict_id)
                if truncated:
                    warnings.append(
                        f"dictionary {entry.dict_id} truncated to max_dictionary_entries"
                    )
            except Exception as exc:
                errors.append(f"dictionary[{idx}]: {exc}")
        return ephemeral

    def _build_dictionary(
        self,
        source: DictionarySource,
        index: int,
        used_ids: set[str],
        max_dictionary_bytes: int,
        max_dictionary_entries: int,
        min_word_length: int,
    ) -> tuple[_StoredDictionary, bool]:
        dict_id = self._coerce_dict_id(source, index, used_ids)
        entries, definition_count, truncated = self._load_dictionary_entries(
            source,
            max_dictionary_bytes,
            max_dictionary_entries,
            min_word_length,
        )
        tags = self._normalize_tags(source.tags)
        markers = self._normalize_markers(source.markers, min_word_length)
        priority = self._infer_priority(source, tags)
        now = self._now_iso()
        entry_count = len(entries)
        return (
            _StoredDictionary(
                dict_id=dict_id,
                name=source.name,
                tags=tags,
                markers=markers,
                priority=priority,
                source=source.source,
                created_at=now,
                updated_at=now,
                entries=entries,
                entry_count=entry_count,
                definition_count=definition_count,
                usage_count=0,
                total_hits=0,
                last_used_at=None,
                persistent=True,
            ),
            truncated,
        )

    def _coerce_dict_id(
        self, source: DictionarySource, index: int, used_ids: set[str]
    ) -> str:
        base = self._candidate_dict_id(source, index)
        candidate = base
        suffix = 2
        while candidate in used_ids:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    def _candidate_dict_id(self, source: DictionarySource, index: int) -> str:
        raw = source.id or source.name or f"dictionary_{index}"
        base = re.sub(r"[^0-9A-Za-z_-]+", "_", str(raw).strip().lower()).strip("_")
        if not base:
            base = f"dictionary_{index}"
        return base

    def _normalize_tags(self, tags: list[str] | None) -> list[str]:
        if not tags:
            return []
        normalized: set[str] = set()
        for tag in tags:
            cleaned = re.sub(r"[^0-9A-Za-z_-]+", "_", str(tag).strip().lower()).strip("_")
            if cleaned:
                normalized.add(cleaned)
        return sorted(normalized)

    def _normalize_markers(
        self, markers: list[str] | None, min_word_length: int
    ) -> set[str]:
        if not markers:
            return set()
        normalized: set[str] = set()
        for marker in markers:
            if marker is None:
                continue
            for token in WORD_RE.findall(str(marker).lower()):
                if len(token) >= min_word_length:
                    normalized.add(token)
        return normalized

    def _infer_priority(self, source: DictionarySource, tags: list[str]) -> int:
        if source.priority is not None:
            return int(source.priority)
        modern_tags = set(self._normalize_tags(self.config.modern_dictionary_tags))
        if modern_tags and set(tags) & modern_tags:
            return self.config.modern_dictionary_priority
        name_hint = f"{source.id or ''} {source.name or ''}".lower()
        if "modern" in name_hint or "standard" in name_hint:
            return self.config.modern_dictionary_priority
        return self.config.default_dictionary_priority

    def _infer_priority_from_meta(
        self, dict_id: str, name: str | None, tags: list[str]
    ) -> int:
        modern_tags = set(self._normalize_tags(self.config.modern_dictionary_tags))
        if modern_tags and set(tags) & modern_tags:
            return self.config.modern_dictionary_priority
        name_hint = f"{dict_id} {name or ''}".lower()
        if "modern" in name_hint or "standard" in name_hint:
            return self.config.modern_dictionary_priority
        return self.config.default_dictionary_priority

    def _load_dictionary_entries(
        self,
        source: DictionarySource,
        max_dictionary_bytes: int,
        max_dictionary_entries: int,
        min_word_length: int,
    ) -> tuple[dict[str, str | None], int, bool]:
        raw_entries: list[DictionaryEntry] = []
        if source.entries is not None:
            raw_entries = self._coerce_entries(source.entries)
        else:
            content = self._load_dictionary_content(source, max_dictionary_bytes)
            fmt = (source.format or "auto").strip().lower()
            if fmt not in SUPPORTED_FORMATS:
                raise ToolError(f"Unsupported dictionary format: {fmt}")
            raw_entries = self._parse_dictionary_content(content, fmt, source.separator)

        entries: dict[str, str | None] = {}
        definition_count = 0
        truncated = False
        for entry in raw_entries:
            normalized = self._normalize_word(entry.word, min_word_length)
            if not normalized:
                continue
            definition = self._normalize_definition(entry.definition)
            if normalized in entries:
                if entries[normalized] is None and definition:
                    entries[normalized] = definition
                continue
            entries[normalized] = definition
            if definition:
                definition_count += 1
            if len(entries) >= max_dictionary_entries:
                truncated = True
                break

        if not entries:
            raise ToolError("dictionary contains no valid entries.")

        return entries, definition_count, truncated

    def _load_dictionary_content(
        self, source: DictionarySource, max_dictionary_bytes: int
    ) -> str:
        if source.content and source.path:
            raise ToolError("Provide content or path, not both.")
        if source.content is not None:
            size = len(source.content.encode("utf-8"))
            if size > max_dictionary_bytes:
                raise ToolError(
                    f"Dictionary content exceeds max_dictionary_bytes ({size} > {max_dictionary_bytes})."
                )
            return source.content

        if source.path:
            path = Path(source.path).expanduser()
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

        raise ToolError("Dictionary source must include content, path, or entries.")

    def _coerce_entries(
        self, entries: list[DictionaryEntry] | dict[str, str] | list[dict]
    ) -> list[DictionaryEntry]:
        if isinstance(entries, dict):
            output: list[DictionaryEntry] = []
            for word, definition in entries.items():
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
            else:
                output.append(DictionaryEntry(word=str(item), definition=None))
        return output

    def _coerce_config_dictionaries(
        self, sources: list[dict], warnings: list[str], errors: list[str]
    ) -> list[DictionarySource]:
        output: list[DictionarySource] = []
        for idx, raw in enumerate(sources, start=1):
            if not isinstance(raw, dict):
                warnings.append(f"default_dictionaries[{idx}] ignored: not a table")
                continue
            try:
                output.append(DictionarySource.model_validate(raw))
            except Exception as exc:
                errors.append(f"default_dictionaries[{idx}]: {exc}")
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
        word_idx = None
        def_idx = None
        for idx, name in enumerate(header):
            if name in {"word", "term", "entry"}:
                word_idx = idx
            if name in {"definition", "meaning", "def"}:
                def_idx = idx
        if word_idx is not None:
            start_idx = 1

        for row in rows[start_idx:]:
            if not row:
                continue
            if word_idx is not None and word_idx < len(row):
                word = row[word_idx].strip()
                definition = None
                if def_idx is not None and def_idx < len(row):
                    definition = row[def_idx].strip() or None
            else:
                word = row[0].strip()
                definition = row[1].strip() if len(row) > 1 else None
            if word:
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
            paragraphs = [p.strip() for p in content.splitlines() if p.strip()]
            return self._group_chunks(paragraphs, size, "\n\n")
        if mode == "sentences":
            sentences = [s.strip() for s in SENTENCE_RE.split(content) if s.strip()]
            return self._group_chunks(sentences, size, " ")
        if mode != "words":
            raise ToolError("chunk_mode must be words, paragraphs, or sentences.")
        if overlap >= size:
            raise ToolError("chunk_overlap must be smaller than chunk_size for word chunking.")
        words = WORD_RE.findall(content)
        chunks: list[str] = []
        start = 0
        while start < len(words):
            end = min(start + size, len(words))
            subset = words[start:end]
            chunks.append(" ".join(subset))
            if end >= len(words):
                break
            start = end - overlap if overlap > 0 else end
        return chunks

    def _group_chunks(self, items: list[str], size: int, joiner: str) -> list[str]:
        if size <= 0:
            raise ToolError("chunk_size must be a positive integer.")
        chunks: list[str] = []
        for idx in range(0, len(items), size):
            subset = items[idx : idx + size]
            if subset:
                chunks.append(joiner.join(subset))
        return chunks

    def _extract_words(self, text: str, min_word_length: int) -> list[str]:
        return [
            word
            for word in (token.lower() for token in WORD_RE.findall(text))
            if len(word) >= min_word_length
        ]

    def _definition_tokens(self, text: str, min_token_length: int) -> set[str]:
        tokens = {
            token.lower()
            for token in WORD_RE.findall(text)
            if len(token) >= min_token_length and token.lower() not in STOPWORDS
        }
        return tokens

    def _preview_text(self, text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def _normalize_embedding(self, embedding: list[float]) -> list[float]:
        norm = sum(x * x for x in embedding) ** 0.5
        if norm == 0:
            return embedding
        return [x / norm for x in embedding]

    def _average_embeddings(self, embeddings: list[list[float]]) -> list[float] | None:
        if not embeddings:
            return None
        dim = len(embeddings[0])
        if dim == 0:
            return None
        totals = [0.0] * dim
        count = 0
        for emb in embeddings:
            if len(emb) != dim:
                continue
            for idx, val in enumerate(emb):
                totals[idx] += val
            count += 1
        if count == 0:
            return None
        avg = [val / count for val in totals]
        return self._normalize_embedding(avg)

    def _unpack_embedding(self, blob: bytes, dim: int | None) -> list[float]:
        arr = array("f")
        arr.frombytes(blob)
        values = list(arr)
        if dim is not None and dim > 0 and len(values) > dim:
            values = values[:dim]
        return self._normalize_embedding([float(x) for x in values])

    def _embed_text(
        self, ollama_url: str, embedding_model: str, text: str
    ) -> list[float]:
        url = ollama_url.rstrip("/") + "/api/embeddings"
        payload = json.dumps({"model": embedding_model, "prompt": text}).encode("utf-8")
        req = request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise ToolError(f"Ollama/GPT-OSS embeddings failed: {exc}") from exc

        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            raise ToolError("Invalid embeddings response from Ollama/GPT-OSS.")
        return self._normalize_embedding([float(x) for x in embedding])

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right) or not left:
            return 0.0
        return sum(l * r for l, r in zip(left, right))

    def _build_neighbors(
        self,
        embeddings: dict[str, list[float]],
        max_neighbors: int,
        min_similarity: float,
    ) -> dict[str, list[DocumentNeighbor]]:
        neighbors: dict[str, list[DocumentNeighbor]] = {}
        ids = list(embeddings.keys())
        for i, left_id in enumerate(ids):
            left_vec = embeddings[left_id]
            scored: list[tuple[float, str]] = []
            for j, right_id in enumerate(ids):
                if i == j:
                    continue
                score = self._cosine_similarity(left_vec, embeddings[right_id])
                if score < min_similarity:
                    continue
                scored.append((score, right_id))
            scored.sort(key=lambda item: (-item[0], item[1]))
            if max_neighbors > 0:
                scored = scored[:max_neighbors]
            neighbors[left_id] = [
                DocumentNeighbor(document_id=doc_id, similarity=round(score, 6))
                for score, doc_id in scored
            ]
        return neighbors

    def _coerce_document_id(
        self, base: str, label: str | None, used: set[str]
    ) -> str:
        candidate = f"{label}:{base}" if label else base
        if candidate not in used:
            return candidate
        suffix = 2
        while True:
            alt = f"{candidate}_{suffix}"
            if alt not in used:
                return alt
            suffix += 1

    def _detect_vector_table(self, conn: sqlite3.Connection) -> str | None:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        if "text_chunks" in tables:
            return "text_chunks"
        if "pdf_chunks" in tables:
            return "pdf_chunks"
        return None

    def _load_vector_documents(
        self,
        stores: list[VectorStore],
        max_docs: int,
        max_chunks_per_doc: int,
        max_source_bytes: int,
        max_total_bytes: int,
        use_embeddings: bool,
        errors: list[str],
        warnings: list[str],
    ) -> tuple[list[TextItem], dict[str, list[float]], dict[str, str]]:
        items: list[TextItem] = []
        embeddings: dict[str, list[float]] = {}
        sources: dict[str, str] = {}
        total_bytes = 0
        total_docs = 0
        used_ids: set[str] = set()

        for store_index, store in enumerate(stores, start=1):
            try:
                path = Path(store.path).expanduser()
                if not path.is_absolute():
                    path = self.config.effective_workdir / path
                path = path.resolve()
                if not path.exists():
                    errors.append(f"vector store not found: {path}")
                    continue

                store_label = store.label or path.stem
                store_max_docs = store.max_docs if store.max_docs is not None else max_docs
                store_max_chunks = (
                    store.max_chunks_per_doc
                    if store.max_chunks_per_doc is not None
                    else max_chunks_per_doc
                )
                if store_max_docs <= 0 or store_max_chunks <= 0:
                    raise ToolError("max docs/chunks must be positive.")

                conn = sqlite3.connect(str(path))
                try:
                    table = store.table
                    if not table:
                        table = self._detect_vector_table(conn)
                    if not table:
                        errors.append(f"{path}: no supported chunks table found.")
                        continue

                    cursor = conn.execute(
                        f"SELECT source_path, chunk_index, content, embedding, embedding_dim "
                        f"FROM {table} ORDER BY source_path, chunk_index"
                    )
                    current_path: str | None = None
                    chunk_texts: list[str] = []
                    chunk_embeddings: list[list[float]] = []
                    chunk_count = 0
                    stop = False
                    store_docs = 0

                    def flush_doc() -> None:
                        nonlocal total_bytes, total_docs, chunk_texts, chunk_embeddings, chunk_count, current_path, stop, store_docs
                        if current_path is None:
                            return
                        if store_docs >= store_max_docs:
                            stop = True
                            return
                        if total_docs >= max_docs:
                            stop = True
                            return
                        doc_text = "\n".join(chunk_texts)
                        if not doc_text:
                            chunk_texts = []
                            chunk_embeddings = []
                            chunk_count = 0
                            return
                        encoded = doc_text.encode("utf-8")
                        if len(encoded) > max_source_bytes:
                            doc_text = doc_text[:max_source_bytes]
                            encoded = doc_text.encode("utf-8")
                        if total_bytes + len(encoded) > max_total_bytes:
                            warnings.append("max_total_bytes exceeded while loading vector stores.")
                            stop = True
                            return
                        doc_id = self._coerce_document_id(current_path, store_label, used_ids)
                        used_ids.add(doc_id)
                        items.append(
                            TextItem(
                                id=doc_id,
                                name=store_label,
                                content=doc_text,
                                source=str(path),
                            )
                        )
                        sources[doc_id] = current_path
                        total_bytes += len(encoded)
                        total_docs += 1
                        store_docs += 1
                        if use_embeddings and chunk_embeddings:
                            doc_embedding = self._average_embeddings(chunk_embeddings)
                            if doc_embedding:
                                embeddings[doc_id] = doc_embedding
                        chunk_texts = []
                        chunk_embeddings = []
                        chunk_count = 0

                    for row in cursor:
                        source_path, _, content, blob, dim = row
                        if source_path != current_path:
                            flush_doc()
                            if stop:
                                break
                            current_path = source_path
                            chunk_texts = []
                            chunk_embeddings = []
                            chunk_count = 0

                        if chunk_count >= store_max_chunks:
                            continue
                        chunk_count += 1
                        if content:
                            chunk_texts.append(str(content))
                        if use_embeddings and blob:
                            try:
                                chunk_embeddings.append(
                                    self._unpack_embedding(blob, dim)
                                )
                            except Exception:
                                continue

                    if not stop:
                        flush_doc()
                finally:
                    conn.close()
            except Exception as exc:
                errors.append(f"vector_store[{store_index}]: {exc}")

        return items, embeddings, sources

    def _build_word_index(
        self, dict_word_sets: dict[str, set[str]]
    ) -> dict[str, list[str]]:
        index: dict[str, list[str]] = {}
        for dict_id, words in dict_word_sets.items():
            for word in words:
                index.setdefault(word, []).append(dict_id)
        return index

    def _analyze_chunk(
        self,
        chunk_text: str,
        word_index: dict[str, list[str]],
        dict_word_sets: dict[str, set[str]],
        dict_info: dict[str, _StoredDictionary],
        union_words: set[str],
        max_words_per_chunk: int,
        min_word_length: int,
        max_word_sources: int,
        max_missing_words: int,
        max_shared_words: int,
        max_sample_words: int,
        max_definition_chars: int,
        definition_min_token_length: int,
        definition_weight: float,
        priority_weight: float,
        size_penalty: float,
        marker_boost: float,
        usage_boost: float,
        preview_chars: int,
    ) -> tuple[ChunkAnalysis, bool, dict[str, int]]:
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
                    dictionary_scores=[],
                    shared_words=[],
                    missing_words=[],
                    word_sources=[],
                    ambiguous_terms=[],
                    truncated=False,
                ),
                False,
                {},
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
        dictionary_scores: list[DictionaryScore] = []
        score_map: dict[str, float] = {}
        matched_counts: dict[str, int] = {}
        context_tokens = self._definition_tokens(
            chunk_text, definition_min_token_length
        )

        for dict_id, words_set in dict_word_sets.items():
            matched = sorted(word_set & words_set)
            matched_count = len(matched)
            coverage = matched_count / len(word_set) if word_set else 0.0
            marker_hits = len(word_set & dict_info[dict_id].markers)
            usage_weight = usage_boost * math.log1p(dict_info[dict_id].usage_count)
            size_norm = math.log1p(len(words_set)) ** size_penalty
            if size_norm <= 0:
                size_norm = 1.0
            priority_score = priority_weight * dict_info[dict_id].priority
            score = (matched_count / size_norm) + (marker_boost * marker_hits) + usage_weight + priority_score
            sample = matched[:max_sample_words] if max_sample_words > 0 else []
            dictionary_scores.append(
                DictionaryScore(
                    dictionary_id=dict_id,
                    name=dict_info[dict_id].name,
                    matched_words=matched_count,
                    coverage=round(coverage, 6),
                    score=round(score, 6),
                    sample_words=sample,
                )
            )
            score_map[dict_id] = score
            matched_counts[dict_id] = matched_count

        dictionary_scores.sort(
            key=lambda item: (-item.score, -item.matched_words, item.dictionary_id)
        )

        shared_candidates: list[tuple[int, str]] = []
        word_sources: list[WordSource] = []
        ambiguous_terms: list[AmbiguousTerm] = []
        for word in word_set:
            dict_ids = word_index.get(word)
            if not dict_ids:
                continue
            sorted_ids = sorted(set(dict_ids))
            if len(sorted_ids) > 1:
                shared_candidates.append((len(sorted_ids), word))
                definition_scores = self._definition_scores(
                    word,
                    sorted_ids,
                    dict_info,
                    context_tokens,
                    definition_min_token_length,
                    max_definition_chars,
                )
                weighted_scores = {
                    dict_id: score * definition_weight
                    for dict_id, score in definition_scores.items()
                }
                resolved_id, reason = self._resolve_ambiguous(
                    sorted_ids,
                    score_map,
                    dict_info,
                    matched_counts,
                    weighted_scores,
                )
                definitions = self._collect_definitions(
                    word, sorted_ids, dict_info, max_definition_chars
                )
                ambiguous_terms.append(
                    AmbiguousTerm(
                        word=word,
                        dictionary_ids=sorted_ids,
                        resolved_dictionary_id=resolved_id,
                        resolved_reason=reason,
                        definitions=definitions,
                        definition_scores=definition_scores,
                    )
                )
            word_sources.append(WordSource(word=word, dictionary_ids=sorted_ids))

        shared_candidates.sort(key=lambda item: (-item[0], item[1]))
        shared_words = [word for _, word in shared_candidates]
        if max_shared_words > 0 and len(shared_words) > max_shared_words:
            truncated = True
            shared_words = shared_words[:max_shared_words]

        word_sources.sort(key=lambda item: (-len(item.dictionary_ids), item.word))
        if max_word_sources > 0 and len(word_sources) > max_word_sources:
            truncated = True
            word_sources = word_sources[:max_word_sources]

        ambiguous_terms.sort(key=lambda item: (-len(item.dictionary_ids), item.word))
        if max_shared_words > 0 and len(ambiguous_terms) > max_shared_words:
            truncated = True
            ambiguous_terms = ambiguous_terms[:max_shared_words]

        missing_words = sorted(word_set - union_words)
        if max_missing_words > 0 and len(missing_words) > max_missing_words:
            truncated = True
            missing_words = missing_words[:max_missing_words]

        preview = self._preview_text(chunk_text, preview_chars)
        return (
            ChunkAnalysis(
                item_index=0,
                item_id=None,
                chunk_index=0,
                preview=preview,
                word_count=word_count,
                unique_words=len(word_set),
                dictionary_scores=dictionary_scores,
                shared_words=shared_words,
                missing_words=missing_words,
                word_sources=word_sources,
                ambiguous_terms=ambiguous_terms,
                truncated=truncated,
            ),
            truncated,
            matched_counts,
        )

    def _resolve_ambiguous(
        self,
        dict_ids: list[str],
        score_map: dict[str, float],
        dict_info: dict[str, _StoredDictionary],
        matched_counts: dict[str, int],
        definition_scores: dict[str, float],
    ) -> tuple[str | None, str | None]:
        candidates: list[tuple[float, float, int, int, int, str]] = []
        for dict_id in dict_ids:
            definition_score = definition_scores.get(dict_id, 0.0)
            score = score_map.get(dict_id, 0.0)
            usage = dict_info[dict_id].usage_count
            matched = matched_counts.get(dict_id, 0)
            priority = dict_info[dict_id].priority
            candidates.append((definition_score, score, priority, usage, matched, dict_id))
        candidates.sort(reverse=True)
        if not candidates:
            return None, None
        if len(candidates) == 1:
            return candidates[0][5], "single"
        top = candidates[0]
        second = candidates[1]
        if top[0] > second[0]:
            return top[5], "definition_match"
        if top[1] > second[1]:
            return top[5], "chunk_score"
        if top[2] > second[2]:
            return top[5], "priority"
        if top[3] > second[3]:
            return top[5], "usage_count"
        if top[4] > second[4]:
            return top[5], "matched_words"
        return None, "tie"

    def _collect_definitions(
        self,
        word: str,
        dict_ids: list[str],
        dict_info: dict[str, _StoredDictionary],
        max_definition_chars: int,
    ) -> dict[str, str]:
        definitions: dict[str, str] = {}
        for dict_id in dict_ids:
            definition = dict_info[dict_id].entries.get(word)
            if definition is None:
                continue
            if max_definition_chars > 0 and len(definition) > max_definition_chars:
                definitions[dict_id] = definition[:max_definition_chars]
            else:
                definitions[dict_id] = definition
        return definitions

    def _definition_scores(
        self,
        word: str,
        dict_ids: list[str],
        dict_info: dict[str, _StoredDictionary],
        context_tokens: set[str],
        min_token_length: int,
        max_definition_chars: int,
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        if not context_tokens:
            return scores
        context_tokens = set(context_tokens)
        if word in context_tokens:
            context_tokens.remove(word)
        for dict_id in dict_ids:
            definition = dict_info[dict_id].entries.get(word)
            if not definition:
                continue
            text = definition
            if max_definition_chars > 0 and len(text) > max_definition_chars:
                text = text[:max_definition_chars]
            def_tokens = self._definition_tokens(text, min_token_length)
            if word in def_tokens:
                def_tokens.remove(word)
            if not def_tokens:
                continue
            overlap = len(def_tokens & context_tokens)
            scores[dict_id] = overlap / len(def_tokens)
        return scores

    def _summarize_dictionaries(
        self, dictionaries: dict[str, _StoredDictionary]
    ) -> list[DictionaryProfileSummary]:
        summaries: list[DictionaryProfileSummary] = []
        for dict_id, entry in dictionaries.items():
            summaries.append(
                DictionaryProfileSummary(
                    dictionary_id=dict_id,
                    name=entry.name,
                    tags=entry.tags,
                    priority=entry.priority,
                    entry_count=entry.entry_count,
                    definition_count=entry.definition_count,
                    usage_count=entry.usage_count,
                    total_hits=entry.total_hits,
                    last_used_at=entry.last_used_at,
                    source=entry.source,
                    persistent=entry.persistent,
                )
            )
        summaries.sort(key=lambda item: item.dictionary_id)
        return summaries

    def _load_profile(
        self, path: Path, errors: list[str]
    ) -> dict[str, _StoredDictionary]:
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"Failed to read profile: {exc}")
            return {}

        entries_raw = data.get("dictionaries", []) if isinstance(data, dict) else data
        dictionaries: dict[str, _StoredDictionary] = {}
        if isinstance(entries_raw, list):
            for entry in entries_raw:
                try:
                    dict_id = str(entry.get("id") or entry.get("dictionary_id"))
                    if not dict_id:
                        continue
                    entries = self._coerce_profile_entries(entry.get("entries", []))
                    tags = self._normalize_tags(entry.get("tags") or [])
                    raw_priority = entry.get("priority")
                    priority = (
                        int(raw_priority)
                        if raw_priority is not None
                        else self._infer_priority_from_meta(dict_id, entry.get("name"), tags)
                    )
                    dictionaries[dict_id] = _StoredDictionary(
                        dict_id=dict_id,
                        name=entry.get("name"),
                        tags=tags,
                        markers=self._normalize_markers(entry.get("markers") or [], self.config.min_word_length),
                        priority=priority,
                        source=entry.get("source"),
                        created_at=str(entry.get("created_at") or entry.get("updated_at") or self._now_iso()),
                        updated_at=str(entry.get("updated_at") or self._now_iso()),
                        entries=entries,
                        entry_count=int(entry.get("entry_count") or len(entries)),
                        definition_count=int(entry.get("definition_count") or self._count_definitions(entries)),
                        usage_count=int(entry.get("usage_count") or 0),
                        total_hits=int(entry.get("total_hits") or 0),
                        last_used_at=entry.get("last_used_at"),
                        persistent=True,
                    )
                except Exception as exc:
                    errors.append(f"Invalid profile entry: {exc}")
        return dictionaries

    def _coerce_profile_entries(self, entries: object) -> dict[str, str | None]:
        if isinstance(entries, dict):
            return {
                self._normalize_word(str(word), self.config.min_word_length) or str(word).lower(): (
                    str(defn) if defn is not None else None
                )
                for word, defn in entries.items()
            }
        output: dict[str, str | None] = {}
        if isinstance(entries, list):
            for item in entries:
                if isinstance(item, dict):
                    word = item.get("word") or item.get("term") or item.get("entry")
                    if not word:
                        continue
                    normalized = self._normalize_word(str(word), self.config.min_word_length)
                    if not normalized:
                        continue
                    definition = item.get("definition") or item.get("meaning") or item.get("def")
                    output[normalized] = (
                        str(definition) if definition is not None else None
                    )
                else:
                    normalized = self._normalize_word(str(item), self.config.min_word_length)
                    if normalized:
                        output[normalized] = None
        return output

    def _count_definitions(self, entries: dict[str, str | None]) -> int:
        return sum(1 for value in entries.values() if value)

    def _save_profile(
        self,
        path: Path,
        dictionaries: dict[str, _StoredDictionary],
        max_profile_bytes: int,
        errors: list[str],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": self._now_iso(),
            "dictionaries": [
                {
                    "id": entry.dict_id,
                    "name": entry.name,
                    "tags": entry.tags,
                    "markers": sorted(entry.markers),
                    "priority": entry.priority,
                    "source": entry.source,
                    "created_at": entry.created_at,
                    "updated_at": entry.updated_at,
                    "usage_count": entry.usage_count,
                    "total_hits": entry.total_hits,
                    "last_used_at": entry.last_used_at,
                    "entry_count": entry.entry_count,
                    "definition_count": entry.definition_count,
                    "entries": [
                        {"word": word, "definition": definition}
                        for word, definition in sorted(entry.entries.items())
                    ],
                }
                for entry in dictionaries.values()
            ],
        }
        try:
            encoded = json.dumps(payload, ensure_ascii=True, indent=2)
            size = len(encoded.encode("utf-8"))
            if size > max_profile_bytes:
                raise ToolError(
                    f"profile exceeds max_profile_bytes ({size} > {max_profile_bytes})."
                )
            path.write_text(encoded, "utf-8")
        except Exception as exc:
            errors.append(f"Failed to write profile: {exc}")

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextDictionaryMemoryArgs):
            return ToolCallDisplay(summary="context_dictionary_memory")
        summary = f"context_dictionary_memory: {event.args.action or 'analyze'}"
        return ToolCallDisplay(
            summary=summary,
            details={
                "action": event.args.action,
                "dictionary_count": len(event.args.dictionaries or []),
                "item_count": len(event.args.items or []),
                "vector_store_count": len(event.args.vector_stores or []),
                "dictionary_ids": event.args.dictionary_ids,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextDictionaryMemoryResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        if event.result.action == "analyze":
            message = (
                f"Processed {event.result.chunk_count} chunk(s) across "
                f"{event.result.dictionary_count} dictionaries"
            )
        else:
            message = (
                f"{event.result.action}: {event.result.dictionary_count} dictionaries"
            )

        warnings = event.result.warnings[:]
        if event.result.truncated:
            warnings.append("Output truncated by size or limits")

        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=warnings,
            details={
                "action": event.result.action,
                "dictionary_count": event.result.dictionary_count,
                "chunk_count": event.result.chunk_count,
                "truncated": event.result.truncated,
                "updated": event.result.updated,
                "embedding_used": event.result.embedding_used,
                "errors": event.result.errors,
                "warnings": event.result.warnings,
                "dictionaries": event.result.dictionaries,
                "dictionary_usage": event.result.dictionary_usage,
                "dictionary_documents": event.result.dictionary_documents,
                "document_usage": event.result.document_usage,
                "chunks": event.result.chunks,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Managing dictionary memory"
