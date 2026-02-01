from __future__ import annotations

from dataclasses import dataclass
from array import array
from heapq import heappush, heappushpop
from pathlib import Path
import csv
import json
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
SUPPORTED_MATCH_TYPES = {"contains", "prefix", "suffix", "regex"}
SUPPORTED_TARGETS = {"path", "title", "source"}
DEFAULT_LABEL = "unknown"


@dataclass
class _VectorRow:
    score: float
    source_path: str
    chunk_index: int
    unit: str
    start_index: int | None
    end_index: int | None
    content: str
    store_label: str | None
    store_path: str | None


@dataclass
class _ChunkRecord:
    item_index: int
    item_id: str | None
    chunk_index: int
    source_path: str | None
    store_label: str | None
    store_path: str | None
    vector_score: float | None
    tags: list[str]
    token_counts: dict[str, int]
    token_set: set[str]
    preview: str


class ContextGenreVectorsConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    ollama_url: str = Field(
        default="http://127.0.0.1:11434",
        description="Base URL for the Ollama/GPT-OSS server.",
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        description="Embedding model to use with Ollama/GPT-OSS.",
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
    min_word_length: int = Field(default=2, description="Minimum word length to include.")
    max_words_per_chunk: int = Field(
        default=2000, description="Maximum unique words per chunk to analyze."
    )
    max_shared_terms: int = Field(
        default=50, description="Maximum shared terms per overlap.")
    max_bridge_terms: int = Field(
        default=50, description="Maximum bridge terms returned.")
    max_label_overlaps: int = Field(
        default=200, description="Maximum label overlap entries returned.")
    max_label_recommendations: int = Field(
        default=5, description="Maximum label recommendations per chunk.")
    similarity_mode: str = Field(
        default="jaccard", description="jaccard or overlap."
    )
    min_label_similarity: float = Field(
        default=0.05, description="Minimum label similarity to report overlaps."
    )
    max_vector_results_per_store: int = Field(
        default=200, description="Maximum vector results per store and query."
    )
    max_vector_total_results: int = Field(
        default=1000, description="Maximum total vector results across queries."
    )
    max_result_chars: int = Field(
        default=1000, description="Maximum characters to return per vector result."
    )


class ContextGenreVectorsState(BaseToolState):
    pass

class GenreRule(BaseModel):
    tag: str = Field(description="Genre or classification tag.")
    kind: str | None = Field(default="genre", description="genre or classification.")
    match: str = Field(description="Match string or regex pattern.")
    match_type: str = Field(
        default="contains", description="contains, prefix, suffix, or regex."
    )
    target: str = Field(
        default="path", description="path, title, or source."
    )


class GenreItem(BaseModel):
    id: str | None = Field(default=None, description="Optional item id.")
    title: str | None = Field(default=None, description="Optional item title.")
    content: str | None = Field(default=None, description="Inline text content.")
    path: str | None = Field(default=None, description="Path to a text file.")
    source: str | None = Field(default=None, description="Source description.")
    genres: list[str] | None = Field(default=None, description="Genre tags.")
    classifications: list[str] | None = Field(
        default=None, description="Classification tags."
    )
    chunk_mode: str | None = Field(
        default=None, description="words, paragraphs, sentences."
    )
    chunk_size: int | None = Field(default=None, description="Override chunk size.")
    chunk_overlap: int | None = Field(
        default=None, description="Override word overlap for word chunking."
    )


class VectorStore(BaseModel):
    path: str = Field(description="Path to a vector DB or manifest file.")
    store_type: str | None = Field(
        default="db", description="db, manifest, or sharded."
    )
    label: str | None = Field(default=None, description="Optional store label.")
    min_score: float | None = Field(default=None, description="Minimum similarity score.")
    max_results: int | None = Field(
        default=None, description="Override max results for this store."
    )
    max_result_chars: int | None = Field(
        default=None, description="Override max chars per result."
    )


class VectorQuery(BaseModel):
    query: str = Field(description="Vector search query.")
    top_k: int | None = Field(default=None, description="Override top_k.")
    min_score: float | None = Field(default=None, description="Override min score.")


class ContextGenreVectorsArgs(BaseModel):
    items: list[GenreItem] | None = Field(default=None, description="Text items.")
    vector_queries: list[VectorQuery] | None = Field(
        default=None, description="Vector search queries."
    )
    vector_stores: list[VectorStore] | None = Field(
        default=None, description="Vector stores for queries."
    )
    genre_rules: list[GenreRule] | None = Field(
        default=None, description="Rules to map paths to tags."
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
    max_shared_terms: int | None = Field(
        default=None, description="Override max_shared_terms."
    )
    max_bridge_terms: int | None = Field(
        default=None, description="Override max_bridge_terms."
    )
    max_label_overlaps: int | None = Field(
        default=None, description="Override max_label_overlaps."
    )
    max_label_recommendations: int | None = Field(
        default=None, description="Override max_label_recommendations."
    )
    similarity_mode: str | None = Field(
        default=None, description="Override similarity_mode."
    )
    min_label_similarity: float | None = Field(
        default=None, description="Override min_label_similarity."
    )
    max_vector_results_per_store: int | None = Field(
        default=None, description="Override max_vector_results_per_store."
    )
    max_vector_total_results: int | None = Field(
        default=None, description="Override max_vector_total_results."
    )
    max_result_chars: int | None = Field(
        default=None, description="Override max_result_chars."
    )
    embedding_model: str | None = Field(
        default=None, description="Override embedding model."
    )


class LabelSummary(BaseModel):
    tag: str
    kind: str
    chunk_count: int
    source_count: int
    keywords: list[str]


class LabelRecommendation(BaseModel):
    tag: str
    score: float
    overlap: int
    coverage: float


class LabelOverlap(BaseModel):
    left_tag: str
    right_tag: str
    shared_terms: list[str]
    shared_count: int
    similarity: float


class BridgeTerm(BaseModel):
    term: str
    label_count: int
    labels: list[str]


class ChunkAnalysis(BaseModel):
    item_index: int
    item_id: str | None
    chunk_index: int
    source_path: str | None
    store_label: str | None
    store_path: str | None
    vector_score: float | None
    tags: list[str]
    preview: str
    token_count: int
    keywords: list[str]
    recommendations: list[LabelRecommendation]
    bridge_terms: list[str]


class ContextGenreVectorsResult(BaseModel):
    labels: list[LabelSummary]
    chunks: list[ChunkAnalysis]
    label_overlaps: list[LabelOverlap]
    bridge_terms: list[BridgeTerm]
    label_count: int
    chunk_count: int
    overlap_count: int
    bridge_count: int
    truncated: bool
    errors: list[str]

class ContextGenreVectors(
    BaseTool[
        ContextGenreVectorsArgs,
        ContextGenreVectorsResult,
        ContextGenreVectorsConfig,
        ContextGenreVectorsState,
    ],
    ToolUIData[ContextGenreVectorsArgs, ContextGenreVectorsResult],
):
    description: ClassVar[str] = (
        "Reason across document genres and classifications with optional vector stores."
    )

    async def run(
        self, args: ContextGenreVectorsArgs
    ) -> ContextGenreVectorsResult:
        items = args.items or []
        vector_queries = args.vector_queries or []
        if not items and not vector_queries:
            raise ToolError("items or vector_queries is required.")

        vector_stores = args.vector_stores or []
        if vector_queries and not vector_stores:
            raise ToolError("vector_stores is required when vector_queries are provided.")

        max_items = args.max_items if args.max_items is not None else self.config.max_items
        if max_items <= 0:
            raise ToolError("max_items must be a positive integer.")
        if len(items) > max_items:
            raise ToolError(f"items exceeds max_items ({len(items)} > {max_items}).")

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
        max_shared_terms = (
            args.max_shared_terms
            if args.max_shared_terms is not None
            else self.config.max_shared_terms
        )
        max_bridge_terms = (
            args.max_bridge_terms
            if args.max_bridge_terms is not None
            else self.config.max_bridge_terms
        )
        max_label_overlaps = (
            args.max_label_overlaps
            if args.max_label_overlaps is not None
            else self.config.max_label_overlaps
        )
        max_label_recommendations = (
            args.max_label_recommendations
            if args.max_label_recommendations is not None
            else self.config.max_label_recommendations
        )
        similarity_mode = (
            args.similarity_mode
            if args.similarity_mode is not None
            else self.config.similarity_mode
        ).strip().lower()
        min_label_similarity = (
            args.min_label_similarity
            if args.min_label_similarity is not None
            else self.config.min_label_similarity
        )
        max_vector_results_per_store = (
            args.max_vector_results_per_store
            if args.max_vector_results_per_store is not None
            else self.config.max_vector_results_per_store
        )
        max_vector_total_results = (
            args.max_vector_total_results
            if args.max_vector_total_results is not None
            else self.config.max_vector_total_results
        )
        max_result_chars = (
            args.max_result_chars
            if args.max_result_chars is not None
            else self.config.max_result_chars
        )
        embedding_model = args.embedding_model or self.config.embedding_model

        if max_source_bytes <= 0:
            raise ToolError("max_source_bytes must be a positive integer.")
        if max_total_bytes <= 0:
            raise ToolError("max_total_bytes must be a positive integer.")
        if max_chunks <= 0:
            raise ToolError("max_chunks must be a positive integer.")
        if preview_chars < 0:
            raise ToolError("preview_chars must be >= 0.")
        if min_word_length <= 0:
            raise ToolError("min_word_length must be a positive integer.")
        if max_words_per_chunk <= 0:
            raise ToolError("max_words_per_chunk must be a positive integer.")
        if max_shared_terms < 0:
            raise ToolError("max_shared_terms must be >= 0.")
        if max_bridge_terms < 0:
            raise ToolError("max_bridge_terms must be >= 0.")
        if max_label_overlaps < 0:
            raise ToolError("max_label_overlaps must be >= 0.")
        if max_label_recommendations < 0:
            raise ToolError("max_label_recommendations must be >= 0.")
        if similarity_mode not in {"jaccard", "overlap"}:
            raise ToolError("similarity_mode must be jaccard or overlap.")
        if min_label_similarity < 0:
            raise ToolError("min_label_similarity must be >= 0.")
        if max_vector_results_per_store <= 0:
            raise ToolError("max_vector_results_per_store must be positive.")
        if max_vector_total_results <= 0:
            raise ToolError("max_vector_total_results must be positive.")
        if max_result_chars <= 0:
            raise ToolError("max_result_chars must be positive.")

        rules = args.genre_rules or []
        self._validate_rules(rules)

        chunks: list[_ChunkRecord] = []
        label_token_counts: dict[str, dict[str, int]] = {}
        label_sources: dict[str, set[str]] = {}
        label_kinds: dict[str, str] = {}
        errors: list[str] = []
        truncated = False
        total_bytes = 0

        for item_idx, item in enumerate(items, start=1):
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

                tags = self._resolve_item_tags(
                    item, source_path, rules, label_kinds
                )
                if not tags:
                    tags = [DEFAULT_LABEL]
                    self._add_label_kind(label_kinds, DEFAULT_LABEL, "unknown")

                chunk_texts = self._chunk_content(
                    content, chunk_mode, chunk_size, chunk_overlap
                )
                for local_idx, chunk_text in enumerate(chunk_texts, start=1):
                    if len(chunks) >= max_chunks:
                        truncated = True
                        break
                    tokens, token_set = self._extract_tokens(
                        chunk_text, min_word_length, max_words_per_chunk
                    )
                    preview = self._preview_text(chunk_text, preview_chars)
                    record = _ChunkRecord(
                        item_index=item_idx,
                        item_id=item.id,
                        chunk_index=local_idx,
                        source_path=source_path,
                        store_label=None,
                        store_path=None,
                        vector_score=None,
                        tags=tags,
                        token_counts=tokens,
                        token_set=token_set,
                        preview=preview,
                    )
                    chunks.append(record)
                    source_key = item.id or source_path or f"item{item_idx}"
                    self._accumulate_labels(
                        tags,
                        tokens,
                        source_key,
                        label_token_counts,
                        label_sources,
                        label_kinds,
                    )
                if len(chunks) >= max_chunks:
                    break
            except ToolError as exc:
                errors.append(f"item[{item_idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{item_idx}]: {exc}")

        if vector_queries:
            try:
                vector_chunks, vector_truncated = self._process_vector_queries(
                vector_queries,
                vector_stores,
                rules,
                label_kinds,
                label_token_counts,
                label_sources,
                embedding_model,
                max_vector_results_per_store,
                max_vector_total_results,
                max_result_chars,
                min_word_length,
                max_words_per_chunk,
                preview_chars,
                similarity_mode,
                min_label_similarity,
                errors,
            )
                for record in vector_chunks:
                    if len(chunks) >= max_chunks:
                        truncated = True
                        break
                    chunks.append(record)
                if vector_truncated:
                    truncated = True
            except ToolError as exc:
                errors.append(str(exc))
            except Exception as exc:
                errors.append(f"vector_queries: {exc}")

        if not chunks:
            raise ToolError("No chunks to process.")

        label_chunk_counts: dict[str, int] = {}
        for record in chunks:
            for tag in record.tags:
                label_chunk_counts[tag] = label_chunk_counts.get(tag, 0) + 1

        label_sets = {tag: set(counts.keys()) for tag, counts in label_token_counts.items()}
        labels = self._build_label_summaries(
            label_token_counts,
            label_sources,
            label_kinds,
            label_chunk_counts,
            max_shared_terms,
        )
        label_overlaps = self._build_label_overlaps(
            label_sets,
            max_shared_terms,
            max_label_overlaps,
            similarity_mode,
            min_label_similarity,
        )
        bridge_terms = self._build_bridge_terms(
            label_sets,
            max_bridge_terms,
        )
        bridge_map = {term.term: term for term in bridge_terms}

        chunks_out: list[ChunkAnalysis] = []
        for record in chunks:
            keywords = self._select_keywords(record.token_counts, max_shared_terms)
            recommendations = self._recommend_labels(
                record.token_set,
                label_sets,
                max_label_recommendations,
                similarity_mode,
            )
            chunk_bridge_terms = [
                term for term in record.token_set if term in bridge_map
            ]
            if max_shared_terms > 0 and len(chunk_bridge_terms) > max_shared_terms:
                chunk_bridge_terms = chunk_bridge_terms[:max_shared_terms]

            chunks_out.append(
                ChunkAnalysis(
                    item_index=record.item_index,
                    item_id=record.item_id,
                    chunk_index=record.chunk_index,
                    source_path=record.source_path,
                    store_label=record.store_label,
                    store_path=record.store_path,
                    vector_score=record.vector_score,
                    tags=record.tags,
                    preview=record.preview,
                    token_count=sum(record.token_counts.values()),
                    keywords=keywords,
                    recommendations=recommendations,
                    bridge_terms=chunk_bridge_terms,
                )
            )

        return ContextGenreVectorsResult(
            labels=labels,
            chunks=chunks_out,
            label_overlaps=label_overlaps,
            bridge_terms=bridge_terms,
            label_count=len(labels),
            chunk_count=len(chunks_out),
            overlap_count=len(label_overlaps),
            bridge_count=len(bridge_terms),
            truncated=truncated,
            errors=errors,
        )
    def _normalize_tag(self, tag: str | None) -> str | None:
        if not tag:
            return None
        normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", tag.strip()).lower()
        return normalized or None

    def _validate_rules(self, rules: list[GenreRule]) -> None:
        for idx, rule in enumerate(rules, start=1):
            tag = self._normalize_tag(rule.tag)
            if not tag:
                raise ToolError(f"genre_rules[{idx}] tag is required.")
            match = (rule.match or "").strip()
            if not match:
                raise ToolError(f"genre_rules[{idx}] match is required.")
            match_type = (rule.match_type or "contains").strip().lower()
            if match_type not in SUPPORTED_MATCH_TYPES:
                raise ToolError(
                    f"genre_rules[{idx}] match_type must be contains, prefix, suffix, or regex."
                )
            target = (rule.target or "path").strip().lower()
            if target not in SUPPORTED_TARGETS:
                raise ToolError(
                    f"genre_rules[{idx}] target must be path, title, or source."
                )
            kind = (rule.kind or "genre").strip().lower()
            if kind not in {"genre", "classification"}:
                raise ToolError(
                    f"genre_rules[{idx}] kind must be genre or classification."
                )
            if match_type == "regex":
                try:
                    re.compile(match)
                except re.error as exc:
                    raise ToolError(f"genre_rules[{idx}] invalid regex: {exc}") from exc

    def _match_rule(self, rule: GenreRule, value: str) -> bool:
        if not value:
            return False
        match_type = (rule.match_type or "contains").strip().lower()
        match = rule.match or ""
        if match_type == "regex":
            try:
                return bool(re.search(match, value, re.IGNORECASE))
            except re.error:
                return False
        haystack = value.lower()
        needle = match.lower()
        if match_type == "contains":
            return needle in haystack
        if match_type == "prefix":
            return haystack.startswith(needle)
        if match_type == "suffix":
            return haystack.endswith(needle)
        return False

    def _resolve_item_tags(
        self,
        item: GenreItem,
        source_path: str | None,
        rules: list[GenreRule],
        label_kinds: dict[str, str],
    ) -> list[str]:
        tags: list[str] = []
        seen: set[str] = set()

        def add(raw: str | None, kind: str) -> None:
            normalized = self._normalize_tag(raw)
            if not normalized:
                return
            if normalized in seen:
                return
            seen.add(normalized)
            tags.append(normalized)
            self._add_label_kind(label_kinds, normalized, kind)

        for tag in item.genres or []:
            add(tag, "genre")
        for tag in item.classifications or []:
            add(tag, "classification")

        targets = {
            "path": source_path or item.path or "",
            "title": item.title or item.id or "",
            "source": item.source or "",
        }

        for rule in rules:
            target_value = targets.get((rule.target or "path").strip().lower(), "")
            if not target_value:
                continue
            if self._match_rule(rule, target_value):
                kind = (rule.kind or "genre").strip().lower()
                if kind not in {"genre", "classification"}:
                    kind = "genre"
                add(rule.tag, kind)

        return tags

    def _add_label_kind(self, label_kinds: dict[str, str], tag: str, kind: str) -> None:
        if not tag:
            return
        normalized_kind = (kind or "unknown").strip().lower()
        if not normalized_kind:
            normalized_kind = "unknown"
        existing = label_kinds.get(tag)
        if existing is None:
            label_kinds[tag] = normalized_kind
            return
        if existing == normalized_kind:
            return
        if existing == "unknown":
            label_kinds[tag] = normalized_kind
            return
        if normalized_kind == "unknown":
            return
        label_kinds[tag] = "mixed"

    def _load_item_content(
        self, item: GenreItem, max_source_bytes: int
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

    def _extract_tokens(
        self, text: str, min_word_length: int, max_words_per_chunk: int
    ) -> tuple[dict[str, int], set[str]]:
        if not text:
            return {}, set()
        word_freq: dict[str, int] = {}
        for token in WORD_RE.findall(text):
            token_l = token.lower()
            if len(token_l) < min_word_length:
                continue
            if token_l in STOPWORDS:
                continue
            word_freq[token_l] = word_freq.get(token_l, 0) + 1

        if not word_freq:
            return {}, set()

        if len(word_freq) > max_words_per_chunk:
            ordered = sorted(word_freq.items(), key=lambda item: (-item[1], item[0]))
            word_freq = dict(ordered[:max_words_per_chunk])

        return word_freq, set(word_freq.keys())

    def _preview_text(self, text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def _accumulate_labels(
        self,
        tags: list[str],
        token_counts: dict[str, int],
        source_key: str,
        label_token_counts: dict[str, dict[str, int]],
        label_sources: dict[str, set[str]],
        label_kinds: dict[str, str],
    ) -> None:
        for tag in tags:
            if tag not in label_kinds:
                label_kinds[tag] = "unknown"
            token_map = label_token_counts.setdefault(tag, {})
            for token, count in token_counts.items():
                token_map[token] = token_map.get(token, 0) + count
            label_sources.setdefault(tag, set()).add(source_key)

    def _select_keywords(
        self, token_counts: dict[str, int], max_terms: int
    ) -> list[str]:
        if max_terms <= 0:
            return []
        ordered = sorted(token_counts.items(), key=lambda item: (-item[1], item[0]))
        return [token for token, _ in ordered[:max_terms]]

    def _build_label_summaries(
        self,
        label_token_counts: dict[str, dict[str, int]],
        label_sources: dict[str, set[str]],
        label_kinds: dict[str, str],
        label_chunk_counts: dict[str, int],
        max_shared_terms: int,
    ) -> list[LabelSummary]:
        summaries: list[LabelSummary] = []
        for tag, tokens in label_token_counts.items():
            keywords = self._select_keywords(tokens, max_shared_terms)
            summaries.append(
                LabelSummary(
                    tag=tag,
                    kind=label_kinds.get(tag, "unknown"),
                    chunk_count=label_chunk_counts.get(tag, 0),
                    source_count=len(label_sources.get(tag, set())),
                    keywords=keywords,
                )
            )
        summaries.sort(key=lambda item: (-item.chunk_count, item.tag))
        return summaries

    def _build_label_overlaps(
        self,
        label_sets: dict[str, set[str]],
        max_shared_terms: int,
        max_label_overlaps: int,
        similarity_mode: str,
        min_label_similarity: float,
    ) -> list[LabelOverlap]:
        tags = sorted(label_sets.keys())
        overlaps: list[LabelOverlap] = []
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                left_tag = tags[i]
                right_tag = tags[j]
                left_terms = label_sets[left_tag]
                right_terms = label_sets[right_tag]
                shared = left_terms & right_terms
                if not shared:
                    continue
                shared_count = len(shared)
                if similarity_mode == "jaccard":
                    union = len(left_terms | right_terms)
                    similarity = shared_count / union if union else 0.0
                else:
                    denom = min(len(left_terms), len(right_terms))
                    similarity = shared_count / denom if denom else 0.0
                if similarity < min_label_similarity:
                    continue
                shared_terms = sorted(shared)
                if max_shared_terms > 0 and len(shared_terms) > max_shared_terms:
                    shared_terms = shared_terms[:max_shared_terms]
                overlaps.append(
                    LabelOverlap(
                        left_tag=left_tag,
                        right_tag=right_tag,
                        shared_terms=shared_terms,
                        shared_count=shared_count,
                        similarity=round(similarity, 6),
                    )
                )

        overlaps.sort(key=lambda item: (-item.similarity, -item.shared_count, item.left_tag))
        if max_label_overlaps > 0 and len(overlaps) > max_label_overlaps:
            overlaps = overlaps[:max_label_overlaps]
        return overlaps

    def _build_bridge_terms(
        self,
        label_sets: dict[str, set[str]],
        max_bridge_terms: int,
    ) -> list[BridgeTerm]:
        if max_bridge_terms <= 0:
            return []
        term_labels: dict[str, set[str]] = {}
        for tag, terms in label_sets.items():
            for term in terms:
                term_labels.setdefault(term, set()).add(tag)

        bridge_terms: list[BridgeTerm] = []
        for term, labels in term_labels.items():
            if len(labels) < 2:
                continue
            bridge_terms.append(
                BridgeTerm(
                    term=term,
                    label_count=len(labels),
                    labels=sorted(labels),
                )
            )

        bridge_terms.sort(key=lambda item: (-item.label_count, item.term))
        if max_bridge_terms > 0 and len(bridge_terms) > max_bridge_terms:
            bridge_terms = bridge_terms[:max_bridge_terms]
        return bridge_terms

    def _recommend_labels(
        self,
        token_set: set[str],
        label_sets: dict[str, set[str]],
        max_label_recommendations: int,
        similarity_mode: str,
    ) -> list[LabelRecommendation]:
        if max_label_recommendations <= 0:
            return []
        recommendations: list[LabelRecommendation] = []
        for tag, label_terms in label_sets.items():
            shared = token_set & label_terms
            if not shared:
                continue
            overlap = len(shared)
            if similarity_mode == "jaccard":
                union = len(token_set | label_terms)
                score = overlap / union if union else 0.0
            else:
                denom = min(len(token_set), len(label_terms))
                score = overlap / denom if denom else 0.0
            coverage = overlap / len(token_set) if token_set else 0.0
            recommendations.append(
                LabelRecommendation(
                    tag=tag,
                    score=round(score, 6),
                    overlap=overlap,
                    coverage=round(coverage, 6),
                )
            )

        recommendations.sort(key=lambda item: (-item.score, -item.overlap, item.tag))
        if max_label_recommendations > 0 and len(recommendations) > max_label_recommendations:
            recommendations = recommendations[:max_label_recommendations]
        return recommendations

    def _normalize_store_type(self, store_type: str | None) -> str:
        store_value = (store_type or "db").strip().lower()
        if store_value == "sharded":
            store_value = "manifest"
        if store_value not in {"db", "manifest"}:
            raise ToolError("store_type must be db, manifest, or sharded.")
        return store_value

    def _resolve_path(self, raw_path: str) -> Path:
        if not raw_path.strip():
            raise ToolError("path cannot be empty.")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.config.effective_workdir / path
        try:
            return path.resolve()
        except ValueError as exc:
            raise ToolError("Security error: cannot resolve the provided path.") from exc

    def _load_manifest(self, path: Path) -> tuple[dict, list[str]]:
        errors: list[str] = []
        if not path.exists():
            errors.append(f"Manifest not found: {path}")
            return {}, errors
        try:
            return json.loads(path.read_text("utf-8")), errors
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"Invalid manifest file: {exc}")
            return {}, errors

    def _search_manifest(
        self,
        manifest_path: Path,
        embedding_model: str,
        query_vec: list[float],
        max_results: int,
        max_chars: int,
        min_score: float | None,
    ) -> tuple[list[_VectorRow], list[str]]:
        manifest, errors = self._load_manifest(manifest_path)
        if not manifest:
            return [], errors
        manifest_model = manifest.get("embedding_model")
        if manifest_model and manifest_model != embedding_model:
            errors.append(
                "Embedding model mismatch with manifest. "
                "Use the matching embedding_model or rebuild the manifest."
            )
            return [], errors

        heap: list[tuple[float, _VectorRow]] = []
        for shard in manifest.get("shards", []):
            shard_path = Path(shard.get("path", ""))
            if not shard_path.is_absolute():
                shard_path = manifest_path.parent / shard_path
            if not shard_path.exists():
                errors.append(f"Shard not found: {shard_path}")
                continue
            try:
                shard_rows = self._search_db(
                    shard_path,
                    query_vec,
                    max_results,
                    max_chars,
                    min_score,
                )
                for row in shard_rows:
                    row.store_path = str(shard_path)
                    if len(heap) < max_results:
                        heappush(heap, (row.score, row))
                    else:
                        heappushpop(heap, (row.score, row))
            except ToolError as exc:
                errors.append(str(exc))

        rows_sorted = [
            item for _, item in sorted(heap, key=lambda r: r[0], reverse=True)
        ]
        return rows_sorted, errors

    def _search_db(
        self,
        db_path: Path,
        query_vec: list[float],
        max_results: int,
        max_chars: int,
        min_score: float | None,
    ) -> list[_VectorRow]:
        if not db_path.exists():
            raise ToolError(f"Store not found: {db_path}")
        if db_path.is_dir():
            raise ToolError(f"Store path is a directory: {db_path}")

        results: list[tuple[float, _VectorRow]] = []
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute(
                """
                SELECT source_path, chunk_index, unit, start_index, end_index, content, embedding, embedding_dim
                FROM pdf_chunks
                """
            )
            for row in cursor:
                embedding_dim = row[7] if len(row) > 7 else None
                if embedding_dim and embedding_dim != len(query_vec):
                    continue
                embedding = self._unpack_embedding(row[6])
                if not embedding:
                    continue
                score = self._dot(query_vec, embedding)
                if min_score is not None and score < min_score:
                    continue
                content = row[5][:max_chars]
                item = _VectorRow(
                    score=score,
                    source_path=row[0],
                    chunk_index=row[1],
                    unit=row[2],
                    start_index=row[3],
                    end_index=row[4],
                    content=content,
                    store_label=None,
                    store_path=str(db_path),
                )
                if len(results) < max_results:
                    heappush(results, (score, item))
                else:
                    heappushpop(results, (score, item))

        return [item for _, item in sorted(results, key=lambda r: r[0], reverse=True)]

    def _embed_text(self, model: str, text: str) -> list[float]:
        payload = json.dumps({"model": model, "prompt": text}).encode("utf-8")
        url = self.config.ollama_url.rstrip("/") + "/api/embeddings"
        req = request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise ToolError(f"Ollama/GPT-OSS embeddings failed: {exc}") from exc

        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            raise ToolError("Invalid embeddings response from Ollama/GPT-OSS.")

        return self._normalize_embedding([float(x) for x in embedding])

    def _normalize_embedding(self, embedding: list[float]) -> list[float]:
        norm = sum(x * x for x in embedding) ** 0.5
        if norm == 0:
            return embedding
        return [x / norm for x in embedding]

    def _unpack_embedding(self, blob: bytes) -> list[float]:
        arr = array("f")
        arr.frombytes(blob)
        return list(arr)

    def _dot(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            return 0.0
        return sum(l * r for l, r in zip(left, right))

    def _process_vector_queries(
        self,
        vector_queries: list[VectorQuery],
        vector_stores: list[VectorStore],
        rules: list[GenreRule],
        label_kinds: dict[str, str],
        label_token_counts: dict[str, dict[str, int]],
        label_sources: dict[str, set[str]],
        embedding_model: str,
        max_vector_results_per_store: int,
        max_vector_total_results: int,
        max_result_chars: int,
        min_word_length: int,
        max_words_per_chunk: int,
        preview_chars: int,
        similarity_mode: str,
        min_label_similarity: float,
        errors: list[str],
    ) -> tuple[list[_ChunkRecord], bool]:
        if not vector_queries:
            return [], False

        query_cache: dict[str, list[float]] = {}
        store_paths: list[Path | None] = []
        for idx, store in enumerate(vector_stores, start=1):
            try:
                store_paths.append(self._resolve_path(store.path))
            except ToolError as exc:
                errors.append(f"vector_stores[{idx}]: {exc}")
                store_paths.append(None)

        output: list[_ChunkRecord] = []
        truncated = False
        total_results = 0

        for query_idx, query in enumerate(vector_queries, start=1):
            query_text = query.query.strip()
            if not query_text:
                raise ToolError(f"vector_queries[{query_idx}] query cannot be empty.")
            top_k = query.top_k if query.top_k is not None else max_vector_results_per_store
            if top_k <= 0:
                raise ToolError(f"vector_queries[{query_idx}] top_k must be positive.")

            cache_key = f"{embedding_model}:{query_text}"
            if cache_key not in query_cache:
                query_cache[cache_key] = self._embed_text(embedding_model, query_text)
            query_vec = query_cache[cache_key]

            for store_idx, store in enumerate(vector_stores, start=1):
                if total_results >= max_vector_total_results:
                    return output, True
                path = store_paths[store_idx - 1]
                if path is None:
                    continue
                try:
                    store_type = self._normalize_store_type(store.store_type)
                except ToolError as exc:
                    errors.append(f"vector_stores[{store_idx}]: {exc}")
                    continue

                store_label = store.label or path.name
                store_max = store.max_results if store.max_results is not None else max_vector_results_per_store
                if store_max <= 0:
                    errors.append(f"vector_stores[{store_idx}]: max_results must be positive.")
                    continue
                store_max = min(store_max, max_vector_results_per_store, top_k)

                store_max_chars = store.max_result_chars if store.max_result_chars is not None else max_result_chars
                if store_max_chars <= 0:
                    errors.append(f"vector_stores[{store_idx}]: max_result_chars must be positive.")
                    continue

                store_min_score = store.min_score if store.min_score is not None else query.min_score

                rows: list[_VectorRow] = []
                if store_type == "db":
                    try:
                        rows = self._search_db(
                            path,
                            query_vec,
                            store_max,
                            store_max_chars,
                            store_min_score,
                        )
                    except ToolError as exc:
                        errors.append(f"{path}: {exc}")
                        continue
                else:
                    rows, manifest_errors = self._search_manifest(
                        path,
                        embedding_model,
                        query_vec,
                        store_max,
                        store_max_chars,
                        store_min_score,
                    )
                    errors.extend(manifest_errors)

                remaining = max_vector_total_results - total_results
                if remaining <= 0:
                    return output, True
                if len(rows) > remaining:
                    rows = rows[:remaining]
                    truncated = True

                for row in rows:
                    if total_results >= max_vector_total_results:
                        truncated = True
                        break
                    item = GenreItem(
                        id=None,
                        title=Path(row.source_path).name if row.source_path else None,
                        content=None,
                        path=row.source_path,
                        source=store_label,
                        genres=None,
                        classifications=None,
                    )
                    tags = self._resolve_item_tags(item, row.source_path, rules, label_kinds)
                    if not tags:
                        tags = [DEFAULT_LABEL]
                        self._add_label_kind(label_kinds, DEFAULT_LABEL, "unknown")

                    tokens, token_set = self._extract_tokens(
                        row.content,
                        min_word_length,
                        max_words_per_chunk,
                    )
                    preview = self._preview_text(row.content, preview_chars)
                    record = _ChunkRecord(
                        item_index=0,
                        item_id=f"vector:{query_idx}",
                        chunk_index=row.chunk_index,
                        source_path=row.source_path,
                        store_label=store_label,
                        store_path=row.store_path or str(path),
                        vector_score=row.score,
                        tags=tags,
                        token_counts=tokens,
                        token_set=token_set,
                        preview=preview,
                    )
                    output.append(record)
                    source_key = row.source_path or f"vector:{query_idx}"
                    self._accumulate_labels(
                        tags,
                        tokens,
                        source_key,
                        label_token_counts,
                        label_sources,
                        label_kinds,
                    )
                    total_results += 1

                if total_results >= max_vector_total_results:
                    truncated = True
                    break

        return output, truncated

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextGenreVectorsArgs):
            return ToolCallDisplay(summary="context_genre_vectors")

        item_count = len(event.args.items or [])
        query_count = len(event.args.vector_queries or [])
        store_count = len(event.args.vector_stores or [])
        summary = (
            f"context_genre_vectors: {item_count} item(s), "
            f"{query_count} query(s)"
        )
        return ToolCallDisplay(
            summary=summary,
            details={
                "item_count": item_count,
                "vector_query_count": query_count,
                "vector_store_count": store_count,
                "chunk_mode": event.args.chunk_mode,
                "chunk_size": event.args.chunk_size,
                "similarity_mode": event.args.similarity_mode,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextGenreVectorsResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = (
            f"Processed {event.result.chunk_count} chunk(s) across "
            f"{event.result.label_count} label(s)"
        )
        warnings = event.result.errors[:]
        if event.result.truncated:
            warnings.append("Output truncated by size or limits")

        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=warnings,
            details={
                "label_count": event.result.label_count,
                "chunk_count": event.result.chunk_count,
                "overlap_count": event.result.overlap_count,
                "bridge_count": event.result.bridge_count,
                "truncated": event.result.truncated,
                "errors": event.result.errors,
                "labels": event.result.labels,
                "chunks": event.result.chunks,
                "label_overlaps": event.result.label_overlaps,
                "bridge_terms": event.result.bridge_terms,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Analyzing genre and classification context"
