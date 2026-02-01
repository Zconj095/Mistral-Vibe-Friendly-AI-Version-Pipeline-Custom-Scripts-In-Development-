
from __future__ import annotations

from dataclasses import dataclass
from array import array
from pathlib import Path
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


WORD_RE = re.compile(r"[A-Za-z0-9_]+")
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

CHUNK_MODES = {"words", "paragraphs", "sentences", "lines", "chars"}


@dataclass
class _VectorDoc:
    doc_id: str
    name: str | None
    source_path: str | None
    store_label: str
    content_parts: list[str]
    size_bytes: int
    included_chunks: int
    token_counts: dict[str, int]
    embedding_sum: list[float] | None
    embedding_count: int
    truncated_content: bool


@dataclass
class _DocumentBuffer:
    doc_id: str
    name: str | None
    source_path: str | None
    source_type: str
    content: str
    size_bytes: int
    chunk_mode: str
    chunk_size: int
    chunk_overlap: int
    embedding: list[float] | None = None


class ContextDenseLayersMultiConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    ollama_url: str = Field(
        default="http://127.0.0.1:11434",
        description="Base URL for the Ollama/GPT-OSS server.",
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        description="Embedding model to use with Ollama/GPT-OSS.",
    )
    default_use_embeddings: bool = Field(
        default=True, description="Use embeddings when not specified."
    )
    max_items: int = Field(default=200, description="Maximum items to process.")
    max_documents: int = Field(
        default=500, description="Maximum total documents to process."
    )
    max_vector_docs: int = Field(
        default=200, description="Maximum documents loaded from vector stores."
    )
    max_vector_chunks_per_doc: int = Field(
        default=200, description="Maximum chunks loaded per vector store document."
    )
    max_source_bytes: int = Field(
        default=5_000_000, description="Maximum size per document (bytes)."
    )
    max_total_bytes: int = Field(
        default=20_000_000, description="Maximum total bytes across documents."
    )
    max_chunk_bytes: int = Field(
        default=200_000, description="Maximum bytes per chunk."
    )
    max_chunks: int = Field(
        default=1000, description="Maximum total chunks returned."
    )
    max_chunks_per_doc: int = Field(
        default=200, description="Maximum chunks per document."
    )
    default_chunk_mode: str = Field(
        default="words",
        description="Default chunking mode: words, paragraphs, sentences, lines, chars.",
    )
    default_chunk_size: int = Field(
        default=200, description="Default chunk size in units."
    )
    default_chunk_overlap: int = Field(
        default=20, description="Default overlap in units."
    )
    preview_chars: int = Field(default=400, description="Preview length per item.")
    min_token_length: int = Field(default=3, description="Minimum token length.")
    max_tokens_per_chunk: int = Field(
        default=200, description="Maximum tokens stored per chunk."
    )
    max_tokens_per_document: int = Field(
        default=500, description="Maximum tokens stored per document."
    )
    max_shared_tokens: int = Field(
        default=10, description="Maximum shared tokens returned per edge."
    )
    min_similarity: float = Field(
        default=0.1, description="Minimum similarity for edges."
    )
    max_neighbors: int = Field(
        default=6, description="Maximum neighbors per node."
    )
    max_edges: int = Field(
        default=2000, description="Maximum total edges returned."
    )
    cluster_min_similarity: float = Field(
        default=0.12, description="Minimum similarity for dense clusters."
    )
    max_clusters: int = Field(default=200, description="Maximum clusters returned.")
    max_cluster_size: int = Field(
        default=20, description="Maximum chunks per cluster."
    )
    max_cluster_keywords: int = Field(
        default=20, description="Maximum keywords stored per cluster."
    )
    cross_document_chunk_similarity: bool = Field(
        default=True,
        description="Allow chunk similarity across documents.",
    )
    max_embedding_chars: int = Field(
        default=8000, description="Maximum characters embedded per document."
    )


class ContextDenseLayersMultiState(BaseToolState):
    pass


class DenseTextItem(BaseModel):
    id: str | None = Field(default=None, description="Optional item id.")
    name: str | None = Field(default=None, description="Optional item name.")
    content: str | None = Field(default=None, description="Inline text content.")
    path: str | None = Field(default=None, description="Path to a text file.")
    source: str | None = Field(default=None, description="Source description.")
    chunk_mode: str | None = Field(
        default=None, description="Override chunk mode for this item."
    )
    chunk_size: int | None = Field(
        default=None, description="Override chunk size for this item."
    )
    chunk_overlap: int | None = Field(
        default=None, description="Override chunk overlap for this item."
    )


class VectorStore(BaseModel):
    path: str = Field(description="Path to a sqlite vector store.")
    store_type: str | None = Field(
        default="auto", description="auto, text, or pdf."
    )
    table: str | None = Field(
        default=None, description="Override table name for chunks."
    )
    label: str | None = Field(default=None, description="Optional store label.")
    max_docs: int | None = Field(
        default=None, description="Override max documents for this store."
    )
    max_chunks_per_doc: int | None = Field(
        default=None, description="Override max chunks per document."
    )


class ContextDenseLayersMultiArgs(BaseModel):
    items: list[DenseTextItem] | None = Field(
        default=None, description="Text items to process."
    )
    vector_stores: list[VectorStore] | None = Field(
        default=None, description="Vector stores containing documents."
    )
    use_embeddings: bool | None = Field(
        default=None, description="Use embeddings for similarity."
    )
    embedding_model: str | None = Field(
        default=None, description="Override embedding model."
    )
    ollama_url: str | None = Field(
        default=None, description="Override Ollama/GPT-OSS URL."
    )
    chunk_mode: str | None = Field(
        default=None, description="Override chunk mode for items."
    )
    chunk_size: int | None = Field(
        default=None, description="Override chunk size for items."
    )
    chunk_overlap: int | None = Field(
        default=None, description="Override chunk overlap for items."
    )
    max_items: int | None = Field(default=None, description="Override max_items.")
    max_documents: int | None = Field(
        default=None, description="Override max_documents."
    )
    max_vector_docs: int | None = Field(
        default=None, description="Override max_vector_docs."
    )
    max_vector_chunks_per_doc: int | None = Field(
        default=None, description="Override max_vector_chunks_per_doc."
    )
    max_source_bytes: int | None = Field(
        default=None, description="Override max_source_bytes."
    )
    max_total_bytes: int | None = Field(
        default=None, description="Override max_total_bytes."
    )
    max_chunk_bytes: int | None = Field(
        default=None, description="Override max_chunk_bytes."
    )
    max_chunks: int | None = Field(default=None, description="Override max_chunks.")
    max_chunks_per_doc: int | None = Field(
        default=None, description="Override max_chunks_per_doc."
    )
    preview_chars: int | None = Field(
        default=None, description="Override preview_chars."
    )
    min_token_length: int | None = Field(
        default=None, description="Override min_token_length."
    )
    max_tokens_per_chunk: int | None = Field(
        default=None, description="Override max_tokens_per_chunk."
    )
    max_tokens_per_document: int | None = Field(
        default=None, description="Override max_tokens_per_document."
    )
    max_shared_tokens: int | None = Field(
        default=None, description="Override max_shared_tokens."
    )
    min_similarity: float | None = Field(
        default=None, description="Override min_similarity."
    )
    max_neighbors: int | None = Field(
        default=None, description="Override max_neighbors."
    )
    max_edges: int | None = Field(default=None, description="Override max_edges.")
    cluster_min_similarity: float | None = Field(
        default=None, description="Override cluster_min_similarity."
    )
    max_clusters: int | None = Field(
        default=None, description="Override max_clusters."
    )
    max_cluster_size: int | None = Field(
        default=None, description="Override max_cluster_size."
    )
    max_cluster_keywords: int | None = Field(
        default=None, description="Override max_cluster_keywords."
    )
    cross_document_chunk_similarity: bool | None = Field(
        default=None, description="Override cross_document_chunk_similarity."
    )
    max_embedding_chars: int | None = Field(
        default=None, description="Override max_embedding_chars."
    )

class DenseDocument(BaseModel):
    index: int
    id: str
    name: str | None
    source_path: str | None
    source_type: str
    chunk_count: int
    token_count: int
    preview: str
    keywords: list[str]
    embedding_used: bool


class DenseChunk(BaseModel):
    index: int
    document_index: int
    unit: str
    start_index: int | None
    end_index: int | None
    preview: str
    token_count: int
    keywords: list[str]
    truncated: bool


class DenseEdge(BaseModel):
    source_kind: str
    source_index: int
    target_kind: str
    target_index: int
    score: float
    relation: str
    shared_tokens: list[str]


class DenseCluster(BaseModel):
    cluster_id: int
    member_indices: list[int]
    center_document_index: int | None
    avg_similarity: float
    keywords: list[str]


class DenseLayer(BaseModel):
    layer: str
    count: int
    description: str


class ProcessStep(BaseModel):
    step: int
    stage: str
    detail: str
    document_indices: list[int]
    chunk_indices: list[int]


class ContextDenseLayersMultiResult(BaseModel):
    documents: list[DenseDocument]
    chunks: list[DenseChunk]
    edges: list[DenseEdge]
    clusters: list[DenseCluster]
    layers: list[DenseLayer]
    process_steps: list[ProcessStep]
    document_count: int
    chunk_count: int
    edge_count: int
    cluster_count: int
    truncated: bool
    embedding_used: bool
    errors: list[str]
    warnings: list[str]


class ContextDenseLayersMulti(
    BaseTool[
        ContextDenseLayersMultiArgs,
        ContextDenseLayersMultiResult,
        ContextDenseLayersMultiConfig,
        ContextDenseLayersMultiState,
    ],
    ToolUIData[ContextDenseLayersMultiArgs, ContextDenseLayersMultiResult],
):
    description: ClassVar[str] = (
        "Build dense context layers across documents and processing stages."
    )

    async def run(
        self, args: ContextDenseLayersMultiArgs
    ) -> ContextDenseLayersMultiResult:
        if not args.items and not args.vector_stores:
            raise ToolError("items or vector_stores is required.")

        max_items = args.max_items if args.max_items is not None else self.config.max_items
        max_documents = (
            args.max_documents
            if args.max_documents is not None
            else self.config.max_documents
        )
        max_vector_docs = (
            args.max_vector_docs
            if args.max_vector_docs is not None
            else self.config.max_vector_docs
        )
        max_vector_chunks_per_doc = (
            args.max_vector_chunks_per_doc
            if args.max_vector_chunks_per_doc is not None
            else self.config.max_vector_chunks_per_doc
        )
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
        max_chunk_bytes = (
            args.max_chunk_bytes
            if args.max_chunk_bytes is not None
            else self.config.max_chunk_bytes
        )
        max_chunks = args.max_chunks if args.max_chunks is not None else self.config.max_chunks
        max_chunks_per_doc = (
            args.max_chunks_per_doc
            if args.max_chunks_per_doc is not None
            else self.config.max_chunks_per_doc
        )
        preview_chars = (
            args.preview_chars
            if args.preview_chars is not None
            else self.config.preview_chars
        )
        min_token_length = (
            args.min_token_length
            if args.min_token_length is not None
            else self.config.min_token_length
        )
        max_tokens_per_chunk = (
            args.max_tokens_per_chunk
            if args.max_tokens_per_chunk is not None
            else self.config.max_tokens_per_chunk
        )
        max_tokens_per_document = (
            args.max_tokens_per_document
            if args.max_tokens_per_document is not None
            else self.config.max_tokens_per_document
        )
        max_shared_tokens = (
            args.max_shared_tokens
            if args.max_shared_tokens is not None
            else self.config.max_shared_tokens
        )
        min_similarity = (
            args.min_similarity
            if args.min_similarity is not None
            else self.config.min_similarity
        )
        max_neighbors = (
            args.max_neighbors
            if args.max_neighbors is not None
            else self.config.max_neighbors
        )
        max_edges = args.max_edges if args.max_edges is not None else self.config.max_edges
        cluster_min_similarity = (
            args.cluster_min_similarity
            if args.cluster_min_similarity is not None
            else self.config.cluster_min_similarity
        )
        max_clusters = (
            args.max_clusters
            if args.max_clusters is not None
            else self.config.max_clusters
        )
        max_cluster_size = (
            args.max_cluster_size
            if args.max_cluster_size is not None
            else self.config.max_cluster_size
        )
        max_cluster_keywords = (
            args.max_cluster_keywords
            if args.max_cluster_keywords is not None
            else self.config.max_cluster_keywords
        )
        cross_document_chunk_similarity = (
            args.cross_document_chunk_similarity
            if args.cross_document_chunk_similarity is not None
            else self.config.cross_document_chunk_similarity
        )
        max_embedding_chars = (
            args.max_embedding_chars
            if args.max_embedding_chars is not None
            else self.config.max_embedding_chars
        )
        default_chunk_mode = (
            args.chunk_mode or self.config.default_chunk_mode
        ).strip().lower()
        default_chunk_size = (
            args.chunk_size if args.chunk_size is not None else self.config.default_chunk_size
        )
        default_chunk_overlap = (
            args.chunk_overlap
            if args.chunk_overlap is not None
            else self.config.default_chunk_overlap
        )

        if max_items <= 0:
            raise ToolError("max_items must be a positive integer.")
        if max_documents <= 0:
            raise ToolError("max_documents must be a positive integer.")
        if max_vector_docs < 0:
            raise ToolError("max_vector_docs must be >= 0.")
        if max_vector_chunks_per_doc <= 0:
            raise ToolError("max_vector_chunks_per_doc must be a positive integer.")
        if max_source_bytes <= 0:
            raise ToolError("max_source_bytes must be a positive integer.")
        if max_total_bytes <= 0:
            raise ToolError("max_total_bytes must be a positive integer.")
        if max_chunk_bytes <= 0:
            raise ToolError("max_chunk_bytes must be a positive integer.")
        if max_chunks <= 0:
            raise ToolError("max_chunks must be a positive integer.")
        if max_chunks_per_doc <= 0:
            raise ToolError("max_chunks_per_doc must be a positive integer.")
        if preview_chars < 0:
            raise ToolError("preview_chars must be >= 0.")
        if min_token_length <= 0:
            raise ToolError("min_token_length must be a positive integer.")
        if max_tokens_per_chunk < 0:
            raise ToolError("max_tokens_per_chunk must be >= 0.")
        if max_tokens_per_document < 0:
            raise ToolError("max_tokens_per_document must be >= 0.")
        if max_shared_tokens < 0:
            raise ToolError("max_shared_tokens must be >= 0.")
        if min_similarity < 0:
            raise ToolError("min_similarity must be >= 0.")
        if max_neighbors < 0:
            raise ToolError("max_neighbors must be >= 0.")
        if max_edges < 0:
            raise ToolError("max_edges must be >= 0.")
        if cluster_min_similarity < 0:
            raise ToolError("cluster_min_similarity must be >= 0.")
        if max_clusters < 0:
            raise ToolError("max_clusters must be >= 0.")
        if max_cluster_size <= 0:
            raise ToolError("max_cluster_size must be a positive integer.")
        if max_cluster_keywords < 0:
            raise ToolError("max_cluster_keywords must be >= 0.")
        if default_chunk_mode not in CHUNK_MODES:
            raise ToolError("chunk_mode must be words, paragraphs, sentences, lines, or chars.")

        self._validate_chunk_settings(
            default_chunk_mode, default_chunk_size, default_chunk_overlap
        )

        if args.items and len(args.items) > max_items:
            raise ToolError(
                f"items exceeds max_items ({len(args.items)} > {max_items})."
            )

        if args.use_embeddings is None:
            use_embeddings = self.config.default_use_embeddings or bool(args.vector_stores)
        else:
            use_embeddings = bool(args.use_embeddings)

        embedding_model = (args.embedding_model or self.config.embedding_model).strip()
        ollama_url = (args.ollama_url or self.config.ollama_url).strip()
        if use_embeddings and not embedding_model:
            raise ToolError("embedding_model cannot be empty when embeddings are enabled.")
        if use_embeddings and not ollama_url:
            raise ToolError("ollama_url cannot be empty when embeddings are enabled.")

        documents: list[_DocumentBuffer] = []
        errors: list[str] = []
        warnings: list[str] = []
        total_bytes = 0
        truncated = False
        if args.items:
            for idx, item in enumerate(args.items, start=1):
                try:
                    content, source_path, size_bytes = self._load_item_content(
                        item, max_source_bytes
                    )
                    if content is None:
                        raise ToolError("Item has no content to process.")
                    if total_bytes + size_bytes > max_total_bytes:
                        truncated = True
                        warnings.append("Total byte limit reached; stopping item ingestion.")
                        break

                    chunk_mode, chunk_size, chunk_overlap = self._resolve_chunk_settings(
                        item.chunk_mode,
                        item.chunk_size,
                        item.chunk_overlap,
                        default_chunk_mode,
                        default_chunk_size,
                        default_chunk_overlap,
                    )

                    doc_id = self._build_doc_id(item, source_path, idx)
                    documents.append(
                        _DocumentBuffer(
                            doc_id=doc_id,
                            name=item.name,
                            source_path=source_path,
                            source_type=item.source or "item",
                            content=content,
                            size_bytes=size_bytes,
                            chunk_mode=chunk_mode,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        )
                    )
                    total_bytes += size_bytes
                except ToolError as exc:
                    errors.append(f"item[{idx}]: {exc}")
                except Exception as exc:
                    errors.append(f"item[{idx}]: {exc}")

        if args.vector_stores:
            (
                vector_docs,
                total_bytes,
                vector_truncated,
                vector_errors,
                vector_warnings,
            ) = self._load_vector_docs(
                args.vector_stores,
                max_documents=max_documents,
                existing_documents=len(documents),
                max_vector_docs=max_vector_docs,
                max_vector_chunks_per_doc=max_vector_chunks_per_doc,
                max_source_bytes=max_source_bytes,
                max_total_bytes=max_total_bytes,
                min_token_length=min_token_length,
                use_embeddings=use_embeddings,
                total_bytes=total_bytes,
                default_chunk_mode=default_chunk_mode,
                default_chunk_size=default_chunk_size,
                default_chunk_overlap=default_chunk_overlap,
            )
            documents.extend(vector_docs)
            errors.extend(vector_errors)
            warnings.extend(vector_warnings)
            if vector_truncated:
                truncated = True

        if len(documents) > max_documents:
            truncated = True
            warnings.append("Document limit reached; truncating document list.")
            documents = documents[:max_documents]

        if not documents:
            raise ToolError("No documents to process.")

        chunks: list[DenseChunk] = []
        chunk_token_sets: list[set[str]] = []
        chunk_token_counts: list[dict[str, int]] = []
        chunk_doc_indices: list[int] = []
        doc_chunk_indices: dict[int, list[int]] = {}
        doc_infos: list[dict[str, object]] = []
        doc_token_sets: list[set[str]] = []
        stop_chunks = False

        for doc_index, doc in enumerate(documents, start=1):
            doc_token_counts: dict[str, int] = {}
            doc_chunk_list: list[int] = []
            doc_chunk_count = 0

            chunk_entries, doc_truncated = self._chunk_text(
                doc.content,
                doc.chunk_mode,
                doc.chunk_size,
                doc.chunk_overlap,
                max_chunks_per_doc,
                max_chunk_bytes,
            )
            if doc_truncated:
                truncated = True
                warnings.append(
                    f"document[{doc_index}] truncated by max_chunks_per_doc limit."
                )

            for chunk_text, start_index, end_index, unit, chunk_truncated in chunk_entries:
                if max_chunks > 0 and len(chunks) >= max_chunks:
                    truncated = True
                    warnings.append("Global chunk limit reached; stopping chunking.")
                    stop_chunks = True
                    break

                token_counts = self._extract_token_counts(
                    chunk_text, min_token_length
                )
                token_count = sum(token_counts.values())
                token_set = self._select_token_set(
                    token_counts, max_tokens_per_chunk
                )
                keywords = self._select_keywords(
                    token_counts, max_tokens_per_chunk
                )

                chunk_index = len(chunks) + 1
                chunks.append(
                    DenseChunk(
                        index=chunk_index,
                        document_index=doc_index,
                        unit=unit,
                        start_index=start_index,
                        end_index=end_index,
                        preview=self._preview_text(chunk_text, preview_chars),
                        token_count=token_count,
                        keywords=keywords,
                        truncated=chunk_truncated,
                    )
                )
                chunk_token_sets.append(token_set)
                chunk_token_counts.append(token_counts)
                chunk_doc_indices.append(doc_index)
                doc_chunk_list.append(chunk_index)
                doc_chunk_count += 1

                for token, count in token_counts.items():
                    doc_token_counts[token] = doc_token_counts.get(token, 0) + count

                if chunk_truncated:
                    warnings.append(
                        f"document[{doc_index}] chunk[{chunk_index}] trimmed by max_chunk_bytes."
                    )

            doc_chunk_indices[doc_index] = doc_chunk_list
            doc_token_sets.append(
                self._select_token_set(doc_token_counts, max_tokens_per_document)
            )
            doc_infos.append(
                {
                    "doc": doc,
                    "chunk_count": doc_chunk_count,
                    "token_counts": doc_token_counts,
                    "token_count": sum(doc_token_counts.values()),
                    "keywords": self._select_keywords(
                        doc_token_counts, max_tokens_per_document
                    ),
                    "preview": self._preview_text(doc.content, preview_chars),
                }
            )

            if stop_chunks:
                break

        if stop_chunks:
            warnings.append("Some documents were not processed due to chunk limits.")
        doc_embeddings: list[list[float] | None] = []
        embedding_used = False
        embedding_cache: dict[str, list[float]] = {}

        if use_embeddings:
            for index, info in enumerate(doc_infos, start=1):
                doc = info["doc"]
                if doc.embedding is None:
                    embedding_text = self._trim_text(
                        doc.content, max_embedding_chars
                    )
                    if embedding_text:
                        try:
                            cached = embedding_cache.get(embedding_text)
                            if cached is None:
                                cached = self._embed_text(
                                    ollama_url, embedding_model, embedding_text
                                )
                                embedding_cache[embedding_text] = cached
                            doc.embedding = cached
                        except ToolError as exc:
                            errors.append(f"embedding[{index}]: {exc}")

                if doc.embedding is not None:
                    doc_embeddings.append(doc.embedding)
                    embedding_used = True
                else:
                    doc_embeddings.append(None)
        else:
            doc_embeddings = [None] * len(doc_infos)

        documents_out: list[DenseDocument] = []
        for index, info in enumerate(doc_infos, start=1):
            doc = info["doc"]
            documents_out.append(
                DenseDocument(
                    index=index,
                    id=str(doc.doc_id),
                    name=doc.name,
                    source_path=doc.source_path,
                    source_type=doc.source_type,
                    chunk_count=int(info["chunk_count"]),
                    token_count=int(info["token_count"]),
                    preview=str(info["preview"]),
                    keywords=list(info["keywords"]),
                    embedding_used=bool(use_embeddings and doc.embedding is not None),
                )
            )

        doc_edges = self._build_doc_edges(
            doc_token_sets,
            doc_embeddings,
            use_embeddings,
            min_similarity,
            max_neighbors,
            max_shared_tokens,
        )

        chunk_adjacent_edges = self._build_adjacent_edges(
            doc_chunk_indices, chunk_token_sets, max_shared_tokens
        )
        pair_scores: dict[tuple[int, int], float] = {}
        if len(chunk_token_sets) > 1:
            pair_threshold = min(min_similarity, cluster_min_similarity)
            pair_scores = self._chunk_pair_scores(
                chunk_token_sets,
                chunk_doc_indices,
                pair_threshold,
                cross_document_chunk_similarity,
            )

        chunk_similarity_edges = self._build_chunk_similarity_edges(
            pair_scores,
            chunk_token_sets,
            min_similarity,
            max_neighbors,
            max_shared_tokens,
        )

        clusters, cluster_truncated = self._build_clusters(
            pair_scores,
            chunk_token_counts,
            chunk_doc_indices,
            cluster_min_similarity,
            max_cluster_size,
            max_clusters,
            max_cluster_keywords,
        )
        if cluster_truncated:
            truncated = True

        edges, edge_truncated = self._limit_edges(
            chunk_adjacent_edges,
            doc_edges + chunk_similarity_edges,
            max_edges,
        )
        if edge_truncated:
            truncated = True

        process_steps = self._build_process_steps(
            len(documents_out),
            len(chunks),
            len(doc_edges),
            len(chunk_adjacent_edges) + len(chunk_similarity_edges),
            len(clusters),
        )

        layers = [
            DenseLayer(
                layer="documents",
                count=len(documents_out),
                description="Document summaries with keywords and signals.",
            ),
            DenseLayer(
                layer="chunks",
                count=len(chunks),
                description="Chunked document units for dense context.",
            ),
            DenseLayer(
                layer="edges",
                count=len(edges),
                description="Adjacency and similarity links across layers.",
            ),
            DenseLayer(
                layer="clusters",
                count=len(clusters),
                description="Dense similarity groups built from chunk signals.",
            ),
            DenseLayer(
                layer="processes",
                count=len(process_steps),
                description="Processing stages used to build the layers.",
            ),
        ]

        return ContextDenseLayersMultiResult(
            documents=documents_out,
            chunks=chunks,
            edges=edges,
            clusters=clusters,
            layers=layers,
            process_steps=process_steps,
            document_count=len(documents_out),
            chunk_count=len(chunks),
            edge_count=len(edges),
            cluster_count=len(clusters),
            truncated=truncated,
            embedding_used=embedding_used,
            errors=errors,
            warnings=warnings,
        )
    def _build_doc_id(
        self, item: DenseTextItem, source_path: str | None, index: int
    ) -> str:
        if item.id:
            return str(item.id)
        if item.name:
            return str(item.name)
        if source_path:
            return str(source_path)
        return f"item-{index}"

    def _load_item_content(
        self, item: DenseTextItem, max_source_bytes: int
    ) -> tuple[str | None, str | None, int]:
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

        return None, None, 0

    def _resolve_chunk_settings(
        self,
        chunk_mode: str | None,
        chunk_size: int | None,
        chunk_overlap: int | None,
        default_mode: str,
        default_size: int,
        default_overlap: int,
    ) -> tuple[str, int, int]:
        mode = (chunk_mode or default_mode).strip().lower()
        if mode not in CHUNK_MODES:
            raise ToolError("chunk_mode must be words, paragraphs, sentences, lines, or chars.")
        size = default_size if chunk_size is None else chunk_size
        overlap = default_overlap if chunk_overlap is None else chunk_overlap
        self._validate_chunk_settings(mode, size, overlap)
        return mode, size, overlap

    def _validate_chunk_settings(self, mode: str, size: int, overlap: int) -> None:
        if size <= 0:
            raise ToolError("chunk_size must be a positive integer.")
        if overlap < 0:
            raise ToolError("chunk_overlap must be >= 0.")
        if overlap >= size:
            raise ToolError("chunk_overlap must be smaller than chunk_size.")

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

    def _preview_text(self, text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def _trim_text(self, text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        data = text.encode("utf-8")
        if len(data) <= max_chars:
            return text
        head_bytes = max_chars // 2
        tail_bytes = max_chars - head_bytes
        head = data[:head_bytes].decode("utf-8", errors="ignore")
        tail = data[-tail_bytes:].decode("utf-8", errors="ignore")
        return f"{head}\n...\n{tail}"

    def _extract_token_counts(self, text: str, min_len: int) -> dict[str, int]:
        tokens: dict[str, int] = {}
        for match in WORD_RE.findall(text.lower()):
            if len(match) < min_len:
                continue
            if match.isdigit():
                continue
            if match in STOPWORDS:
                continue
            tokens[match] = tokens.get(match, 0) + 1
        return tokens

    def _select_keywords(
        self, token_counts: dict[str, int], max_tokens: int
    ) -> list[str]:
        if not token_counts:
            return []
        ordered = sorted(token_counts.items(), key=lambda item: (-item[1], item[0]))
        if max_tokens > 0:
            ordered = ordered[:max_tokens]
        return [token for token, _ in ordered]

    def _select_token_set(
        self, token_counts: dict[str, int], max_tokens: int
    ) -> set[str]:
        if not token_counts:
            return set()
        ordered = sorted(token_counts.items(), key=lambda item: (-item[1], item[0]))
        if max_tokens > 0:
            ordered = ordered[:max_tokens]
        return {token for token, _ in ordered}

    def _update_token_counts(
        self, token_counts: dict[str, int], text: str, min_len: int
    ) -> None:
        for match in WORD_RE.findall(text.lower()):
            if len(match) < min_len:
                continue
            if match.isdigit():
                continue
            if match in STOPWORDS:
                continue
            token_counts[match] = token_counts.get(match, 0) + 1

    def _chunk_text(
        self,
        content: str,
        mode: str,
        size: int,
        overlap: int,
        max_chunks: int,
        max_chunk_bytes: int,
    ) -> tuple[list[tuple[str, int | None, int | None, str, bool]], bool]:
        if mode not in CHUNK_MODES:
            raise ToolError("chunk_mode must be words, paragraphs, sentences, lines, or chars.")
        self._validate_chunk_settings(mode, size, overlap)

        truncated = False
        chunks: list[tuple[str, int | None, int | None, str, bool]] = []

        if mode == "chars":
            step = size - overlap
            index = 0
            chunk_index = 0
            while index < len(content) and chunk_index < max_chunks:
                chunk_text = content[index : index + size]
                if not chunk_text:
                    break
                chunk_text, trimmed = self._limit_chunk_bytes(
                    chunk_text, max_chunk_bytes
                )
                end_index = index + len(chunk_text) - 1
                chunks.append((chunk_text, index, end_index, "chars", trimmed))
                chunk_index += 1
                index += step
            if index < len(content):
                truncated = True
            return chunks, truncated

        units: list[str] = []
        unit_label = mode
        joiner = "\n"
        if mode == "words":
            units = content.split()
            joiner = " "
        elif mode == "sentences":
            units = [part for part in SENTENCE_RE.split(content) if part.strip()]
            joiner = " "
        elif mode == "paragraphs":
            units = self._split_paragraphs(content)
            joiner = "\n\n"
        elif mode == "lines":
            units = content.splitlines()
            joiner = "\n"

        step = size - overlap
        index = 0
        chunk_index = 0
        while index < len(units) and chunk_index < max_chunks:
            subset = units[index : index + size]
            if not subset:
                break
            chunk_text = joiner.join(subset)
            chunk_text, trimmed = self._limit_chunk_bytes(
                chunk_text, max_chunk_bytes
            )
            start_index = index + 1
            end_index = start_index + len(subset) - 1
            chunks.append((chunk_text, start_index, end_index, unit_label, trimmed))
            chunk_index += 1
            index += step

        if index < len(units):
            truncated = True
        return chunks, truncated

    def _split_paragraphs(self, content: str) -> list[str]:
        paragraphs: list[str] = []
        current: list[str] = []
        for line in content.splitlines():
            if line.strip():
                current.append(line)
            elif current:
                paragraphs.append("\n".join(current))
                current = []
        if current:
            paragraphs.append("\n".join(current))
        return paragraphs

    def _limit_chunk_bytes(self, text: str, max_bytes: int) -> tuple[str, bool]:
        if max_bytes <= 0:
            return "", True
        data = text.encode("utf-8")
        if len(data) <= max_bytes:
            return text, False
        trimmed = data[:max_bytes].decode("utf-8", errors="ignore")
        return trimmed, True
    def _load_vector_docs(
        self,
        stores: list[VectorStore],
        *,
        max_documents: int,
        existing_documents: int,
        max_vector_docs: int,
        max_vector_chunks_per_doc: int,
        max_source_bytes: int,
        max_total_bytes: int,
        min_token_length: int,
        use_embeddings: bool,
        total_bytes: int,
        default_chunk_mode: str,
        default_chunk_size: int,
        default_chunk_overlap: int,
    ) -> tuple[list[_DocumentBuffer], int, bool, list[str], list[str]]:
        vector_docs: list[_DocumentBuffer] = []
        errors: list[str] = []
        warnings: list[str] = []
        truncated = False
        doc_map: dict[str, _VectorDoc] = {}
        doc_order: list[str] = []
        total_docs = existing_documents
        stop_all = False

        for idx, store in enumerate(stores, start=1):
            store_label = store.label or Path(store.path).name
            store_errors: list[str] = []
            try:
                path = self._resolve_path(store.path)
            except ToolError as exc:
                errors.append(f"vector_stores[{idx}]: {exc}")
                continue
            if not path.exists():
                errors.append(f"vector_stores[{idx}]: store not found: {path}")
                continue
            if path.is_dir():
                errors.append(f"vector_stores[{idx}]: store path is a directory: {path}")
                continue

            store_type = (store.store_type or "auto").strip().lower()
            if store_type not in {"auto", "text", "pdf"}:
                errors.append(
                    f"vector_stores[{idx}]: store_type must be auto, text, or pdf."
                )
                continue

            store_max_docs = (
                store.max_docs if store.max_docs is not None else max_vector_docs
            )
            store_max_chunks = (
                store.max_chunks_per_doc
                if store.max_chunks_per_doc is not None
                else max_vector_chunks_per_doc
            )
            if store_max_docs <= 0:
                errors.append(
                    f"vector_stores[{idx}]: max_docs must be a positive integer."
                )
                continue
            if store_max_chunks <= 0:
                errors.append(
                    f"vector_stores[{idx}]: max_chunks_per_doc must be a positive integer."
                )
                continue

            try:
                conn = sqlite3.connect(str(path))
            except Exception as exc:
                errors.append(f"vector_stores[{idx}]: failed to open store: {exc}")
                continue

            try:
                table = store.table or self._select_chunk_table(conn, store_type)
                if not table:
                    errors.append(
                        f"vector_stores[{idx}]: no valid chunks table in {path}"
                    )
                    continue

                cursor = conn.execute(
                    f"SELECT source_path, chunk_index, unit, start_index, end_index, content, embedding, embedding_dim FROM {table} ORDER BY source_path, chunk_index"
                )
                store_doc_count = 0
                for row in cursor:
                    source_path = row[0] or ""
                    doc_id = f"{store_label}|{source_path}"
                    doc = doc_map.get(doc_id)
                    if doc is None:
                        if (
                            total_docs >= max_documents
                            or len(doc_order) >= max_vector_docs
                            or store_doc_count >= store_max_docs
                        ):
                            truncated = True
                            stop_all = True
                            break
                        doc = _VectorDoc(
                            doc_id=doc_id,
                            name=Path(source_path).name if source_path else None,
                            source_path=source_path or None,
                            store_label=store_label,
                            content_parts=[],
                            size_bytes=0,
                            included_chunks=0,
                            token_counts={},
                            embedding_sum=None,
                            embedding_count=0,
                            truncated_content=False,
                        )
                        doc_map[doc_id] = doc
                        doc_order.append(doc_id)
                        total_docs += 1
                        store_doc_count += 1

                    if doc.included_chunks >= store_max_chunks:
                        doc.truncated_content = True
                        continue

                    content = row[5] or ""
                    if not content:
                        continue
                    content_bytes = len(content.encode("utf-8"))
                    if doc.size_bytes + content_bytes > max_source_bytes:
                        doc.truncated_content = True
                        continue
                    if total_bytes + content_bytes > max_total_bytes:
                        truncated = True
                        stop_all = True
                        break

                    doc.content_parts.append(content)
                    doc.size_bytes += content_bytes
                    total_bytes += content_bytes
                    doc.included_chunks += 1
                    self._update_token_counts(doc.token_counts, content, min_token_length)

                    if use_embeddings and doc.embedding_count < store_max_chunks:
                        blob = row[6] if len(row) > 6 else None
                        dim = row[7] if len(row) > 7 else None
                        if blob:
                            embedding = self._unpack_embedding(blob)
                            if embedding:
                                if dim and dim != len(embedding):
                                    continue
                                self._add_embedding(doc, embedding)

                if stop_all:
                    break
            except sqlite3.Error as exc:
                store_errors.append(f"Failed reading store {path}: {exc}")
            finally:
                conn.close()

            errors.extend(store_errors)

            if stop_all:
                break

        for doc_id in doc_order:
            doc = doc_map[doc_id]
            content = "\n\n".join(doc.content_parts)
            if not content:
                warnings.append(f"vector_store doc empty: {doc.source_path}")
                continue

            embedding: list[float] | None = None
            if use_embeddings and doc.embedding_sum and doc.embedding_count > 0:
                averaged = [
                    value / doc.embedding_count for value in doc.embedding_sum
                ]
                embedding = self._normalize_embedding(averaged)

            if doc.truncated_content:
                warnings.append(f"vector_store doc truncated: {doc.source_path}")

            vector_docs.append(
                _DocumentBuffer(
                    doc_id=doc.doc_id,
                    name=doc.name,
                    source_path=doc.source_path,
                    source_type=f"vector_store:{doc.store_label}",
                    content=content,
                    size_bytes=doc.size_bytes,
                    chunk_mode=default_chunk_mode,
                    chunk_size=default_chunk_size,
                    chunk_overlap=default_chunk_overlap,
                    embedding=embedding,
                )
            )

        return vector_docs, total_bytes, truncated, errors, warnings

    def _select_chunk_table(self, conn: sqlite3.Connection, store_type: str) -> str | None:
        if store_type == "text":
            return "text_chunks" if self._table_exists(conn, "text_chunks") else None
        if store_type == "pdf":
            return "pdf_chunks" if self._table_exists(conn, "pdf_chunks") else None
        if self._table_exists(conn, "text_chunks"):
            return "text_chunks"
        if self._table_exists(conn, "pdf_chunks"):
            return "pdf_chunks"
        return None

    def _table_exists(self, conn: sqlite3.Connection, name: str) -> bool:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        ).fetchone()
        return bool(row)

    def _embed_text(self, ollama_url: str, model: str, text: str) -> list[float]:
        payload = json.dumps({"model": model, "prompt": text}).encode("utf-8")
        url = ollama_url.rstrip("/") + "/api/embeddings"
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

    def _normalize_embedding(self, embedding: list[float]) -> list[float]:
        norm = sum(value * value for value in embedding) ** 0.5
        if norm == 0:
            return embedding
        return [value / norm for value in embedding]

    def _unpack_embedding(self, blob: bytes) -> list[float]:
        arr = array("f")
        try:
            arr.frombytes(blob)
        except Exception:
            return []
        return list(arr)

    def _add_embedding(self, doc: _VectorDoc, embedding: list[float]) -> None:
        if doc.embedding_sum is None:
            doc.embedding_sum = [0.0] * len(embedding)
        if len(embedding) != len(doc.embedding_sum):
            return
        for index, value in enumerate(embedding):
            doc.embedding_sum[index] += value
        doc.embedding_count += 1

    def _dot(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            return 0.0
        return sum(a * b for a, b in zip(left, right))

    def _jaccard(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        overlap = len(left & right)
        union = len(left) + len(right) - overlap
        if union == 0:
            return 0.0
        return overlap / union

    def _shared_tokens(
        self, left: set[str], right: set[str], max_shared_tokens: int
    ) -> list[str]:
        shared = sorted(left & right)
        if max_shared_tokens > 0 and len(shared) > max_shared_tokens:
            shared = shared[:max_shared_tokens]
        return shared
    def _build_doc_edges(
        self,
        token_sets: list[set[str]],
        embeddings: list[list[float] | None],
        use_embeddings: bool,
        min_similarity: float,
        max_neighbors: int,
        max_shared_tokens: int,
    ) -> list[DenseEdge]:
        count = len(token_sets)
        if count < 2 or max_neighbors == 0:
            return []

        neighbors: list[list[tuple[int, float]]] = [[] for _ in range(count)]
        for i in range(count):
            for j in range(i + 1, count):
                score = self._doc_similarity(
                    token_sets[i],
                    token_sets[j],
                    embeddings[i],
                    embeddings[j],
                    use_embeddings,
                )
                if score < min_similarity:
                    continue
                neighbors[i].append((j, score))
                neighbors[j].append((i, score))

        for idx in range(count):
            neighbors[idx].sort(key=lambda item: (-item[1], item[0]))
            if max_neighbors > 0:
                neighbors[idx] = neighbors[idx][:max_neighbors]

        edges: list[DenseEdge] = []
        for i, items in enumerate(neighbors):
            for j, score in items:
                if i >= j:
                    continue
                shared = self._shared_tokens(
                    token_sets[i], token_sets[j], max_shared_tokens
                )
                edges.append(
                    DenseEdge(
                        source_kind="document",
                        source_index=i + 1,
                        target_kind="document",
                        target_index=j + 1,
                        score=round(score, 6),
                        relation="similar",
                        shared_tokens=shared,
                    )
                )
        return edges

    def _doc_similarity(
        self,
        left_tokens: set[str],
        right_tokens: set[str],
        left_embedding: list[float] | None,
        right_embedding: list[float] | None,
        use_embeddings: bool,
    ) -> float:
        if use_embeddings and left_embedding and right_embedding:
            return self._dot(left_embedding, right_embedding)
        return self._jaccard(left_tokens, right_tokens)

    def _build_adjacent_edges(
        self,
        doc_chunk_indices: dict[int, list[int]],
        token_sets: list[set[str]],
        max_shared_tokens: int,
    ) -> list[DenseEdge]:
        edges: list[DenseEdge] = []
        for chunk_list in doc_chunk_indices.values():
            for idx in range(len(chunk_list) - 1):
                left = chunk_list[idx]
                right = chunk_list[idx + 1]
                shared = self._shared_tokens(
                    token_sets[left - 1], token_sets[right - 1], max_shared_tokens
                )
                edges.append(
                    DenseEdge(
                        source_kind="chunk",
                        source_index=left,
                        target_kind="chunk",
                        target_index=right,
                        score=1.0,
                        relation="adjacent",
                        shared_tokens=shared,
                    )
                )
        return edges

    def _chunk_pair_scores(
        self,
        token_sets: list[set[str]],
        doc_indices: list[int],
        threshold: float,
        cross_document: bool,
    ) -> dict[tuple[int, int], float]:
        scores: dict[tuple[int, int], float] = {}
        count = len(token_sets)
        if count < 2:
            return scores

        for i in range(count):
            for j in range(i + 1, count):
                if not cross_document and doc_indices[i] != doc_indices[j]:
                    continue
                score = self._jaccard(token_sets[i], token_sets[j])
                if score < threshold:
                    continue
                scores[(i, j)] = score
        return scores

    def _build_chunk_similarity_edges(
        self,
        pair_scores: dict[tuple[int, int], float],
        token_sets: list[set[str]],
        min_similarity: float,
        max_neighbors: int,
        max_shared_tokens: int,
    ) -> list[DenseEdge]:
        count = len(token_sets)
        if count < 2 or max_neighbors == 0:
            return []

        neighbors: list[list[tuple[int, float]]] = [[] for _ in range(count)]
        for (left, right), score in pair_scores.items():
            if score < min_similarity:
                continue
            neighbors[left].append((right, score))
            neighbors[right].append((left, score))

        for idx in range(count):
            neighbors[idx].sort(key=lambda item: (-item[1], item[0]))
            if max_neighbors > 0:
                neighbors[idx] = neighbors[idx][:max_neighbors]

        edges: list[DenseEdge] = []
        for i, items in enumerate(neighbors):
            for j, score in items:
                if i >= j:
                    continue
                shared = self._shared_tokens(
                    token_sets[i], token_sets[j], max_shared_tokens
                )
                edges.append(
                    DenseEdge(
                        source_kind="chunk",
                        source_index=i + 1,
                        target_kind="chunk",
                        target_index=j + 1,
                        score=round(score, 6),
                        relation="similar",
                        shared_tokens=shared,
                    )
                )
        return edges

    def _build_clusters(
        self,
        pair_scores: dict[tuple[int, int], float],
        chunk_token_counts: list[dict[str, int]],
        chunk_doc_indices: list[int],
        min_similarity: float,
        max_cluster_size: int,
        max_clusters: int,
        max_keywords: int,
    ) -> tuple[list[DenseCluster], bool]:
        if not pair_scores or max_clusters == 0:
            return [], False

        count = len(chunk_token_counts)
        adjacency: dict[int, set[int]] = {i: set() for i in range(count)}
        for (left, right), score in pair_scores.items():
            if score < min_similarity:
                continue
            adjacency[left].add(right)
            adjacency[right].add(left)

        visited: set[int] = set()
        clusters: list[DenseCluster] = []
        truncated = False

        for node in range(count):
            if node in visited:
                continue
            if not adjacency[node]:
                visited.add(node)
                continue

            stack = [node]
            component: list[int] = []
            visited.add(node)
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)

            if len(component) < 2:
                continue

            component.sort()
            groups: list[list[int]] = []
            if max_cluster_size > 0 and len(component) > max_cluster_size:
                truncated = True
                for start in range(0, len(component), max_cluster_size):
                    groups.append(component[start : start + max_cluster_size])
            else:
                groups = [component]

            for members in groups:
                if max_clusters > 0 and len(clusters) >= max_clusters:
                    truncated = True
                    return clusters, truncated

                avg_similarity = self._cluster_similarity(members, pair_scores)
                keywords = self._cluster_keywords(
                    members, chunk_token_counts, max_keywords
                )
                center_doc = self._cluster_center_doc(members, chunk_doc_indices)
                clusters.append(
                    DenseCluster(
                        cluster_id=len(clusters) + 1,
                        member_indices=[index + 1 for index in members],
                        center_document_index=center_doc,
                        avg_similarity=round(avg_similarity, 6),
                        keywords=keywords,
                    )
                )

        return clusters, truncated

    def _cluster_center_doc(
        self, members: list[int], chunk_doc_indices: list[int]
    ) -> int | None:
        counts: dict[int, int] = {}
        for idx in members:
            doc_index = chunk_doc_indices[idx]
            counts[doc_index] = counts.get(doc_index, 0) + 1
        if not counts:
            return None
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return ranked[0][0]

    def _cluster_similarity(
        self, members: list[int], pair_scores: dict[tuple[int, int], float]
    ) -> float:
        if len(members) < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                left = members[i]
                right = members[j]
                key = (left, right) if left < right else (right, left)
                score = pair_scores.get(key)
                if score is None:
                    continue
                total += score
                count += 1
        if count == 0:
            return 0.0
        return total / count

    def _cluster_keywords(
        self,
        members: list[int],
        chunk_token_counts: list[dict[str, int]],
        max_keywords: int,
    ) -> list[str]:
        combined: dict[str, int] = {}
        for idx in members:
            for token, count in chunk_token_counts[idx].items():
                combined[token] = combined.get(token, 0) + count
        return self._select_keywords(combined, max_keywords)

    def _limit_edges(
        self,
        adjacency_edges: list[DenseEdge],
        other_edges: list[DenseEdge],
        max_edges: int,
    ) -> tuple[list[DenseEdge], bool]:
        if max_edges <= 0:
            return [], False

        edges: list[DenseEdge] = []
        truncated = False
        if adjacency_edges:
            if len(adjacency_edges) >= max_edges:
                return adjacency_edges[:max_edges], True
            edges.extend(adjacency_edges)

        remaining = max_edges - len(edges)
        if remaining <= 0:
            return edges, True

        other_edges.sort(
            key=lambda item: (-item.score, item.source_kind, item.target_kind)
        )
        edges.extend(other_edges[:remaining])
        if len(other_edges) > remaining:
            truncated = True
        return edges, truncated

    def _build_process_steps(
        self,
        document_count: int,
        chunk_count: int,
        doc_edge_count: int,
        chunk_edge_count: int,
        cluster_count: int,
    ) -> list[ProcessStep]:
        doc_sample = list(range(1, min(document_count, 10) + 1))
        chunk_sample = list(range(1, min(chunk_count, 10) + 1))

        steps = [
            ProcessStep(
                step=1,
                stage="ingest",
                detail=f"Loaded {document_count} document(s).",
                document_indices=doc_sample,
                chunk_indices=[],
            ),
            ProcessStep(
                step=2,
                stage="chunk",
                detail=f"Generated {chunk_count} chunk(s).",
                document_indices=[],
                chunk_indices=chunk_sample,
            ),
            ProcessStep(
                step=3,
                stage="connect",
                detail=(
                    f"Built {doc_edge_count} document edge(s) and "
                    f"{chunk_edge_count} chunk edge(s)."
                ),
                document_indices=doc_sample,
                chunk_indices=chunk_sample,
            ),
            ProcessStep(
                step=4,
                stage="cluster",
                detail=f"Formed {cluster_count} dense cluster(s).",
                document_indices=doc_sample,
                chunk_indices=[],
            ),
        ]
        return steps

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextDenseLayersMultiArgs):
            return ToolCallDisplay(summary="context_dense_layers_multi")

        item_count = len(event.args.items or [])
        store_count = len(event.args.vector_stores or [])
        summary = f"context_dense_layers_multi: {item_count} item(s), {store_count} store(s)"
        return ToolCallDisplay(
            summary=summary,
            details={
                "item_count": item_count,
                "store_count": store_count,
                "max_documents": event.args.max_documents,
                "max_chunks": event.args.max_chunks,
                "min_similarity": event.args.min_similarity,
                "cluster_min_similarity": event.args.cluster_min_similarity,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextDenseLayersMultiResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = (
            f"Processed {event.result.document_count} document(s), "
            f"{event.result.chunk_count} chunk(s), "
            f"{event.result.edge_count} edge(s)"
        )
        warnings = event.result.warnings[:]
        warnings.extend(event.result.errors)
        if event.result.truncated:
            warnings.append("Output truncated by size or limits")

        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=warnings,
            details={
                "document_count": event.result.document_count,
                "chunk_count": event.result.chunk_count,
                "edge_count": event.result.edge_count,
                "cluster_count": event.result.cluster_count,
                "truncated": event.result.truncated,
                "embedding_used": event.result.embedding_used,
                "errors": event.result.errors,
                "warnings": event.result.warnings,
                "documents": event.result.documents,
                "chunks": event.result.chunks,
                "edges": event.result.edges,
                "clusters": event.result.clusters,
                "layers": event.result.layers,
                "process_steps": event.result.process_steps,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Building dense context layers"
