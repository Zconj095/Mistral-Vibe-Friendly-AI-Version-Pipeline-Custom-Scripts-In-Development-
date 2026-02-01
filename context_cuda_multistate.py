
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
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
class _DocBuffer:
    doc_id: str
    name: str | None
    source_path: str | None
    source_type: str
    content: str
    size_bytes: int
    chunk_mode: str
    chunk_size: int
    chunk_overlap: int


@dataclass
class _ChunkState:
    token_count: int
    unique_tokens: int
    unique_ratio: float
    line_count: int
    sentence_count: int
    avg_token_length: float


class ContextCudaMultistateConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    default_use_cuda: bool = Field(
        default=True, description="Use CUDA when available."
    )
    require_cuda: bool = Field(
        default=False, description="Fail if CUDA backend is unavailable."
    )
    feature_dim: int = Field(
        default=2048, description="Hashed feature dimension."
    )
    warp_group_size: int = Field(
        default=32, description="Chunk group size for warp-style batching."
    )
    max_pairwise_chunks: int = Field(
        default=2000, description="Maximum chunks for pairwise similarity."
    )
    pairwise_block_size: int = Field(
        default=256, description="Block size for pairwise similarity."
    )
    lexical_weight: float = Field(
        default=0.6, description="Weight for lexical similarity."
    )
    structural_weight: float = Field(
        default=0.2, description="Weight for structural similarity."
    )
    density_weight: float = Field(
        default=0.2, description="Weight for density similarity."
    )
    min_score: float = Field(
        default=0.1, description="Minimum combined score for edges."
    )
    max_neighbors: int = Field(
        default=6, description="Maximum neighbors per chunk."
    )
    max_edges: int = Field(
        default=2000, description="Maximum total edges returned."
    )
    max_items: int = Field(default=200, description="Maximum items to process.")
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
        default=10, description="Maximum shared tokens per edge."
    )


class ContextCudaMultistateState(BaseToolState):
    pass


class CudaDocumentItem(BaseModel):
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


class ContextCudaMultistateArgs(BaseModel):
    items: list[CudaDocumentItem] = Field(
        description="Document items to process."
    )
    use_cuda: bool | None = Field(
        default=None, description="Use CUDA when available."
    )
    require_cuda: bool | None = Field(
        default=None, description="Fail if CUDA backend is unavailable."
    )
    feature_dim: int | None = Field(
        default=None, description="Override feature_dim."
    )
    warp_group_size: int | None = Field(
        default=None, description="Override warp_group_size."
    )
    max_pairwise_chunks: int | None = Field(
        default=None, description="Override max_pairwise_chunks."
    )
    pairwise_block_size: int | None = Field(
        default=None, description="Override pairwise_block_size."
    )
    lexical_weight: float | None = Field(
        default=None, description="Override lexical_weight."
    )
    structural_weight: float | None = Field(
        default=None, description="Override structural_weight."
    )
    density_weight: float | None = Field(
        default=None, description="Override density_weight."
    )
    min_score: float | None = Field(
        default=None, description="Override min_score."
    )
    max_neighbors: int | None = Field(
        default=None, description="Override max_neighbors."
    )
    max_edges: int | None = Field(default=None, description="Override max_edges.")
    max_items: int | None = Field(default=None, description="Override max_items.")
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
    chunk_mode: str | None = Field(
        default=None, description="Override chunk mode for items."
    )
    chunk_size: int | None = Field(
        default=None, description="Override chunk size for items."
    )
    chunk_overlap: int | None = Field(
        default=None, description="Override chunk overlap for items."
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

class CudaDocumentSummary(BaseModel):
    index: int
    id: str
    name: str | None
    source_path: str | None
    source_type: str
    chunk_count: int
    token_count: int
    preview: str
    keywords: list[str]
    line_count: int
    sentence_count: int
    unique_ratio: float
    avg_token_length: float
    truncated: bool


class CudaChunk(BaseModel):
    index: int
    document_index: int
    unit: str
    start_index: int | None
    end_index: int | None
    preview: str
    token_count: int
    unique_tokens: int
    unique_ratio: float
    line_count: int
    sentence_count: int
    avg_token_length: float
    keywords: list[str]
    truncated: bool


class CudaEdge(BaseModel):
    source_index: int
    target_index: int
    score: float
    lexical_score: float
    structural_score: float
    density_score: float
    shared_tokens: list[str]


class CudaWarpGroup(BaseModel):
    group_id: int
    document_index: int
    chunk_indices: list[int]
    avg_token_count: float
    avg_unique_ratio: float
    avg_sentence_count: float


class ContextCudaMultistateResult(BaseModel):
    documents: list[CudaDocumentSummary]
    chunks: list[CudaChunk]
    edges: list[CudaEdge]
    warp_groups: list[CudaWarpGroup]
    document_count: int
    chunk_count: int
    edge_count: int
    warp_group_count: int
    backend: str
    use_cuda: bool
    cuda_available: bool
    truncated: bool
    errors: list[str]
    warnings: list[str]


class ContextCudaMultistate(
    BaseTool[
        ContextCudaMultistateArgs,
        ContextCudaMultistateResult,
        ContextCudaMultistateConfig,
        ContextCudaMultistateState,
    ],
    ToolUIData[ContextCudaMultistateArgs, ContextCudaMultistateResult],
):
    description: ClassVar[str] = (
        "Use CUDA-friendly batches to build multistate chunk reasoning across documents."
    )

    async def run(
        self, args: ContextCudaMultistateArgs
    ) -> ContextCudaMultistateResult:
        if not args.items:
            raise ToolError("items is required.")

        max_items = args.max_items if args.max_items is not None else self.config.max_items
        if max_items <= 0:
            raise ToolError("max_items must be a positive integer.")
        if len(args.items) > max_items:
            raise ToolError(f"items exceeds max_items ({len(args.items)} > {max_items}).")

        use_cuda = (
            self.config.default_use_cuda
            if args.use_cuda is None
            else bool(args.use_cuda)
        )
        require_cuda = (
            self.config.require_cuda
            if args.require_cuda is None
            else bool(args.require_cuda)
        )
        feature_dim = args.feature_dim if args.feature_dim is not None else self.config.feature_dim
        warp_group_size = (
            args.warp_group_size
            if args.warp_group_size is not None
            else self.config.warp_group_size
        )
        max_pairwise_chunks = (
            args.max_pairwise_chunks
            if args.max_pairwise_chunks is not None
            else self.config.max_pairwise_chunks
        )
        pairwise_block_size = (
            args.pairwise_block_size
            if args.pairwise_block_size is not None
            else self.config.pairwise_block_size
        )
        lexical_weight = (
            args.lexical_weight
            if args.lexical_weight is not None
            else self.config.lexical_weight
        )
        structural_weight = (
            args.structural_weight
            if args.structural_weight is not None
            else self.config.structural_weight
        )
        density_weight = (
            args.density_weight
            if args.density_weight is not None
            else self.config.density_weight
        )
        min_score = args.min_score if args.min_score is not None else self.config.min_score
        max_neighbors = (
            args.max_neighbors
            if args.max_neighbors is not None
            else self.config.max_neighbors
        )
        max_edges = args.max_edges if args.max_edges is not None else self.config.max_edges
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

        if feature_dim <= 0:
            raise ToolError("feature_dim must be a positive integer.")
        if warp_group_size <= 0:
            raise ToolError("warp_group_size must be a positive integer.")
        if max_pairwise_chunks <= 0:
            raise ToolError("max_pairwise_chunks must be a positive integer.")
        if pairwise_block_size <= 0:
            raise ToolError("pairwise_block_size must be a positive integer.")
        if lexical_weight < 0 or structural_weight < 0 or density_weight < 0:
            raise ToolError("weights must be >= 0.")
        if min_score < 0:
            raise ToolError("min_score must be >= 0.")
        if max_neighbors < 0:
            raise ToolError("max_neighbors must be >= 0.")
        if max_edges < 0:
            raise ToolError("max_edges must be >= 0.")
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
        if default_chunk_mode not in CHUNK_MODES:
            raise ToolError("chunk_mode must be words, paragraphs, sentences, lines, or chars.")
        if default_chunk_size <= 0:
            raise ToolError("chunk_size must be a positive integer.")
        if default_chunk_overlap < 0:
            raise ToolError("chunk_overlap must be >= 0.")
        if default_chunk_overlap >= default_chunk_size:
            raise ToolError("chunk_overlap must be smaller than chunk_size.")

        backend_name, xp, cuda_available = self._resolve_backend(use_cuda)
        if require_cuda and not cuda_available:
            raise ToolError("CUDA backend requested but cupy is not available.")

        documents, truncated, errors, warnings = self._load_documents(
            args.items,
            max_source_bytes,
            max_total_bytes,
            default_chunk_mode,
            default_chunk_size,
            default_chunk_overlap,
        )
        if not documents:
            raise ToolError("No documents to process.")

        chunks, chunk_states, chunk_token_sets, truncated_chunks = self._chunk_documents(
            documents,
            max_chunk_bytes,
            max_chunks,
            max_chunks_per_doc,
            min_token_length,
            max_tokens_per_chunk,
            preview_chars,
        )
        if truncated_chunks:
            truncated = True

        document_summaries = self._build_document_summaries(
            documents,
            chunks,
            chunk_states,
            max_tokens_per_document,
            min_token_length,
            preview_chars,
        )

        features = self._build_feature_matrix(
            chunk_token_sets, feature_dim
        )

        edges, edge_truncated = self._build_edges(
            features,
            chunk_states,
            chunk_token_sets,
            lexical_weight,
            structural_weight,
            density_weight,
            min_score,
            max_neighbors,
            max_edges,
            max_pairwise_chunks,
            pairwise_block_size,
            max_shared_tokens,
            xp,
            backend_name,
        )
        if edge_truncated:
            truncated = True

        warp_groups = self._build_warp_groups(
            chunks, chunk_states, warp_group_size
        )

        return ContextCudaMultistateResult(
            documents=document_summaries,
            chunks=chunks,
            edges=edges,
            warp_groups=warp_groups,
            document_count=len(document_summaries),
            chunk_count=len(chunks),
            edge_count=len(edges),
            warp_group_count=len(warp_groups),
            backend=backend_name,
            use_cuda=bool(use_cuda),
            cuda_available=cuda_available,
            truncated=truncated,
            errors=errors,
            warnings=warnings,
        )
    def _resolve_backend(self, use_cuda: bool) -> tuple[str, object | None, bool]:
        if use_cuda:
            try:
                import cupy as cp

                return "cupy", cp, True
            except Exception:
                pass
        try:
            import numpy as np

            return "numpy", np, False
        except Exception:
            return "python", None, False

    def _load_documents(
        self,
        items: list[CudaDocumentItem],
        max_source_bytes: int,
        max_total_bytes: int,
        default_chunk_mode: str,
        default_chunk_size: int,
        default_chunk_overlap: int,
    ) -> tuple[list[_DocBuffer], bool, list[str], list[str]]:
        documents: list[_DocBuffer] = []
        errors: list[str] = []
        warnings: list[str] = []
        truncated = False
        total_bytes = 0

        for idx, item in enumerate(items, start=1):
            try:
                if item.content and item.path:
                    raise ToolError("Provide content or path, not both.")

                content = item.content
                source_path: str | None = None
                size_bytes = 0
                if item.path:
                    path = self._resolve_path(item.path)
                    if not path.exists():
                        raise ToolError(f"Path not found: {path}")
                    if path.is_dir():
                        raise ToolError(f"Path is a directory, not a file: {path}")
                    size_bytes = path.stat().st_size
                    if size_bytes > max_source_bytes:
                        raise ToolError(
                            f"{path} exceeds max_source_bytes ({size_bytes} > {max_source_bytes})."
                        )
                    content = path.read_text("utf-8", errors="ignore")
                    source_path = str(path)
                elif content is not None:
                    size_bytes = len(content.encode("utf-8"))
                    if size_bytes > max_source_bytes:
                        raise ToolError(
                            f"content exceeds max_source_bytes ({size_bytes} > {max_source_bytes})."
                        )
                else:
                    raise ToolError("Item has no content to process.")

                if total_bytes + size_bytes > max_total_bytes:
                    truncated = True
                    warnings.append("Total byte limit reached; stopping item ingestion.")
                    break

                chunk_mode = (item.chunk_mode or default_chunk_mode).strip().lower()
                if chunk_mode not in CHUNK_MODES:
                    raise ToolError(
                        "chunk_mode must be words, paragraphs, sentences, lines, or chars."
                    )
                chunk_size = (
                    default_chunk_size if item.chunk_size is None else item.chunk_size
                )
                chunk_overlap = (
                    default_chunk_overlap
                    if item.chunk_overlap is None
                    else item.chunk_overlap
                )
                if chunk_size <= 0:
                    raise ToolError("chunk_size must be a positive integer.")
                if chunk_overlap < 0:
                    raise ToolError("chunk_overlap must be >= 0.")
                if chunk_overlap >= chunk_size:
                    raise ToolError("chunk_overlap must be smaller than chunk_size.")

                doc_id = item.id or item.name or source_path or f"item-{idx}"
                documents.append(
                    _DocBuffer(
                        doc_id=str(doc_id),
                        name=item.name,
                        source_path=source_path,
                        source_type=item.source or "item",
                        content=content or "",
                        size_bytes=size_bytes,
                        chunk_mode=chunk_mode,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                )
                total_bytes += size_bytes
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")

        return documents, truncated, errors, warnings

    def _chunk_documents(
        self,
        documents: list[_DocBuffer],
        max_chunk_bytes: int,
        max_chunks: int,
        max_chunks_per_doc: int,
        min_token_length: int,
        max_tokens_per_chunk: int,
        preview_chars: int,
    ) -> tuple[list[CudaChunk], list[_ChunkState], list[set[str]], bool]:
        chunks: list[CudaChunk] = []
        chunk_states: list[_ChunkState] = []
        chunk_token_sets: list[set[str]] = []
        truncated = False
        total_chunks = 0

        for doc_index, doc in enumerate(documents, start=1):
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

            for chunk_text, start_index, end_index, unit, chunk_trimmed in chunk_entries:
                if max_chunks > 0 and total_chunks >= max_chunks:
                    truncated = True
                    return chunks, chunk_states, chunk_token_sets, truncated

                token_counts, token_set, state = self._chunk_state(
                    chunk_text, min_token_length
                )
                keywords = self._select_keywords(
                    token_counts, max_tokens_per_chunk
                )

                total_chunks += 1
                chunks.append(
                    CudaChunk(
                        index=total_chunks,
                        document_index=doc_index,
                        unit=unit,
                        start_index=start_index,
                        end_index=end_index,
                        preview=self._preview_text(chunk_text, preview_chars),
                        token_count=state.token_count,
                        unique_tokens=state.unique_tokens,
                        unique_ratio=state.unique_ratio,
                        line_count=state.line_count,
                        sentence_count=state.sentence_count,
                        avg_token_length=state.avg_token_length,
                        keywords=keywords,
                        truncated=chunk_trimmed,
                    )
                )
                chunk_states.append(state)
                chunk_token_sets.append(token_set)

        return chunks, chunk_states, chunk_token_sets, truncated

    def _build_document_summaries(
        self,
        documents: list[_DocBuffer],
        chunks: list[CudaChunk],
        chunk_states: list[_ChunkState],
        max_tokens_per_document: int,
        min_token_length: int,
        preview_chars: int,
    ) -> list[CudaDocumentSummary]:
        doc_token_counts: list[dict[str, int]] = [
            {} for _ in range(len(documents))
        ]
        doc_metrics: list[dict[str, float]] = [
            {"token_count": 0, "line_count": 0, "sentence_count": 0, "unique_ratio": 0.0, "avg_token_length": 0.0, "chunks": 0}
            for _ in range(len(documents))
        ]

        for chunk, state in zip(chunks, chunk_states):
            doc_idx = chunk.document_index - 1
            doc_metrics[doc_idx]["token_count"] += state.token_count
            doc_metrics[doc_idx]["line_count"] += state.line_count
            doc_metrics[doc_idx]["sentence_count"] += state.sentence_count
            doc_metrics[doc_idx]["unique_ratio"] += state.unique_ratio
            doc_metrics[doc_idx]["avg_token_length"] += state.avg_token_length
            doc_metrics[doc_idx]["chunks"] += 1

            token_counts, _, _ = self._chunk_state(
                chunk.preview, min_token_length
            )
            for token, count in token_counts.items():
                doc_token_counts[doc_idx][token] = (
                    doc_token_counts[doc_idx].get(token, 0) + count
                )

        summaries: list[CudaDocumentSummary] = []
        for idx, doc in enumerate(documents, start=1):
            metrics = doc_metrics[idx - 1]
            chunk_count = int(metrics["chunks"] or 0)
            if chunk_count:
                avg_unique_ratio = metrics["unique_ratio"] / chunk_count
                avg_token_length = metrics["avg_token_length"] / chunk_count
            else:
                avg_unique_ratio = 0.0
                avg_token_length = 0.0

            summaries.append(
                CudaDocumentSummary(
                    index=idx,
                    id=str(doc.doc_id),
                    name=doc.name,
                    source_path=doc.source_path,
                    source_type=doc.source_type,
                    chunk_count=chunk_count,
                    token_count=int(metrics["token_count"]),
                    preview=self._preview_text(doc.content, preview_chars),
                    keywords=self._select_keywords(
                        doc_token_counts[idx - 1], max_tokens_per_document
                    ),
                    line_count=int(metrics["line_count"]),
                    sentence_count=int(metrics["sentence_count"]),
                    unique_ratio=round(avg_unique_ratio, 6),
                    avg_token_length=round(avg_token_length, 6),
                    truncated=False,
                )
            )

        return summaries
    def _chunk_state(
        self, content: str, min_token_length: int
    ) -> tuple[dict[str, int], set[str], _ChunkState]:
        token_counts: dict[str, int] = {}
        token_lengths: list[int] = []
        for match in WORD_RE.findall(content.lower()):
            if len(match) < min_token_length:
                continue
            if match.isdigit():
                continue
            if match in STOPWORDS:
                continue
            token_counts[match] = token_counts.get(match, 0) + 1
            token_lengths.append(len(match))

        token_count = sum(token_counts.values())
        unique_tokens = len(token_counts)
        unique_ratio = unique_tokens / token_count if token_count else 0.0
        line_count = len(content.splitlines())
        sentence_count = len([part for part in SENTENCE_RE.split(content) if part.strip()])
        avg_token_length = (
            sum(token_lengths) / len(token_lengths) if token_lengths else 0.0
        )
        return (
            token_counts,
            set(token_counts.keys()),
            _ChunkState(
                token_count=token_count,
                unique_tokens=unique_tokens,
                unique_ratio=unique_ratio,
                line_count=line_count,
                sentence_count=sentence_count,
                avg_token_length=avg_token_length,
            ),
        )

    def _build_feature_matrix(
        self, token_sets: list[set[str]], feature_dim: int
    ) -> list[list[float]]:
        features: list[list[float]] = []
        for tokens in token_sets:
            row = [0.0] * feature_dim
            for token in tokens:
                idx = self._hash_token(token, feature_dim)
                row[idx] += 1.0
            features.append(row)
        return features

    def _hash_token(self, token: str, feature_dim: int) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        value = int.from_bytes(digest[:4], "little")
        return value % feature_dim

    def _build_edges(
        self,
        features: list[list[float]],
        chunk_states: list[_ChunkState],
        token_sets: list[set[str]],
        lexical_weight: float,
        structural_weight: float,
        density_weight: float,
        min_score: float,
        max_neighbors: int,
        max_edges: int,
        max_pairwise_chunks: int,
        pairwise_block_size: int,
        max_shared_tokens: int,
        xp: object | None,
        backend_name: str,
    ) -> tuple[list[CudaEdge], bool]:
        count = len(features)
        if count < 2 or max_neighbors == 0 or max_edges == 0:
            return [], False

        truncated = False
        if count > max_pairwise_chunks:
            truncated = True
            count = max_pairwise_chunks

        neighbors: list[list[tuple[int, float]]] = [[] for _ in range(count)]
        if backend_name in {"cupy", "numpy"} and xp is not None:
            matrix = xp.asarray(features[:count], dtype=xp.float32)
            norms = xp.linalg.norm(matrix, axis=1)
            norms = xp.where(norms == 0, 1.0, norms)
            matrix = matrix / norms[:, None]

            block = max(1, min(pairwise_block_size, count))
            for start in range(0, count, block):
                end = min(count, start + block)
                scores = matrix[start:end] @ matrix.T
                scores = scores if backend_name == "numpy" else scores.get()
                for row_index in range(scores.shape[0]):
                    i = start + row_index
                    row = scores[row_index]
                    for j, value in enumerate(row.tolist()):
                        if j <= i:
                            continue
                        if value <= 0:
                            continue
                        neighbors[i].append((j, float(value)))
                        neighbors[j].append((i, float(value)))
        else:
            for i in range(count):
                for j in range(i + 1, count):
                    score = self._dot(features[i], features[j])
                    if score <= 0:
                        continue
                    neighbors[i].append((j, score))
                    neighbors[j].append((i, score))

        for idx in range(count):
            neighbors[idx].sort(key=lambda item: (-item[1], item[0]))
            if max_neighbors > 0:
                neighbors[idx] = neighbors[idx][:max_neighbors]

        edges: list[CudaEdge] = []
        for i, entries in enumerate(neighbors):
            for j, lexical_score in entries:
                if i >= j:
                    continue
                structural_score = self._structural_similarity(
                    chunk_states[i], chunk_states[j]
                )
                density_score = self._density_similarity(
                    chunk_states[i], chunk_states[j]
                )
                score = (
                    lexical_weight * lexical_score
                    + structural_weight * structural_score
                    + density_weight * density_score
                )
                if score < min_score:
                    continue
                shared = self._shared_tokens(
                    token_sets[i], token_sets[j], max_shared_tokens
                )
                edges.append(
                    CudaEdge(
                        source_index=i + 1,
                        target_index=j + 1,
                        score=round(score, 6),
                        lexical_score=round(lexical_score, 6),
                        structural_score=round(structural_score, 6),
                        density_score=round(density_score, 6),
                        shared_tokens=shared,
                    )
                )

        edges.sort(key=lambda item: (-item.score, item.source_index, item.target_index))
        if max_edges > 0 and len(edges) > max_edges:
            truncated = True
            edges = edges[:max_edges]

        return edges, truncated

    def _build_warp_groups(
        self,
        chunks: list[CudaChunk],
        chunk_states: list[_ChunkState],
        warp_group_size: int,
    ) -> list[CudaWarpGroup]:
        groups: list[CudaWarpGroup] = []
        current: list[int] = []
        current_doc = None
        totals = {"token": 0.0, "ratio": 0.0, "sent": 0.0}

        def flush() -> None:
            nonlocal current, totals, current_doc
            if not current:
                return
            count = len(current)
            groups.append(
                CudaWarpGroup(
                    group_id=len(groups) + 1,
                    document_index=current_doc or 1,
                    chunk_indices=current[:],
                    avg_token_count=totals["token"] / count if count else 0.0,
                    avg_unique_ratio=totals["ratio"] / count if count else 0.0,
                    avg_sentence_count=totals["sent"] / count if count else 0.0,
                )
            )
            current = []
            totals = {"token": 0.0, "ratio": 0.0, "sent": 0.0}

        for idx, chunk in enumerate(chunks):
            if current_doc is None:
                current_doc = chunk.document_index
            if chunk.document_index != current_doc or len(current) >= warp_group_size:
                flush()
                current_doc = chunk.document_index
            current.append(chunk.index)
            state = chunk_states[idx]
            totals["token"] += state.token_count
            totals["ratio"] += state.unique_ratio
            totals["sent"] += state.sentence_count

        flush()
        return groups

    def _structural_similarity(self, left: _ChunkState, right: _ChunkState) -> float:
        line_score = self._ratio_similarity(left.line_count, right.line_count)
        sentence_score = self._ratio_similarity(
            left.sentence_count, right.sentence_count
        )
        token_score = self._ratio_similarity(left.token_count, right.token_count)
        return (line_score + sentence_score + token_score) / 3.0

    def _density_similarity(self, left: _ChunkState, right: _ChunkState) -> float:
        ratio_score = self._ratio_similarity(left.unique_ratio, right.unique_ratio)
        length_score = self._ratio_similarity(
            left.avg_token_length, right.avg_token_length
        )
        return (ratio_score + length_score) / 2.0

    def _ratio_similarity(self, left: float, right: float) -> float:
        if left == 0 and right == 0:
            return 1.0
        max_value = max(left, right)
        if max_value == 0:
            return 0.0
        return max(0.0, 1.0 - (abs(left - right) / max_value))

    def _dot(self, left: list[float], right: list[float]) -> float:
        total = 0.0
        for a, b in zip(left, right):
            total += a * b
        return total

    def _shared_tokens(
        self, left: set[str], right: set[str], max_shared_tokens: int
    ) -> list[str]:
        shared = sorted(left & right)
        if max_shared_tokens > 0 and len(shared) > max_shared_tokens:
            shared = shared[:max_shared_tokens]
        return shared

    def _preview_text(self, text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def _chunk_text(
        self,
        content: str,
        mode: str,
        size: int,
        overlap: int,
        max_chunks: int,
        max_chunk_bytes: int,
    ) -> tuple[list[tuple[str, int | None, int | None, str, bool]], bool]:
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

        units: list[str]
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
        else:
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

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextCudaMultistateArgs):
            return ToolCallDisplay(summary="context_cuda_multistate")

        summary = f"context_cuda_multistate: {len(event.args.items)} item(s)"
        return ToolCallDisplay(
            summary=summary,
            details={
                "item_count": len(event.args.items),
                "use_cuda": event.args.use_cuda,
                "max_edges": event.args.max_edges,
                "max_neighbors": event.args.max_neighbors,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextCudaMultistateResult):
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
                "warp_group_count": event.result.warp_group_count,
                "backend": event.result.backend,
                "use_cuda": event.result.use_cuda,
                "cuda_available": event.result.cuda_available,
                "truncated": event.result.truncated,
                "errors": event.result.errors,
                "warnings": event.result.warnings,
                "documents": event.result.documents,
                "chunks": event.result.chunks,
                "edges": event.result.edges,
                "warp_groups": event.result.warp_groups,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "CUDA multistate processing"
