from __future__ import annotations

from dataclasses import dataclass
from array import array
from heapq import heappush, heappushpop
from pathlib import Path
import json
import re
import sqlite3
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


@dataclass
class _DocAggregate:
    doc_id: str
    source_path: str
    store_label: str
    store_path: str
    chunk_count: int
    included_chunks: int
    embedding_sum: list[float] | None
    embedding_count: int
    token_counts: dict[str, int]
    subchunks: list["SubChunk"]


class ContextVectorCrossrefMultiConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_documents: int = Field(
        default=200, description="Maximum documents to process across stores."
    )
    max_chunks_per_doc: int = Field(
        default=10, description="Maximum subchunks stored per document."
    )
    max_total_chunks: int = Field(
        default=1000, description="Maximum subchunks stored across all documents."
    )
    max_subchunk_chars: int = Field(
        default=1200, description="Maximum characters per subchunk."
    )
    max_embedding_chunks_per_doc: int = Field(
        default=50, description="Maximum chunks per document used for embeddings."
    )
    min_similarity: float = Field(
        default=0.15, description="Minimum similarity for cross references."
    )
    max_crossrefs_per_doc: int = Field(
        default=5, description="Maximum cross references per document."
    )
    max_crossrefs_total: int = Field(
        default=500, description="Maximum cross references overall."
    )
    min_word_length: int = Field(
        default=3, description="Minimum word length for keywords."
    )
    max_keywords: int = Field(
        default=20, description="Maximum keywords per document."
    )


class ContextVectorCrossrefMultiState(BaseToolState):
    pass


class VectorStore(BaseModel):
    path: str = Field(description="Path to a sqlite DB or manifest file.")
    store_type: str | None = Field(
        default="db", description="db, manifest, or sharded."
    )
    label: str | None = Field(default=None, description="Optional store label.")
    max_documents: int | None = Field(
        default=None, description="Override max documents for this store."
    )
    max_chunks_per_doc: int | None = Field(
        default=None, description="Override max subchunks per document."
    )
    max_embedding_chunks_per_doc: int | None = Field(
        default=None, description="Override embedding chunks per document."
    )
    max_subchunk_chars: int | None = Field(
        default=None, description="Override max chars per subchunk."
    )


class ContextVectorCrossrefMultiArgs(BaseModel):
    stores: list[VectorStore] = Field(description="Vector stores to scan.")
    max_documents: int | None = Field(
        default=None, description="Override max_documents."
    )
    max_chunks_per_doc: int | None = Field(
        default=None, description="Override max_chunks_per_doc."
    )
    max_total_chunks: int | None = Field(
        default=None, description="Override max_total_chunks."
    )
    max_subchunk_chars: int | None = Field(
        default=None, description="Override max_subchunk_chars."
    )
    max_embedding_chunks_per_doc: int | None = Field(
        default=None, description="Override max_embedding_chunks_per_doc."
    )
    min_similarity: float | None = Field(
        default=None, description="Override min_similarity."
    )
    max_crossrefs_per_doc: int | None = Field(
        default=None, description="Override max_crossrefs_per_doc."
    )
    max_crossrefs_total: int | None = Field(
        default=None, description="Override max_crossrefs_total."
    )
    min_word_length: int | None = Field(
        default=None, description="Override min_word_length."
    )
    max_keywords: int | None = Field(
        default=None, description="Override max_keywords."
    )


class SubChunk(BaseModel):
    chunk_index: int
    unit: str
    start_index: int | None
    end_index: int | None
    content: str


class DocumentSummary(BaseModel):
    doc_id: str
    source_path: str
    store_label: str
    store_path: str
    chunk_count: int
    included_chunks: int
    embedding_count: int
    keywords: list[str]
    subchunks: list[SubChunk]


class DocumentCrossref(BaseModel):
    doc_id: str
    source_path: str
    store_label: str
    related_doc_id: str
    related_source_path: str
    related_store_label: str
    similarity: float


class StoreSummary(BaseModel):
    label: str
    store_path: str
    store_type: str
    document_count: int
    chunk_count: int
    truncated: bool
    errors: list[str]


class ContextVectorCrossrefMultiResult(BaseModel):
    documents: list[DocumentSummary]
    crossrefs: list[DocumentCrossref]
    document_count: int
    crossref_count: int
    truncated: bool
    store_summaries: list[StoreSummary]
    errors: list[str]


class ContextVectorCrossrefMulti(
    BaseTool[
        ContextVectorCrossrefMultiArgs,
        ContextVectorCrossrefMultiResult,
        ContextVectorCrossrefMultiConfig,
        ContextVectorCrossrefMultiState,
    ],
    ToolUIData[ContextVectorCrossrefMultiArgs, ContextVectorCrossrefMultiResult],
):
    description: ClassVar[str] = (
        "Cross reference documents across multiple vector databases."
    )

    async def run(
        self, args: ContextVectorCrossrefMultiArgs
    ) -> ContextVectorCrossrefMultiResult:
        if not args.stores:
            raise ToolError("stores is required.")

        max_documents = (
            args.max_documents
            if args.max_documents is not None
            else self.config.max_documents
        )
        max_chunks_per_doc = (
            args.max_chunks_per_doc
            if args.max_chunks_per_doc is not None
            else self.config.max_chunks_per_doc
        )
        max_total_chunks = (
            args.max_total_chunks
            if args.max_total_chunks is not None
            else self.config.max_total_chunks
        )
        max_subchunk_chars = (
            args.max_subchunk_chars
            if args.max_subchunk_chars is not None
            else self.config.max_subchunk_chars
        )
        max_embedding_chunks_per_doc = (
            args.max_embedding_chunks_per_doc
            if args.max_embedding_chunks_per_doc is not None
            else self.config.max_embedding_chunks_per_doc
        )
        min_similarity = (
            args.min_similarity
            if args.min_similarity is not None
            else self.config.min_similarity
        )
        max_crossrefs_per_doc = (
            args.max_crossrefs_per_doc
            if args.max_crossrefs_per_doc is not None
            else self.config.max_crossrefs_per_doc
        )
        max_crossrefs_total = (
            args.max_crossrefs_total
            if args.max_crossrefs_total is not None
            else self.config.max_crossrefs_total
        )
        min_word_length = (
            args.min_word_length
            if args.min_word_length is not None
            else self.config.min_word_length
        )
        max_keywords = (
            args.max_keywords
            if args.max_keywords is not None
            else self.config.max_keywords
        )

        if max_documents <= 0:
            raise ToolError("max_documents must be a positive integer.")
        if max_chunks_per_doc <= 0:
            raise ToolError("max_chunks_per_doc must be a positive integer.")
        if max_total_chunks <= 0:
            raise ToolError("max_total_chunks must be a positive integer.")
        if max_subchunk_chars <= 0:
            raise ToolError("max_subchunk_chars must be a positive integer.")
        if max_embedding_chunks_per_doc <= 0:
            raise ToolError("max_embedding_chunks_per_doc must be a positive integer.")
        if min_similarity < 0:
            raise ToolError("min_similarity must be >= 0.")
        if max_crossrefs_per_doc < 0:
            raise ToolError("max_crossrefs_per_doc must be >= 0.")
        if max_crossrefs_total < 0:
            raise ToolError("max_crossrefs_total must be >= 0.")
        if min_word_length <= 0:
            raise ToolError("min_word_length must be a positive integer.")
        if max_keywords < 0:
            raise ToolError("max_keywords must be >= 0.")

        documents: dict[str, _DocAggregate] = {}
        doc_order: list[str] = []
        store_summaries: list[StoreSummary] = []
        errors: list[str] = []
        truncated = False
        total_chunks = 0

        for idx, store in enumerate(args.stores, start=1):
            store_errors: list[str] = []
            store_truncated = False
            store_label = store.label or Path(store.path).name
            try:
                store_type = self._normalize_store_type(store.store_type)
            except ToolError as exc:
                errors.append(f"stores[{idx}]: {exc}")
                store_summaries.append(
                    StoreSummary(
                        label=store_label,
                        store_path=store.path,
                        store_type=str(store.store_type or "db"),
                        document_count=0,
                        chunk_count=0,
                        truncated=False,
                        errors=[str(exc)],
                    )
                )
                continue

            try:
                path = self._resolve_path(store.path)
            except ToolError as exc:
                errors.append(f"stores[{idx}]: {exc}")
                store_summaries.append(
                    StoreSummary(
                        label=store_label,
                        store_path=store.path,
                        store_type=store_type,
                        document_count=0,
                        chunk_count=0,
                        truncated=False,
                        errors=[str(exc)],
                    )
                )
                continue

            store_max_docs = (
                store.max_documents if store.max_documents is not None else max_documents
            )
            store_max_chunks = (
                store.max_chunks_per_doc
                if store.max_chunks_per_doc is not None
                else max_chunks_per_doc
            )
            store_max_embed = (
                store.max_embedding_chunks_per_doc
                if store.max_embedding_chunks_per_doc is not None
                else max_embedding_chunks_per_doc
            )
            store_max_chars = (
                store.max_subchunk_chars
                if store.max_subchunk_chars is not None
                else max_subchunk_chars
            )

            if store_max_docs <= 0:
                store_errors.append("max_documents must be a positive integer.")
                store_summaries.append(
                    StoreSummary(
                        label=store_label,
                        store_path=str(path),
                        store_type=store_type,
                        document_count=0,
                        chunk_count=0,
                        truncated=False,
                        errors=store_errors,
                    )
                )
                errors.extend(store_errors)
                continue
            if store_max_chunks <= 0:
                store_errors.append("max_chunks_per_doc must be a positive integer.")
            if store_max_embed <= 0:
                store_errors.append("max_embedding_chunks_per_doc must be positive.")
            if store_max_chars <= 0:
                store_errors.append("max_subchunk_chars must be positive.")
            if store_errors:
                store_summaries.append(
                    StoreSummary(
                        label=store_label,
                        store_path=str(path),
                        store_type=store_type,
                        document_count=0,
                        chunk_count=0,
                        truncated=False,
                        errors=store_errors,
                    )
                )
                errors.extend(store_errors)
                continue

            store_doc_count = 0
            store_chunk_count = 0

            if store_type == "db":
                (
                    store_doc_count,
                    store_chunk_count,
                    total_chunks,
                    store_truncated,
                ) = self._scan_db(
                    path,
                    store_label,
                    str(path),
                    documents,
                    doc_order,
                    max_documents,
                    store_max_docs,
                    store_max_chunks,
                    store_max_embed,
                    store_max_chars,
                    max_total_chunks,
                    total_chunks,
                    min_word_length,
                    max_keywords,
                    store_errors,
                )
            else:
                manifest, manifest_errors = self._load_manifest(path)
                store_errors.extend(manifest_errors)
                if manifest:
                    shards = manifest.get("shards", [])
                    if not shards:
                        store_errors.append("Manifest contains no shards.")
                    for shard in shards:
                        shard_path = Path(shard.get("path", ""))
                        if not shard_path.is_absolute():
                            shard_path = path.parent / shard_path
                        if not shard_path.exists():
                            store_errors.append(f"Shard not found: {shard_path}")
                            continue
                        (
                            added_docs,
                            added_chunks,
                            total_chunks,
                            shard_truncated,
                        ) = self._scan_db(
                            shard_path,
                            store_label,
                            str(path),
                            documents,
                            doc_order,
                            max_documents,
                            store_max_docs,
                            store_max_chunks,
                            store_max_embed,
                            store_max_chars,
                            max_total_chunks,
                            total_chunks,
                            min_word_length,
                            max_keywords,
                            store_errors,
                        )
                        store_doc_count += added_docs
                        store_chunk_count += added_chunks
                        if shard_truncated:
                            store_truncated = True

            if store_truncated:
                truncated = True

            store_summaries.append(
                StoreSummary(
                    label=store_label,
                    store_path=str(path),
                    store_type=store_type,
                    document_count=store_doc_count,
                    chunk_count=store_chunk_count,
                    truncated=store_truncated,
                    errors=store_errors,
                )
            )
            errors.extend(store_errors)

        if not documents:
            raise ToolError("No documents were loaded from the vector stores.")

        doc_embeddings: dict[str, list[float]] = {}
        for doc_id in doc_order:
            doc = documents[doc_id]
            if doc.embedding_sum and doc.embedding_count > 0:
                averaged = [value / doc.embedding_count for value in doc.embedding_sum]
                doc_embeddings[doc_id] = self._normalize_embedding(averaged)

        document_summaries: list[DocumentSummary] = []
        for doc_id in doc_order:
            doc = documents[doc_id]
            keywords = self._select_keywords(doc.token_counts, max_keywords)
            document_summaries.append(
                DocumentSummary(
                    doc_id=doc.doc_id,
                    source_path=doc.source_path,
                    store_label=doc.store_label,
                    store_path=doc.store_path,
                    chunk_count=doc.chunk_count,
                    included_chunks=doc.included_chunks,
                    embedding_count=doc.embedding_count,
                    keywords=keywords,
                    subchunks=doc.subchunks,
                )
            )

        crossrefs: list[DocumentCrossref] = []
        crossref_count = 0
        if max_crossrefs_total > 0 and max_crossrefs_per_doc > 0:
            for doc_id in doc_order:
                if crossref_count >= max_crossrefs_total:
                    truncated = True
                    break
                embedding = doc_embeddings.get(doc_id)
                if not embedding:
                    continue
                heap: list[tuple[float, str]] = []
                for other_id in doc_order:
                    if other_id == doc_id:
                        continue
                    other_embedding = doc_embeddings.get(other_id)
                    if not other_embedding:
                        continue
                    score = self._dot(embedding, other_embedding)
                    if score < min_similarity:
                        continue
                    if len(heap) < max_crossrefs_per_doc:
                        heappush(heap, (score, other_id))
                    else:
                        heappushpop(heap, (score, other_id))

                ranked = sorted(heap, key=lambda item: item[0], reverse=True)
                for score, other_id in ranked:
                    if crossref_count >= max_crossrefs_total:
                        truncated = True
                        break
                    source_doc = documents[doc_id]
                    target_doc = documents[other_id]
                    crossrefs.append(
                        DocumentCrossref(
                            doc_id=source_doc.doc_id,
                            source_path=source_doc.source_path,
                            store_label=source_doc.store_label,
                            related_doc_id=target_doc.doc_id,
                            related_source_path=target_doc.source_path,
                            related_store_label=target_doc.store_label,
                            similarity=round(score, 6),
                        )
                    )
                    crossref_count += 1

        return ContextVectorCrossrefMultiResult(
            documents=document_summaries,
            crossrefs=crossrefs,
            document_count=len(document_summaries),
            crossref_count=len(crossrefs),
            truncated=truncated,
            store_summaries=store_summaries,
            errors=errors,
        )

    def _scan_db(
        self,
        db_path: Path,
        store_label: str,
        store_path: str,
        documents: dict[str, _DocAggregate],
        doc_order: list[str],
        max_documents: int,
        store_max_docs: int,
        max_chunks_per_doc: int,
        max_embedding_chunks_per_doc: int,
        max_subchunk_chars: int,
        max_total_chunks: int,
        total_chunks: int,
        min_word_length: int,
        max_keywords: int,
        errors: list[str],
    ) -> tuple[int, int, int, bool]:
        if not db_path.exists():
            errors.append(f"Store not found: {db_path}")
            return 0, 0, total_chunks, False
        if db_path.is_dir():
            errors.append(f"Store path is a directory: {db_path}")
            return 0, 0, total_chunks, False

        doc_count = 0
        chunk_count = 0
        truncated = False

        try:
            conn = sqlite3.connect(str(db_path))
        except Exception as exc:
            errors.append(f"Failed to open store: {exc}")
            return 0, 0, total_chunks, False

        try:
            cursor = conn.execute(
                """
                SELECT source_path, chunk_index, unit, start_index, end_index, content, embedding, embedding_dim
                FROM pdf_chunks
                ORDER BY source_path, chunk_index
                """
            )
            for row in cursor:
                source_path = row[0] or ""
                doc_id = self._doc_id(store_label, store_path, source_path)
                doc = documents.get(doc_id)
                if doc is None:
                    if len(documents) >= max_documents or doc_count >= store_max_docs:
                        truncated = True
                        break
                    doc = _DocAggregate(
                        doc_id=doc_id,
                        source_path=source_path,
                        store_label=store_label,
                        store_path=store_path,
                        chunk_count=0,
                        included_chunks=0,
                        embedding_sum=None,
                        embedding_count=0,
                        token_counts={},
                        subchunks=[],
                    )
                    documents[doc_id] = doc
                    doc_order.append(doc_id)
                    doc_count += 1

                doc.chunk_count += 1
                chunk_count += 1

                if (
                    doc.included_chunks < max_chunks_per_doc
                    and total_chunks < max_total_chunks
                ):
                    content = row[5] or ""
                    content = self._truncate_text(content, max_subchunk_chars)
                    doc.subchunks.append(
                        SubChunk(
                            chunk_index=int(row[1]),
                            unit=str(row[2]),
                            start_index=row[3],
                            end_index=row[4],
                            content=content,
                        )
                    )
                    doc.included_chunks += 1
                    total_chunks += 1
                    self._update_token_counts(
                        doc.token_counts, content, min_word_length
                    )
                elif total_chunks >= max_total_chunks:
                    truncated = True

                if doc.embedding_count < max_embedding_chunks_per_doc:
                    embedding_blob = row[6]
                    embedding_dim = row[7] if len(row) > 7 else None
                    if embedding_blob:
                        embedding = self._unpack_embedding(embedding_blob)
                        if embedding:
                            if embedding_dim and embedding_dim != len(embedding):
                                continue
                            self._add_embedding(doc, embedding)

            if len(documents) >= max_documents and total_chunks >= max_total_chunks:
                truncated = True
        except sqlite3.Error as exc:
            errors.append(f"Failed reading store {db_path}: {exc}")
        finally:
            conn.close()

        return doc_count, chunk_count, total_chunks, truncated

    def _doc_id(self, store_label: str, store_path: str, source_path: str) -> str:
        return f"{store_label}|{store_path}|{source_path}"

    def _truncate_text(self, text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def _update_token_counts(
        self, token_counts: dict[str, int], text: str, min_word_length: int
    ) -> None:
        for token in WORD_RE.findall(text):
            lowered = token.lower()
            if len(lowered) < min_word_length:
                continue
            if lowered in STOPWORDS:
                continue
            token_counts[lowered] = token_counts.get(lowered, 0) + 1

    def _select_keywords(self, token_counts: dict[str, int], max_keywords: int) -> list[str]:
        if max_keywords <= 0:
            return []
        ordered = sorted(token_counts.items(), key=lambda item: (-item[1], item[0]))
        return [token for token, _ in ordered[:max_keywords]]

    def _add_embedding(self, doc: _DocAggregate, embedding: list[float]) -> None:
        if doc.embedding_sum is None:
            doc.embedding_sum = [0.0] * len(embedding)
        if len(embedding) != len(doc.embedding_sum):
            return
        for index, value in enumerate(embedding):
            doc.embedding_sum[index] += value
        doc.embedding_count += 1

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

    def _dot(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            return 0.0
        return sum(l * r for l, r in zip(left, right))

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

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextVectorCrossrefMultiArgs):
            return ToolCallDisplay(summary="context_vector_crossref_multi")

        summary = f"context_vector_crossref_multi: {len(event.args.stores)} store(s)"
        return ToolCallDisplay(
            summary=summary,
            details={
                "store_count": len(event.args.stores),
                "max_documents": event.args.max_documents,
                "max_chunks_per_doc": event.args.max_chunks_per_doc,
                "max_total_chunks": event.args.max_total_chunks,
                "min_similarity": event.args.min_similarity,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextVectorCrossrefMultiResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = (
            f"Processed {event.result.document_count} document(s) across "
            f"{len(event.result.store_summaries)} store(s)"
        )
        warnings = event.result.errors[:]
        if event.result.truncated:
            warnings.append("Output truncated by size or limits")

        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=warnings,
            details={
                "document_count": event.result.document_count,
                "crossref_count": event.result.crossref_count,
                "truncated": event.result.truncated,
                "errors": event.result.errors,
                "store_summaries": event.result.store_summaries,
                "documents": event.result.documents,
                "crossrefs": event.result.crossrefs,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Cross referencing vector databases"
