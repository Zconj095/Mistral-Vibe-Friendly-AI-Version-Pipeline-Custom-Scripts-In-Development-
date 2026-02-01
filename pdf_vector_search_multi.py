from __future__ import annotations

import json
from array import array
from dataclasses import dataclass
from heapq import heappush, heappushpop
from pathlib import Path
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


DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"


@dataclass
class _SearchRow:
    score: float
    source_path: str
    chunk_index: int
    unit: str
    start_index: int | None
    end_index: int | None
    content: str
    shard_path: str | None


class PdfVectorSearchMultiConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    ollama_url: str = Field(
        default="http://127.0.0.1:11434",
        description="Base URL for the Ollama/GPT-OSS server.",
    )
    embedding_model: str = Field(
        default=DEFAULT_EMBEDDING_MODEL,
        description="Default embedding model to use with Ollama/GPT-OSS.",
    )
    max_results_per_store: int = Field(
        default=500, description="Maximum results to return per store."
    )
    max_total_results: int = Field(
        default=2000, description="Maximum total results to return."
    )
    max_result_chars: int = Field(
        default=2000, description="Maximum characters to return per result."
    )
    base_budget_bytes: int = Field(
        default=1_000_000, description="Base budget per embedding (bytes)."
    )
    budget_step_bytes: int = Field(
        default=1_000_000, description="Budget increment per step (bytes)."
    )
    budget_steps: float = Field(
        default=0.0, description="Fractional steps to increase the per-embedding budget."
    )
    cap_by_storage: bool = Field(
        default=True, description="Cap budgets by on-disk store sizes."
    )
    bind_to_first_embedding: bool = Field(
        default=True, description="Bind per-embedding budget to the first store."
    )


class PdfVectorSearchMultiState(BaseToolState):
    pass


class EmbeddingStore(BaseModel):
    path: str = Field(description="Path to a sqlite DB or manifest file.")
    store_type: str | None = Field(
        default="db", description="db, manifest, or sharded."
    )
    embedding_model: str | None = Field(
        default=None, description="Override embedding model for this store."
    )
    label: str | None = Field(default=None, description="Optional store label.")
    min_score: float | None = Field(default=None, description="Minimum similarity score.")
    max_results: int | None = Field(
        default=None, description="Override max results for this store."
    )
    max_result_chars: int | None = Field(
        default=None, description="Override max chars per result for this store."
    )
    budget_bytes: int | None = Field(
        default=None, description="Override budget for this store (bytes)."
    )


class PdfVectorSearchMultiArgs(BaseModel):
    query: str = Field(description="Search query.")
    stores: list[EmbeddingStore] = Field(description="Embedding stores to search.")
    base_budget_bytes: int | None = Field(
        default=None, description="Override base budget per embedding (bytes)."
    )
    budget_step_bytes: int | None = Field(
        default=None, description="Override budget step size (bytes)."
    )
    budget_steps: float | None = Field(
        default=None, description="Override budget steps (fractional allowed)."
    )
    max_results_per_store: int | None = Field(
        default=None, description="Override max results per store."
    )
    max_total_results: int | None = Field(
        default=None, description="Override max total results."
    )
    max_result_chars: int | None = Field(
        default=None, description="Override max chars per result."
    )
    min_score: float | None = Field(default=None, description="Minimum similarity score.")
    cap_by_storage: bool | None = Field(
        default=None, description="Cap budgets by on-disk store sizes."
    )
    bind_to_first_embedding: bool | None = Field(
        default=None, description="Bind per-embedding budget to the first store."
    )


class PdfVectorSearchMultiResultItem(BaseModel):
    score: float
    store_label: str
    store_path: str
    store_type: str
    embedding_model: str
    source_path: str
    chunk_index: int
    unit: str
    start_index: int | None
    end_index: int | None
    content: str
    shard_path: str | None


class EmbeddingStoreSummary(BaseModel):
    label: str
    store_path: str
    store_type: str
    embedding_model: str
    budget_bytes: int
    used_bytes: int
    storage_bytes: int | None
    result_count: int
    truncated: bool
    errors: list[str]


class PdfVectorSearchMultiResult(BaseModel):
    results: list[PdfVectorSearchMultiResultItem]
    count: int
    store_count: int
    total_budget_bytes: int
    total_used_bytes: int
    truncated: bool
    store_summaries: list[EmbeddingStoreSummary]
    errors: list[str]


class PdfVectorSearchMulti(
    BaseTool[
        PdfVectorSearchMultiArgs,
        PdfVectorSearchMultiResult,
        PdfVectorSearchMultiConfig,
        PdfVectorSearchMultiState,
    ],
    ToolUIData[PdfVectorSearchMultiArgs, PdfVectorSearchMultiResult],
):
    description: ClassVar[str] = (
        "Search multiple embedding stores and allocate per-embedding context budgets."
    )

    async def run(self, args: PdfVectorSearchMultiArgs) -> PdfVectorSearchMultiResult:
        if not args.query.strip():
            raise ToolError("query cannot be empty.")
        if not args.stores:
            raise ToolError("At least one store is required.")

        base_budget = (
            args.base_budget_bytes
            if args.base_budget_bytes is not None
            else self.config.base_budget_bytes
        )
        budget_step = (
            args.budget_step_bytes
            if args.budget_step_bytes is not None
            else self.config.budget_step_bytes
        )
        budget_steps = (
            args.budget_steps
            if args.budget_steps is not None
            else self.config.budget_steps
        )
        if base_budget <= 0:
            raise ToolError("base_budget_bytes must be a positive integer.")
        if budget_step <= 0:
            raise ToolError("budget_step_bytes must be a positive integer.")

        per_embedding_budget = base_budget + int(budget_step * budget_steps)
        if per_embedding_budget <= 0:
            raise ToolError("Budget per embedding must be positive.")

        max_results_per_store = (
            args.max_results_per_store
            if args.max_results_per_store is not None
            else self.config.max_results_per_store
        )
        max_total_results = (
            args.max_total_results
            if args.max_total_results is not None
            else self.config.max_total_results
        )
        max_result_chars = (
            args.max_result_chars
            if args.max_result_chars is not None
            else self.config.max_result_chars
        )
        if max_results_per_store <= 0:
            raise ToolError("max_results_per_store must be a positive integer.")
        if max_total_results <= 0:
            raise ToolError("max_total_results must be a positive integer.")
        if max_result_chars <= 0:
            raise ToolError("max_result_chars must be a positive integer.")

        cap_by_storage = (
            args.cap_by_storage
            if args.cap_by_storage is not None
            else self.config.cap_by_storage
        )
        bind_to_first = (
            args.bind_to_first_embedding
            if args.bind_to_first_embedding is not None
            else self.config.bind_to_first_embedding
        )

        store_paths = [self._resolve_path(store.path) for store in args.stores]
        store_sizes: list[int | None] = []
        store_size_errors: list[list[str]] = []
        for store, path in zip(args.stores, store_paths):
            store_type = self._normalize_store_type(store.store_type)
            size_bytes, size_errors = self._get_store_size(path, store_type)
            store_sizes.append(size_bytes)
            store_size_errors.append(size_errors)

        if bind_to_first and store_sizes:
            first_size = store_sizes[0]
            if cap_by_storage and first_size:
                per_embedding_budget = min(per_embedding_budget, first_size)

        total_budget_bytes = 0
        total_used_bytes = 0
        results: list[PdfVectorSearchMultiResultItem] = []
        summaries: list[EmbeddingStoreSummary] = []
        errors: list[str] = []
        truncated = False

        query_cache: dict[str, list[float]] = {}

        for index, store in enumerate(args.stores):
            if len(results) >= max_total_results:
                truncated = True
                break

            label = store.label or Path(store.path).name
            store_type = self._normalize_store_type(store.store_type)
            path = store_paths[index]
            store_size = store_sizes[index]
            store_errors = list(store_size_errors[index])

            budget = store.budget_bytes if store.budget_bytes is not None else per_embedding_budget
            if budget <= 0:
                store_errors.append("budget_bytes must be a positive integer.")
                summaries.append(
                    EmbeddingStoreSummary(
                        label=label,
                        store_path=str(path),
                        store_type=store_type,
                        embedding_model=store.embedding_model or self.config.embedding_model,
                        budget_bytes=0,
                        used_bytes=0,
                        storage_bytes=store_size,
                        result_count=0,
                        truncated=False,
                        errors=store_errors,
                    )
                )
                errors.extend(store_errors)
                continue

            if cap_by_storage and store_size:
                budget = min(budget, store_size)

            total_budget_bytes += budget

            embedding_model = store.embedding_model or self.config.embedding_model
            if embedding_model not in query_cache:
                query_cache[embedding_model] = self._embed_text(
                    embedding_model, args.query
                )

            query_vec = query_cache[embedding_model]

            store_max_results = store.max_results or max_results_per_store
            store_max_chars = store.max_result_chars or max_result_chars
            store_min_score = (
                store.min_score if store.min_score is not None else args.min_score
            )

            rows: list[_SearchRow] = []
            if store_type == "db":
                try:
                    rows = self._search_db(
                        path,
                        query_vec,
                        store_max_results,
                        store_max_chars,
                        store_min_score,
                    )
                except ToolError as exc:
                    store_errors.append(str(exc))
            else:
                rows, manifest_errors = self._search_manifest(
                    path,
                    embedding_model,
                    query_vec,
                    store_max_results,
                    store_max_chars,
                    store_min_score,
                )
                store_errors.extend(manifest_errors)

            rows, used_bytes, store_truncated = self._apply_budget(rows, budget)

            remaining_total = max_total_results - len(results)
            if remaining_total <= 0:
                truncated = True
                store_truncated = True
                used_bytes = 0
                rows = []
            elif len(rows) > remaining_total:
                rows = rows[:remaining_total]
                used_bytes = sum(len(r.content.encode("utf-8")) for r in rows)
                truncated = True
                store_truncated = True

            for row in rows:
                results.append(
                    PdfVectorSearchMultiResultItem(
                        score=row.score,
                        store_label=label,
                        store_path=str(path),
                        store_type=store_type,
                        embedding_model=embedding_model,
                        source_path=row.source_path,
                        chunk_index=row.chunk_index,
                        unit=row.unit,
                        start_index=row.start_index,
                        end_index=row.end_index,
                        content=row.content,
                        shard_path=row.shard_path,
                    )
                )

            total_used_bytes += used_bytes
            summaries.append(
                EmbeddingStoreSummary(
                    label=label,
                    store_path=str(path),
                    store_type=store_type,
                    embedding_model=embedding_model,
                    budget_bytes=budget,
                    used_bytes=used_bytes,
                    storage_bytes=store_size,
                    result_count=len(rows),
                    truncated=store_truncated,
                    errors=store_errors,
                )
            )
            errors.extend(store_errors)

        return PdfVectorSearchMultiResult(
            results=results,
            count=len(results),
            store_count=len(args.stores),
            total_budget_bytes=total_budget_bytes,
            total_used_bytes=total_used_bytes,
            truncated=truncated,
            store_summaries=summaries,
            errors=errors,
        )

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

    def _get_store_size(self, path: Path, store_type: str) -> tuple[int | None, list[str]]:
        errors: list[str] = []
        if store_type == "db":
            if not path.exists():
                errors.append(f"Store not found: {path}")
                return None, errors
            if path.is_dir():
                errors.append(f"Store path is a directory: {path}")
                return None, errors
            return path.stat().st_size, errors

        manifest, manifest_errors = self._load_manifest(path)
        errors.extend(manifest_errors)
        total = 0
        shard_count = 0
        for shard in manifest.get("shards", []):
            shard_path = Path(shard.get("path", ""))
            if not shard_path.is_absolute():
                shard_path = path.parent / shard_path
            if not shard_path.exists():
                errors.append(f"Shard not found: {shard_path}")
                continue
            shard_count += 1
            total += shard_path.stat().st_size
        if shard_count == 0:
            return None, errors
        return total, errors

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
    ) -> tuple[list[_SearchRow], list[str]]:
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

        heap: list[tuple[float, _SearchRow]] = []
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
                    row.shard_path = str(shard_path)
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
    ) -> list[_SearchRow]:
        if not db_path.exists():
            raise ToolError(f"Store not found: {db_path}")
        if db_path.is_dir():
            raise ToolError(f"Store path is a directory: {db_path}")

        results: list[tuple[float, _SearchRow]] = []
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
                item = _SearchRow(
                    score=score,
                    source_path=row[0],
                    chunk_index=row[1],
                    unit=row[2],
                    start_index=row[3],
                    end_index=row[4],
                    content=content,
                    shard_path=None,
                )
                if len(results) < max_results:
                    heappush(results, (score, item))
                else:
                    heappushpop(results, (score, item))

        return [item for _, item in sorted(results, key=lambda r: r[0], reverse=True)]

    def _apply_budget(
        self, rows: list[_SearchRow], budget_bytes: int
    ) -> tuple[list[_SearchRow], int, bool]:
        if budget_bytes <= 0:
            return [], 0, True

        selected: list[_SearchRow] = []
        used = 0
        truncated = False
        remaining = budget_bytes

        for row in rows:
            if remaining <= 0:
                truncated = True
                break
            content = row.content
            content_bytes = len(content.encode("utf-8"))
            if content_bytes > remaining:
                content = self._truncate_to_bytes(content, remaining)
                content_bytes = len(content.encode("utf-8"))
                truncated = True
            remaining -= content_bytes
            used += content_bytes
            selected.append(
                _SearchRow(
                    score=row.score,
                    source_path=row.source_path,
                    chunk_index=row.chunk_index,
                    unit=row.unit,
                    start_index=row.start_index,
                    end_index=row.end_index,
                    content=content,
                    shard_path=row.shard_path,
                )
            )
            if truncated:
                break

        return selected, used, truncated

    def _truncate_to_bytes(self, text: str, max_bytes: int) -> str:
        if max_bytes <= 0:
            return ""
        data = text.encode("utf-8")
        if len(data) <= max_bytes:
            return text
        return data[:max_bytes].decode("utf-8", errors="ignore")

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

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, PdfVectorSearchMultiArgs):
            return ToolCallDisplay(summary="pdf_vector_search_multi")

        summary = f"pdf_vector_search_multi: {len(event.args.stores)} store(s)"
        return ToolCallDisplay(
            summary=summary,
            details={
                "query": event.args.query,
                "stores": len(event.args.stores),
                "base_budget_bytes": event.args.base_budget_bytes,
                "budget_step_bytes": event.args.budget_step_bytes,
                "budget_steps": event.args.budget_steps,
                "max_results_per_store": event.args.max_results_per_store,
                "max_total_results": event.args.max_total_results,
                "max_result_chars": event.args.max_result_chars,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, PdfVectorSearchMultiResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = (
            f"Found {event.result.count} result(s) across "
            f"{event.result.store_count} store(s)"
        )
        warnings = event.result.errors[:]
        if event.result.truncated:
            warnings.append("Results truncated by budget or max_total_results limits")

        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=warnings,
            details={
                "count": event.result.count,
                "store_count": event.result.store_count,
                "total_budget_bytes": event.result.total_budget_bytes,
                "total_used_bytes": event.result.total_used_bytes,
                "truncated": event.result.truncated,
                "errors": event.result.errors,
                "store_summaries": event.result.store_summaries,
                "results": event.result.results,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Searching multiple embedding stores"
