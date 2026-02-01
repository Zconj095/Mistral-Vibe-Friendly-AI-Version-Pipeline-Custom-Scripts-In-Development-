from __future__ import annotations

from dataclasses import dataclass
from array import array
from heapq import heappush, heappushpop
from pathlib import Path
import csv
import json
import sqlite3
from typing import TYPE_CHECKING, ClassVar, Iterable
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


SUPPORTED_CATALOG_FORMATS = {"auto", "json", "jsonl", "csv", "tsv"}
SUPPORTED_SCORE_MODES = {"rank", "raw", "combined"}


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


class VectorSearchFederatedConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    ollama_url: str = Field(
        default="http://127.0.0.1:11434",
        description="Base URL for the Ollama/GPT-OSS server.",
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        description="Default embedding model to use with Ollama/GPT-OSS.",
    )
    max_stores: int = Field(
        default=5000, description="Maximum stores to scan."
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
    score_mode: str = Field(
        default="rank", description="rank, raw, or combined."
    )


class VectorSearchFederatedState(BaseToolState):
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


class VectorSearchFederatedArgs(BaseModel):
    query: str = Field(description="Search query.")
    stores: list[EmbeddingStore] | None = Field(
        default=None, description="Embedding stores to search."
    )
    catalog_path: str | None = Field(
        default=None, description="Optional catalog file listing stores."
    )
    catalog_format: str | None = Field(
        default="auto", description="auto, json, jsonl, csv, tsv."
    )
    max_stores: int | None = Field(
        default=None, description="Override max_stores."
    )
    max_results_per_store: int | None = Field(
        default=None, description="Override max_results_per_store."
    )
    max_total_results: int | None = Field(
        default=None, description="Override max_total_results."
    )
    max_result_chars: int | None = Field(
        default=None, description="Override max_result_chars."
    )
    min_score: float | None = Field(default=None, description="Minimum similarity score.")
    score_mode: str | None = Field(
        default=None, description="rank, raw, or combined."
    )


class VectorSearchFederatedResultItem(BaseModel):
    score: float
    rank_score: float
    store_rank: int
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


class StoreSummary(BaseModel):
    label: str
    store_path: str
    store_type: str
    embedding_model: str
    result_count: int
    truncated: bool
    errors: list[str]


class VectorSearchFederatedResult(BaseModel):
    results: list[VectorSearchFederatedResultItem]
    count: int
    store_count: int
    truncated: bool
    store_summaries: list[StoreSummary]
    errors: list[str]


class VectorSearchFederated(
    BaseTool[
        VectorSearchFederatedArgs,
        VectorSearchFederatedResult,
        VectorSearchFederatedConfig,
        VectorSearchFederatedState,
    ],
    ToolUIData[VectorSearchFederatedArgs, VectorSearchFederatedResult],
):
    description: ClassVar[str] = (
        "Search many embedding stores with heterogeneous embedding sizes."
    )

    async def run(self, args: VectorSearchFederatedArgs) -> VectorSearchFederatedResult:
        if not args.query.strip():
            raise ToolError("query cannot be empty.")

        stores: list[EmbeddingStore] = []
        if args.stores:
            stores.extend(args.stores)

        errors: list[str] = []
        truncated = False

        if args.catalog_path:
            catalog_stores, catalog_errors = self._load_catalog(
                args.catalog_path, args.catalog_format
            )
            stores.extend(catalog_stores)
            errors.extend(catalog_errors)

        if not stores:
            raise ToolError("At least one store is required.")

        max_stores = args.max_stores if args.max_stores is not None else self.config.max_stores
        if max_stores <= 0:
            raise ToolError("max_stores must be a positive integer.")
        if len(stores) > max_stores:
            stores = stores[:max_stores]
            truncated = True
            errors.append("stores truncated to max_stores")

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
        score_mode = (
            args.score_mode if args.score_mode is not None else self.config.score_mode
        ).strip().lower()

        if max_results_per_store <= 0:
            raise ToolError("max_results_per_store must be a positive integer.")
        if max_total_results <= 0:
            raise ToolError("max_total_results must be a positive integer.")
        if max_result_chars <= 0:
            raise ToolError("max_result_chars must be a positive integer.")
        if score_mode not in SUPPORTED_SCORE_MODES:
            raise ToolError("score_mode must be rank, raw, or combined.")

        query_cache: dict[str, list[float]] = {}
        results: list[VectorSearchFederatedResultItem] = []
        summaries: list[StoreSummary] = []

        for index, store in enumerate(stores, start=1):
            label = store.label or Path(store.path).name
            store_errors: list[str] = []
            try:
                store_type = self._normalize_store_type(store.store_type)
            except ToolError as exc:
                store_errors.append(str(exc))
                summaries.append(
                    StoreSummary(
                        label=label,
                        store_path=store.path,
                        store_type=str(store.store_type or "db"),
                        embedding_model=store.embedding_model or self.config.embedding_model,
                        result_count=0,
                        truncated=False,
                        errors=store_errors,
                    )
                )
                errors.extend(store_errors)
                continue

            try:
                path = self._resolve_path(store.path)
            except ToolError as exc:
                store_errors.append(str(exc))
                summaries.append(
                    StoreSummary(
                        label=label,
                        store_path=store.path,
                        store_type=store_type,
                        embedding_model=store.embedding_model or self.config.embedding_model,
                        result_count=0,
                        truncated=False,
                        errors=store_errors,
                    )
                )
                errors.extend(store_errors)
                continue

            store_max_results = store.max_results or max_results_per_store
            store_max_chars = store.max_result_chars or max_result_chars
            store_min_score = store.min_score if store.min_score is not None else args.min_score
            if store_max_results <= 0:
                store_errors.append("max_results must be a positive integer.")
            if store_max_chars <= 0:
                store_errors.append("max_result_chars must be a positive integer.")
            if store_errors:
                summaries.append(
                    StoreSummary(
                        label=label,
                        store_path=str(path),
                        store_type=store_type,
                        embedding_model=store.embedding_model or self.config.embedding_model,
                        result_count=0,
                        truncated=False,
                        errors=store_errors,
                    )
                )
                errors.extend(store_errors)
                continue

            embedding_model = store.embedding_model or self.config.embedding_model
            if embedding_model not in query_cache:
                query_cache[embedding_model] = self._embed_text(
                    embedding_model, args.query
                )
            query_vec = query_cache[embedding_model]

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

            ranked = self._rank_rows(rows, store_max_results)
            store_truncated = len(rows) >= store_max_results

            for store_rank, row, rank_score in ranked:
                results.append(
                    VectorSearchFederatedResultItem(
                        score=row.score,
                        rank_score=rank_score,
                        store_rank=store_rank,
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

            summaries.append(
                StoreSummary(
                    label=label,
                    store_path=str(path),
                    store_type=store_type,
                    embedding_model=embedding_model,
                    result_count=len(rows),
                    truncated=store_truncated,
                    errors=store_errors,
                )
            )
            errors.extend(store_errors)

        if not results:
            raise ToolError("No results returned from the selected stores.")

        results = self._sort_results(results, score_mode)
        if len(results) > max_total_results:
            results = results[:max_total_results]
            truncated = True

        return VectorSearchFederatedResult(
            results=results,
            count=len(results),
            store_count=len(stores),
            truncated=truncated,
            store_summaries=summaries,
            errors=errors,
        )

    def _rank_rows(
        self, rows: list[_SearchRow], max_results: int
    ) -> list[tuple[int, _SearchRow, float]]:
        if not rows:
            return []
        denom = max(1, len(rows) - 1)
        ranked: list[tuple[int, _SearchRow, float]] = []
        for idx, row in enumerate(rows, start=1):
            rank_score = 1.0 if len(rows) == 1 else 1.0 - ((idx - 1) / denom)
            if max_results > 1:
                rank_score = max(rank_score, 0.0)
            ranked.append((idx, row, round(rank_score, 6)))
        return ranked

    def _sort_results(
        self, results: list[VectorSearchFederatedResultItem], score_mode: str
    ) -> list[VectorSearchFederatedResultItem]:
        if score_mode == "raw":
            return sorted(results, key=lambda item: (-item.score, item.store_label))
        if score_mode == "combined":
            return sorted(
                results,
                key=lambda item: (-(item.rank_score + max(item.score, 0.0)) / 2, item.store_label),
            )
        return sorted(results, key=lambda item: (-item.rank_score, item.store_label))

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

    def _load_catalog(
        self, raw_path: str, fmt: str | None
    ) -> tuple[list[EmbeddingStore], list[str]]:
        errors: list[str] = []
        stores: list[EmbeddingStore] = []

        path = self._resolve_path(raw_path)
        if not path.exists():
            errors.append(f"Catalog not found: {path}")
            return [], errors
        if path.is_dir():
            errors.append(f"Catalog path is a directory: {path}")
            return [], errors

        format_value = (fmt or "auto").strip().lower()
        if format_value not in SUPPORTED_CATALOG_FORMATS:
            errors.append("catalog_format must be auto, json, jsonl, csv, or tsv.")
            return [], errors

        if format_value == "auto":
            format_value = self._detect_format(path)

        if format_value == "json":
            try:
                payload = json.loads(path.read_text("utf-8", errors="ignore"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"Invalid catalog JSON: {exc}")
                return [], errors
            entries = payload
            if isinstance(payload, dict):
                entries = payload.get("stores") or payload.get("items") or []
            if isinstance(entries, list):
                for entry in entries:
                    store = self._coerce_store_entry(entry, errors)
                    if store:
                        stores.append(store)
            else:
                errors.append("Catalog JSON must contain a list or stores/items array.")
            return stores, errors

        if format_value == "jsonl":
            for entry in self._iter_jsonl_entries(path, errors):
                store = self._coerce_store_entry(entry, errors)
                if store:
                    stores.append(store)
            return stores, errors

        delimiter = "," if format_value == "csv" else "\t"
        for entry in self._iter_csv_entries(path, delimiter, errors):
            store = self._coerce_store_entry(entry, errors)
            if store:
                stores.append(store)
        return stores, errors

    def _detect_format(self, path: Path) -> str:
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    if stripped.startswith("{") or stripped.startswith("["):
                        return "json"
                    if "\t" in stripped:
                        return "tsv"
                    if "," in stripped:
                        return "csv"
                    return "jsonl"
        except OSError:
            return "json"
        return "jsonl"

    def _iter_jsonl_entries(
        self, path: Path, errors: list[str]
    ) -> Iterable[dict]:
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        data = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        errors.append(f"Invalid JSONL line: {exc}")
                        continue
                    if isinstance(data, dict):
                        yield data
        except OSError as exc:
            errors.append(f"Failed to read catalog: {exc}")

    def _iter_csv_entries(
        self, path: Path, delimiter: str, errors: list[str]
    ) -> Iterable[dict]:
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                reader = csv.DictReader(handle, delimiter=delimiter)
                for row in reader:
                    if not row:
                        continue
                    yield row
        except OSError as exc:
            errors.append(f"Failed to read catalog: {exc}")

    def _coerce_store_entry(
        self, entry: object, errors: list[str]
    ) -> EmbeddingStore | None:
        if isinstance(entry, EmbeddingStore):
            return entry
        if isinstance(entry, dict):
            if not entry.get("path"):
                errors.append("Catalog entry missing path.")
                return None
            try:
                return EmbeddingStore(
                    path=str(entry.get("path", "")),
                    store_type=entry.get("store_type"),
                    embedding_model=entry.get("embedding_model"),
                    label=entry.get("label"),
                    min_score=self._coerce_float(entry.get("min_score")),
                    max_results=self._coerce_int(entry.get("max_results")),
                    max_result_chars=self._coerce_int(entry.get("max_result_chars")),
                )
            except Exception as exc:
                errors.append(f"Invalid catalog entry: {exc}")
        return None

    def _coerce_int(self, value: object) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _coerce_float(self, value: object) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

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
            with request.urlopen(req, timeout=120) as resp:
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
        try:
            arr.frombytes(blob)
        except Exception:
            return []
        return list(arr)

    def _dot(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            return 0.0
        return sum(l * r for l, r in zip(left, right))

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, VectorSearchFederatedArgs):
            return ToolCallDisplay(summary="vector_search_federated")

        summary = f"vector_search_federated: {len(event.args.stores or [])} store(s)"
        return ToolCallDisplay(
            summary=summary,
            details={
                "query": event.args.query,
                "store_count": len(event.args.stores or []),
                "catalog_path": event.args.catalog_path,
                "max_results_per_store": event.args.max_results_per_store,
                "max_total_results": event.args.max_total_results,
                "score_mode": event.args.score_mode,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, VectorSearchFederatedResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = (
            f"Found {event.result.count} result(s) across "
            f"{event.result.store_count} store(s)"
        )
        warnings = event.result.errors[:]
        if event.result.truncated:
            warnings.append("Results truncated by limits")

        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=warnings,
            details={
                "count": event.result.count,
                "store_count": event.result.store_count,
                "truncated": event.result.truncated,
                "errors": event.result.errors,
                "store_summaries": event.result.store_summaries,
                "results": event.result.results,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Searching federated embedding stores"
