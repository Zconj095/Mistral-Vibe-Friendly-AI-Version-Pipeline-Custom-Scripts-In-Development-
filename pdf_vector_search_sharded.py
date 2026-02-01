from __future__ import annotations

import json
from array import array
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


DEFAULT_MANIFEST_PATH = Path.home() / ".vibe" / "vectorstores" / "pdf_vectors_manifest.json"


class PdfVectorSearchShardedConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    manifest_path: Path = Field(
        default=DEFAULT_MANIFEST_PATH,
        description="Path to the manifest JSON file.",
    )
    ollama_url: str = Field(
        default="http://127.0.0.1:11434",
        description="Base URL for the Ollama/GPT-OSS server.",
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        description="Embedding model to use with Ollama/GPT-OSS.",
    )
    max_results: int = Field(
        default=5, description="Maximum number of results to return."
    )
    max_result_chars: int = Field(
        default=800, description="Maximum characters to return per result."
    )


class PdfVectorSearchShardedState(BaseToolState):
    pass


class PdfVectorSearchShardedArgs(BaseModel):
    query: str = Field(description="Search query.")
    manifest_path: str | None = Field(default=None, description="Override manifest path.")
    embedding_model: str | None = Field(default=None, description="Override embedding model.")
    top_k: int | None = Field(default=None, description="Number of results to return.")
    min_score: float | None = Field(default=None, description="Minimum similarity score.")
    max_result_chars: int | None = Field(default=None, description="Override snippet length.")


class PdfVectorSearchShardedResultItem(BaseModel):
    score: float
    source_path: str
    chunk_index: int
    unit: str
    start_index: int | None
    end_index: int | None
    content: str
    shard_path: str


class PdfVectorSearchShardedResult(BaseModel):
    results: list[PdfVectorSearchShardedResultItem]
    count: int
    shard_count: int
    manifest_path: str
    errors: list[str]


class PdfVectorSearchSharded(
    BaseTool[
        PdfVectorSearchShardedArgs,
        PdfVectorSearchShardedResult,
        PdfVectorSearchShardedConfig,
        PdfVectorSearchShardedState,
    ],
    ToolUIData[PdfVectorSearchShardedArgs, PdfVectorSearchShardedResult],
):
    description: ClassVar[str] = (
        "Search sharded PDF vectors using Ollama/GPT-OSS embeddings."
    )

    async def run(self, args: PdfVectorSearchShardedArgs) -> PdfVectorSearchShardedResult:
        if not args.query.strip():
            raise ToolError("query cannot be empty.")

        manifest_path = (
            Path(args.manifest_path).expanduser()
            if args.manifest_path
            else self.config.manifest_path
        )
        embedding_model = args.embedding_model or self.config.embedding_model
        top_k = args.top_k if args.top_k is not None else self.config.max_results
        max_chars = (
            args.max_result_chars
            if args.max_result_chars is not None
            else self.config.max_result_chars
        )
        if top_k <= 0:
            raise ToolError("top_k must be a positive integer.")
        if max_chars <= 0:
            raise ToolError("max_result_chars must be a positive integer.")

        manifest = self._load_manifest(manifest_path)
        if not manifest.get("shards"):
            raise ToolError("Manifest has no shards to search.")
        if manifest.get("embedding_model") and manifest.get("embedding_model") != embedding_model:
            raise ToolError(
                "Embedding model mismatch with manifest. "
                "Use the same embedding_model or create a new manifest."
            )

        query_vec = self._embed_text(embedding_model, args.query)

        results: list[tuple[float, PdfVectorSearchShardedResultItem]] = []
        errors: list[str] = []
        shard_count = 0

        for shard in manifest.get("shards", []):
            shard_path = Path(shard.get("path", ""))
            if not shard_path.exists():
                errors.append(f"Shard not found: {shard_path}")
                continue
            shard_count += 1
            try:
                with sqlite3.connect(str(shard_path)) as conn:
                    cursor = conn.execute(
                        """
                        SELECT source_path, chunk_index, unit, start_index, end_index, content, embedding, embedding_dim
                        FROM pdf_chunks
                        """
                    )
                    for row in cursor:
                        if row[7] and row[7] != len(query_vec):
                            continue
                        embedding = self._unpack_embedding(row[6])
                        if not embedding:
                            continue
                        score = self._dot(query_vec, embedding)
                        if args.min_score is not None and score < args.min_score:
                            continue
                        item = PdfVectorSearchShardedResultItem(
                            score=score,
                            source_path=row[0],
                            chunk_index=row[1],
                            unit=row[2],
                            start_index=row[3],
                            end_index=row[4],
                            content=row[5][:max_chars],
                            shard_path=str(shard_path),
                        )
                        if len(results) < top_k:
                            heappush(results, (score, item))
                        else:
                            heappushpop(results, (score, item))
            except Exception as exc:
                errors.append(f"{shard_path}: {exc}")

        results_sorted = [
            item for _, item in sorted(results, key=lambda r: r[0], reverse=True)
        ]
        return PdfVectorSearchShardedResult(
            results=results_sorted,
            count=len(results_sorted),
            shard_count=shard_count,
            manifest_path=str(manifest_path),
            errors=errors,
        )

    def _load_manifest(self, path: Path) -> dict:
        if not path.exists():
            raise ToolError(f"Manifest not found at: {path}")
        try:
            return json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ToolError(f"Invalid manifest file: {exc}") from exc

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
        if not isinstance(event.args, PdfVectorSearchShardedArgs):
            return ToolCallDisplay(summary="pdf_vector_search_sharded")

        summary = "pdf_vector_search_sharded"
        return ToolCallDisplay(
            summary=summary,
            details={
                "query": event.args.query,
                "top_k": event.args.top_k,
                "min_score": event.args.min_score,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, PdfVectorSearchShardedResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = f"Found {event.result.count} result(s) across {event.result.shard_count} shard(s)"
        warnings = event.result.errors[:]
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=warnings,
            details={
                "count": event.result.count,
                "shard_count": event.result.shard_count,
                "manifest_path": event.result.manifest_path,
                "results": event.result.results,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Searching sharded PDF vectors"
