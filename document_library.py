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


DEFAULT_DB_PATH = Path.home() / ".vibe" / "vectorstores" / "pdf_vectors.sqlite"


class DocumentLibraryConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    db_path: Path = Field(
        default=DEFAULT_DB_PATH,
        description="Path to the sqlite database file.",
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


class DocumentLibraryState(BaseToolState):
    pass


class DocumentLibraryArgs(BaseModel):
    query: str = Field(description="Search query.")
    db_path: str | None = Field(default=None, description="Override database path.")
    embedding_model: str | None = Field(
        default=None, description="Override embedding model."
    )
    top_k: int | None = Field(default=None, description="Number of results to return.")
    min_score: float | None = Field(default=None, description="Minimum similarity score.")
    max_result_chars: int | None = Field(
        default=None, description="Override snippet length."
    )


class DocumentLibraryResultItem(BaseModel):
    score: float
    source_path: str
    chunk_index: int
    unit: str
    start_index: int | None
    end_index: int | None
    content: str


class DocumentLibraryResult(BaseModel):
    results: list[DocumentLibraryResultItem]
    count: int
    db_path: str


class DocumentLibrary(
    BaseTool[
        DocumentLibraryArgs,
        DocumentLibraryResult,
        DocumentLibraryConfig,
        DocumentLibraryState,
    ],
    ToolUIData[DocumentLibraryArgs, DocumentLibraryResult],
):
    description: ClassVar[str] = (
        "Search a local document library using embeddings in a sqlite vector store."
    )

    async def run(self, args: DocumentLibraryArgs) -> DocumentLibraryResult:
        if not args.query.strip():
            raise ToolError("query cannot be empty.")

        db_path = Path(args.db_path).expanduser() if args.db_path else self.config.db_path
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

        query_vec = self._embed_text(embedding_model, args.query)

        if not db_path.exists():
            raise ToolError(f"Database not found at: {db_path}")

        results: list[tuple[float, DocumentLibraryResultItem]] = []
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute(
                """
                SELECT source_path, chunk_index, unit, start_index, end_index, content, embedding
                FROM pdf_chunks
                """
            )
            for row in cursor:
                embedding = self._unpack_embedding(row[6])
                if not embedding:
                    continue
                score = self._dot(query_vec, embedding)
                if args.min_score is not None and score < args.min_score:
                    continue
                item = DocumentLibraryResultItem(
                    score=score,
                    source_path=row[0],
                    chunk_index=row[1],
                    unit=row[2],
                    start_index=row[3],
                    end_index=row[4],
                    content=row[5][:max_chars],
                )
                if len(results) < top_k:
                    heappush(results, (score, item))
                else:
                    heappushpop(results, (score, item))

        results_sorted = [
            item for _, item in sorted(results, key=lambda r: r[0], reverse=True)
        ]
        return DocumentLibraryResult(
            results=results_sorted,
            count=len(results_sorted),
            db_path=str(db_path),
        )

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
        if not isinstance(event.args, DocumentLibraryArgs):
            return ToolCallDisplay(summary="document_library")

        return ToolCallDisplay(
            summary="document_library",
            details={
                "query": event.args.query,
                "top_k": event.args.top_k,
                "min_score": event.args.min_score,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, DocumentLibraryResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = f"Found {event.result.count} result(s)"
        return ToolResultDisplay(
            success=True,
            message=message,
            details={
                "db_path": event.result.db_path,
                "count": event.result.count,
                "results": event.result.results,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Searching document library"
