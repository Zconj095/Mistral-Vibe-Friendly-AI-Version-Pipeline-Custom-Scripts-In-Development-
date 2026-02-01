
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import os
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
WORKER_MODES = {"auto", "thread", "process"}


@dataclass
class _WorkItem:
    doc_index: int
    doc_id: str
    name: str | None
    source: str | None
    path: str | None
    content: str | None
    max_source_bytes: int
    chunk_mode: str
    chunk_size: int
    chunk_overlap: int
    max_chunks_per_doc: int
    max_chunk_bytes: int
    min_token_length: int
    max_tokens_per_chunk: int
    max_tokens_per_document: int
    preview_chars: int


def _preview_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _split_paragraphs(content: str) -> list[str]:
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


def _limit_chunk_bytes(text: str, max_bytes: int) -> tuple[str, bool]:
    if max_bytes <= 0:
        return "", True
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text, False
    trimmed = data[:max_bytes].decode("utf-8", errors="ignore")
    return trimmed, True


def _chunk_text(
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
            chunk_text, trimmed = _limit_chunk_bytes(chunk_text, max_chunk_bytes)
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
        units = _split_paragraphs(content)
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
        chunk_text, trimmed = _limit_chunk_bytes(chunk_text, max_chunk_bytes)
        start_index = index + 1
        end_index = start_index + len(subset) - 1
        chunks.append((chunk_text, start_index, end_index, unit_label, trimmed))
        chunk_index += 1
        index += step

    if index < len(units):
        truncated = True
    return chunks, truncated


def _extract_token_counts(text: str, min_len: int) -> dict[str, int]:
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


def _select_keywords(token_counts: dict[str, int], max_tokens: int) -> list[str]:
    if not token_counts:
        return []
    ordered = sorted(token_counts.items(), key=lambda item: (-item[1], item[0]))
    if max_tokens > 0:
        ordered = ordered[:max_tokens]
    return [token for token, _ in ordered]


def _process_document(payload: dict) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    chunks_out: list[dict] = []
    doc_data: dict | None = None

    try:
        content = payload.get("content")
        path = payload.get("path")
        if content and path:
            raise ToolError("Provide content or path, not both.")

        source_path: str | None = None
        if path:
            path_obj = Path(path)
            if not path_obj.exists():
                raise ToolError(f"Path not found: {path_obj}")
            if path_obj.is_dir():
                raise ToolError(f"Path is a directory, not a file: {path_obj}")
            size_bytes = path_obj.stat().st_size
            if size_bytes > payload["max_source_bytes"]:
                raise ToolError(
                    f"{path_obj} exceeds max_source_bytes ({size_bytes} > {payload['max_source_bytes']})."
                )
            content = path_obj.read_text("utf-8", errors="ignore")
            source_path = str(path_obj)
        elif content is not None:
            size_bytes = len(content.encode("utf-8"))
            if size_bytes > payload["max_source_bytes"]:
                raise ToolError(
                    f"content exceeds max_source_bytes ({size_bytes} > {payload['max_source_bytes']})."
                )
        else:
            raise ToolError("Item has no content to process.")

        chunk_entries, doc_truncated = _chunk_text(
            content,
            payload["chunk_mode"],
            payload["chunk_size"],
            payload["chunk_overlap"],
            payload["max_chunks_per_doc"],
            payload["max_chunk_bytes"],
        )
        if doc_truncated:
            warnings.append("Truncated by max_chunks_per_doc.")

        doc_token_counts: dict[str, int] = {}
        for chunk_text, start_index, end_index, unit, chunk_trimmed in chunk_entries:
            token_counts = _extract_token_counts(
                chunk_text, payload["min_token_length"]
            )
            token_count = sum(token_counts.values())
            keywords = _select_keywords(
                token_counts, payload["max_tokens_per_chunk"]
            )
            for token, count in token_counts.items():
                doc_token_counts[token] = doc_token_counts.get(token, 0) + count
            chunks_out.append(
                {
                    "document_index": payload["doc_index"],
                    "unit": unit,
                    "start_index": start_index,
                    "end_index": end_index,
                    "preview": _preview_text(chunk_text, payload["preview_chars"]),
                    "token_count": token_count,
                    "keywords": keywords,
                    "truncated": chunk_trimmed,
                }
            )

        doc_keywords = _select_keywords(
            doc_token_counts, payload["max_tokens_per_document"]
        )
        doc_data = {
            "doc_index": payload["doc_index"],
            "doc_id": payload["doc_id"],
            "name": payload.get("name"),
            "source_path": source_path,
            "source_type": payload.get("source") or "item",
            "chunk_count": len(chunk_entries),
            "token_count": sum(doc_token_counts.values()),
            "preview": _preview_text(content, payload["preview_chars"]),
            "keywords": doc_keywords,
            "truncated": doc_truncated,
        }
    except Exception as exc:
        errors.append(str(exc))

    return {
        "doc_index": payload.get("doc_index"),
        "doc": doc_data,
        "chunks": chunks_out,
        "errors": errors,
        "warnings": warnings,
    }

class ContextParallelChunkingConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    worker_mode: str = Field(
        default="auto", description="auto, thread, or process."
    )
    max_workers: int = Field(
        default=0, description="0 uses os.cpu_count()."
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


class ContextParallelChunkingState(BaseToolState):
    pass


class ParallelDocumentItem(BaseModel):
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


class ContextParallelChunkingArgs(BaseModel):
    items: list[ParallelDocumentItem] = Field(
        description="Document items to process."
    )
    worker_mode: str | None = Field(
        default=None, description="Override worker_mode."
    )
    max_workers: int | None = Field(
        default=None, description="Override max_workers."
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


class ParallelDocumentSummary(BaseModel):
    index: int
    id: str
    name: str | None
    source_path: str | None
    source_type: str
    chunk_count: int
    token_count: int
    preview: str
    keywords: list[str]
    truncated: bool


class ParallelChunk(BaseModel):
    index: int
    document_index: int
    unit: str
    start_index: int | None
    end_index: int | None
    preview: str
    token_count: int
    keywords: list[str]
    truncated: bool


class ContextParallelChunkingResult(BaseModel):
    documents: list[ParallelDocumentSummary]
    chunks: list[ParallelChunk]
    document_count: int
    chunk_count: int
    worker_mode: str
    workers_used: int
    truncated: bool
    errors: list[str]
    warnings: list[str]


class ContextParallelChunking(
    BaseTool[
        ContextParallelChunkingArgs,
        ContextParallelChunkingResult,
        ContextParallelChunkingConfig,
        ContextParallelChunkingState,
    ],
    ToolUIData[ContextParallelChunkingArgs, ContextParallelChunkingResult],
):
    description: ClassVar[str] = (
        "Parallelize document chunking and analysis across multiple workers."
    )

    async def run(
        self, args: ContextParallelChunkingArgs
    ) -> ContextParallelChunkingResult:
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
        if default_chunk_mode not in CHUNK_MODES:
            raise ToolError("chunk_mode must be words, paragraphs, sentences, lines, or chars.")
        if default_chunk_size <= 0:
            raise ToolError("chunk_size must be a positive integer.")
        if default_chunk_overlap < 0:
            raise ToolError("chunk_overlap must be >= 0.")
        if default_chunk_overlap >= default_chunk_size:
            raise ToolError("chunk_overlap must be smaller than chunk_size.")

        worker_mode = (
            args.worker_mode if args.worker_mode is not None else self.config.worker_mode
        ).strip().lower()
        if worker_mode not in WORKER_MODES:
            raise ToolError("worker_mode must be auto, thread, or process.")

        workers_requested = (
            args.max_workers if args.max_workers is not None else self.config.max_workers
        )
        workers_used = self._resolve_workers(workers_requested, len(args.items))
        worker_mode = self._resolve_worker_mode(worker_mode, workers_used)

        work_items, truncated, errors, warnings = self._prepare_work_items(
            args.items,
            max_source_bytes,
            max_total_bytes,
            default_chunk_mode,
            default_chunk_size,
            default_chunk_overlap,
            max_chunks_per_doc,
            max_chunk_bytes,
            min_token_length,
            max_tokens_per_chunk,
            max_tokens_per_document,
            preview_chars,
        )

        if not work_items:
            raise ToolError("No valid items to process.")

        results, used_mode, used_workers = self._run_parallel(
            work_items, worker_mode, workers_used
        )
        if used_mode != worker_mode:
            warnings.append(
                f"worker_mode changed to {used_mode} due to runtime constraints."
            )
        worker_mode = used_mode
        workers_used = used_workers

        documents: list[ParallelDocumentSummary] = []
        chunks: list[ParallelChunk] = []
        total_chunks = 0
        for result in results:
            doc = result.get("doc")
            doc_index = result.get("doc_index")
            if doc is None:
                for error in result.get("errors", []):
                    errors.append(f"document[{doc_index}]: {error}")
                continue

            documents.append(
                ParallelDocumentSummary(
                    index=doc["doc_index"],
                    id=str(doc["doc_id"]),
                    name=doc.get("name"),
                    source_path=doc.get("source_path"),
                    source_type=str(doc.get("source_type") or "item"),
                    chunk_count=int(doc.get("chunk_count") or 0),
                    token_count=int(doc.get("token_count") or 0),
                    preview=str(doc.get("preview") or ""),
                    keywords=list(doc.get("keywords") or []),
                    truncated=bool(doc.get("truncated")),
                )
            )

            for warning in result.get("warnings", []):
                warnings.append(f"document[{doc_index}]: {warning}")

            for chunk in result.get("chunks", []):
                if max_chunks > 0 and total_chunks >= max_chunks:
                    truncated = True
                    break
                total_chunks += 1
                chunks.append(
                    ParallelChunk(
                        index=total_chunks,
                        document_index=int(chunk["document_index"]),
                        unit=str(chunk["unit"]),
                        start_index=chunk.get("start_index"),
                        end_index=chunk.get("end_index"),
                        preview=str(chunk.get("preview") or ""),
                        token_count=int(chunk.get("token_count") or 0),
                        keywords=list(chunk.get("keywords") or []),
                        truncated=bool(chunk.get("truncated")),
                    )
                )

        documents.sort(key=lambda item: item.index)

        return ContextParallelChunkingResult(
            documents=documents,
            chunks=chunks,
            document_count=len(documents),
            chunk_count=len(chunks),
            worker_mode=worker_mode,
            workers_used=workers_used,
            truncated=truncated,
            errors=errors,
            warnings=warnings,
        )

    def _resolve_workers(self, requested: int, total_items: int) -> int:
        if requested is None:
            requested = 0
        if requested <= 0:
            requested = os.cpu_count() or 1
        if total_items > 0:
            return max(1, min(requested, total_items))
        return max(1, requested)

    def _resolve_worker_mode(self, mode: str, workers: int) -> str:
        if mode != "auto":
            return mode
        if workers <= 1:
            return "thread"
        return "process"

    def _prepare_work_items(
        self,
        items: list[ParallelDocumentItem],
        max_source_bytes: int,
        max_total_bytes: int,
        default_chunk_mode: str,
        default_chunk_size: int,
        default_chunk_overlap: int,
        max_chunks_per_doc: int,
        max_chunk_bytes: int,
        min_token_length: int,
        max_tokens_per_chunk: int,
        max_tokens_per_document: int,
        preview_chars: int,
    ) -> tuple[list[_WorkItem], bool, list[str], list[str]]:
        work_items: list[_WorkItem] = []
        errors: list[str] = []
        warnings: list[str] = []
        truncated = False
        total_bytes = 0

        for idx, item in enumerate(items, start=1):
            try:
                if item.content and item.path:
                    raise ToolError("Provide content or path, not both.")

                size_bytes = 0
                resolved_path: str | None = None
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
                    resolved_path = str(path)
                elif item.content is not None:
                    size_bytes = len(item.content.encode("utf-8"))
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

                doc_id = item.id or item.name or resolved_path or f"item-{idx}"
                work_items.append(
                    _WorkItem(
                        doc_index=len(work_items) + 1,
                        doc_id=str(doc_id),
                        name=item.name,
                        source=item.source,
                        path=resolved_path,
                        content=item.content,
                        max_source_bytes=max_source_bytes,
                        chunk_mode=chunk_mode,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        max_chunks_per_doc=max_chunks_per_doc,
                        max_chunk_bytes=max_chunk_bytes,
                        min_token_length=min_token_length,
                        max_tokens_per_chunk=max_tokens_per_chunk,
                        max_tokens_per_document=max_tokens_per_document,
                        preview_chars=preview_chars,
                    )
                )
                total_bytes += size_bytes
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")

        return work_items, truncated, errors, warnings

    def _run_parallel(
        self, work_items: list[_WorkItem], worker_mode: str, workers_used: int
    ) -> tuple[list[dict], str, int]:
        payloads = [item.__dict__ for item in work_items]
        if workers_used <= 1:
            results = [_process_document(payload) for payload in payloads]
            results.sort(key=lambda item: item.get("doc_index") or 0)
            return results, "thread", 1

        if worker_mode == "process":
            try:
                with ProcessPoolExecutor(max_workers=workers_used) as executor:
                    results = self._collect_futures(executor, payloads)
                results.sort(key=lambda item: item.get("doc_index") or 0)
                return results, "process", workers_used
            except Exception:
                worker_mode = "thread"

        with ThreadPoolExecutor(max_workers=workers_used) as executor:
            results = self._collect_futures(executor, payloads)
        results.sort(key=lambda item: item.get("doc_index") or 0)
        return results, "thread", workers_used

    def _collect_futures(self, executor, payloads: list[dict]) -> list[dict]:
        futures = {executor.submit(_process_document, payload): payload for payload in payloads}
        results: list[dict] = []
        for future in as_completed(futures):
            payload = futures[future]
            doc_index = payload.get("doc_index")
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(
                    {
                        "doc_index": doc_index,
                        "doc": None,
                        "chunks": [],
                        "errors": [str(exc)],
                        "warnings": [],
                    }
                )
        return results

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
        if not isinstance(event.args, ContextParallelChunkingArgs):
            return ToolCallDisplay(summary="context_parallel_chunking")

        summary = f"context_parallel_chunking: {len(event.args.items)} item(s)"
        return ToolCallDisplay(
            summary=summary,
            details={
                "item_count": len(event.args.items),
                "worker_mode": event.args.worker_mode,
                "max_workers": event.args.max_workers,
                "max_chunks": event.args.max_chunks,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextParallelChunkingResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = (
            f"Processed {event.result.document_count} document(s), "
            f"{event.result.chunk_count} chunk(s)"
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
                "worker_mode": event.result.worker_mode,
                "workers_used": event.result.workers_used,
                "truncated": event.result.truncated,
                "errors": event.result.errors,
                "warnings": event.result.warnings,
                "documents": event.result.documents,
                "chunks": event.result.chunks,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Parallel chunking documents"
