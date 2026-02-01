from __future__ import annotations

import fnmatch
from pathlib import Path
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


class ProcessDocumentsRecursiveConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_source_bytes: int = Field(
        default=5_000_000, description="Maximum size per file (bytes)."
    )
    max_total_bytes: int = Field(
        default=20_000_000, description="Maximum total bytes across files."
    )
    max_chunk_bytes: int = Field(
        default=200_000, description="Maximum size per chunk (bytes)."
    )
    max_chunks: int = Field(
        default=500, description="Maximum number of chunks to return."
    )
    max_files: int = Field(
        default=500, description="Maximum files to process."
    )
    default_unit: str = Field(default="lines", description="lines or chars.")
    default_chunk_size: int = Field(default=200, description="Chunk size in units.")
    default_overlap: int = Field(default=0, description="Overlap in units.")
    default_mode: str = Field(
        default="chunk", description="chunk or read."
    )
    default_exclude_globs: list[str] = Field(
        default=[
            "**/.git/**",
            "**/.svn/**",
            "**/.hg/**",
            "**/node_modules/**",
            "**/.venv/**",
            "**/venv/**",
            "**/.idea/**",
            "**/.vscode/**",
            "**/dist/**",
            "**/build/**",
            "**/target/**",
            "**/.mypy_cache/**",
            "**/.pytest_cache/**",
            "**/.cache/**",
            "**/__pycache__/**",
        ],
        description="Default glob patterns excluded during auto-discovery.",
    )


class ProcessDocumentsRecursiveState(BaseToolState):
    pass


class ProcessDocumentsRecursiveArgs(BaseModel):
    paths: list[str] = Field(description="Root directories or files to scan.")
    extension: str | None = Field(
        default=None, description="Single extension to match (e.g. .pdf)."
    )
    extensions: list[str] | None = Field(
        default=None, description="Extensions to match (e.g. ['.md', '.txt'])."
    )
    include_globs: list[str] | None = Field(
        default=None, description="Optional include globs."
    )
    exclude_globs: list[str] | None = Field(
        default=None, description="Optional exclude globs."
    )
    mode: str | None = Field(default=None, description="chunk or read.")
    chunk_unit: str | None = Field(default=None, description="lines or chars.")
    chunk_size: int | None = Field(default=None, description="Chunk size in units.")
    overlap: int | None = Field(default=None, description="Overlap in units.")
    max_chunks: int | None = Field(
        default=None, description="Override max chunks limit."
    )
    max_files: int | None = Field(default=None, description="Override max files limit.")
    max_source_bytes: int | None = Field(
        default=None, description="Override max_source_bytes."
    )
    max_total_bytes: int | None = Field(
        default=None, description="Override max_total_bytes."
    )
    max_chunk_bytes: int | None = Field(
        default=None, description="Override max_chunk_bytes."
    )


class DocumentEntry(BaseModel):
    path: str
    bytes_total: int
    content: str


class DocumentChunk(BaseModel):
    path: str
    chunk_index: int
    total_chunks: int
    unit: str
    start: int | None
    end: int | None
    content: str


class ProcessDocumentsRecursiveResult(BaseModel):
    mode: str
    documents: list[DocumentEntry]
    chunks: list[DocumentChunk]
    count: int
    document_count: int
    chunk_count: int
    total_bytes: int
    truncated: bool
    errors: list[str]


class ProcessDocumentsRecursive(
    BaseTool[
        ProcessDocumentsRecursiveArgs,
        ProcessDocumentsRecursiveResult,
        ProcessDocumentsRecursiveConfig,
        ProcessDocumentsRecursiveState,
    ],
    ToolUIData[ProcessDocumentsRecursiveArgs, ProcessDocumentsRecursiveResult],
):
    description: ClassVar[str] = (
        "Recursively find documents by extension and read or chunk them."
    )

    async def run(
        self, args: ProcessDocumentsRecursiveArgs
    ) -> ProcessDocumentsRecursiveResult:
        if not args.paths:
            raise ToolError("At least one path is required.")

        mode = (args.mode or self.config.default_mode).strip().lower()
        if mode not in {"chunk", "read"}:
            raise ToolError("mode must be chunk or read.")

        max_files = args.max_files if args.max_files is not None else self.config.max_files
        if max_files <= 0:
            raise ToolError("max_files must be a positive integer.")

        max_chunks = args.max_chunks if args.max_chunks is not None else self.config.max_chunks
        if max_chunks <= 0:
            raise ToolError("max_chunks must be a positive integer.")

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
        if max_source_bytes <= 0:
            raise ToolError("max_source_bytes must be a positive integer.")
        if max_total_bytes <= 0:
            raise ToolError("max_total_bytes must be a positive integer.")
        if max_chunk_bytes <= 0:
            raise ToolError("max_chunk_bytes must be a positive integer.")

        unit = (args.chunk_unit or self.config.default_unit).strip().lower()
        size = args.chunk_size if args.chunk_size is not None else self.config.default_chunk_size
        overlap = args.overlap if args.overlap is not None else self.config.default_overlap
        if mode == "chunk":
            if unit not in {"lines", "chars"}:
                raise ToolError("chunk_unit must be lines or chars.")
            if size <= 0:
                raise ToolError("chunk_size must be a positive integer.")
            if overlap < 0:
                raise ToolError("overlap must be a non-negative integer.")
            if overlap >= size:
                raise ToolError("overlap must be smaller than chunk_size.")

        include_globs = self._resolve_include_globs(args)
        exclude_globs = self._normalize_globs(
            args.exclude_globs, self.config.default_exclude_globs
        )

        files = self._gather_files(args.paths, include_globs, exclude_globs, max_files)
        if not files:
            raise ToolError("No files matched the provided paths/patterns.")

        documents: list[DocumentEntry] = []
        chunks: list[DocumentChunk] = []
        errors: list[str] = []
        total_bytes = 0
        truncated = False
        document_count = 0

        for file_path in files:
            if truncated:
                break
            if mode == "chunk" and len(chunks) >= max_chunks:
                truncated = True
                break

            try:
                size_bytes = file_path.stat().st_size
                if size_bytes > max_source_bytes:
                    raise ToolError(
                        f"{file_path} exceeds max_source_bytes ({size_bytes} > {max_source_bytes})."
                    )
                if total_bytes + size_bytes > max_total_bytes:
                    truncated = True
                    break

                content = file_path.read_text("utf-8", errors="ignore")
                total_bytes += size_bytes
                document_count += 1

                if mode == "read":
                    documents.append(
                        DocumentEntry(
                            path=str(file_path),
                            bytes_total=size_bytes,
                            content=content,
                        )
                    )
                    continue

                source_chunks = self._chunk_content(
                    content,
                    unit,
                    size,
                    overlap,
                    max_chunk_bytes,
                )

                if not source_chunks:
                    continue

                remaining = max_chunks - len(chunks)
                if len(source_chunks) > remaining:
                    source_chunks = source_chunks[:remaining]
                    self._finalize_chunks(source_chunks)
                    truncated = True

                chunks.extend(
                    [
                        DocumentChunk(
                            path=str(file_path),
                            chunk_index=chunk.chunk_index,
                            total_chunks=chunk.total_chunks,
                            unit=chunk.unit,
                            start=chunk.start,
                            end=chunk.end,
                            content=chunk.content,
                        )
                        for chunk in source_chunks
                    ]
                )
            except ToolError as exc:
                errors.append(str(exc))
            except Exception as exc:
                errors.append(f"{file_path}: {exc}")

        if mode == "chunk" and len(chunks) > max_chunks:
            chunks = chunks[:max_chunks]
            truncated = True

        return ProcessDocumentsRecursiveResult(
            mode=mode,
            documents=documents,
            chunks=chunks,
            count=len(documents) + len(chunks),
            document_count=document_count,
            chunk_count=len(chunks),
            total_bytes=total_bytes,
            truncated=truncated,
            errors=errors,
        )

    def _resolve_include_globs(
        self, args: ProcessDocumentsRecursiveArgs
    ) -> list[str]:
        if args.include_globs:
            return self._normalize_globs(args.include_globs, [])

        if args.extension and args.extensions:
            raise ToolError("Provide extension or extensions, not both.")

        ext_list: list[str] = []
        if args.extension:
            ext_list = [args.extension]
        elif args.extensions:
            ext_list = args.extensions

        normalized = self._normalize_extensions(ext_list)
        if not normalized:
            raise ToolError("Provide extension(s) or include_globs.")

        return [f"**/*{ext}" for ext in normalized]

    def _normalize_extensions(self, ext_list: list[str]) -> list[str]:
        normalized: list[str] = []
        for ext in ext_list:
            value = (ext or "").strip()
            if not value:
                continue
            if not value.startswith("."):
                value = f".{value}"
            normalized.append(value.lower())
        return sorted(set(normalized))

    def _normalize_globs(
        self, value: list[str] | None, defaults: list[str]
    ) -> list[str]:
        globs = value if value is not None else defaults
        globs = [g.strip() for g in globs if g and g.strip()]
        if not globs:
            return []
        return globs

    def _gather_files(
        self,
        paths: list[str],
        include_globs: list[str],
        exclude_globs: list[str],
        max_files: int,
    ) -> list[Path]:
        discovered: list[Path] = []
        seen: set[str] = set()

        for raw in paths:
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = self.config.effective_workdir / path
            path = path.resolve()

            if path.is_dir():
                for pattern in include_globs:
                    for file_path in path.glob(pattern):
                        if not file_path.is_file():
                            continue
                        rel = file_path.relative_to(path).as_posix()
                        if self._is_excluded(rel, exclude_globs):
                            continue
                        key = str(file_path)
                        if key in seen:
                            continue
                        seen.add(key)
                        discovered.append(file_path)
                        if len(discovered) >= max_files:
                            return discovered
            elif path.is_file():
                key = str(path)
                if key not in seen:
                    if not self._is_excluded(path.name, exclude_globs):
                        seen.add(key)
                        discovered.append(path)
                        if len(discovered) >= max_files:
                            return discovered
            else:
                raise ToolError(f"Path not found: {path}")

        return discovered

    def _is_excluded(self, rel_path: str, exclude_globs: list[str]) -> bool:
        if not exclude_globs:
            return False
        return any(fnmatch.fnmatch(rel_path, pattern) for pattern in exclude_globs)

    def _chunk_content(
        self,
        content: str,
        unit: str,
        size: int,
        overlap: int,
        max_chunk_bytes: int,
    ) -> list[DocumentChunk]:
        if not content:
            return []
        if unit == "lines":
            chunks = self._chunk_lines(content, size, overlap, max_chunk_bytes)
        else:
            chunks = self._chunk_chars(content, size, overlap, max_chunk_bytes)
        return self._finalize_chunks(chunks)

    def _chunk_lines(
        self, content: str, size: int, overlap: int, max_chunk_bytes: int
    ) -> list[DocumentChunk]:
        lines = content.splitlines()
        if not lines:
            return []
        step = size - overlap
        index = 0
        chunk_index = 0
        chunks: list[DocumentChunk] = []
        while index < len(lines):
            subset = lines[index : index + size]
            if not subset:
                break
            chunk_text = "\n".join(subset)
            self._check_chunk_bytes(chunk_text, max_chunk_bytes)
            start_line = index + 1
            end_line = start_line + len(subset) - 1
            chunk_index += 1
            chunks.append(
                DocumentChunk(
                    path="",
                    chunk_index=chunk_index,
                    total_chunks=0,
                    unit="lines",
                    start=start_line,
                    end=end_line,
                    content=chunk_text,
                )
            )
            index += step
        return chunks

    def _chunk_chars(
        self, content: str, size: int, overlap: int, max_chunk_bytes: int
    ) -> list[DocumentChunk]:
        step = size - overlap
        index = 0
        chunk_index = 0
        chunks: list[DocumentChunk] = []
        while index < len(content):
            chunk_text = content[index : index + size]
            if not chunk_text:
                break
            self._check_chunk_bytes(chunk_text, max_chunk_bytes)
            start = index
            end = index + len(chunk_text) - 1
            chunk_index += 1
            chunks.append(
                DocumentChunk(
                    path="",
                    chunk_index=chunk_index,
                    total_chunks=0,
                    unit="chars",
                    start=start,
                    end=end,
                    content=chunk_text,
                )
            )
            index += step
        return chunks

    def _check_chunk_bytes(self, content: str, max_chunk_bytes: int) -> None:
        size = len(content.encode("utf-8"))
        if size > max_chunk_bytes:
            raise ToolError(
                f"Chunk exceeds max_chunk_bytes ({size} > {max_chunk_bytes})."
            )

    def _finalize_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total
        return chunks

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ProcessDocumentsRecursiveArgs):
            return ToolCallDisplay(summary="process_documents_recursive")

        summary = f"process_documents_recursive: {len(event.args.paths)} path(s)"
        return ToolCallDisplay(
            summary=summary,
            details={
                "paths": event.args.paths,
                "extension": event.args.extension,
                "extensions": event.args.extensions,
                "include_globs": event.args.include_globs,
                "exclude_globs": event.args.exclude_globs,
                "mode": event.args.mode,
                "chunk_unit": event.args.chunk_unit,
                "chunk_size": event.args.chunk_size,
                "overlap": event.args.overlap,
                "max_chunks": event.args.max_chunks,
                "max_files": event.args.max_files,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ProcessDocumentsRecursiveResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = (
            f"Processed {event.result.document_count} file(s) "
            f"into {event.result.chunk_count} chunk(s)"
        )
        warnings = event.result.errors[:]
        if event.result.truncated:
            warnings.append("Output truncated by size or limits")

        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=warnings,
            details={
                "mode": event.result.mode,
                "document_count": event.result.document_count,
                "chunk_count": event.result.chunk_count,
                "total_bytes": event.result.total_bytes,
                "truncated": event.result.truncated,
                "errors": event.result.errors,
                "documents": event.result.documents,
                "chunks": event.result.chunks,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Processing documents recursively"
