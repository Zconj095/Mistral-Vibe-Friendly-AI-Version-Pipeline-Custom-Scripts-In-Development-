from __future__ import annotations

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


class ChunkToolchainMultiConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_source_bytes: int = Field(
        default=5_000_000, description="Maximum size per source (bytes)."
    )
    max_total_bytes: int = Field(
        default=20_000_000, description="Maximum total size across sources (bytes)."
    )
    max_chunk_bytes: int = Field(
        default=200_000, description="Maximum size per chunk (bytes)."
    )
    max_chunks: int = Field(
        default=500, description="Maximum number of chunks to return."
    )
    max_sources_per_kind: int = Field(
        default=50, description="Maximum sources per data kind."
    )
    default_unit: str = Field(default="lines", description="lines or chars.")
    default_chunk_size: int = Field(default=200, description="Chunk size in units.")
    default_overlap: int = Field(default=0, description="Overlap in units.")


class ChunkToolchainMultiState(BaseToolState):
    pass


class ToolchainSource(BaseModel):
    path: str | None = Field(default=None, description="Path to a file.")
    content: str | None = Field(default=None, description="Inline content.")
    label: str | None = Field(default=None, description="Optional label.")


class ChunkToolchainMultiArgs(BaseModel):
    compiler_sources: list[ToolchainSource] | None = Field(
        default=None, description="Compiler data sources."
    )
    interpreter_sources: list[ToolchainSource] | None = Field(
        default=None, description="Interpreter data sources."
    )
    transpiled_sources: list[ToolchainSource] | None = Field(
        default=None, description="Transpiled data sources."
    )
    chunk_unit: str | None = Field(default=None, description="lines or chars.")
    chunk_size: int | None = Field(
        default=None, description="Chunk size in units."
    )
    overlap: int | None = Field(
        default=None, description="Overlap in units."
    )
    max_chunks: int | None = Field(
        default=None, description="Override the configured max chunks limit."
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
    max_sources_per_kind: int | None = Field(
        default=None, description="Override max_sources_per_kind."
    )


class ToolchainChunk(BaseModel):
    kind: str
    label: str
    source_path: str | None
    chunk_index: int
    total_chunks: int
    unit: str
    start: int | None
    end: int | None
    content: str


class ChunkToolchainMultiResult(BaseModel):
    chunks: list[ToolchainChunk]
    count: int
    compiler_count: int
    interpreter_count: int
    transpiled_count: int
    source_count: int
    total_bytes: int
    truncated: bool
    errors: list[str]


class ChunkToolchainMulti(
    BaseTool[
        ChunkToolchainMultiArgs,
        ChunkToolchainMultiResult,
        ChunkToolchainMultiConfig,
        ChunkToolchainMultiState,
    ],
    ToolUIData[ChunkToolchainMultiArgs, ChunkToolchainMultiResult],
):
    description: ClassVar[str] = (
        "Chunk compiler, interpreter, and transpiled data together."
    )

    async def run(self, args: ChunkToolchainMultiArgs) -> ChunkToolchainMultiResult:
        compiler_sources = args.compiler_sources or []
        interpreter_sources = args.interpreter_sources or []
        transpiled_sources = args.transpiled_sources or []

        if not compiler_sources and not interpreter_sources and not transpiled_sources:
            raise ToolError("Provide at least one compiler/interpreter/transpiled source.")

        unit = (args.chunk_unit or self.config.default_unit).strip().lower()
        if unit not in {"lines", "chars"}:
            raise ToolError("chunk_unit must be lines or chars.")

        size = args.chunk_size if args.chunk_size is not None else self.config.default_chunk_size
        if size <= 0:
            raise ToolError("chunk_size must be a positive integer.")

        overlap = args.overlap if args.overlap is not None else self.config.default_overlap
        if overlap < 0:
            raise ToolError("overlap must be a non-negative integer.")
        if overlap >= size:
            raise ToolError("overlap must be smaller than chunk_size.")

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
        max_sources_per_kind = (
            args.max_sources_per_kind
            if args.max_sources_per_kind is not None
            else self.config.max_sources_per_kind
        )
        if max_source_bytes <= 0:
            raise ToolError("max_source_bytes must be a positive integer.")
        if max_total_bytes <= 0:
            raise ToolError("max_total_bytes must be a positive integer.")
        if max_chunk_bytes <= 0:
            raise ToolError("max_chunk_bytes must be a positive integer.")
        if max_sources_per_kind <= 0:
            raise ToolError("max_sources_per_kind must be a positive integer.")

        total_bytes = 0
        total_chunks = 0
        source_count = 0
        truncated = False
        errors: list[str] = []
        chunks: list[ToolchainChunk] = []
        compiler_count = 0
        interpreter_count = 0
        transpiled_count = 0

        def handle_sources(kind: str, sources: list[ToolchainSource]) -> None:
            nonlocal total_bytes, total_chunks, source_count, truncated
            nonlocal compiler_count, interpreter_count, transpiled_count

            if not sources or truncated or total_chunks >= max_chunks:
                return
            if len(sources) > max_sources_per_kind:
                sources = sources[:max_sources_per_kind]
                truncated = True

            for index, source in enumerate(sources, start=1):
                if truncated or total_chunks >= max_chunks:
                    break
                label = self._source_label(kind, index, source)
                try:
                    content, size_bytes, source_path = self._load_source(source)
                    if size_bytes > max_source_bytes:
                        raise ToolError(
                            f"{label} exceeds max_source_bytes "
                            f"({size_bytes} > {max_source_bytes})."
                        )
                    if total_bytes + size_bytes > max_total_bytes:
                        truncated = True
                        break

                    total_bytes += size_bytes
                    source_count += 1

                    source_chunks = self._chunk_content(
                        kind,
                        label,
                        source_path,
                        content,
                        unit,
                        size,
                        overlap,
                        max_chunk_bytes,
                    )

                    if not source_chunks:
                        continue

                    remaining = max_chunks - total_chunks
                    if len(source_chunks) > remaining:
                        source_chunks = source_chunks[:remaining]
                        self._finalize_chunks(source_chunks)
                        truncated = True

                    chunks.extend(source_chunks)
                    total_chunks += len(source_chunks)
                    if kind == "compiler":
                        compiler_count += len(source_chunks)
                    elif kind == "interpreter":
                        interpreter_count += len(source_chunks)
                    else:
                        transpiled_count += len(source_chunks)

                except ToolError as exc:
                    errors.append(f"{label}: {exc}")

        handle_sources("compiler", compiler_sources)
        handle_sources("interpreter", interpreter_sources)
        handle_sources("transpiled", transpiled_sources)

        return ChunkToolchainMultiResult(
            chunks=chunks,
            count=len(chunks),
            compiler_count=compiler_count,
            interpreter_count=interpreter_count,
            transpiled_count=transpiled_count,
            source_count=source_count,
            total_bytes=total_bytes,
            truncated=truncated,
            errors=errors,
        )

    def _source_label(
        self, kind: str, index: int, source: ToolchainSource
    ) -> str:
        if source.label and source.label.strip():
            return source.label.strip()
        if source.path and source.path.strip():
            return Path(source.path).name
        return f"{kind}-{index}"

    def _load_source(self, source: ToolchainSource) -> tuple[str, int, str | None]:
        if (source.path is None and source.content is None) or (
            source.path is not None and source.content is not None
        ):
            raise ToolError("Provide either path or content, but not both.")

        if source.path is not None:
            path = self._resolve_path(source.path)
            data = path.read_bytes()
            content = data.decode("utf-8", errors="ignore")
            return content, len(data), str(path)

        content = source.content or ""
        size_bytes = len(content.encode("utf-8"))
        return content, size_bytes, None

    def _resolve_path(self, raw_path: str) -> Path:
        if not raw_path.strip():
            raise ToolError("Path cannot be empty.")

        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.config.effective_workdir / path

        try:
            resolved = path.resolve()
        except ValueError as exc:
            raise ToolError("Security error: cannot resolve the provided path.") from exc

        if not resolved.exists():
            raise ToolError(f"File not found at: {resolved}")
        if resolved.is_dir():
            raise ToolError(f"Path is a directory, not a file: {resolved}")

        return resolved

    def _chunk_content(
        self,
        kind: str,
        label: str,
        source_path: str | None,
        content: str,
        unit: str,
        size: int,
        overlap: int,
        max_chunk_bytes: int,
    ) -> list[ToolchainChunk]:
        if not content:
            return []
        if unit == "lines":
            chunks = self._chunk_lines(
                kind, label, source_path, content, size, overlap, max_chunk_bytes
            )
        else:
            chunks = self._chunk_chars(
                kind, label, source_path, content, size, overlap, max_chunk_bytes
            )
        return self._finalize_chunks(chunks)

    def _chunk_lines(
        self,
        kind: str,
        label: str,
        source_path: str | None,
        content: str,
        size: int,
        overlap: int,
        max_chunk_bytes: int,
    ) -> list[ToolchainChunk]:
        lines = content.splitlines()
        if not lines:
            return []
        step = size - overlap
        index = 0
        chunk_index = 0
        chunks: list[ToolchainChunk] = []
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
                ToolchainChunk(
                    kind=kind,
                    label=label,
                    source_path=source_path,
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
        self,
        kind: str,
        label: str,
        source_path: str | None,
        content: str,
        size: int,
        overlap: int,
        max_chunk_bytes: int,
    ) -> list[ToolchainChunk]:
        step = size - overlap
        index = 0
        chunk_index = 0
        chunks: list[ToolchainChunk] = []
        while index < len(content):
            chunk_text = content[index : index + size]
            if not chunk_text:
                break
            self._check_chunk_bytes(chunk_text, max_chunk_bytes)
            start = index
            end = index + len(chunk_text) - 1
            chunk_index += 1
            chunks.append(
                ToolchainChunk(
                    kind=kind,
                    label=label,
                    source_path=source_path,
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

    def _finalize_chunks(self, chunks: list[ToolchainChunk]) -> list[ToolchainChunk]:
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total
        return chunks

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ChunkToolchainMultiArgs):
            return ToolCallDisplay(summary="chunk_toolchain_multi")

        compiler_count = len(event.args.compiler_sources or [])
        interpreter_count = len(event.args.interpreter_sources or [])
        transpiled_count = len(event.args.transpiled_sources or [])
        summary = (
            "chunk_toolchain_multi: "
            f"{compiler_count} compiler, {interpreter_count} interpreter, "
            f"{transpiled_count} transpiled"
        )
        return ToolCallDisplay(
            summary=summary,
            details={
                "compiler_sources": compiler_count,
                "interpreter_sources": interpreter_count,
                "transpiled_sources": transpiled_count,
                "chunk_unit": event.args.chunk_unit,
                "chunk_size": event.args.chunk_size,
                "overlap": event.args.overlap,
                "max_chunks": event.args.max_chunks,
                "max_source_bytes": event.args.max_source_bytes,
                "max_total_bytes": event.args.max_total_bytes,
                "max_chunk_bytes": event.args.max_chunk_bytes,
                "max_sources_per_kind": event.args.max_sources_per_kind,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ChunkToolchainMultiResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = (
            f"Created {event.result.count} chunk(s) "
            f"({event.result.compiler_count} compiler, "
            f"{event.result.interpreter_count} interpreter, "
            f"{event.result.transpiled_count} transpiled)"
        )
        warnings = event.result.errors[:]
        if event.result.truncated:
            warnings.append("Chunk list truncated by size or limits")

        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=warnings,
            details={
                "count": event.result.count,
                "compiler_count": event.result.compiler_count,
                "interpreter_count": event.result.interpreter_count,
                "transpiled_count": event.result.transpiled_count,
                "source_count": event.result.source_count,
                "total_bytes": event.result.total_bytes,
                "truncated": event.result.truncated,
                "errors": event.result.errors,
                "chunks": event.result.chunks,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Chunking toolchain data"
