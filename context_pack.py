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


class ContextPackConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_source_bytes: int = Field(
        default=1_000_000, description="Maximum size per source (bytes)."
    )
    max_total_bytes: int = Field(
        default=3_000_000, description="Maximum total size across sources (bytes)."
    )
    max_chunk_bytes: int = Field(
        default=1_000_000, description="Maximum size per chunk (bytes)."
    )
    max_chunks: int = Field(
        default=200, description="Maximum number of chunks to return."
    )


class ContextPackState(BaseToolState):
    pass


class ContextSource(BaseModel):
    id: str | None = Field(default=None, description="Optional source identifier.")
    label: str | None = Field(default=None, description="Optional label for the source.")
    language: str | None = Field(default=None, description="Language hint.")
    content: str | None = Field(default=None, description="Inline content.")
    path: str | None = Field(default=None, description="Path to a file.")
    start_line: int | None = Field(
        default=None, description="1-based start line (inclusive)."
    )
    end_line: int | None = Field(
        default=None, description="1-based end line (inclusive)."
    )


class ContextPackArgs(BaseModel):
    sources: list[ContextSource]
    chunk_unit: str = Field(
        default="none", description="none, lines, or chars."
    )
    chunk_size: int | None = Field(
        default=None, description="Chunk size in units for lines/chars."
    )
    overlap: int = Field(
        default=0, description="Overlap in units for lines/chars."
    )
    max_chunks: int | None = Field(
        default=None, description="Override the configured max chunks limit."
    )


class ContextChunk(BaseModel):
    source_id: str
    label: str | None
    path: str | None
    language: str | None
    chunk_index: int
    total_chunks: int
    unit: str
    start: int | None
    end: int | None
    content: str


class ContextPackResult(BaseModel):
    chunks: list[ContextChunk]
    count: int
    source_count: int
    truncated: bool
    errors: list[str]


class ContextPack(
    BaseTool[ContextPackArgs, ContextPackResult, ContextPackConfig, ContextPackState],
    ToolUIData[ContextPackArgs, ContextPackResult],
):
    description: ClassVar[str] = (
        "Bundle multiple sources into chunked context blocks for reasoning."
    )

    async def run(self, args: ContextPackArgs) -> ContextPackResult:
        if not args.sources:
            raise ToolError("At least one source is required.")

        unit = args.chunk_unit.strip().lower()
        if unit not in {"none", "lines", "chars"}:
            raise ToolError("chunk_unit must be one of: none, lines, chars.")

        if unit == "none":
            if args.chunk_size is not None:
                raise ToolError("chunk_size is not used when chunk_unit is none.")
            if args.overlap:
                raise ToolError("overlap is not used when chunk_unit is none.")
        else:
            size = self._require_positive(args.chunk_size, "chunk_size")
            overlap = self._require_non_negative(args.overlap, "overlap")
            if overlap >= size:
                raise ToolError("overlap must be smaller than chunk_size.")

        max_chunks = args.max_chunks if args.max_chunks is not None else self.config.max_chunks
        if max_chunks <= 0:
            raise ToolError("max_chunks must be a positive integer.")

        chunks: list[ContextChunk] = []
        errors: list[str] = []
        truncated = False
        total_bytes = 0
        source_count = 0

        for index, source in enumerate(args.sources, start=1):
            try:
                source_id, label, language, path, content = self._load_source(source, index)
                content, base_line = self._slice_lines(
                    content, source.start_line, source.end_line
                )
                content_bytes = len(content.encode("utf-8"))

                if content_bytes > self.config.max_source_bytes:
                    raise ToolError(
                        f"Source '{source_id}' exceeds max_source_bytes "
                        f"({content_bytes} > {self.config.max_source_bytes})."
                    )

                if total_bytes + content_bytes > self.config.max_total_bytes:
                    truncated = True
                    break

                total_bytes += content_bytes

                source_chunks = self._chunk_content(
                    content,
                    unit,
                    args.chunk_size,
                    args.overlap,
                    base_line,
                    source_id,
                    label,
                    path,
                    language,
                )

                if len(chunks) + len(source_chunks) > max_chunks:
                    remaining = max_chunks - len(chunks)
                    if remaining > 0:
                        chunks.extend(source_chunks[:remaining])
                    truncated = True
                    break

                chunks.extend(source_chunks)
                source_count += 1
            except ToolError as exc:
                errors.append(str(exc))

        return ContextPackResult(
            chunks=chunks,
            count=len(chunks),
            source_count=source_count,
            truncated=truncated,
            errors=errors,
        )

    def _load_source(
        self, source: ContextSource, index: int
    ) -> tuple[str, str | None, str | None, str | None, str]:
        if source.content and source.path:
            raise ToolError("Provide content or path per source, not both.")
        if source.content is None and source.path is None:
            raise ToolError("Each source must provide content or path.")

        source_id = source.id or source.label or source.path or f"source-{index}"
        label = source.label
        language = source.language.strip().lower() if source.language else None

        if source.content is not None:
            return source_id, label, language, None, source.content

        path = self._resolve_path(source.path or "")
        content = path.read_text("utf-8", errors="ignore")
        return source_id, label, language, str(path), content

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

    def _slice_lines(
        self, content: str, start_line: int | None, end_line: int | None
    ) -> tuple[str, int]:
        if start_line is None and end_line is None:
            return content, 1

        if start_line is not None and start_line <= 0:
            raise ToolError("start_line must be a positive integer.")
        if end_line is not None and end_line <= 0:
            raise ToolError("end_line must be a positive integer.")

        lines = content.splitlines()
        start_index = (start_line - 1) if start_line is not None else 0
        end_index = end_line if end_line is not None else len(lines)

        if start_index >= len(lines):
            raise ToolError("start_line is beyond the end of the content.")
        if end_index < start_index:
            raise ToolError("end_line must be greater than or equal to start_line.")

        sliced = lines[start_index:end_index]
        if not sliced:
            raise ToolError("Selected line range is empty.")

        return "\n".join(sliced), start_index + 1

    def _chunk_content(
        self,
        content: str,
        unit: str,
        chunk_size: int | None,
        overlap: int,
        base_line: int,
        source_id: str,
        label: str | None,
        path: str | None,
        language: str | None,
    ) -> list[ContextChunk]:
        if unit == "none":
            self._check_chunk_bytes(content, source_id)
            return [
                ContextChunk(
                    source_id=source_id,
                    label=label,
                    path=path,
                    language=language,
                    chunk_index=1,
                    total_chunks=1,
                    unit=unit,
                    start=None,
                    end=None,
                    content=content,
                )
            ]

        size = self._require_positive(chunk_size, "chunk_size")
        if unit == "lines":
            return self._chunk_lines(
                content,
                size,
                overlap,
                base_line,
                source_id,
                label,
                path,
                language,
            )
        return self._chunk_chars(
            content,
            size,
            overlap,
            source_id,
            label,
            path,
            language,
        )

    def _chunk_lines(
        self,
        content: str,
        size: int,
        overlap: int,
        base_line: int,
        source_id: str,
        label: str | None,
        path: str | None,
        language: str | None,
    ) -> list[ContextChunk]:
        lines = content.splitlines()
        step = size - overlap
        chunks: list[ContextChunk] = []
        index = 0
        chunk_index = 0
        while index < len(lines):
            subset = lines[index : index + size]
            if not subset:
                break
            chunk_text = "\n".join(subset)
            self._check_chunk_bytes(chunk_text, source_id)
            start_line = base_line + index
            end_line = start_line + len(subset) - 1
            chunk_index += 1
            chunks.append(
                ContextChunk(
                    source_id=source_id,
                    label=label,
                    path=path,
                    language=language,
                    chunk_index=chunk_index,
                    total_chunks=0,
                    unit="lines",
                    start=start_line,
                    end=end_line,
                    content=chunk_text,
                )
            )
            index += step

        return self._finalize_chunks(chunks)

    def _chunk_chars(
        self,
        content: str,
        size: int,
        overlap: int,
        source_id: str,
        label: str | None,
        path: str | None,
        language: str | None,
    ) -> list[ContextChunk]:
        step = size - overlap
        chunks: list[ContextChunk] = []
        index = 0
        chunk_index = 0
        length = len(content)
        while index < length:
            chunk_text = content[index : index + size]
            if not chunk_text:
                break
            self._check_chunk_bytes(chunk_text, source_id)
            chunk_index += 1
            start = index
            end = index + len(chunk_text) - 1
            chunks.append(
                ContextChunk(
                    source_id=source_id,
                    label=label,
                    path=path,
                    language=language,
                    chunk_index=chunk_index,
                    total_chunks=0,
                    unit="chars",
                    start=start,
                    end=end,
                    content=chunk_text,
                )
            )
            index += step

        return self._finalize_chunks(chunks)

    def _finalize_chunks(self, chunks: list[ContextChunk]) -> list[ContextChunk]:
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total
        return chunks

    def _check_chunk_bytes(self, content: str, source_id: str) -> None:
        size = len(content.encode("utf-8"))
        if size > self.config.max_chunk_bytes:
            raise ToolError(
                f"Chunk from '{source_id}' exceeds max_chunk_bytes "
                f"({size} > {self.config.max_chunk_bytes})."
            )

    def _require_positive(self, value: int | None, name: str) -> int:
        if value is None:
            raise ToolError(f"{name} is required for this mode.")
        if value <= 0:
            raise ToolError(f"{name} must be a positive integer.")
        return value

    def _require_non_negative(self, value: int, name: str) -> int:
        if value < 0:
            raise ToolError(f"{name} must be a non-negative integer.")
        return value

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextPackArgs):
            return ToolCallDisplay(summary="context_pack")

        summary = f"context_pack: {len(event.args.sources)} source(s)"
        return ToolCallDisplay(
            summary=summary,
            details={
                "sources": len(event.args.sources),
                "chunk_unit": event.args.chunk_unit,
                "chunk_size": event.args.chunk_size,
                "overlap": event.args.overlap,
                "max_chunks": event.args.max_chunks,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextPackResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = f"Prepared {event.result.count} chunk(s)"
        if event.result.truncated:
            message += " (truncated)"

        warnings = event.result.errors[:]
        if event.result.truncated:
            warnings.append("Chunk list truncated by size or max_chunks limits")

        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=warnings,
            details={
                "count": event.result.count,
                "source_count": event.result.source_count,
                "truncated": event.result.truncated,
                "errors": event.result.errors,
                "chunks": event.result.chunks,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Packing context"
