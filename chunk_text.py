from __future__ import annotations

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


SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class ChunkTextConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_input_bytes: int = Field(
        default=1_000_000,
        description="Maximum bytes allowed for input content.",
    )
    max_chunks: int = Field(
        default=200,
        description="Maximum number of chunks to return.",
    )


class ChunkTextState(BaseToolState):
    pass


class ChunkTextArgs(BaseModel):
    content: str | None = Field(default=None, description="Raw text to chunk.")
    path: str | None = Field(default=None, description="Path to a text file to chunk.")
    mode: str = Field(
        default="fixed",
        description="Chunking mode: fixed, sliding, variable, paragraph, sentence, regex.",
    )
    unit: str = Field(
        default="chars",
        description="Unit for size and overlap: chars or lines.",
    )
    size: int | None = Field(
        default=None,
        description="Chunk size in units for fixed or sliding modes.",
    )
    sizes: list[int] | None = Field(
        default=None,
        description="List of sizes for variable mode.",
    )
    overlap: int = Field(
        default=0,
        description="Overlap in units for sliding mode.",
    )
    separator: str | None = Field(
        default=None,
        description="Regex pattern for regex mode.",
    )
    include_separator: bool = Field(
        default=False,
        description="Include separators at the start of chunks for regex mode.",
    )
    repeat_sizes: bool = Field(
        default=True,
        description="Repeat the sizes list when input exceeds it (variable mode).",
    )
    max_chunks: int | None = Field(
        default=None,
        description="Override the configured max chunks limit.",
    )


class ChunkTextResult(BaseModel):
    chunks: list[str]
    count: int
    truncated: bool
    mode: str
    unit: str
    chunk_sizes: list[int]
    input_bytes: int


class ChunkText(
    BaseTool[ChunkTextArgs, ChunkTextResult, ChunkTextConfig, ChunkTextState],
    ToolUIData[ChunkTextArgs, ChunkTextResult],
):
    description: ClassVar[str] = (
        "Chunk text into different shapes and sizes using multiple strategies."
    )

    async def run(self, args: ChunkTextArgs) -> ChunkTextResult:
        content = self._load_content(args)
        mode = args.mode.strip().lower()
        unit = args.unit.strip().lower()

        if not content:
            return ChunkTextResult(
                chunks=[],
                count=0,
                truncated=False,
                mode=mode,
                unit=unit,
                chunk_sizes=[],
                input_bytes=0,
            )

        if mode in {"fixed", "sliding", "variable"}:
            if unit not in {"chars", "lines"}:
                raise ToolError("Unit must be 'chars' or 'lines' for this mode.")

            if mode == "fixed":
                size = self._require_positive(args.size, "size")
                chunks_with_sizes = self._chunk_fixed(content, unit, size)
            elif mode == "sliding":
                size = self._require_positive(args.size, "size")
                overlap = self._require_non_negative(args.overlap, "overlap")
                if overlap >= size:
                    raise ToolError("Overlap must be smaller than size.")
                chunks_with_sizes = self._chunk_sliding(content, unit, size, overlap)
            else:
                sizes = self._require_sizes(args.sizes)
                chunks_with_sizes = self._chunk_variable(
                    content, unit, sizes, args.repeat_sizes
                )

        elif mode == "paragraph":
            chunks_with_sizes = self._chunk_paragraphs(content, args.size)
        elif mode == "sentence":
            chunks_with_sizes = self._chunk_sentences(content, args.size)
        elif mode == "regex":
            pattern = (args.separator or "").strip()
            if not pattern:
                raise ToolError("separator is required for regex mode.")
            chunks_with_sizes = self._chunk_regex(
                content, pattern, args.include_separator, args.size
            )
        else:
            raise ToolError(
                "Mode must be one of: fixed, sliding, variable, paragraph, sentence, regex."
            )

        max_chunks = args.max_chunks if args.max_chunks is not None else self.config.max_chunks
        if max_chunks <= 0:
            raise ToolError("max_chunks must be a positive integer.")

        truncated = len(chunks_with_sizes) > max_chunks
        if truncated:
            chunks_with_sizes = chunks_with_sizes[:max_chunks]

        chunks = [chunk for chunk, _ in chunks_with_sizes]
        chunk_sizes = [size for _, size in chunks_with_sizes]

        return ChunkTextResult(
            chunks=chunks,
            count=len(chunks),
            truncated=truncated,
            mode=mode,
            unit=unit,
            chunk_sizes=chunk_sizes,
            input_bytes=len(content.encode("utf-8")),
        )

    def _load_content(self, args: ChunkTextArgs) -> str:
        if args.content and args.path:
            raise ToolError("Provide either content or path, not both.")
        if args.content is None and args.path is None:
            raise ToolError("Provide content or path.")

        if args.content is not None:
            data = args.content.encode("utf-8")
            self._validate_input_size(len(data))
            return args.content

        path = self._resolve_path(args.path or "")
        size = path.stat().st_size
        self._validate_input_size(size)
        return path.read_text("utf-8", errors="ignore")

    def _validate_input_size(self, size: int) -> None:
        if size > self.config.max_input_bytes:
            raise ToolError(
                f"Input is {size} bytes, which exceeds the limit of {self.config.max_input_bytes} bytes."
            )

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

    def _require_sizes(self, sizes: list[int] | None) -> list[int]:
        if not sizes:
            raise ToolError("sizes must be provided for variable mode.")
        cleaned: list[int] = []
        for size in sizes:
            if size <= 0:
                raise ToolError("All sizes must be positive integers.")
            cleaned.append(size)
        return cleaned

    def _chunk_fixed(self, content: str, unit: str, size: int) -> list[tuple[str, int]]:
        if unit == "chars":
            return [
                (content[start : start + size], len(content[start : start + size]))
                for start in range(0, len(content), size)
            ]

        lines = content.splitlines()
        chunks: list[tuple[str, int]] = []
        for start in range(0, len(lines), size):
            subset = lines[start : start + size]
            if subset:
                chunks.append(("\n".join(subset), len(subset)))
        return chunks

    def _chunk_sliding(
        self, content: str, unit: str, size: int, overlap: int
    ) -> list[tuple[str, int]]:
        step = size - overlap
        if unit == "chars":
            chunks: list[tuple[str, int]] = []
            start = 0
            while start < len(content):
                chunk = content[start : start + size]
                chunks.append((chunk, len(chunk)))
                start += step
            return chunks

        lines = content.splitlines()
        chunks = []
        start = 0
        while start < len(lines):
            subset = lines[start : start + size]
            if subset:
                chunks.append(("\n".join(subset), len(subset)))
            start += step
        return chunks

    def _chunk_variable(
        self, content: str, unit: str, sizes: list[int], repeat_sizes: bool
    ) -> list[tuple[str, int]]:
        if unit == "chars":
            chunks: list[tuple[str, int]] = []
            index = 0
            pos = 0
            while pos < len(content):
                size = sizes[index]
                chunk = content[pos : pos + size]
                chunks.append((chunk, len(chunk)))
                pos += size
                index += 1
                if index >= len(sizes):
                    if repeat_sizes:
                        index = 0
                    else:
                        if pos < len(content):
                            chunk = content[pos:]
                            chunks.append((chunk, len(chunk)))
                        break
            return chunks

        lines = content.splitlines()
        chunks = []
        index = 0
        pos = 0
        while pos < len(lines):
            size = sizes[index]
            subset = lines[pos : pos + size]
            if subset:
                chunks.append(("\n".join(subset), len(subset)))
            pos += size
            index += 1
            if index >= len(sizes):
                if repeat_sizes:
                    index = 0
                else:
                    if pos < len(lines):
                        subset = lines[pos:]
                        if subset:
                            chunks.append(("\n".join(subset), len(subset)))
                    break
        return chunks

    def _chunk_paragraphs(self, content: str, size: int | None) -> list[tuple[str, int]]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", content) if p.strip()]
        return self._group_items(paragraphs, size, "\n\n")

    def _chunk_sentences(self, content: str, size: int | None) -> list[tuple[str, int]]:
        sentences = [s.strip() for s in SENTENCE_SPLIT.split(content) if s.strip()]
        return self._group_items(sentences, size, " ")

    def _chunk_regex(
        self, content: str, pattern: str, include_separator: bool, size: int | None
    ) -> list[tuple[str, int]]:
        regex = re.compile(pattern, re.MULTILINE)
        matches = list(regex.finditer(content))
        for match in matches:
            if match.start() == match.end():
                raise ToolError("Separator pattern must not be zero-length.")

        if not matches:
            return self._group_items([content], size, "")

        if include_separator:
            boundaries = [0]
            for match in matches:
                if match.start() > 0 and match.start() != boundaries[-1]:
                    boundaries.append(match.start())
            if boundaries[-1] != len(content):
                boundaries.append(len(content))

            segments = [
                content[boundaries[i] : boundaries[i + 1]]
                for i in range(len(boundaries) - 1)
                if boundaries[i] < boundaries[i + 1]
            ]
        else:
            segments = []
            pos = 0
            for match in matches:
                if pos < match.start():
                    segments.append(content[pos : match.start()])
                pos = match.end()
            if pos < len(content):
                segments.append(content[pos:])

        segments = [seg for seg in segments if seg.strip()]
        return self._group_items(segments, size, "")

    def _group_items(
        self, items: list[str], size: int | None, joiner: str
    ) -> list[tuple[str, int]]:
        if not items:
            return []

        if size is None:
            return [(item, 1) for item in items]

        if size <= 0:
            raise ToolError("size must be a positive integer.")

        chunks: list[tuple[str, int]] = []
        for start in range(0, len(items), size):
            subset = items[start : start + size]
            chunks.append((joiner.join(subset), len(subset)))
        return chunks

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ChunkTextArgs):
            return ToolCallDisplay(summary="chunk_text")

        summary = f"chunk_text: {event.args.mode}"
        details = {
            "mode": event.args.mode,
            "unit": event.args.unit,
            "size": event.args.size,
            "sizes": event.args.sizes,
            "overlap": event.args.overlap,
            "separator": event.args.separator,
            "include_separator": event.args.include_separator,
            "repeat_sizes": event.args.repeat_sizes,
            "max_chunks": event.args.max_chunks,
        }
        return ToolCallDisplay(summary=summary, details=details)

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ChunkTextResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = f"Created {event.result.count} chunks"
        if event.result.truncated:
            message += " (truncated)"

        warnings = []
        if event.result.truncated:
            warnings.append("Chunk list truncated by max_chunks limit")

        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=warnings,
            details={
                "count": event.result.count,
                "truncated": event.result.truncated,
                "mode": event.result.mode,
                "unit": event.result.unit,
                "chunk_sizes": event.result.chunk_sizes,
                "input_bytes": event.result.input_bytes,
                "chunks": event.result.chunks,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Chunking text"
