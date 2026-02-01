from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class _ScanState:
    in_block_comment: bool = False
    in_string: str | None = None


class ChunkCodeNestedConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_input_bytes: int = Field(
        default=5_000_000, description="Maximum input size in bytes."
    )
    max_chunk_bytes: int = Field(
        default=200_000, description="Maximum chunk size in bytes."
    )
    max_chunks: int = Field(
        default=200, description="Maximum number of chunks to return."
    )
    default_unit: str = Field(default="lines", description="lines or chars.")
    default_chunk_size: int = Field(default=200, description="Chunk size in units.")
    default_split_depth: int = Field(
        default=0, description="Depth at or below which splitting is allowed."
    )
    default_mode: str = Field(default="auto", description="auto, indent, brace.")
    respect_brackets: bool = Field(
        default=True,
        description="Avoid splitting while inside bracketed data structures.",
    )


class ChunkCodeNestedState(BaseToolState):
    pass


class ChunkCodeNestedArgs(BaseModel):
    content: str | None = Field(default=None, description="Raw code to chunk.")
    path: str | None = Field(default=None, description="Path to a code file.")
    language: str | None = Field(default=None, description="Language hint.")
    mode: str | None = Field(default=None, description="auto, indent, or brace.")
    unit: str | None = Field(default=None, description="lines or chars.")
    size: int | None = Field(default=None, description="Chunk size in units.")
    split_depth: int | None = Field(
        default=None, description="Depth at or below which splitting is allowed."
    )
    respect_brackets: bool | None = Field(
        default=None, description="Respect nested bracket structures."
    )
    hard_split: bool = Field(
        default=True,
        description="Allow splitting even if no safe boundary exists.",
    )
    max_chunks: int | None = Field(
        default=None, description="Override the configured max chunks limit."
    )


class CodeChunk(BaseModel):
    index: int
    start_line: int
    end_line: int
    min_depth: int
    max_depth: int
    unit: str
    content: str


class ChunkCodeNestedResult(BaseModel):
    chunks: list[CodeChunk]
    count: int
    truncated: bool
    mode: str
    unit: str


class ChunkCodeNested(
    BaseTool[
        ChunkCodeNestedArgs,
        ChunkCodeNestedResult,
        ChunkCodeNestedConfig,
        ChunkCodeNestedState,
    ],
    ToolUIData[ChunkCodeNestedArgs, ChunkCodeNestedResult],
):
    description: ClassVar[str] = (
        "Chunk code while respecting nested blocks and data structures."
    )

    async def run(self, args: ChunkCodeNestedArgs) -> ChunkCodeNestedResult:
        content = self._load_content(args)
        if not content:
            return ChunkCodeNestedResult(
                chunks=[],
                count=0,
                truncated=False,
                mode="auto",
                unit=self.config.default_unit,
            )

        mode = self._normalize_mode(args)
        unit = self._normalize_unit(args)
        size = args.size if args.size is not None else self.config.default_chunk_size
        split_depth = (
            args.split_depth
            if args.split_depth is not None
            else self.config.default_split_depth
        )
        if size <= 0:
            raise ToolError("size must be a positive integer.")
        if split_depth < 0:
            raise ToolError("split_depth must be >= 0.")

        max_chunks = args.max_chunks if args.max_chunks is not None else self.config.max_chunks
        if max_chunks <= 0:
            raise ToolError("max_chunks must be a positive integer.")

        respect_brackets = (
            args.respect_brackets
            if args.respect_brackets is not None
            else self.config.respect_brackets
        )

        if unit == "chars" and size > self.config.max_chunk_bytes:
            raise ToolError("size exceeds max_chunk_bytes for char-based chunking.")

        lines = content.splitlines()
        boundaries, depths = self._compute_boundaries(
            lines, mode, split_depth, respect_brackets
        )

        chunks, truncated = self._build_chunks(
            lines,
            boundaries,
            depths,
            unit,
            size,
            max_chunks,
            args.hard_split,
        )

        return ChunkCodeNestedResult(
            chunks=chunks,
            count=len(chunks),
            truncated=truncated,
            mode=mode,
            unit=unit,
        )

    def _load_content(self, args: ChunkCodeNestedArgs) -> str:
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
                f"Input is {size} bytes, which exceeds max_input_bytes "
                f"({self.config.max_input_bytes})."
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

    def _normalize_mode(self, args: ChunkCodeNestedArgs) -> str:
        mode = (args.mode or self.config.default_mode).strip().lower()
        if mode not in {"auto", "indent", "brace"}:
            raise ToolError("mode must be auto, indent, or brace.")
        if mode == "auto":
            language = (args.language or "").strip().lower()
            if language in {"python", "py"}:
                return "indent"
            if args.path:
                ext = Path(args.path).suffix.lower()
                if ext == ".py":
                    return "indent"
            return "brace"
        return mode

    def _normalize_unit(self, args: ChunkCodeNestedArgs) -> str:
        unit = (args.unit or self.config.default_unit).strip().lower()
        if unit not in {"lines", "chars"}:
            raise ToolError("unit must be lines or chars.")
        return unit

    def _compute_boundaries(
        self,
        lines: list[str],
        mode: str,
        split_depth: int,
        respect_brackets: bool,
    ) -> tuple[list[bool], list[int]]:
        if mode == "indent":
            return self._compute_indent_boundaries(lines, split_depth, respect_brackets)
        return self._compute_brace_boundaries(lines, split_depth, respect_brackets)

    def _compute_brace_boundaries(
        self, lines: list[str], split_depth: int, respect_brackets: bool
    ) -> tuple[list[bool], list[int]]:
        boundaries: list[bool] = []
        depths: list[int] = []
        block_depth = 0
        bracket_depth = 0
        state = _ScanState()

        for line in lines:
            sanitized, state = self._scan_line(line, state, hash_comments=False)
            for ch in sanitized:
                if ch == "{":
                    block_depth += 1
                elif ch == "}":
                    block_depth = max(block_depth - 1, 0)
                elif ch == "[":
                    bracket_depth += 1
                elif ch == "]":
                    bracket_depth = max(bracket_depth - 1, 0)

            safe = (
                block_depth <= split_depth
                and (not respect_brackets or bracket_depth == 0)
                and not state.in_string
                and not state.in_block_comment
            )
            boundaries.append(safe)
            depths.append(block_depth)

        return boundaries, depths

    def _compute_indent_boundaries(
        self, lines: list[str], split_depth: int, respect_brackets: bool
    ) -> tuple[list[bool], list[int]]:
        boundaries: list[bool] = []
        depths: list[int] = []
        indent_stack: list[int] = []
        bracket_depth = 0
        state = _ScanState()

        for line in lines:
            sanitized, state = self._scan_line(line, state, hash_comments=True)
            stripped = sanitized.strip()
            if stripped:
                if bracket_depth == 0:
                    indent = self._count_indent(line)
                    current = indent_stack[-1] if indent_stack else 0
                    if indent > current:
                        indent_stack.append(indent)
                    elif indent < current:
                        while indent_stack and indent_stack[-1] > indent:
                            indent_stack.pop()
            for ch in sanitized:
                if ch in "{[":
                    bracket_depth += 1
                elif ch in "]}":
                    bracket_depth = max(bracket_depth - 1, 0)
                elif ch == "(":
                    bracket_depth += 1
                elif ch == ")":
                    bracket_depth = max(bracket_depth - 1, 0)

            depth = len(indent_stack)
            safe = (
                depth <= split_depth
                and (not respect_brackets or bracket_depth == 0)
                and not state.in_string
                and not state.in_block_comment
            )
            boundaries.append(safe)
            depths.append(depth)

        return boundaries, depths

    def _scan_line(
        self, line: str, state: _ScanState, *, hash_comments: bool
    ) -> tuple[str, _ScanState]:
        result: list[str] = []
        i = 0
        length = len(line)
        while i < length:
            ch = line[i]
            next_two = line[i : i + 2]

            if state.in_block_comment:
                if next_two == "*/":
                    state.in_block_comment = False
                    i += 2
                    continue
                i += 1
                continue

            if state.in_string:
                if ch == "\\":
                    i += 2
                    continue
                if ch == state.in_string:
                    state.in_string = None
                i += 1
                continue

            if next_two == "/*":
                state.in_block_comment = True
                i += 2
                continue
            if next_two == "//":
                break
            if hash_comments and ch == "#":
                break
            if ch in {"'", '"', "`"}:
                state.in_string = ch
                i += 1
                continue

            result.append(ch)
            i += 1

        return "".join(result), state

    def _count_indent(self, line: str) -> int:
        count = 0
        for ch in line:
            if ch == " ":
                count += 1
            elif ch == "\t":
                count += 4
            else:
                break
        return count

    def _build_chunks(
        self,
        lines: list[str],
        boundaries: list[bool],
        depths: list[int],
        unit: str,
        size: int,
        max_chunks: int,
        hard_split: bool,
    ) -> tuple[list[CodeChunk], bool]:
        chunks: list[CodeChunk] = []
        truncated = False
        buffer_lines: list[str] = []
        buffer_sizes: list[int] = []
        buffer_depths: list[int] = []
        start_line = 1
        last_safe_index: int | None = None

        def current_count() -> int:
            if unit == "lines":
                return len(buffer_lines)
            return sum(buffer_sizes)

        def flush(end_index: int) -> None:
            nonlocal start_line
            if end_index <= 0:
                return
            chunk_lines = buffer_lines[:end_index]
            chunk_depths = buffer_depths[:end_index]
            content = "\n".join(chunk_lines)
            content_bytes = len(content.encode("utf-8"))
            if content_bytes > self.config.max_chunk_bytes:
                raise ToolError(
                    f"Chunk exceeds max_chunk_bytes ({content_bytes} > {self.config.max_chunk_bytes})."
                )
            chunk = CodeChunk(
                index=len(chunks) + 1,
                start_line=start_line,
                end_line=start_line + len(chunk_lines) - 1,
                min_depth=min(chunk_depths) if chunk_depths else 0,
                max_depth=max(chunk_depths) if chunk_depths else 0,
                unit=unit,
                content=content,
            )
            chunks.append(chunk)
            del buffer_lines[:end_index]
            del buffer_sizes[:end_index]
            del buffer_depths[:end_index]
            start_line = chunk.end_line + 1

        for idx, line in enumerate(lines):
            buffer_lines.append(line)
            buffer_sizes.append(len(line) + 1)
            depth = depths[idx] if idx < len(depths) else 0
            buffer_depths.append(depth)

            if boundaries[idx]:
                last_safe_index = len(buffer_lines)

            if current_count() >= size:
                if last_safe_index is not None:
                    flush(last_safe_index)
                    last_safe_index = None
                elif hard_split:
                    flush(len(buffer_lines))
                else:
                    continue

                if len(chunks) >= max_chunks:
                    truncated = True
                    break

        if not truncated and buffer_lines:
            flush(len(buffer_lines))

        if len(chunks) > max_chunks:
            chunks = chunks[:max_chunks]
            truncated = True

        return chunks, truncated

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ChunkCodeNestedArgs):
            return ToolCallDisplay(summary="chunk_code_nested")

        summary = f"chunk_code_nested: {event.args.mode or 'auto'}"
        return ToolCallDisplay(
            summary=summary,
            details={
                "mode": event.args.mode,
                "unit": event.args.unit,
                "size": event.args.size,
                "split_depth": event.args.split_depth,
                "respect_brackets": event.args.respect_brackets,
                "hard_split": event.args.hard_split,
                "max_chunks": event.args.max_chunks,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ChunkCodeNestedResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = f"Created {event.result.count} chunk(s)"
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
                "chunks": event.result.chunks,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Chunking nested code"
