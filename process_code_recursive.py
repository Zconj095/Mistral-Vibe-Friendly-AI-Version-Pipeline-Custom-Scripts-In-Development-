from __future__ import annotations

from dataclasses import dataclass
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


INDENT_EXTS = {".py", ".pyi", ".yml", ".yaml"}
INDENT_LANGS = {"python", "py", "yaml", "yml"}


@dataclass
class _ScanState:
    in_block_comment: bool = False
    in_string: str | None = None


class ProcessCodeRecursiveConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_input_bytes: int = Field(
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
    default_split_depth: int = Field(
        default=0, description="Depth at or below which splitting is allowed."
    )
    default_code_mode: str = Field(default="auto", description="auto, indent, brace.")
    default_process_mode: str = Field(default="chunk", description="chunk or read.")
    respect_brackets: bool = Field(
        default=True, description="Avoid splitting inside bracketed structures."
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


class ProcessCodeRecursiveState(BaseToolState):
    pass


class ProcessCodeRecursiveArgs(BaseModel):
    paths: list[str] = Field(description="Root directories or files to scan.")
    extension: str | None = Field(
        default=None, description="Single extension to match (e.g. .py)."
    )
    extensions: list[str] | None = Field(
        default=None, description="Extensions to match (e.g. ['.py', '.ts'])."
    )
    include_globs: list[str] | None = Field(
        default=None, description="Optional include globs."
    )
    exclude_globs: list[str] | None = Field(
        default=None, description="Optional exclude globs."
    )
    mode: str | None = Field(default=None, description="chunk or read.")
    code_mode: str | None = Field(
        default=None, description="auto, indent, or brace."
    )
    chunk_unit: str | None = Field(default=None, description="lines or chars.")
    chunk_size: int | None = Field(default=None, description="Chunk size in units.")
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
        default=None, description="Override max chunks limit."
    )
    max_files: int | None = Field(default=None, description="Override max files limit.")
    max_input_bytes: int | None = Field(
        default=None, description="Override max_input_bytes."
    )
    max_total_bytes: int | None = Field(
        default=None, description="Override max_total_bytes."
    )
    max_chunk_bytes: int | None = Field(
        default=None, description="Override max_chunk_bytes."
    )


class CodeDocumentEntry(BaseModel):
    path: str
    language: str | None
    bytes_total: int
    content: str


class CodeChunkData(BaseModel):
    index: int
    total_chunks: int
    start_line: int
    end_line: int
    min_depth: int
    max_depth: int
    unit: str
    content: str


class CodeChunk(BaseModel):
    path: str
    language: str | None
    mode: str
    chunk_index: int
    total_chunks: int
    start_line: int
    end_line: int
    min_depth: int
    max_depth: int
    unit: str
    content: str


class ProcessCodeRecursiveResult(BaseModel):
    mode: str
    documents: list[CodeDocumentEntry]
    chunks: list[CodeChunk]
    count: int
    document_count: int
    chunk_count: int
    total_bytes: int
    truncated: bool
    errors: list[str]


class ProcessCodeRecursive(
    BaseTool[
        ProcessCodeRecursiveArgs,
        ProcessCodeRecursiveResult,
        ProcessCodeRecursiveConfig,
        ProcessCodeRecursiveState,
    ],
    ToolUIData[ProcessCodeRecursiveArgs, ProcessCodeRecursiveResult],
):
    description: ClassVar[str] = (
        "Recursively find code files by extension and read or chunk them."
    )

    async def run(
        self, args: ProcessCodeRecursiveArgs
    ) -> ProcessCodeRecursiveResult:
        if not args.paths:
            raise ToolError("At least one path is required.")

        process_mode = (args.mode or self.config.default_process_mode).strip().lower()
        if process_mode not in {"chunk", "read"}:
            raise ToolError("mode must be chunk or read.")

        max_files = args.max_files if args.max_files is not None else self.config.max_files
        if max_files <= 0:
            raise ToolError("max_files must be a positive integer.")

        max_chunks = args.max_chunks if args.max_chunks is not None else self.config.max_chunks
        if max_chunks <= 0:
            raise ToolError("max_chunks must be a positive integer.")

        max_input_bytes = (
            args.max_input_bytes
            if args.max_input_bytes is not None
            else self.config.max_input_bytes
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
        if max_input_bytes <= 0:
            raise ToolError("max_input_bytes must be a positive integer.")
        if max_total_bytes <= 0:
            raise ToolError("max_total_bytes must be a positive integer.")
        if max_chunk_bytes <= 0:
            raise ToolError("max_chunk_bytes must be a positive integer.")

        unit = (args.chunk_unit or self.config.default_unit).strip().lower()
        size = args.chunk_size if args.chunk_size is not None else self.config.default_chunk_size
        split_depth = (
            args.split_depth
            if args.split_depth is not None
            else self.config.default_split_depth
        )
        if process_mode == "chunk":
            if unit not in {"lines", "chars"}:
                raise ToolError("chunk_unit must be lines or chars.")
            if size <= 0:
                raise ToolError("chunk_size must be a positive integer.")
            if unit == "chars" and size > max_chunk_bytes:
                raise ToolError("chunk_size exceeds max_chunk_bytes for char chunking.")
            if split_depth < 0:
                raise ToolError("split_depth must be >= 0.")

        code_mode = (args.code_mode or self.config.default_code_mode).strip().lower()
        if code_mode not in {"auto", "indent", "brace"}:
            raise ToolError("code_mode must be auto, indent, or brace.")

        respect_brackets = (
            args.respect_brackets
            if args.respect_brackets is not None
            else self.config.respect_brackets
        )

        include_globs = self._resolve_include_globs(args)
        exclude_globs = self._normalize_globs(
            args.exclude_globs, self.config.default_exclude_globs
        )

        files = self._gather_files(args.paths, include_globs, exclude_globs, max_files)
        if not files:
            raise ToolError("No files matched the provided paths/patterns.")

        documents: list[CodeDocumentEntry] = []
        chunks: list[CodeChunk] = []
        errors: list[str] = []
        total_bytes = 0
        truncated = False
        document_count = 0

        for file_path in files:
            if truncated:
                break
            if process_mode == "chunk" and len(chunks) >= max_chunks:
                truncated = True
                break

            try:
                size_bytes = file_path.stat().st_size
                if size_bytes > max_input_bytes:
                    raise ToolError(
                        f"{file_path} exceeds max_input_bytes ({size_bytes} > {max_input_bytes})."
                    )
                if total_bytes + size_bytes > max_total_bytes:
                    truncated = True
                    break

                content = file_path.read_text("utf-8", errors="ignore")
                total_bytes += size_bytes
                document_count += 1
                language = self._resolve_language(file_path)

                if process_mode == "read":
                    documents.append(
                        CodeDocumentEntry(
                            path=str(file_path),
                            language=language,
                            bytes_total=size_bytes,
                            content=content,
                        )
                    )
                    continue

                resolved_mode = self._resolve_mode(code_mode, language, file_path)
                lines = content.splitlines()
                boundaries, depths = self._compute_boundaries(
                    lines, resolved_mode, split_depth, respect_brackets
                )

                remaining = max_chunks - len(chunks)
                file_chunks, file_truncated = self._build_chunks(
                    lines,
                    boundaries,
                    depths,
                    unit,
                    size,
                    remaining,
                    args.hard_split,
                    max_chunk_bytes,
                )
                self._finalize_chunks(file_chunks)

                for chunk in file_chunks:
                    chunks.append(
                        CodeChunk(
                            path=str(file_path),
                            language=language,
                            mode=resolved_mode,
                            chunk_index=chunk.index,
                            total_chunks=chunk.total_chunks,
                            start_line=chunk.start_line,
                            end_line=chunk.end_line,
                            min_depth=chunk.min_depth,
                            max_depth=chunk.max_depth,
                            unit=chunk.unit,
                            content=chunk.content,
                        )
                    )

                if file_truncated or len(chunks) >= max_chunks:
                    truncated = True
                    break
            except ToolError as exc:
                errors.append(str(exc))
            except Exception as exc:
                errors.append(f"{file_path}: {exc}")

        if process_mode == "chunk" and len(chunks) > max_chunks:
            chunks = chunks[:max_chunks]
            truncated = True

        return ProcessCodeRecursiveResult(
            mode=process_mode,
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
        self, args: ProcessCodeRecursiveArgs
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

    def _resolve_language(self, path: Path) -> str | None:
        ext = path.suffix.lower().lstrip(".")
        return ext if ext else None

    def _resolve_mode(self, mode: str, language: str | None, path: Path) -> str:
        if mode != "auto":
            return mode
        if language and language in INDENT_LANGS:
            return "indent"
        if path.suffix.lower() in INDENT_EXTS:
            return "indent"
        return "brace"

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
                elif ch in "[(":
                    bracket_depth += 1
                elif ch in "])":
                    bracket_depth = max(bracket_depth - 1, 0)

            depth = block_depth + bracket_depth
            safe = (
                block_depth <= split_depth
                and (not respect_brackets or bracket_depth == 0)
                and not state.in_string
                and not state.in_block_comment
            )
            boundaries.append(safe)
            depths.append(depth)

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
                if ch in "{[(":
                    bracket_depth += 1
                elif ch in "]})":
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
        max_chunk_bytes: int,
    ) -> tuple[list[CodeChunkData], bool]:
        chunks: list[CodeChunkData] = []
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
            if content_bytes > max_chunk_bytes:
                raise ToolError(
                    f"Chunk exceeds max_chunk_bytes ({content_bytes} > {max_chunk_bytes})."
                )
            chunk = CodeChunkData(
                index=len(chunks) + 1,
                total_chunks=0,
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

    def _finalize_chunks(self, chunks: list[CodeChunkData]) -> None:
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ProcessCodeRecursiveArgs):
            return ToolCallDisplay(summary="process_code_recursive")

        summary = f"process_code_recursive: {len(event.args.paths)} path(s)"
        return ToolCallDisplay(
            summary=summary,
            details={
                "paths": event.args.paths,
                "extension": event.args.extension,
                "extensions": event.args.extensions,
                "include_globs": event.args.include_globs,
                "exclude_globs": event.args.exclude_globs,
                "mode": event.args.mode,
                "code_mode": event.args.code_mode,
                "chunk_unit": event.args.chunk_unit,
                "chunk_size": event.args.chunk_size,
                "split_depth": event.args.split_depth,
                "respect_brackets": event.args.respect_brackets,
                "hard_split": event.args.hard_split,
                "max_chunks": event.args.max_chunks,
                "max_files": event.args.max_files,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ProcessCodeRecursiveResult):
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
        return "Processing code recursively"
