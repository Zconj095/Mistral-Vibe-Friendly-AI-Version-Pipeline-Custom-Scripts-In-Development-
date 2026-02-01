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


class ChunkCodeNestedMultiConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_input_bytes: int = Field(
        default=5_000_000, description="Maximum input size per source in bytes."
    )
    max_total_bytes: int = Field(
        default=20_000_000, description="Maximum total bytes across sources."
    )
    max_chunk_bytes: int = Field(
        default=200_000, description="Maximum chunk size in bytes."
    )
    max_chunks: int = Field(
        default=500, description="Maximum number of chunks to return."
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
    max_files: int = Field(
        default=500, description="Maximum files to process."
    )
    default_include_globs: list[str] = Field(
        default=[
            "**/*.py",
            "**/*.pyi",
            "**/*.js",
            "**/*.ts",
            "**/*.tsx",
            "**/*.jsx",
            "**/*.java",
            "**/*.cs",
            "**/*.cpp",
            "**/*.c",
            "**/*.h",
            "**/*.hpp",
            "**/*.go",
            "**/*.rs",
            "**/*.php",
            "**/*.rb",
            "**/*.swift",
            "**/*.kt",
            "**/*.m",
            "**/*.mm",
            "**/*.json",
            "**/*.yaml",
            "**/*.yml",
            "**/*.toml",
        ],
        description="Default glob patterns used when include_globs is not provided.",
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


class ChunkCodeNestedMultiState(BaseToolState):
    pass


class CodeSnippet(BaseModel):
    name: str | None = Field(default=None, description="Optional snippet identifier.")
    content: str = Field(description="Snippet content.")
    language: str | None = Field(
        default=None, description="Optional language hint for this snippet."
    )


class ChunkCodeNestedMultiArgs(BaseModel):
    paths: list[str] | None = Field(
        default=None, description="File or directory paths."
    )
    snippets: list[CodeSnippet] | None = Field(
        default=None, description="Inline code snippets to chunk."
    )
    include_globs: list[str] | None = Field(
        default=None, description="Glob patterns for auto-discovery."
    )
    exclude_globs: list[str] | None = Field(
        default=None, description="Glob patterns to exclude."
    )
    max_files: int | None = Field(default=None, description="Override max files limit.")
    language: str | None = Field(
        default=None, description="Optional language hint for all sources."
    )
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


class CodeChunkMulti(BaseModel):
    source_path: str
    language: str | None
    mode: str
    index: int
    start_line: int
    end_line: int
    min_depth: int
    max_depth: int
    unit: str
    content: str


class ChunkCodeNestedMultiResult(BaseModel):
    chunks: list[CodeChunkMulti]
    count: int
    source_count: int
    truncated: bool
    errors: list[str]


class ChunkCodeNestedMulti(
    BaseTool[
        ChunkCodeNestedMultiArgs,
        ChunkCodeNestedMultiResult,
        ChunkCodeNestedMultiConfig,
        ChunkCodeNestedMultiState,
    ],
    ToolUIData[ChunkCodeNestedMultiArgs, ChunkCodeNestedMultiResult],
):
    description: ClassVar[str] = (
        "Chunk nested code across multiple files, snippets, and language extensions."
    )

    async def run(self, args: ChunkCodeNestedMultiArgs) -> ChunkCodeNestedMultiResult:
        paths = args.paths or []
        snippets = args.snippets or []
        if not paths and not snippets:
            raise ToolError("At least one path or snippet is required.")

        files: list[Path] = []
        if paths:
            include_globs = self._normalize_globs(
                args.include_globs, self.config.default_include_globs
            )
            exclude_globs = self._normalize_globs(
                args.exclude_globs, self.config.default_exclude_globs
            )
            max_files = (
                args.max_files if args.max_files is not None else self.config.max_files
            )
            if max_files <= 0:
                raise ToolError("max_files must be a positive integer.")

            files = self._gather_files(paths, include_globs, exclude_globs, max_files)
            if not files:
                raise ToolError("No files matched the provided paths/patterns.")

        mode = (args.mode or self.config.default_mode).strip().lower()
        if mode not in {"auto", "indent", "brace"}:
            raise ToolError("mode must be auto, indent, or brace.")

        unit = (args.unit or self.config.default_unit).strip().lower()
        if unit not in {"lines", "chars"}:
            raise ToolError("unit must be lines or chars.")

        size = args.size if args.size is not None else self.config.default_chunk_size
        if size <= 0:
            raise ToolError("size must be a positive integer.")

        split_depth = (
            args.split_depth
            if args.split_depth is not None
            else self.config.default_split_depth
        )
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

        total_bytes = 0
        chunks: list[CodeChunkMulti] = []
        errors: list[str] = []
        truncated = False
        source_count = 0

        def snippet_label(snippet: CodeSnippet, index: int) -> str:
            if snippet.name and snippet.name.strip():
                return snippet.name.strip()
            return f"snippet:{index}"

        for file_path in files:
            try:
                content = file_path.read_text("utf-8", errors="ignore")
                size_bytes = len(content.encode("utf-8"))
                if size_bytes > self.config.max_input_bytes:
                    raise ToolError(
                        f"{file_path} exceeds max_input_bytes ({size_bytes} > {self.config.max_input_bytes})."
                    )

                if total_bytes + size_bytes > self.config.max_total_bytes:
                    truncated = True
                    break

                total_bytes += size_bytes
                language = self._resolve_language(args.language, file_path)
                resolved_mode = self._resolve_mode(mode, language, file_path)
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
                )

                for chunk in file_chunks:
                    chunks.append(
                        CodeChunkMulti(
                            source_path=str(file_path),
                            language=language,
                            mode=resolved_mode,
                            index=chunk.index,
                            start_line=chunk.start_line,
                            end_line=chunk.end_line,
                            min_depth=chunk.min_depth,
                            max_depth=chunk.max_depth,
                            unit=chunk.unit,
                            content=chunk.content,
                        )
                    )

                source_count += 1
                if file_truncated or len(chunks) >= max_chunks:
                    truncated = True
                    break

            except ToolError as exc:
                errors.append(str(exc))

        for index, snippet in enumerate(snippets, start=1):
            if truncated or len(chunks) >= max_chunks:
                break
            label = snippet_label(snippet, index)
            try:
                content = snippet.content
                if not content:
                    raise ToolError("Snippet content is empty.")
                size_bytes = len(content.encode("utf-8"))
                if size_bytes > self.config.max_input_bytes:
                    raise ToolError(
                        f"{label} exceeds max_input_bytes ({size_bytes} > {self.config.max_input_bytes})."
                    )

                if total_bytes + size_bytes > self.config.max_total_bytes:
                    truncated = True
                    break

                total_bytes += size_bytes
                language = snippet.language or args.language
                if language:
                    language = language.strip().lower()
                resolved_mode = self._resolve_mode(mode, language, None)
                lines = content.splitlines()
                boundaries, depths = self._compute_boundaries(
                    lines, resolved_mode, split_depth, respect_brackets
                )

                remaining = max_chunks - len(chunks)
                snippet_chunks, snippet_truncated = self._build_chunks(
                    lines,
                    boundaries,
                    depths,
                    unit,
                    size,
                    remaining,
                    args.hard_split,
                )

                for chunk in snippet_chunks:
                    chunks.append(
                        CodeChunkMulti(
                            source_path=label,
                            language=language,
                            mode=resolved_mode,
                            index=chunk.index,
                            start_line=chunk.start_line,
                            end_line=chunk.end_line,
                            min_depth=chunk.min_depth,
                            max_depth=chunk.max_depth,
                            unit=chunk.unit,
                            content=chunk.content,
                        )
                    )

                source_count += 1
                if snippet_truncated or len(chunks) >= max_chunks:
                    truncated = True
                    break

            except ToolError as exc:
                errors.append(f"{label}: {exc}")

        return ChunkCodeNestedMultiResult(
            chunks=chunks,
            count=len(chunks),
            source_count=source_count,
            truncated=truncated,
            errors=errors,
        )

    def _normalize_globs(
        self, value: list[str] | None, defaults: list[str]
    ) -> list[str]:
        globs = value if value is not None else defaults
        globs = [g.strip() for g in globs if g and g.strip()]
        if not globs:
            raise ToolError("include_globs/exclude_globs cannot be empty.")
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
                    seen.add(key)
                    discovered.append(path)
                    if len(discovered) >= max_files:
                        return discovered
            else:
                raise ToolError(f"Path not found: {path}")

        return discovered

    def _is_excluded(self, rel_path: str, exclude_globs: list[str]) -> bool:
        return any(fnmatch.fnmatch(rel_path, pattern) for pattern in exclude_globs)

    def _resolve_language(self, language: str | None, path: Path) -> str | None:
        if language:
            return language.strip().lower()
        ext = path.suffix.lower().lstrip(".")
        return ext if ext else None

    def _resolve_mode(self, mode: str, language: str | None, path: Path | None) -> str:
        if mode != "auto":
            return mode
        if language and language in INDENT_LANGS:
            return "indent"
        if path and path.suffix.lower() in INDENT_EXTS:
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
    ) -> tuple[list[CodeChunkMulti], bool]:
        chunks: list[CodeChunkMulti] = []
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
            chunk = CodeChunkMulti(
                source_path="",
                language=None,
                mode="",
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
        if not isinstance(event.args, ChunkCodeNestedMultiArgs):
            return ToolCallDisplay(summary="chunk_code_nested_multi")

        paths = event.args.paths or []
        snippets = event.args.snippets or []
        snippet_labels: list[str] = []
        for index, snippet in enumerate(snippets, start=1):
            label = snippet.name.strip() if snippet.name and snippet.name.strip() else f"snippet:{index}"
            if snippet.language:
                label = f"{label} ({snippet.language})"
            snippet_labels.append(label)

        summary = (
            f"chunk_code_nested_multi: {len(paths)} path(s), {len(snippets)} snippet(s)"
        )
        return ToolCallDisplay(
            summary=summary,
            details={
                "paths": paths,
                "snippets": snippet_labels,
                "include_globs": event.args.include_globs,
                "exclude_globs": event.args.exclude_globs,
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
        if not isinstance(event.result, ChunkCodeNestedMultiResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = f"Created {event.result.count} chunk(s) from {event.result.source_count} file(s)"
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
        return "Chunking nested code across files"
