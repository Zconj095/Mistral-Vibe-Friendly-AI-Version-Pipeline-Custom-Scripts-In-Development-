from __future__ import annotations

from pathlib import Path
import fnmatch
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


class ContextPackMultiConfig(BaseToolConfig):
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
    max_files_per_codebase: int = Field(
        default=200, description="Maximum files to auto-discover per codebase."
    )
    default_include_globs: list[str] = Field(
        default=[
            "**/*.py",
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
            "**/*.md",
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


class ContextPackMultiState(BaseToolState):
    pass


class ContextRootSource(BaseModel):
    id: str | None = Field(default=None, description="Optional source identifier.")
    label: str | None = Field(default=None, description="Optional label for the source.")
    language: str | None = Field(default=None, description="Language hint.")
    root: str = Field(description="Root directory for the codebase.")
    path: str | None = Field(
        default=None,
        description="Relative file path within the codebase.",
    )
    include_globs: list[str] | None = Field(
        default=None,
        description="Glob patterns for auto-discovery within the codebase.",
    )
    exclude_globs: list[str] | None = Field(
        default=None,
        description="Glob patterns to exclude during auto-discovery.",
    )
    max_files: int | None = Field(
        default=None,
        description="Maximum number of files to discover for this codebase.",
    )
    start_line: int | None = Field(
        default=None, description="1-based start line (inclusive)."
    )
    end_line: int | None = Field(
        default=None, description="1-based end line (inclusive)."
    )


class ContextPackMultiArgs(BaseModel):
    sources: list[ContextRootSource]
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
    root: str
    path: str
    relative_path: str
    language: str | None
    chunk_index: int
    total_chunks: int
    unit: str
    start: int | None
    end: int | None
    content: str


class ContextPackMultiResult(BaseModel):
    chunks: list[ContextChunk]
    count: int
    source_count: int
    truncated: bool
    errors: list[str]


class ContextPackMulti(
    BaseTool[
        ContextPackMultiArgs,
        ContextPackMultiResult,
        ContextPackMultiConfig,
        ContextPackMultiState,
    ],
    ToolUIData[ContextPackMultiArgs, ContextPackMultiResult],
):
    description: ClassVar[str] = (
        "Bundle context chunks across multiple scripts or codebases."
    )

    async def run(self, args: ContextPackMultiArgs) -> ContextPackMultiResult:
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
                expanded_sources = self._expand_sources(source, index)
                for expanded in expanded_sources:
                    source_id, label, language, root, rel_path, content = self._load_source(
                        expanded, index
                    )
                    content, base_line = self._slice_lines(
                        content, expanded.start_line, expanded.end_line
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
                        root,
                        rel_path,
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
                if truncated:
                    break
            except ToolError as exc:
                errors.append(str(exc))

        return ContextPackMultiResult(
            chunks=chunks,
            count=len(chunks),
            source_count=source_count,
            truncated=truncated,
            errors=errors,
        )

    def _expand_sources(
        self, source: ContextRootSource, index: int
    ) -> list[ContextRootSource]:
        if source.path:
            if source.include_globs or source.exclude_globs:
                raise ToolError("Provide either path or include_globs/exclude_globs, not both.")
            return [source]

        root_path = self._resolve_root(source.root)
        include_globs = self._normalize_globs(
            source.include_globs, self.config.default_include_globs
        )
        exclude_globs = self._normalize_globs(
            source.exclude_globs, self.config.default_exclude_globs
        )
        max_files = (
            source.max_files
            if source.max_files is not None
            else self.config.max_files_per_codebase
        )
        if max_files <= 0:
            raise ToolError("max_files must be a positive integer.")

        discovered = self._discover_files(root_path, include_globs, exclude_globs, max_files)
        if not discovered:
            raise ToolError(f"No files matched include_globs under root {root_path}.")

        expanded: list[ContextRootSource] = []
        for rel_path in discovered:
            expanded.append(
                ContextRootSource(
                    id=self._make_source_id(source, rel_path),
                    label=source.label,
                    language=source.language,
                    root=str(root_path),
                    path=rel_path,
                    start_line=source.start_line,
                    end_line=source.end_line,
                )
            )
        return expanded

    def _load_source(
        self, source: ContextRootSource, index: int
    ) -> tuple[str, str | None, str | None, str, str, str]:
        root_path = self._resolve_root(source.root)
        rel_path = (source.path or "").strip()
        if not rel_path:
            raise ToolError("path cannot be empty.")
        if Path(rel_path).is_absolute():
            raise ToolError("path must be relative to root.")

        full_path = (root_path / rel_path).resolve()
        if not self._is_within_root(full_path, root_path):
            raise ToolError(f"path '{rel_path}' is outside root '{root_path}'.")
        if not full_path.exists():
            raise ToolError(f"File not found at: {full_path}")
        if full_path.is_dir():
            raise ToolError(f"Path is a directory, not a file: {full_path}")

        source_id = source.id or source.label or rel_path or f"source-{index}"
        label = source.label
        language = source.language.strip().lower() if source.language else None
        content = full_path.read_text("utf-8", errors="ignore")
        return source_id, label, language, str(root_path), rel_path, content

    def _resolve_root(self, root: str) -> Path:
        if not root.strip():
            raise ToolError("root cannot be empty.")
        path = Path(root).expanduser()
        if not path.is_absolute():
            path = self.config.effective_workdir / path
        resolved = path.resolve()
        if not resolved.exists():
            raise ToolError(f"Root not found at: {resolved}")
        if not resolved.is_dir():
            raise ToolError(f"Root is not a directory: {resolved}")
        return resolved

    def _is_within_root(self, path: Path, root: Path) -> bool:
        try:
            return path.is_relative_to(root)
        except AttributeError:
            try:
                path.relative_to(root)
                return True
            except ValueError:
                return False

    def _normalize_globs(
        self, value: list[str] | None, defaults: list[str]
    ) -> list[str]:
        globs = value if value is not None else defaults
        globs = [g.strip() for g in globs if g and g.strip()]
        if not globs:
            raise ToolError("include_globs/exclude_globs cannot be empty.")
        return globs

    def _discover_files(
        self,
        root: Path,
        include_globs: list[str],
        exclude_globs: list[str],
        max_files: int,
    ) -> list[str]:
        discovered: list[str] = []
        seen: set[str] = set()

        for pattern in include_globs:
            for path in root.glob(pattern):
                if not path.is_file():
                    continue
                rel = path.relative_to(root).as_posix()
                if self._is_excluded(rel, exclude_globs):
                    continue
                if rel in seen:
                    continue
                seen.add(rel)
                discovered.append(rel)
                if len(discovered) >= max_files:
                    return sorted(discovered)

        return sorted(discovered)

    def _is_excluded(self, rel_path: str, exclude_globs: list[str]) -> bool:
        return any(fnmatch.fnmatch(rel_path, pattern) for pattern in exclude_globs)

    def _make_source_id(self, source: ContextRootSource, rel_path: str) -> str:
        prefix = source.id or source.label
        if prefix:
            return f"{prefix}:{rel_path}"
        return rel_path

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
        root: str,
        rel_path: str,
        language: str | None,
    ) -> list[ContextChunk]:
        if unit == "none":
            self._check_chunk_bytes(content, source_id)
            return [
                ContextChunk(
                    source_id=source_id,
                    label=label,
                    root=root,
                    path=str(Path(root) / rel_path),
                    relative_path=rel_path,
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
                root,
                rel_path,
                language,
            )
        return self._chunk_chars(
            content,
            size,
            overlap,
            source_id,
            label,
            root,
            rel_path,
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
        root: str,
        rel_path: str,
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
                    root=root,
                    path=str(Path(root) / rel_path),
                    relative_path=rel_path,
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
        root: str,
        rel_path: str,
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
                    root=root,
                    path=str(Path(root) / rel_path),
                    relative_path=rel_path,
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
        if not isinstance(event.args, ContextPackMultiArgs):
            return ToolCallDisplay(summary="context_pack_multi")

        summary = f"context_pack_multi: {len(event.args.sources)} source(s)"
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
        if not isinstance(event.result, ContextPackMultiResult):
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
        return "Packing multi-codebase context"
