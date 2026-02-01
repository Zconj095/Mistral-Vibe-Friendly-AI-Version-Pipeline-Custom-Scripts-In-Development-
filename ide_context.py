from __future__ import annotations

import fnmatch
import os
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


DEFAULT_EXCLUDE_GLOBS = [
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
]


class FileRequest(BaseModel):
    path: str
    start_line: int | None = Field(
        default=None, description="1-based start line."
    )
    end_line: int | None = Field(
        default=None, description="1-based end line (inclusive)."
    )
    around_line: int | None = Field(
        default=None, description="1-based center line for context."
    )
    radius: int | None = Field(
        default=None, description="Lines around the center line."
    )
    max_lines: int | None = Field(
        default=None, description="Maximum lines to return."
    )


class FileSnippet(BaseModel):
    path: str
    start_line: int
    end_line: int
    total_lines: int
    content: str
    truncated: bool


class IdeContextArgs(BaseModel):
    paths: list[str] | None = Field(
        default=None, description="File paths to read."
    )
    files: list[FileRequest] | None = Field(
        default=None, description="Detailed file slice requests."
    )
    include_tree: bool = Field(
        default=False, description="Include a file tree snapshot."
    )
    tree_root: str | None = Field(
        default=None, description="Root directory for the tree snapshot."
    )
    include_globs: list[str] | None = Field(
        default=None, description="Include glob filters for tree listing."
    )
    exclude_globs: list[str] | None = Field(
        default=None, description="Exclude glob filters for tree listing."
    )
    max_depth: int | None = Field(
        default=None, description="Maximum depth for tree listing."
    )
    max_tree_files: int | None = Field(
        default=None, description="Maximum files in tree listing."
    )


class IdeContextResult(BaseModel):
    snippets: list[FileSnippet]
    tree: list[str]
    total_bytes: int
    warnings: list[str]


class IdeContextConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_file_bytes: int = Field(
        default=200_000, description="Maximum bytes to read per file."
    )
    max_total_bytes: int = Field(
        default=2_000_000, description="Maximum bytes across all snippets."
    )
    default_max_lines: int = Field(
        default=200, description="Default max lines per file."
    )
    default_radius: int = Field(
        default=30, description="Default radius for around_line."
    )
    default_exclude_globs: list[str] = Field(
        default=DEFAULT_EXCLUDE_GLOBS,
        description="Default glob patterns excluded during tree listing.",
    )
    default_max_tree_files: int = Field(
        default=500, description="Default max files for tree listing."
    )
    default_max_depth: int = Field(
        default=8, description="Default max depth for tree listing."
    )


class IdeContextState(BaseToolState):
    pass


class IdeContext(
    BaseTool[IdeContextArgs, IdeContextResult, IdeContextConfig, IdeContextState],
    ToolUIData[IdeContextArgs, IdeContextResult],
):
    description: ClassVar[str] = (
        "Collect file slices and optional file tree context for IDE-style workflows."
    )

    async def run(self, args: IdeContextArgs) -> IdeContextResult:
        base = self.config.effective_workdir.resolve()
        requests = self._build_requests(args)
        snippets: list[FileSnippet] = []
        warnings: list[str] = []
        total_bytes = 0

        for request in requests:
            file_path = self._resolve_path(request.path, base)
            snippet = self._read_slice(file_path, request)
            snippet_bytes = len(snippet.content.encode("utf-8"))
            if (
                self.config.max_total_bytes > 0
                and total_bytes + snippet_bytes > self.config.max_total_bytes
            ):
                warnings.append(
                    f"Total snippet bytes exceeded limit at {file_path}."
                )
                break
            total_bytes += snippet_bytes
            snippets.append(snippet)

        tree: list[str] = []
        if args.include_tree:
            tree_root = self._resolve_tree_root(args.tree_root, base)
            tree = self._build_tree(
                tree_root,
                args.include_globs,
                args.exclude_globs,
                args.max_depth,
                args.max_tree_files,
            )

        return IdeContextResult(
            snippets=snippets,
            tree=tree,
            total_bytes=total_bytes,
            warnings=warnings,
        )

    def _build_requests(self, args: IdeContextArgs) -> list[FileRequest]:
        requests: list[FileRequest] = []
        if args.paths:
            for path in args.paths:
                requests.append(FileRequest(path=path))
        if args.files:
            requests.extend(args.files)
        if not requests:
            raise ToolError("Provide paths or files to collect context.")
        return requests

    def _resolve_path(self, raw: str, base: Path) -> Path:
        if not raw or not raw.strip():
            raise ToolError("File path cannot be empty.")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = base / path
        path = path.resolve()
        try:
            path.relative_to(base)
        except ValueError as exc:
            raise ToolError(f"Path must stay within project root: {base}") from exc
        if not path.exists():
            raise ToolError(f"File not found: {path}")
        if not path.is_file():
            raise ToolError(f"Path is not a file: {path}")
        return path

    def _resolve_tree_root(self, raw: str | None, base: Path) -> Path:
        if raw:
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = base / path
            path = path.resolve()
        else:
            path = base
        try:
            path.relative_to(base)
        except ValueError as exc:
            raise ToolError(
                f"Tree root must stay within project root: {base}"
            ) from exc
        if not path.exists():
            raise ToolError(f"Tree root not found: {path}")
        if not path.is_dir():
            raise ToolError(f"Tree root is not a directory: {path}")
        return path

    def _read_slice(self, path: Path, request: FileRequest) -> FileSnippet:
        max_bytes = self.config.max_file_bytes
        truncated_by_bytes = False
        data = b""
        try:
            with path.open("rb") as handle:
                if max_bytes and max_bytes > 0:
                    data = handle.read(max_bytes + 1)
                    if len(data) > max_bytes:
                        data = data[:max_bytes]
                        truncated_by_bytes = True
                else:
                    data = handle.read()
        except OSError as exc:
            raise ToolError(f"Error reading {path}: {exc}") from exc

        text = data.decode("utf-8", errors="ignore")
        lines = text.splitlines()
        total_lines = len(lines)

        start_line, end_line = self._resolve_line_window(request, total_lines)
        if total_lines == 0:
            start_line = 0
            end_line = 0
            content = ""
            truncated = truncated_by_bytes
        else:
            start_idx = max(start_line - 1, 0)
            end_idx = min(end_line, total_lines)
            content = "\n".join(lines[start_idx:end_idx])
            truncated = truncated_by_bytes or end_idx < total_lines

        return FileSnippet(
            path=str(path),
            start_line=start_line,
            end_line=end_line,
            total_lines=total_lines,
            content=content,
            truncated=truncated,
        )

    def _resolve_line_window(
        self, request: FileRequest, total_lines: int
    ) -> tuple[int, int]:
        if total_lines <= 0:
            return 0, 0

        radius = request.radius or self.config.default_radius
        max_lines = (
            request.max_lines
            if request.max_lines is not None
            else self.config.default_max_lines
        )
        if max_lines <= 0:
            max_lines = total_lines

        if request.around_line:
            center = max(1, request.around_line)
            start_line = max(1, center - radius)
            end_line = min(total_lines, center + radius)
        else:
            start_line = request.start_line or 1
            end_line = request.end_line or 0
            if end_line <= 0:
                end_line = min(total_lines, start_line + max_lines - 1)
            if end_line < start_line:
                end_line = start_line

        if end_line - start_line + 1 > max_lines:
            end_line = min(total_lines, start_line + max_lines - 1)

        return max(1, start_line), min(total_lines, end_line)

    def _build_tree(
        self,
        root: Path,
        include_globs: list[str] | None,
        exclude_globs: list[str] | None,
        max_depth: int | None,
        max_files: int | None,
    ) -> list[str]:
        include = include_globs or ["**/*"]
        exclude = exclude_globs or self.config.default_exclude_globs
        depth_limit = (
            max_depth
            if max_depth is not None
            else self.config.default_max_depth
        )
        file_limit = (
            max_files
            if max_files is not None
            else self.config.default_max_tree_files
        )

        results: list[str] = []
        root_depth = len(root.parts)

        for dirpath, dirnames, filenames in os.walk(root):
            depth = len(Path(dirpath).parts) - root_depth
            if depth_limit >= 0 and depth > depth_limit:
                dirnames[:] = []
                continue

            for name in filenames:
                rel_path = Path(dirpath) / name
                rel = rel_path.relative_to(root).as_posix()
                if not self._matches_any(rel, include):
                    continue
                if self._matches_any(rel, exclude):
                    continue
                results.append(rel)
                if file_limit > 0 and len(results) >= file_limit:
                    return results
        return results

    @staticmethod
    def _matches_any(path: str, patterns: list[str]) -> bool:
        for pattern in patterns:
            if fnmatch.fnmatch(path, pattern):
                return True
        return False

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, IdeContextArgs):
            return ToolCallDisplay(summary="ide_context")
        return ToolCallDisplay(
            summary="ide_context",
            details={
                "paths": event.args.paths,
                "files": [item.path for item in event.args.files or []],
                "include_tree": event.args.include_tree,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, IdeContextResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = f"Collected {len(event.result.snippets)} snippet(s)"
        if event.result.tree:
            message += f" and {len(event.result.tree)} tree entries"
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=event.result.warnings,
            details={
                "snippets": [snippet.model_dump() for snippet in event.result.snippets],
                "tree": event.result.tree,
                "total_bytes": event.result.total_bytes,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Collecting IDE context"
