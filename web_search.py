from __future__ import annotations

import json
import re
import shutil
import subprocess
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


DEFAULT_ROOT = Path.home() / ".vibe" / "web_cache"
DEFAULT_GLOBS = ["**/*.md", "**/*.txt", "**/*.html", "**/*.htm", "**/*.json"]


class WebSearchConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    search_roots: list[Path] = Field(
        default_factory=lambda: [DEFAULT_ROOT],
        description="Directories to search for local cached content.",
    )
    include_globs: list[str] = Field(
        default_factory=lambda: list(DEFAULT_GLOBS),
        description="Glob patterns to include while searching.",
    )
    max_results: int = Field(default=10, description="Maximum matches to return.")
    context_lines: int = Field(default=0, description="Context lines around matches.")
    case_sensitive: bool = Field(default=False, description="Case sensitive search.")
    use_regex: bool = Field(default=False, description="Interpret query as regex.")
    max_file_bytes: int = Field(
        default=2_000_000, description="Maximum file size to scan in bytes."
    )
    allowlist: list[str] = Field(default_factory=list)
    denylist: list[str] = Field(default_factory=list)


class WebSearchState(BaseToolState):
    pass


class WebSearchArgs(BaseModel):
    query: str = Field(description="Search query.")
    roots: list[str] | None = Field(
        default=None, description="Override search roots."
    )
    include_globs: list[str] | None = Field(
        default=None, description="Override include globs."
    )
    max_results: int | None = Field(default=None, description="Max results.")
    context_lines: int | None = Field(default=None, description="Context lines.")
    case_sensitive: bool | None = Field(default=None, description="Case sensitive.")
    use_regex: bool | None = Field(default=None, description="Use regex search.")


class WebSearchResultItem(BaseModel):
    path: str
    line_number: int
    snippet: str


class WebSearchResult(BaseModel):
    query: str
    results: list[WebSearchResultItem]
    count: int
    searched_roots: list[str]


class WebSearch(
    BaseTool[WebSearchArgs, WebSearchResult, WebSearchConfig, WebSearchState],
    ToolUIData[WebSearchArgs, WebSearchResult],
):
    description: ClassVar[str] = (
        "Search local cached files for a query. This tool does not access the internet."
    )

    async def run(self, args: WebSearchArgs) -> WebSearchResult:
        if not args.query.strip():
            raise ToolError("query cannot be empty.")

        roots = self._resolve_roots(args.roots)
        if not roots:
            raise ToolError("No valid search roots configured.")

        include_globs = args.include_globs or self.config.include_globs
        max_results = args.max_results or self.config.max_results
        context_lines = (
            args.context_lines
            if args.context_lines is not None
            else self.config.context_lines
        )
        case_sensitive = (
            args.case_sensitive
            if args.case_sensitive is not None
            else self.config.case_sensitive
        )
        use_regex = (
            args.use_regex
            if args.use_regex is not None
            else self.config.use_regex
        )

        if max_results <= 0:
            raise ToolError("max_results must be a positive integer.")
        if context_lines < 0:
            raise ToolError("context_lines cannot be negative.")

        results: list[WebSearchResultItem] = []
        if self._rg_available() and context_lines == 0:
            results = self._search_with_rg(
                args.query,
                roots,
                include_globs,
                max_results,
                case_sensitive,
                use_regex,
            )
        else:
            results = self._search_with_python(
                args.query,
                roots,
                include_globs,
                max_results,
                context_lines,
                case_sensitive,
                use_regex,
            )

        return WebSearchResult(
            query=args.query,
            results=results,
            count=len(results),
            searched_roots=[str(root) for root in roots],
        )

    def _resolve_roots(self, raw_roots: list[str] | None) -> list[Path]:
        roots: list[Path] = []
        candidates = raw_roots or [str(p) for p in self.config.search_roots]
        for raw in candidates:
            if not raw:
                continue
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = self.config.effective_workdir / path
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if not resolved.exists():
                continue
            if not self._is_allowed_root(resolved):
                continue
            roots.append(resolved)
        return roots

    def _is_allowed_root(self, root: Path) -> bool:
        import fnmatch

        root_str = str(root)
        for pattern in self.config.denylist:
            if fnmatch.fnmatch(root_str, pattern):
                return False
        if self.config.allowlist:
            return any(fnmatch.fnmatch(root_str, pattern) for pattern in self.config.allowlist)
        return True

    def _rg_available(self) -> bool:
        return shutil.which("rg") is not None

    def _search_with_rg(
        self,
        query: str,
        roots: list[Path],
        include_globs: list[str],
        max_results: int,
        case_sensitive: bool,
        use_regex: bool,
    ) -> list[WebSearchResultItem]:
        cmd = ["rg", "--json"]
        if not use_regex:
            cmd.append("-F")
        if not case_sensitive:
            cmd.append("-i")
        for glob in include_globs:
            cmd.extend(["--glob", glob])
        cmd.append(query)
        cmd.extend([str(root) for root in roots])

        results: list[WebSearchResultItem] = []
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as exc:
            raise ToolError(f"Failed to run rg: {exc}") from exc

        assert proc.stdout is not None
        for line in proc.stdout:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "match":
                continue
            data = event.get("data", {})
            path = data.get("path", {}).get("text")
            line_number = data.get("line_number")
            snippet = data.get("lines", {}).get("text", "").rstrip("\n")
            if not path or line_number is None:
                continue
            results.append(
                WebSearchResultItem(
                    path=path,
                    line_number=int(line_number),
                    snippet=snippet,
                )
            )
            if len(results) >= max_results:
                proc.terminate()
                break

        proc.communicate(timeout=5)
        return results

    def _search_with_python(
        self,
        query: str,
        roots: list[Path],
        include_globs: list[str],
        max_results: int,
        context_lines: int,
        case_sensitive: bool,
        use_regex: bool,
    ) -> list[WebSearchResultItem]:
        results: list[WebSearchResultItem] = []
        regex = None
        if use_regex:
            flags = 0 if case_sensitive else re.IGNORECASE
            regex = re.compile(query, flags=flags)
        needle = query if case_sensitive else query.lower()

        for root in roots:
            files = self._gather_files(root, include_globs)
            for path in files:
                if len(results) >= max_results:
                    return results
                try:
                    if path.stat().st_size > self.config.max_file_bytes:
                        continue
                except OSError:
                    continue
                try:
                    content = path.read_text("utf-8", errors="ignore")
                except OSError:
                    continue
                lines = content.splitlines()
                for idx, line in enumerate(lines):
                    if use_regex:
                        if not regex or not regex.search(line):
                            continue
                    else:
                        hay = line if case_sensitive else line.lower()
                        if needle not in hay:
                            continue
                    start = max(0, idx - context_lines)
                    end = min(len(lines), idx + context_lines + 1)
                    snippet = "\n".join(lines[start:end])
                    results.append(
                        WebSearchResultItem(
                            path=str(path),
                            line_number=idx + 1,
                            snippet=snippet,
                        )
                    )
                    if len(results) >= max_results:
                        return results
        return results

    def _gather_files(self, root: Path, include_globs: list[str]) -> list[Path]:
        files: set[Path] = set()
        for pattern in include_globs:
            for path in root.rglob(pattern):
                if path.is_file():
                    files.add(path)
        return sorted(files)

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, WebSearchArgs):
            return ToolCallDisplay(summary="web_search")
        return ToolCallDisplay(
            summary=f"web_search: {event.args.query}",
            details={
                "query": event.args.query,
                "roots": event.args.roots,
                "max_results": event.args.max_results,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, WebSearchResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = f"Found {event.result.count} result(s)"
        return ToolResultDisplay(
            success=True,
            message=message,
            details={
                "query": event.result.query,
                "count": event.result.count,
                "results": event.result.results,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Searching local cache"
