from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import os
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


INDENT_EXTS = {".py", ".pyi", ".yml", ".yaml"}
INDENT_LANGS = {"python", "py", "yaml", "yml"}

EXTENSION_HINTS = [
    ".py",
    ".pyi",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".mjs",
    ".cjs",
    ".java",
    ".kt",
    ".cs",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".go",
    ".rs",
    ".php",
    ".rb",
    ".swift",
    ".m",
    ".mm",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
]

LANG_ALIASES = {
    "python": "py",
    "py": "py",
    "pyi": "py",
    "javascript": "js",
    "js": "js",
    "typescript": "ts",
    "ts": "ts",
    "tsx": "tsx",
    "jsx": "jsx",
    "mjs": "mjs",
    "cjs": "cjs",
    "java": "java",
    "kotlin": "kt",
    "kt": "kt",
    "scala": "scala",
    "csharp": "cs",
    "cs": "cs",
    "golang": "go",
    "go": "go",
    "rust": "rs",
    "rs": "rs",
    "c": "c",
    "cpp": "cpp",
    "cxx": "cpp",
    "cc": "cpp",
    "objective-c": "m",
    "objc": "m",
    "objective-c++": "mm",
    "objcpp": "mm",
    "php": "php",
    "ruby": "rb",
    "rb": "rb",
    "swift": "swift",
}

PY_IMPORT_RE = re.compile(r"^\s*import\s+(.+)")
PY_FROM_RE = re.compile(r"^\s*from\s+([A-Za-z0-9_\.]+)\s+import\s+")

JS_IMPORT_FROM_RE = re.compile(r"^\s*import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]")
JS_IMPORT_SIDE_RE = re.compile(r"^\s*import\s+['\"]([^'\"]+)['\"]")
JS_EXPORT_FROM_RE = re.compile(r"^\s*export\s+.*?\s+from\s+['\"]([^'\"]+)['\"]")
JS_REQUIRE_RE = re.compile(r"\brequire\(\s*['\"]([^'\"]+)['\"]\s*\)")
JS_DYNAMIC_IMPORT_RE = re.compile(r"\bimport\(\s*['\"]([^'\"]+)['\"]\s*\)")

C_INCLUDE_RE = re.compile(r"^\s*#\s*include\s+[<\"]([^>\"]+)[>\"]")

JAVA_IMPORT_RE = re.compile(r"^\s*import\s+(static\s+)?([A-Za-z0-9_\.]+)")
CSHARP_USING_RE = re.compile(r"^\s*using\s+([A-Za-z0-9_\.]+)")
SWIFT_IMPORT_RE = re.compile(r"^\s*import\s+([A-Za-z0-9_\.]+)")

GO_IMPORT_SINGLE_RE = re.compile(r'^\s*import\s+"([^"]+)"')
GO_IMPORT_BLOCK_START_RE = re.compile(r"^\s*import\s*\(")
GO_IMPORT_SPEC_RE = re.compile(r'^\s*(?:[A-Za-z0-9_\.]+\s+)?\"([^\"]+)\"')

RUST_MOD_RE = re.compile(r"^\s*mod\s+([A-Za-z0-9_]+)\s*;")
RUST_USE_RE = re.compile(r"^\s*use\s+([A-Za-z0-9_:]+)")
RUST_EXTERN_CRATE_RE = re.compile(r"^\s*extern\s+crate\s+([A-Za-z0-9_]+)")

PHP_INCLUDE_RE = re.compile(
    r"^\s*(include|include_once|require|require_once)\s*[\"']([^\"']+)[\"']"
)
RUBY_REQUIRE_RE = re.compile(r"^\s*require(?:_relative)?\s+[\"']([^\"']+)[\"']")


@dataclass
class _ScanState:
    in_block_comment: bool = False
    in_string: str | None = None


@dataclass
class _SourceItem:
    source_path: str
    path: Path | None
    language: str | None
    content: str
    lines: list[str]
    module_name: str | None = None
    rel_path: str | None = None
    rel_no_ext: str | None = None
    stem: str | None = None
    dir_path: Path | None = None


@dataclass
class _Reference:
    source: _SourceItem
    target: str
    relation: str
    line: int | None
    raw: str | None


class ChunkCodeConnectedMultiConfig(BaseToolConfig):
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
    max_connections: int = Field(
        default=2000, description="Maximum number of connections to return."
    )
    include_unresolved: bool = Field(
        default=True, description="Include unresolved/external references."
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
    max_files: int = Field(default=500, description="Maximum files to process.")
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


class ChunkCodeConnectedMultiState(BaseToolState):
    pass


class CodeSnippet(BaseModel):
    name: str | None = Field(default=None, description="Optional snippet identifier.")
    content: str = Field(description="Snippet content.")
    language: str | None = Field(
        default=None, description="Optional language hint for this snippet."
    )


class ChunkCodeConnectedMultiArgs(BaseModel):
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
    analyze_connections: bool = Field(
        default=True, description="Extract cross-file references and connections."
    )
    max_connections: int | None = Field(
        default=None, description="Override the configured max connections limit."
    )
    include_unresolved: bool | None = Field(
        default=None, description="Include unresolved/external references."
    )


class CodeChunkConnected(BaseModel):
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


class CodeConnection(BaseModel):
    id: int
    source_path: str
    source_language: str | None
    target_path: str | None
    target_language: str | None
    target_ref: str
    relation: str
    line: int | None
    source_chunk_index: int | None
    resolved: bool
    notes: str | None


class ChunkCodeConnectedMultiResult(BaseModel):
    chunks: list[CodeChunkConnected]
    connections: list[CodeConnection]
    count: int
    connection_count: int
    source_count: int
    truncated: bool
    connections_truncated: bool
    errors: list[str]


class _SourceIndex:
    def __init__(self, sources: list[_SourceItem], common_root: Path | None) -> None:
        self.common_root = common_root
        self.by_path: dict[str, list[_SourceItem]] = {}
        self.by_path_no_ext: dict[str, list[_SourceItem]] = {}
        self.by_rel: dict[str, list[_SourceItem]] = {}
        self.by_rel_no_ext: dict[str, list[_SourceItem]] = {}
        self.by_dotted: dict[str, list[_SourceItem]] = {}
        self.by_stem: dict[str, list[_SourceItem]] = {}
        self.by_basename: dict[str, list[_SourceItem]] = {}
        self.by_dir: dict[str, list[_SourceItem]] = {}
        self._index_sources(sources)

    def _normalize_path(self, path: Path | str) -> str:
        return os.path.normcase(os.path.normpath(str(path)))

    def _add_key(
        self, mapping: dict[str, list[_SourceItem]], key: str | None, source: _SourceItem
    ) -> None:
        if not key:
            return
        mapping.setdefault(key, []).append(source)

    def _index_sources(self, sources: list[_SourceItem]) -> None:
        for source in sources:
            if source.path:
                source.dir_path = source.path.parent
                source.stem = source.path.stem
                rel_path = None
                if self.common_root:
                    try:
                        rel_path = source.path.relative_to(self.common_root).as_posix()
                    except ValueError:
                        rel_path = source.path.name
                else:
                    rel_path = source.path.name
                source.rel_path = rel_path
                if rel_path:
                    rel_no_ext = Path(rel_path).with_suffix("").as_posix()
                    source.rel_no_ext = rel_no_ext
                    module_name = rel_no_ext.replace("/", ".")
                    dir_rel = Path(rel_path).parent.as_posix()
                    if dir_rel == ".":
                        dir_rel = ""
                    if source.stem in {"__init__", "index"} and dir_rel:
                        module_name = dir_rel.replace("/", ".")
                        self._add_key(self.by_rel_no_ext, dir_rel, source)
                        self._add_key(self.by_dotted, module_name, source)
                    source.module_name = module_name
                    if dir_rel:
                        self._add_key(self.by_dir, dir_rel, source)

                resolved = self._normalize_path(source.path.resolve())
                self._add_key(self.by_path, resolved, source)
                self._add_key(
                    self.by_path_no_ext,
                    self._normalize_path(source.path.with_suffix("")),
                    source,
                )
                self._add_key(self.by_rel, source.rel_path, source)
                self._add_key(self.by_rel_no_ext, source.rel_no_ext, source)
                self._add_key(self.by_dotted, source.module_name, source)
                self._add_key(self.by_stem, source.stem, source)
                self._add_key(self.by_basename, source.path.name, source)
            else:
                if source.source_path:
                    source.stem = Path(source.source_path).stem
                    self._add_key(self.by_stem, source.stem, source)

    def _unique(self, items: list[_SourceItem]) -> list[_SourceItem]:
        seen: set[str] = set()
        unique: list[_SourceItem] = []
        for item in items:
            key = item.source_path
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique

    def _match_path(self, path: Path) -> list[_SourceItem]:
        candidates: list[_SourceItem] = []
        try:
            resolved = self._normalize_path(path.resolve())
        except (OSError, ValueError):
            resolved = self._normalize_path(path)
        candidates.extend(self.by_path.get(resolved, []))
        if path.suffix:
            no_ext = self._normalize_path(path.with_suffix(""))
            candidates.extend(self.by_path_no_ext.get(no_ext, []))
        return candidates

    def _rel_matches(self, target: str) -> list[_SourceItem]:
        norm = target.replace("\\", "/")
        norm = norm.lstrip("./")
        candidates: list[_SourceItem] = []
        candidates.extend(self.by_rel.get(norm, []))
        norm_no_ext = Path(norm).with_suffix("").as_posix()
        candidates.extend(self.by_rel_no_ext.get(norm_no_ext, []))
        if norm.startswith("@/"):
            alias = norm[2:]
            candidates.extend(self.by_rel.get(alias, []))
            alias_no_ext = Path(alias).with_suffix("").as_posix()
            candidates.extend(self.by_rel_no_ext.get(alias_no_ext, []))
        return candidates

    def resolve_path(self, target: str, source: _SourceItem) -> list[_SourceItem]:
        candidates: list[_SourceItem] = []
        candidates.extend(self._rel_matches(target))

        path = Path(target)
        if path.is_absolute():
            candidates.extend(self._match_path(path))

        if source.path:
            base = source.path.parent
            candidates.extend(self._match_path(base / target))
            if not path.suffix:
                for ext in EXTENSION_HINTS:
                    candidates.extend(self._match_path((base / target).with_suffix(ext)))
                for ext in EXTENSION_HINTS:
                    candidates.extend(self._match_path(base / target / f"index{ext}"))
                candidates.extend(self._match_path(base / target / "__init__.py"))

        if self.common_root:
            candidates.extend(self._match_path(self.common_root / target))
            if not path.suffix:
                for ext in EXTENSION_HINTS:
                    candidates.extend(
                        self._match_path((self.common_root / target).with_suffix(ext))
                    )
                for ext in EXTENSION_HINTS:
                    candidates.extend(
                        self._match_path(self.common_root / target / f"index{ext}")
                    )
                candidates.extend(self._match_path(self.common_root / target / "__init__.py"))

        return self._unique(candidates)

    def resolve_mod(self, target: str, source: _SourceItem) -> list[_SourceItem]:
        if not source.path:
            return []
        base = source.path.parent
        candidates: list[_SourceItem] = []
        candidates.extend(self._match_path(base / f"{target}.rs"))
        candidates.extend(self._match_path(base / target / "mod.rs"))
        return self._unique(candidates)

    def resolve_module(self, target: str, source: _SourceItem) -> list[_SourceItem]:
        module = target.strip().strip(";")
        module = module.replace("::", ".")
        if module.endswith(".*"):
            module = module[:-2]
        if module.startswith("."):
            resolved = _resolve_relative_module(source.module_name, module)
            if resolved:
                module = resolved
            else:
                module = module.lstrip(".")

        candidates: list[_SourceItem] = []
        candidates.extend(self.by_dotted.get(module, []))
        candidates.extend(self.by_rel_no_ext.get(module.replace(".", "/"), []))
        if module in self.by_dir:
            candidates.extend(self.by_dir.get(module, []))
        if not candidates and module:
            tail = module.split(".")[-1]
            candidates.extend(self.by_stem.get(tail, []))

        return self._unique(candidates)


class ChunkCodeConnectedMulti(
    BaseTool[
        ChunkCodeConnectedMultiArgs,
        ChunkCodeConnectedMultiResult,
        ChunkCodeConnectedMultiConfig,
        ChunkCodeConnectedMultiState,
    ],
    ToolUIData[ChunkCodeConnectedMultiArgs, ChunkCodeConnectedMultiResult],
):
    description: ClassVar[str] = (
        "Chunk multi-language code and return a cross-file connection map."
    )

    async def run(self, args: ChunkCodeConnectedMultiArgs) -> ChunkCodeConnectedMultiResult:
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

        max_connections = (
            args.max_connections
            if args.max_connections is not None
            else self.config.max_connections
        )
        if max_connections <= 0:
            raise ToolError("max_connections must be a positive integer.")

        include_unresolved = (
            args.include_unresolved
            if args.include_unresolved is not None
            else self.config.include_unresolved
        )

        total_bytes = 0
        sources: list[_SourceItem] = []
        errors: list[str] = []
        truncated = False

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
                language = self._normalize_language(args.language, file_path)
                sources.append(
                    _SourceItem(
                        source_path=str(file_path),
                        path=file_path,
                        language=language,
                        content=content,
                        lines=content.splitlines(),
                    )
                )
            except ToolError as exc:
                errors.append(str(exc))

        for index, snippet in enumerate(snippets, start=1):
            if truncated:
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
                language = self._normalize_language(snippet.language or args.language, None)
                sources.append(
                    _SourceItem(
                        source_path=label,
                        path=None,
                        language=language,
                        content=content,
                        lines=content.splitlines(),
                    )
                )
            except ToolError as exc:
                errors.append(f"{label}: {exc}")

        if not sources:
            return ChunkCodeConnectedMultiResult(
                chunks=[],
                connections=[],
                count=0,
                connection_count=0,
                source_count=0,
                truncated=truncated,
                connections_truncated=False,
                errors=errors,
            )

        common_root = self._common_root([s.path for s in sources if s.path])
        index = _SourceIndex(sources, common_root)

        chunks: list[CodeChunkConnected] = []
        chunk_ranges: dict[str, list[tuple[int, int, int]]] = {}
        source_count = 0

        for source in sources:
            if len(chunks) >= max_chunks:
                truncated = True
                break
            try:
                resolved_mode = self._resolve_mode(mode, source.language, source.path)
                lines = source.lines
                boundaries, depths = self._compute_boundaries(
                    lines, resolved_mode, split_depth, respect_brackets
                )

                remaining = max_chunks - len(chunks)
                source_chunks, source_truncated = self._build_chunks(
                    lines,
                    boundaries,
                    depths,
                    unit,
                    size,
                    remaining,
                    args.hard_split,
                )

                for chunk in source_chunks:
                    chunks.append(
                        CodeChunkConnected(
                            source_path=source.source_path,
                            language=source.language,
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
                    chunk_ranges.setdefault(source.source_path, []).append(
                        (chunk.start_line, chunk.end_line, chunk.index)
                    )

                source_count += 1
                if source_truncated or len(chunks) >= max_chunks:
                    truncated = True
                    break
            except ToolError as exc:
                errors.append(str(exc))

        connections: list[CodeConnection] = []
        connections_truncated = False
        if args.analyze_connections and sources:
            connection_id = 1
            seen: set[tuple[str, str, str, int | None, str | None]] = set()
            for source in sources:
                if connections_truncated:
                    break
                refs = self._extract_references(source)
                for ref in refs:
                    if len(connections) >= max_connections:
                        connections_truncated = True
                        break

                    resolved_targets: list[_SourceItem] = []
                    if ref.relation == "mod":
                        resolved_targets = index.resolve_mod(ref.target, source)
                    elif self._is_path_like(ref.target):
                        resolved_targets = index.resolve_path(ref.target, source)
                    else:
                        resolved_targets = index.resolve_module(ref.target, source)

                    if resolved_targets:
                        for target in resolved_targets:
                            key = (
                                source.source_path,
                                ref.target,
                                ref.relation,
                                ref.line,
                                target.source_path,
                            )
                            if key in seen:
                                continue
                            seen.add(key)
                            connections.append(
                                CodeConnection(
                                    id=connection_id,
                                    source_path=source.source_path,
                                    source_language=source.language,
                                    target_path=target.source_path,
                                    target_language=target.language,
                                    target_ref=ref.target,
                                    relation=ref.relation,
                                    line=ref.line,
                                    source_chunk_index=self._find_chunk_index(
                                        chunk_ranges.get(source.source_path, []),
                                        ref.line,
                                    ),
                                    resolved=True,
                                    notes=None,
                                )
                            )
                            connection_id += 1
                            if len(connections) >= max_connections:
                                connections_truncated = True
                                break
                    elif include_unresolved:
                        key = (
                            source.source_path,
                            ref.target,
                            ref.relation,
                            ref.line,
                            None,
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        connections.append(
                            CodeConnection(
                                id=connection_id,
                                source_path=source.source_path,
                                source_language=source.language,
                                target_path=None,
                                target_language=None,
                                target_ref=ref.target,
                                relation=ref.relation,
                                line=ref.line,
                                source_chunk_index=self._find_chunk_index(
                                    chunk_ranges.get(source.source_path, []),
                                    ref.line,
                                ),
                                resolved=False,
                                notes="Unresolved or external reference",
                            )
                        )
                        connection_id += 1
                        if len(connections) >= max_connections:
                            connections_truncated = True
                            break

        return ChunkCodeConnectedMultiResult(
            chunks=chunks,
            connections=connections,
            count=len(chunks),
            connection_count=len(connections),
            source_count=source_count,
            truncated=truncated,
            connections_truncated=connections_truncated,
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

    def _normalize_language(self, language: str | None, path: Path | None) -> str | None:
        if language:
            key = language.strip().lower()
            return LANG_ALIASES.get(key, key)
        if path:
            ext = path.suffix.lower().lstrip(".")
            if ext:
                return LANG_ALIASES.get(ext, ext)
        return None

    def _resolve_mode(self, mode: str, language: str | None, path: Path | None) -> str:
        if mode != "auto":
            return mode
        if language and language in INDENT_LANGS:
            return "indent"
        if path and path.suffix.lower() in INDENT_EXTS:
            return "indent"
        return "brace"

    def _common_root(self, paths: list[Path]) -> Path | None:
        if not paths:
            return None
        try:
            root = os.path.commonpath([str(p) for p in paths])
        except ValueError:
            return None
        return Path(root)

    def _is_path_like(self, target: str) -> bool:
        if target.startswith((".", "/", "\\")):
            return True
        if "/" in target or "\\" in target:
            return True
        lower = target.lower()
        return any(lower.endswith(ext) for ext in EXTENSION_HINTS)

    def _extract_references(self, source: _SourceItem) -> list[_Reference]:
        lang = source.language or ""
        refs: list[_Reference] = []
        if lang in {"py"}:
            return self._extract_python_refs(source)
        if lang in {"js", "jsx", "ts", "tsx", "mjs", "cjs"}:
            return self._extract_js_refs(source)
        if lang in {"java", "kt", "scala"}:
            return self._extract_java_refs(source)
        if lang in {"cs"}:
            return self._extract_csharp_refs(source)
        if lang in {"go"}:
            return self._extract_go_refs(source)
        if lang in {"rs"}:
            return self._extract_rust_refs(source)
        if lang in {"c", "cpp", "h", "hpp", "m", "mm"}:
            return self._extract_c_refs(source)
        if lang in {"php"}:
            return self._extract_php_refs(source)
        if lang in {"rb"}:
            return self._extract_ruby_refs(source)
        if lang in {"swift"}:
            return self._extract_swift_refs(source)
        return refs

    def _extract_python_refs(self, source: _SourceItem) -> list[_Reference]:
        refs: list[_Reference] = []
        for line_no, line in enumerate(source.lines, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            match = PY_FROM_RE.match(line)
            if match:
                module = match.group(1).strip()
                if module:
                    refs.append(
                        _Reference(source, module, "import", line_no, line.strip())
                    )
                continue
            match = PY_IMPORT_RE.match(line)
            if match:
                modules = self._split_python_imports(match.group(1))
                for module in modules:
                    refs.append(
                        _Reference(source, module, "import", line_no, line.strip())
                    )
        return refs

    def _split_python_imports(self, raw: str) -> list[str]:
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        modules: list[str] = []
        for part in parts:
            if " as " in part:
                part = part.split(" as ")[0].strip()
            modules.append(part)
        return modules

    def _extract_js_refs(self, source: _SourceItem) -> list[_Reference]:
        refs: list[_Reference] = []
        for line_no, line in enumerate(source.lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            match = JS_IMPORT_FROM_RE.match(line)
            if match:
                refs.append(
                    _Reference(source, match.group(1), "import", line_no, stripped)
                )
                continue
            match = JS_IMPORT_SIDE_RE.match(line)
            if match:
                refs.append(
                    _Reference(source, match.group(1), "import", line_no, stripped)
                )
            match = JS_EXPORT_FROM_RE.match(line)
            if match:
                refs.append(
                    _Reference(source, match.group(1), "export", line_no, stripped)
                )
            for target in JS_REQUIRE_RE.findall(line):
                refs.append(_Reference(source, target, "require", line_no, stripped))
            for target in JS_DYNAMIC_IMPORT_RE.findall(line):
                refs.append(_Reference(source, target, "import", line_no, stripped))
        return refs

    def _extract_java_refs(self, source: _SourceItem) -> list[_Reference]:
        refs: list[_Reference] = []
        for line_no, line in enumerate(source.lines, start=1):
            match = JAVA_IMPORT_RE.match(line)
            if match:
                module = match.group(2)
                if module:
                    refs.append(
                        _Reference(source, module, "import", line_no, line.strip())
                    )
        return refs

    def _extract_csharp_refs(self, source: _SourceItem) -> list[_Reference]:
        refs: list[_Reference] = []
        for line_no, line in enumerate(source.lines, start=1):
            if " using " in f" {line} " and "=" in line:
                continue
            if " static " in f" {line} ":
                continue
            match = CSHARP_USING_RE.match(line)
            if match:
                module = match.group(1)
                refs.append(_Reference(source, module, "using", line_no, line.strip()))
        return refs

    def _extract_swift_refs(self, source: _SourceItem) -> list[_Reference]:
        refs: list[_Reference] = []
        for line_no, line in enumerate(source.lines, start=1):
            match = SWIFT_IMPORT_RE.match(line)
            if match:
                module = match.group(1)
                refs.append(_Reference(source, module, "import", line_no, line.strip()))
        return refs

    def _extract_go_refs(self, source: _SourceItem) -> list[_Reference]:
        refs: list[_Reference] = []
        in_block = False
        for line_no, line in enumerate(source.lines, start=1):
            if in_block:
                if line.strip().startswith(")"):
                    in_block = False
                    continue
                match = GO_IMPORT_SPEC_RE.match(line)
                if match:
                    refs.append(
                        _Reference(source, match.group(1), "import", line_no, line.strip())
                    )
                continue

            match = GO_IMPORT_SINGLE_RE.match(line)
            if match:
                refs.append(
                    _Reference(source, match.group(1), "import", line_no, line.strip())
                )
                continue
            if GO_IMPORT_BLOCK_START_RE.match(line):
                in_block = True
        return refs

    def _extract_rust_refs(self, source: _SourceItem) -> list[_Reference]:
        refs: list[_Reference] = []
        for line_no, line in enumerate(source.lines, start=1):
            match = RUST_MOD_RE.match(line)
            if match:
                refs.append(
                    _Reference(source, match.group(1), "mod", line_no, line.strip())
                )
                continue
            match = RUST_USE_RE.match(line)
            if match:
                refs.append(
                    _Reference(source, match.group(1), "use", line_no, line.strip())
                )
                continue
            match = RUST_EXTERN_CRATE_RE.match(line)
            if match:
                refs.append(
                    _Reference(source, match.group(1), "extern", line_no, line.strip())
                )
        return refs

    def _extract_c_refs(self, source: _SourceItem) -> list[_Reference]:
        refs: list[_Reference] = []
        for line_no, line in enumerate(source.lines, start=1):
            match = C_INCLUDE_RE.match(line)
            if match:
                refs.append(
                    _Reference(source, match.group(1), "include", line_no, line.strip())
                )
        return refs

    def _extract_php_refs(self, source: _SourceItem) -> list[_Reference]:
        refs: list[_Reference] = []
        for line_no, line in enumerate(source.lines, start=1):
            match = PHP_INCLUDE_RE.match(line)
            if match:
                refs.append(
                    _Reference(source, match.group(2), "include", line_no, line.strip())
                )
        return refs

    def _extract_ruby_refs(self, source: _SourceItem) -> list[_Reference]:
        refs: list[_Reference] = []
        for line_no, line in enumerate(source.lines, start=1):
            match = RUBY_REQUIRE_RE.match(line)
            if match:
                refs.append(
                    _Reference(source, match.group(1), "require", line_no, line.strip())
                )
        return refs

    def _find_chunk_index(
        self, ranges: list[tuple[int, int, int]], line: int | None
    ) -> int | None:
        if line is None:
            return None
        for start, end, index in ranges:
            if start <= line <= end:
                return index
        return None

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
    ) -> tuple[list[CodeChunkConnected], bool]:
        chunks: list[CodeChunkConnected] = []
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
            chunk = CodeChunkConnected(
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
        if not isinstance(event.args, ChunkCodeConnectedMultiArgs):
            return ToolCallDisplay(summary="chunk_code_connected_multi")

        paths = event.args.paths or []
        snippets = event.args.snippets or []
        snippet_labels: list[str] = []
        for index, snippet in enumerate(snippets, start=1):
            label = (
                snippet.name.strip()
                if snippet.name and snippet.name.strip()
                else f"snippet:{index}"
            )
            if snippet.language:
                label = f"{label} ({snippet.language})"
            snippet_labels.append(label)

        summary = (
            f"chunk_code_connected_multi: {len(paths)} path(s), {len(snippets)} snippet(s)"
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
                "analyze_connections": event.args.analyze_connections,
                "max_connections": event.args.max_connections,
                "include_unresolved": event.args.include_unresolved,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ChunkCodeConnectedMultiResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = (
            f"Created {event.result.count} chunk(s), "
            f"{event.result.connection_count} connection(s)"
        )
        warnings = event.result.errors[:]
        if event.result.truncated:
            warnings.append("Chunk list truncated by size or max_chunks limits")
        if event.result.connections_truncated:
            warnings.append("Connections truncated by max_connections limits")

        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=warnings,
            details={
                "count": event.result.count,
                "connection_count": event.result.connection_count,
                "source_count": event.result.source_count,
                "truncated": event.result.truncated,
                "connections_truncated": event.result.connections_truncated,
                "errors": event.result.errors,
                "chunks": event.result.chunks,
                "connections": event.result.connections,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Chunking code and mapping connections"


def _resolve_relative_module(base_module: str | None, ref: str) -> str | None:
    if not base_module:
        return None
    dots = len(ref) - len(ref.lstrip("."))
    remainder = ref[dots:]
    base_parts = base_module.split(".")
    if dots > len(base_parts):
        return remainder or None
    prefix = base_parts[: len(base_parts) - dots]
    if remainder:
        return ".".join(prefix + remainder.split("."))
    return ".".join(prefix)
