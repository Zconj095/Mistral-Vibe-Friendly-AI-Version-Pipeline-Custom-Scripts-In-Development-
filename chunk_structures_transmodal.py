from __future__ import annotations

from dataclasses import dataclass
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
CLASS_KEYWORDS = {"class", "struct", "interface", "record"}
CLASS_PATTERN = re.compile(r"\b(class|struct|interface|record)\s+([A-Za-z_][A-Za-z0-9_]*)")
PY_BASE_PATTERN = re.compile(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)")
EXTENDS_PATTERN = re.compile(r"\bextends\s+([^\{]+)")
IMPLEMENTS_PATTERN = re.compile(r"\bimplements\s+([^\{]+)")
COLON_PATTERN = re.compile(r":\s*([^\{]+)")
BASE_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_\.]*")
GENERIC_RE = re.compile(r"<[^>]*>")

ATX_RE = re.compile(r"^\s*(#{1,6})\s*(.+?)\s*#*\s*$")
SETEXT_RE = re.compile(r"^\s*(=+|-+)\s*$")
NUMBERED_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:[.)])?\s+(.+)$")
FENCE_RE = re.compile(r"^\s*(```+|~~~+)")

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


@dataclass
class _ScanState:
    in_block_comment: bool = False
    in_string: str | None = None


@dataclass
class _ClassEntry:
    name: str
    kind: str
    start_line: int
    indent: int
    depth: int
    bases: list[str]
    start_depth: int | None = None


@dataclass
class _ClassSpan:
    name: str
    kind: str
    start_line: int
    end_line: int
    bases: list[str]
    content: str


@dataclass
class _CodeGroup:
    span_indices: list[int]
    class_names: list[str]
    start_line: int
    end_line: int
    max_depth: int
    line_span: int
    density: float
    density_per_100_lines: float
    content: str


@dataclass
class _HeadingEntry:
    title: str
    depth: int
    start_line: int
    end_line: int | None = None
    parent_index: int | None = None


@dataclass
class _TextGroup:
    root_index: int
    heading_indices: list[int]
    heading_titles: list[str]
    start_line: int
    end_line: int
    line_span: int
    max_depth: int
    density: float
    density_per_100_lines: float
    content: str


@dataclass
class _LinkCandidate:
    code_index: int
    text_index: int
    overlap: int
    score: float
    shared_tokens: list[str]


class ChunkStructuresTransmodalConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_input_bytes: int = Field(
        default=5_000_000, description="Maximum input size in bytes."
    )
    max_chunk_bytes: int = Field(
        default=500_000, description="Maximum structure chunk size in bytes."
    )
    max_code_chunks: int = Field(
        default=200, description="Maximum number of code structure chunks."
    )
    max_text_chunks: int = Field(
        default=200, description="Maximum number of text structure chunks."
    )
    default_code_mode: str = Field(
        default="auto", description="auto, indent, or brace for code."
    )
    code_include_nested: bool = Field(
        default=True, description="Include nested classes in code results."
    )
    code_include_singletons: bool = Field(
        default=False, description="Include single classes without links."
    )
    text_include_nested: bool = Field(
        default=True, description="Include nested headings in text results."
    )
    text_include_singletons: bool = Field(
        default=True, description="Include single-heading text sections."
    )
    include_markdown_headings: bool = Field(
        default=True, description="Detect Markdown # style headings."
    )
    include_setext_headings: bool = Field(
        default=True, description="Detect setext-style headings (---/===)."
    )
    include_numbered_headings: bool = Field(
        default=True, description="Detect numbered headings (e.g., 1.2 Title)."
    )
    default_class_keywords: list[str] = Field(
        default=sorted(CLASS_KEYWORDS),
        description="Keywords treated as class declarations.",
    )
    max_tokens_per_chunk: int = Field(
        default=500, description="Maximum tokens stored per chunk."
    )
    min_token_length: int = Field(default=3, description="Minimum token length.")
    min_shared_tokens: int = Field(
        default=1, description="Minimum shared tokens for a link."
    )
    max_shared_tokens: int = Field(
        default=10, description="Maximum shared tokens returned per link."
    )
    min_score: float = Field(default=0.0, description="Minimum link score.")
    score_mode: str = Field(default="overlap", description="overlap or jaccard.")
    max_links: int = Field(default=500, description="Maximum links returned.")
    max_links_per_code: int = Field(
        default=5, description="Maximum links per code chunk."
    )
    max_links_per_text: int = Field(
        default=5, description="Maximum links per text chunk."
    )


class ChunkStructuresTransmodalState(BaseToolState):
    pass


class ChunkStructuresTransmodalArgs(BaseModel):
    code_content: str | None = Field(default=None, description="Raw code content.")
    code_path: str | None = Field(default=None, description="Path to a code file.")
    code_language: str | None = Field(
        default=None, description="Language hint for code."
    )
    code_mode: str | None = Field(default=None, description="auto, indent, or brace.")
    code_include_nested: bool | None = Field(
        default=None, description="Include nested classes."
    )
    code_include_singletons: bool | None = Field(
        default=None, description="Include single classes without links."
    )
    class_keywords: list[str] | None = Field(
        default=None, description="Override class keywords to detect."
    )
    text_content: str | None = Field(default=None, description="Raw text content.")
    text_path: str | None = Field(default=None, description="Path to a text file.")
    text_include_nested: bool | None = Field(
        default=None, description="Include nested headings."
    )
    text_include_singletons: bool | None = Field(
        default=None, description="Include single-heading sections."
    )
    include_markdown_headings: bool | None = Field(
        default=None, description="Detect Markdown # headings."
    )
    include_setext_headings: bool | None = Field(
        default=None, description="Detect setext headings."
    )
    include_numbered_headings: bool | None = Field(
        default=None, description="Detect numbered headings."
    )
    max_code_chunks: int | None = Field(
        default=None, description="Override the configured code chunk limit."
    )
    max_text_chunks: int | None = Field(
        default=None, description="Override the configured text chunk limit."
    )
    max_tokens_per_chunk: int | None = Field(
        default=None, description="Override max tokens stored per chunk."
    )
    min_token_length: int | None = Field(
        default=None, description="Override min token length."
    )
    min_shared_tokens: int | None = Field(
        default=None, description="Minimum shared tokens for a link."
    )
    max_shared_tokens: int | None = Field(
        default=None, description="Maximum shared tokens per link."
    )
    min_score: float | None = Field(default=None, description="Minimum link score.")
    score_mode: str | None = Field(default=None, description="overlap or jaccard.")
    max_links: int | None = Field(
        default=None, description="Maximum number of links returned."
    )
    max_links_per_code: int | None = Field(
        default=None, description="Maximum links per code chunk."
    )
    max_links_per_text: int | None = Field(
        default=None, description="Maximum links per text chunk."
    )

class InheritanceClass(BaseModel):
    index: int
    name: str
    kind: str
    start_line: int
    end_line: int
    bases: list[str]
    depth: int
    group_index: int


class InheritanceChunk(BaseModel):
    index: int
    class_names: list[str]
    class_count: int
    start_line: int
    end_line: int
    line_span: int
    max_depth: int
    density: float
    density_per_100_lines: float
    content: str


class HeadingNode(BaseModel):
    index: int
    title: str
    depth: int
    relative_depth: int
    start_line: int
    end_line: int
    parent_index: int | None
    group_index: int


class TextStructureChunk(BaseModel):
    index: int
    heading_titles: list[str]
    heading_count: int
    start_line: int
    end_line: int
    line_span: int
    max_depth: int
    density: float
    density_per_100_lines: float
    content: str


class TransmodalLink(BaseModel):
    code_chunk_index: int
    text_chunk_index: int
    overlap: int
    score: float
    shared_tokens: list[str]


class ChunkStructuresTransmodalResult(BaseModel):
    code_mode: str | None
    code_include_nested: bool
    code_include_singletons: bool
    text_include_nested: bool
    text_include_singletons: bool
    score_mode: str
    code_chunks: list[InheritanceChunk]
    code_classes: list[InheritanceClass]
    text_chunks: list[TextStructureChunk]
    text_headings: list[HeadingNode]
    links: list[TransmodalLink]
    code_count: int
    text_count: int
    link_count: int
    code_truncated: bool
    text_truncated: bool
    links_truncated: bool


class ChunkStructuresTransmodal(
    BaseTool[
        ChunkStructuresTransmodalArgs,
        ChunkStructuresTransmodalResult,
        ChunkStructuresTransmodalConfig,
        ChunkStructuresTransmodalState,
    ],
    ToolUIData[ChunkStructuresTransmodalArgs, ChunkStructuresTransmodalResult],
):
    description: ClassVar[str] = (
        "Chunk code and text structures and build transmodal links."
    )

    async def run(
        self, args: ChunkStructuresTransmodalArgs
    ) -> ChunkStructuresTransmodalResult:
        code_content, code_path = self._load_code_content(args)
        text_content, text_path = self._load_text_content(args)

        if code_content is None and text_content is None:
            raise ToolError(
                "Provide code_content/code_path and/or text_content/text_path."
            )

        code_include_nested = self._resolve_flag(
            args.code_include_nested, self.config.code_include_nested
        )
        code_include_singletons = self._resolve_flag(
            args.code_include_singletons, self.config.code_include_singletons
        )
        text_include_nested = self._resolve_flag(
            args.text_include_nested, self.config.text_include_nested
        )
        text_include_singletons = self._resolve_flag(
            args.text_include_singletons, self.config.text_include_singletons
        )

        max_code_chunks = (
            args.max_code_chunks
            if args.max_code_chunks is not None
            else self.config.max_code_chunks
        )
        if max_code_chunks <= 0:
            raise ToolError("max_code_chunks must be a positive integer.")

        max_text_chunks = (
            args.max_text_chunks
            if args.max_text_chunks is not None
            else self.config.max_text_chunks
        )
        if max_text_chunks <= 0:
            raise ToolError("max_text_chunks must be a positive integer.")

        code_mode = None
        code_chunks: list[InheritanceChunk] = []
        code_classes: list[InheritanceClass] = []
        code_truncated = False

        if code_content is not None:
            if code_content:
                code_mode = self._resolve_code_mode(args, code_path)
                keywords = self._resolve_keywords(args)
                match code_mode:
                    case "indent":
                        spans = self._find_classes_indent(
                            code_content, keywords, code_include_nested
                        )
                    case "brace":
                        spans = self._find_classes_brace(
                            code_content, keywords, code_include_nested
                        )
                    case _:
                        raise ToolError("code_mode must be auto, indent, or brace.")

                groups, depth_map = self._build_code_groups(
                    spans, code_include_singletons
                )
                groups = sorted(groups, key=lambda item: (item.start_line, item.end_line))

                code_truncated = len(groups) > max_code_chunks
                if code_truncated:
                    groups = groups[:max_code_chunks]

                code_chunks, code_classes = self._finalize_code_groups(
                    groups, spans, depth_map
                )
                self._validate_code_chunk_sizes(code_chunks)
            else:
                if args.code_mode or args.code_language or code_path:
                    code_mode = self._resolve_code_mode(args, code_path)

        text_chunks: list[TextStructureChunk] = []
        text_headings: list[HeadingNode] = []
        text_truncated = False

        if text_content is not None:
            if text_content:
                lines = text_content.splitlines()
                headings = self._extract_headings(
                    lines,
                    include_markdown=self._resolve_flag(
                        args.include_markdown_headings,
                        self.config.include_markdown_headings,
                    ),
                    include_setext=self._resolve_flag(
                        args.include_setext_headings,
                        self.config.include_setext_headings,
                    ),
                    include_numbered=self._resolve_flag(
                        args.include_numbered_headings,
                        self.config.include_numbered_headings,
                    ),
                )

                if not headings:
                    if text_include_singletons:
                        chunk = self._build_singleton_chunk(lines)
                        self._validate_text_chunk_sizes([chunk])
                        text_chunks = [chunk]
                    else:
                        text_chunks = []
                    text_headings = []
                else:
                    last_line = len(lines)
                    self._assign_parents_and_ends(headings, last_line)
                    root_depth = min(entry.depth for entry in headings)

                    groups = self._build_text_groups(
                        headings,
                        lines,
                        root_depth,
                        include_nested=text_include_nested,
                        include_singletons=text_include_singletons,
                    )
                    groups = sorted(groups, key=lambda item: (item.start_line, item.end_line))

                    text_truncated = len(groups) > max_text_chunks
                    if text_truncated:
                        groups = groups[:max_text_chunks]

                    text_chunks, text_headings = self._finalize_text_groups(
                        groups, headings, root_depth
                    )
                    self._validate_text_chunk_sizes(text_chunks)

        score_mode = self._resolve_score_mode(args)
        max_tokens = (
            args.max_tokens_per_chunk
            if args.max_tokens_per_chunk is not None
            else self.config.max_tokens_per_chunk
        )
        min_token_length = (
            args.min_token_length
            if args.min_token_length is not None
            else self.config.min_token_length
        )
        min_shared_tokens = (
            args.min_shared_tokens
            if args.min_shared_tokens is not None
            else self.config.min_shared_tokens
        )
        max_shared_tokens = (
            args.max_shared_tokens
            if args.max_shared_tokens is not None
            else self.config.max_shared_tokens
        )
        min_score = args.min_score if args.min_score is not None else self.config.min_score
        max_links = args.max_links if args.max_links is not None else self.config.max_links
        max_links_per_code = (
            args.max_links_per_code
            if args.max_links_per_code is not None
            else self.config.max_links_per_code
        )
        max_links_per_text = (
            args.max_links_per_text
            if args.max_links_per_text is not None
            else self.config.max_links_per_text
        )

        if max_tokens < 0:
            raise ToolError("max_tokens_per_chunk must be >= 0.")
        if min_token_length < 0:
            raise ToolError("min_token_length must be >= 0.")
        if min_shared_tokens < 0:
            raise ToolError("min_shared_tokens must be >= 0.")
        if max_shared_tokens < 0:
            raise ToolError("max_shared_tokens must be >= 0.")
        if min_score < 0:
            raise ToolError("min_score must be >= 0.")
        if max_links < 0:
            raise ToolError("max_links must be >= 0.")
        if max_links_per_code < 0:
            raise ToolError("max_links_per_code must be >= 0.")
        if max_links_per_text < 0:
            raise ToolError("max_links_per_text must be >= 0.")

        links: list[TransmodalLink] = []
        links_truncated = False

        if code_chunks and text_chunks:
            code_tokens = [
                self._extract_tokens(chunk.content, min_token_length, max_tokens)
                for chunk in code_chunks
            ]
            text_tokens = [
                self._extract_tokens(chunk.content, min_token_length, max_tokens)
                for chunk in text_chunks
            ]
            links, links_truncated = self._build_links(
                code_tokens,
                text_tokens,
                min_shared_tokens=min_shared_tokens,
                max_shared_tokens=max_shared_tokens,
                min_score=min_score,
                score_mode=score_mode,
                max_links=max_links,
                max_links_per_code=max_links_per_code,
                max_links_per_text=max_links_per_text,
            )

        return ChunkStructuresTransmodalResult(
            code_mode=code_mode,
            code_include_nested=code_include_nested,
            code_include_singletons=code_include_singletons,
            text_include_nested=text_include_nested,
            text_include_singletons=text_include_singletons,
            score_mode=score_mode,
            code_chunks=code_chunks,
            code_classes=code_classes,
            text_chunks=text_chunks,
            text_headings=text_headings,
            links=links,
            code_count=len(code_chunks),
            text_count=len(text_chunks),
            link_count=len(links),
            code_truncated=code_truncated,
            text_truncated=text_truncated,
            links_truncated=links_truncated,
        )

    def _resolve_flag(self, value: bool | None, default: bool) -> bool:
        if value is None:
            return default
        return value

    def _resolve_score_mode(self, args: ChunkStructuresTransmodalArgs) -> str:
        mode = (args.score_mode or self.config.score_mode).strip().lower()
        if mode not in {"overlap", "jaccard"}:
            raise ToolError("score_mode must be overlap or jaccard.")
        return mode

    def _load_code_content(
        self, args: ChunkStructuresTransmodalArgs
    ) -> tuple[str | None, Path | None]:
        if args.code_content and args.code_path:
            raise ToolError("Provide code_content or code_path, not both.")
        if args.code_content is None and args.code_path is None:
            return None, None

        if args.code_content is not None:
            data = args.code_content.encode("utf-8")
            self._validate_input_size(len(data))
            return args.code_content, None

        path = self._resolve_path(args.code_path or "")
        size = path.stat().st_size
        self._validate_input_size(size)
        return path.read_text("utf-8", errors="ignore"), path

    def _load_text_content(
        self, args: ChunkStructuresTransmodalArgs
    ) -> tuple[str | None, Path | None]:
        if args.text_content and args.text_path:
            raise ToolError("Provide text_content or text_path, not both.")
        if args.text_content is None and args.text_path is None:
            return None, None

        if args.text_content is not None:
            data = args.text_content.encode("utf-8")
            self._validate_input_size(len(data))
            return args.text_content, None

        path = self._resolve_path(args.text_path or "")
        size = path.stat().st_size
        self._validate_input_size(size)
        return path.read_text("utf-8", errors="ignore"), path

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
        except OSError as exc:
            raise ToolError(f"Failed to resolve path: {exc}") from exc

        if not resolved.exists():
            raise ToolError(f"Path not found: {resolved}")
        if resolved.is_dir():
            raise ToolError(f"Path is a directory, not a file: {resolved}")
        return resolved

    def _resolve_code_mode(
        self, args: ChunkStructuresTransmodalArgs, source_path: Path | None
    ) -> str:
        mode = (args.code_mode or self.config.default_code_mode).strip().lower()
        if mode not in {"auto", "indent", "brace"}:
            raise ToolError("code_mode must be auto, indent, or brace.")
        if mode in {"indent", "brace"}:
            return mode

        language = (args.code_language or "").strip().lower()
        if language in INDENT_LANGS:
            return "indent"

        if source_path and source_path.suffix.lower() in INDENT_EXTS:
            return "indent"
        return "brace"

    def _resolve_keywords(self, args: ChunkStructuresTransmodalArgs) -> list[str]:
        raw = (
            args.class_keywords
            if args.class_keywords is not None
            else self.config.default_class_keywords
        )
        keywords = []
        for value in raw:
            if not value:
                continue
            cleaned = value.strip().lower()
            if cleaned:
                keywords.append(cleaned)
        if not keywords:
            raise ToolError("class_keywords must not be empty.")
        return sorted(set(keywords))

    def _validate_code_chunk_sizes(self, chunks: list[InheritanceChunk]) -> None:
        max_bytes = self.config.max_chunk_bytes
        for chunk in chunks:
            size = len(chunk.content.encode("utf-8"))
            if size > max_bytes:
                raise ToolError(
                    f"Chunk {chunk.index} exceeds max_chunk_bytes ({size} > {max_bytes})."
                )

    def _validate_text_chunk_sizes(self, chunks: list[TextStructureChunk]) -> None:
        max_bytes = self.config.max_chunk_bytes
        for chunk in chunks:
            size = len(chunk.content.encode("utf-8"))
            if size > max_bytes:
                raise ToolError(
                    f"Chunk {chunk.index} exceeds max_chunk_bytes ({size} > {max_bytes})."
                )

    def _find_classes_indent(
        self, content: str, keywords: list[str], include_nested: bool
    ) -> list[_ClassSpan]:
        lines = content.splitlines()
        state = _ScanState()
        spans: list[_ClassSpan] = []
        stack: list[_ClassEntry] = []
        keyword_set = set(keywords)

        for idx, line in enumerate(lines, start=1):
            sanitized, state = self._scan_line(line, state, hash_comments=True)
            stripped = sanitized.strip()
            indent = self._count_indent(line)
            is_significant = bool(stripped)

            if is_significant:
                while stack and indent <= stack[-1].indent:
                    entry = stack.pop()
                    if include_nested or entry.depth == 1:
                        spans.append(self._build_span(entry, lines, idx - 1))

            if not stripped:
                continue

            if match := self._detect_class(stripped, keyword_set):
                kind, name = match
                bases = self._extract_bases(stripped, name, mode="indent")
                depth = len(stack) + 1
                stack.append(
                    _ClassEntry(
                        name=name,
                        kind=kind,
                        start_line=idx,
                        indent=indent,
                        depth=depth,
                        bases=bases,
                    )
                )

        last_line = len(lines)
        while stack:
            entry = stack.pop()
            if include_nested or entry.depth == 1:
                spans.append(self._build_span(entry, lines, last_line))

        return spans

    def _find_classes_brace(
        self, content: str, keywords: list[str], include_nested: bool
    ) -> list[_ClassSpan]:
        lines = content.splitlines()
        state = _ScanState()
        spans: list[_ClassSpan] = []
        stack: list[_ClassEntry] = []
        pending: _ClassEntry | None = None
        keyword_set = set(keywords)
        brace_depth = 0

        for idx, line in enumerate(lines, start=1):
            sanitized, state = self._scan_line(line, state, hash_comments=False)
            stripped = sanitized.strip()
            if stripped and not pending:
                if match := self._detect_class(stripped, keyword_set):
                    kind, name = match
                    bases = self._extract_bases(stripped, name, mode="brace")
                    pending = _ClassEntry(
                        name=name,
                        kind=kind,
                        start_line=idx,
                        indent=0,
                        depth=len(stack) + 1,
                        bases=bases,
                    )

            for ch in sanitized:
                if ch == "{":
                    if pending:
                        brace_depth += 1
                        pending.start_depth = brace_depth
                        stack.append(pending)
                        pending = None
                        continue
                    brace_depth += 1
                elif ch == "}":
                    brace_depth = max(brace_depth - 1, 0)
                    while stack and stack[-1].start_depth is not None:
                        if brace_depth < (stack[-1].start_depth or 0):
                            entry = stack.pop()
                            if include_nested or entry.depth == 1:
                                spans.append(self._build_span(entry, lines, idx))
                            continue
                        break
                elif ch == ";" and pending:
                    pending = None

        last_line = len(lines)
        while stack:
            entry = stack.pop()
            if include_nested or entry.depth == 1:
                spans.append(self._build_span(entry, lines, last_line))

        return spans

    def _build_span(
        self, entry: _ClassEntry, lines: list[str], end_line: int
    ) -> _ClassSpan:
        start_line = entry.start_line
        end_line = max(end_line, start_line)
        content = "\n".join(lines[start_line - 1 : end_line])
        return _ClassSpan(
            name=entry.name,
            kind=entry.kind,
            start_line=start_line,
            end_line=end_line,
            bases=entry.bases,
            content=content,
        )

    def _detect_class(
        self, line: str, keywords: set[str]
    ) -> tuple[str, str] | None:
        matches = list(CLASS_PATTERN.finditer(line))
        if not matches:
            return None
        match = matches[-1]
        kind = match.group(1).lower()
        name = match.group(2)
        if kind not in keywords:
            return None
        return kind, name

    def _extract_bases(self, line: str, name: str, mode: str) -> list[str]:
        if mode == "indent":
            return self._extract_python_bases(line, name)
        return self._extract_brace_bases(line)

    def _extract_python_bases(self, line: str, name: str) -> list[str]:
        match = PY_BASE_PATTERN.search(line)
        if not match:
            return []
        if match.group(1) != name:
            return []
        return self._split_bases(match.group(2))

    def _extract_brace_bases(self, line: str) -> list[str]:
        bases: list[str] = []
        if match := EXTENDS_PATTERN.search(line):
            bases.extend(self._split_bases(match.group(1)))
        if match := IMPLEMENTS_PATTERN.search(line):
            bases.extend(self._split_bases(match.group(1)))
        if not bases and ":" in line:
            if match := COLON_PATTERN.search(line):
                bases.extend(self._split_bases(match.group(1)))

        ordered: list[str] = []
        seen: set[str] = set()
        for base in bases:
            if base not in seen:
                ordered.append(base)
                seen.add(base)
        return ordered

    def _split_bases(self, raw: str) -> list[str]:
        cleaned = GENERIC_RE.sub("", raw)
        cleaned = cleaned.replace("::", ".")
        parts = [part.strip() for part in cleaned.split(",") if part.strip()]
        bases: list[str] = []
        for part in parts:
            tokens = BASE_TOKEN_RE.findall(part)
            if not tokens:
                continue
            base = tokens[0]
            base = base.split(".")[-1]
            if base and base not in bases:
                bases.append(base)
        return bases

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

    def _build_code_groups(
        self, spans: list[_ClassSpan], include_singletons: bool
    ) -> tuple[list[_CodeGroup], dict[str, int]]:
        if not spans:
            return [], {}

        name_to_indices: dict[str, list[int]] = {}
        for idx, span in enumerate(spans):
            name_to_indices.setdefault(span.name, []).append(idx)

        bases_map: dict[str, set[str]] = {name: set() for name in name_to_indices}
        for span in spans:
            for base in span.bases:
                if base in name_to_indices:
                    bases_map.setdefault(span.name, set()).add(base)

        depth_map = self._compute_depths(bases_map)

        adjacency: dict[int, set[int]] = {idx: set() for idx in range(len(spans))}
        for idx, span in enumerate(spans):
            for base in span.bases:
                for base_idx in name_to_indices.get(base, []):
                    adjacency[idx].add(base_idx)
                    adjacency[base_idx].add(idx)

        visited: set[int] = set()
        groups: list[_CodeGroup] = []
        for idx in range(len(spans)):
            if idx in visited:
                continue
            stack = [idx]
            component: list[int] = []
            visited.add(idx)
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)

            if not include_singletons and len(component) < 2:
                continue

            component = sorted(component, key=lambda item: spans[item].start_line)
            class_names = [spans[item].name for item in component]
            start_line = min(spans[item].start_line for item in component)
            end_line = max(spans[item].end_line for item in component)
            max_depth = max(depth_map.get(spans[item].name, 1) for item in component)
            line_span = max(end_line - start_line + 1, 1)
            class_count = len(component)
            density = class_count / line_span
            density_per_100 = density * 100.0
            content = "\n\n".join(spans[item].content for item in component)

            groups.append(
                _CodeGroup(
                    span_indices=component,
                    class_names=class_names,
                    start_line=start_line,
                    end_line=end_line,
                    max_depth=max_depth,
                    line_span=line_span,
                    density=density,
                    density_per_100_lines=density_per_100,
                    content=content,
                )
            )

        return groups, depth_map

    def _compute_depths(self, bases_map: dict[str, set[str]]) -> dict[str, int]:
        memo: dict[str, int] = {}

        def depth(name: str, stack: set[str]) -> int:
            if name in memo:
                return memo[name]
            if name in stack:
                return 1
            stack.add(name)
            bases = bases_map.get(name) or set()
            if not bases:
                memo[name] = 1
            else:
                memo[name] = 1 + max(depth(base, stack) for base in bases)
            stack.remove(name)
            return memo[name]

        for name in bases_map:
            depth(name, set())
        return memo

    def _finalize_code_groups(
        self,
        groups: list[_CodeGroup],
        spans: list[_ClassSpan],
        depth_map: dict[str, int],
    ) -> tuple[list[InheritanceChunk], list[InheritanceClass]]:
        chunks: list[InheritanceChunk] = []
        classes: list[InheritanceClass] = []

        for index, group in enumerate(groups, start=1):
            chunk = InheritanceChunk(
                index=index,
                class_names=group.class_names,
                class_count=len(group.class_names),
                start_line=group.start_line,
                end_line=group.end_line,
                line_span=group.line_span,
                max_depth=group.max_depth,
                density=group.density,
                density_per_100_lines=group.density_per_100_lines,
                content=group.content,
            )
            chunks.append(chunk)

            for span_idx in group.span_indices:
                span = spans[span_idx]
                classes.append(
                    InheritanceClass(
                        index=0,
                        name=span.name,
                        kind=span.kind,
                        start_line=span.start_line,
                        end_line=span.end_line,
                        bases=span.bases,
                        depth=depth_map.get(span.name, 1),
                        group_index=index,
                    )
                )

        ordered = sorted(
            classes, key=lambda item: (item.start_line, item.end_line, item.name)
        )
        classes = [
            InheritanceClass(
                index=idx + 1,
                name=item.name,
                kind=item.kind,
                start_line=item.start_line,
                end_line=item.end_line,
                bases=item.bases,
                depth=item.depth,
                group_index=item.group_index,
            )
            for idx, item in enumerate(ordered)
        ]

        return chunks, classes

    def _extract_headings(
        self,
        lines: list[str],
        *,
        include_markdown: bool,
        include_setext: bool,
        include_numbered: bool,
    ) -> list[_HeadingEntry]:
        skip_lines = self._mark_fence_lines(lines)
        headings: list[_HeadingEntry] = []
        heading_lines: set[int] = set()

        for idx, line in enumerate(lines):
            if skip_lines[idx]:
                continue

            raw = line.rstrip("\r\n")
            if include_setext:
                if match := SETEXT_RE.match(raw):
                    prev_index = idx - 1
                    if prev_index >= 0 and not skip_lines[prev_index]:
                        title = lines[prev_index].strip()
                        if title and (prev_index + 1) not in heading_lines:
                            depth = 1 if match.group(1).startswith("=") else 2
                            headings.append(
                                _HeadingEntry(
                                    title=title,
                                    depth=depth,
                                    start_line=prev_index + 1,
                                )
                            )
                            heading_lines.add(prev_index + 1)
                    continue

            if include_markdown:
                if match := ATX_RE.match(raw):
                    title = match.group(2).strip()
                    if title and (idx + 1) not in heading_lines:
                        depth = len(match.group(1))
                        headings.append(
                            _HeadingEntry(title=title, depth=depth, start_line=idx + 1)
                        )
                        heading_lines.add(idx + 1)
                    continue

            if include_numbered:
                if match := NUMBERED_RE.match(raw):
                    title = match.group(2).strip()
                    if title and (idx + 1) not in heading_lines:
                        depth = len(match.group(1).split("."))
                        headings.append(
                            _HeadingEntry(title=title, depth=depth, start_line=idx + 1)
                        )
                        heading_lines.add(idx + 1)

        headings.sort(key=lambda item: item.start_line)
        return headings

    def _mark_fence_lines(self, lines: list[str]) -> list[bool]:
        skip = [False] * len(lines)
        in_fence = False
        fence_char: str | None = None

        for idx, line in enumerate(lines):
            if match := FENCE_RE.match(line):
                skip[idx] = True
                marker = match.group(1)
                marker_char = marker[0]
                if not in_fence:
                    in_fence = True
                    fence_char = marker_char
                elif fence_char == marker_char:
                    in_fence = False
                    fence_char = None
                continue

            if in_fence:
                skip[idx] = True

        return skip

    def _assign_parents_and_ends(self, headings: list[_HeadingEntry], last_line: int) -> None:
        stack: list[int] = []
        for idx, heading in enumerate(headings):
            while stack and heading.depth <= headings[stack[-1]].depth:
                prev = stack.pop()
                headings[prev].end_line = heading.start_line - 1

            heading.parent_index = stack[-1] if stack else None
            stack.append(idx)

        while stack:
            prev = stack.pop()
            headings[prev].end_line = last_line

    def _build_text_groups(
        self,
        headings: list[_HeadingEntry],
        lines: list[str],
        root_depth: int,
        *,
        include_nested: bool,
        include_singletons: bool,
    ) -> list[_TextGroup]:
        groups: list[_TextGroup] = []
        last_line = len(lines)
        root_indices = [
            idx for idx, entry in enumerate(headings) if entry.depth == root_depth
        ]

        for root_idx in root_indices:
            root = headings[root_idx]
            start_line = root.start_line
            end_line = root.end_line or last_line
            heading_indices = [
                idx
                for idx, entry in enumerate(headings)
                if start_line <= entry.start_line <= end_line
                and (include_nested or entry.depth == root_depth)
            ]

            if not include_singletons and len(heading_indices) < 2:
                continue

            heading_titles = [headings[idx].title for idx in heading_indices]
            line_span = max(end_line - start_line + 1, 1)

            if include_nested and heading_indices:
                max_depth = max(
                    headings[idx].depth - root_depth + 1 for idx in heading_indices
                )
            elif heading_indices:
                max_depth = 1
            else:
                max_depth = 0

            density = len(heading_indices) / line_span
            density_per_100 = density * 100.0
            content = "\n".join(lines[start_line - 1 : end_line])

            groups.append(
                _TextGroup(
                    root_index=root_idx,
                    heading_indices=heading_indices,
                    heading_titles=heading_titles,
                    start_line=start_line,
                    end_line=end_line,
                    line_span=line_span,
                    max_depth=max_depth,
                    density=density,
                    density_per_100_lines=density_per_100,
                    content=content,
                )
            )

        return groups

    def _finalize_text_groups(
        self,
        groups: list[_TextGroup],
        headings: list[_HeadingEntry],
        root_depth: int,
    ) -> tuple[list[TextStructureChunk], list[HeadingNode]]:
        chunks: list[TextStructureChunk] = []
        heading_group: dict[int, int] = {}
        included_indices: list[int] = []

        for group_index, group in enumerate(groups, start=1):
            chunks.append(
                TextStructureChunk(
                    index=group_index,
                    heading_titles=group.heading_titles,
                    heading_count=len(group.heading_titles),
                    start_line=group.start_line,
                    end_line=group.end_line,
                    line_span=group.line_span,
                    max_depth=group.max_depth,
                    density=group.density,
                    density_per_100_lines=group.density_per_100_lines,
                    content=group.content,
                )
            )
            for idx in group.heading_indices:
                heading_group[idx] = group_index
            included_indices.extend(group.heading_indices)

        unique_indices = sorted(
            set(included_indices), key=lambda idx: headings[idx].start_line
        )
        index_map = {
            old_idx: new_idx for new_idx, old_idx in enumerate(unique_indices, start=1)
        }

        nodes: list[HeadingNode] = []
        for old_idx in unique_indices:
            entry = headings[old_idx]
            parent_old = entry.parent_index
            parent_index = index_map.get(parent_old) if parent_old in index_map else None
            group_index = heading_group.get(old_idx)
            if group_index is None:
                continue

            relative_depth = entry.depth - root_depth + 1
            nodes.append(
                HeadingNode(
                    index=index_map[old_idx],
                    title=entry.title,
                    depth=entry.depth,
                    relative_depth=relative_depth,
                    start_line=entry.start_line,
                    end_line=entry.end_line or entry.start_line,
                    parent_index=parent_index,
                    group_index=group_index,
                )
            )

        return chunks, nodes

    def _build_singleton_chunk(self, lines: list[str]) -> TextStructureChunk:
        if not lines:
            return TextStructureChunk(
                index=1,
                heading_titles=[],
                heading_count=0,
                start_line=1,
                end_line=1,
                line_span=1,
                max_depth=0,
                density=0.0,
                density_per_100_lines=0.0,
                content="",
            )

        start_line = 1
        end_line = len(lines)
        line_span = max(end_line - start_line + 1, 1)
        content = "\n".join(lines)
        return TextStructureChunk(
            index=1,
            heading_titles=[],
            heading_count=0,
            start_line=start_line,
            end_line=end_line,
            line_span=line_span,
            max_depth=0,
            density=0.0,
            density_per_100_lines=0.0,
            content=content,
        )

    def _extract_tokens(
        self, content: str, min_len: int, max_tokens: int
    ) -> set[str]:
        tokens: dict[str, int] = {}
        for match in TOKEN_RE.findall(content.lower()):
            if len(match) < min_len:
                continue
            if match.isdigit():
                continue
            if match in STOPWORDS:
                continue
            tokens[match] = tokens.get(match, 0) + 1

        if not tokens:
            return set()

        sorted_tokens = sorted(tokens.items(), key=lambda item: (-item[1], item[0]))
        if max_tokens > 0:
            sorted_tokens = sorted_tokens[:max_tokens]
        return {token for token, _ in sorted_tokens}

    def _build_links(
        self,
        code_tokens: list[set[str]],
        text_tokens: list[set[str]],
        *,
        min_shared_tokens: int,
        max_shared_tokens: int,
        min_score: float,
        score_mode: str,
        max_links: int,
        max_links_per_code: int,
        max_links_per_text: int,
    ) -> tuple[list[TransmodalLink], bool]:
        if not code_tokens or not text_tokens:
            return [], False

        token_map: dict[str, list[int]] = {}
        for idx, tokens in enumerate(text_tokens):
            for token in tokens:
                token_map.setdefault(token, []).append(idx)

        candidates: list[_LinkCandidate] = []
        for code_idx, tokens in enumerate(code_tokens):
            if not tokens:
                continue
            overlap_counts: dict[int, int] = {}
            for token in tokens:
                for text_idx in token_map.get(token, []):
                    overlap_counts[text_idx] = overlap_counts.get(text_idx, 0) + 1

            for text_idx, overlap in overlap_counts.items():
                if overlap < min_shared_tokens:
                    continue
                text_set = text_tokens[text_idx]
                if score_mode == "jaccard":
                    union = len(tokens) + len(text_set) - overlap
                    score = overlap / union if union else 0.0
                else:
                    score = float(overlap)
                if score < min_score:
                    continue
                shared = sorted(tokens & text_set)
                if max_shared_tokens > 0 and len(shared) > max_shared_tokens:
                    shared = shared[:max_shared_tokens]
                candidates.append(
                    _LinkCandidate(
                        code_index=code_idx + 1,
                        text_index=text_idx + 1,
                        overlap=overlap,
                        score=score,
                        shared_tokens=shared,
                    )
                )

        candidates.sort(
            key=lambda item: (-item.score, -item.overlap, item.code_index, item.text_index)
        )

        selected: list[TransmodalLink] = []
        counts_code: dict[int, int] = {}
        counts_text: dict[int, int] = {}
        for candidate in candidates:
            if max_links > 0 and len(selected) >= max_links:
                break
            if max_links_per_code > 0 and counts_code.get(candidate.code_index, 0) >= max_links_per_code:
                continue
            if max_links_per_text > 0 and counts_text.get(candidate.text_index, 0) >= max_links_per_text:
                continue
            selected.append(
                TransmodalLink(
                    code_chunk_index=candidate.code_index,
                    text_chunk_index=candidate.text_index,
                    overlap=candidate.overlap,
                    score=round(candidate.score, 6),
                    shared_tokens=candidate.shared_tokens,
                )
            )
            counts_code[candidate.code_index] = counts_code.get(candidate.code_index, 0) + 1
            counts_text[candidate.text_index] = counts_text.get(candidate.text_index, 0) + 1

        limit_active = any(
            value > 0 for value in (max_links, max_links_per_code, max_links_per_text)
        )
        truncated = limit_active and len(selected) < len(candidates)
        return selected, truncated

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ChunkStructuresTransmodalArgs):
            return ToolCallDisplay(summary="chunk_structures_transmodal")

        return ToolCallDisplay(
            summary="chunk_structures_transmodal",
            details={
                "code_path": event.args.code_path,
                "code_language": event.args.code_language,
                "code_mode": event.args.code_mode,
                "code_include_nested": event.args.code_include_nested,
                "code_include_singletons": event.args.code_include_singletons,
                "class_keywords": event.args.class_keywords,
                "text_path": event.args.text_path,
                "text_include_nested": event.args.text_include_nested,
                "text_include_singletons": event.args.text_include_singletons,
                "include_markdown_headings": event.args.include_markdown_headings,
                "include_setext_headings": event.args.include_setext_headings,
                "include_numbered_headings": event.args.include_numbered_headings,
                "max_code_chunks": event.args.max_code_chunks,
                "max_text_chunks": event.args.max_text_chunks,
                "max_tokens_per_chunk": event.args.max_tokens_per_chunk,
                "min_token_length": event.args.min_token_length,
                "min_shared_tokens": event.args.min_shared_tokens,
                "max_shared_tokens": event.args.max_shared_tokens,
                "min_score": event.args.min_score,
                "score_mode": event.args.score_mode,
                "max_links": event.args.max_links,
                "max_links_per_code": event.args.max_links_per_code,
                "max_links_per_text": event.args.max_links_per_text,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ChunkStructuresTransmodalResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = (
            f"Built {event.result.code_count} code chunk(s), "
            f"{event.result.text_count} text chunk(s), "
            f"{event.result.link_count} link(s)"
        )
        warnings: list[str] = []
        if event.result.code_truncated:
            warnings.append("Code chunk list truncated by max_code_chunks limit")
        if event.result.text_truncated:
            warnings.append("Text chunk list truncated by max_text_chunks limit")
        if event.result.links_truncated:
            warnings.append("Link list truncated by link limits")

        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=warnings,
            details={
                "code_mode": event.result.code_mode,
                "code_include_nested": event.result.code_include_nested,
                "code_include_singletons": event.result.code_include_singletons,
                "text_include_nested": event.result.text_include_nested,
                "text_include_singletons": event.result.text_include_singletons,
                "score_mode": event.result.score_mode,
                "code_count": event.result.code_count,
                "text_count": event.result.text_count,
                "link_count": event.result.link_count,
                "code_truncated": event.result.code_truncated,
                "text_truncated": event.result.text_truncated,
                "links_truncated": event.result.links_truncated,
                "code_chunks": event.result.code_chunks,
                "code_classes": event.result.code_classes,
                "text_chunks": event.result.text_chunks,
                "text_headings": event.result.text_headings,
                "links": event.result.links,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Chunking transmodal structures"
