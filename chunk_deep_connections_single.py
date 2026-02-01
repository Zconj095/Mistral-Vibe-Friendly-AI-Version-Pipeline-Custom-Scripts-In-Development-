from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
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
CODE_EXTS = {
    ".py",
    ".pyi",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
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
    ".kt",
    ".m",
    ".mm",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
}
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


class ChunkDeepConnectionsSingleConfig(BaseToolConfig):
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
    default_overlap: int = Field(default=0, description="Overlap in units.")
    default_split_depth: int = Field(
        default=0, description="Depth at or below which splitting is allowed."
    )
    default_code_mode: str = Field(default="auto", description="auto, indent, brace.")
    default_kind: str = Field(default="auto", description="auto, code, document.")
    respect_brackets: bool = Field(
        default=True,
        description="Avoid splitting while inside bracketed data structures.",
    )
    max_tokens_per_chunk: int = Field(
        default=700, description="Maximum tokens stored per chunk."
    )
    min_token_length: int = Field(
        default=3, description="Minimum token length."
    )
    max_df_ratio: float = Field(
        default=0.75,
        description="Ignore tokens that appear in more than this ratio of chunks.",
    )
    min_similarity: float = Field(
        default=0.08, description="Minimum similarity for direct edges."
    )
    max_neighbors: int = Field(
        default=6, description="Maximum direct neighbors per chunk."
    )
    max_depth: int = Field(
        default=3, description="Maximum depth for deep connections."
    )
    min_deep_score: float = Field(
        default=0.03, description="Minimum score for deep connections."
    )
    max_deep_connections: int = Field(
        default=6, description="Maximum deep connections per chunk."
    )
    max_shared_tokens: int = Field(
        default=8, description="Maximum shared tokens returned per connection."
    )
    default_code_extensions: list[str] = Field(
        default=sorted(CODE_EXTS),
        description="Extensions treated as code for auto detection.",
    )


class ChunkDeepConnectionsSingleState(BaseToolState):
    pass


class ChunkDeepConnectionsSingleArgs(BaseModel):
    content: str | None = Field(default=None, description="Content to chunk.")
    path: str | None = Field(default=None, description="Path to a file.")
    kind: str | None = Field(default=None, description="auto, code, or document.")
    language: str | None = Field(default=None, description="Language hint for code.")
    code_mode: str | None = Field(default=None, description="auto, indent, brace.")
    unit: str | None = Field(default=None, description="lines or chars.")
    size: int | None = Field(default=None, description="Chunk size in units.")
    overlap: int | None = Field(default=None, description="Overlap in units.")
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
    max_depth: int | None = Field(
        default=None, description="Override max depth for deep connections."
    )
    min_similarity: float | None = Field(
        default=None, description="Override minimum similarity for direct edges."
    )
    min_deep_score: float | None = Field(
        default=None, description="Override minimum deep connection score."
    )
    max_neighbors: int | None = Field(
        default=None, description="Override maximum direct neighbors."
    )
    max_deep_connections: int | None = Field(
        default=None, description="Override maximum deep connections."
    )
    max_shared_tokens: int | None = Field(
        default=None, description="Override maximum shared tokens per connection."
    )


class CrossrefChunk(BaseModel):
    index: int
    kind: str
    unit: str
    start: int | None
    end: int | None
    start_line: int | None
    end_line: int | None
    min_depth: int | None
    max_depth: int | None
    content: str


class DocumentChunk(BaseModel):
    chunk_index: int
    unit: str
    start: int
    end: int
    content: str


class CodeChunkData(BaseModel):
    index: int
    start_line: int
    end_line: int
    min_depth: int
    max_depth: int
    unit: str
    content: str


class DirectConnection(BaseModel):
    source_index: int
    target_index: int
    score: float
    shared_tokens: list[str]


class DeepConnection(BaseModel):
    source_index: int
    target_index: int
    score: float
    depth: int
    path: list[int]


class ChunkDeepConnectionsSingleResult(BaseModel):
    kind: str
    unit: str
    code_mode: str | None
    chunks: list[CrossrefChunk]
    connections: list[DirectConnection]
    deep_connections: list[DeepConnection]
    count: int
    connection_count: int
    deep_connection_count: int
    truncated: bool


class ChunkDeepConnectionsSingle(
    BaseTool[
        ChunkDeepConnectionsSingleArgs,
        ChunkDeepConnectionsSingleResult,
        ChunkDeepConnectionsSingleConfig,
        ChunkDeepConnectionsSingleState,
    ],
    ToolUIData[ChunkDeepConnectionsSingleArgs, ChunkDeepConnectionsSingleResult],
):
    description: ClassVar[str] = (
        "Chunk a single file and build deep multi-hop connections between chunks."
    )

    async def run(
        self, args: ChunkDeepConnectionsSingleArgs
    ) -> ChunkDeepConnectionsSingleResult:
        content, source_path = self._load_content(args)
        if not content:
            return ChunkDeepConnectionsSingleResult(
                kind="document",
                unit=self.config.default_unit,
                code_mode=None,
                chunks=[],
                connections=[],
                deep_connections=[],
                count=0,
                connection_count=0,
                deep_connection_count=0,
                truncated=False,
            )

        kind = self._resolve_kind(args, source_path)
        unit = (args.unit or self.config.default_unit).strip().lower()
        size = args.size if args.size is not None else self.config.default_chunk_size
        overlap = args.overlap if args.overlap is not None else self.config.default_overlap
        split_depth = (
            args.split_depth
            if args.split_depth is not None
            else self.config.default_split_depth
        )
        if unit not in {"lines", "chars"}:
            raise ToolError("unit must be lines or chars.")
        if size <= 0:
            raise ToolError("size must be a positive integer.")
        if overlap < 0:
            raise ToolError("overlap must be a non-negative integer.")
        if overlap >= size:
            raise ToolError("overlap must be smaller than size.")
        if split_depth < 0:
            raise ToolError("split_depth must be >= 0.")

        max_chunks = args.max_chunks if args.max_chunks is not None else self.config.max_chunks
        if max_chunks <= 0:
            raise ToolError("max_chunks must be a positive integer.")

        max_depth = args.max_depth if args.max_depth is not None else self.config.max_depth
        if max_depth <= 0:
            raise ToolError("max_depth must be a positive integer.")

        min_similarity = (
            args.min_similarity
            if args.min_similarity is not None
            else self.config.min_similarity
        )
        min_deep_score = (
            args.min_deep_score
            if args.min_deep_score is not None
            else self.config.min_deep_score
        )
        max_neighbors = (
            args.max_neighbors
            if args.max_neighbors is not None
            else self.config.max_neighbors
        )
        max_deep_connections = (
            args.max_deep_connections
            if args.max_deep_connections is not None
            else self.config.max_deep_connections
        )
        max_shared_tokens = (
            args.max_shared_tokens
            if args.max_shared_tokens is not None
            else self.config.max_shared_tokens
        )

        if min_similarity < 0:
            raise ToolError("min_similarity must be >= 0.")
        if min_deep_score < 0:
            raise ToolError("min_deep_score must be >= 0.")
        if max_neighbors < 0:
            raise ToolError("max_neighbors must be >= 0.")
        if max_deep_connections < 0:
            raise ToolError("max_deep_connections must be >= 0.")
        if max_shared_tokens < 0:
            raise ToolError("max_shared_tokens must be >= 0.")

        respect_brackets = (
            args.respect_brackets
            if args.respect_brackets is not None
            else self.config.respect_brackets
        )

        code_mode = None
        if kind == "code":
            code_mode = self._normalize_code_mode(args)
            if unit == "chars" and size > self.config.max_chunk_bytes:
                raise ToolError("size exceeds max_chunk_bytes for char-based chunking.")

        if kind == "code":
            chunks, truncated = self._chunk_code(
                content,
                source_path,
                args,
                code_mode,
                unit,
                size,
                split_depth,
                respect_brackets,
                max_chunks,
            )
        else:
            chunks, truncated = self._chunk_document(
                content,
                unit,
                size,
                overlap,
                max_chunks,
            )

        token_maps = [self._extract_tokens(chunk.content) for chunk in chunks]
        connections = self._build_direct_connections(
            token_maps, min_similarity, max_neighbors, max_shared_tokens
        )
        adjacency = self._connections_to_adjacency(connections, len(chunks))
        deep_connections = self._build_deep_connections(
            adjacency, max_depth, min_deep_score, max_deep_connections
        )

        return ChunkDeepConnectionsSingleResult(
            kind=kind,
            unit=unit,
            code_mode=code_mode,
            chunks=chunks,
            connections=connections,
            deep_connections=deep_connections,
            count=len(chunks),
            connection_count=len(connections),
            deep_connection_count=len(deep_connections),
            truncated=truncated,
        )

    def _load_content(
        self, args: ChunkDeepConnectionsSingleArgs
    ) -> tuple[str, Path | None]:
        if args.content and args.path:
            raise ToolError("Provide either content or path, not both.")
        if args.content is None and args.path is None:
            raise ToolError("Provide content or path.")

        if args.content is not None:
            data = args.content.encode("utf-8")
            self._validate_input_size(len(data))
            return args.content, None

        path = self._resolve_path(args.path or "")
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
        return resolved

    def _resolve_kind(
        self, args: ChunkDeepConnectionsSingleArgs, source_path: Path | None
    ) -> str:
        kind = (args.kind or self.config.default_kind).strip().lower()
        if kind not in {"auto", "code", "document"}:
            raise ToolError("kind must be auto, code, or document.")

        if kind != "auto":
            return kind

        if args.language:
            return "code"

        if source_path is not None:
            ext = source_path.suffix.lower()
            if ext in set(self.config.default_code_extensions):
                return "code"
            if ext in CODE_EXTS:
                return "code"

        return "document"

    def _normalize_code_mode(self, args: ChunkDeepConnectionsSingleArgs) -> str:
        mode = (args.code_mode or self.config.default_code_mode).strip().lower()
        if mode not in {"auto", "indent", "brace"}:
            raise ToolError("code_mode must be auto, indent, or brace.")
        return mode

    def _resolve_language(
        self, args: ChunkDeepConnectionsSingleArgs, source_path: Path | None
    ) -> str | None:
        if args.language:
            return args.language.strip().lower()
        if source_path is None:
            return None
        ext = source_path.suffix.lower().lstrip(".")
        return ext if ext else None

    def _resolve_code_mode(
        self, mode: str, language: str | None, source_path: Path | None
    ) -> str:
        if mode != "auto":
            return mode
        if language and language in INDENT_LANGS:
            return "indent"
        if source_path and source_path.suffix.lower() in INDENT_EXTS:
            return "indent"
        return "brace"

    def _chunk_code(
        self,
        content: str,
        source_path: Path | None,
        args: ChunkDeepConnectionsSingleArgs,
        mode: str,
        unit: str,
        size: int,
        split_depth: int,
        respect_brackets: bool,
        max_chunks: int,
    ) -> tuple[list[CrossrefChunk], bool]:
        language = self._resolve_language(args, source_path)
        resolved_mode = self._resolve_code_mode(mode, language, source_path)
        lines = content.splitlines()
        boundaries, depths = self._compute_boundaries(
            lines, resolved_mode, split_depth, respect_brackets
        )
        chunks, truncated = self._build_code_chunks(
            lines,
            boundaries,
            depths,
            unit,
            size,
            max_chunks,
            args.hard_split,
        )
        return (
            [
                CrossrefChunk(
                    index=chunk.index,
                    kind="code",
                    unit=chunk.unit,
                    start=None,
                    end=None,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    min_depth=chunk.min_depth,
                    max_depth=chunk.max_depth,
                    content=chunk.content,
                )
                for chunk in chunks
            ],
            truncated,
        )

    def _chunk_document(
        self,
        content: str,
        unit: str,
        size: int,
        overlap: int,
        max_chunks: int,
    ) -> tuple[list[CrossrefChunk], bool]:
        if unit == "lines":
            chunks, truncated = self._chunk_lines(content, size, overlap, max_chunks)
        else:
            chunks, truncated = self._chunk_chars(content, size, overlap, max_chunks)

        crossref_chunks: list[CrossrefChunk] = []
        for chunk in chunks:
            start_line = chunk.start if chunk.unit == "lines" else None
            end_line = chunk.end if chunk.unit == "lines" else None
            crossref_chunks.append(
                CrossrefChunk(
                    index=chunk.chunk_index,
                    kind="document",
                    unit=chunk.unit,
                    start=chunk.start if chunk.unit == "chars" else None,
                    end=chunk.end if chunk.unit == "chars" else None,
                    start_line=start_line,
                    end_line=end_line,
                    min_depth=None,
                    max_depth=None,
                    content=chunk.content,
                )
            )

        return crossref_chunks, truncated

    def _chunk_lines(
        self, content: str, size: int, overlap: int, max_chunks: int
    ) -> tuple[list[DocumentChunk], bool]:
        lines = content.splitlines()
        if not lines:
            return [], False
        step = size - overlap
        index = 0
        chunk_index = 0
        chunks: list[DocumentChunk] = []
        while index < len(lines) and chunk_index < max_chunks:
            subset = lines[index : index + size]
            if not subset:
                break
            chunk_text = "\n".join(subset)
            self._check_chunk_bytes(chunk_text)
            start_line = index + 1
            end_line = start_line + len(subset) - 1
            chunk_index += 1
            chunks.append(
                DocumentChunk(
                    chunk_index=chunk_index,
                    unit="lines",
                    start=start_line,
                    end=end_line,
                    content=chunk_text,
                )
            )
            index += step
        truncated = chunk_index >= max_chunks and index < len(lines)
        return chunks, truncated

    def _chunk_chars(
        self, content: str, size: int, overlap: int, max_chunks: int
    ) -> tuple[list[DocumentChunk], bool]:
        step = size - overlap
        index = 0
        chunk_index = 0
        chunks: list[DocumentChunk] = []
        while index < len(content) and chunk_index < max_chunks:
            chunk_text = content[index : index + size]
            if not chunk_text:
                break
            self._check_chunk_bytes(chunk_text)
            start = index
            end = index + len(chunk_text) - 1
            chunk_index += 1
            chunks.append(
                DocumentChunk(
                    chunk_index=chunk_index,
                    unit="chars",
                    start=start,
                    end=end,
                    content=chunk_text,
                )
            )
            index += step
        truncated = chunk_index >= max_chunks and index < len(content)
        return chunks, truncated

    def _check_chunk_bytes(self, content: str) -> None:
        size = len(content.encode("utf-8"))
        if size > self.config.max_chunk_bytes:
            raise ToolError(
                f"Chunk exceeds max_chunk_bytes ({size} > {self.config.max_chunk_bytes})."
            )

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

    def _build_code_chunks(
        self,
        lines: list[str],
        boundaries: list[bool],
        depths: list[int],
        unit: str,
        size: int,
        max_chunks: int,
        hard_split: bool,
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
            self._check_chunk_bytes(content)
            chunk = CodeChunkData(
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

    def _extract_tokens(self, content: str) -> dict[str, float]:
        min_len = self.config.min_token_length
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
            return {}

        sorted_tokens = sorted(tokens.items(), key=lambda item: (-item[1], item[0]))
        max_tokens = self.config.max_tokens_per_chunk
        if max_tokens > 0:
            sorted_tokens = sorted_tokens[:max_tokens]
        return {token: float(count) for token, count in sorted_tokens}

    def _build_direct_connections(
        self,
        token_maps: list[dict[str, float]],
        min_similarity: float,
        max_neighbors: int,
        max_shared_tokens: int,
    ) -> list[DirectConnection]:
        if len(token_maps) < 2:
            return []

        df: dict[str, int] = {}
        for tokens in token_maps:
            for token in tokens:
                df[token] = df.get(token, 0) + 1

        chunk_count = len(token_maps)
        idf: dict[str, float] = {}
        for token, count in df.items():
            ratio = count / chunk_count
            if ratio > self.config.max_df_ratio:
                continue
            idf[token] = math.log((chunk_count + 1) / (count + 1)) + 1.0

        tfidf: list[dict[str, float]] = []
        norms: list[float] = []
        for tokens in token_maps:
            weighted: dict[str, float] = {}
            total = 0.0
            for token, count in tokens.items():
                weight = count * idf.get(token, 0.0)
                if weight <= 0:
                    continue
                weighted[token] = weight
                total += weight * weight
            tfidf.append(weighted)
            norms.append(math.sqrt(total) if total > 0 else 0.0)

        dot: dict[tuple[int, int], float] = {}
        postings: dict[str, list[tuple[int, float]]] = {}
        for idx, weights in enumerate(tfidf):
            for token, weight in weights.items():
                postings.setdefault(token, []).append((idx, weight))

        for token, items in postings.items():
            if len(items) < 2:
                continue
            for i in range(len(items)):
                idx_i, weight_i = items[i]
                for j in range(i + 1, len(items)):
                    idx_j, weight_j = items[j]
                    key = (idx_i, idx_j)
                    dot[key] = dot.get(key, 0.0) + weight_i * weight_j

        neighbor_lists: list[list[tuple[int, float]]] = [
            [] for _ in range(chunk_count)
        ]
        for (idx_i, idx_j), value in dot.items():
            norm_i = norms[idx_i]
            norm_j = norms[idx_j]
            if norm_i == 0 or norm_j == 0:
                continue
            similarity = value / (norm_i * norm_j)
            if similarity < min_similarity:
                continue
            neighbor_lists[idx_i].append((idx_j, similarity))
            neighbor_lists[idx_j].append((idx_i, similarity))

        for idx in range(chunk_count):
            neighbor_lists[idx].sort(key=lambda item: (-item[1], item[0]))
            if max_neighbors > 0:
                neighbor_lists[idx] = neighbor_lists[idx][:max_neighbors]

        connections: list[DirectConnection] = []
        for source_idx, neighbors in enumerate(neighbor_lists):
            for target_idx, score in neighbors:
                if source_idx >= target_idx:
                    continue
                shared = sorted(set(token_maps[source_idx]) & set(token_maps[target_idx]))
                if max_shared_tokens > 0 and len(shared) > max_shared_tokens:
                    shared = shared[:max_shared_tokens]
                connections.append(
                    DirectConnection(
                        source_index=source_idx + 1,
                        target_index=target_idx + 1,
                        score=round(score, 6),
                        shared_tokens=shared,
                    )
                )

        return connections

    def _connections_to_adjacency(
        self, connections: list[DirectConnection], count: int
    ) -> list[list[tuple[int, float]]]:
        adjacency: list[list[tuple[int, float]]] = [[] for _ in range(count)]
        for conn in connections:
            src = conn.source_index - 1
            tgt = conn.target_index - 1
            adjacency[src].append((tgt, conn.score))
            adjacency[tgt].append((src, conn.score))
        for idx in range(count):
            adjacency[idx].sort(key=lambda item: (-item[1], item[0]))
        return adjacency

    def _build_deep_connections(
        self,
        adjacency: list[list[tuple[int, float]]],
        max_depth: int,
        min_deep_score: float,
        max_deep_connections: int,
    ) -> list[DeepConnection]:
        results: list[DeepConnection] = []
        node_count = len(adjacency)
        if node_count == 0:
            return results

        for source in range(node_count):
            best: dict[int, tuple[float, int, list[int]]] = {}
            heap: list[tuple[float, int, int, list[int]]] = []
            for target, score in adjacency[source]:
                if score < min_deep_score:
                    continue
                heapq.heappush(heap, (-score, 1, target, [source, target]))

            while heap:
                neg_score, depth, node, path = heapq.heappop(heap)
                score = -neg_score
                if node == source:
                    continue
                existing = best.get(node)
                if existing is None or score > existing[0]:
                    best[node] = (score, depth, path)

                if depth >= max_depth:
                    continue
                for next_node, edge_score in adjacency[node]:
                    if next_node in path:
                        continue
                    new_score = score * edge_score
                    if new_score < min_deep_score:
                        continue
                    heapq.heappush(
                        heap,
                        (-new_score, depth + 1, next_node, path + [next_node]),
                    )

            items = sorted(
                best.items(), key=lambda item: (-item[1][0], item[0])
            )
            if max_deep_connections > 0:
                items = items[:max_deep_connections]
            for target, (score, depth, path) in items:
                results.append(
                    DeepConnection(
                        source_index=source + 1,
                        target_index=target + 1,
                        score=round(score, 6),
                        depth=depth,
                        path=[idx + 1 for idx in path],
                    )
                )

        return results

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ChunkDeepConnectionsSingleArgs):
            return ToolCallDisplay(summary="chunk_deep_connections_single")

        summary = "chunk_deep_connections_single"
        return ToolCallDisplay(
            summary=summary,
            details={
                "path": event.args.path,
                "kind": event.args.kind,
                "language": event.args.language,
                "code_mode": event.args.code_mode,
                "unit": event.args.unit,
                "size": event.args.size,
                "overlap": event.args.overlap,
                "split_depth": event.args.split_depth,
                "respect_brackets": event.args.respect_brackets,
                "hard_split": event.args.hard_split,
                "max_chunks": event.args.max_chunks,
                "max_depth": event.args.max_depth,
                "min_similarity": event.args.min_similarity,
                "min_deep_score": event.args.min_deep_score,
                "max_neighbors": event.args.max_neighbors,
                "max_deep_connections": event.args.max_deep_connections,
                "max_shared_tokens": event.args.max_shared_tokens,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ChunkDeepConnectionsSingleResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = (
            f"Built {event.result.count} chunk(s), "
            f"{event.result.connection_count} direct link(s), "
            f"{event.result.deep_connection_count} deep link(s)"
        )
        warnings: list[str] = []
        if event.result.truncated:
            warnings.append("Output truncated by size or limits")

        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=warnings,
            details={
                "kind": event.result.kind,
                "unit": event.result.unit,
                "code_mode": event.result.code_mode,
                "count": event.result.count,
                "connection_count": event.result.connection_count,
                "deep_connection_count": event.result.deep_connection_count,
                "truncated": event.result.truncated,
                "chunks": event.result.chunks,
                "connections": event.result.connections,
                "deep_connections": event.result.deep_connections,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Building deep context connections"
