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


class ChunkCrossrefSingleConfig(BaseToolConfig):
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
        default=500, description="Maximum tokens stored per chunk."
    )
    min_token_length: int = Field(
        default=3, description="Minimum token length."
    )
    min_shared_tokens: int = Field(
        default=1, description="Minimum shared tokens for a cross-reference."
    )
    max_refs_per_chunk: int = Field(
        default=5, description="Maximum references per chunk."
    )
    max_shared_tokens: int = Field(
        default=10, description="Maximum shared tokens returned per reference."
    )
    default_code_extensions: list[str] = Field(
        default=sorted(CODE_EXTS),
        description="Extensions treated as code for auto detection.",
    )


class ChunkCrossrefSingleState(BaseToolState):
    pass


class ChunkCrossrefSingleArgs(BaseModel):
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
    min_shared_tokens: int | None = Field(
        default=None, description="Minimum shared tokens for a reference."
    )
    max_refs_per_chunk: int | None = Field(
        default=None, description="Maximum references per chunk."
    )
    max_shared_tokens: int | None = Field(
        default=None, description="Maximum shared tokens per reference."
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


class ChunkCrossrefTarget(BaseModel):
    chunk_index: int
    overlap: int
    shared_tokens: list[str]


class ChunkCrossref(BaseModel):
    source_index: int
    targets: list[ChunkCrossrefTarget]


class ChunkCrossrefSingleResult(BaseModel):
    kind: str
    unit: str
    code_mode: str | None
    chunks: list[CrossrefChunk]
    crossrefs: list[ChunkCrossref]
    count: int
    reference_count: int
    truncated: bool


class ChunkCrossrefSingle(
    BaseTool[
        ChunkCrossrefSingleArgs,
        ChunkCrossrefSingleResult,
        ChunkCrossrefSingleConfig,
        ChunkCrossrefSingleState,
    ],
    ToolUIData[ChunkCrossrefSingleArgs, ChunkCrossrefSingleResult],
):
    description: ClassVar[str] = (
        "Chunk a single document or script and cross-reference related chunks."
    )

    async def run(self, args: ChunkCrossrefSingleArgs) -> ChunkCrossrefSingleResult:
        content, source_path = self._load_content(args)
        if not content:
            return ChunkCrossrefSingleResult(
                kind="document",
                unit=self.config.default_unit,
                code_mode=None,
                chunks=[],
                crossrefs=[],
                count=0,
                reference_count=0,
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

        min_shared = (
            args.min_shared_tokens
            if args.min_shared_tokens is not None
            else self.config.min_shared_tokens
        )
        max_refs = (
            args.max_refs_per_chunk
            if args.max_refs_per_chunk is not None
            else self.config.max_refs_per_chunk
        )
        max_shared_tokens = (
            args.max_shared_tokens
            if args.max_shared_tokens is not None
            else self.config.max_shared_tokens
        )
        if min_shared < 0:
            raise ToolError("min_shared_tokens must be >= 0.")
        if max_refs < 0:
            raise ToolError("max_refs_per_chunk must be >= 0.")
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

        token_sets = [self._extract_tokens(chunk.content) for chunk in chunks]
        crossrefs = self._build_crossrefs(
            token_sets, min_shared, max_refs, max_shared_tokens
        )
        reference_count = sum(len(item.targets) for item in crossrefs)

        return ChunkCrossrefSingleResult(
            kind=kind,
            unit=unit,
            code_mode=code_mode,
            chunks=chunks,
            crossrefs=crossrefs,
            count=len(chunks),
            reference_count=reference_count,
            truncated=truncated,
        )

    def _load_content(
        self, args: ChunkCrossrefSingleArgs
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
        self, args: ChunkCrossrefSingleArgs, source_path: Path | None
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

    def _normalize_code_mode(self, args: ChunkCrossrefSingleArgs) -> str:
        mode = (args.code_mode or self.config.default_code_mode).strip().lower()
        if mode not in {"auto", "indent", "brace"}:
            raise ToolError("code_mode must be auto, indent, or brace.")
        return mode

    def _resolve_language(
        self, args: ChunkCrossrefSingleArgs, source_path: Path | None
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
        args: ChunkCrossrefSingleArgs,
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

    def _extract_tokens(self, content: str) -> set[str]:
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
            return set()

        sorted_tokens = sorted(tokens.items(), key=lambda item: (-item[1], item[0]))
        max_tokens = self.config.max_tokens_per_chunk
        trimmed = sorted_tokens[:max_tokens] if max_tokens > 0 else sorted_tokens
        return {token for token, _ in trimmed}

    def _build_crossrefs(
        self,
        token_sets: list[set[str]],
        min_shared: int,
        max_refs: int,
        max_shared_tokens: int,
    ) -> list[ChunkCrossref]:
        overlaps: list[dict[int, int]] = [dict() for _ in token_sets]
        token_map: dict[str, list[int]] = {}
        for idx, tokens in enumerate(token_sets):
            for token in tokens:
                token_map.setdefault(token, []).append(idx)

        for indices in token_map.values():
            if len(indices) < 2:
                continue
            for i in range(len(indices)):
                source = indices[i]
                for target in indices[i + 1 :]:
                    overlaps[source][target] = overlaps[source].get(target, 0) + 1
                    overlaps[target][source] = overlaps[target].get(source, 0) + 1

        crossrefs: list[ChunkCrossref] = []
        for idx, counts in enumerate(overlaps):
            targets: list[ChunkCrossrefTarget] = []
            for target_idx, overlap in counts.items():
                if overlap < min_shared:
                    continue
                shared = sorted(token_sets[idx] & token_sets[target_idx])
                if max_shared_tokens > 0 and len(shared) > max_shared_tokens:
                    shared = shared[:max_shared_tokens]
                targets.append(
                    ChunkCrossrefTarget(
                        chunk_index=target_idx + 1,
                        overlap=overlap,
                        shared_tokens=shared,
                    )
                )

            targets.sort(key=lambda item: (-item.overlap, item.chunk_index))
            if max_refs > 0:
                targets = targets[:max_refs]
            crossrefs.append(ChunkCrossref(source_index=idx + 1, targets=targets))

        return crossrefs

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ChunkCrossrefSingleArgs):
            return ToolCallDisplay(summary="chunk_crossref_single")

        summary = "chunk_crossref_single"
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
                "min_shared_tokens": event.args.min_shared_tokens,
                "max_refs_per_chunk": event.args.max_refs_per_chunk,
                "max_shared_tokens": event.args.max_shared_tokens,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ChunkCrossrefSingleResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = (
            f"Built {event.result.count} chunk(s) with "
            f"{event.result.reference_count} cross-reference(s)"
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
                "reference_count": event.result.reference_count,
                "truncated": event.result.truncated,
                "chunks": event.result.chunks,
                "crossrefs": event.result.crossrefs,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Cross-referencing chunks"
