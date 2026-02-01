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
    start_depth: int | None = None


class ChunkCodeByClassPrimaryConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_input_bytes: int = Field(
        default=5_000_000, description="Maximum input size in bytes."
    )
    max_chunk_bytes: int = Field(
        default=500_000, description="Maximum class chunk size in bytes."
    )
    max_chunks: int = Field(
        default=200, description="Maximum number of class chunks to return."
    )
    default_mode: str = Field(default="auto", description="auto, indent, brace.")
    default_class_keywords: list[str] = Field(
        default=sorted(CLASS_KEYWORDS),
        description="Keywords treated as class declarations.",
    )


class ChunkCodeByClassPrimaryState(BaseToolState):
    pass


class ChunkCodeByClassPrimaryArgs(BaseModel):
    content: str | None = Field(default=None, description="Raw code to chunk.")
    path: str | None = Field(default=None, description="Path to a code file.")
    language: str | None = Field(default=None, description="Language hint.")
    mode: str | None = Field(default=None, description="auto, indent, or brace.")
    class_keywords: list[str] | None = Field(
        default=None, description="Override class keywords to detect."
    )
    max_chunks: int | None = Field(
        default=None, description="Override the configured max chunks limit."
    )


class ClassChunk(BaseModel):
    index: int
    name: str
    kind: str
    start_line: int
    end_line: int
    content: str


class ChunkCodeByClassPrimaryResult(BaseModel):
    mode: str
    chunks: list[ClassChunk]
    count: int
    truncated: bool


class ChunkCodeByClassPrimary(
    BaseTool[
        ChunkCodeByClassPrimaryArgs,
        ChunkCodeByClassPrimaryResult,
        ChunkCodeByClassPrimaryConfig,
        ChunkCodeByClassPrimaryState,
    ],
    ToolUIData[ChunkCodeByClassPrimaryArgs, ChunkCodeByClassPrimaryResult],
):
    description: ClassVar[str] = (
        "Chunk a script by each top-level class definition."
    )

    async def run(
        self, args: ChunkCodeByClassPrimaryArgs
    ) -> ChunkCodeByClassPrimaryResult:
        content, source_path = self._load_content(args)
        if not content:
            return ChunkCodeByClassPrimaryResult(
                mode="auto",
                chunks=[],
                count=0,
                truncated=False,
            )

        mode = self._resolve_mode(args, source_path)
        keywords = self._resolve_keywords(args)

        max_chunks = args.max_chunks if args.max_chunks is not None else self.config.max_chunks
        if max_chunks <= 0:
            raise ToolError("max_chunks must be a positive integer.")

        include_nested = False
        match mode:
            case "indent":
                chunks = self._find_classes_indent(content, keywords, include_nested)
            case "brace":
                chunks = self._find_classes_brace(content, keywords, include_nested)
            case _:
                raise ToolError("mode must be auto, indent, or brace.")

        chunks = self._normalize_chunk_order(chunks)

        truncated = len(chunks) > max_chunks
        if truncated:
            chunks = chunks[:max_chunks]

        self._validate_chunk_sizes(chunks)

        return ChunkCodeByClassPrimaryResult(
            mode=mode,
            chunks=chunks,
            count=len(chunks),
            truncated=truncated,
        )

    def _load_content(
        self, args: ChunkCodeByClassPrimaryArgs
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
        if resolved.is_dir():
            raise ToolError(f"Path is a directory, not a file: {resolved}")
        return resolved

    def _resolve_mode(
        self, args: ChunkCodeByClassPrimaryArgs, source_path: Path | None
    ) -> str:
        mode = (args.mode or self.config.default_mode).strip().lower()
        if mode != "auto":
            return mode

        language = (args.language or "").strip().lower()
        if language in INDENT_LANGS:
            return "indent"

        if source_path and source_path.suffix.lower() in INDENT_EXTS:
            return "indent"
        return "brace"

    def _resolve_keywords(self, args: ChunkCodeByClassPrimaryArgs) -> list[str]:
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

    def _validate_chunk_sizes(self, chunks: list[ClassChunk]) -> None:
        max_bytes = self.config.max_chunk_bytes
        for chunk in chunks:
            size = len(chunk.content.encode("utf-8"))
            if size > max_bytes:
                raise ToolError(
                    f"Chunk '{chunk.name}' exceeds max_chunk_bytes ({size} > {max_bytes})."
                )

    def _normalize_chunk_order(self, chunks: list[ClassChunk]) -> list[ClassChunk]:
        ordered = sorted(chunks, key=lambda item: (item.start_line, item.end_line))
        return [
            ClassChunk(
                index=idx + 1,
                name=chunk.name,
                kind=chunk.kind,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                content=chunk.content,
            )
            for idx, chunk in enumerate(ordered)
        ]

    def _find_classes_indent(
        self, content: str, keywords: list[str], include_nested: bool
    ) -> list[ClassChunk]:
        lines = content.splitlines()
        state = _ScanState()
        chunks: list[ClassChunk] = []
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
                        chunks.append(self._build_chunk(entry, lines, idx - 1, len(chunks)))

            if not stripped:
                continue

            if match := self._detect_class(stripped, keyword_set):
                kind, name = match
                depth = len(stack) + 1
                stack.append(
                    _ClassEntry(
                        name=name,
                        kind=kind,
                        start_line=idx,
                        indent=indent,
                        depth=depth,
                    )
                )

        last_line = len(lines)
        while stack:
            entry = stack.pop()
            if include_nested or entry.depth == 1:
                chunks.append(self._build_chunk(entry, lines, last_line, len(chunks)))

        return chunks

    def _find_classes_brace(
        self, content: str, keywords: list[str], include_nested: bool
    ) -> list[ClassChunk]:
        lines = content.splitlines()
        state = _ScanState()
        chunks: list[ClassChunk] = []
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
                    pending = _ClassEntry(
                        name=name,
                        kind=kind,
                        start_line=idx,
                        indent=0,
                        depth=len(stack) + 1,
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
                                chunks.append(
                                    self._build_chunk(entry, lines, idx, len(chunks))
                                )
                            continue
                        break
                elif ch == ";" and pending:
                    pending = None

        last_line = len(lines)
        while stack:
            entry = stack.pop()
            if include_nested or entry.depth == 1:
                chunks.append(self._build_chunk(entry, lines, last_line, len(chunks)))

        return chunks

    def _build_chunk(
        self, entry: _ClassEntry, lines: list[str], end_line: int, index: int
    ) -> ClassChunk:
        start_line = entry.start_line
        end_line = max(end_line, start_line)
        content = "\n".join(lines[start_line - 1 : end_line])
        return ClassChunk(
            index=index + 1,
            name=entry.name,
            kind=entry.kind,
            start_line=start_line,
            end_line=end_line,
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

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ChunkCodeByClassPrimaryArgs):
            return ToolCallDisplay(summary="chunk_code_by_class_primary")

        summary = "chunk_code_by_class_primary"
        return ToolCallDisplay(
            summary=summary,
            details={
                "path": event.args.path,
                "language": event.args.language,
                "mode": event.args.mode,
                "class_keywords": event.args.class_keywords,
                "max_chunks": event.args.max_chunks,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ChunkCodeByClassPrimaryResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = f"Found {event.result.count} primary class chunk(s)"
        warnings: list[str] = []
        if event.result.truncated:
            warnings.append("Chunk list truncated by max_chunks limit")

        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=warnings,
            details={
                "mode": event.result.mode,
                "count": event.result.count,
                "truncated": event.result.truncated,
                "chunks": event.result.chunks,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Chunking primary classes"
