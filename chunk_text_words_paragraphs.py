from __future__ import annotations

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


WORD_RE = re.compile(r"[A-Za-z0-9_]+")
PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")


class ChunkTextWordsParagraphsConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_input_bytes: int = Field(
        default=1_000_000,
        description="Maximum bytes allowed for input content.",
    )
    max_chunks: int = Field(
        default=200,
        description="Maximum number of chunks to return.",
    )
    default_mode: str = Field(
        default="words",
        description="Default mode: words or paragraphs.",
    )
    default_word_size: int = Field(
        default=200,
        description="Default chunk size in words.",
    )
    default_paragraph_size: int = Field(
        default=1,
        description="Default chunk size in paragraphs.",
    )
    default_word_overlap: int = Field(
        default=0,
        description="Default overlap in words.",
    )
    default_paragraph_overlap: int = Field(
        default=0,
        description="Default overlap in paragraphs.",
    )


class ChunkTextWordsParagraphsState(BaseToolState):
    pass


class ChunkTextWordsParagraphsArgs(BaseModel):
    content: str | None = Field(default=None, description="Raw text to chunk.")
    path: str | None = Field(default=None, description="Path to a text file to chunk.")
    mode: str | None = Field(
        default=None,
        description="Chunking mode: words or paragraphs.",
    )
    size: int | None = Field(
        default=None,
        description="Chunk size in words or paragraphs.",
    )
    overlap: int | None = Field(
        default=None,
        description="Overlap in words or paragraphs.",
    )
    max_chunks: int | None = Field(
        default=None,
        description="Override the configured max chunks limit.",
    )


class WordParagraphChunk(BaseModel):
    index: int
    unit: str
    start: int
    end: int
    word_count: int
    paragraph_count: int
    content: str


class ChunkTextWordsParagraphsResult(BaseModel):
    chunks: list[WordParagraphChunk]
    count: int
    truncated: bool
    mode: str
    unit: str
    total_words: int
    total_paragraphs: int
    input_bytes: int


class ChunkTextWordsParagraphs(
    BaseTool[
        ChunkTextWordsParagraphsArgs,
        ChunkTextWordsParagraphsResult,
        ChunkTextWordsParagraphsConfig,
        ChunkTextWordsParagraphsState,
    ],
    ToolUIData[ChunkTextWordsParagraphsArgs, ChunkTextWordsParagraphsResult],
):
    description: ClassVar[str] = (
        "Chunk text by word count or paragraph count instead of tokens."
    )

    async def run(self, args: ChunkTextWordsParagraphsArgs) -> ChunkTextWordsParagraphsResult:
        content = self._load_content(args)
        mode = (args.mode or self.config.default_mode).strip().lower()
        if mode not in {"words", "paragraphs"}:
            raise ToolError("mode must be words or paragraphs.")

        total_words = self._count_words(content)
        total_paragraphs = self._count_paragraphs(content)

        if not content:
            return ChunkTextWordsParagraphsResult(
                chunks=[],
                count=0,
                truncated=False,
                mode=mode,
                unit=mode,
                total_words=total_words,
                total_paragraphs=total_paragraphs,
                input_bytes=0,
            )

        size = args.size
        overlap = args.overlap
        if mode == "words":
            size = size if size is not None else self.config.default_word_size
            overlap = (
                overlap if overlap is not None else self.config.default_word_overlap
            )
        else:
            size = size if size is not None else self.config.default_paragraph_size
            overlap = (
                overlap
                if overlap is not None
                else self.config.default_paragraph_overlap
            )

        if size <= 0:
            raise ToolError("size must be a positive integer.")
        if overlap < 0:
            raise ToolError("overlap must be a non-negative integer.")
        if overlap >= size:
            raise ToolError("overlap must be smaller than size.")

        max_chunks = args.max_chunks if args.max_chunks is not None else self.config.max_chunks
        if max_chunks <= 0:
            raise ToolError("max_chunks must be a positive integer.")

        if mode == "words":
            chunks = self._chunk_words(content, size, overlap)
        else:
            chunks = self._chunk_paragraphs(content, size, overlap)

        truncated = len(chunks) > max_chunks
        if truncated:
            chunks = chunks[:max_chunks]

        return ChunkTextWordsParagraphsResult(
            chunks=chunks,
            count=len(chunks),
            truncated=truncated,
            mode=mode,
            unit=mode,
            total_words=total_words,
            total_paragraphs=total_paragraphs,
            input_bytes=len(content.encode("utf-8")),
        )

    def _load_content(self, args: ChunkTextWordsParagraphsArgs) -> str:
        if args.content and args.path:
            raise ToolError("Provide either content or path, not both.")
        if args.content is None and args.path is None:
            raise ToolError("Provide content or path.")

        if args.content is not None:
            data = args.content.encode("utf-8")
            self._validate_input_size(len(data))
            return args.content

        path = self._resolve_path(args.path or "")
        size = path.stat().st_size
        self._validate_input_size(size)
        return path.read_text("utf-8", errors="ignore")

    def _validate_input_size(self, size: int) -> None:
        if size > self.config.max_input_bytes:
            raise ToolError(
                f"Input is {size} bytes, which exceeds the limit of {self.config.max_input_bytes} bytes."
            )

    def _resolve_path(self, raw_path: str) -> Path:
        if not raw_path.strip():
            raise ToolError("Path cannot be empty.")

        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.config.effective_workdir / path

        try:
            resolved = path.resolve()
        except ValueError as exc:
            raise ToolError("Security error: cannot resolve the provided path.") from exc

        if not resolved.exists():
            raise ToolError(f"File not found at: {resolved}")
        if resolved.is_dir():
            raise ToolError(f"Path is a directory, not a file: {resolved}")
        return resolved

    def _count_words(self, content: str) -> int:
        return len(WORD_RE.findall(content))

    def _count_paragraphs(self, content: str) -> int:
        paragraphs = [p.strip() for p in PARAGRAPH_SPLIT.split(content) if p.strip()]
        return len(paragraphs)

    def _chunk_words(self, content: str, size: int, overlap: int) -> list[WordParagraphChunk]:
        matches = list(WORD_RE.finditer(content))
        if not matches:
            return []
        step = size - overlap
        chunks: list[WordParagraphChunk] = []
        index = 0
        chunk_index = 0
        total = len(matches)

        while index < total:
            end_index = min(index + size - 1, total - 1)
            start_pos = matches[index].start()
            end_pos = matches[end_index].end()
            chunk_text = content[start_pos:end_pos]
            chunk_index += 1
            chunks.append(
                WordParagraphChunk(
                    index=chunk_index,
                    unit="words",
                    start=index + 1,
                    end=end_index + 1,
                    word_count=end_index - index + 1,
                    paragraph_count=self._count_paragraphs(chunk_text),
                    content=chunk_text,
                )
            )
            index += step

        return chunks

    def _chunk_paragraphs(
        self, content: str, size: int, overlap: int
    ) -> list[WordParagraphChunk]:
        paragraphs = [p.strip() for p in PARAGRAPH_SPLIT.split(content) if p.strip()]
        if not paragraphs:
            return []
        step = size - overlap
        index = 0
        chunk_index = 0
        chunks: list[WordParagraphChunk] = []
        total = len(paragraphs)

        while index < total:
            subset = paragraphs[index : index + size]
            if not subset:
                break
            chunk_text = "\n\n".join(subset)
            chunk_index += 1
            start_paragraph = index + 1
            end_paragraph = start_paragraph + len(subset) - 1
            chunks.append(
                WordParagraphChunk(
                    index=chunk_index,
                    unit="paragraphs",
                    start=start_paragraph,
                    end=end_paragraph,
                    word_count=self._count_words(chunk_text),
                    paragraph_count=len(subset),
                    content=chunk_text,
                )
            )
            index += step

        return chunks

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ChunkTextWordsParagraphsArgs):
            return ToolCallDisplay(summary="chunk_text_words_paragraphs")

        summary = f"chunk_text_words_paragraphs: {event.args.mode or 'words'}"
        return ToolCallDisplay(
            summary=summary,
            details={
                "mode": event.args.mode,
                "size": event.args.size,
                "overlap": event.args.overlap,
                "max_chunks": event.args.max_chunks,
                "path": event.args.path,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ChunkTextWordsParagraphsResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = f"Created {event.result.count} chunks"
        if event.result.truncated:
            message += " (truncated)"

        warnings = []
        if event.result.truncated:
            warnings.append("Chunk list truncated by max_chunks limit")

        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=warnings,
            details={
                "count": event.result.count,
                "truncated": event.result.truncated,
                "mode": event.result.mode,
                "unit": event.result.unit,
                "total_words": event.result.total_words,
                "total_paragraphs": event.result.total_paragraphs,
                "input_bytes": event.result.input_bytes,
                "chunks": event.result.chunks,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Chunking by words or paragraphs"
