from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TYPE_CHECKING

from pydantic import BaseModel, Field

from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    ToolCallDisplay,
    ToolError,
    ToolPermission,
    ToolResultDisplay,
    ToolUIData,
)

if TYPE_CHECKING:
    from vibe.core.types import ToolCallEvent, ToolResultEvent


WORD_RE = re.compile(r"[A-Za-z0-9_']+")
SPEAKER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _.'-]{0,40}):\s*(.*)$")


@dataclass
class _Segment:
    index: int
    speaker: str | None
    start: int
    end: int
    text: str


class ContextSpokenWordChunkingConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_input_bytes: int = Field(
        default=3_000_000,
        description="Maximum bytes allowed for input content.",
    )
    max_chunks: int = Field(
        default=200,
        description="Maximum number of chunks to return.",
    )
    default_chunk_size: int = Field(
        default=200,
        description="Default chunk size in words.",
    )
    default_overlap: int = Field(
        default=0,
        description="Default overlap in words.",
    )
    preserve_speakers: bool = Field(
        default=True,
        description="Keep chunks within speaker segments when possible.",
    )
    min_segment_chars: int = Field(
        default=20, description="Minimum segment length before merging."
    )


class ContextSpokenWordChunkingState(BaseToolState):
    pass


class SpokenWordChunkingArgs(BaseModel):
    content: str | None = Field(default=None, description="Raw text to chunk.")
    path: str | None = Field(default=None, description="Path to a text file to chunk.")
    chunk_size: int | None = Field(
        default=None, description="Chunk size in words."
    )
    overlap: int | None = Field(default=None, description="Overlap in words.")
    max_chunks: int | None = Field(
        default=None, description="Override the configured max chunks."
    )
    preserve_speakers: bool | None = Field(
        default=None, description="Override speaker boundary preservation."
    )


class SpokenWordChunk(BaseModel):
    index: int
    segment_index: int
    speaker: str | None
    word_start: int
    word_end: int
    word_count: int
    start: int
    end: int
    content: str


class SpokenWordChunkingResult(BaseModel):
    chunks: list[SpokenWordChunk]
    count: int
    truncated: bool
    total_words: int
    total_segments: int
    input_bytes: int


class ContextSpokenWordChunking(
    BaseTool[
        SpokenWordChunkingArgs,
        SpokenWordChunkingResult,
        ContextSpokenWordChunkingConfig,
        ContextSpokenWordChunkingState,
    ],
    ToolUIData[SpokenWordChunkingArgs, SpokenWordChunkingResult],
):
    description: ClassVar[str] = (
        "Chunk spoken words into word windows while preserving dialogue segments."
    )

    async def run(self, args: SpokenWordChunkingArgs) -> SpokenWordChunkingResult:
        content = self._load_content(args)
        chunk_size = args.chunk_size or self.config.default_chunk_size
        overlap = args.overlap if args.overlap is not None else self.config.default_overlap
        max_chunks = args.max_chunks if args.max_chunks is not None else self.config.max_chunks
        preserve_speakers = (
            args.preserve_speakers
            if args.preserve_speakers is not None
            else self.config.preserve_speakers
        )

        if chunk_size <= 0:
            raise ToolError("chunk_size must be a positive integer.")
        if overlap < 0:
            raise ToolError("overlap must be a non-negative integer.")
        if overlap >= chunk_size:
            raise ToolError("overlap must be smaller than chunk_size.")
        if max_chunks <= 0:
            raise ToolError("max_chunks must be a positive integer.")

        segments = (
            self._segment_conversation(content)
            if preserve_speakers
            else [self._single_segment(content)]
        )
        total_words = sum(len(WORD_RE.findall(seg.text)) for seg in segments)
        chunks: list[SpokenWordChunk] = []
        chunk_index = 0

        for segment in segments:
            seg_chunks = self._chunk_segment(segment, chunk_size, overlap)
            for chunk in seg_chunks:
                chunk_index += 1
                chunks.append(
                    SpokenWordChunk(
                        index=chunk_index,
                        segment_index=segment.index,
                        speaker=segment.speaker,
                        word_start=chunk.word_start,
                        word_end=chunk.word_end,
                        word_count=chunk.word_count,
                        start=chunk.start,
                        end=chunk.end,
                        content=chunk.content,
                    )
                )
                if len(chunks) >= max_chunks:
                    return SpokenWordChunkingResult(
                        chunks=chunks,
                        count=len(chunks),
                        truncated=True,
                        total_words=total_words,
                        total_segments=len(segments),
                        input_bytes=len(content.encode("utf-8")),
                    )

        return SpokenWordChunkingResult(
            chunks=chunks,
            count=len(chunks),
            truncated=False,
            total_words=total_words,
            total_segments=len(segments),
            input_bytes=len(content.encode("utf-8")),
        )

    def _load_content(self, args: SpokenWordChunkingArgs) -> str:
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

    def _segment_conversation(self, text: str) -> list[_Segment]:
        segments: list[_Segment] = []
        lines = text.splitlines()
        current_lines: list[str] = []
        current_speaker: str | None = None
        seg_start = 0
        cursor = 0

        def flush(end_offset: int) -> None:
            nonlocal current_lines, current_speaker, seg_start
            if not current_lines:
                return
            seg_text = "\n".join(current_lines).strip()
            if not seg_text:
                current_lines = []
                return
            if len(seg_text) < self.config.min_segment_chars and segments:
                segments[-1].text = f"{segments[-1].text}\n{seg_text}".strip()
                segments[-1].end = end_offset
            else:
                segments.append(
                    _Segment(
                        index=len(segments) + 1,
                        speaker=current_speaker,
                        start=seg_start,
                        end=end_offset,
                        text=seg_text,
                    )
                )
            current_lines = []
            current_speaker = None

        for line in lines:
            line_len = len(line) + 1
            speaker_match = SPEAKER_RE.match(line)
            if speaker_match:
                flush(cursor)
                current_speaker = speaker_match.group(1).strip()
                current_lines.append(speaker_match.group(2))
                seg_start = cursor
            elif not line.strip():
                flush(cursor)
            else:
                if not current_lines:
                    seg_start = cursor
                current_lines.append(line)
            cursor += line_len

        flush(cursor)
        if not segments and text.strip():
            segments.append(self._single_segment(text))
        return segments

    def _single_segment(self, text: str) -> _Segment:
        return _Segment(index=1, speaker=None, start=0, end=len(text), text=text.strip())

    def _chunk_segment(
        self, segment: _Segment, size: int, overlap: int
    ) -> list[SpokenWordChunk]:
        matches = list(WORD_RE.finditer(segment.text))
        if not matches:
            return []
        step = size - overlap
        chunks: list[SpokenWordChunk] = []
        index = 0
        total = len(matches)

        while index < total:
            end_index = min(index + size - 1, total - 1)
            start_pos = matches[index].start()
            end_pos = matches[end_index].end()
            chunk_text = segment.text[start_pos:end_pos]
            chunks.append(
                SpokenWordChunk(
                    index=0,
                    segment_index=segment.index,
                    speaker=segment.speaker,
                    word_start=index + 1,
                    word_end=end_index + 1,
                    word_count=end_index - index + 1,
                    start=segment.start + start_pos,
                    end=segment.start + end_pos,
                    content=chunk_text,
                )
            )
            index += step

        return chunks

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, SpokenWordChunkingArgs):
            return ToolCallDisplay(summary="context_spoken_word_chunking")
        return ToolCallDisplay(
            summary="context_spoken_word_chunking",
            details={
                "chunk_size": event.args.chunk_size,
                "overlap": event.args.overlap,
                "max_chunks": event.args.max_chunks,
                "path": event.args.path,
                "preserve_speakers": event.args.preserve_speakers,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, SpokenWordChunkingResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = f"Created {event.result.count} spoken word chunks"
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
                "total_words": event.result.total_words,
                "total_segments": event.result.total_segments,
                "input_bytes": event.result.input_bytes,
                "chunks": event.result.chunks,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Chunking spoken words"
