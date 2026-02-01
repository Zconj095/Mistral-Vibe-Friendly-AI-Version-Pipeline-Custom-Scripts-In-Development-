from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import ClassVar, TYPE_CHECKING

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


WORD_RE = re.compile(r"[A-Za-z0-9_']+")
SENTENCE_RE = re.compile(r"[^.!...]+[.!...]*", re.S)

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


class ContextSpeechExplainDocumentConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_source_bytes: int = Field(default=50_000_000, description="Maximum bytes per document.")
    max_chunks: int = Field(
        default=5000, description="Maximum chunks to return (0 for unlimited)."
    )
    preview_chars: int = Field(default=240, description="Preview snippet length.")
    default_chunk_mode: str = Field(default="words", description="Default chunk mode.")
    default_chunk_size: int = Field(default=200, description="Default chunk size.")
    default_overlap: int = Field(default=0, description="Default chunk overlap.")
    min_token_length: int = Field(default=3, description="Minimum token length.")
    max_keywords: int = Field(default=12, description="Max keywords per chunk.")
    max_key_sentences: int = Field(default=3, description="Max key sentences per chunk.")
    max_sentence_chars: int = Field(default=400, description="Max sentence chars.")


class ContextSpeechExplainDocumentState(BaseToolState):
    pass


class ContextSpeechExplainDocumentArgs(BaseModel):
    content: str | None = Field(default=None, description="Document content.")
    path: str | None = Field(default=None, description="Path to a document.")
    chunk_mode: str | None = Field(
        default=None,
        description="Chunk mode: words, lines, paragraphs, sentences.",
    )
    chunk_size: int | None = Field(
        default=None,
        description="Chunk size in units for the selected mode.",
    )
    overlap: int | None = Field(default=None, description="Chunk overlap in units.")
    max_chunks: int | None = Field(
        default=None, description="Override max chunk limit (0 for unlimited)."
    )
    include_key_sentences: bool = Field(
        default=True, description="Include key sentences per chunk."
    )


class SpeechChunk(BaseModel):
    index: int
    start: int
    end: int
    word_count: int
    char_count: int
    preview: str
    keywords: list[str]
    key_sentences: list[str]
    speech_prompt: str


class SpeechCue(BaseModel):
    index: int
    chunk_index: int
    cue: str


class ContextSpeechExplainDocumentResult(BaseModel):
    chunks: list[SpeechChunk]
    chunk_count: int
    truncated: bool
    chunk_mode: str
    chunk_size: int
    overlap: int
    input_bytes: int
    overall_keywords: list[str]
    speech_opening: str
    speech_closing: str
    speech_outline: list[SpeechCue]
    warnings: list[str]


class ContextSpeechExplainDocument(
    BaseTool[
        ContextSpeechExplainDocumentArgs,
        ContextSpeechExplainDocumentResult,
        ContextSpeechExplainDocumentConfig,
        ContextSpeechExplainDocumentState,
    ],
    ToolUIData[
        ContextSpeechExplainDocumentArgs,
        ContextSpeechExplainDocumentResult,
    ],
):
    description: ClassVar[str] = (
        "Chunk a document completely and prepare speech prompts for each chunk."
    )

    async def run(
        self, args: ContextSpeechExplainDocumentArgs
    ) -> ContextSpeechExplainDocumentResult:
        content = self._load_content(args)
        mode = (args.chunk_mode or self.config.default_chunk_mode).strip().lower()
        size = args.chunk_size or self.config.default_chunk_size
        overlap = args.overlap if args.overlap is not None else self.config.default_overlap
        max_chunks = (
            args.max_chunks if args.max_chunks is not None else self.config.max_chunks
        )
        if max_chunks <= 0:
            max_chunks = None
        warnings: list[str] = []

        if size <= 0:
            raise ToolError("chunk_size must be positive.")
        if overlap < 0:
            raise ToolError("overlap must be non-negative.")
        if overlap >= size:
            raise ToolError("overlap must be smaller than chunk_size.")

        units = self._build_units(content, mode)
        if not units:
            return ContextSpeechExplainDocumentResult(
                chunks=[],
                chunk_count=0,
                truncated=False,
                chunk_mode=mode,
                chunk_size=size,
                overlap=overlap,
                input_bytes=len(content.encode("utf-8")),
                overall_keywords=[],
                speech_opening="",
                speech_closing="",
                speech_outline=[],
                warnings=["No content to chunk."],
            )

        joiner = self._joiner_for_mode(mode)
        chunk_units = self._chunk_units(units, size, overlap, joiner)
        truncated = False
        if max_chunks is not None and len(chunk_units) > max_chunks:
            truncated = True
            warnings.append("Chunk limit reached; output truncated.")
            chunk_units = chunk_units[:max_chunks]

        chunks: list[SpeechChunk] = []
        overall_tokens = Counter()
        for idx, (start, end, text) in enumerate(chunk_units, start=1):
            tokens = self._tokenize(text)
            overall_tokens.update(tokens)
            keywords = self._top_keywords(tokens, self.config.max_keywords)
            key_sentences = []
            if args.include_key_sentences:
                key_sentences = self._key_sentences(text, keywords)
            prompt = self._speech_prompt(idx, keywords, key_sentences)
            chunks.append(
                SpeechChunk(
                    index=idx,
                    start=start,
                    end=end,
                    word_count=len(tokens),
                    char_count=len(text),
                    preview=self._preview(text),
                    keywords=keywords,
                    key_sentences=key_sentences,
                    speech_prompt=prompt,
                )
            )

        overall_keywords = [
            word for word, _ in overall_tokens.most_common(self.config.max_keywords)
        ]
        speech_opening = self._speech_opening(overall_keywords)
        speech_closing = self._speech_closing(overall_keywords)
        outline = self._speech_outline(chunks)

        return ContextSpeechExplainDocumentResult(
            chunks=chunks,
            chunk_count=len(chunks),
            truncated=truncated,
            chunk_mode=mode,
            chunk_size=size,
            overlap=overlap,
            input_bytes=len(content.encode("utf-8")),
            overall_keywords=overall_keywords,
            speech_opening=speech_opening,
            speech_closing=speech_closing,
            speech_outline=outline,
            warnings=warnings,
        )

    def _load_content(self, args: ContextSpeechExplainDocumentArgs) -> str:
        if args.content and args.path:
            raise ToolError("Provide content or path, not both.")
        if args.content is None and args.path is None:
            raise ToolError("Provide content or path.")

        if args.content is not None:
            data = args.content.encode("utf-8")
            if len(data) > self.config.max_source_bytes:
                raise ToolError(
                    f"content exceeds max_source_bytes ({len(data)} > {self.config.max_source_bytes})."
                )
            return args.content

        path = self._resolve_path(args.path or "")
        size = path.stat().st_size
        if size > self.config.max_source_bytes:
            raise ToolError(
                f"{path} exceeds max_source_bytes ({size} > {self.config.max_source_bytes})."
            )
        return path.read_text("utf-8", errors="ignore")

    def _resolve_path(self, raw_path: str) -> Path:
        if not raw_path.strip():
            raise ToolError("Path cannot be empty.")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.config.effective_workdir / path
        path = path.resolve()
        if not path.exists():
            raise ToolError(f"Path not found: {path}")
        if path.is_dir():
            raise ToolError(f"Path is a directory: {path}")
        return path

    def _build_units(self, text: str, mode: str) -> list[tuple[int, int, str]]:
        if mode == "words":
            return self._word_units(text)
        if mode == "lines":
            return self._line_units(text)
        if mode == "paragraphs":
            return self._paragraph_units(text)
        if mode == "sentences":
            return self._sentence_units(text)
        raise ToolError("chunk_mode must be one of: words, lines, paragraphs, sentences.")

    def _word_units(self, text: str) -> list[tuple[int, int, str]]:
        units = []
        for match in WORD_RE.finditer(text):
            units.append((match.start(), match.end(), match.group(0)))
        return units

    def _line_units(self, text: str) -> list[tuple[int, int, str]]:
        units: list[tuple[int, int, str]] = []
        start = 0
        for line in text.splitlines(True):
            end = start + len(line)
            if line.strip():
                units.append((start, end, line))
            start = end
        return units

    def _paragraph_units(self, text: str) -> list[tuple[int, int, str]]:
        units: list[tuple[int, int, str]] = []
        start = 0
        for match in re.finditer(r"\n\s*\n", text):
            end = match.start()
            chunk = text[start:end]
            if chunk.strip():
                units.append((start, end, chunk))
            start = match.end()
        if start < len(text):
            chunk = text[start:]
            if chunk.strip():
                units.append((start, len(text), chunk))
        return units

    def _sentence_units(self, text: str) -> list[tuple[int, int, str]]:
        units: list[tuple[int, int, str]] = []
        for match in SENTENCE_RE.finditer(text):
            chunk = match.group(0)
            if chunk.strip():
                units.append((match.start(), match.end(), chunk))
        return units

    def _joiner_for_mode(self, mode: str) -> str:
        if mode == "paragraphs":
            return "\n\n"
        if mode in {"words", "sentences"}:
            return " "
        return ""

    def _chunk_units(
        self,
        units: list[tuple[int, int, str]],
        size: int,
        overlap: int,
        joiner: str,
    ) -> list[tuple[int, int, str]]:
        chunks: list[tuple[int, int, str]] = []
        step = size - overlap
        if step <= 0:
            raise ToolError("overlap must be smaller than chunk_size.")
        for i in range(0, len(units), step):
            segment = units[i : i + size]
            if not segment:
                break
            start = segment[0][0]
            end = segment[-1][1]
            text = joiner.join(piece for _, _, piece in segment).strip()
            if text:
                chunks.append((start, end, text))
        return chunks

    def _tokenize(self, text: str) -> list[str]:
        return [
            token.lower()
            for token in WORD_RE.findall(text)
            if len(token) >= self.config.min_token_length
            and token.lower() not in STOPWORDS
        ]

    def _top_keywords(self, tokens: list[str], max_items: int) -> list[str]:
        return [word for word, _ in Counter(tokens).most_common(max_items)]

    def _key_sentences(self, text: str, keywords: list[str]) -> list[str]:
        if not keywords:
            return []
        keyword_set = {word.lower() for word in keywords}
        scored: list[tuple[float, str]] = []
        for match in SENTENCE_RE.finditer(text):
            sentence = match.group(0).strip()
            if not sentence:
                continue
            tokens = [t.lower() for t in WORD_RE.findall(sentence)]
            hit_count = sum(1 for token in tokens if token in keyword_set)
            score = hit_count * 2 + min(len(sentence) / 200, 1.0)
            scored.append((score, sentence))
        scored.sort(key=lambda item: item[0], reverse=True)
        results: list[str] = []
        for _, sentence in scored[: self.config.max_key_sentences]:
            trimmed = (
                sentence[: self.config.max_sentence_chars] + "..."
                if len(sentence) > self.config.max_sentence_chars
                else sentence
            )
            results.append(trimmed)
        return results

    def _speech_prompt(
        self, index: int, keywords: list[str], key_sentences: list[str]
    ) -> str:
        parts = [f"Chunk {index} focus"]
        if keywords:
            parts.append(f"keywords: {', '.join(keywords)}")
        if key_sentences:
            parts.append(f"key sentences: {' | '.join(key_sentences)}")
        return "; ".join(parts) + "."

    def _speech_opening(self, keywords: list[str]) -> str:
        if not keywords:
            return "Introduce the document and its main themes."
        return (
            "Introduce the document and its main themes: "
            + ", ".join(keywords[: self.config.max_keywords])
            + "."
        )

    def _speech_closing(self, keywords: list[str]) -> str:
        if not keywords:
            return "Summarize the main points and provide a closing remark."
        return (
            "Summarize the main points and restate key themes: "
            + ", ".join(keywords[: self.config.max_keywords])
            + "."
        )

    def _speech_outline(self, chunks: list[SpeechChunk]) -> list[SpeechCue]:
        outline: list[SpeechCue] = []
        for idx, chunk in enumerate(chunks, start=1):
            outline.append(
                SpeechCue(
                    index=idx,
                    chunk_index=chunk.index,
                    cue=chunk.speech_prompt,
                )
            )
        return outline

    def _preview(self, text: str) -> str:
        max_chars = self.config.preview_chars
        if max_chars <= 0:
            return ""
        return text if len(text) <= max_chars else text[:max_chars]

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextSpeechExplainDocumentArgs):
            return ToolCallDisplay(summary="context_speech_explain_document")
        return ToolCallDisplay(
            summary="context_speech_explain_document",
            details={
                "path": event.args.path,
                "chunk_mode": event.args.chunk_mode,
                "chunk_size": event.args.chunk_size,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpeechExplainDocumentResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=True,
            message=(
                f"Prepared speech outline with {event.result.chunk_count} chunk(s)"
            ),
            warnings=event.result.warnings,
            details={
                "chunk_count": event.result.chunk_count,
                "truncated": event.result.truncated,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Preparing speech explanation for document"
