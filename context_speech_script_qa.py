from __future__ import annotations

import re
from collections import Counter
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


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
FUNC_DEF_RE = re.compile(
    r"^\s*(?:async\s+)?(def|function|fn|func)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.M,
)
CLASS_DEF_RE = re.compile(
    r"^\s*(class|struct|interface|record)\s+([A-Za-z_][A-Za-z0-9_]*)", re.M
)

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

CODE_KEYWORDS = {
    "abstract",
    "async",
    "await",
    "bool",
    "break",
    "case",
    "catch",
    "class",
    "const",
    "continue",
    "def",
    "default",
    "delete",
    "do",
    "elif",
    "else",
    "enum",
    "export",
    "extends",
    "false",
    "finally",
    "for",
    "foreach",
    "from",
    "function",
    "if",
    "import",
    "in",
    "interface",
    "let",
    "match",
    "namespace",
    "new",
    "null",
    "public",
    "private",
    "protected",
    "raise",
    "record",
    "return",
    "self",
    "static",
    "struct",
    "super",
    "switch",
    "this",
    "throw",
    "true",
    "try",
    "typedef",
    "var",
    "void",
    "while",
    "with",
}

IGNORED_TOKENS = STOPWORDS | CODE_KEYWORDS


class ContextSpeechScriptQAConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_source_bytes: int = Field(
        default=50_000_000, description="Maximum bytes per script."
    )
    max_chunks: int = Field(
        default=2000, description="Maximum chunks to return (0 for unlimited)."
    )
    max_chunks_per_question: int = Field(
        default=5, description="Maximum relevant chunks per question."
    )
    preview_chars: int = Field(default=240, description="Preview snippet length.")
    default_chunk_size: int = Field(default=400, description="Default chunk size.")
    default_overlap: int = Field(default=40, description="Default chunk overlap.")
    min_token_length: int = Field(default=3, description="Minimum token length.")
    max_keywords: int = Field(default=12, description="Max keywords per chunk.")
    max_symbols: int = Field(default=12, description="Max symbols per chunk.")
    max_definitions: int = Field(
        default=10, description="Max functions/classes per chunk."
    )


class ContextSpeechScriptQAState(BaseToolState):
    pass


class ContextSpeechScriptQAArgs(BaseModel):
    content: str | None = Field(default=None, description="Script content.")
    path: str | None = Field(default=None, description="Path to a script.")
    question: str | None = Field(default=None, description="Single question.")
    questions: list[str] | None = Field(default=None, description="Question list.")
    chunk_size: int | None = Field(default=None, description="Chunk size in lines.")
    overlap: int | None = Field(default=None, description="Chunk overlap in lines.")
    max_chunks: int | None = Field(
        default=None, description="Override max chunks (0 for unlimited)."
    )
    max_chunks_per_question: int | None = Field(
        default=None, description="Override relevant chunks per question."
    )


class ScriptChunk(BaseModel):
    index: int
    start_line: int
    end_line: int
    line_count: int
    preview: str
    keywords: list[str]
    symbols: list[str]
    functions: list[str]
    classes: list[str]


class QuestionAnswer(BaseModel):
    question: str
    relevant_chunks: list[int]
    cues: list[str]
    speech_answer: str


class ContextSpeechScriptQAResult(BaseModel):
    total_lines: int
    chunk_count: int
    truncated: bool
    chunks: list[ScriptChunk]
    question_answers: list[QuestionAnswer]
    speech_opening: str
    speech_closing: str
    warnings: list[str]


class ContextSpeechScriptQA(
    BaseTool[
        ContextSpeechScriptQAArgs,
        ContextSpeechScriptQAResult,
        ContextSpeechScriptQAConfig,
        ContextSpeechScriptQAState,
    ],
    ToolUIData[ContextSpeechScriptQAArgs, ContextSpeechScriptQAResult],
):
    description: ClassVar[str] = (
        "Chunk a large script and prepare speech answers to questions about it."
    )

    async def run(self, args: ContextSpeechScriptQAArgs) -> ContextSpeechScriptQAResult:
        content = self._load_content(args)
        lines = content.splitlines()
        total_lines = len(lines)

        chunk_size = args.chunk_size or self.config.default_chunk_size
        overlap = (
            args.overlap if args.overlap is not None else self.config.default_overlap
        )
        max_chunks = args.max_chunks if args.max_chunks is not None else self.config.max_chunks
        max_chunks_per_question = (
            args.max_chunks_per_question
            if args.max_chunks_per_question is not None
            else self.config.max_chunks_per_question
        )
        warnings: list[str] = []

        if chunk_size <= 0:
            raise ToolError("chunk_size must be positive.")
        if overlap < 0:
            raise ToolError("overlap must be non-negative.")
        if overlap >= chunk_size:
            raise ToolError("overlap must be smaller than chunk_size.")
        if max_chunks <= 0:
            max_chunks = None
        if max_chunks_per_question <= 0:
            raise ToolError("max_chunks_per_question must be positive.")

        chunks, truncated = self._chunk_lines(
            lines, chunk_size, overlap, max_chunks
        )
        if truncated:
            warnings.append("Chunk limit reached; output truncated.")

        question_list = self._normalize_questions(args)
        if not question_list:
            warnings.append("No questions provided.")

        question_answers = self._build_answers(
            question_list, chunks, max_chunks_per_question
        )

        speech_opening = self._speech_opening(total_lines, len(chunks))
        speech_closing = self._speech_closing(question_list)

        return ContextSpeechScriptQAResult(
            total_lines=total_lines,
            chunk_count=len(chunks),
            truncated=truncated,
            chunks=chunks,
            question_answers=question_answers,
            speech_opening=speech_opening,
            speech_closing=speech_closing,
            warnings=warnings,
        )

    def _load_content(self, args: ContextSpeechScriptQAArgs) -> str:
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

    def _chunk_lines(
        self,
        lines: list[str],
        chunk_size: int,
        overlap: int,
        max_chunks: int | None,
    ) -> tuple[list[ScriptChunk], bool]:
        chunks: list[ScriptChunk] = []
        truncated = False
        step = chunk_size - overlap
        if step <= 0:
            raise ToolError("overlap must be smaller than chunk_size.")

        total_lines = len(lines)
        for idx, start in enumerate(range(0, total_lines, step), start=1):
            if max_chunks is not None and idx > max_chunks:
                truncated = True
                break
            segment = lines[start : start + chunk_size]
            if not segment:
                break
            text = "\n".join(segment)
            chunk = self._build_chunk(idx, start, segment, text)
            chunks.append(chunk)

        return chunks, truncated

    def _build_chunk(
        self, index: int, start_index: int, segment: list[str], text: str
    ) -> ScriptChunk:
        start_line = start_index + 1
        end_line = start_line + len(segment) - 1
        keywords = self._extract_keywords(text, self.config.max_keywords)
        symbols = self._extract_symbols(text, self.config.max_symbols)
        functions = self._extract_definitions(FUNC_DEF_RE, text)
        classes = self._extract_definitions(CLASS_DEF_RE, text)
        return ScriptChunk(
            index=index,
            start_line=start_line,
            end_line=end_line,
            line_count=len(segment),
            preview=self._preview(text),
            keywords=keywords,
            symbols=symbols,
            functions=functions,
            classes=classes,
        )

    def _extract_keywords(self, text: str, max_items: int) -> list[str]:
        tokens = []
        for token in TOKEN_RE.findall(text):
            lower = token.lower()
            if len(lower) < self.config.min_token_length:
                continue
            if lower in IGNORED_TOKENS:
                continue
            tokens.append(lower)
        return [word for word, _ in Counter(tokens).most_common(max_items)]

    def _extract_symbols(self, text: str, max_items: int) -> list[str]:
        counts: Counter[str] = Counter()
        case_map: dict[str, str] = {}
        for token in TOKEN_RE.findall(text):
            lower = token.lower()
            if len(lower) < self.config.min_token_length:
                continue
            if lower in IGNORED_TOKENS:
                continue
            counts[lower] += 1
            if lower not in case_map:
                case_map[lower] = token
        return [
            case_map[word] for word, _ in counts.most_common(max_items)
        ]

    def _extract_definitions(self, pattern: re.Pattern[str], text: str) -> list[str]:
        names: list[str] = []
        for match in pattern.findall(text):
            if isinstance(match, tuple):
                name = match[1]
            else:
                name = match
            if name not in names:
                names.append(name)
            if len(names) >= self.config.max_definitions:
                break
        return names

    def _normalize_questions(self, args: ContextSpeechScriptQAArgs) -> list[str]:
        questions: list[str] = []
        if args.question:
            questions.append(args.question.strip())
        if args.questions:
            questions.extend(q.strip() for q in args.questions if q and q.strip())
        return [q for q in questions if q]

    def _build_answers(
        self,
        questions: list[str],
        chunks: list[ScriptChunk],
        max_chunks_per_question: int,
    ) -> list[QuestionAnswer]:
        answers: list[QuestionAnswer] = []
        for question in questions:
            scored = self._score_chunks(question, chunks)
            selected = [idx for idx, score in scored if score > 0]
            if not selected:
                selected = [chunk.index for chunk in chunks[:1]]
            selected = selected[:max_chunks_per_question]
            cues = self._build_cues(question, chunks, selected)
            speech = self._speech_answer(question, selected, cues)
            answers.append(
                QuestionAnswer(
                    question=question,
                    relevant_chunks=selected,
                    cues=cues,
                    speech_answer=speech,
                )
            )
        return answers

    def _score_chunks(
        self, question: str, chunks: list[ScriptChunk]
    ) -> list[tuple[int, int]]:
        question_tokens = self._question_tokens(question)
        results: list[tuple[int, int]] = []
        for chunk in chunks:
            score = 0
            score += 2 * len(question_tokens & set(chunk.keywords))
            score += 3 * len(question_tokens & self._lower_set(chunk.symbols))
            score += 4 * len(question_tokens & self._lower_set(chunk.functions))
            score += 4 * len(question_tokens & self._lower_set(chunk.classes))
            results.append((chunk.index, score))
        results.sort(key=lambda item: item[1], reverse=True)
        return results

    def _question_tokens(self, question: str) -> set[str]:
        tokens = set()
        for token in TOKEN_RE.findall(question):
            lower = token.lower()
            if len(lower) < self.config.min_token_length:
                continue
            if lower in IGNORED_TOKENS:
                continue
            tokens.add(lower)
        return tokens

    def _lower_set(self, values: list[str]) -> set[str]:
        return {value.lower() for value in values}

    def _build_cues(
        self, question: str, chunks: list[ScriptChunk], selected: list[int]
    ) -> list[str]:
        cues: list[str] = []
        cues.append(f"Question focus: {question}")
        for chunk in chunks:
            if chunk.index not in selected:
                continue
            parts = [f"Chunk {chunk.index} lines {chunk.start_line}-{chunk.end_line}"]
            if chunk.functions:
                parts.append(f"functions: {', '.join(chunk.functions[:4])}")
            if chunk.classes:
                parts.append(f"classes: {', '.join(chunk.classes[:4])}")
            if chunk.keywords:
                parts.append(f"keywords: {', '.join(chunk.keywords[:6])}")
            cues.append("; ".join(parts))
        return cues

    def _speech_answer(
        self, question: str, selected: list[int], cues: list[str]
    ) -> str:
        cue_text = " | ".join(cues[1:]) if len(cues) > 1 else ""
        base = f"Answer: {question}. Reference chunks {', '.join(map(str, selected))}."
        if cue_text:
            return f"{base} Use cues: {cue_text}."
        return base

    def _speech_opening(self, total_lines: int, chunk_count: int) -> str:
        return (
            f"We are reviewing a script with {total_lines} lines across "
            f"{chunk_count} sections."
        )

    def _speech_closing(self, questions: list[str]) -> str:
        if not questions:
            return "Summarize the main modules and offer to answer follow-up questions."
        return "Wrap up by confirming each question was addressed and offer next steps."

    def _preview(self, text: str) -> str:
        max_chars = self.config.preview_chars
        if max_chars <= 0:
            return ""
        return text if len(text) <= max_chars else text[:max_chars]

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextSpeechScriptQAArgs):
            return ToolCallDisplay(summary="context_speech_script_qa")
        return ToolCallDisplay(
            summary="context_speech_script_qa",
            details={
                "path": event.args.path,
                "chunk_size": event.args.chunk_size,
                "overlap": event.args.overlap,
                "question": event.args.question,
                "questions": event.args.questions,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpeechScriptQAResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = (
            f"Prepared {event.result.chunk_count} chunk(s) and "
            f"{len(event.result.question_answers)} answer prompt(s)"
        )
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=event.result.warnings,
            details={
                "chunk_count": event.result.chunk_count,
                "total_lines": event.result.total_lines,
                "truncated": event.result.truncated,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Preparing speech answers for script"
