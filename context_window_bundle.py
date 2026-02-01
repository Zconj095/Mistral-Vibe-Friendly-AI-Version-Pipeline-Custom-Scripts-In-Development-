from __future__ import annotations

import importlib.util
import json
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
import urllib.error
import urllib.request

from pydantic import BaseModel, Field

try:
    from actions_lib import validate_args
except ModuleNotFoundError:  # Fallback when tools directory is not on sys.path.
    _actions_path = Path(__file__).with_name("actions_lib.py")
    _spec = importlib.util.spec_from_file_location("actions_lib", _actions_path)
    if not _spec or not _spec.loader:
        raise
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    validate_args = _module.validate_args

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


DEFAULT_DB_PATH = Path.home() / ".vibe" / "vectorstores" / "pdf_vectors.sqlite"

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "keywords": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class _QueryResult:
    score: float
    source_path: str
    chunk_index: int
    unit: str
    start_index: int | None
    end_index: int | None
    content: str
    query: str


class MessageItem(BaseModel):
    role: str
    content: str


class ConversationSummary(BaseModel):
    summary: str
    key_points: list[str] | None = None
    keywords: list[str] | None = None


class RetrievalItem(BaseModel):
    score: float
    source_path: str
    chunk_index: int
    unit: str
    start_index: int | None
    end_index: int | None
    content: str
    query: str


class ContextWindowBundleArgs(BaseModel):
    messages: list[MessageItem] = Field(description="Conversation messages.")
    question: str | None = Field(
        default=None, description="Optional question for retrieval and summary."
    )
    recent_messages: int | None = Field(
        default=None, description="Number of recent messages to keep verbatim."
    )
    include_summary: bool = Field(
        default=True, description="Include a summary of older messages."
    )
    include_recent: bool = Field(
        default=True, description="Include recent messages in the bundle."
    )
    include_retrieval: bool = Field(
        default=True, description="Include retrieval context."
    )
    retrieval_queries: list[str] | None = Field(
        default=None, description="Override retrieval queries."
    )
    db_path: str | None = Field(default=None, description="Override database path.")
    embedding_model: str | None = Field(
        default=None, description="Override embedding model."
    )
    top_k: int | None = Field(
        default=None, description="Max retrieval results."
    )
    min_score: float | None = Field(
        default=None, description="Minimum retrieval similarity score."
    )
    max_result_chars: int | None = Field(
        default=None, description="Max chars per retrieval snippet."
    )
    memory_items: list[str] | None = Field(
        default=None, description="Inline memory notes."
    )
    memory_paths: list[str] | None = Field(
        default=None, description="Paths to memory note files."
    )
    max_memory_chars: int | None = Field(
        default=None, description="Max chars per memory note."
    )
    max_bundle_chars: int | None = Field(
        default=None, description="Max total bundle characters."
    )
    llm_api_base: str | None = Field(
        default=None, description="OpenAI-compatible API base URL."
    )
    llm_model: str | None = Field(default=None, description="LLM model name.")
    llm_temperature: float = Field(default=0.2, description="LLM temperature.")
    llm_max_tokens: int = Field(default=600, description="LLM max tokens.")
    llm_stream: bool = Field(default=False, description="Stream LLM output.")
    max_retries: int | None = Field(
        default=None, description="Max retries for JSON output."
    )


class ContextWindowBundleResult(BaseModel):
    summary: ConversationSummary | None
    recent_messages: list[MessageItem]
    retrieval: list[RetrievalItem]
    memory_notes: list[str]
    bundle_text: str
    warnings: list[str]
    errors: list[str]


class ContextWindowBundleConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    recent_messages: int = Field(default=8, description="Recent messages to keep.")
    llm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL.",
    )
    llm_model: str = Field(
        default="gpt-oss:latest", description="Default LLM model name."
    )
    max_retries: int = Field(default=2, description="Max JSON retries.")
    db_path: Path = Field(
        default=DEFAULT_DB_PATH,
        description="Path to the sqlite database file.",
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        description="Embedding model to use with Ollama/GPT-OSS.",
    )
    max_results: int = Field(
        default=5, description="Maximum number of retrieval results."
    )
    max_result_chars: int = Field(
        default=800, description="Maximum characters per retrieval snippet."
    )
    max_memory_chars: int = Field(
        default=2000, description="Maximum characters per memory note."
    )
    max_bundle_chars: int = Field(
        default=16000, description="Maximum total bundle characters."
    )
    ollama_url: str = Field(
        default="http://127.0.0.1:11434",
        description="Base URL for the local Ollama server.",
    )


class ContextWindowBundleState(BaseToolState):
    pass


class ContextWindowBundle(
    BaseTool[
        ContextWindowBundleArgs,
        ContextWindowBundleResult,
        ContextWindowBundleConfig,
        ContextWindowBundleState,
    ],
    ToolUIData[ContextWindowBundleArgs, ContextWindowBundleResult],
):
    description: ClassVar[str] = (
        "Build a long-context bundle with summaries, retrieval, and memory notes."
    )

    async def run(
        self, args: ContextWindowBundleArgs
    ) -> ContextWindowBundleResult:
        if not args.messages:
            raise ToolError("messages is required.")

        warnings: list[str] = []
        errors: list[str] = []

        summary, recent = self._summarize_window(args, warnings, errors)
        retrieval = self._retrieve_context(args, warnings)
        memory_notes = self._collect_memory(args, warnings)
        bundle_text = self._build_bundle(
            summary, recent, retrieval, memory_notes, args
        )

        return ContextWindowBundleResult(
            summary=summary,
            recent_messages=recent,
            retrieval=retrieval,
            memory_notes=memory_notes,
            bundle_text=bundle_text,
            warnings=warnings,
            errors=errors,
        )

    def _summarize_window(
        self,
        args: ContextWindowBundleArgs,
        warnings: list[str],
        errors: list[str],
    ) -> tuple[ConversationSummary | None, list[MessageItem]]:
        recent_count = (
            args.recent_messages
            if args.recent_messages is not None
            else self.config.recent_messages
        )
        if recent_count < 0:
            raise ToolError("recent_messages must be >= 0.")

        messages = [MessageItem(**msg.model_dump()) for msg in args.messages]
        if not args.include_recent:
            recent = []
            older = messages
        elif recent_count == 0:
            recent = []
            older = messages
        else:
            recent = messages[-recent_count:]
            older = messages[:-recent_count]

        if not args.include_summary or not older:
            return None, recent

        transcript = "\n".join(
            f"{msg.role}: {msg.content.strip()}" for msg in older if msg.content.strip()
        )
        if not transcript:
            return None, recent

        prompt = (
            "Summarize the conversation history. Focus on decisions, facts, and open questions."
        )
        if args.question:
            prompt += f" Prioritize details relevant to: {args.question}"

        summary_payload, summary_errors = self._summarize_text(
            transcript,
            args,
            prompt_prefix=prompt,
        )
        if summary_errors:
            errors.extend(summary_errors)
        if summary_payload is None:
            return None, recent

        summary = ConversationSummary(
            summary=summary_payload.get("summary", ""),
            key_points=summary_payload.get("key_points"),
            keywords=summary_payload.get("keywords"),
        )
        return summary, recent

    def _summarize_text(
        self,
        text: str,
        args: ContextWindowBundleArgs,
        prompt_prefix: str,
    ) -> tuple[dict | None, list[str]]:
        errors: list[str] = []
        if not text.strip():
            return None, ["Empty input for summary."]

        system_prompt = (
            "Reply ONLY with valid JSON that conforms to the JSON schema below. "
            "Do not include extra keys, comments, or markdown.\n"
            f"JSON schema:\n{json.dumps(SUMMARY_SCHEMA, ensure_ascii=True)}"
        )
        user_prompt = f"{prompt_prefix}\n\n{text}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload, call_errors = self._call_llm_json(messages, args)
        errors.extend(call_errors)
        if not isinstance(payload, dict):
            return None, errors
        return payload, errors

    def _call_llm_json(
        self, messages: list[dict[str, str]], args: ContextWindowBundleArgs
    ) -> tuple[dict | None, list[str]]:
        errors: list[str] = []
        max_retries = (
            args.max_retries if args.max_retries is not None else self.config.max_retries
        )

        for attempt in range(max_retries + 1):
            raw = self._call_llm(messages, args)
            parsed, parse_error = self._parse_json(raw)
            if parse_error:
                errors.append(parse_error)
                if attempt < max_retries:
                    messages = self._append_retry(messages, raw, parse_error)
                    continue
                return None, errors

            validation_errors = validate_args(SUMMARY_SCHEMA, parsed)
            if validation_errors:
                errors.extend(validation_errors)
                if attempt < max_retries:
                    messages = self._append_retry(
                        messages, raw, "; ".join(validation_errors)
                    )
                    continue
                return None, errors

            return parsed, errors
        return None, errors

    def _append_retry(
        self, messages: list[dict[str, str]], raw: str, error: str
    ) -> list[dict[str, str]]:
        retry_prompt = (
            "Your previous output was invalid.\n"
            f"Error: {error}\n"
            "Reply again with ONLY valid JSON that matches the schema."
        )
        return messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": retry_prompt},
        ]

    def _parse_json(self, raw: str) -> tuple[Any | None, str | None]:
        text = raw.strip()
        if not text:
            return None, "Empty output"
        try:
            return json.loads(text), None
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            start = min(
                [i for i in (text.find("{"), text.find("[")) if i != -1],
                default=-1,
            )
            if start == -1:
                return None, "No JSON object or array found"
            try:
                obj, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError as exc:
                return None, f"JSON parse error: {exc}"
            return obj, None

    def _call_llm(
        self, messages: list[dict[str, str]], args: ContextWindowBundleArgs
    ) -> str:
        api_base = (args.llm_api_base or self.config.llm_api_base).rstrip("/")
        url = api_base + "/chat/completions"
        payload = {
            "model": args.llm_model or self.config.llm_model,
            "messages": messages,
            "temperature": args.llm_temperature,
            "max_tokens": args.llm_max_tokens,
            "stream": bool(args.llm_stream),
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=600)
        except urllib.error.URLError as exc:
            raise ToolError(f"LLM request failed: {exc}") from exc

        if not args.llm_stream:
            body = resp.read().decode("utf-8")
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError as exc:
                raise ToolError(f"LLM response parse failed: {exc}") from exc
            return parsed["choices"][0]["message"].get("content", "").strip()

        return self._read_streaming_response(resp)

    def _read_streaming_response(self, resp) -> str:
        parts: list[str] = []
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            if line.startswith("data:"):
                line = line[len("data:") :].strip()
            if line == "[DONE]":
                break
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            choice = chunk.get("choices", [{}])[0]
            delta = choice.get("delta") or choice.get("message") or {}
            content = delta.get("content")
            if content:
                parts.append(content)
                sys.stdout.write(content)
                sys.stdout.flush()
        if parts:
            sys.stdout.write("\n")
            sys.stdout.flush()
        return "".join(parts).strip()

    def _retrieve_context(
        self, args: ContextWindowBundleArgs, warnings: list[str]
    ) -> list[RetrievalItem]:
        if not args.include_retrieval:
            return []

        queries = self._resolve_queries(args, warnings)
        if not queries:
            return []

        db_path = Path(args.db_path).expanduser() if args.db_path else self.config.db_path
        if not db_path.exists():
            warnings.append(f"Retrieval database not found: {db_path}")
            return []

        embedding_model = args.embedding_model or self.config.embedding_model
        top_k = args.top_k if args.top_k is not None else self.config.max_results
        if top_k <= 0:
            raise ToolError("top_k must be a positive integer.")
        max_chars = (
            args.max_result_chars
            if args.max_result_chars is not None
            else self.config.max_result_chars
        )
        if max_chars <= 0:
            raise ToolError("max_result_chars must be positive.")
        min_score = args.min_score
        if min_score is not None and (min_score < -1.0 or min_score > 1.0):
            raise ToolError("min_score must be between -1.0 and 1.0.")

        results: list[_QueryResult] = []
        for query in queries:
            results.extend(
                self._search_db(
                    db_path,
                    embedding_model,
                    query,
                    top_k,
                    min_score,
                    max_chars,
                    warnings,
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        if len(results) > top_k:
            results = results[:top_k]

        return [
            RetrievalItem(
                score=item.score,
                source_path=item.source_path,
                chunk_index=item.chunk_index,
                unit=item.unit,
                start_index=item.start_index,
                end_index=item.end_index,
                content=item.content,
                query=item.query,
            )
            for item in results
        ]

    def _resolve_queries(
        self, args: ContextWindowBundleArgs, warnings: list[str]
    ) -> list[str]:
        queries: list[str] = []
        if args.retrieval_queries:
            queries.extend([q.strip() for q in args.retrieval_queries if q and q.strip()])

        if not queries and args.question:
            queries.append(args.question.strip())

        if not queries:
            last_user = next(
                (msg.content.strip() for msg in reversed(args.messages) if msg.role == "user"),
                "",
            )
            if last_user:
                queries.append(last_user)
            else:
                warnings.append("No retrieval query available.")

        return queries

    def _search_db(
        self,
        db_path: Path,
        embedding_model: str,
        query: str,
        top_k: int,
        min_score: float | None,
        max_chars: int,
        warnings: list[str],
    ) -> list[_QueryResult]:
        query_vec = self._embed_text(embedding_model, query, warnings)
        if query_vec is None:
            return []

        results: list[tuple[float, _QueryResult]] = []
        import sqlite3

        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute(
                """
                SELECT source_path, chunk_index, unit, start_index, end_index, content, embedding
                FROM pdf_chunks
                """
            )
            for row in cursor:
                embedding = self._unpack_embedding(row[6])
                if not embedding:
                    continue
                score = self._dot(query_vec, embedding)
                if min_score is not None and score < min_score:
                    continue
                item = _QueryResult(
                    score=score,
                    source_path=row[0],
                    chunk_index=row[1],
                    unit=row[2],
                    start_index=row[3],
                    end_index=row[4],
                    content=row[5][:max_chars],
                    query=query,
                )
                if len(results) < top_k:
                    results.append((score, item))
                    results.sort(key=lambda r: r[0])
                else:
                    if score > results[0][0]:
                        results[0] = (score, item)
                        results.sort(key=lambda r: r[0])

        return [item for _, item in sorted(results, key=lambda r: r[0], reverse=True)]

    def _embed_text(
        self, model: str, text: str, warnings: list[str]
    ) -> list[float] | None:
        payload = json.dumps({"model": model, "prompt": text}).encode("utf-8")
        url = self.config.ollama_url.rstrip("/") + "/api/embeddings"
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            warnings.append(f"Ollama embeddings failed: {exc}")
            return None

        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            warnings.append("Invalid embeddings response from Ollama.")
            return None

        return self._normalize_embedding([float(x) for x in embedding])

    def _normalize_embedding(self, embedding: list[float]) -> list[float]:
        norm = sum(x * x for x in embedding) ** 0.5
        if norm == 0:
            return embedding
        return [x / norm for x in embedding]

    def _unpack_embedding(self, blob: bytes) -> list[float]:
        arr = array("f")
        arr.frombytes(blob)
        return list(arr)

    def _dot(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            return 0.0
        return sum(l * r for l, r in zip(left, right))

    def _collect_memory(
        self, args: ContextWindowBundleArgs, warnings: list[str]
    ) -> list[str]:
        notes: list[str] = []
        max_chars = (
            args.max_memory_chars
            if args.max_memory_chars is not None
            else self.config.max_memory_chars
        )
        if max_chars <= 0:
            raise ToolError("max_memory_chars must be positive.")

        if args.memory_items:
            for item in args.memory_items:
                if not item.strip():
                    continue
                notes.append(self._truncate_text(item, max_chars))

        if args.memory_paths:
            for raw_path in args.memory_paths:
                path = self._resolve_path(raw_path)
                try:
                    content = path.read_text("utf-8", errors="ignore")
                except OSError as exc:
                    warnings.append(f"Failed to read memory file: {exc}")
                    continue
                if not content.strip():
                    continue
                notes.append(self._truncate_text(content, max_chars))

        return notes

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

    def _truncate_text(self, text: str, max_chars: int) -> str:
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "..."

    def _build_bundle(
        self,
        summary: ConversationSummary | None,
        recent: list[MessageItem],
        retrieval: list[RetrievalItem],
        memory_notes: list[str],
        args: ContextWindowBundleArgs,
    ) -> str:
        parts: list[str] = []

        if summary:
            parts.append("Conversation summary:")
            parts.append(summary.summary.strip())
            if summary.key_points:
                parts.append("Key points:")
                parts.extend([f"- {item}" for item in summary.key_points if item.strip()])
            if summary.keywords:
                parts.append("Keywords: " + ", ".join(summary.keywords))

        if args.include_recent and recent:
            parts.append("Recent messages:")
            for msg in recent:
                parts.append(f"{msg.role}: {msg.content.strip()}")

        if memory_notes:
            parts.append("Memory notes:")
            parts.extend([f"- {note.strip()}" for note in memory_notes if note.strip()])

        if retrieval:
            parts.append("Retrieved context:")
            for item in retrieval:
                snippet = item.content.strip()
                parts.append(
                    f"- {item.source_path} (chunk {item.chunk_index}, score {item.score:.3f})"
                )
                parts.append(snippet)

        bundle = "\n".join(parts).strip()
        max_bundle = (
            args.max_bundle_chars
            if args.max_bundle_chars is not None
            else self.config.max_bundle_chars
        )
        if max_bundle <= 0:
            return bundle
        return self._truncate_text(bundle, max_bundle)

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextWindowBundleArgs):
            return ToolCallDisplay(summary="context_window_bundle")
        return ToolCallDisplay(
            summary="context_window_bundle",
            details={
                "messages": len(event.args.messages),
                "recent_messages": event.args.recent_messages,
                "include_retrieval": event.args.include_retrieval,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextWindowBundleResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = "Context window bundle ready"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "summary": event.result.summary,
                "recent_messages": event.result.recent_messages,
                "retrieval": event.result.retrieval,
                "memory_notes": event.result.memory_notes,
                "bundle_text": event.result.bundle_text,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Building context window bundle"
