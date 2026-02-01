
from __future__ import annotations

import fnmatch
import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Iterable
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
class _Chunk:
    source_id: str
    source_path: str | None
    unit: str
    chunk_index: int
    start_index: int | None
    end_index: int | None
    content: str


class LongContextItem(BaseModel):
    id: str | None = Field(default=None, description="Optional item id.")
    path: str | None = Field(default=None, description="Path to a file.")
    content: str | None = Field(default=None, description="Inline content.")
    source: str | None = Field(default=None, description="Source label.")


class MessageItem(BaseModel):
    role: str
    content: str


class ChunkSummary(BaseModel):
    summary_id: str
    source_id: str
    source_path: str | None
    chunk_index: int
    unit: str
    start_index: int | None
    end_index: int | None
    summary: str
    key_points: list[str] | None = None
    keywords: list[str] | None = None


class LevelSummary(BaseModel):
    level: int
    summary_id: str
    source_ids: list[str]
    summary: str
    key_points: list[str] | None = None
    keywords: list[str] | None = None


class ConversationSummary(BaseModel):
    summary: str
    key_points: list[str] | None = None
    keywords: list[str] | None = None


class ContextLongContextArgs(BaseModel):
    action: str | None = Field(
        default="summarize",
        description="summarize or window.",
    )
    question: str | None = Field(
        default=None, description="Optional focus question for summaries."
    )
    items: list[LongContextItem] | None = Field(
        default=None, description="Inline items to summarize."
    )
    paths: list[str] | None = Field(
        default=None, description="Files or directories to summarize."
    )
    include_globs: list[str] | None = Field(
        default=None, description="Optional include globs for directories."
    )
    exclude_globs: list[str] | None = Field(
        default=None, description="Optional exclude globs for directories."
    )
    max_files: int | None = Field(
        default=None, description="Maximum files to gather."
    )
    max_source_bytes: int | None = Field(
        default=None, description="Maximum bytes per source."
    )
    max_total_bytes: int | None = Field(
        default=None, description="Maximum total bytes across sources."
    )
    max_item_chars: int | None = Field(
        default=None, description="Maximum chars per item."
    )
    chunk_unit: str | None = Field(
        default=None, description="chars or lines."
    )
    chunk_size: int | None = Field(
        default=None, description="Chunk size in units."
    )
    overlap: int | None = Field(
        default=None, description="Overlap in units."
    )
    max_chunks_per_item: int | None = Field(
        default=None, description="Maximum chunks per item."
    )
    hierarchy_levels: int | None = Field(
        default=None, description="Number of summary hierarchy levels."
    )
    summary_group_size: int | None = Field(
        default=None, description="Chunk summaries per group."
    )
    llm_api_base: str | None = Field(
        default=None, description="OpenAI-compatible API base URL."
    )
    llm_model: str | None = Field(
        default=None, description="LLM model name."
    )
    llm_temperature: float = Field(default=0.2, description="LLM temperature.")
    llm_max_tokens: int = Field(default=600, description="LLM max tokens.")
    llm_stream: bool = Field(default=False, description="Stream LLM output.")
    max_retries: int | None = Field(
        default=None, description="Max retries for JSON output."
    )
    messages: list[MessageItem] | None = Field(
        default=None, description="Conversation messages for window action."
    )
    recent_messages: int | None = Field(
        default=None, description="Number of recent messages to keep."
    )


class ContextLongContextResult(BaseModel):
    mode: str
    summary_at: str
    items_count: int
    chunks_count: int
    chunk_summaries: list[ChunkSummary]
    level_summaries: list[LevelSummary]
    final_summary: str | None
    conversation_summary: ConversationSummary | None
    recent_messages: list[MessageItem] | None
    warnings: list[str]
    errors: list[str]


class ContextLongContextConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    max_source_bytes: int = Field(
        default=5_000_000, description="Maximum size per source (bytes)."
    )
    max_total_bytes: int = Field(
        default=20_000_000, description="Maximum total size across sources (bytes)."
    )
    max_item_chars: int = Field(
        default=200_000, description="Maximum characters per item."
    )
    max_chunk_chars: int = Field(
        default=6000, description="Maximum chars per chunk summary input."
    )
    chunk_unit: str = Field(default="chars", description="chars or lines.")
    chunk_size: int = Field(default=2000, description="Chunk size in units.")
    overlap: int = Field(default=200, description="Overlap in units.")
    max_chunks_per_item: int = Field(
        default=500, description="Maximum chunks per item."
    )
    hierarchy_levels: int = Field(
        default=2, description="Number of summary hierarchy levels."
    )
    summary_group_size: int = Field(
        default=8, description="Chunk summaries per group."
    )
    fallback_summary_chars: int = Field(
        default=400, description="Fallback summary length when LLM fails."
    )
    max_files: int = Field(default=500, description="Maximum files to gather.")
    default_include_globs: list[str] = Field(
        default=[
            "**/*.txt",
            "**/*.md",
            "**/*.rst",
            "**/*.py",
            "**/*.js",
            "**/*.ts",
            "**/*.tsx",
            "**/*.jsx",
            "**/*.json",
            "**/*.yaml",
            "**/*.yml",
            "**/*.toml",
            "**/*.csv",
        ],
        description="Default glob patterns when include_globs not provided.",
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
    llm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL.",
    )
    llm_model: str = Field(
        default="gpt-oss:latest", description="Default LLM model name."
    )
    max_retries: int = Field(default=2, description="Max JSON retries.")


class ContextLongContextState(BaseToolState):
    pass


class ContextLongContext(
    BaseTool[
        ContextLongContextArgs,
        ContextLongContextResult,
        ContextLongContextConfig,
        ContextLongContextState,
    ],
    ToolUIData[ContextLongContextArgs, ContextLongContextResult],
):
    description: ClassVar[str] = (
        "Summarize large contexts into hierarchical summaries or rolling windows."
    )

    async def run(
        self, args: ContextLongContextArgs
    ) -> ContextLongContextResult:
        mode = (args.action or "summarize").strip().lower()
        if mode not in {"summarize", "window"}:
            raise ToolError("action must be summarize or window.")

        warnings: list[str] = []
        errors: list[str] = []

        chunk_summaries: list[ChunkSummary] = []
        level_summaries: list[LevelSummary] = []
        final_summary: str | None = None
        conversation_summary: ConversationSummary | None = None
        recent_messages: list[MessageItem] | None = None

        summary_at = datetime.now(timezone.utc).isoformat()

        if mode == "window":
            conversation_summary, recent_messages, win_errors = self._summarize_window(
                args
            )
            errors.extend(win_errors)
            return ContextLongContextResult(
                mode=mode,
                summary_at=summary_at,
                items_count=0,
                chunks_count=0,
                chunk_summaries=[],
                level_summaries=[],
                final_summary=None,
                conversation_summary=conversation_summary,
                recent_messages=recent_messages,
                warnings=warnings,
                errors=errors,
            )

        items, gather_warnings = self._gather_items(args)
        warnings.extend(gather_warnings)
        if not items:
            raise ToolError("No items provided for summarization.")

        chunks = self._chunk_items(items, args, warnings)
        if not chunks:
            raise ToolError("No chunks produced for summarization.")

        for chunk in chunks:
            summary = self._summarize_chunk(chunk, args, warnings)
            chunk_summaries.append(summary)

        level_summaries = self._build_summary_hierarchy(
            chunk_summaries, args, warnings
        )
        if level_summaries:
            final_summary = level_summaries[-1].summary
        else:
            final_summary = " ".join([item.summary for item in chunk_summaries])

        return ContextLongContextResult(
            mode=mode,
            summary_at=summary_at,
            items_count=len(items),
            chunks_count=len(chunks),
            chunk_summaries=chunk_summaries,
            level_summaries=level_summaries,
            final_summary=final_summary,
            conversation_summary=None,
            recent_messages=None,
            warnings=warnings,
            errors=errors,
        )
    def _summarize_window(
        self, args: ContextLongContextArgs
    ) -> tuple[ConversationSummary | None, list[MessageItem] | None, list[str]]:
        if not args.messages:
            raise ToolError("messages is required for window action.")

        recent_count = args.recent_messages or 8
        if recent_count < 0:
            raise ToolError("recent_messages must be >= 0.")

        messages = [MessageItem(**msg.model_dump()) for msg in args.messages]
        if recent_count == 0:
            recent = []
            older = messages
        else:
            recent = messages[-recent_count:]
            older = messages[:-recent_count]

        if not older:
            return None, recent, []

        transcript = "\n".join(
            f"{msg.role}: {msg.content.strip()}" for msg in older if msg.content.strip()
        )
        if not transcript:
            return None, recent, []

        summary_payload, errors = self._summarize_text(
            transcript,
            args,
            prompt_prefix=(
                "Summarize the conversation history. Focus on decisions, facts, and open questions."
            ),
        )
        if summary_payload is None:
            return None, recent, errors

        summary = ConversationSummary(
            summary=summary_payload.get("summary", ""),
            key_points=summary_payload.get("key_points"),
            keywords=summary_payload.get("keywords"),
        )
        return summary, recent, errors

    def _gather_items(
        self, args: ContextLongContextArgs
    ) -> tuple[list[LongContextItem], list[str]]:
        warnings: list[str] = []
        items: list[LongContextItem] = []
        if args.items:
            items.extend(args.items)

        if args.paths:
            include_globs = args.include_globs or self.config.default_include_globs
            exclude_globs = args.exclude_globs or self.config.default_exclude_globs
            max_files = args.max_files if args.max_files is not None else self.config.max_files
            files = self._gather_files(args.paths, include_globs, exclude_globs, max_files)
            for path in files:
                items.append(
                    LongContextItem(
                        id=str(path),
                        path=str(path),
                        content=None,
                        source=path.name,
                    )
                )

        return items, warnings

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
                    if not self._is_excluded(path.name, exclude_globs):
                        seen.add(key)
                        discovered.append(path)
                        if len(discovered) >= max_files:
                            return discovered
            else:
                raise ToolError(f"Path not found: {path}")

        return discovered

    def _is_excluded(self, rel_path: str, exclude_globs: list[str]) -> bool:
        if not exclude_globs:
            return False
        return any(fnmatch.fnmatch(rel_path, pattern) for pattern in exclude_globs)

    def _chunk_items(
        self,
        items: list[LongContextItem],
        args: ContextLongContextArgs,
        warnings: list[str],
    ) -> list[_Chunk]:
        max_source_bytes = (
            args.max_source_bytes
            if args.max_source_bytes is not None
            else self.config.max_source_bytes
        )
        max_total_bytes = (
            args.max_total_bytes
            if args.max_total_bytes is not None
            else self.config.max_total_bytes
        )
        max_item_chars = (
            args.max_item_chars
            if args.max_item_chars is not None
            else self.config.max_item_chars
        )
        unit = (args.chunk_unit or self.config.chunk_unit).strip().lower()
        size = args.chunk_size or self.config.chunk_size
        overlap = args.overlap if args.overlap is not None else self.config.overlap
        max_chunks = (
            args.max_chunks_per_item
            if args.max_chunks_per_item is not None
            else self.config.max_chunks_per_item
        )

        if unit not in {"chars", "lines"}:
            raise ToolError("chunk_unit must be chars or lines.")
        if size <= 0:
            raise ToolError("chunk_size must be a positive integer.")
        if overlap < 0:
            raise ToolError("overlap must be a non-negative integer.")
        if overlap >= size:
            raise ToolError("overlap must be smaller than chunk_size.")

        chunks: list[_Chunk] = []
        total_bytes = 0

        for idx, item in enumerate(items, start=1):
            try:
                source_id, source_path, content = self._load_item(item, idx, max_source_bytes)
            except ToolError as exc:
                warnings.append(str(exc))
                continue

            if content is None:
                warnings.append(f"Item {idx} has no content.")
                continue

            if max_item_chars > 0 and len(content) > max_item_chars:
                content = content[:max_item_chars]
                warnings.append(f"Item {idx} truncated to max_item_chars.")

            content_bytes = len(content.encode("utf-8"))
            if total_bytes + content_bytes > max_total_bytes:
                warnings.append("max_total_bytes reached; truncating further items.")
                break
            total_bytes += content_bytes

            for chunk in self._chunk_text(content, unit, size, overlap, max_chunks):
                chunks.append(
                    _Chunk(
                        source_id=source_id,
                        source_path=source_path,
                        unit=chunk[3],
                        chunk_index=chunk[4],
                        start_index=chunk[1],
                        end_index=chunk[2],
                        content=chunk[0],
                    )
                )

        return chunks

    def _load_item(
        self, item: LongContextItem, index: int, max_source_bytes: int
    ) -> tuple[str, str | None, str | None]:
        if item.content and item.path:
            raise ToolError("Provide content or path per item, not both.")
        source_id = item.id or item.source or item.path or f"item-{index}"
        if item.content is not None:
            content_bytes = len(item.content.encode("utf-8"))
            if content_bytes > max_source_bytes:
                raise ToolError(
                    f"Item '{source_id}' exceeds max_source_bytes ({content_bytes} > {max_source_bytes})."
                )
            return source_id, item.path, item.content

        if item.path:
            path = Path(item.path).expanduser()
            if not path.is_absolute():
                path = self.config.effective_workdir / path
            path = path.resolve()
            if not path.exists():
                raise ToolError(f"Path not found: {path}")
            if path.is_dir():
                raise ToolError(f"Path is a directory, not a file: {path}")
            content = path.read_text("utf-8", errors="ignore")
            content_bytes = len(content.encode("utf-8"))
            if content_bytes > max_source_bytes:
                raise ToolError(
                    f"{path} exceeds max_source_bytes ({content_bytes} > {max_source_bytes})."
                )
            return source_id, str(path), content
        return source_id, None, None

    def _chunk_text(
        self,
        content: str,
        unit: str,
        size: int,
        overlap: int,
        max_chunks: int,
    ) -> Iterable[tuple[str, int | None, int | None, str, int]]:
        chunk_index = 0
        if unit == "chars":
            step = size - overlap
            index = 0
            length = len(content)
            while index < length and chunk_index < max_chunks:
                chunk_text = content[index : index + size]
                if not chunk_text:
                    break
                chunk_index += 1
                start = index
                end = index + len(chunk_text) - 1
                yield chunk_text, start, end, "chars", chunk_index
                index += step
            return

        lines = content.splitlines()
        step = size - overlap
        index = 0
        while index < len(lines) and chunk_index < max_chunks:
            subset = lines[index : index + size]
            if not subset:
                break
            chunk_text = "\n".join(subset)
            chunk_index += 1
            start = index + 1
            end = start + len(subset) - 1
            yield chunk_text, start, end, "lines", chunk_index
            index += step
    def _summarize_chunk(
        self, chunk: _Chunk, args: ContextLongContextArgs, warnings: list[str]
    ) -> ChunkSummary:
        max_chars = self.config.max_chunk_chars
        content = chunk.content
        if max_chars > 0 and len(content) > max_chars:
            content = content[:max_chars]
        summary_payload, errors = self._summarize_text(
            content,
            args,
            prompt_prefix=self._chunk_prompt_prefix(args.question),
        )
        if errors:
            warnings.extend(errors)

        if summary_payload is None:
            fallback = content[: self.config.fallback_summary_chars].strip()
            summary_text = fallback or "(summary unavailable)"
            return ChunkSummary(
                summary_id=f"{chunk.source_id}:{chunk.chunk_index}",
                source_id=chunk.source_id,
                source_path=chunk.source_path,
                chunk_index=chunk.chunk_index,
                unit=chunk.unit,
                start_index=chunk.start_index,
                end_index=chunk.end_index,
                summary=summary_text,
                key_points=None,
                keywords=None,
            )

        return ChunkSummary(
            summary_id=f"{chunk.source_id}:{chunk.chunk_index}",
            source_id=chunk.source_id,
            source_path=chunk.source_path,
            chunk_index=chunk.chunk_index,
            unit=chunk.unit,
            start_index=chunk.start_index,
            end_index=chunk.end_index,
            summary=summary_payload.get("summary", ""),
            key_points=summary_payload.get("key_points"),
            keywords=summary_payload.get("keywords"),
        )

    def _build_summary_hierarchy(
        self,
        chunk_summaries: list[ChunkSummary],
        args: ContextLongContextArgs,
        warnings: list[str],
    ) -> list[LevelSummary]:
        if not chunk_summaries:
            return []

        hierarchy_levels = (
            args.hierarchy_levels
            if args.hierarchy_levels is not None
            else self.config.hierarchy_levels
        )
        group_size = (
            args.summary_group_size
            if args.summary_group_size is not None
            else self.config.summary_group_size
        )
        if hierarchy_levels <= 1:
            return []
        if group_size <= 0:
            raise ToolError("summary_group_size must be a positive integer.")

        current_level = 1
        summaries: list[LevelSummary] = []
        source_texts = [
            (summary.summary_id, summary.summary) for summary in chunk_summaries
        ]

        while current_level <= hierarchy_levels - 1 and len(source_texts) > 1:
            grouped = list(self._group_summaries(source_texts, group_size))
            next_texts: list[tuple[str, str]] = []
            for group_index, group in enumerate(grouped, start=1):
                source_ids = [item[0] for item in group]
                prompt = self._group_prompt(group, args.question)
                payload, errors = self._summarize_text(prompt, args)
                if errors:
                    warnings.extend(errors)
                summary_text = None
                if payload is not None:
                    summary_text = payload.get("summary")
                if not summary_text:
                    summary_text = self._fallback_group_summary(group)
                summary_id = f"L{current_level}-{group_index}"
                level_summary = LevelSummary(
                    level=current_level,
                    summary_id=summary_id,
                    source_ids=source_ids,
                    summary=summary_text,
                    key_points=payload.get("key_points") if payload else None,
                    keywords=payload.get("keywords") if payload else None,
                )
                summaries.append(level_summary)
                next_texts.append((summary_id, summary_text))

            source_texts = next_texts
            current_level += 1

        return summaries

    def _summarize_text(
        self,
        text: str,
        args: ContextLongContextArgs,
        prompt_prefix: str | None = None,
    ) -> tuple[dict | None, list[str]]:
        errors: list[str] = []
        if not text.strip():
            return None, ["Empty input for summary."]

        system_prompt = (
            "Reply ONLY with valid JSON that conforms to the JSON schema below. "
            "Do not include extra keys, comments, or markdown.\n"
            f"JSON schema:\n{json.dumps(SUMMARY_SCHEMA, ensure_ascii=True)}"
        )
        user_prompt = text
        if prompt_prefix:
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
        self, messages: list[dict[str, str]], args: ContextLongContextArgs
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

    def _call_llm(self, messages: list[dict[str, str]], args: ContextLongContextArgs) -> str:
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

    def _group_summaries(
        self, summaries: list[tuple[str, str]], group_size: int
    ) -> Iterable[list[tuple[str, str]]]:
        current: list[tuple[str, str]] = []
        for item in summaries:
            current.append(item)
            if len(current) >= group_size:
                yield current
                current = []
        if current:
            yield current

    def _group_prompt(self, group: list[tuple[str, str]], question: str | None) -> str:
        lines = []
        for summary_id, text in group:
            clean = text.replace("\n", " ").strip()
            lines.append(f"{summary_id}: {clean}")
        header = "Summarize the following chunk summaries into a concise overview."
        if question:
            header += f" Focus on this question: {question}"
        return f"{header}\n\n" + "\n".join(lines)

    def _fallback_group_summary(self, group: list[tuple[str, str]]) -> str:
        combined = " ".join(text for _, text in group)
        combined = combined.strip()
        if not combined:
            return "(summary unavailable)"
        return combined[: self.config.fallback_summary_chars]

    def _chunk_prompt_prefix(self, question: str | None) -> str:
        if question:
            return (
                "Summarize this chunk with focus on answering the question:\n"
                f"{question}"
            )
        return "Summarize this chunk in 1-3 concise sentences."

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextLongContextArgs):
            return ToolCallDisplay(summary="context_long_context")
        return ToolCallDisplay(
            summary="context_long_context",
            details={
                "action": event.args.action,
                "items": len(event.args.items or []),
                "paths": len(event.args.paths or []),
                "messages": len(event.args.messages or []),
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextLongContextResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = (
            f"Context summarized ({event.result.chunks_count} chunk(s))"
            if event.result.mode == "summarize"
            else "Context window summarized"
        )
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "mode": event.result.mode,
                "items_count": event.result.items_count,
                "chunks_count": event.result.chunks_count,
                "final_summary": event.result.final_summary,
                "conversation_summary": event.result.conversation_summary,
                "recent_messages": event.result.recent_messages,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Summarizing long context"
