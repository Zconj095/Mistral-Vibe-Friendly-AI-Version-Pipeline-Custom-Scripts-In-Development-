from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from urllib import request

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


VALID_EXAMPLE_SELECTION = {"all", "first", "semantic"}
VALID_CONTEXT_ROLES = {"system", "user"}


@dataclass(frozen=True)
class _ContextSource:
    content: str
    role: str
    label: str | None
    source_path: str | None


class PromptingStrategiesConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    default_role: str = Field(
        default="You are a helpful assistant.",
        description="Default system role prompt.",
    )
    default_selection: str = Field(
        default="semantic",
        description="Default example selection: all, first, or semantic.",
    )
    default_max_examples: int = Field(
        default=3, description="Default maximum examples to include."
    )
    max_context_chars: int = Field(
        default=4000, description="Maximum characters per context block."
    )
    max_context_total_chars: int = Field(
        default=12000, description="Maximum total context characters."
    )
    ollama_url: str = Field(
        default="http://127.0.0.1:11434",
        description="Base URL for the local Ollama server.",
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        description="Embedding model for semantic example selection.",
    )


class PromptingStrategiesState(BaseToolState):
    pass


class PromptContextItem(BaseModel):
    id: str | None = Field(default=None, description="Optional context id.")
    label: str | None = Field(default=None, description="Optional label.")
    content: str | None = Field(default=None, description="Inline content.")
    path: str | None = Field(default=None, description="Path to a file.")
    role: str | None = Field(
        default=None, description="system or user."
    )


class PromptExample(BaseModel):
    input: str
    output: str
    description: str | None = None
    score: float | None = None


class PromptMessage(BaseModel):
    role: str
    content: str


class PromptingStrategiesArgs(BaseModel):
    task: str = Field(description="Primary task or question.")
    role: str | None = Field(default=None, description="System role prompt.")
    goal: str | None = Field(default=None, description="Overall goal.")
    constraints: list[str] | None = Field(
        default=None, description="Hard constraints."
    )
    preferences: list[str] | None = Field(
        default=None, description="Soft preferences."
    )
    checklist: list[str] | None = Field(
        default=None, description="Quality checklist."
    )
    include_plan: bool = Field(
        default=False, description="Ask for a brief plan before the answer."
    )
    reveal_reasoning: bool = Field(
        default=False, description="Allow chain-of-thought in output."
    )
    output_format: str | None = Field(
        default=None, description="Output format hint."
    )
    output_schema: dict | str | None = Field(
        default=None, description="JSON schema dict or JSON string."
    )
    strict_json: bool = Field(
        default=True, description="Require JSON-only output when schema is set."
    )
    context: list[PromptContextItem] | None = Field(
        default=None, description="Context items."
    )
    context_paths: list[str] | None = Field(
        default=None, description="Additional context file paths."
    )
    max_context_chars: int | None = Field(
        default=None, description="Override max context chars per item."
    )
    max_context_total_chars: int | None = Field(
        default=None, description="Override max total context chars."
    )
    examples: list[PromptExample] | None = Field(
        default=None, description="Inline few-shot examples."
    )
    example_paths: list[str] | None = Field(
        default=None, description="JSON or JSONL example files."
    )
    example_selection: str | None = Field(
        default=None, description="all, first, or semantic."
    )
    max_examples: int | None = Field(
        default=None, description="Maximum examples to include."
    )
    embedding_model: str | None = Field(
        default=None, description="Override embedding model."
    )


class ContextBlock(BaseModel):
    role: str
    label: str | None
    source_path: str | None
    content: str
    truncated: bool


class PromptingStrategiesResult(BaseModel):
    system_prompt: str
    user_prompt: str
    messages: list[PromptMessage]
    selected_examples: list[PromptExample]
    context_blocks: list[ContextBlock]
    warnings: list[str]
    errors: list[str]


class PromptingStrategies(
    BaseTool[
        PromptingStrategiesArgs,
        PromptingStrategiesResult,
        PromptingStrategiesConfig,
        PromptingStrategiesState,
    ],
    ToolUIData[PromptingStrategiesArgs, PromptingStrategiesResult],
):
    description: ClassVar[str] = (
        "Build structured prompts with constraints, examples, and context."
    )

    async def run(
        self, args: PromptingStrategiesArgs
    ) -> PromptingStrategiesResult:
        if not args.task.strip():
            raise ToolError("task cannot be empty.")

        warnings: list[str] = []
        errors: list[str] = []

        context_blocks, system_context, user_context = self._collect_context(
            args, warnings
        )
        examples = self._collect_examples(args, warnings)
        selected_examples = self._select_examples(args, examples, warnings)
        schema = self._load_schema(args.output_schema, warnings)

        system_prompt = self._build_system_prompt(
            args, system_context, schema
        )
        user_prompt = self._build_user_prompt(
            args, user_context, schema
        )

        messages = self._build_messages(
            system_prompt, user_prompt, selected_examples
        )

        return PromptingStrategiesResult(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            messages=messages,
            selected_examples=selected_examples,
            context_blocks=context_blocks,
            warnings=warnings,
            errors=errors,
        )

    def _collect_context(
        self,
        args: PromptingStrategiesArgs,
        warnings: list[str],
    ) -> tuple[list[ContextBlock], list[str], list[str]]:
        max_item = (
            args.max_context_chars
            if args.max_context_chars is not None
            else self.config.max_context_chars
        )
        max_total = (
            args.max_context_total_chars
            if args.max_context_total_chars is not None
            else self.config.max_context_total_chars
        )
        if max_item <= 0 or max_total <= 0:
            raise ToolError("max_context_chars and max_context_total_chars must be positive.")

        context_sources = self._resolve_context_sources(args, warnings)
        context_blocks: list[ContextBlock] = []
        system_context: list[str] = []
        user_context: list[str] = []
        total_chars = 0

        for source in context_sources:
            if total_chars >= max_total:
                warnings.append("max_context_total_chars reached; truncating context.")
                break
            content = source.content
            truncated = False
            if len(content) > max_item:
                content = content[:max_item]
                truncated = True
            if total_chars + len(content) > max_total:
                content = content[: max_total - total_chars]
                truncated = True
            total_chars += len(content)

            block = ContextBlock(
                role=source.role,
                label=source.label,
                source_path=source.source_path,
                content=content,
                truncated=truncated,
            )
            context_blocks.append(block)

            label = source.label or source.source_path or "context"
            header = f"[{label}]"
            text_block = f"{header}\n{content}".strip()
            if source.role == "system":
                system_context.append(text_block)
            else:
                user_context.append(text_block)

        return context_blocks, system_context, user_context

    def _resolve_context_sources(
        self, args: PromptingStrategiesArgs, warnings: list[str]
    ) -> list[_ContextSource]:
        sources: list[_ContextSource] = []
        if args.context:
            for item in args.context:
                sources.append(self._load_context_item(item))

        if args.context_paths:
            for raw_path in args.context_paths:
                path = self._resolve_path(raw_path)
                content = path.read_text("utf-8", errors="ignore")
                sources.append(
                    _ContextSource(
                        content=content,
                        role="user",
                        label=path.name,
                        source_path=str(path),
                    )
                )

        return sources

    def _load_context_item(self, item: PromptContextItem) -> _ContextSource:
        if item.content and item.path:
            raise ToolError("Provide content or path per context item, not both.")
        if not item.content and not item.path:
            raise ToolError("Each context item must provide content or path.")

        role = (item.role or "user").strip().lower()
        if role not in VALID_CONTEXT_ROLES:
            raise ToolError("context role must be system or user.")

        label = item.label or item.id or item.path
        if item.content is not None:
            return _ContextSource(
                content=item.content,
                role=role,
                label=label,
                source_path=None,
            )

        path = self._resolve_path(item.path or "")
        content = path.read_text("utf-8", errors="ignore")
        return _ContextSource(
            content=content,
            role=role,
            label=label or path.name,
            source_path=str(path),
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

    def _collect_examples(
        self, args: PromptingStrategiesArgs, warnings: list[str]
    ) -> list[PromptExample]:
        examples: list[PromptExample] = []
        if args.examples:
            examples.extend(args.examples)

        if args.example_paths:
            for raw_path in args.example_paths:
                path = self._resolve_path(raw_path)
                loaded, load_warnings = self._load_examples_from_path(path)
                warnings.extend(load_warnings)
                examples.extend(loaded)

        return examples

    def _load_examples_from_path(
        self, path: Path
    ) -> tuple[list[PromptExample], list[str]]:
        warnings: list[str] = []
        try:
            content = path.read_text("utf-8", errors="ignore")
        except OSError as exc:
            warnings.append(f"Failed to read examples: {exc}")
            return [], warnings

        examples: list[PromptExample] = []
        if path.suffix.lower() == ".jsonl":
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    warnings.append("Invalid JSONL line; skipping.")
                    continue
                example = self._parse_example_payload(payload)
                if example:
                    examples.append(example)
            return examples, warnings

        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            warnings.append("Example file must be JSON or JSONL.")
            return [], warnings

        if isinstance(payload, dict) and "examples" in payload:
            payload = payload["examples"]

        if isinstance(payload, dict):
            example = self._parse_example_payload(payload)
            if example:
                examples.append(example)
        elif isinstance(payload, list):
            for item in payload:
                example = self._parse_example_payload(item)
                if example:
                    examples.append(example)
        else:
            warnings.append("Example JSON must be an object or array.")
        return examples, warnings

    def _parse_example_payload(self, payload: Any) -> PromptExample | None:
        if not isinstance(payload, dict):
            return None
        input_text = payload.get("input")
        output_text = payload.get("output")
        if not isinstance(input_text, str) or not isinstance(output_text, str):
            return None
        description = payload.get("description")
        return PromptExample(
            input=input_text,
            output=output_text,
            description=description if isinstance(description, str) else None,
        )

    def _select_examples(
        self,
        args: PromptingStrategiesArgs,
        examples: list[PromptExample],
        warnings: list[str],
    ) -> list[PromptExample]:
        if not examples:
            return []

        selection = (
            args.example_selection or self.config.default_selection
        ).strip().lower()
        if selection not in VALID_EXAMPLE_SELECTION:
            raise ToolError("example_selection must be all, first, or semantic.")

        max_examples = (
            args.max_examples if args.max_examples is not None else self.config.default_max_examples
        )
        if max_examples is not None and max_examples < 0:
            raise ToolError("max_examples must be >= 0.")
        if max_examples == 0:
            return []

        filtered = [ex for ex in examples if ex.output.strip()]
        if len(filtered) < len(examples):
            warnings.append("Examples missing outputs were skipped.")

        if selection == "first":
            return filtered[:max_examples] if max_examples else filtered
        if selection == "all":
            return filtered[:max_examples] if max_examples else filtered

        return self._semantic_select(args, filtered, max_examples, warnings)

    def _semantic_select(
        self,
        args: PromptingStrategiesArgs,
        examples: list[PromptExample],
        max_examples: int | None,
        warnings: list[str],
    ) -> list[PromptExample]:
        model = args.embedding_model or self.config.embedding_model
        query = args.task.strip()
        if args.goal:
            query = f"{query}\nGoal: {args.goal.strip()}"

        query_vec = self._embed_text(model, query, warnings)
        if not query_vec:
            return examples[:max_examples] if max_examples else examples

        scored: list[PromptExample] = []
        for example in examples:
            text = f"{example.input}\n{example.output}"
            embedding = self._embed_text(model, text, warnings)
            if not embedding:
                continue
            score = self._dot(query_vec, embedding)
            scored.append(
                PromptExample(
                    input=example.input,
                    output=example.output,
                    description=example.description,
                    score=score,
                )
            )

        scored.sort(key=lambda item: item.score or 0.0, reverse=True)
        if max_examples:
            scored = scored[:max_examples]
        return scored

    def _build_system_prompt(
        self,
        args: PromptingStrategiesArgs,
        system_context: list[str],
        schema: dict | None,
    ) -> str:
        role = (args.role or self.config.default_role).strip()
        lines = [role]

        if args.goal:
            lines.append(f"Goal: {args.goal.strip()}")

        if args.constraints:
            constraints = [item.strip() for item in args.constraints if item.strip()]
            if constraints:
                lines.append("Constraints:")
                lines.extend([f"- {item}" for item in constraints])

        if args.preferences:
            preferences = [item.strip() for item in args.preferences if item.strip()]
            if preferences:
                lines.append("Preferences:")
                lines.extend([f"- {item}" for item in preferences])

        if schema is not None:
            schema_text = json.dumps(schema, ensure_ascii=True)
            if args.strict_json:
                lines.append(
                    "Reply ONLY with valid JSON that matches the schema below."
                )
            lines.append(f"JSON schema: {schema_text}")

        if args.output_format:
            lines.append(f"Output format: {args.output_format.strip()}")

        if not args.reveal_reasoning:
            lines.append("Do not reveal chain-of-thought reasoning.")

        if system_context:
            lines.append("System context:")
            lines.extend(system_context)

        return "\n".join(lines).strip()

    def _build_user_prompt(
        self,
        args: PromptingStrategiesArgs,
        user_context: list[str],
        schema: dict | None,
    ) -> str:
        lines = [f"Task: {args.task.strip()}"]

        if user_context:
            lines.append("Context:")
            lines.extend(user_context)

        if args.checklist:
            checklist = [item.strip() for item in args.checklist if item.strip()]
            if checklist:
                lines.append("Quality checklist:")
                lines.extend([f"- {item}" for item in checklist])

        if args.include_plan:
            lines.append("Provide a short plan (2-5 bullets) before the answer.")

        if schema is not None and args.strict_json:
            lines.append("Return JSON only. Do not add extra text.")

        return "\n".join(lines).strip()

    def _build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        examples: list[PromptExample],
    ) -> list[PromptMessage]:
        messages: list[PromptMessage] = []
        messages.append(PromptMessage(role="system", content=system_prompt))
        for example in examples:
            messages.append(PromptMessage(role="user", content=example.input))
            messages.append(PromptMessage(role="assistant", content=example.output))
        messages.append(PromptMessage(role="user", content=user_prompt))
        return messages

    def _load_schema(
        self, schema: dict | str | None, warnings: list[str]
    ) -> dict | None:
        if schema is None:
            return None
        if isinstance(schema, dict):
            return schema
        if isinstance(schema, str):
            value = schema.strip()
            if not value:
                warnings.append("output_schema provided but empty.")
                return None
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                warnings.append(f"output_schema JSON parse error: {exc}")
                return None
            if isinstance(parsed, dict):
                return parsed
            warnings.append("output_schema JSON must be an object.")
            return None
        warnings.append("output_schema must be a dict or JSON string.")
        return None

    def _embed_text(
        self, model: str, text: str, warnings: list[str]
    ) -> list[float] | None:
        payload = json.dumps({"model": model, "prompt": text}).encode("utf-8")
        url = self.config.ollama_url.rstrip("/") + "/api/embeddings"
        req = request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=60) as resp:
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

    def _dot(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            return 0.0
        return sum(l * r for l, r in zip(left, right))

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, PromptingStrategiesArgs):
            return ToolCallDisplay(summary="prompting_strategies")
        return ToolCallDisplay(
            summary="prompting_strategies",
            details={
                "task": event.args.task,
                "examples": len(event.args.examples or []),
                "context": len(event.args.context or []),
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, PromptingStrategiesResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = f"Prompt bundle built ({len(event.result.messages)} message(s))"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "system_prompt": event.result.system_prompt,
                "user_prompt": event.result.user_prompt,
                "selected_examples": event.result.selected_examples,
                "context_blocks": event.result.context_blocks,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Building prompt bundle"
