from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
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


VALID_MEMORY_TYPES = {"long-term", "short-term"}
MEMORY_TYPE_ALIASES = {
    "long": "long-term",
    "longterm": "long-term",
    "ltm": "long-term",
    "short": "short-term",
    "shortterm": "short-term",
    "stm": "short-term",
}

DEFAULT_PROMPT_TEMPLATE = """### ADVANCED RECOLLECTION MODE (OPT-IN)
Bridge short-term and long-term memory to recall and connect relevant information.
- Weigh recency vs durability; surface conflicts or mismatches.
- Highlight likely retrieval cues and missing links.
- Keep internal recollection private; do not mention this mode in the response.
- If the user asks for steps or show_steps is enabled, give a concise outline.

Recollection style: {recollection_style}
Memory focus: {memory_focus}
Show steps: {show_steps}
Max links: {max_links}
"""

TOOL_PROMPT = (
    "Use `advanced_recollection` to bridge short-term and long-term memory. "
    "Provide `prompt` or `messages`, and optionally set `recollection_style`, "
    "`memory_focus`, `show_steps`, and `max_links`."
)


@dataclass(frozen=True)
class _MemorySource:
    content: str
    label: str | None
    memory_type: str | None
    source_path: str | None


class AdvancedRecollectionMessage(BaseModel):
    role: str
    content: str


class MemoryRecallItem(BaseModel):
    id: str | None = Field(default=None, description="Optional memory id.")
    label: str | None = Field(default=None, description="Optional label.")
    memory_type: str | None = Field(
        default=None, description="Memory type (long-term, short-term)."
    )
    context: str | None = Field(default=None, description="Context summary.")
    memory: str | None = Field(default=None, description="Memory content.")
    content: str | None = Field(default=None, description="Inline content.")
    tags: list[str] | None = Field(default=None, description="Optional tags.")
    path: str | None = Field(default=None, description="Path to a memory file.")


class AdvancedRecollectionArgs(BaseModel):
    prompt: str | None = Field(
        default=None, description="User prompt to solve."
    )
    messages: list[AdvancedRecollectionMessage] | None = Field(
        default=None, description="Optional chat messages."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prefix."
    )
    recollection_style: str | None = Field(
        default=None, description="Recollection style label."
    )
    memory_focus: str | None = Field(
        default=None, description="Memory focus label."
    )
    show_steps: bool | None = Field(
        default=None, description="Whether to include a step outline."
    )
    max_links: int | None = Field(
        default=None, description="Maximum links in the outline."
    )
    memories: list[MemoryRecallItem] | None = Field(
        default=None, description="Memory recall items."
    )
    memory_paths: list[str] | None = Field(
        default=None, description="Additional memory file paths."
    )
    max_memory_chars: int | None = Field(
        default=None, description="Max chars per memory item."
    )
    max_memory_total_chars: int | None = Field(
        default=None, description="Max total memory chars."
    )
    llm_api_base: str | None = Field(
        default=None, description="OpenAI-compatible API base URL."
    )
    llm_model: str | None = Field(
        default=None, description="LLM model name."
    )
    llm_temperature: float = Field(
        default=0.3, description="LLM temperature."
    )
    llm_max_tokens: int = Field(
        default=700, description="LLM max tokens."
    )
    llm_stream: bool = Field(
        default=False, description="Stream LLM tokens."
    )


class MemoryBlock(BaseModel):
    label: str | None
    memory_type: str | None
    source_path: str | None
    content: str
    truncated: bool


class AdvancedRecollectionResult(BaseModel):
    answer: str
    system_prompt: str
    messages: list[AdvancedRecollectionMessage]
    used_recollection_style: str
    used_memory_focus: str
    show_steps: bool
    max_links: int
    template_source: str
    memory_blocks: list[MemoryBlock]
    warnings: list[str]
    errors: list[str]
    llm_model: str


class AdvancedRecollectionConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    llm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL.",
    )
    llm_model: str = Field(
        default="gpt-oss:latest", description="Default LLM model name."
    )
    default_recollection_style: str = Field(
        default="bridge",
        description="Default recollection style.",
    )
    default_memory_focus: str = Field(
        default="balanced",
        description="Default memory focus.",
    )
    default_show_steps: bool = Field(
        default=False, description="Default for show_steps."
    )
    default_max_links: int = Field(
        default=6, description="Default max links."
    )
    prompt_path: Path | None = Field(
        default=Path.home()
        / "mistral-vibe"
        / "vibe"
        / "core"
        / "prompts"
        / "advanced_recollection.md",
        description="Optional path to a prompt template.",
    )
    prompt_max_chars: int = Field(
        default=8000, description="Maximum template characters to load."
    )
    max_memory_chars: int = Field(
        default=4000, description="Maximum characters per memory item."
    )
    max_memory_total_chars: int = Field(
        default=12000, description="Maximum total memory characters."
    )


class AdvancedRecollectionState(BaseToolState):
    pass


class AdvancedRecollection(
    BaseTool[
        AdvancedRecollectionArgs,
        AdvancedRecollectionResult,
        AdvancedRecollectionConfig,
        AdvancedRecollectionState,
    ],
    ToolUIData[AdvancedRecollectionArgs, AdvancedRecollectionResult],
):
    description: ClassVar[str] = (
        "Bridge short-term and long-term memory using a prompt template."
    )

    @classmethod
    def get_tool_prompt(cls) -> str | None:
        return TOOL_PROMPT

    async def run(
        self, args: AdvancedRecollectionArgs
    ) -> AdvancedRecollectionResult:
        warnings: list[str] = []
        errors: list[str] = []

        recollection_style = (
            args.recollection_style or self.config.default_recollection_style
        ).strip()
        if not recollection_style:
            recollection_style = "bridge"

        memory_focus = (args.memory_focus or self.config.default_memory_focus).strip()
        if not memory_focus:
            memory_focus = "balanced"

        show_steps = (
            args.show_steps
            if args.show_steps is not None
            else self.config.default_show_steps
        )
        max_links = (
            args.max_links
            if args.max_links is not None
            else self.config.default_max_links
        )
        if max_links <= 0:
            raise ToolError("max_links must be positive.")

        self._validate_llm_settings(args)

        template, template_source = self._load_template(warnings)
        system_prompt = self._build_system_prompt(
            template,
            recollection_style,
            memory_focus,
            show_steps,
            max_links,
            args.system_prompt,
        )

        memory_blocks = self._collect_memory_context(args, warnings)
        messages = self._normalize_messages(args, system_prompt)
        messages = self._inject_memory_context(messages, memory_blocks)
        answer = self._call_llm(messages, args)

        return AdvancedRecollectionResult(
            answer=answer,
            system_prompt=system_prompt,
            messages=messages,
            used_recollection_style=recollection_style,
            used_memory_focus=memory_focus,
            show_steps=show_steps,
            max_links=max_links,
            template_source=template_source,
            memory_blocks=memory_blocks,
            warnings=warnings,
            errors=errors,
            llm_model=self._resolve_model(args),
        )

    def _validate_llm_settings(self, args: AdvancedRecollectionArgs) -> None:
        if args.llm_temperature < 0:
            raise ToolError("llm_temperature cannot be negative.")
        if args.llm_max_tokens <= 0:
            raise ToolError("llm_max_tokens must be positive.")

    def _collect_memory_context(
        self, args: AdvancedRecollectionArgs, warnings: list[str]
    ) -> list[MemoryBlock]:
        max_item = (
            args.max_memory_chars
            if args.max_memory_chars is not None
            else self.config.max_memory_chars
        )
        max_total = (
            args.max_memory_total_chars
            if args.max_memory_total_chars is not None
            else self.config.max_memory_total_chars
        )
        if max_item <= 0 or max_total <= 0:
            raise ToolError(
                "max_memory_chars and max_memory_total_chars must be positive."
            )

        sources = self._resolve_memory_sources(args, warnings)
        blocks: list[MemoryBlock] = []
        total_chars = 0
        for source in sources:
            if total_chars >= max_total:
                warnings.append(
                    "max_memory_total_chars reached; truncating memory."
                )
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
            blocks.append(
                MemoryBlock(
                    label=source.label,
                    memory_type=source.memory_type,
                    source_path=source.source_path,
                    content=content,
                    truncated=truncated,
                )
            )
        if not blocks:
            warnings.append("No memory context provided.")
        return blocks

    def _resolve_memory_sources(
        self, args: AdvancedRecollectionArgs, warnings: list[str]
    ) -> list[_MemorySource]:
        sources: list[_MemorySource] = []
        if args.memories:
            for item in args.memories:
                sources.append(self._load_memory_item(item, warnings))

        if args.memory_paths:
            for raw_path in args.memory_paths:
                path = self._resolve_path(raw_path)
                content = path.read_text("utf-8", errors="ignore")
                sources.append(
                    _MemorySource(
                        content=content,
                        label=path.name,
                        memory_type=None,
                        source_path=str(path),
                    )
                )

        if not sources and not warnings:
            warnings.append("No memory context provided.")
        return sources

    def _load_memory_item(
        self, item: MemoryRecallItem, warnings: list[str]
    ) -> _MemorySource:
        if item.content and item.path:
            raise ToolError("Provide content or path per memory item, not both.")
        if not item.content and not item.path and not item.memory and not item.context:
            raise ToolError(
                "Each memory item must provide content, memory, context, or path."
            )

        label = item.label or item.id or item.path
        memory_type = (
            self._normalize_memory_type(item.memory_type, warnings)
            if item.memory_type
            else None
        )
        text = self._format_memory_text(item)

        if item.path:
            path = self._resolve_path(item.path)
            content = path.read_text("utf-8", errors="ignore")
            text = self._join_sections(text, "Content", content)
            return _MemorySource(
                content=text,
                label=label or path.name,
                memory_type=memory_type,
                source_path=str(path),
            )

        return _MemorySource(
            content=text,
            label=label,
            memory_type=memory_type,
            source_path=None,
        )

    def _normalize_memory_type(self, value: str, warnings: list[str]) -> str:
        normalized = value.strip().lower()
        normalized = MEMORY_TYPE_ALIASES.get(normalized, normalized)
        if normalized not in VALID_MEMORY_TYPES:
            warnings.append(f"Unknown memory type '{value}'; using as-is.")
        return normalized

    def _format_memory_text(self, item: MemoryRecallItem) -> str:
        lines: list[str] = []
        if item.label or item.id:
            lines.append(f"Label: {item.label or item.id}")
        if item.memory_type:
            lines.append(f"Memory type: {item.memory_type}")
        if item.tags:
            tags = ", ".join(tag for tag in item.tags if tag)
            if tags:
                lines.append(f"Tags: {tags}")
        if item.context:
            lines.append("Context:")
            lines.append(item.context)
        if item.memory:
            lines.append("Memory:")
            lines.append(item.memory)
        if item.content:
            lines.append("Content:")
            lines.append(item.content)
        return "\n".join(lines).strip()

    def _join_sections(self, base: str, title: str, content: str) -> str:
        if not content.strip():
            return base
        if base:
            return f"{base}\n\n{title}:\n{content}".strip()
        return f"{title}:\n{content}".strip()

    def _resolve_path(self, raw_path: str) -> Path:
        if not raw_path.strip():
            raise ToolError("Path cannot be empty.")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.config.effective_workdir / path
        try:
            resolved = path.resolve()
        except ValueError as exc:
            raise ToolError(
                "Security error: cannot resolve the provided path."
            ) from exc
        if not resolved.exists():
            raise ToolError(f"File not found at: {resolved}")
        if resolved.is_dir():
            raise ToolError(f"Path is a directory, not a file: {resolved}")
        return resolved

    def _format_memory_blocks(self, blocks: list[MemoryBlock]) -> str:
        if not blocks:
            return ""
        parts = ["Memory recall context:"]
        for block in blocks:
            label = block.label or block.source_path or "memory"
            type_suffix = f" | {block.memory_type}" if block.memory_type else ""
            trunc = " (truncated)" if block.truncated else ""
            header = f"[{label}{type_suffix}{trunc}]"
            parts.append(f"{header}\n{block.content}".strip())
        return "\n\n".join(parts).strip()

    def _inject_memory_context(
        self,
        messages: list[AdvancedRecollectionMessage],
        blocks: list[MemoryBlock],
    ) -> list[AdvancedRecollectionMessage]:
        if not blocks:
            return messages
        memory_text = self._format_memory_blocks(blocks)
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].role == "user":
                content = messages[idx].content.rstrip()
                messages[idx].content = f"{content}\n\n{memory_text}".strip()
                return messages
        messages.append(
            AdvancedRecollectionMessage(role="user", content=memory_text)
        )
        return messages

    def _load_template(self, warnings: list[str]) -> tuple[str, str]:
        template = DEFAULT_PROMPT_TEMPLATE
        source = "embedded"

        if not self.config.prompt_path:
            return self._truncate_template(template, warnings), source

        path = self._resolve_prompt_path(self.config.prompt_path)
        if not path.exists():
            warnings.append(f"Prompt template not found: {path}")
            return self._truncate_template(template, warnings), source
        if path.is_dir():
            warnings.append(f"Prompt template is a directory: {path}")
            return self._truncate_template(template, warnings), source

        try:
            text = path.read_text("utf-8", errors="ignore").strip()
        except OSError as exc:
            warnings.append(f"Failed to read prompt template: {exc}")
            return self._truncate_template(template, warnings), source

        if not text:
            warnings.append(f"Prompt template empty: {path}")
            return self._truncate_template(template, warnings), source

        template = text
        source = str(path)
        return self._truncate_template(template, warnings), source

    def _resolve_prompt_path(self, raw_path: Path | str) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.config.effective_workdir / path
        return path.resolve()

    def _truncate_template(self, template: str, warnings: list[str]) -> str:
        max_chars = self.config.prompt_max_chars
        if max_chars > 0 and len(template) > max_chars:
            warnings.append("Prompt template truncated to prompt_max_chars.")
            return template[:max_chars].rstrip()
        return template

    def _build_system_prompt(
        self,
        template: str,
        recollection_style: str,
        memory_focus: str,
        show_steps: bool,
        max_links: int,
        prefix: str | None,
    ) -> str:
        show_steps_text = "yes" if show_steps else "no"
        rendered = self._render_template(
            template,
            recollection_style,
            memory_focus,
            show_steps_text,
            max_links,
        )
        if prefix and prefix.strip():
            return f"{prefix.strip()}\n\n{rendered}".strip()
        return rendered.strip()

    def _render_template(
        self,
        template: str,
        recollection_style: str,
        memory_focus: str,
        show_steps_text: str,
        max_links: int,
    ) -> str:
        had_placeholders = (
            "{recollection_style}" in template
            or "{memory_focus}" in template
            or "{show_steps}" in template
            or "{max_links}" in template
        )
        rendered = template
        if "{recollection_style}" in template:
            rendered = rendered.replace("{recollection_style}", recollection_style)
        if "{memory_focus}" in template:
            rendered = rendered.replace("{memory_focus}", memory_focus)
        if "{show_steps}" in template:
            rendered = rendered.replace("{show_steps}", show_steps_text)
        if "{max_links}" in template:
            rendered = rendered.replace("{max_links}", str(max_links))

        if had_placeholders:
            return rendered

        extra = (
            f"Recollection style: {recollection_style}\n"
            f"Memory focus: {memory_focus}\n"
            f"Show steps: {show_steps_text}\n"
            f"Max links: {max_links}"
        )
        return f"{rendered.rstrip()}\n\n{extra}"

    def _normalize_messages(
        self, args: AdvancedRecollectionArgs, system_prompt: str
    ) -> list[AdvancedRecollectionMessage]:
        messages: list[AdvancedRecollectionMessage] = []
        if args.messages:
            for msg in args.messages:
                role = (msg.role or "").strip()
                content = (msg.content or "").strip()
                if not content:
                    continue
                if not role:
                    role = "user"
                messages.append(
                    AdvancedRecollectionMessage(role=role, content=content)
                )
        elif args.prompt and args.prompt.strip():
            messages.append(
                AdvancedRecollectionMessage(role="user", content=args.prompt.strip())
            )
        else:
            raise ToolError("Provide prompt or messages.")

        if not messages:
            raise ToolError("No usable messages provided.")

        if system_prompt.strip():
            messages.insert(
                0,
                AdvancedRecollectionMessage(
                    role="system", content=system_prompt.strip()
                ),
            )
        return messages

    def _call_llm(
        self,
        messages: list[AdvancedRecollectionMessage],
        args: AdvancedRecollectionArgs,
    ) -> str:
        api_base = (args.llm_api_base or self.config.llm_api_base).rstrip("/")
        url = api_base + "/chat/completions"
        payload = {
            "model": self._resolve_model(args),
            "messages": [msg.model_dump() for msg in messages],
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

    def _resolve_model(self, args: AdvancedRecollectionArgs) -> str:
        return args.llm_model or self.config.llm_model

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, AdvancedRecollectionArgs):
            return ToolCallDisplay(summary="advanced_recollection")
        return ToolCallDisplay(
            summary="advanced_recollection",
            details={
                "recollection_style": event.args.recollection_style,
                "memory_focus": event.args.memory_focus,
                "show_steps": event.args.show_steps,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, AdvancedRecollectionResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = "Advanced recollection complete"
        if event.result.errors:
            message = "Advanced recollection finished with errors"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "answer": event.result.answer,
                "used_recollection_style": event.result.used_recollection_style,
                "used_memory_focus": event.result.used_memory_focus,
                "show_steps": event.result.show_steps,
                "max_links": event.result.max_links,
                "template_source": event.result.template_source,
                "memory_blocks": len(event.result.memory_blocks),
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Advanced recollection"
