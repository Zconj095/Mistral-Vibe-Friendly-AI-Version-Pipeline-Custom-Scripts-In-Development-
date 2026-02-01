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


DEFAULT_PROMPT_TEMPLATE = """### RECOLLECTION TRIGGERS MODE (OPT-IN)
Recognize when recollection can be triggered by events and circumstances.
- Identify cues, contexts, and conditions that can surface memories.
- Distinguish observed signals from inferred triggers; call out uncertainty.
- Highlight likely retrieval cues and gaps.
- Keep internal recollection framing private; do not mention this mode in the response.
- If the user asks for steps or show_steps is enabled, give a concise outline.

Recollection style: {recollection_style}
Trigger focus: {trigger_focus}
Show steps: {show_steps}
Max triggers: {max_triggers}
"""

TOOL_PROMPT = (
    "Use `recollection_triggers` to recognize when recollection can occur "
    "from events and circumstances. Provide `prompt` or `messages`, and "
    "optionally set `recollection_style`, `trigger_focus`, `show_steps`, "
    "and `max_triggers`."
)


@dataclass(frozen=True)
class _TriggerSource:
    content: str
    label: str | None
    source_type: str | None
    source_path: str | None


class RecollectionTriggerMessage(BaseModel):
    role: str
    content: str


class EventCircumstanceItem(BaseModel):
    id: str | None = Field(default=None, description="Optional item id.")
    label: str | None = Field(default=None, description="Optional label.")
    source_type: str | None = Field(
        default=None, description="Source type (event, circumstance, signal)."
    )
    event: str | None = Field(default=None, description="Event description.")
    circumstance: str | None = Field(
        default=None, description="Circumstance description."
    )
    context: str | None = Field(default=None, description="Context summary.")
    trigger: str | None = Field(default=None, description="Potential trigger.")
    timestamp: str | None = Field(default=None, description="Time or timeframe.")
    location: str | None = Field(default=None, description="Location or setting.")
    content: str | None = Field(default=None, description="Inline content.")
    tags: list[str] | None = Field(default=None, description="Optional tags.")
    path: str | None = Field(default=None, description="Path to a context file.")


class RecollectionTriggersArgs(BaseModel):
    prompt: str | None = Field(
        default=None, description="User prompt to solve."
    )
    messages: list[RecollectionTriggerMessage] | None = Field(
        default=None, description="Optional chat messages."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prefix."
    )
    recollection_style: str | None = Field(
        default=None, description="Recollection style label."
    )
    trigger_focus: str | None = Field(
        default=None, description="Trigger focus label."
    )
    show_steps: bool | None = Field(
        default=None, description="Whether to include a step outline."
    )
    max_triggers: int | None = Field(
        default=None, description="Maximum triggers in the outline."
    )
    items: list[EventCircumstanceItem] | None = Field(
        default=None, description="Event or circumstance items."
    )
    item_paths: list[str] | None = Field(
        default=None, description="Additional context file paths."
    )
    max_item_chars: int | None = Field(
        default=None, description="Max chars per item."
    )
    max_item_total_chars: int | None = Field(
        default=None, description="Max total item chars."
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


class TriggerBlock(BaseModel):
    label: str | None
    source_type: str | None
    source_path: str | None
    content: str
    truncated: bool


class RecollectionTriggersResult(BaseModel):
    answer: str
    system_prompt: str
    messages: list[RecollectionTriggerMessage]
    used_recollection_style: str
    used_trigger_focus: str
    show_steps: bool
    max_triggers: int
    template_source: str
    trigger_blocks: list[TriggerBlock]
    warnings: list[str]
    errors: list[str]
    llm_model: str


class RecollectionTriggersConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    llm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL.",
    )
    llm_model: str = Field(
        default="gpt-oss:latest", description="Default LLM model name."
    )
    default_recollection_style: str = Field(
        default="contextual",
        description="Default recollection style.",
    )
    default_trigger_focus: str = Field(
        default="events-and-circumstances",
        description="Default trigger focus.",
    )
    default_show_steps: bool = Field(
        default=False, description="Default for show_steps."
    )
    default_max_triggers: int = Field(
        default=6, description="Default max triggers."
    )
    prompt_path: Path | None = Field(
        default=Path.home()
        / "mistral-vibe"
        / "vibe"
        / "core"
        / "prompts"
        / "recollection_triggers.md",
        description="Optional path to a prompt template.",
    )
    prompt_max_chars: int = Field(
        default=8000, description="Maximum template characters to load."
    )
    max_item_chars: int = Field(
        default=4000, description="Maximum characters per item."
    )
    max_item_total_chars: int = Field(
        default=12000, description="Maximum total item characters."
    )


class RecollectionTriggersState(BaseToolState):
    pass


class RecollectionTriggers(
    BaseTool[
        RecollectionTriggersArgs,
        RecollectionTriggersResult,
        RecollectionTriggersConfig,
        RecollectionTriggersState,
    ],
    ToolUIData[RecollectionTriggersArgs, RecollectionTriggersResult],
):
    description: ClassVar[str] = (
        "Recognize recollection triggers from events and circumstances."
    )

    @classmethod
    def get_tool_prompt(cls) -> str | None:
        return TOOL_PROMPT

    async def run(
        self, args: RecollectionTriggersArgs
    ) -> RecollectionTriggersResult:
        warnings: list[str] = []
        errors: list[str] = []

        recollection_style = (
            args.recollection_style or self.config.default_recollection_style
        ).strip()
        if not recollection_style:
            recollection_style = "contextual"

        trigger_focus = (
            args.trigger_focus or self.config.default_trigger_focus
        ).strip()
        if not trigger_focus:
            trigger_focus = "events-and-circumstances"

        show_steps = (
            args.show_steps
            if args.show_steps is not None
            else self.config.default_show_steps
        )
        max_triggers = (
            args.max_triggers
            if args.max_triggers is not None
            else self.config.default_max_triggers
        )
        if max_triggers <= 0:
            raise ToolError("max_triggers must be positive.")

        self._validate_llm_settings(args)

        template, template_source = self._load_template(warnings)
        system_prompt = self._build_system_prompt(
            template,
            recollection_style,
            trigger_focus,
            show_steps,
            max_triggers,
            args.system_prompt,
        )

        trigger_blocks = self._collect_triggers(args, warnings)
        messages = self._normalize_messages(args, system_prompt)
        messages = self._inject_triggers(messages, trigger_blocks)
        answer = self._call_llm(messages, args)

        return RecollectionTriggersResult(
            answer=answer,
            system_prompt=system_prompt,
            messages=messages,
            used_recollection_style=recollection_style,
            used_trigger_focus=trigger_focus,
            show_steps=show_steps,
            max_triggers=max_triggers,
            template_source=template_source,
            trigger_blocks=trigger_blocks,
            warnings=warnings,
            errors=errors,
            llm_model=self._resolve_model(args),
        )

    def _validate_llm_settings(self, args: RecollectionTriggersArgs) -> None:
        if args.llm_temperature < 0:
            raise ToolError("llm_temperature cannot be negative.")
        if args.llm_max_tokens <= 0:
            raise ToolError("llm_max_tokens must be positive.")

    def _collect_triggers(
        self, args: RecollectionTriggersArgs, warnings: list[str]
    ) -> list[TriggerBlock]:
        max_item = (
            args.max_item_chars
            if args.max_item_chars is not None
            else self.config.max_item_chars
        )
        max_total = (
            args.max_item_total_chars
            if args.max_item_total_chars is not None
            else self.config.max_item_total_chars
        )
        if max_item <= 0 or max_total <= 0:
            raise ToolError(
                "max_item_chars and max_item_total_chars must be positive."
            )

        sources = self._resolve_trigger_sources(args, warnings)
        blocks: list[TriggerBlock] = []
        total_chars = 0
        for source in sources:
            if total_chars >= max_total:
                warnings.append(
                    "max_item_total_chars reached; truncating trigger inputs."
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
                TriggerBlock(
                    label=source.label,
                    source_type=source.source_type,
                    source_path=source.source_path,
                    content=content,
                    truncated=truncated,
                )
            )

        if not blocks:
            warnings.append("No trigger context provided.")
        return blocks

    def _resolve_trigger_sources(
        self, args: RecollectionTriggersArgs, warnings: list[str]
    ) -> list[_TriggerSource]:
        sources: list[_TriggerSource] = []
        if args.items:
            for item in args.items:
                sources.append(self._load_trigger_item(item))

        if args.item_paths:
            for raw_path in args.item_paths:
                path = self._resolve_path(raw_path)
                content = path.read_text("utf-8", errors="ignore")
                sources.append(
                    _TriggerSource(
                        content=content,
                        label=path.name,
                        source_type=None,
                        source_path=str(path),
                    )
                )

        if not sources and not warnings:
            warnings.append("No trigger context provided.")
        return sources

    def _load_trigger_item(self, item: EventCircumstanceItem) -> _TriggerSource:
        if item.content and item.path:
            raise ToolError("Provide content or path per item, not both.")
        if not (
            item.content
            or item.path
            or item.event
            or item.circumstance
            or item.context
            or item.trigger
        ):
            raise ToolError(
                "Each item must provide content, details, or path."
            )

        label = item.label or item.id or item.path
        text = self._format_trigger_text(item)

        if item.path:
            path = self._resolve_path(item.path)
            file_content = path.read_text("utf-8", errors="ignore")
            text = self._join_sections(text, "Content", file_content)
            return _TriggerSource(
                content=text,
                label=label or path.name,
                source_type=item.source_type,
                source_path=str(path),
            )

        return _TriggerSource(
            content=text,
            label=label,
            source_type=item.source_type,
            source_path=None,
        )

    def _format_trigger_text(self, item: EventCircumstanceItem) -> str:
        lines: list[str] = []
        if item.label or item.id:
            lines.append(f"Label: {item.label or item.id}")
        if item.source_type:
            lines.append(f"Source type: {item.source_type}")
        if item.tags:
            tags = ", ".join(tag for tag in item.tags if tag)
            if tags:
                lines.append(f"Tags: {tags}")
        if item.event:
            lines.append("Event:")
            lines.append(item.event)
        if item.circumstance:
            lines.append("Circumstance:")
            lines.append(item.circumstance)
        if item.context:
            lines.append("Context:")
            lines.append(item.context)
        if item.trigger:
            lines.append("Trigger:")
            lines.append(item.trigger)
        if item.timestamp:
            lines.append(f"Timestamp: {item.timestamp}")
        if item.location:
            lines.append(f"Location: {item.location}")
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

    def _format_trigger_blocks(self, blocks: list[TriggerBlock]) -> str:
        if not blocks:
            return ""
        parts = ["Trigger context:"]
        for block in blocks:
            label = block.label or block.source_path or "trigger"
            type_suffix = f" | {block.source_type}" if block.source_type else ""
            trunc = " (truncated)" if block.truncated else ""
            header = f"[{label}{type_suffix}{trunc}]"
            parts.append(f"{header}\n{block.content}".strip())
        return "\n\n".join(parts).strip()

    def _inject_triggers(
        self,
        messages: list[RecollectionTriggerMessage],
        blocks: list[TriggerBlock],
    ) -> list[RecollectionTriggerMessage]:
        if not blocks:
            return messages
        trigger_text = self._format_trigger_blocks(blocks)
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].role == "user":
                content = messages[idx].content.rstrip()
                messages[idx].content = f"{content}\n\n{trigger_text}".strip()
                return messages
        messages.append(
            RecollectionTriggerMessage(role="user", content=trigger_text)
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
        trigger_focus: str,
        show_steps: bool,
        max_triggers: int,
        prefix: str | None,
    ) -> str:
        show_steps_text = "yes" if show_steps else "no"
        rendered = self._render_template(
            template,
            recollection_style,
            trigger_focus,
            show_steps_text,
            max_triggers,
        )
        if prefix and prefix.strip():
            return f"{prefix.strip()}\n\n{rendered}".strip()
        return rendered.strip()

    def _render_template(
        self,
        template: str,
        recollection_style: str,
        trigger_focus: str,
        show_steps_text: str,
        max_triggers: int,
    ) -> str:
        had_placeholders = (
            "{recollection_style}" in template
            or "{trigger_focus}" in template
            or "{show_steps}" in template
            or "{max_triggers}" in template
        )
        rendered = template
        if "{recollection_style}" in template:
            rendered = rendered.replace("{recollection_style}", recollection_style)
        if "{trigger_focus}" in template:
            rendered = rendered.replace("{trigger_focus}", trigger_focus)
        if "{show_steps}" in template:
            rendered = rendered.replace("{show_steps}", show_steps_text)
        if "{max_triggers}" in template:
            rendered = rendered.replace("{max_triggers}", str(max_triggers))

        if had_placeholders:
            return rendered

        extra = (
            f"Recollection style: {recollection_style}\n"
            f"Trigger focus: {trigger_focus}\n"
            f"Show steps: {show_steps_text}\n"
            f"Max triggers: {max_triggers}"
        )
        return f"{rendered.rstrip()}\n\n{extra}"

    def _normalize_messages(
        self, args: RecollectionTriggersArgs, system_prompt: str
    ) -> list[RecollectionTriggerMessage]:
        messages: list[RecollectionTriggerMessage] = []
        if args.messages:
            for msg in args.messages:
                role = (msg.role or "").strip()
                content = (msg.content or "").strip()
                if not content:
                    continue
                if not role:
                    role = "user"
                messages.append(RecollectionTriggerMessage(role=role, content=content))
        elif args.prompt and args.prompt.strip():
            messages.append(
                RecollectionTriggerMessage(role="user", content=args.prompt.strip())
            )
        else:
            raise ToolError("Provide prompt or messages.")

        if not messages:
            raise ToolError("No usable messages provided.")

        if system_prompt.strip():
            messages.insert(
                0,
                RecollectionTriggerMessage(
                    role="system", content=system_prompt.strip()
                ),
            )
        return messages

    def _call_llm(
        self,
        messages: list[RecollectionTriggerMessage],
        args: RecollectionTriggersArgs,
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

    def _resolve_model(self, args: RecollectionTriggersArgs) -> str:
        return args.llm_model or self.config.llm_model

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, RecollectionTriggersArgs):
            return ToolCallDisplay(summary="recollection_triggers")
        return ToolCallDisplay(
            summary="recollection_triggers",
            details={
                "recollection_style": event.args.recollection_style,
                "trigger_focus": event.args.trigger_focus,
                "show_steps": event.args.show_steps,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, RecollectionTriggersResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = "Recollection triggers complete"
        if event.result.errors:
            message = "Recollection triggers finished with errors"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "answer": event.result.answer,
                "used_recollection_style": event.result.used_recollection_style,
                "used_trigger_focus": event.result.used_trigger_focus,
                "show_steps": event.result.show_steps,
                "max_triggers": event.result.max_triggers,
                "template_source": event.result.template_source,
                "trigger_blocks": len(event.result.trigger_blocks),
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Recollection triggers"
