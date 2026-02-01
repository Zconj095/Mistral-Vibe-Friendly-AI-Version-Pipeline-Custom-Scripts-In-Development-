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


DEFAULT_PROMPT_TEMPLATE = """### FEELING MANAGEMENT + CONTROL MODE (OPT-IN)
Manage and regulate feelings before answering.
- Identify current feelings, triggers, and desired state.
- Suggest grounded regulation strategies and safeguards.
- Separate observed signals from inferences; call out uncertainty.
- Keep internal management private; do not mention this mode in the response.
- If the user asks for steps or show_steps is enabled, give a concise outline.

Feeling style: {feeling_style}
Control focus: {control_focus}
Show steps: {show_steps}
Max actions: {max_actions}
"""

TOOL_PROMPT = (
    "Use `feeling_management_control` to manage and regulate feelings. "
    "Provide `prompt` or `messages`, and optionally set `feeling_style`, "
    "`control_focus`, `show_steps`, and `max_actions`."
)


@dataclass(frozen=True)
class _FeelingSource:
    content: str
    label: str | None
    source_type: str | None
    source_path: str | None


class FeelingManagementMessage(BaseModel):
    role: str
    content: str


class FeelingContextItem(BaseModel):
    id: str | None = Field(default=None, description="Optional item id.")
    label: str | None = Field(default=None, description="Optional label.")
    source_type: str | None = Field(
        default=None, description="Source type (feeling, trigger, feedback, etc.)."
    )
    feeling: str | None = Field(default=None, description="Feeling description.")
    intensity: str | None = Field(default=None, description="Intensity rating.")
    trigger: str | None = Field(default=None, description="Trigger description.")
    context: str | None = Field(default=None, description="Context summary.")
    desired_state: str | None = Field(
        default=None, description="Target feeling state."
    )
    signals: str | None = Field(default=None, description="Observed signals.")
    content: str | None = Field(default=None, description="Inline content.")
    tags: list[str] | None = Field(default=None, description="Optional tags.")
    path: str | None = Field(default=None, description="Path to a context file.")


class FeelingManagementControlArgs(BaseModel):
    prompt: str | None = Field(
        default=None, description="User prompt to solve."
    )
    messages: list[FeelingManagementMessage] | None = Field(
        default=None, description="Optional chat messages."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prefix."
    )
    feeling_style: str | None = Field(
        default=None, description="Feeling management style label."
    )
    control_focus: str | None = Field(
        default=None, description="Control focus label."
    )
    show_steps: bool | None = Field(
        default=None, description="Whether to include a step outline."
    )
    max_actions: int | None = Field(
        default=None, description="Maximum actions in the outline."
    )
    items: list[FeelingContextItem] | None = Field(
        default=None, description="Feeling context items."
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
        default=0.2, description="LLM temperature."
    )
    llm_max_tokens: int = Field(
        default=700, description="LLM max tokens."
    )
    llm_stream: bool = Field(
        default=False, description="Stream LLM tokens."
    )


class FeelingBlock(BaseModel):
    label: str | None
    source_type: str | None
    source_path: str | None
    content: str
    truncated: bool


class FeelingManagementControlResult(BaseModel):
    answer: str
    system_prompt: str
    messages: list[FeelingManagementMessage]
    used_feeling_style: str
    used_control_focus: str
    show_steps: bool
    max_actions: int
    template_source: str
    feeling_blocks: list[FeelingBlock]
    warnings: list[str]
    errors: list[str]
    llm_model: str


class FeelingManagementControlConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    llm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL.",
    )
    llm_model: str = Field(
        default="gpt-oss:latest", description="Default LLM model name."
    )
    default_feeling_style: str = Field(
        default="grounded",
        description="Default feeling management style.",
    )
    default_control_focus: str = Field(
        default="stability-and-alignment",
        description="Default control focus.",
    )
    default_show_steps: bool = Field(
        default=False, description="Default for show_steps."
    )
    default_max_actions: int = Field(
        default=6, description="Default max actions."
    )
    prompt_path: Path | None = Field(
        default=Path.home()
        / "mistral-vibe"
        / "vibe"
        / "core"
        / "prompts"
        / "feeling_management_control.md",
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


class FeelingManagementControlState(BaseToolState):
    pass


class FeelingManagementControl(
    BaseTool[
        FeelingManagementControlArgs,
        FeelingManagementControlResult,
        FeelingManagementControlConfig,
        FeelingManagementControlState,
    ],
    ToolUIData[FeelingManagementControlArgs, FeelingManagementControlResult],
):
    description: ClassVar[str] = (
        "Manage and regulate feelings using a prompt template."
    )

    @classmethod
    def get_tool_prompt(cls) -> str | None:
        return TOOL_PROMPT

    async def run(
        self, args: FeelingManagementControlArgs
    ) -> FeelingManagementControlResult:
        warnings: list[str] = []
        errors: list[str] = []

        feeling_style = (
            args.feeling_style or self.config.default_feeling_style
        ).strip()
        if not feeling_style:
            feeling_style = "grounded"

        control_focus = (
            args.control_focus or self.config.default_control_focus
        ).strip()
        if not control_focus:
            control_focus = "stability-and-alignment"

        show_steps = (
            args.show_steps
            if args.show_steps is not None
            else self.config.default_show_steps
        )
        max_actions = (
            args.max_actions
            if args.max_actions is not None
            else self.config.default_max_actions
        )
        if max_actions <= 0:
            raise ToolError("max_actions must be positive.")

        self._validate_llm_settings(args)

        template, template_source = self._load_template(warnings)
        system_prompt = self._build_system_prompt(
            template,
            feeling_style,
            control_focus,
            show_steps,
            max_actions,
            args.system_prompt,
        )

        feeling_blocks = self._collect_feeling_context(args, warnings)
        messages = self._normalize_messages(args, system_prompt)
        messages = self._inject_feeling_context(messages, feeling_blocks)
        answer = self._call_llm(messages, args)

        return FeelingManagementControlResult(
            answer=answer,
            system_prompt=system_prompt,
            messages=messages,
            used_feeling_style=feeling_style,
            used_control_focus=control_focus,
            show_steps=show_steps,
            max_actions=max_actions,
            template_source=template_source,
            feeling_blocks=feeling_blocks,
            warnings=warnings,
            errors=errors,
            llm_model=self._resolve_model(args),
        )

    def _validate_llm_settings(self, args: FeelingManagementControlArgs) -> None:
        if args.llm_temperature < 0:
            raise ToolError("llm_temperature cannot be negative.")
        if args.llm_max_tokens <= 0:
            raise ToolError("llm_max_tokens must be positive.")

    def _collect_feeling_context(
        self, args: FeelingManagementControlArgs, warnings: list[str]
    ) -> list[FeelingBlock]:
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

        sources = self._resolve_feeling_sources(args, warnings)
        blocks: list[FeelingBlock] = []
        total_chars = 0
        for source in sources:
            if total_chars >= max_total:
                warnings.append(
                    "max_item_total_chars reached; truncating feeling context."
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
                FeelingBlock(
                    label=source.label,
                    source_type=source.source_type,
                    source_path=source.source_path,
                    content=content,
                    truncated=truncated,
                )
            )

        if not blocks:
            warnings.append("No feeling context provided.")
        return blocks

    def _resolve_feeling_sources(
        self, args: FeelingManagementControlArgs, warnings: list[str]
    ) -> list[_FeelingSource]:
        sources: list[_FeelingSource] = []
        if args.items:
            for item in args.items:
                sources.append(self._load_feeling_item(item))

        if args.item_paths:
            for raw_path in args.item_paths:
                path = self._resolve_path(raw_path)
                content = path.read_text("utf-8", errors="ignore")
                sources.append(
                    _FeelingSource(
                        content=content,
                        label=path.name,
                        source_type=None,
                        source_path=str(path),
                    )
                )

        if not sources and not warnings:
            warnings.append("No feeling context provided.")
        return sources

    def _load_feeling_item(self, item: FeelingContextItem) -> _FeelingSource:
        if item.content and item.path:
            raise ToolError("Provide content or path per item, not both.")
        if not (
            item.content
            or item.path
            or item.feeling
            or item.context
            or item.desired_state
        ):
            raise ToolError(
                "Each item must provide content, details, or path."
            )

        label = item.label or item.id or item.path
        text = self._format_feeling_text(item)

        if item.path:
            path = self._resolve_path(item.path)
            file_content = path.read_text("utf-8", errors="ignore")
            text = self._join_sections(text, "Content", file_content)
            return _FeelingSource(
                content=text,
                label=label or path.name,
                source_type=item.source_type,
                source_path=str(path),
            )

        return _FeelingSource(
            content=text,
            label=label,
            source_type=item.source_type,
            source_path=None,
        )

    def _format_feeling_text(self, item: FeelingContextItem) -> str:
        lines: list[str] = []
        if item.label or item.id:
            lines.append(f"Label: {item.label or item.id}")
        if item.source_type:
            lines.append(f"Source type: {item.source_type}")
        if item.tags:
            tags = ", ".join(tag for tag in item.tags if tag)
            if tags:
                lines.append(f"Tags: {tags}")
        if item.feeling:
            lines.append("Feeling:")
            lines.append(item.feeling)
        if item.intensity:
            lines.append(f"Intensity: {item.intensity}")
        if item.trigger:
            lines.append("Trigger:")
            lines.append(item.trigger)
        if item.context:
            lines.append("Context:")
            lines.append(item.context)
        if item.desired_state:
            lines.append("Desired state:")
            lines.append(item.desired_state)
        if item.signals:
            lines.append("Signals:")
            lines.append(item.signals)
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

    def _format_feeling_blocks(self, blocks: list[FeelingBlock]) -> str:
        if not blocks:
            return ""
        parts = ["Feeling context:"]
        for block in blocks:
            label = block.label or block.source_path or "feeling"
            type_suffix = f" | {block.source_type}" if block.source_type else ""
            trunc = " (truncated)" if block.truncated else ""
            header = f"[{label}{type_suffix}{trunc}]"
            parts.append(f"{header}\n{block.content}".strip())
        return "\n\n".join(parts).strip()

    def _inject_feeling_context(
        self,
        messages: list[FeelingManagementMessage],
        blocks: list[FeelingBlock],
    ) -> list[FeelingManagementMessage]:
        if not blocks:
            return messages
        feeling_text = self._format_feeling_blocks(blocks)
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].role == "user":
                content = messages[idx].content.rstrip()
                messages[idx].content = f"{content}\n\n{feeling_text}".strip()
                return messages
        messages.append(FeelingManagementMessage(role="user", content=feeling_text))
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
        feeling_style: str,
        control_focus: str,
        show_steps: bool,
        max_actions: int,
        prefix: str | None,
    ) -> str:
        show_steps_text = "yes" if show_steps else "no"
        rendered = self._render_template(
            template,
            feeling_style,
            control_focus,
            show_steps_text,
            max_actions,
        )
        if prefix and prefix.strip():
            return f"{prefix.strip()}\n\n{rendered}".strip()
        return rendered.strip()

    def _render_template(
        self,
        template: str,
        feeling_style: str,
        control_focus: str,
        show_steps_text: str,
        max_actions: int,
    ) -> str:
        had_placeholders = (
            "{feeling_style}" in template
            or "{control_focus}" in template
            or "{show_steps}" in template
            or "{max_actions}" in template
        )
        rendered = template
        if "{feeling_style}" in template:
            rendered = rendered.replace("{feeling_style}", feeling_style)
        if "{control_focus}" in template:
            rendered = rendered.replace("{control_focus}", control_focus)
        if "{show_steps}" in template:
            rendered = rendered.replace("{show_steps}", show_steps_text)
        if "{max_actions}" in template:
            rendered = rendered.replace("{max_actions}", str(max_actions))

        if had_placeholders:
            return rendered

        extra = (
            f"Feeling style: {feeling_style}\n"
            f"Control focus: {control_focus}\n"
            f"Show steps: {show_steps_text}\n"
            f"Max actions: {max_actions}"
        )
        return f"{rendered.rstrip()}\n\n{extra}"

    def _normalize_messages(
        self, args: FeelingManagementControlArgs, system_prompt: str
    ) -> list[FeelingManagementMessage]:
        messages: list[FeelingManagementMessage] = []
        if args.messages:
            for msg in args.messages:
                role = (msg.role or "").strip()
                content = (msg.content or "").strip()
                if not content:
                    continue
                if not role:
                    role = "user"
                messages.append(FeelingManagementMessage(role=role, content=content))
        elif args.prompt and args.prompt.strip():
            messages.append(
                FeelingManagementMessage(role="user", content=args.prompt.strip())
            )
        else:
            raise ToolError("Provide prompt or messages.")

        if not messages:
            raise ToolError("No usable messages provided.")

        if system_prompt.strip():
            messages.insert(
                0,
                FeelingManagementMessage(
                    role="system", content=system_prompt.strip()
                ),
            )
        return messages

    def _call_llm(
        self,
        messages: list[FeelingManagementMessage],
        args: FeelingManagementControlArgs,
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

    def _resolve_model(self, args: FeelingManagementControlArgs) -> str:
        return args.llm_model or self.config.llm_model

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, FeelingManagementControlArgs):
            return ToolCallDisplay(summary="feeling_management_control")
        return ToolCallDisplay(
            summary="feeling_management_control",
            details={
                "feeling_style": event.args.feeling_style,
                "control_focus": event.args.control_focus,
                "show_steps": event.args.show_steps,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, FeelingManagementControlResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = "Feeling management and control complete"
        if event.result.errors:
            message = "Feeling management and control finished with errors"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "answer": event.result.answer,
                "used_feeling_style": event.result.used_feeling_style,
                "used_control_focus": event.result.used_control_focus,
                "show_steps": event.result.show_steps,
                "max_actions": event.result.max_actions,
                "template_source": event.result.template_source,
                "feeling_blocks": len(event.result.feeling_blocks),
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Feeling management and control"
