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


KNOWN_FORMS = {
    "structured",
    "causal",
    "mechanistic",
    "counterfactual",
    "probabilistic",
    "diagnostic",
    "systems",
    "optimization",
    "temporal",
    "analogical",
    "deductive",
    "inductive",
    "abductive",
    "critical",
    "strategic",
}
FORM_ALIASES = {
    "counter-factual": "counterfactual",
    "bayesian": "probabilistic",
    "systems-thinking": "systems",
    "systemic": "systems",
}

DEFAULT_PROMPT_TEMPLATE = """### ADVANCED REASONING MODE (OPT-IN)
Develop an advanced form of reasoning for the task.
- Make assumptions explicit and test alternatives.
- Separate evidence from inference; quantify uncertainty when possible.
- Prefer structured explanations with causal or mechanistic links.
- Keep internal reasoning private; do not mention this mode in the response.
- If the user asks for steps or show_steps is enabled, give a concise outline.

Reasoning form: {reasoning_form}
Reasoning focus: {reasoning_focus}
Reasoning depth: {reasoning_depth}
Show steps: {show_steps}
Max steps: {max_steps}
"""

TOOL_PROMPT = (
    "Use `advanced_reasoning` to apply an advanced reasoning mode. "
    "Provide `prompt` or `messages`, and optionally set `reasoning_form`, "
    "`reasoning_focus`, `reasoning_depth`, `show_steps`, and `max_steps`."
)


@dataclass(frozen=True)
class _ReasoningSource:
    content: str
    label: str | None
    form: str | None
    source_path: str | None


class AdvancedReasoningMessage(BaseModel):
    role: str
    content: str


class ReasoningContextItem(BaseModel):
    id: str | None = Field(default=None, description="Optional context id.")
    label: str | None = Field(default=None, description="Optional label.")
    form: str | None = Field(default=None, description="Reasoning form label.")
    content: str | None = Field(default=None, description="Inline context content.")
    path: str | None = Field(default=None, description="Path to a context file.")
    tags: list[str] | None = Field(default=None, description="Optional tags.")


class AdvancedReasoningArgs(BaseModel):
    prompt: str | None = Field(
        default=None, description="User prompt to solve."
    )
    messages: list[AdvancedReasoningMessage] | None = Field(
        default=None, description="Optional chat messages."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prefix."
    )
    reasoning_form: str | None = Field(
        default=None, description="Reasoning form label."
    )
    reasoning_focus: str | None = Field(
        default=None, description="Reasoning focus label."
    )
    reasoning_depth: str | None = Field(
        default=None, description="Reasoning depth label."
    )
    show_steps: bool | None = Field(
        default=None, description="Whether to include a step outline."
    )
    max_steps: int | None = Field(
        default=None, description="Maximum steps in the outline."
    )
    reasoning_inputs: list[ReasoningContextItem] | None = Field(
        default=None, description="Reasoning context items."
    )
    reasoning_paths: list[str] | None = Field(
        default=None, description="Additional reasoning context file paths."
    )
    max_reasoning_chars: int | None = Field(
        default=None, description="Max chars per reasoning item."
    )
    max_reasoning_total_chars: int | None = Field(
        default=None, description="Max total reasoning chars."
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


class ReasoningBlock(BaseModel):
    label: str | None
    form: str | None
    source_path: str | None
    content: str
    truncated: bool


class AdvancedReasoningResult(BaseModel):
    answer: str
    system_prompt: str
    messages: list[AdvancedReasoningMessage]
    used_reasoning_form: str
    used_reasoning_focus: str
    used_reasoning_depth: str
    show_steps: bool
    max_steps: int
    template_source: str
    reasoning_blocks: list[ReasoningBlock]
    warnings: list[str]
    errors: list[str]
    llm_model: str


class AdvancedReasoningConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    llm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL.",
    )
    llm_model: str = Field(
        default="gpt-oss:latest", description="Default LLM model name."
    )
    default_reasoning_form: str = Field(
        default="structured",
        description="Default reasoning form.",
    )
    default_reasoning_focus: str = Field(
        default="balanced",
        description="Default reasoning focus.",
    )
    default_reasoning_depth: str = Field(
        default="deep",
        description="Default reasoning depth.",
    )
    default_show_steps: bool = Field(
        default=False, description="Default for show_steps."
    )
    default_max_steps: int = Field(
        default=6, description="Default max steps."
    )
    prompt_path: Path | None = Field(
        default=Path.home()
        / "mistral-vibe"
        / "vibe"
        / "core"
        / "prompts"
        / "advanced_reasoning.md",
        description="Optional path to a prompt template.",
    )
    prompt_max_chars: int = Field(
        default=8000, description="Maximum template characters to load."
    )
    max_reasoning_chars: int = Field(
        default=4000, description="Maximum characters per reasoning item."
    )
    max_reasoning_total_chars: int = Field(
        default=12000, description="Maximum total reasoning characters."
    )


class AdvancedReasoningState(BaseToolState):
    pass


class AdvancedReasoning(
    BaseTool[
        AdvancedReasoningArgs,
        AdvancedReasoningResult,
        AdvancedReasoningConfig,
        AdvancedReasoningState,
    ],
    ToolUIData[AdvancedReasoningArgs, AdvancedReasoningResult],
):
    description: ClassVar[str] = (
        "Apply an advanced reasoning mode using a dedicated prompt template."
    )

    @classmethod
    def get_tool_prompt(cls) -> str | None:
        return TOOL_PROMPT

    async def run(
        self, args: AdvancedReasoningArgs
    ) -> AdvancedReasoningResult:
        warnings: list[str] = []
        errors: list[str] = []

        reasoning_form = (
            args.reasoning_form or self.config.default_reasoning_form
        ).strip()
        if not reasoning_form:
            reasoning_form = "structured"
        reasoning_form = self._normalize_form(reasoning_form, warnings)

        reasoning_focus = (
            args.reasoning_focus or self.config.default_reasoning_focus
        ).strip()
        if not reasoning_focus:
            reasoning_focus = "balanced"

        reasoning_depth = (
            args.reasoning_depth or self.config.default_reasoning_depth
        ).strip()
        if not reasoning_depth:
            reasoning_depth = "deep"

        show_steps = (
            args.show_steps
            if args.show_steps is not None
            else self.config.default_show_steps
        )
        max_steps = (
            args.max_steps
            if args.max_steps is not None
            else self.config.default_max_steps
        )
        if max_steps <= 0:
            raise ToolError("max_steps must be positive.")

        self._validate_llm_settings(args)

        template, template_source = self._load_template(warnings)
        system_prompt = self._build_system_prompt(
            template,
            reasoning_form,
            reasoning_focus,
            reasoning_depth,
            show_steps,
            max_steps,
            args.system_prompt,
        )

        reasoning_blocks = self._collect_reasoning_inputs(args, warnings)
        messages = self._normalize_messages(args, system_prompt)
        messages = self._inject_reasoning_context(messages, reasoning_blocks)
        answer = self._call_llm(messages, args)

        return AdvancedReasoningResult(
            answer=answer,
            system_prompt=system_prompt,
            messages=messages,
            used_reasoning_form=reasoning_form,
            used_reasoning_focus=reasoning_focus,
            used_reasoning_depth=reasoning_depth,
            show_steps=show_steps,
            max_steps=max_steps,
            template_source=template_source,
            reasoning_blocks=reasoning_blocks,
            warnings=warnings,
            errors=errors,
            llm_model=self._resolve_model(args),
        )

    def _normalize_form(self, form: str, warnings: list[str]) -> str:
        value = form.strip().lower()
        value = FORM_ALIASES.get(value, value)
        if value not in KNOWN_FORMS:
            warnings.append(f"Unknown reasoning form '{form}'; using as-is.")
        return value

    def _validate_llm_settings(self, args: AdvancedReasoningArgs) -> None:
        if args.llm_temperature < 0:
            raise ToolError("llm_temperature cannot be negative.")
        if args.llm_max_tokens <= 0:
            raise ToolError("llm_max_tokens must be positive.")

    def _collect_reasoning_inputs(
        self, args: AdvancedReasoningArgs, warnings: list[str]
    ) -> list[ReasoningBlock]:
        max_item = (
            args.max_reasoning_chars
            if args.max_reasoning_chars is not None
            else self.config.max_reasoning_chars
        )
        max_total = (
            args.max_reasoning_total_chars
            if args.max_reasoning_total_chars is not None
            else self.config.max_reasoning_total_chars
        )
        if max_item <= 0 or max_total <= 0:
            raise ToolError(
                "max_reasoning_chars and max_reasoning_total_chars must be positive."
            )

        sources = self._resolve_reasoning_sources(args, warnings)
        blocks: list[ReasoningBlock] = []
        total_chars = 0

        for source in sources:
            if total_chars >= max_total:
                warnings.append(
                    "max_reasoning_total_chars reached; truncating reasoning inputs."
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
                ReasoningBlock(
                    label=source.label,
                    form=source.form,
                    source_path=source.source_path,
                    content=content,
                    truncated=truncated,
                )
            )

        if not blocks:
            warnings.append("No reasoning context provided.")
        return blocks

    def _resolve_reasoning_sources(
        self, args: AdvancedReasoningArgs, warnings: list[str]
    ) -> list[_ReasoningSource]:
        sources: list[_ReasoningSource] = []
        if args.reasoning_inputs:
            for item in args.reasoning_inputs:
                sources.append(self._load_reasoning_item(item, warnings))

        if args.reasoning_paths:
            for raw_path in args.reasoning_paths:
                path = self._resolve_path(raw_path)
                content = path.read_text("utf-8", errors="ignore")
                sources.append(
                    _ReasoningSource(
                        content=content,
                        label=path.name,
                        form=None,
                        source_path=str(path),
                    )
                )

        if not sources and not warnings:
            warnings.append("No reasoning context provided.")
        return sources

    def _load_reasoning_item(
        self, item: ReasoningContextItem, warnings: list[str]
    ) -> _ReasoningSource:
        if item.content and item.path:
            raise ToolError("Provide content or path per reasoning item, not both.")
        if not item.content and not item.path:
            raise ToolError("Each reasoning item must provide content or path.")

        label = item.label or item.id or item.path
        form = self._normalize_form(item.form, warnings) if item.form else None
        content = self._format_reasoning_text(item)

        if item.path:
            path = self._resolve_path(item.path)
            file_content = path.read_text("utf-8", errors="ignore")
            combined = self._join_sections(content, "Content", file_content)
            return _ReasoningSource(
                content=combined,
                label=label or path.name,
                form=form,
                source_path=str(path),
            )

        return _ReasoningSource(
            content=content,
            label=label,
            form=form,
            source_path=None,
        )

    def _format_reasoning_text(self, item: ReasoningContextItem) -> str:
        lines: list[str] = []
        if item.label or item.id:
            lines.append(f"Label: {item.label or item.id}")
        if item.form:
            lines.append(f"Form: {item.form}")
        if item.tags:
            tags = ", ".join(tag for tag in item.tags if tag)
            if tags:
                lines.append(f"Tags: {tags}")
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

    def _format_reasoning_blocks(self, blocks: list[ReasoningBlock]) -> str:
        if not blocks:
            return ""
        parts = ["Reasoning context:"]
        for block in blocks:
            label = block.label or block.source_path or "reasoning"
            form_suffix = f" | {block.form}" if block.form else ""
            trunc = " (truncated)" if block.truncated else ""
            header = f"[{label}{form_suffix}{trunc}]"
            parts.append(f"{header}\n{block.content}".strip())
        return "\n\n".join(parts).strip()

    def _inject_reasoning_context(
        self,
        messages: list[AdvancedReasoningMessage],
        blocks: list[ReasoningBlock],
    ) -> list[AdvancedReasoningMessage]:
        if not blocks:
            return messages
        reasoning_text = self._format_reasoning_blocks(blocks)
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].role == "user":
                content = messages[idx].content.rstrip()
                messages[idx].content = f"{content}\n\n{reasoning_text}".strip()
                return messages
        messages.append(AdvancedReasoningMessage(role="user", content=reasoning_text))
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
        reasoning_form: str,
        reasoning_focus: str,
        reasoning_depth: str,
        show_steps: bool,
        max_steps: int,
        prefix: str | None,
    ) -> str:
        show_steps_text = "yes" if show_steps else "no"
        rendered = self._render_template(
            template,
            reasoning_form,
            reasoning_focus,
            reasoning_depth,
            show_steps_text,
            max_steps,
        )
        if prefix and prefix.strip():
            return f"{prefix.strip()}\n\n{rendered}".strip()
        return rendered.strip()

    def _render_template(
        self,
        template: str,
        reasoning_form: str,
        reasoning_focus: str,
        reasoning_depth: str,
        show_steps_text: str,
        max_steps: int,
    ) -> str:
        had_placeholders = (
            "{reasoning_form}" in template
            or "{reasoning_focus}" in template
            or "{reasoning_depth}" in template
            or "{show_steps}" in template
            or "{max_steps}" in template
        )
        rendered = template
        if "{reasoning_form}" in template:
            rendered = rendered.replace("{reasoning_form}", reasoning_form)
        if "{reasoning_focus}" in template:
            rendered = rendered.replace("{reasoning_focus}", reasoning_focus)
        if "{reasoning_depth}" in template:
            rendered = rendered.replace("{reasoning_depth}", reasoning_depth)
        if "{show_steps}" in template:
            rendered = rendered.replace("{show_steps}", show_steps_text)
        if "{max_steps}" in template:
            rendered = rendered.replace("{max_steps}", str(max_steps))

        if had_placeholders:
            return rendered

        extra = (
            f"Reasoning form: {reasoning_form}\n"
            f"Reasoning focus: {reasoning_focus}\n"
            f"Reasoning depth: {reasoning_depth}\n"
            f"Show steps: {show_steps_text}\n"
            f"Max steps: {max_steps}"
        )
        return f"{rendered.rstrip()}\n\n{extra}"

    def _normalize_messages(
        self, args: AdvancedReasoningArgs, system_prompt: str
    ) -> list[AdvancedReasoningMessage]:
        messages: list[AdvancedReasoningMessage] = []
        if args.messages:
            for msg in args.messages:
                role = (msg.role or "").strip()
                content = (msg.content or "").strip()
                if not content:
                    continue
                if not role:
                    role = "user"
                messages.append(AdvancedReasoningMessage(role=role, content=content))
        elif args.prompt and args.prompt.strip():
            messages.append(
                AdvancedReasoningMessage(role="user", content=args.prompt.strip())
            )
        else:
            raise ToolError("Provide prompt or messages.")

        if not messages:
            raise ToolError("No usable messages provided.")

        if system_prompt.strip():
            messages.insert(
                0,
                AdvancedReasoningMessage(role="system", content=system_prompt.strip()),
            )
        return messages

    def _call_llm(
        self,
        messages: list[AdvancedReasoningMessage],
        args: AdvancedReasoningArgs,
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

    def _resolve_model(self, args: AdvancedReasoningArgs) -> str:
        return args.llm_model or self.config.llm_model

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, AdvancedReasoningArgs):
            return ToolCallDisplay(summary="advanced_reasoning")
        return ToolCallDisplay(
            summary="advanced_reasoning",
            details={
                "reasoning_form": event.args.reasoning_form,
                "reasoning_focus": event.args.reasoning_focus,
                "reasoning_depth": event.args.reasoning_depth,
                "show_steps": event.args.show_steps,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, AdvancedReasoningResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = "Advanced reasoning complete"
        if event.result.errors:
            message = "Advanced reasoning finished with errors"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "answer": event.result.answer,
                "used_reasoning_form": event.result.used_reasoning_form,
                "used_reasoning_focus": event.result.used_reasoning_focus,
                "used_reasoning_depth": event.result.used_reasoning_depth,
                "show_steps": event.result.show_steps,
                "max_steps": event.result.max_steps,
                "template_source": event.result.template_source,
                "reasoning_blocks": len(event.result.reasoning_blocks),
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Advanced reasoning"
