from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
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


DEFAULT_PROMPT_TEMPLATE = """### FILTER + PROCESS MODE (OPT-IN)
Filter information and process information simultaneously.
- Apply inclusion/exclusion criteria while summarizing or transforming.
- Provide concise filtered outputs with confidence notes.
- Keep reasoning concise; avoid revealing chain-of-thought.

Focus: {focus}
Show steps: {show_steps}
Max outputs: {max_outputs}
"""

TOOL_PROMPT = (
    "Use `simultaneous_filter_process_information` to filter and process "
    "information simultaneously. Provide `prompt` or `messages`, and optionally "
    "set `focus`, `show_steps`, and `max_outputs`."
)


class FilterProcessMessage(BaseModel):
    role: str
    content: str


class FilterProcessArgs(BaseModel):
    prompt: str | None = Field(
        default=None, description="User prompt to solve."
    )
    messages: list[FilterProcessMessage] | None = Field(
        default=None, description="Optional chat messages."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prefix."
    )
    focus: str | None = Field(
        default=None, description="Focus label or domain."
    )
    show_steps: bool | None = Field(
        default=None, description="Whether to include a step outline."
    )
    max_outputs: int | None = Field(
        default=None, description="Maximum outputs in the outline."
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
        default=900, description="LLM max tokens."
    )
    llm_stream: bool = Field(
        default=False, description="Stream LLM tokens."
    )


class FilterProcessResult(BaseModel):
    answer: str
    system_prompt: str
    messages: list[FilterProcessMessage]
    used_focus: str
    show_steps: bool
    max_outputs: int
    template_source: str
    warnings: list[str]
    errors: list[str]
    llm_model: str


class FilterProcessConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    llm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL.",
    )
    llm_model: str = Field(
        default="gpt-oss:latest", description="Default LLM model name."
    )
    default_focus: str = Field(
        default="Filter and process information simultaneously.",
        description="Default focus label.",
    )
    default_show_steps: bool = Field(
        default=False, description="Default for show_steps."
    )
    default_max_outputs: int = Field(
        default=4, description="Default max outputs."
    )
    prompt_path: Path | None = Field(
        default=Path.home()
        / "mistral-vibe"
        / "vibe"
        / "core"
        / "prompts"
        / "simultaneous_filter_process_information.md",
        description="Optional path to a prompt template.",
    )
    prompt_max_chars: int = Field(
        default=8000, description="Maximum template characters to load."
    )


class FilterProcessState(BaseToolState):
    pass


class SimultaneousFilterProcessInformation(
    BaseTool[
        FilterProcessArgs,
        FilterProcessResult,
        FilterProcessConfig,
        FilterProcessState,
    ],
    ToolUIData[FilterProcessArgs, FilterProcessResult],
):
    description: ClassVar[str] = (
        "Filter information and process information simultaneously."
    )

    @classmethod
    def get_tool_prompt(cls) -> str | None:
        return TOOL_PROMPT

    async def run(
        self, args: FilterProcessArgs
    ) -> FilterProcessResult:
        warnings: list[str] = []
        errors: list[str] = []

        focus = (args.focus or self.config.default_focus).strip()
        if not focus:
            focus = "filter and process information simultaneously"

        show_steps = (
            args.show_steps
            if args.show_steps is not None
            else self.config.default_show_steps
        )
        max_outputs = (
            args.max_outputs
            if args.max_outputs is not None
            else self.config.default_max_outputs
        )
        if max_outputs <= 0:
            raise ToolError("max_outputs must be positive.")

        self._validate_llm_settings(args)

        template, template_source = self._load_template(warnings)
        system_prompt = self._build_system_prompt(
            template, focus, show_steps, max_outputs, args.system_prompt
        )

        messages = self._normalize_messages(args, system_prompt)
        answer = self._call_llm(messages, args)

        return FilterProcessResult(
            answer=answer,
            system_prompt=system_prompt,
            messages=messages,
            used_focus=focus,
            show_steps=show_steps,
            max_outputs=max_outputs,
            template_source=template_source,
            warnings=warnings,
            errors=errors,
            llm_model=self._resolve_model(args),
        )

    def _validate_llm_settings(self, args: FilterProcessArgs) -> None:
        if args.llm_temperature < 0:
            raise ToolError("llm_temperature cannot be negative.")
        if args.llm_max_tokens <= 0:
            raise ToolError("llm_max_tokens must be positive.")

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
        focus: str,
        show_steps: bool,
        max_outputs: int,
        prefix: str | None,
    ) -> str:
        show_steps_text = "yes" if show_steps else "no"
        rendered = self._render_template(
            template, focus, show_steps_text, max_outputs
        )
        if prefix and prefix.strip():
            return f"{prefix.strip()}\\n\\n{rendered}".strip()
        return rendered.strip()

    def _render_template(
        self,
        template: str,
        focus: str,
        show_steps_text: str,
        max_outputs: int,
    ) -> str:
        had_placeholders = (
            "{focus}" in template
            or "{show_steps}" in template
            or "{max_outputs}" in template
        )
        rendered = template
        if "{focus}" in template:
            rendered = rendered.replace("{focus}", focus)
        if "{show_steps}" in template:
            rendered = rendered.replace("{show_steps}", show_steps_text)
        if "{max_outputs}" in template:
            rendered = rendered.replace("{max_outputs}", str(max_outputs))

        if had_placeholders:
            return rendered

        extra = (
            f"Focus: {focus}\n"
            f"Show steps: {show_steps_text}\n"
            f"Max outputs: {max_outputs}"
        )
        return f"{rendered.rstrip()}\n\n{extra}"

    def _normalize_messages(
        self, args: FilterProcessArgs, system_prompt: str
    ) -> list[FilterProcessMessage]:
        messages: list[FilterProcessMessage] = []
        if args.messages:
            for msg in args.messages:
                role = (msg.role or "").strip()
                content = (msg.content or "").strip()
                if not content:
                    continue
                if not role:
                    role = "user"
                messages.append(
                    FilterProcessMessage(role=role, content=content)
                )
        elif args.prompt and args.prompt.strip():
            messages.append(
                FilterProcessMessage(
                    role="user", content=args.prompt.strip()
                )
            )
        else:
            raise ToolError("Provide prompt or messages.")

        if not messages:
            raise ToolError("No usable messages provided.")

        if system_prompt.strip():
            messages.insert(
                0,
                FilterProcessMessage(
                    role="system", content=system_prompt.strip()
                ),
            )
        return messages

    def _call_llm(
        self,
        messages: list[FilterProcessMessage],
        args: FilterProcessArgs,
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

    def _resolve_model(self, args: FilterProcessArgs) -> str:
        return args.llm_model or self.config.llm_model

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, FilterProcessArgs):
            return ToolCallDisplay(
                summary="simultaneous_filter_process_information"
            )
        return ToolCallDisplay(
            summary="simultaneous_filter_process_information",
            details={
                "focus": event.args.focus,
                "show_steps": event.args.show_steps,
                "max_outputs": event.args.max_outputs,
            },
        )

    @classmethod
    def get_result_display(
        cls, event: ToolResultEvent
    ) -> ToolResultDisplay:
        if not isinstance(event.result, FilterProcessResult):
            return ToolResultDisplay(
                success=False,
                message=event.error or event.skip_reason or "No result",
            )
        message = "Filter + process complete"
        if event.result.errors:
            message = "Filter + process finished with errors"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "answer": event.result.answer,
                "used_focus": event.result.used_focus,
                "show_steps": event.result.show_steps,
                "max_outputs": event.result.max_outputs,
                "template_source": event.result.template_source,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Filter and process information simultaneously"
