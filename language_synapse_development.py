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


DEFAULT_PROMPT_TEMPLATE = """### LANGUAGE SYNAPSE DEVELOPMENT MODE (OPT-IN)
Develop cognitive features through language and language arts.
- Use structured language practice to build connections.
- Provide exercises and feedback loops.
- Keep reasoning concise; avoid revealing chain-of-thought.

Focus: {focus}
Practice cadence: {practice_cadence}
Show steps: {show_steps}
Max steps: {max_steps}
"""

TOOL_PROMPT = (
    "Use `language_synapse_development` to develop cognitive features through language. "
    "Provide `prompt` or `messages`, and optionally set `focus`, `practice_cadence`, "
    "`show_steps`, and `max_steps`."
)


class LanguageSynapseMessage(BaseModel):
    role: str
    content: str


class LanguageSynapseArgs(BaseModel):
    prompt: str | None = Field(
        default=None, description="User prompt to solve."
    )
    messages: list[LanguageSynapseMessage] | None = Field(
        default=None, description="Optional chat messages."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prefix."
    )
    focus: str | None = Field(
        default=None, description="Focus label or language domain."
    )
    practice_cadence: str | None = Field(
        default=None, description="Practice cadence or schedule."
    )
    show_steps: bool | None = Field(
        default=None, description="Whether to include a step outline."
    )
    max_steps: int | None = Field(
        default=None, description="Maximum steps in the outline."
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
        default=800, description="LLM max tokens."
    )
    llm_stream: bool = Field(
        default=False, description="Stream LLM tokens."
    )


class LanguageSynapseResult(BaseModel):
    answer: str
    system_prompt: str
    messages: list[LanguageSynapseMessage]
    used_focus: str
    practice_cadence: str
    show_steps: bool
    max_steps: int
    template_source: str
    warnings: list[str]
    errors: list[str]
    llm_model: str


class LanguageSynapseConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    llm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL.",
    )
    llm_model: str = Field(
        default="gpt-oss:latest", description="Default LLM model name."
    )
    default_focus: str = Field(
        default="Language arts and cognitive development.",
        description="Default focus label.",
    )
    default_practice_cadence: str = Field(
        default="Daily practice.",
        description="Default practice cadence.",
    )
    default_show_steps: bool = Field(
        default=False, description="Default for show_steps."
    )
    default_max_steps: int = Field(
        default=7, description="Default max steps."
    )
    prompt_path: Path | None = Field(
        default=Path.home()
        / "mistral-vibe"
        / "vibe"
        / "core"
        / "prompts"
        / "language_synapse_development.md",
        description="Optional path to a prompt template.",
    )
    prompt_max_chars: int = Field(
        default=8000, description="Maximum template characters to load."
    )


class LanguageSynapseState(BaseToolState):
    pass


class LanguageSynapseDevelopment(
    BaseTool[
        LanguageSynapseArgs,
        LanguageSynapseResult,
        LanguageSynapseConfig,
        LanguageSynapseState,
    ],
    ToolUIData[LanguageSynapseArgs, LanguageSynapseResult],
):
    description: ClassVar[str] = (
        "Develop cognitive features through language and language arts."
    )

    @classmethod
    def get_tool_prompt(cls) -> str | None:
        return TOOL_PROMPT

    async def run(self, args: LanguageSynapseArgs) -> LanguageSynapseResult:
        warnings: list[str] = []
        errors: list[str] = []

        focus = (args.focus or self.config.default_focus).strip()
        if not focus:
            focus = "language arts and cognitive development"
        cadence = (args.practice_cadence or self.config.default_practice_cadence).strip()
        if not cadence:
            cadence = "daily practice"

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
            template, focus, cadence, show_steps, max_steps, args.system_prompt
        )

        messages = self._normalize_messages(args, system_prompt)
        answer = self._call_llm(messages, args)

        return LanguageSynapseResult(
            answer=answer,
            system_prompt=system_prompt,
            messages=messages,
            used_focus=focus,
            practice_cadence=cadence,
            show_steps=show_steps,
            max_steps=max_steps,
            template_source=template_source,
            warnings=warnings,
            errors=errors,
            llm_model=self._resolve_model(args),
        )

    def _validate_llm_settings(self, args: LanguageSynapseArgs) -> None:
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
        cadence: str,
        show_steps: bool,
        max_steps: int,
        prefix: str | None,
    ) -> str:
        show_steps_text = "yes" if show_steps else "no"
        rendered = self._render_template(
            template, focus, cadence, show_steps_text, max_steps
        )
        if prefix and prefix.strip():
            return f"{prefix.strip()}\n\n{rendered}".strip()
        return rendered.strip()

    def _render_template(
        self,
        template: str,
        focus: str,
        cadence: str,
        show_steps_text: str,
        max_steps: int,
    ) -> str:
        had_placeholders = (
            "{focus}" in template
            or "{practice_cadence}" in template
            or "{show_steps}" in template
            or "{max_steps}" in template
        )
        rendered = template
        if "{focus}" in template:
            rendered = rendered.replace("{focus}", focus)
        if "{practice_cadence}" in template:
            rendered = rendered.replace("{practice_cadence}", cadence)
        if "{show_steps}" in template:
            rendered = rendered.replace("{show_steps}", show_steps_text)
        if "{max_steps}" in template:
            rendered = rendered.replace("{max_steps}", str(max_steps))

        if had_placeholders:
            return rendered

        extra = (
            f"Focus: {focus}\n"
            f"Practice cadence: {cadence}\n"
            f"Show steps: {show_steps_text}\n"
            f"Max steps: {max_steps}"
        )
        return f"{rendered.rstrip()}\n\n{extra}"

    def _normalize_messages(
        self, args: LanguageSynapseArgs, system_prompt: str
    ) -> list[LanguageSynapseMessage]:
        messages: list[LanguageSynapseMessage] = []
        if args.messages:
            for msg in args.messages:
                role = (msg.role or "").strip()
                content = (msg.content or "").strip()
                if not content:
                    continue
                if not role:
                    role = "user"
                messages.append(
                    LanguageSynapseMessage(role=role, content=content)
                )
        elif args.prompt and args.prompt.strip():
            messages.append(
                LanguageSynapseMessage(
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
                LanguageSynapseMessage(
                    role="system", content=system_prompt.strip()
                ),
            )
        return messages

    def _call_llm(
        self,
        messages: list[LanguageSynapseMessage],
        args: LanguageSynapseArgs,
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

    def _resolve_model(self, args: LanguageSynapseArgs) -> str:
        return args.llm_model or self.config.llm_model

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, LanguageSynapseArgs):
            return ToolCallDisplay(summary="language_synapse_development")
        return ToolCallDisplay(
            summary="language_synapse_development",
            details={
                "focus": event.args.focus,
                "practice_cadence": event.args.practice_cadence,
                "show_steps": event.args.show_steps,
                "max_steps": event.args.max_steps,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, LanguageSynapseResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = "Language synapse development complete"
        if event.result.errors:
            message = "Language synapse development finished with errors"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "answer": event.result.answer,
                "used_focus": event.result.used_focus,
                "practice_cadence": event.result.practice_cadence,
                "show_steps": event.result.show_steps,
                "max_steps": event.result.max_steps,
                "template_source": event.result.template_source,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Language synapse development"
