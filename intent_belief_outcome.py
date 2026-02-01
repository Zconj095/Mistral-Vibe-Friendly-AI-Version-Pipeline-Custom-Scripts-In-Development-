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


DEFAULT_PROMPT_TEMPLATE = """### INTENT & BELIEF OUTCOME MODE (OPT-IN)
Use intent and belief as motivational frames to drive outcomes.
- Translate intent into actions and checkpoints.
- Treat belief as focus and resilience, not supernatural causation.
- Provide practical steps and realistic constraints.
- Keep reasoning concise; avoid revealing chain-of-thought.

Intent: {intent}
Belief frame: {belief_frame}
Show steps: {show_steps}
Max steps: {max_steps}
"""

TOOL_PROMPT = (
    "Use `intent_belief_outcome` to shape outcomes via intent and belief framing. "
    "Provide `prompt` or `messages`, and optionally set `intent`, `belief_frame`, "
    "`show_steps`, and `max_steps`."
)


class IntentBeliefMessage(BaseModel):
    role: str
    content: str


class IntentBeliefArgs(BaseModel):
    prompt: str | None = Field(
        default=None, description="User prompt to solve."
    )
    messages: list[IntentBeliefMessage] | None = Field(
        default=None, description="Optional chat messages."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prefix."
    )
    intent: str | None = Field(
        default=None, description="Intent label or goal."
    )
    belief_frame: str | None = Field(
        default=None, description="Belief framing label."
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


class IntentBeliefResult(BaseModel):
    answer: str
    system_prompt: str
    messages: list[IntentBeliefMessage]
    used_intent: str
    used_belief_frame: str
    show_steps: bool
    max_steps: int
    template_source: str
    warnings: list[str]
    errors: list[str]
    llm_model: str


class IntentBeliefConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    llm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL.",
    )
    llm_model: str = Field(
        default="gpt-oss:latest", description="Default LLM model name."
    )
    default_intent: str = Field(
        default="Clear, actionable intent.",
        description="Default intent label.",
    )
    default_belief_frame: str = Field(
        default="Focus, resilience, and follow-through.",
        description="Default belief framing.",
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
        / "intent_belief_outcome.md",
        description="Optional path to a prompt template.",
    )
    prompt_max_chars: int = Field(
        default=8000, description="Maximum template characters to load."
    )


class IntentBeliefState(BaseToolState):
    pass


class IntentBeliefOutcome(
    BaseTool[
        IntentBeliefArgs,
        IntentBeliefResult,
        IntentBeliefConfig,
        IntentBeliefState,
    ],
    ToolUIData[IntentBeliefArgs, IntentBeliefResult],
):
    description: ClassVar[str] = (
        "Shape outcomes using intent and belief framing with practical actions."
    )

    @classmethod
    def get_tool_prompt(cls) -> str | None:
        return TOOL_PROMPT

    async def run(
        self, args: IntentBeliefArgs
    ) -> IntentBeliefResult:
        warnings: list[str] = []
        errors: list[str] = []

        intent = (args.intent or self.config.default_intent).strip()
        if not intent:
            intent = "clear, actionable intent"
        belief_frame = (args.belief_frame or self.config.default_belief_frame).strip()
        if not belief_frame:
            belief_frame = "focus, resilience, and follow-through"

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
            template, intent, belief_frame, show_steps, max_steps, args.system_prompt
        )

        messages = self._normalize_messages(args, system_prompt)
        answer = self._call_llm(messages, args)

        return IntentBeliefResult(
            answer=answer,
            system_prompt=system_prompt,
            messages=messages,
            used_intent=intent,
            used_belief_frame=belief_frame,
            show_steps=show_steps,
            max_steps=max_steps,
            template_source=template_source,
            warnings=warnings,
            errors=errors,
            llm_model=self._resolve_model(args),
        )

    def _validate_llm_settings(self, args: IntentBeliefArgs) -> None:
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
        intent: str,
        belief_frame: str,
        show_steps: bool,
        max_steps: int,
        prefix: str | None,
    ) -> str:
        show_steps_text = "yes" if show_steps else "no"
        rendered = self._render_template(
            template, intent, belief_frame, show_steps_text, max_steps
        )
        if prefix and prefix.strip():
            return f"{prefix.strip()}\n\n{rendered}".strip()
        return rendered.strip()

    def _render_template(
        self,
        template: str,
        intent: str,
        belief_frame: str,
        show_steps_text: str,
        max_steps: int,
    ) -> str:
        had_placeholders = (
            "{intent}" in template
            or "{belief_frame}" in template
            or "{show_steps}" in template
            or "{max_steps}" in template
        )
        rendered = template
        if "{intent}" in template:
            rendered = rendered.replace("{intent}", intent)
        if "{belief_frame}" in template:
            rendered = rendered.replace("{belief_frame}", belief_frame)
        if "{show_steps}" in template:
            rendered = rendered.replace("{show_steps}", show_steps_text)
        if "{max_steps}" in template:
            rendered = rendered.replace("{max_steps}", str(max_steps))

        if had_placeholders:
            return rendered

        extra = (
            f"Intent: {intent}\n"
            f"Belief frame: {belief_frame}\n"
            f"Show steps: {show_steps_text}\n"
            f"Max steps: {max_steps}"
        )
        return f"{rendered.rstrip()}\n\n{extra}"

    def _normalize_messages(
        self, args: IntentBeliefArgs, system_prompt: str
    ) -> list[IntentBeliefMessage]:
        messages: list[IntentBeliefMessage] = []
        if args.messages:
            for msg in args.messages:
                role = (msg.role or "").strip()
                content = (msg.content or "").strip()
                if not content:
                    continue
                if not role:
                    role = "user"
                messages.append(
                    IntentBeliefMessage(role=role, content=content)
                )
        elif args.prompt and args.prompt.strip():
            messages.append(
                IntentBeliefMessage(
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
                IntentBeliefMessage(
                    role="system", content=system_prompt.strip()
                ),
            )
        return messages

    def _call_llm(
        self,
        messages: list[IntentBeliefMessage],
        args: IntentBeliefArgs,
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

    def _resolve_model(self, args: IntentBeliefArgs) -> str:
        return args.llm_model or self.config.llm_model

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, IntentBeliefArgs):
            return ToolCallDisplay(summary="intent_belief_outcome")
        return ToolCallDisplay(
            summary="intent_belief_outcome",
            details={
                "intent": event.args.intent,
                "belief_frame": event.args.belief_frame,
                "show_steps": event.args.show_steps,
                "max_steps": event.args.max_steps,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, IntentBeliefResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = "Intent & belief outcome complete"
        if event.result.errors:
            message = "Intent & belief outcome finished with errors"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "answer": event.result.answer,
                "used_intent": event.result.used_intent,
                "used_belief_frame": event.result.used_belief_frame,
                "show_steps": event.result.show_steps,
                "max_steps": event.result.max_steps,
                "template_source": event.result.template_source,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Intent & belief outcome"
