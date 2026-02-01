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


DEFAULT_PROMPT_TEMPLATE = """### CLASSIFYING INFORMATION MODE (OPT-IN)
Classify items into labeled classes before answering.
- Assign each item to a class using clear, consistent criteria.
- Note ambiguous cases or overlaps when they affect the answer.
- Keep internal classification private; do not mention this mode in the response.
- If the user asks for steps or show_steps is enabled, give a concise outline.

Classification scheme: {classification_scheme}
Show steps: {show_steps}
Max classes: {max_classes}
"""

TOOL_PROMPT = (
    "Use `classifying_information` to classify items into labeled classes. "
    "Provide `prompt` or `messages`, and optionally set `classification_scheme`, "
    "`show_steps`, and `max_classes` to control the classification."
)


class ClassifyingInformationMessage(BaseModel):
    role: str
    content: str


class ClassifyingInformationArgs(BaseModel):
    prompt: str | None = Field(
        default=None, description="User prompt to solve."
    )
    messages: list[ClassifyingInformationMessage] | None = Field(
        default=None, description="Optional chat messages."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prefix."
    )
    classification_scheme: str | None = Field(
        default=None, description="Classification scheme label."
    )
    show_steps: bool | None = Field(
        default=None, description="Whether to include a step outline."
    )
    max_classes: int | None = Field(
        default=None, description="Maximum classes in the outline."
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


class ClassifyingInformationResult(BaseModel):
    answer: str
    system_prompt: str
    messages: list[ClassifyingInformationMessage]
    used_classification_scheme: str
    show_steps: bool
    max_classes: int
    template_source: str
    warnings: list[str]
    errors: list[str]
    llm_model: str


class ClassifyingInformationConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    llm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL.",
    )
    llm_model: str = Field(
        default="gpt-oss:latest", description="Default LLM model name."
    )
    default_classification_scheme: str = Field(
        default="taxonomy",
        description="Default classification scheme.",
    )
    default_show_steps: bool = Field(
        default=False, description="Default for show_steps."
    )
    default_max_classes: int = Field(
        default=6, description="Default max classes."
    )
    prompt_path: Path | None = Field(
        default=Path.home()
        / "mistral-vibe"
        / "vibe"
        / "core"
        / "prompts"
        / "classifying_information.md",
        description="Optional path to a prompt template.",
    )
    prompt_max_chars: int = Field(
        default=8000, description="Maximum template characters to load."
    )


class ClassifyingInformationState(BaseToolState):
    pass


class ClassifyingInformation(
    BaseTool[
        ClassifyingInformationArgs,
        ClassifyingInformationResult,
        ClassifyingInformationConfig,
        ClassifyingInformationState,
    ],
    ToolUIData[ClassifyingInformationArgs, ClassifyingInformationResult],
):
    description: ClassVar[str] = (
        "Generate responses guided by explicit classification."
    )

    @classmethod
    def get_tool_prompt(cls) -> str | None:
        return TOOL_PROMPT

    async def run(
        self, args: ClassifyingInformationArgs
    ) -> ClassifyingInformationResult:
        warnings: list[str] = []
        errors: list[str] = []

        classification_scheme = (
            args.classification_scheme or self.config.default_classification_scheme
        ).strip()
        if not classification_scheme:
            classification_scheme = "taxonomy"

        show_steps = (
            args.show_steps
            if args.show_steps is not None
            else self.config.default_show_steps
        )
        max_classes = (
            args.max_classes
            if args.max_classes is not None
            else self.config.default_max_classes
        )
        if max_classes <= 0:
            raise ToolError("max_classes must be positive.")

        self._validate_llm_settings(args)

        template, template_source = self._load_template(warnings)
        system_prompt = self._build_system_prompt(
            template,
            classification_scheme,
            show_steps,
            max_classes,
            args.system_prompt,
        )

        messages = self._normalize_messages(args, system_prompt)
        answer = self._call_llm(messages, args)

        return ClassifyingInformationResult(
            answer=answer,
            system_prompt=system_prompt,
            messages=messages,
            used_classification_scheme=classification_scheme,
            show_steps=show_steps,
            max_classes=max_classes,
            template_source=template_source,
            warnings=warnings,
            errors=errors,
            llm_model=self._resolve_model(args),
        )

    def _validate_llm_settings(self, args: ClassifyingInformationArgs) -> None:
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
        classification_scheme: str,
        show_steps: bool,
        max_classes: int,
        prefix: str | None,
    ) -> str:
        show_steps_text = "yes" if show_steps else "no"
        rendered = self._render_template(
            template,
            classification_scheme,
            show_steps_text,
            max_classes,
        )
        if prefix and prefix.strip():
            return f"{prefix.strip()}\n\n{rendered}".strip()
        return rendered.strip()

    def _render_template(
        self,
        template: str,
        classification_scheme: str,
        show_steps_text: str,
        max_classes: int,
    ) -> str:
        had_placeholders = (
            "{classification_scheme}" in template
            or "{show_steps}" in template
            or "{max_classes}" in template
        )
        rendered = template
        if "{classification_scheme}" in template:
            rendered = rendered.replace("{classification_scheme}", classification_scheme)
        if "{show_steps}" in template:
            rendered = rendered.replace("{show_steps}", show_steps_text)
        if "{max_classes}" in template:
            rendered = rendered.replace("{max_classes}", str(max_classes))

        if had_placeholders:
            return rendered

        extra = (
            f"Classification scheme: {classification_scheme}\n"
            f"Show steps: {show_steps_text}\n"
            f"Max classes: {max_classes}"
        )
        return f"{rendered.rstrip()}\n\n{extra}"

    def _normalize_messages(
        self, args: ClassifyingInformationArgs, system_prompt: str
    ) -> list[ClassifyingInformationMessage]:
        messages: list[ClassifyingInformationMessage] = []
        if args.messages:
            for msg in args.messages:
                role = (msg.role or "").strip()
                content = (msg.content or "").strip()
                if not content:
                    continue
                if not role:
                    role = "user"
                messages.append(
                    ClassifyingInformationMessage(role=role, content=content)
                )
        elif args.prompt and args.prompt.strip():
            messages.append(
                ClassifyingInformationMessage(
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
                ClassifyingInformationMessage(
                    role="system", content=system_prompt.strip()
                ),
            )
        return messages

    def _call_llm(
        self,
        messages: list[ClassifyingInformationMessage],
        args: ClassifyingInformationArgs,
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

    def _resolve_model(self, args: ClassifyingInformationArgs) -> str:
        return args.llm_model or self.config.llm_model

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ClassifyingInformationArgs):
            return ToolCallDisplay(summary="classifying_information")
        return ToolCallDisplay(
            summary="classifying_information",
            details={
                "classification_scheme": event.args.classification_scheme,
                "show_steps": event.args.show_steps,
                "max_classes": event.args.max_classes,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ClassifyingInformationResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = "Classifying information complete"
        if event.result.errors:
            message = "Classifying information finished with errors"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "answer": event.result.answer,
                "used_classification_scheme": event.result.used_classification_scheme,
                "show_steps": event.result.show_steps,
                "max_classes": event.result.max_classes,
                "template_source": event.result.template_source,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Classifying information"
