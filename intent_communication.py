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


VALID_MODES = {"send", "receive", "both"}
VALID_CHANNELS = {"text", "speech"}


DEFAULT_PROMPT_TEMPLATE = """### INTENT COMMUNICATION MODE (OPT-IN)
Use intent to craft or interpret communication through text and spoken word.
- For sending: draft a text message and a spoken script that express the intent.
- For receiving: infer the sender intent and suggest a concise response.
- Keep internal reasoning private; do not mention this mode in the response.
- If the user asks for steps or show_steps is enabled, give a concise outline.

Mode: {mode}
Channels: {channels}
Show steps: {show_steps}
Max text chars: {max_text_chars}
Max speech chars: {max_speech_chars}
"""

TOOL_PROMPT = (
    "Use `intent_communication` to turn intent into text and spoken messaging, or "
    "interpret incoming text/speech. Provide `intent` and/or incoming content, "
    "and optionally set `mode`, `channels`, `show_steps`, and length limits."
)


class IntentCommunicationMessage(BaseModel):
    role: str
    content: str


class IntentCommunicationArgs(BaseModel):
    prompt: str | None = Field(
        default=None, description="User prompt to solve."
    )
    messages: list[IntentCommunicationMessage] | None = Field(
        default=None, description="Optional chat messages."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prefix."
    )
    intent: str | None = Field(
        default=None, description="Intent to communicate."
    )
    incoming_text: str | None = Field(
        default=None, description="Incoming text to interpret."
    )
    incoming_speech: str | None = Field(
        default=None, description="Incoming speech transcript to interpret."
    )
    mode: str | None = Field(
        default=None, description="send, receive, or both."
    )
    channels: list[str] | None = Field(
        default=None, description="Output channels: text, speech."
    )
    audience: str | None = Field(
        default=None, description="Audience description."
    )
    tone: str | None = Field(
        default=None, description="Tone guidance."
    )
    language: str | None = Field(
        default=None, description="Language guidance."
    )
    constraints: list[str] | None = Field(
        default=None, description="Constraints for the message."
    )
    max_text_chars: int | None = Field(
        default=None, description="Max characters for text output."
    )
    max_speech_chars: int | None = Field(
        default=None, description="Max characters for speech output."
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


class IntentCommunicationResult(BaseModel):
    answer: str
    system_prompt: str
    messages: list[IntentCommunicationMessage]
    used_mode: str
    used_channels: list[str]
    max_text_chars: int
    max_speech_chars: int
    template_source: str
    warnings: list[str]
    errors: list[str]
    llm_model: str


class IntentCommunicationConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    llm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL.",
    )
    llm_model: str = Field(
        default="gpt-oss:latest", description="Default LLM model name."
    )
    default_mode: str = Field(
        default="send", description="Default mode."
    )
    default_channels: list[str] = Field(
        default_factory=lambda: ["text", "speech"],
        description="Default output channels.",
    )
    default_show_steps: bool = Field(
        default=False, description="Default for show_steps."
    )
    default_max_text_chars: int = Field(
        default=600, description="Default max characters for text."
    )
    default_max_speech_chars: int = Field(
        default=900, description="Default max characters for speech."
    )
    prompt_path: Path | None = Field(
        default=Path.home()
        / "mistral-vibe"
        / "vibe"
        / "core"
        / "prompts"
        / "intent_communication.md",
        description="Optional path to a prompt template.",
    )
    prompt_max_chars: int = Field(
        default=8000, description="Maximum template characters to load."
    )


class IntentCommunicationState(BaseToolState):
    pass


class IntentCommunication(
    BaseTool[
        IntentCommunicationArgs,
        IntentCommunicationResult,
        IntentCommunicationConfig,
        IntentCommunicationState,
    ],
    ToolUIData[IntentCommunicationArgs, IntentCommunicationResult],
):
    description: ClassVar[str] = (
        "Turn intent into text and spoken messaging, or interpret incoming content."
    )

    @classmethod
    def get_tool_prompt(cls) -> str | None:
        return TOOL_PROMPT

    async def run(
        self, args: IntentCommunicationArgs
    ) -> IntentCommunicationResult:
        warnings: list[str] = []
        errors: list[str] = []

        mode = (args.mode or self.config.default_mode).strip().lower()
        if mode not in VALID_MODES:
            raise ToolError(
                f"mode must be one of: {', '.join(sorted(VALID_MODES))}"
            )

        channels = self._normalize_channels(args, mode)
        max_text_chars = (
            args.max_text_chars
            if args.max_text_chars is not None
            else self.config.default_max_text_chars
        )
        max_speech_chars = (
            args.max_speech_chars
            if args.max_speech_chars is not None
            else self.config.default_max_speech_chars
        )
        if max_text_chars <= 0 or max_speech_chars <= 0:
            raise ToolError("max_text_chars and max_speech_chars must be positive.")

        show_steps = (
            args.show_steps
            if args.show_steps is not None
            else self.config.default_show_steps
        )

        self._validate_llm_settings(args)

        template, template_source = self._load_template(warnings)
        system_prompt = self._build_system_prompt(
            template,
            mode,
            channels,
            show_steps,
            max_text_chars,
            max_speech_chars,
            args.system_prompt,
        )

        user_prompt = self._build_user_prompt(args, mode, channels)
        messages = self._normalize_messages(args, system_prompt, user_prompt)
        answer = self._call_llm(messages, args)

        return IntentCommunicationResult(
            answer=answer,
            system_prompt=system_prompt,
            messages=messages,
            used_mode=mode,
            used_channels=channels,
            max_text_chars=max_text_chars,
            max_speech_chars=max_speech_chars,
            template_source=template_source,
            warnings=warnings,
            errors=errors,
            llm_model=self._resolve_model(args),
        )

    def _normalize_channels(
        self, args: IntentCommunicationArgs, mode: str
    ) -> list[str]:
        if mode == "receive":
            return []
        raw = args.channels or self.config.default_channels
        channels = [item.strip().lower() for item in raw if item and item.strip()]
        if not channels:
            raise ToolError("channels cannot be empty for send mode.")
        invalid = [item for item in channels if item not in VALID_CHANNELS]
        if invalid:
            raise ToolError(
                f"channels must be in {sorted(VALID_CHANNELS)}; invalid: {invalid}"
            )
        return channels

    def _validate_llm_settings(self, args: IntentCommunicationArgs) -> None:
        if args.llm_temperature < 0:
            raise ToolError("llm_temperature cannot be negative.")
        if args.llm_max_tokens <= 0:
            raise ToolError("llm_max_tokens must be positive.")

    def _build_user_prompt(
        self, args: IntentCommunicationArgs, mode: str, channels: list[str]
    ) -> str:
        lines: list[str] = []

        intent_value = (args.intent or args.prompt or "").strip()
        incoming_text = (args.incoming_text or "").strip()
        incoming_speech = (args.incoming_speech or "").strip()

        if mode in {"send", "both"} and not intent_value:
            raise ToolError("intent or prompt is required for send mode.")
        if mode in {"receive", "both"} and not (incoming_text or incoming_speech):
            raise ToolError(
                "incoming_text or incoming_speech is required for receive mode."
            )

        lines.append(f"Mode: {mode}")
        if channels:
            lines.append(f"Channels: {', '.join(channels)}")

        if intent_value:
            lines.append(f"Intent: {intent_value}")
        if incoming_text:
            lines.append("Incoming text:")
            lines.append(incoming_text)
        if incoming_speech:
            lines.append("Incoming speech:")
            lines.append(incoming_speech)

        if args.audience:
            lines.append(f"Audience: {args.audience.strip()}")
        if args.tone:
            lines.append(f"Tone: {args.tone.strip()}")
        if args.language:
            lines.append(f"Language: {args.language.strip()}")
        if args.constraints:
            constraints = [item.strip() for item in args.constraints if item.strip()]
            if constraints:
                lines.append("Constraints:")
                lines.extend([f"- {item}" for item in constraints])

        lines.append("Output format:")
        lines.append("- For send: TEXT: <text message> and/or SPEECH: <spoken script>.")
        lines.append(
            "- For receive: RECEIVED_INTENT: <summary>, KEY_POINTS: <bullets>, RESPONSE: <suggested reply>."
        )
        lines.append("Only include sections relevant to the selected mode.")

        return "\n".join(lines).strip()

    def _normalize_messages(
        self,
        args: IntentCommunicationArgs,
        system_prompt: str,
        user_prompt: str,
    ) -> list[IntentCommunicationMessage]:
        messages: list[IntentCommunicationMessage] = []
        if args.messages:
            for msg in args.messages:
                role = (msg.role or "").strip()
                content = (msg.content or "").strip()
                if not content:
                    continue
                if not role:
                    role = "user"
                messages.append(IntentCommunicationMessage(role=role, content=content))

        if user_prompt:
            messages.append(
                IntentCommunicationMessage(role="user", content=user_prompt)
            )

        if not messages:
            raise ToolError("Provide prompt, intent, incoming content, or messages.")

        if system_prompt.strip():
            messages.insert(
                0,
                IntentCommunicationMessage(
                    role="system", content=system_prompt.strip()
                ),
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
        mode: str,
        channels: list[str],
        show_steps: bool,
        max_text_chars: int,
        max_speech_chars: int,
        prefix: str | None,
    ) -> str:
        show_steps_text = "yes" if show_steps else "no"
        rendered = self._render_template(
            template,
            mode,
            ", ".join(channels) if channels else "receive-only",
            show_steps_text,
            max_text_chars,
            max_speech_chars,
        )
        if prefix and prefix.strip():
            return f"{prefix.strip()}\n\n{rendered}".strip()
        return rendered.strip()

    def _render_template(
        self,
        template: str,
        mode: str,
        channels: str,
        show_steps_text: str,
        max_text_chars: int,
        max_speech_chars: int,
    ) -> str:
        rendered = template
        rendered = rendered.replace("{mode}", mode)
        rendered = rendered.replace("{channels}", channels)
        rendered = rendered.replace("{show_steps}", show_steps_text)
        rendered = rendered.replace("{max_text_chars}", str(max_text_chars))
        rendered = rendered.replace("{max_speech_chars}", str(max_speech_chars))
        return rendered

    def _call_llm(
        self,
        messages: list[IntentCommunicationMessage],
        args: IntentCommunicationArgs,
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

    def _resolve_model(self, args: IntentCommunicationArgs) -> str:
        return args.llm_model or self.config.llm_model

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, IntentCommunicationArgs):
            return ToolCallDisplay(summary="intent_communication")
        return ToolCallDisplay(
            summary="intent_communication",
            details={
                "mode": event.args.mode,
                "channels": event.args.channels,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, IntentCommunicationResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = "Intent communication complete"
        if event.result.errors:
            message = "Intent communication finished with errors"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "answer": event.result.answer,
                "used_mode": event.result.used_mode,
                "used_channels": event.result.used_channels,
                "template_source": event.result.template_source,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Intent communication"
