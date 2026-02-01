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


VALID_SENSES = {"sight", "hearing", "touch", "smell", "taste"}
SENSE_ALIASES = {
    "vision": "sight",
    "visual": "sight",
    "audition": "hearing",
    "audio": "hearing",
    "sound": "hearing",
    "tactile": "touch",
    "olfaction": "smell",
    "odor": "smell",
    "gustation": "taste",
}


DEFAULT_PROMPT_TEMPLATE = """### FIVE SENSES MODE (OPT-IN)
Ground understanding in the five senses to enrich interpretation.
- Consider sight, hearing, touch, smell, and taste when relevant.
- Call out missing or weak sensory cues when they affect the answer.
- Keep internal grounding private; do not mention this mode in the response.
- If the user asks for steps or show_steps is enabled, give a concise outline.

Style: {style}
Focus senses: {focus_senses}
Show steps: {show_steps}
Max cues: {max_cues}
"""

TOOL_PROMPT = (
    "Use `five_senses_understanding` to ground responses in sensory cues. "
    "Provide `prompt` or `messages`, and optionally set `style`, `focus_senses`, "
    "`show_steps`, and `max_cues` to control the output."
)


@dataclass(frozen=True)
class _SensorySource:
    content: str
    label: str | None
    sense: str | None
    source_path: str | None


class FiveSensesMessage(BaseModel):
    role: str
    content: str


class SensoryInputItem(BaseModel):
    id: str | None = Field(default=None, description="Optional input id.")
    label: str | None = Field(default=None, description="Optional label.")
    sense: str | None = Field(default=None, description="Sense label.")
    content: str | None = Field(default=None, description="Inline sensory content.")
    path: str | None = Field(default=None, description="Path to a sensory file.")


class FiveSensesUnderstandingArgs(BaseModel):
    prompt: str | None = Field(
        default=None, description="User prompt to solve."
    )
    messages: list[FiveSensesMessage] | None = Field(
        default=None, description="Optional chat messages."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prefix."
    )
    style: str | None = Field(
        default=None, description="Sensory grounding style label."
    )
    focus_senses: list[str] | None = Field(
        default=None, description="Subset of senses to prioritize."
    )
    show_steps: bool | None = Field(
        default=None, description="Whether to include a step outline."
    )
    max_cues: int | None = Field(
        default=None, description="Maximum cues in the outline."
    )
    sensory_inputs: list[SensoryInputItem] | None = Field(
        default=None, description="Sensory input items."
    )
    sensory_paths: list[str] | None = Field(
        default=None, description="Additional sensory file paths."
    )
    max_sensory_chars: int | None = Field(
        default=None, description="Max chars per sensory item."
    )
    max_sensory_total_chars: int | None = Field(
        default=None, description="Max total sensory chars."
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


class SensoryBlock(BaseModel):
    label: str | None
    sense: str | None
    source_path: str | None
    content: str
    truncated: bool


class FiveSensesUnderstandingResult(BaseModel):
    answer: str
    system_prompt: str
    messages: list[FiveSensesMessage]
    used_style: str
    focus_senses: list[str]
    show_steps: bool
    max_cues: int
    template_source: str
    sensory_blocks: list[SensoryBlock]
    warnings: list[str]
    errors: list[str]
    llm_model: str


class FiveSensesUnderstandingConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    llm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL.",
    )
    llm_model: str = Field(
        default="gpt-oss:latest", description="Default LLM model name."
    )
    default_style: str = Field(
        default="grounded",
        description="Default sensory grounding style.",
    )
    default_show_steps: bool = Field(
        default=False, description="Default for show_steps."
    )
    default_max_cues: int = Field(
        default=6, description="Default max cues."
    )
    prompt_path: Path | None = Field(
        default=Path.home()
        / "mistral-vibe"
        / "vibe"
        / "core"
        / "prompts"
        / "five_senses_understanding.md",
        description="Optional path to a prompt template.",
    )
    prompt_max_chars: int = Field(
        default=8000, description="Maximum template characters to load."
    )
    max_sensory_chars: int = Field(
        default=4000, description="Maximum characters per sensory item."
    )
    max_sensory_total_chars: int = Field(
        default=12000, description="Maximum total sensory characters."
    )


class FiveSensesUnderstandingState(BaseToolState):
    pass


class FiveSensesUnderstanding(
    BaseTool[
        FiveSensesUnderstandingArgs,
        FiveSensesUnderstandingResult,
        FiveSensesUnderstandingConfig,
        FiveSensesUnderstandingState,
    ],
    ToolUIData[FiveSensesUnderstandingArgs, FiveSensesUnderstandingResult],
):
    description: ClassVar[str] = (
        "Ground responses in the five senses using a prompt template."
    )

    @classmethod
    def get_tool_prompt(cls) -> str | None:
        return TOOL_PROMPT

    async def run(
        self, args: FiveSensesUnderstandingArgs
    ) -> FiveSensesUnderstandingResult:
        warnings: list[str] = []
        errors: list[str] = []

        style = (args.style or self.config.default_style).strip()
        if not style:
            style = "grounded"

        focus_senses = self._normalize_focus_senses(args.focus_senses)
        show_steps = (
            args.show_steps
            if args.show_steps is not None
            else self.config.default_show_steps
        )
        max_cues = (
            args.max_cues
            if args.max_cues is not None
            else self.config.default_max_cues
        )
        if max_cues <= 0:
            raise ToolError("max_cues must be positive.")

        self._validate_llm_settings(args)

        template, template_source = self._load_template(warnings)
        system_prompt = self._build_system_prompt(
            template,
            style,
            focus_senses,
            show_steps,
            max_cues,
            args.system_prompt,
        )

        sensory_blocks = self._collect_sensory_inputs(args, warnings)
        messages = self._normalize_messages(args, system_prompt)
        messages = self._inject_sensory_context(messages, sensory_blocks)
        answer = self._call_llm(messages, args)

        return FiveSensesUnderstandingResult(
            answer=answer,
            system_prompt=system_prompt,
            messages=messages,
            used_style=style,
            focus_senses=focus_senses,
            show_steps=show_steps,
            max_cues=max_cues,
            template_source=template_source,
            sensory_blocks=sensory_blocks,
            warnings=warnings,
            errors=errors,
            llm_model=self._resolve_model(args),
        )

    def _normalize_focus_senses(self, senses: list[str] | None) -> list[str]:
        if not senses:
            return sorted(VALID_SENSES)
        normalized: list[str] = []
        for sense in senses:
            if not sense or not sense.strip():
                continue
            value = self._normalize_sense(sense)
            if value not in normalized:
                normalized.append(value)
        if not normalized:
            raise ToolError("focus_senses must include at least one sense.")
        return normalized

    def _normalize_sense(self, sense: str) -> str:
        value = sense.strip().lower()
        value = SENSE_ALIASES.get(value, value)
        if value not in VALID_SENSES:
            raise ToolError(
                f"Unknown sense '{sense}'. Expected one of: {sorted(VALID_SENSES)}"
            )
        return value

    def _validate_llm_settings(self, args: FiveSensesUnderstandingArgs) -> None:
        if args.llm_temperature < 0:
            raise ToolError("llm_temperature cannot be negative.")
        if args.llm_max_tokens <= 0:
            raise ToolError("llm_max_tokens must be positive.")

    def _collect_sensory_inputs(
        self, args: FiveSensesUnderstandingArgs, warnings: list[str]
    ) -> list[SensoryBlock]:
        max_item = (
            args.max_sensory_chars
            if args.max_sensory_chars is not None
            else self.config.max_sensory_chars
        )
        max_total = (
            args.max_sensory_total_chars
            if args.max_sensory_total_chars is not None
            else self.config.max_sensory_total_chars
        )
        if max_item <= 0 or max_total <= 0:
            raise ToolError(
                "max_sensory_chars and max_sensory_total_chars must be positive."
            )

        sources = self._resolve_sensory_sources(args, warnings)
        blocks: list[SensoryBlock] = []
        total_chars = 0

        for source in sources:
            if total_chars >= max_total:
                warnings.append(
                    "max_sensory_total_chars reached; truncating sensory inputs."
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
                SensoryBlock(
                    label=source.label,
                    sense=source.sense,
                    source_path=source.source_path,
                    content=content,
                    truncated=truncated,
                )
            )

        if not blocks:
            warnings.append("No sensory context provided.")
        return blocks

    def _resolve_sensory_sources(
        self, args: FiveSensesUnderstandingArgs, warnings: list[str]
    ) -> list[_SensorySource]:
        sources: list[_SensorySource] = []
        if args.sensory_inputs:
            for item in args.sensory_inputs:
                sources.append(self._load_sensory_item(item))

        if args.sensory_paths:
            for raw_path in args.sensory_paths:
                path = self._resolve_path(raw_path)
                content = path.read_text("utf-8", errors="ignore")
                sources.append(
                    _SensorySource(
                        content=content,
                        label=path.name,
                        sense=None,
                        source_path=str(path),
                    )
                )

        if not sources and not warnings:
            warnings.append("No sensory context provided.")
        return sources

    def _load_sensory_item(self, item: SensoryInputItem) -> _SensorySource:
        if item.content and item.path:
            raise ToolError("Provide content or path per sensory item, not both.")
        if not item.content and not item.path:
            raise ToolError("Each sensory item must provide content or path.")

        label = item.label or item.id or item.path
        sense = self._normalize_sense(item.sense) if item.sense else None
        if item.content is not None:
            return _SensorySource(
                content=item.content,
                label=label,
                sense=sense,
                source_path=None,
            )

        path = self._resolve_path(item.path or "")
        content = path.read_text("utf-8", errors="ignore")
        return _SensorySource(
            content=content,
            label=label or path.name,
            sense=sense,
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
            raise ToolError(
                "Security error: cannot resolve the provided path."
            ) from exc
        if not resolved.exists():
            raise ToolError(f"File not found at: {resolved}")
        if resolved.is_dir():
            raise ToolError(f"Path is a directory, not a file: {resolved}")
        return resolved

    def _format_sensory_blocks(self, blocks: list[SensoryBlock]) -> str:
        if not blocks:
            return ""
        parts = ["Sensory context:"]
        for block in blocks:
            label = block.label or block.source_path or "sensory"
            sense_suffix = f" | {block.sense}" if block.sense else ""
            trunc = " (truncated)" if block.truncated else ""
            header = f"[{label}{sense_suffix}{trunc}]"
            parts.append(f"{header}\n{block.content}".strip())
        return "\n\n".join(parts).strip()

    def _inject_sensory_context(
        self,
        messages: list[FiveSensesMessage],
        blocks: list[SensoryBlock],
    ) -> list[FiveSensesMessage]:
        if not blocks:
            return messages
        sensory_text = self._format_sensory_blocks(blocks)
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].role == "user":
                content = messages[idx].content.rstrip()
                messages[idx].content = f"{content}\n\n{sensory_text}".strip()
                return messages
        messages.append(FiveSensesMessage(role="user", content=sensory_text))
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
        style: str,
        focus_senses: list[str],
        show_steps: bool,
        max_cues: int,
        prefix: str | None,
    ) -> str:
        show_steps_text = "yes" if show_steps else "no"
        rendered = self._render_template(
            template,
            style,
            ", ".join(focus_senses),
            show_steps_text,
            max_cues,
        )
        if prefix and prefix.strip():
            return f"{prefix.strip()}\n\n{rendered}".strip()
        return rendered.strip()

    def _render_template(
        self,
        template: str,
        style: str,
        focus_senses: str,
        show_steps_text: str,
        max_cues: int,
    ) -> str:
        rendered = template
        rendered = rendered.replace("{style}", style)
        rendered = rendered.replace("{focus_senses}", focus_senses)
        rendered = rendered.replace("{show_steps}", show_steps_text)
        rendered = rendered.replace("{max_cues}", str(max_cues))
        return rendered

    def _normalize_messages(
        self, args: FiveSensesUnderstandingArgs, system_prompt: str
    ) -> list[FiveSensesMessage]:
        messages: list[FiveSensesMessage] = []
        if args.messages:
            for msg in args.messages:
                role = (msg.role or "").strip()
                content = (msg.content or "").strip()
                if not content:
                    continue
                if not role:
                    role = "user"
                messages.append(FiveSensesMessage(role=role, content=content))
        elif args.prompt and args.prompt.strip():
            messages.append(
                FiveSensesMessage(role="user", content=args.prompt.strip())
            )
        else:
            raise ToolError("Provide prompt or messages.")

        if not messages:
            raise ToolError("No usable messages provided.")

        if system_prompt.strip():
            messages.insert(
                0,
                FiveSensesMessage(role="system", content=system_prompt.strip())
            )
        return messages

    def _call_llm(
        self,
        messages: list[FiveSensesMessage],
        args: FiveSensesUnderstandingArgs,
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

    def _resolve_model(self, args: FiveSensesUnderstandingArgs) -> str:
        return args.llm_model or self.config.llm_model

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, FiveSensesUnderstandingArgs):
            return ToolCallDisplay(summary="five_senses_understanding")
        return ToolCallDisplay(
            summary="five_senses_understanding",
            details={
                "style": event.args.style,
                "focus_senses": event.args.focus_senses,
                "show_steps": event.args.show_steps,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, FiveSensesUnderstandingResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = "Five senses understanding complete"
        if event.result.errors:
            message = "Five senses understanding finished with errors"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "answer": event.result.answer,
                "used_style": event.result.used_style,
                "focus_senses": event.result.focus_senses,
                "show_steps": event.result.show_steps,
                "max_cues": event.result.max_cues,
                "template_source": event.result.template_source,
                "sensory_blocks": len(event.result.sensory_blocks),
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Five senses understanding"
