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


KNOWN_MODALITIES = {
    "sight",
    "hearing",
    "touch",
    "smell",
    "taste",
    "proprioception",
    "balance",
    "temperature",
    "pain",
    "interoception",
}
MODALITY_ALIASES = {
    "vision": "sight",
    "visual": "sight",
    "audition": "hearing",
    "audio": "hearing",
    "sound": "hearing",
    "tactile": "touch",
    "somatosensory": "touch",
    "olfaction": "smell",
    "odor": "smell",
    "scent": "smell",
    "gustation": "taste",
    "flavor": "taste",
    "vestibular": "balance",
    "equilibrium": "balance",
    "kinesthetic": "proprioception",
    "kinaesthetic": "proprioception",
    "kinesthesia": "proprioception",
    "thermal": "temperature",
    "heat": "temperature",
    "cold": "temperature",
    "nociception": "pain",
    "interoceptive": "interoception",
}

DEFAULT_PROMPT_TEMPLATE = """### SENSORY CAPABILITIES MODE (OPT-IN)
Develop sensory capabilities by grounding perception, calibration, and integration.
- Map signals to modalities and call out uncertainty or missing cues.
- Distinguish raw sensory cues from inferences.
- Blend modalities when helpful but avoid inventing missing signals.
- Keep internal calibration private; do not mention this mode in the response.
- If the user asks for steps or show_steps is enabled, give a concise outline.

Capability style: {capability_style}
Modalities: {modalities}
Capability goals: {capability_goals}
Show steps: {show_steps}
Max capabilities: {max_capabilities}
"""

TOOL_PROMPT = (
    "Use `sensory_capabilities` to develop and refine sensory grounding. "
    "Provide `prompt` or `messages`, and optionally set `capability_style`, "
    "`modalities`, `capability_goals`, `show_steps`, and `max_capabilities`."
)


@dataclass(frozen=True)
class _SensoryCapabilitySource:
    content: str
    label: str | None
    modality: str | None
    source_path: str | None


class SensoryCapabilitiesMessage(BaseModel):
    role: str
    content: str


class SensoryCapabilityInputItem(BaseModel):
    id: str | None = Field(default=None, description="Optional input id.")
    label: str | None = Field(default=None, description="Optional label.")
    modality: str | None = Field(default=None, description="Modality label.")
    content: str | None = Field(default=None, description="Inline sensory content.")
    path: str | None = Field(default=None, description="Path to a sensory file.")
    tags: list[str] | None = Field(default=None, description="Optional tags.")


class SensoryCapabilitiesArgs(BaseModel):
    prompt: str | None = Field(
        default=None, description="User prompt to solve."
    )
    messages: list[SensoryCapabilitiesMessage] | None = Field(
        default=None, description="Optional chat messages."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prefix."
    )
    capability_style: str | None = Field(
        default=None, description="Capability development style label."
    )
    modalities: list[str] | None = Field(
        default=None, description="Modalities to prioritize."
    )
    capability_goals: list[str] | None = Field(
        default=None, description="Capability goals or targets."
    )
    show_steps: bool | None = Field(
        default=None, description="Whether to include a step outline."
    )
    max_capabilities: int | None = Field(
        default=None, description="Maximum capabilities in the outline."
    )
    sensory_inputs: list[SensoryCapabilityInputItem] | None = Field(
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


class SensoryCapabilityBlock(BaseModel):
    label: str | None
    modality: str | None
    source_path: str | None
    content: str
    truncated: bool


class SensoryCapabilitiesResult(BaseModel):
    answer: str
    system_prompt: str
    messages: list[SensoryCapabilitiesMessage]
    used_capability_style: str
    used_modalities: list[str]
    capability_goals: list[str]
    show_steps: bool
    max_capabilities: int
    template_source: str
    sensory_blocks: list[SensoryCapabilityBlock]
    warnings: list[str]
    errors: list[str]
    llm_model: str


class SensoryCapabilitiesConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    llm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL.",
    )
    llm_model: str = Field(
        default="gpt-oss:latest", description="Default LLM model name."
    )
    default_capability_style: str = Field(
        default="adaptive",
        description="Default capability development style.",
    )
    default_modalities: list[str] = Field(
        default_factory=lambda: [
            "sight",
            "hearing",
            "touch",
            "smell",
            "taste",
            "proprioception",
            "balance",
            "temperature",
            "pain",
        ],
        description="Default modalities to include.",
    )
    default_show_steps: bool = Field(
        default=False, description="Default for show_steps."
    )
    default_max_capabilities: int = Field(
        default=6, description="Default max capabilities."
    )
    prompt_path: Path | None = Field(
        default=Path.home()
        / "mistral-vibe"
        / "vibe"
        / "core"
        / "prompts"
        / "sensory_capabilities.md",
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


class SensoryCapabilitiesState(BaseToolState):
    pass


class SensoryCapabilities(
    BaseTool[
        SensoryCapabilitiesArgs,
        SensoryCapabilitiesResult,
        SensoryCapabilitiesConfig,
        SensoryCapabilitiesState,
    ],
    ToolUIData[SensoryCapabilitiesArgs, SensoryCapabilitiesResult],
):
    description: ClassVar[str] = (
        "Develop sensory capabilities using a dedicated prompt template."
    )

    @classmethod
    def get_tool_prompt(cls) -> str | None:
        return TOOL_PROMPT

    async def run(
        self, args: SensoryCapabilitiesArgs
    ) -> SensoryCapabilitiesResult:
        warnings: list[str] = []
        errors: list[str] = []

        capability_style = (
            args.capability_style or self.config.default_capability_style
        ).strip()
        if not capability_style:
            capability_style = "adaptive"

        modalities = self._normalize_modalities(args.modalities, warnings)
        capability_goals = self._normalize_capability_goals(args.capability_goals)

        show_steps = (
            args.show_steps
            if args.show_steps is not None
            else self.config.default_show_steps
        )
        max_capabilities = (
            args.max_capabilities
            if args.max_capabilities is not None
            else self.config.default_max_capabilities
        )
        if max_capabilities <= 0:
            raise ToolError("max_capabilities must be positive.")

        self._validate_llm_settings(args)

        template, template_source = self._load_template(warnings)
        system_prompt = self._build_system_prompt(
            template,
            capability_style,
            modalities,
            capability_goals,
            show_steps,
            max_capabilities,
            args.system_prompt,
        )

        sensory_blocks = self._collect_sensory_inputs(args, warnings)
        messages = self._normalize_messages(args, system_prompt)
        messages = self._inject_sensory_context(messages, sensory_blocks)
        answer = self._call_llm(messages, args)

        return SensoryCapabilitiesResult(
            answer=answer,
            system_prompt=system_prompt,
            messages=messages,
            used_capability_style=capability_style,
            used_modalities=modalities,
            capability_goals=capability_goals,
            show_steps=show_steps,
            max_capabilities=max_capabilities,
            template_source=template_source,
            sensory_blocks=sensory_blocks,
            warnings=warnings,
            errors=errors,
            llm_model=self._resolve_model(args),
        )

    def _normalize_modalities(
        self, modalities: list[str] | None, warnings: list[str]
    ) -> list[str]:
        values = modalities or list(self.config.default_modalities)
        normalized: list[str] = []
        warned: set[str] = set()
        for modality in values:
            if not modality or not modality.strip():
                continue
            value = modality.strip().lower()
            value = MODALITY_ALIASES.get(value, value)
            if value not in normalized:
                normalized.append(value)
            if value not in KNOWN_MODALITIES and value not in warned:
                warnings.append(f"Unknown modality '{modality}'; using as-is.")
                warned.add(value)
        if not normalized:
            raise ToolError("modalities must include at least one modality.")
        return normalized

    def _normalize_capability_goals(self, goals: list[str] | None) -> list[str]:
        if not goals:
            return []
        normalized: list[str] = []
        for goal in goals:
            if goal and goal.strip():
                normalized.append(goal.strip())
        return normalized

    def _validate_llm_settings(self, args: SensoryCapabilitiesArgs) -> None:
        if args.llm_temperature < 0:
            raise ToolError("llm_temperature cannot be negative.")
        if args.llm_max_tokens <= 0:
            raise ToolError("llm_max_tokens must be positive.")

    def _collect_sensory_inputs(
        self, args: SensoryCapabilitiesArgs, warnings: list[str]
    ) -> list[SensoryCapabilityBlock]:
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
        blocks: list[SensoryCapabilityBlock] = []
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
                SensoryCapabilityBlock(
                    label=source.label,
                    modality=source.modality,
                    source_path=source.source_path,
                    content=content,
                    truncated=truncated,
                )
            )

        if not blocks:
            warnings.append("No sensory context provided.")
        return blocks

    def _resolve_sensory_sources(
        self, args: SensoryCapabilitiesArgs, warnings: list[str]
    ) -> list[_SensoryCapabilitySource]:
        sources: list[_SensoryCapabilitySource] = []
        if args.sensory_inputs:
            for item in args.sensory_inputs:
                sources.append(self._load_sensory_item(item, warnings))

        if args.sensory_paths:
            for raw_path in args.sensory_paths:
                path = self._resolve_path(raw_path)
                content = path.read_text("utf-8", errors="ignore")
                sources.append(
                    _SensoryCapabilitySource(
                        content=content,
                        label=path.name,
                        modality=None,
                        source_path=str(path),
                    )
                )

        if not sources and not warnings:
            warnings.append("No sensory context provided.")
        return sources

    def _load_sensory_item(
        self, item: SensoryCapabilityInputItem, warnings: list[str]
    ) -> _SensoryCapabilitySource:
        if item.content and item.path:
            raise ToolError("Provide content or path per sensory item, not both.")
        if not item.content and not item.path:
            raise ToolError("Each sensory item must provide content or path.")

        label = item.label or item.id or item.path
        modality = (
            self._normalize_modality(item.modality, warnings)
            if item.modality
            else None
        )
        content = self._format_sensory_text(item)

        if item.path:
            path = self._resolve_path(item.path)
            file_content = path.read_text("utf-8", errors="ignore")
            combined = self._join_sections(content, "Content", file_content)
            return _SensoryCapabilitySource(
                content=combined,
                label=label or path.name,
                modality=modality,
                source_path=str(path),
            )

        return _SensoryCapabilitySource(
            content=content,
            label=label,
            modality=modality,
            source_path=None,
        )

    def _normalize_modality(
        self, modality: str, warnings: list[str]
    ) -> str:
        value = modality.strip().lower()
        value = MODALITY_ALIASES.get(value, value)
        if value not in KNOWN_MODALITIES:
            warnings.append(f"Unknown modality '{modality}'; using as-is.")
        return value

    def _format_sensory_text(self, item: SensoryCapabilityInputItem) -> str:
        lines: list[str] = []
        if item.label or item.id:
            lines.append(f"Label: {item.label or item.id}")
        if item.modality:
            lines.append(f"Modality: {item.modality}")
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

    def _format_sensory_blocks(self, blocks: list[SensoryCapabilityBlock]) -> str:
        if not blocks:
            return ""
        parts = ["Sensory context:"]
        for block in blocks:
            label = block.label or block.source_path or "sensory"
            modality_suffix = f" | {block.modality}" if block.modality else ""
            trunc = " (truncated)" if block.truncated else ""
            header = f"[{label}{modality_suffix}{trunc}]"
            parts.append(f"{header}\n{block.content}".strip())
        return "\n\n".join(parts).strip()

    def _inject_sensory_context(
        self,
        messages: list[SensoryCapabilitiesMessage],
        blocks: list[SensoryCapabilityBlock],
    ) -> list[SensoryCapabilitiesMessage]:
        if not blocks:
            return messages
        sensory_text = self._format_sensory_blocks(blocks)
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].role == "user":
                content = messages[idx].content.rstrip()
                messages[idx].content = f"{content}\n\n{sensory_text}".strip()
                return messages
        messages.append(SensoryCapabilitiesMessage(role="user", content=sensory_text))
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
        capability_style: str,
        modalities: list[str],
        capability_goals: list[str],
        show_steps: bool,
        max_capabilities: int,
        prefix: str | None,
    ) -> str:
        show_steps_text = "yes" if show_steps else "no"
        modalities_text = ", ".join(modalities)
        goals_text = ", ".join(capability_goals) if capability_goals else "none"
        rendered = self._render_template(
            template,
            capability_style,
            modalities_text,
            goals_text,
            show_steps_text,
            max_capabilities,
        )
        if prefix and prefix.strip():
            return f"{prefix.strip()}\n\n{rendered}".strip()
        return rendered.strip()

    def _render_template(
        self,
        template: str,
        capability_style: str,
        modalities_text: str,
        goals_text: str,
        show_steps_text: str,
        max_capabilities: int,
    ) -> str:
        had_placeholders = (
            "{capability_style}" in template
            or "{modalities}" in template
            or "{capability_goals}" in template
            or "{show_steps}" in template
            or "{max_capabilities}" in template
        )
        rendered = template
        if "{capability_style}" in template:
            rendered = rendered.replace("{capability_style}", capability_style)
        if "{modalities}" in template:
            rendered = rendered.replace("{modalities}", modalities_text)
        if "{capability_goals}" in template:
            rendered = rendered.replace("{capability_goals}", goals_text)
        if "{show_steps}" in template:
            rendered = rendered.replace("{show_steps}", show_steps_text)
        if "{max_capabilities}" in template:
            rendered = rendered.replace("{max_capabilities}", str(max_capabilities))

        if had_placeholders:
            return rendered

        extra = (
            f"Capability style: {capability_style}\n"
            f"Modalities: {modalities_text}\n"
            f"Capability goals: {goals_text}\n"
            f"Show steps: {show_steps_text}\n"
            f"Max capabilities: {max_capabilities}"
        )
        return f"{rendered.rstrip()}\n\n{extra}"

    def _normalize_messages(
        self, args: SensoryCapabilitiesArgs, system_prompt: str
    ) -> list[SensoryCapabilitiesMessage]:
        messages: list[SensoryCapabilitiesMessage] = []
        if args.messages:
            for msg in args.messages:
                role = (msg.role or "").strip()
                content = (msg.content or "").strip()
                if not content:
                    continue
                if not role:
                    role = "user"
                messages.append(SensoryCapabilitiesMessage(role=role, content=content))
        elif args.prompt and args.prompt.strip():
            messages.append(
                SensoryCapabilitiesMessage(role="user", content=args.prompt.strip())
            )
        else:
            raise ToolError("Provide prompt or messages.")

        if not messages:
            raise ToolError("No usable messages provided.")

        if system_prompt.strip():
            messages.insert(
                0,
                SensoryCapabilitiesMessage(role="system", content=system_prompt.strip()),
            )
        return messages

    def _call_llm(
        self,
        messages: list[SensoryCapabilitiesMessage],
        args: SensoryCapabilitiesArgs,
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

    def _resolve_model(self, args: SensoryCapabilitiesArgs) -> str:
        return args.llm_model or self.config.llm_model

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, SensoryCapabilitiesArgs):
            return ToolCallDisplay(summary="sensory_capabilities")
        return ToolCallDisplay(
            summary="sensory_capabilities",
            details={
                "capability_style": event.args.capability_style,
                "modalities": event.args.modalities,
                "capability_goals": event.args.capability_goals,
                "show_steps": event.args.show_steps,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, SensoryCapabilitiesResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = "Sensory capabilities complete"
        if event.result.errors:
            message = "Sensory capabilities finished with errors"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "answer": event.result.answer,
                "used_capability_style": event.result.used_capability_style,
                "used_modalities": event.result.used_modalities,
                "capability_goals": event.result.capability_goals,
                "show_steps": event.result.show_steps,
                "max_capabilities": event.result.max_capabilities,
                "template_source": event.result.template_source,
                "sensory_blocks": len(event.result.sensory_blocks),
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Sensory capabilities"
