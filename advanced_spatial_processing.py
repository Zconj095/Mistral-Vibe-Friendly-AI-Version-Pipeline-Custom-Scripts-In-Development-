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


DEFAULT_PROMPT_TEMPLATE = """### ADVANCED SPATIAL PROCESSING MODE (OPT-IN)
Process advanced spatial environments before answering.
- Represent spatial structure, scale, adjacency, and navigation constraints.
- Distinguish observed layout details from inferred structure.
- Call out ambiguity, missing measurements, or conflicting spatial cues.
- Keep internal spatial reasoning private; do not mention this mode in the response.
- If the user asks for steps or show_steps is enabled, give a concise outline.

Processing style: {processing_style}
Environment focus: {environment_focus}
Show steps: {show_steps}
Max features: {max_features}
"""

TOOL_PROMPT = (
    "Use `advanced_spatial_processing` to interpret spatial environments. "
    "Provide `prompt` or `messages`, and optionally set `processing_style`, "
    "`environment_focus`, `show_steps`, and `max_features`."
)


@dataclass(frozen=True)
class _SpatialSource:
    content: str
    label: str | None
    source_type: str | None
    source_path: str | None


class SpatialProcessingMessage(BaseModel):
    role: str
    content: str


class SpatialEnvironmentItem(BaseModel):
    id: str | None = Field(default=None, description="Optional item id.")
    label: str | None = Field(default=None, description="Optional label.")
    source_type: str | None = Field(
        default=None, description="Source type (map, floorplan, scene, route, etc.)."
    )
    environment: str | None = Field(
        default=None, description="Environment description."
    )
    layout: str | None = Field(default=None, description="Layout details.")
    coordinates: str | None = Field(
        default=None, description="Coordinate system or reference frame."
    )
    constraints: str | None = Field(
        default=None, description="Navigation or physical constraints."
    )
    context: str | None = Field(default=None, description="Context summary.")
    content: str | None = Field(default=None, description="Inline content.")
    tags: list[str] | None = Field(default=None, description="Optional tags.")
    path: str | None = Field(default=None, description="Path to a context file.")


class AdvancedSpatialProcessingArgs(BaseModel):
    prompt: str | None = Field(
        default=None, description="User prompt to solve."
    )
    messages: list[SpatialProcessingMessage] | None = Field(
        default=None, description="Optional chat messages."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prefix."
    )
    processing_style: str | None = Field(
        default=None, description="Processing style label."
    )
    environment_focus: str | None = Field(
        default=None, description="Environment focus label."
    )
    show_steps: bool | None = Field(
        default=None, description="Whether to include a step outline."
    )
    max_features: int | None = Field(
        default=None, description="Maximum features in the outline."
    )
    environments: list[SpatialEnvironmentItem] | None = Field(
        default=None, description="Spatial environment items."
    )
    environment_paths: list[str] | None = Field(
        default=None, description="Additional environment context file paths."
    )
    max_environment_chars: int | None = Field(
        default=None, description="Max chars per environment item."
    )
    max_environment_total_chars: int | None = Field(
        default=None, description="Max total environment chars."
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


class SpatialBlock(BaseModel):
    label: str | None
    source_type: str | None
    source_path: str | None
    content: str
    truncated: bool


class AdvancedSpatialProcessingResult(BaseModel):
    answer: str
    system_prompt: str
    messages: list[SpatialProcessingMessage]
    used_processing_style: str
    used_environment_focus: str
    show_steps: bool
    max_features: int
    template_source: str
    spatial_blocks: list[SpatialBlock]
    warnings: list[str]
    errors: list[str]
    llm_model: str


class AdvancedSpatialProcessingConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    llm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL.",
    )
    llm_model: str = Field(
        default="gpt-oss:latest", description="Default LLM model name."
    )
    default_processing_style: str = Field(
        default="spatial-analytic",
        description="Default processing style.",
    )
    default_environment_focus: str = Field(
        default="structure-and-navigation",
        description="Default environment focus.",
    )
    default_show_steps: bool = Field(
        default=False, description="Default for show_steps."
    )
    default_max_features: int = Field(
        default=6, description="Default max features."
    )
    prompt_path: Path | None = Field(
        default=Path.home()
        / "mistral-vibe"
        / "vibe"
        / "core"
        / "prompts"
        / "advanced_spatial_processing.md",
        description="Optional path to a prompt template.",
    )
    prompt_max_chars: int = Field(
        default=8000, description="Maximum template characters to load."
    )
    max_environment_chars: int = Field(
        default=4000, description="Maximum characters per environment item."
    )
    max_environment_total_chars: int = Field(
        default=12000, description="Maximum total environment characters."
    )


class AdvancedSpatialProcessingState(BaseToolState):
    pass


class AdvancedSpatialProcessing(
    BaseTool[
        AdvancedSpatialProcessingArgs,
        AdvancedSpatialProcessingResult,
        AdvancedSpatialProcessingConfig,
        AdvancedSpatialProcessingState,
    ],
    ToolUIData[AdvancedSpatialProcessingArgs, AdvancedSpatialProcessingResult],
):
    description: ClassVar[str] = (
        "Process advanced spatial environments using a prompt template."
    )

    @classmethod
    def get_tool_prompt(cls) -> str | None:
        return TOOL_PROMPT

    async def run(
        self, args: AdvancedSpatialProcessingArgs
    ) -> AdvancedSpatialProcessingResult:
        warnings: list[str] = []
        errors: list[str] = []

        processing_style = (
            args.processing_style or self.config.default_processing_style
        ).strip()
        if not processing_style:
            processing_style = "spatial-analytic"

        environment_focus = (
            args.environment_focus or self.config.default_environment_focus
        ).strip()
        if not environment_focus:
            environment_focus = "structure-and-navigation"

        show_steps = (
            args.show_steps
            if args.show_steps is not None
            else self.config.default_show_steps
        )
        max_features = (
            args.max_features
            if args.max_features is not None
            else self.config.default_max_features
        )
        if max_features <= 0:
            raise ToolError("max_features must be positive.")

        self._validate_llm_settings(args)

        template, template_source = self._load_template(warnings)
        system_prompt = self._build_system_prompt(
            template,
            processing_style,
            environment_focus,
            show_steps,
            max_features,
            args.system_prompt,
        )

        spatial_blocks = self._collect_spatial_context(args, warnings)
        messages = self._normalize_messages(args, system_prompt)
        messages = self._inject_spatial_context(messages, spatial_blocks)
        answer = self._call_llm(messages, args)

        return AdvancedSpatialProcessingResult(
            answer=answer,
            system_prompt=system_prompt,
            messages=messages,
            used_processing_style=processing_style,
            used_environment_focus=environment_focus,
            show_steps=show_steps,
            max_features=max_features,
            template_source=template_source,
            spatial_blocks=spatial_blocks,
            warnings=warnings,
            errors=errors,
            llm_model=self._resolve_model(args),
        )

    def _validate_llm_settings(self, args: AdvancedSpatialProcessingArgs) -> None:
        if args.llm_temperature < 0:
            raise ToolError("llm_temperature cannot be negative.")
        if args.llm_max_tokens <= 0:
            raise ToolError("llm_max_tokens must be positive.")

    def _collect_spatial_context(
        self, args: AdvancedSpatialProcessingArgs, warnings: list[str]
    ) -> list[SpatialBlock]:
        max_item = (
            args.max_environment_chars
            if args.max_environment_chars is not None
            else self.config.max_environment_chars
        )
        max_total = (
            args.max_environment_total_chars
            if args.max_environment_total_chars is not None
            else self.config.max_environment_total_chars
        )
        if max_item <= 0 or max_total <= 0:
            raise ToolError(
                "max_environment_chars and max_environment_total_chars must be positive."
            )

        sources = self._resolve_spatial_sources(args, warnings)
        blocks: list[SpatialBlock] = []
        total_chars = 0
        for source in sources:
            if total_chars >= max_total:
                warnings.append(
                    "max_environment_total_chars reached; truncating environments."
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
                SpatialBlock(
                    label=source.label,
                    source_type=source.source_type,
                    source_path=source.source_path,
                    content=content,
                    truncated=truncated,
                )
            )

        if not blocks:
            warnings.append("No spatial context provided.")
        return blocks

    def _resolve_spatial_sources(
        self, args: AdvancedSpatialProcessingArgs, warnings: list[str]
    ) -> list[_SpatialSource]:
        sources: list[_SpatialSource] = []
        if args.environments:
            for item in args.environments:
                sources.append(self._load_environment_item(item))

        if args.environment_paths:
            for raw_path in args.environment_paths:
                path = self._resolve_path(raw_path)
                content = path.read_text("utf-8", errors="ignore")
                sources.append(
                    _SpatialSource(
                        content=content,
                        label=path.name,
                        source_type=None,
                        source_path=str(path),
                    )
                )

        if not sources and not warnings:
            warnings.append("No spatial context provided.")
        return sources

    def _load_environment_item(
        self, item: SpatialEnvironmentItem
    ) -> _SpatialSource:
        if item.content and item.path:
            raise ToolError("Provide content or path per environment item, not both.")
        if not (
            item.content
            or item.path
            or item.environment
            or item.layout
            or item.coordinates
            or item.constraints
            or item.context
        ):
            raise ToolError(
                "Each environment item must provide content, details, or path."
            )

        label = item.label or item.id or item.path
        text = self._format_environment_text(item)

        if item.path:
            path = self._resolve_path(item.path)
            file_content = path.read_text("utf-8", errors="ignore")
            text = self._join_sections(text, "Content", file_content)
            return _SpatialSource(
                content=text,
                label=label or path.name,
                source_type=item.source_type,
                source_path=str(path),
            )

        return _SpatialSource(
            content=text,
            label=label,
            source_type=item.source_type,
            source_path=None,
        )

    def _format_environment_text(self, item: SpatialEnvironmentItem) -> str:
        lines: list[str] = []
        if item.label or item.id:
            lines.append(f"Label: {item.label or item.id}")
        if item.source_type:
            lines.append(f"Source type: {item.source_type}")
        if item.tags:
            tags = ", ".join(tag for tag in item.tags if tag)
            if tags:
                lines.append(f"Tags: {tags}")
        if item.environment:
            lines.append("Environment:")
            lines.append(item.environment)
        if item.layout:
            lines.append("Layout:")
            lines.append(item.layout)
        if item.coordinates:
            lines.append("Coordinates:")
            lines.append(item.coordinates)
        if item.constraints:
            lines.append("Constraints:")
            lines.append(item.constraints)
        if item.context:
            lines.append("Context:")
            lines.append(item.context)
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

    def _format_spatial_blocks(self, blocks: list[SpatialBlock]) -> str:
        if not blocks:
            return ""
        parts = ["Spatial context:"]
        for block in blocks:
            label = block.label or block.source_path or "environment"
            type_suffix = f" | {block.source_type}" if block.source_type else ""
            trunc = " (truncated)" if block.truncated else ""
            header = f"[{label}{type_suffix}{trunc}]"
            parts.append(f"{header}\n{block.content}".strip())
        return "\n\n".join(parts).strip()

    def _inject_spatial_context(
        self,
        messages: list[SpatialProcessingMessage],
        blocks: list[SpatialBlock],
    ) -> list[SpatialProcessingMessage]:
        if not blocks:
            return messages
        spatial_text = self._format_spatial_blocks(blocks)
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].role == "user":
                content = messages[idx].content.rstrip()
                messages[idx].content = f"{content}\n\n{spatial_text}".strip()
                return messages
        messages.append(SpatialProcessingMessage(role="user", content=spatial_text))
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
        processing_style: str,
        environment_focus: str,
        show_steps: bool,
        max_features: int,
        prefix: str | None,
    ) -> str:
        show_steps_text = "yes" if show_steps else "no"
        rendered = self._render_template(
            template,
            processing_style,
            environment_focus,
            show_steps_text,
            max_features,
        )
        if prefix and prefix.strip():
            return f"{prefix.strip()}\n\n{rendered}".strip()
        return rendered.strip()

    def _render_template(
        self,
        template: str,
        processing_style: str,
        environment_focus: str,
        show_steps_text: str,
        max_features: int,
    ) -> str:
        had_placeholders = (
            "{processing_style}" in template
            or "{environment_focus}" in template
            or "{show_steps}" in template
            or "{max_features}" in template
        )
        rendered = template
        if "{processing_style}" in template:
            rendered = rendered.replace("{processing_style}", processing_style)
        if "{environment_focus}" in template:
            rendered = rendered.replace("{environment_focus}", environment_focus)
        if "{show_steps}" in template:
            rendered = rendered.replace("{show_steps}", show_steps_text)
        if "{max_features}" in template:
            rendered = rendered.replace("{max_features}", str(max_features))

        if had_placeholders:
            return rendered

        extra = (
            f"Processing style: {processing_style}\n"
            f"Environment focus: {environment_focus}\n"
            f"Show steps: {show_steps_text}\n"
            f"Max features: {max_features}"
        )
        return f"{rendered.rstrip()}\n\n{extra}"

    def _normalize_messages(
        self, args: AdvancedSpatialProcessingArgs, system_prompt: str
    ) -> list[SpatialProcessingMessage]:
        messages: list[SpatialProcessingMessage] = []
        if args.messages:
            for msg in args.messages:
                role = (msg.role or "").strip()
                content = (msg.content or "").strip()
                if not content:
                    continue
                if not role:
                    role = "user"
                messages.append(SpatialProcessingMessage(role=role, content=content))
        elif args.prompt and args.prompt.strip():
            messages.append(
                SpatialProcessingMessage(role="user", content=args.prompt.strip())
            )
        else:
            raise ToolError("Provide prompt or messages.")

        if not messages:
            raise ToolError("No usable messages provided.")

        if system_prompt.strip():
            messages.insert(
                0,
                SpatialProcessingMessage(role="system", content=system_prompt.strip()),
            )
        return messages

    def _call_llm(
        self,
        messages: list[SpatialProcessingMessage],
        args: AdvancedSpatialProcessingArgs,
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

    def _resolve_model(self, args: AdvancedSpatialProcessingArgs) -> str:
        return args.llm_model or self.config.llm_model

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, AdvancedSpatialProcessingArgs):
            return ToolCallDisplay(summary="advanced_spatial_processing")
        return ToolCallDisplay(
            summary="advanced_spatial_processing",
            details={
                "processing_style": event.args.processing_style,
                "environment_focus": event.args.environment_focus,
                "show_steps": event.args.show_steps,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, AdvancedSpatialProcessingResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = "Advanced spatial processing complete"
        if event.result.errors:
            message = "Advanced spatial processing finished with errors"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "answer": event.result.answer,
                "used_processing_style": event.result.used_processing_style,
                "used_environment_focus": event.result.used_environment_focus,
                "show_steps": event.result.show_steps,
                "max_features": event.result.max_features,
                "template_source": event.result.template_source,
                "spatial_blocks": len(event.result.spatial_blocks),
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Advanced spatial processing"
