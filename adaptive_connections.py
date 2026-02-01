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


DEFAULT_PROMPT_TEMPLATE = """### ADAPTIVE CONNECTIONS MODE (OPT-IN)
Adapt to new information and form new connections before answering.
- Update prior links when new signals appear.
- Propose novel connections while avoiding spurious links.
- Separate observed signals from inferred links and call out uncertainty.
- Keep internal adaptation private; do not mention this mode in the response.
- If the user asks for steps or show_steps is enabled, give a concise outline.

Adaptation style: {adaptation_style}
Connection style: {connection_style}
Adaptation focus: {adaptation_focus}
Show steps: {show_steps}
Max links: {max_links}
"""

TOOL_PROMPT = (
    "Use `adaptive_connections` to adapt and form new connections. "
    "Provide `prompt` or `messages`, and optionally set `adaptation_style`, "
    "`connection_style`, `adaptation_focus`, `show_steps`, and `max_links`."
)


@dataclass(frozen=True)
class _ConnectionSource:
    content: str
    label: str | None
    source_type: str | None
    source_path: str | None


class AdaptiveConnectionsMessage(BaseModel):
    role: str
    content: str


class ConnectionInputItem(BaseModel):
    id: str | None = Field(default=None, description="Optional input id.")
    label: str | None = Field(default=None, description="Optional label.")
    source_type: str | None = Field(
        default=None, description="Source type (memory, observation, feedback, etc.)."
    )
    content: str | None = Field(default=None, description="Inline connection content.")
    path: str | None = Field(default=None, description="Path to a context file.")
    tags: list[str] | None = Field(default=None, description="Optional tags.")


class AdaptiveConnectionsArgs(BaseModel):
    prompt: str | None = Field(
        default=None, description="User prompt to solve."
    )
    messages: list[AdaptiveConnectionsMessage] | None = Field(
        default=None, description="Optional chat messages."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prefix."
    )
    adaptation_style: str | None = Field(
        default=None, description="Adaptation style label."
    )
    connection_style: str | None = Field(
        default=None, description="Connection style label."
    )
    adaptation_focus: str | None = Field(
        default=None, description="Adaptation focus label."
    )
    show_steps: bool | None = Field(
        default=None, description="Whether to include a step outline."
    )
    max_links: int | None = Field(
        default=None, description="Maximum links in the outline."
    )
    connection_inputs: list[ConnectionInputItem] | None = Field(
        default=None, description="Connection context items."
    )
    connection_paths: list[str] | None = Field(
        default=None, description="Additional connection context file paths."
    )
    max_connection_chars: int | None = Field(
        default=None, description="Max chars per connection item."
    )
    max_connection_total_chars: int | None = Field(
        default=None, description="Max total connection chars."
    )
    llm_api_base: str | None = Field(
        default=None, description="OpenAI-compatible API base URL."
    )
    llm_model: str | None = Field(
        default=None, description="LLM model name."
    )
    llm_temperature: float = Field(
        default=0.3, description="LLM temperature."
    )
    llm_max_tokens: int = Field(
        default=700, description="LLM max tokens."
    )
    llm_stream: bool = Field(
        default=False, description="Stream LLM tokens."
    )


class ConnectionBlock(BaseModel):
    label: str | None
    source_type: str | None
    source_path: str | None
    content: str
    truncated: bool


class AdaptiveConnectionsResult(BaseModel):
    answer: str
    system_prompt: str
    messages: list[AdaptiveConnectionsMessage]
    used_adaptation_style: str
    used_connection_style: str
    used_adaptation_focus: str
    show_steps: bool
    max_links: int
    template_source: str
    connection_blocks: list[ConnectionBlock]
    warnings: list[str]
    errors: list[str]
    llm_model: str


class AdaptiveConnectionsConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    llm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL.",
    )
    llm_model: str = Field(
        default="gpt-oss:latest", description="Default LLM model name."
    )
    default_adaptation_style: str = Field(
        default="responsive",
        description="Default adaptation style.",
    )
    default_connection_style: str = Field(
        default="relational",
        description="Default connection style.",
    )
    default_adaptation_focus: str = Field(
        default="novelty-aware",
        description="Default adaptation focus.",
    )
    default_show_steps: bool = Field(
        default=False, description="Default for show_steps."
    )
    default_max_links: int = Field(
        default=6, description="Default max links."
    )
    prompt_path: Path | None = Field(
        default=Path.home()
        / "mistral-vibe"
        / "vibe"
        / "core"
        / "prompts"
        / "adaptive_connections.md",
        description="Optional path to a prompt template.",
    )
    prompt_max_chars: int = Field(
        default=8000, description="Maximum template characters to load."
    )
    max_connection_chars: int = Field(
        default=4000, description="Maximum characters per connection item."
    )
    max_connection_total_chars: int = Field(
        default=12000, description="Maximum total connection characters."
    )


class AdaptiveConnectionsState(BaseToolState):
    pass


class AdaptiveConnections(
    BaseTool[
        AdaptiveConnectionsArgs,
        AdaptiveConnectionsResult,
        AdaptiveConnectionsConfig,
        AdaptiveConnectionsState,
    ],
    ToolUIData[AdaptiveConnectionsArgs, AdaptiveConnectionsResult],
):
    description: ClassVar[str] = (
        "Adapt and form new connections using a dedicated prompt template."
    )

    @classmethod
    def get_tool_prompt(cls) -> str | None:
        return TOOL_PROMPT

    async def run(
        self, args: AdaptiveConnectionsArgs
    ) -> AdaptiveConnectionsResult:
        warnings: list[str] = []
        errors: list[str] = []

        adaptation_style = (
            args.adaptation_style or self.config.default_adaptation_style
        ).strip()
        if not adaptation_style:
            adaptation_style = "responsive"

        connection_style = (
            args.connection_style or self.config.default_connection_style
        ).strip()
        if not connection_style:
            connection_style = "relational"

        adaptation_focus = (
            args.adaptation_focus or self.config.default_adaptation_focus
        ).strip()
        if not adaptation_focus:
            adaptation_focus = "novelty-aware"

        show_steps = (
            args.show_steps
            if args.show_steps is not None
            else self.config.default_show_steps
        )
        max_links = (
            args.max_links
            if args.max_links is not None
            else self.config.default_max_links
        )
        if max_links <= 0:
            raise ToolError("max_links must be positive.")

        self._validate_llm_settings(args)

        template, template_source = self._load_template(warnings)
        system_prompt = self._build_system_prompt(
            template,
            adaptation_style,
            connection_style,
            adaptation_focus,
            show_steps,
            max_links,
            args.system_prompt,
        )

        connection_blocks = self._collect_connections(args, warnings)
        messages = self._normalize_messages(args, system_prompt)
        messages = self._inject_connections(messages, connection_blocks)
        answer = self._call_llm(messages, args)

        return AdaptiveConnectionsResult(
            answer=answer,
            system_prompt=system_prompt,
            messages=messages,
            used_adaptation_style=adaptation_style,
            used_connection_style=connection_style,
            used_adaptation_focus=adaptation_focus,
            show_steps=show_steps,
            max_links=max_links,
            template_source=template_source,
            connection_blocks=connection_blocks,
            warnings=warnings,
            errors=errors,
            llm_model=self._resolve_model(args),
        )

    def _validate_llm_settings(self, args: AdaptiveConnectionsArgs) -> None:
        if args.llm_temperature < 0:
            raise ToolError("llm_temperature cannot be negative.")
        if args.llm_max_tokens <= 0:
            raise ToolError("llm_max_tokens must be positive.")

    def _collect_connections(
        self, args: AdaptiveConnectionsArgs, warnings: list[str]
    ) -> list[ConnectionBlock]:
        max_item = (
            args.max_connection_chars
            if args.max_connection_chars is not None
            else self.config.max_connection_chars
        )
        max_total = (
            args.max_connection_total_chars
            if args.max_connection_total_chars is not None
            else self.config.max_connection_total_chars
        )
        if max_item <= 0 or max_total <= 0:
            raise ToolError(
                "max_connection_chars and max_connection_total_chars must be positive."
            )

        sources = self._resolve_connection_sources(args, warnings)
        blocks: list[ConnectionBlock] = []
        total_chars = 0
        for source in sources:
            if total_chars >= max_total:
                warnings.append(
                    "max_connection_total_chars reached; truncating connections."
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
                ConnectionBlock(
                    label=source.label,
                    source_type=source.source_type,
                    source_path=source.source_path,
                    content=content,
                    truncated=truncated,
                )
            )

        if not blocks:
            warnings.append("No connection context provided.")
        return blocks

    def _resolve_connection_sources(
        self, args: AdaptiveConnectionsArgs, warnings: list[str]
    ) -> list[_ConnectionSource]:
        sources: list[_ConnectionSource] = []
        if args.connection_inputs:
            for item in args.connection_inputs:
                sources.append(self._load_connection_item(item))

        if args.connection_paths:
            for raw_path in args.connection_paths:
                path = self._resolve_path(raw_path)
                content = path.read_text("utf-8", errors="ignore")
                sources.append(
                    _ConnectionSource(
                        content=content,
                        label=path.name,
                        source_type=None,
                        source_path=str(path),
                    )
                )

        if not sources and not warnings:
            warnings.append("No connection context provided.")
        return sources

    def _load_connection_item(self, item: ConnectionInputItem) -> _ConnectionSource:
        if item.content and item.path:
            raise ToolError("Provide content or path per connection item, not both.")
        if not item.content and not item.path:
            raise ToolError("Each connection item must provide content or path.")

        label = item.label or item.id or item.path
        content = self._format_connection_text(item)

        if item.path:
            path = self._resolve_path(item.path)
            file_content = path.read_text("utf-8", errors="ignore")
            combined = self._join_sections(content, "Content", file_content)
            return _ConnectionSource(
                content=combined,
                label=label or path.name,
                source_type=item.source_type,
                source_path=str(path),
            )

        return _ConnectionSource(
            content=content,
            label=label,
            source_type=item.source_type,
            source_path=None,
        )

    def _format_connection_text(self, item: ConnectionInputItem) -> str:
        lines: list[str] = []
        if item.label or item.id:
            lines.append(f"Label: {item.label or item.id}")
        if item.source_type:
            lines.append(f"Source type: {item.source_type}")
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

    def _format_connection_blocks(self, blocks: list[ConnectionBlock]) -> str:
        if not blocks:
            return ""
        parts = ["Connection context:"]
        for block in blocks:
            label = block.label or block.source_path or "connection"
            type_suffix = f" | {block.source_type}" if block.source_type else ""
            trunc = " (truncated)" if block.truncated else ""
            header = f"[{label}{type_suffix}{trunc}]"
            parts.append(f"{header}\n{block.content}".strip())
        return "\n\n".join(parts).strip()

    def _inject_connections(
        self,
        messages: list[AdaptiveConnectionsMessage],
        blocks: list[ConnectionBlock],
    ) -> list[AdaptiveConnectionsMessage]:
        if not blocks:
            return messages
        connection_text = self._format_connection_blocks(blocks)
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].role == "user":
                content = messages[idx].content.rstrip()
                messages[idx].content = f"{content}\n\n{connection_text}".strip()
                return messages
        messages.append(
            AdaptiveConnectionsMessage(role="user", content=connection_text)
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
        adaptation_style: str,
        connection_style: str,
        adaptation_focus: str,
        show_steps: bool,
        max_links: int,
        prefix: str | None,
    ) -> str:
        show_steps_text = "yes" if show_steps else "no"
        rendered = self._render_template(
            template,
            adaptation_style,
            connection_style,
            adaptation_focus,
            show_steps_text,
            max_links,
        )
        if prefix and prefix.strip():
            return f"{prefix.strip()}\n\n{rendered}".strip()
        return rendered.strip()

    def _render_template(
        self,
        template: str,
        adaptation_style: str,
        connection_style: str,
        adaptation_focus: str,
        show_steps_text: str,
        max_links: int,
    ) -> str:
        had_placeholders = (
            "{adaptation_style}" in template
            or "{connection_style}" in template
            or "{adaptation_focus}" in template
            or "{show_steps}" in template
            or "{max_links}" in template
        )
        rendered = template
        if "{adaptation_style}" in template:
            rendered = rendered.replace("{adaptation_style}", adaptation_style)
        if "{connection_style}" in template:
            rendered = rendered.replace("{connection_style}", connection_style)
        if "{adaptation_focus}" in template:
            rendered = rendered.replace("{adaptation_focus}", adaptation_focus)
        if "{show_steps}" in template:
            rendered = rendered.replace("{show_steps}", show_steps_text)
        if "{max_links}" in template:
            rendered = rendered.replace("{max_links}", str(max_links))

        if had_placeholders:
            return rendered

        extra = (
            f"Adaptation style: {adaptation_style}\n"
            f"Connection style: {connection_style}\n"
            f"Adaptation focus: {adaptation_focus}\n"
            f"Show steps: {show_steps_text}\n"
            f"Max links: {max_links}"
        )
        return f"{rendered.rstrip()}\n\n{extra}"

    def _normalize_messages(
        self, args: AdaptiveConnectionsArgs, system_prompt: str
    ) -> list[AdaptiveConnectionsMessage]:
        messages: list[AdaptiveConnectionsMessage] = []
        if args.messages:
            for msg in args.messages:
                role = (msg.role or "").strip()
                content = (msg.content or "").strip()
                if not content:
                    continue
                if not role:
                    role = "user"
                messages.append(
                    AdaptiveConnectionsMessage(role=role, content=content)
                )
        elif args.prompt and args.prompt.strip():
            messages.append(
                AdaptiveConnectionsMessage(
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
                AdaptiveConnectionsMessage(
                    role="system", content=system_prompt.strip()
                ),
            )
        return messages

    def _call_llm(
        self,
        messages: list[AdaptiveConnectionsMessage],
        args: AdaptiveConnectionsArgs,
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

    def _resolve_model(self, args: AdaptiveConnectionsArgs) -> str:
        return args.llm_model or self.config.llm_model

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, AdaptiveConnectionsArgs):
            return ToolCallDisplay(summary="adaptive_connections")
        return ToolCallDisplay(
            summary="adaptive_connections",
            details={
                "adaptation_style": event.args.adaptation_style,
                "connection_style": event.args.connection_style,
                "adaptation_focus": event.args.adaptation_focus,
                "show_steps": event.args.show_steps,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, AdaptiveConnectionsResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = "Adaptive connections complete"
        if event.result.errors:
            message = "Adaptive connections finished with errors"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "answer": event.result.answer,
                "used_adaptation_style": event.result.used_adaptation_style,
                "used_connection_style": event.result.used_connection_style,
                "used_adaptation_focus": event.result.used_adaptation_focus,
                "show_steps": event.result.show_steps,
                "max_links": event.result.max_links,
                "template_source": event.result.template_source,
                "connection_blocks": len(event.result.connection_blocks),
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Adaptive connections"
