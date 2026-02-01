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


DEFAULT_PROMPT_TEMPLATE = """### DEEP CONNECTION ANALYSIS MODE (OPT-IN)
Analyze information deeply and trace where it connects from before answering.
- Identify root causes, dependencies, and upstream sources.
- Call out uncertainty or missing context when relevant.
- Keep internal analysis private; do not mention this mode in the response.
- If the user asks for steps or show_steps is enabled, give a concise outline.

Focus: {focus}
Show steps: {show_steps}
Max connections: {max_connections}
"""

TOOL_PROMPT = (
    "Use `deep_connection_analysis` to trace upstream connections and dependencies. "
    "Provide `prompt` or `messages`, and optionally set `focus`, `show_steps`, "
    "and `max_connections` to control the analysis."
)


@dataclass(frozen=True)
class _ConnectionSource:
    content: str
    label: str | None
    source_type: str | None
    source_path: str | None


class DeepConnectionMessage(BaseModel):
    role: str
    content: str


class ConnectionSourceItem(BaseModel):
    id: str | None = Field(default=None, description="Optional source id.")
    label: str | None = Field(default=None, description="Optional label.")
    source_type: str | None = Field(
        default=None, description="Source type (note, log, report, etc.)."
    )
    content: str | None = Field(default=None, description="Inline source content.")
    path: str | None = Field(default=None, description="Path to a source file.")


class DeepConnectionAnalysisArgs(BaseModel):
    prompt: str | None = Field(
        default=None, description="User prompt to solve."
    )
    messages: list[DeepConnectionMessage] | None = Field(
        default=None, description="Optional chat messages."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prefix."
    )
    focus: str | None = Field(
        default=None, description="Analysis focus label."
    )
    show_steps: bool | None = Field(
        default=None, description="Whether to include a step outline."
    )
    max_connections: int | None = Field(
        default=None, description="Maximum connections in the outline."
    )
    sources: list[ConnectionSourceItem] | None = Field(
        default=None, description="Connection source items."
    )
    source_paths: list[str] | None = Field(
        default=None, description="Additional source file paths."
    )
    max_source_chars: int | None = Field(
        default=None, description="Max chars per source item."
    )
    max_source_total_chars: int | None = Field(
        default=None, description="Max total source chars."
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


class SourceBlock(BaseModel):
    label: str | None
    source_type: str | None
    source_path: str | None
    content: str
    truncated: bool


class DeepConnectionAnalysisResult(BaseModel):
    answer: str
    system_prompt: str
    messages: list[DeepConnectionMessage]
    used_focus: str
    show_steps: bool
    max_connections: int
    template_source: str
    source_blocks: list[SourceBlock]
    warnings: list[str]
    errors: list[str]
    llm_model: str


class DeepConnectionAnalysisConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    llm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL.",
    )
    llm_model: str = Field(
        default="gpt-oss:latest", description="Default LLM model name."
    )
    default_focus: str = Field(
        default="dependency-trace",
        description="Default analysis focus.",
    )
    default_show_steps: bool = Field(
        default=False, description="Default for show_steps."
    )
    default_max_connections: int = Field(
        default=6, description="Default max connections."
    )
    prompt_path: Path | None = Field(
        default=Path.home()
        / "mistral-vibe"
        / "vibe"
        / "core"
        / "prompts"
        / "deep_connection_analysis.md",
        description="Optional path to a prompt template.",
    )
    prompt_max_chars: int = Field(
        default=8000, description="Maximum template characters to load."
    )
    max_source_chars: int = Field(
        default=4000, description="Maximum characters per source item."
    )
    max_source_total_chars: int = Field(
        default=12000, description="Maximum total source characters."
    )


class DeepConnectionAnalysisState(BaseToolState):
    pass


class DeepConnectionAnalysis(
    BaseTool[
        DeepConnectionAnalysisArgs,
        DeepConnectionAnalysisResult,
        DeepConnectionAnalysisConfig,
        DeepConnectionAnalysisState,
    ],
    ToolUIData[DeepConnectionAnalysisArgs, DeepConnectionAnalysisResult],
):
    description: ClassVar[str] = (
        "Analyze deep connections and upstream origins using a prompt template."
    )

    @classmethod
    def get_tool_prompt(cls) -> str | None:
        return TOOL_PROMPT

    async def run(
        self, args: DeepConnectionAnalysisArgs
    ) -> DeepConnectionAnalysisResult:
        warnings: list[str] = []
        errors: list[str] = []

        focus = (args.focus or self.config.default_focus).strip()
        if not focus:
            focus = "dependency-trace"

        show_steps = (
            args.show_steps
            if args.show_steps is not None
            else self.config.default_show_steps
        )
        max_connections = (
            args.max_connections
            if args.max_connections is not None
            else self.config.default_max_connections
        )
        if max_connections <= 0:
            raise ToolError("max_connections must be positive.")

        self._validate_llm_settings(args)

        template, template_source = self._load_template(warnings)
        system_prompt = self._build_system_prompt(
            template,
            focus,
            show_steps,
            max_connections,
            args.system_prompt,
        )

        source_blocks = self._collect_sources(args, warnings)
        messages = self._normalize_messages(args, system_prompt)
        messages = self._inject_sources(messages, source_blocks)
        answer = self._call_llm(messages, args)

        return DeepConnectionAnalysisResult(
            answer=answer,
            system_prompt=system_prompt,
            messages=messages,
            used_focus=focus,
            show_steps=show_steps,
            max_connections=max_connections,
            template_source=template_source,
            source_blocks=source_blocks,
            warnings=warnings,
            errors=errors,
            llm_model=self._resolve_model(args),
        )

    def _validate_llm_settings(self, args: DeepConnectionAnalysisArgs) -> None:
        if args.llm_temperature < 0:
            raise ToolError("llm_temperature cannot be negative.")
        if args.llm_max_tokens <= 0:
            raise ToolError("llm_max_tokens must be positive.")

    def _collect_sources(
        self, args: DeepConnectionAnalysisArgs, warnings: list[str]
    ) -> list[SourceBlock]:
        max_item = (
            args.max_source_chars
            if args.max_source_chars is not None
            else self.config.max_source_chars
        )
        max_total = (
            args.max_source_total_chars
            if args.max_source_total_chars is not None
            else self.config.max_source_total_chars
        )
        if max_item <= 0 or max_total <= 0:
            raise ToolError(
                "max_source_chars and max_source_total_chars must be positive."
            )

        sources = self._resolve_sources(args, warnings)
        blocks: list[SourceBlock] = []
        total_chars = 0

        for source in sources:
            if total_chars >= max_total:
                warnings.append(
                    "max_source_total_chars reached; truncating sources."
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
                SourceBlock(
                    label=source.label,
                    source_type=source.source_type,
                    source_path=source.source_path,
                    content=content,
                    truncated=truncated,
                )
            )

        if not blocks:
            warnings.append("No source context provided.")
        return blocks

    def _resolve_sources(
        self, args: DeepConnectionAnalysisArgs, warnings: list[str]
    ) -> list[_ConnectionSource]:
        sources: list[_ConnectionSource] = []
        if args.sources:
            for item in args.sources:
                sources.append(self._load_source_item(item))

        if args.source_paths:
            for raw_path in args.source_paths:
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
            warnings.append("No source context provided.")
        return sources

    def _load_source_item(self, item: ConnectionSourceItem) -> _ConnectionSource:
        if item.content and item.path:
            raise ToolError("Provide content or path per source item, not both.")
        if not item.content and not item.path:
            raise ToolError("Each source item must provide content or path.")

        label = item.label or item.id or item.path
        if item.content is not None:
            return _ConnectionSource(
                content=item.content,
                label=label,
                source_type=item.source_type,
                source_path=None,
            )

        path = self._resolve_path(item.path or "")
        content = path.read_text("utf-8", errors="ignore")
        return _ConnectionSource(
            content=content,
            label=label or path.name,
            source_type=item.source_type,
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

    def _format_source_blocks(self, blocks: list[SourceBlock]) -> str:
        if not blocks:
            return ""
        parts = ["Connection context:"]
        for block in blocks:
            label = block.label or block.source_path or "source"
            type_suffix = f" | {block.source_type}" if block.source_type else ""
            trunc = " (truncated)" if block.truncated else ""
            header = f"[{label}{type_suffix}{trunc}]"
            parts.append(f"{header}\n{block.content}".strip())
        return "\n\n".join(parts).strip()

    def _inject_sources(
        self,
        messages: list[DeepConnectionMessage],
        blocks: list[SourceBlock],
    ) -> list[DeepConnectionMessage]:
        if not blocks:
            return messages
        source_text = self._format_source_blocks(blocks)
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].role == "user":
                content = messages[idx].content.rstrip()
                messages[idx].content = f"{content}\n\n{source_text}".strip()
                return messages
        messages.append(DeepConnectionMessage(role="user", content=source_text))
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
        focus: str,
        show_steps: bool,
        max_connections: int,
        prefix: str | None,
    ) -> str:
        show_steps_text = "yes" if show_steps else "no"
        rendered = self._render_template(
            template,
            focus,
            show_steps_text,
            max_connections,
        )
        if prefix and prefix.strip():
            return f"{prefix.strip()}\n\n{rendered}".strip()
        return rendered.strip()

    def _render_template(
        self,
        template: str,
        focus: str,
        show_steps_text: str,
        max_connections: int,
    ) -> str:
        had_placeholders = (
            "{focus}" in template
            or "{show_steps}" in template
            or "{max_connections}" in template
        )
        rendered = template
        if "{focus}" in template:
            rendered = rendered.replace("{focus}", focus)
        if "{show_steps}" in template:
            rendered = rendered.replace("{show_steps}", show_steps_text)
        if "{max_connections}" in template:
            rendered = rendered.replace("{max_connections}", str(max_connections))

        if had_placeholders:
            return rendered

        extra = (
            f"Focus: {focus}\n"
            f"Show steps: {show_steps_text}\n"
            f"Max connections: {max_connections}"
        )
        return f"{rendered.rstrip()}\n\n{extra}"

    def _normalize_messages(
        self, args: DeepConnectionAnalysisArgs, system_prompt: str
    ) -> list[DeepConnectionMessage]:
        messages: list[DeepConnectionMessage] = []
        if args.messages:
            for msg in args.messages:
                role = (msg.role or "").strip()
                content = (msg.content or "").strip()
                if not content:
                    continue
                if not role:
                    role = "user"
                messages.append(DeepConnectionMessage(role=role, content=content))
        elif args.prompt and args.prompt.strip():
            messages.append(
                DeepConnectionMessage(role="user", content=args.prompt.strip())
            )
        else:
            raise ToolError("Provide prompt or messages.")

        if not messages:
            raise ToolError("No usable messages provided.")

        if system_prompt.strip():
            messages.insert(
                0,
                DeepConnectionMessage(
                    role="system", content=system_prompt.strip()
                ),
            )
        return messages

    def _call_llm(
        self,
        messages: list[DeepConnectionMessage],
        args: DeepConnectionAnalysisArgs,
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

    def _resolve_model(self, args: DeepConnectionAnalysisArgs) -> str:
        return args.llm_model or self.config.llm_model

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, DeepConnectionAnalysisArgs):
            return ToolCallDisplay(summary="deep_connection_analysis")
        return ToolCallDisplay(
            summary="deep_connection_analysis",
            details={
                "focus": event.args.focus,
                "show_steps": event.args.show_steps,
                "max_connections": event.args.max_connections,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, DeepConnectionAnalysisResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = "Deep connection analysis complete"
        if event.result.errors:
            message = "Deep connection analysis finished with errors"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "answer": event.result.answer,
                "used_focus": event.result.used_focus,
                "show_steps": event.result.show_steps,
                "max_connections": event.result.max_connections,
                "template_source": event.result.template_source,
                "source_blocks": len(event.result.source_blocks),
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Deep connection analysis"
