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


DEFAULT_PROMPT_TEMPLATE = """### INNATE WISDOM MODE (OPT-IN)
Develop wisdom across multiple fields or domains before answering.
- Synthesize cross-domain principles, tradeoffs, and long-term implications.
- Separate evidence from inference; call out uncertainty.
- Emphasize practical guidance with balanced judgment.
- Keep internal wisdom framing private; do not mention this mode in the response.
- If the user asks for steps or show_steps is enabled, give a concise outline.

Wisdom style: {wisdom_style}
Domains: {domains}
Wisdom focus: {wisdom_focus}
Show steps: {show_steps}
Max insights: {max_insights}
"""

TOOL_PROMPT = (
    "Use `innate_wisdom` to synthesize insights across domains. "
    "Provide `prompt` or `messages`, and optionally set `wisdom_style`, "
    "`domains`, `wisdom_focus`, `show_steps`, and `max_insights`."
)


@dataclass(frozen=True)
class _WisdomSource:
    content: str
    label: str | None
    domain: str | None
    source_path: str | None


class InnateWisdomMessage(BaseModel):
    role: str
    content: str


class WisdomInputItem(BaseModel):
    id: str | None = Field(default=None, description="Optional input id.")
    label: str | None = Field(default=None, description="Optional label.")
    domain: str | None = Field(default=None, description="Domain label.")
    content: str | None = Field(default=None, description="Inline wisdom content.")
    path: str | None = Field(default=None, description="Path to a context file.")
    tags: list[str] | None = Field(default=None, description="Optional tags.")


class InnateWisdomArgs(BaseModel):
    prompt: str | None = Field(
        default=None, description="User prompt to solve."
    )
    messages: list[InnateWisdomMessage] | None = Field(
        default=None, description="Optional chat messages."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prefix."
    )
    wisdom_style: str | None = Field(
        default=None, description="Wisdom style label."
    )
    domains: list[str] | None = Field(
        default=None, description="Domains to include."
    )
    wisdom_focus: str | None = Field(
        default=None, description="Wisdom focus label."
    )
    show_steps: bool | None = Field(
        default=None, description="Whether to include a step outline."
    )
    max_insights: int | None = Field(
        default=None, description="Maximum insights in the outline."
    )
    wisdom_inputs: list[WisdomInputItem] | None = Field(
        default=None, description="Wisdom context items."
    )
    wisdom_paths: list[str] | None = Field(
        default=None, description="Additional wisdom context file paths."
    )
    max_wisdom_chars: int | None = Field(
        default=None, description="Max chars per wisdom item."
    )
    max_wisdom_total_chars: int | None = Field(
        default=None, description="Max total wisdom chars."
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


class WisdomBlock(BaseModel):
    label: str | None
    domain: str | None
    source_path: str | None
    content: str
    truncated: bool


class InnateWisdomResult(BaseModel):
    answer: str
    system_prompt: str
    messages: list[InnateWisdomMessage]
    used_wisdom_style: str
    used_domains: list[str]
    used_wisdom_focus: str
    show_steps: bool
    max_insights: int
    template_source: str
    wisdom_blocks: list[WisdomBlock]
    warnings: list[str]
    errors: list[str]
    llm_model: str


class InnateWisdomConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    llm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL.",
    )
    llm_model: str = Field(
        default="gpt-oss:latest", description="Default LLM model name."
    )
    default_wisdom_style: str = Field(
        default="integrative",
        description="Default wisdom style.",
    )
    default_domains: list[str] = Field(
        default_factory=lambda: [
            "science",
            "engineering",
            "technology",
            "business",
            "health",
            "psychology",
            "philosophy",
            "history",
            "education",
            "art",
        ],
        description="Default domains to include.",
    )
    default_wisdom_focus: str = Field(
        default="cross-domain",
        description="Default wisdom focus.",
    )
    default_show_steps: bool = Field(
        default=False, description="Default for show_steps."
    )
    default_max_insights: int = Field(
        default=6, description="Default max insights."
    )
    prompt_path: Path | None = Field(
        default=Path.home()
        / "mistral-vibe"
        / "vibe"
        / "core"
        / "prompts"
        / "innate_wisdom.md",
        description="Optional path to a prompt template.",
    )
    prompt_max_chars: int = Field(
        default=8000, description="Maximum template characters to load."
    )
    max_wisdom_chars: int = Field(
        default=4000, description="Maximum characters per wisdom item."
    )
    max_wisdom_total_chars: int = Field(
        default=12000, description="Maximum total wisdom characters."
    )


class InnateWisdomState(BaseToolState):
    pass


class InnateWisdom(
    BaseTool[
        InnateWisdomArgs,
        InnateWisdomResult,
        InnateWisdomConfig,
        InnateWisdomState,
    ],
    ToolUIData[InnateWisdomArgs, InnateWisdomResult],
):
    description: ClassVar[str] = (
        "Synthesize cross-domain wisdom using a dedicated prompt template."
    )

    @classmethod
    def get_tool_prompt(cls) -> str | None:
        return TOOL_PROMPT

    async def run(
        self, args: InnateWisdomArgs
    ) -> InnateWisdomResult:
        warnings: list[str] = []
        errors: list[str] = []

        wisdom_style = (
            args.wisdom_style or self.config.default_wisdom_style
        ).strip()
        if not wisdom_style:
            wisdom_style = "integrative"

        domains = self._normalize_domains(args.domains)

        wisdom_focus = (
            args.wisdom_focus or self.config.default_wisdom_focus
        ).strip()
        if not wisdom_focus:
            wisdom_focus = "cross-domain"

        show_steps = (
            args.show_steps
            if args.show_steps is not None
            else self.config.default_show_steps
        )
        max_insights = (
            args.max_insights
            if args.max_insights is not None
            else self.config.default_max_insights
        )
        if max_insights <= 0:
            raise ToolError("max_insights must be positive.")

        self._validate_llm_settings(args)

        template, template_source = self._load_template(warnings)
        system_prompt = self._build_system_prompt(
            template,
            wisdom_style,
            domains,
            wisdom_focus,
            show_steps,
            max_insights,
            args.system_prompt,
        )

        wisdom_blocks = self._collect_wisdom_inputs(args, warnings)
        messages = self._normalize_messages(args, system_prompt)
        messages = self._inject_wisdom_context(messages, wisdom_blocks)
        answer = self._call_llm(messages, args)

        return InnateWisdomResult(
            answer=answer,
            system_prompt=system_prompt,
            messages=messages,
            used_wisdom_style=wisdom_style,
            used_domains=domains,
            used_wisdom_focus=wisdom_focus,
            show_steps=show_steps,
            max_insights=max_insights,
            template_source=template_source,
            wisdom_blocks=wisdom_blocks,
            warnings=warnings,
            errors=errors,
            llm_model=self._resolve_model(args),
        )

    def _normalize_domains(self, domains: list[str] | None) -> list[str]:
        values = domains or list(self.config.default_domains)
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            if not value or not value.strip():
                continue
            clean = value.strip()
            key = clean.lower()
            if key not in seen:
                normalized.append(clean)
                seen.add(key)
        if not normalized:
            raise ToolError("domains must include at least one domain.")
        return normalized

    def _validate_llm_settings(self, args: InnateWisdomArgs) -> None:
        if args.llm_temperature < 0:
            raise ToolError("llm_temperature cannot be negative.")
        if args.llm_max_tokens <= 0:
            raise ToolError("llm_max_tokens must be positive.")

    def _collect_wisdom_inputs(
        self, args: InnateWisdomArgs, warnings: list[str]
    ) -> list[WisdomBlock]:
        max_item = (
            args.max_wisdom_chars
            if args.max_wisdom_chars is not None
            else self.config.max_wisdom_chars
        )
        max_total = (
            args.max_wisdom_total_chars
            if args.max_wisdom_total_chars is not None
            else self.config.max_wisdom_total_chars
        )
        if max_item <= 0 or max_total <= 0:
            raise ToolError(
                "max_wisdom_chars and max_wisdom_total_chars must be positive."
            )

        sources = self._resolve_wisdom_sources(args, warnings)
        blocks: list[WisdomBlock] = []
        total_chars = 0

        for source in sources:
            if total_chars >= max_total:
                warnings.append(
                    "max_wisdom_total_chars reached; truncating wisdom inputs."
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
                WisdomBlock(
                    label=source.label,
                    domain=source.domain,
                    source_path=source.source_path,
                    content=content,
                    truncated=truncated,
                )
            )

        if not blocks:
            warnings.append("No wisdom context provided.")
        return blocks

    def _resolve_wisdom_sources(
        self, args: InnateWisdomArgs, warnings: list[str]
    ) -> list[_WisdomSource]:
        sources: list[_WisdomSource] = []
        if args.wisdom_inputs:
            for item in args.wisdom_inputs:
                sources.append(self._load_wisdom_item(item))

        if args.wisdom_paths:
            for raw_path in args.wisdom_paths:
                path = self._resolve_path(raw_path)
                content = path.read_text("utf-8", errors="ignore")
                sources.append(
                    _WisdomSource(
                        content=content,
                        label=path.name,
                        domain=None,
                        source_path=str(path),
                    )
                )

        if not sources and not warnings:
            warnings.append("No wisdom context provided.")
        return sources

    def _load_wisdom_item(self, item: WisdomInputItem) -> _WisdomSource:
        if item.content and item.path:
            raise ToolError("Provide content or path per wisdom item, not both.")
        if not item.content and not item.path:
            raise ToolError("Each wisdom item must provide content or path.")

        label = item.label or item.id or item.path
        content = self._format_wisdom_text(item)

        if item.path:
            path = self._resolve_path(item.path)
            file_content = path.read_text("utf-8", errors="ignore")
            combined = self._join_sections(content, "Content", file_content)
            return _WisdomSource(
                content=combined,
                label=label or path.name,
                domain=item.domain,
                source_path=str(path),
            )

        return _WisdomSource(
            content=content,
            label=label,
            domain=item.domain,
            source_path=None,
        )

    def _format_wisdom_text(self, item: WisdomInputItem) -> str:
        lines: list[str] = []
        if item.label or item.id:
            lines.append(f"Label: {item.label or item.id}")
        if item.domain:
            lines.append(f"Domain: {item.domain}")
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

    def _format_wisdom_blocks(self, blocks: list[WisdomBlock]) -> str:
        if not blocks:
            return ""
        parts = ["Wisdom context:"]
        for block in blocks:
            label = block.label or block.source_path or "wisdom"
            domain_suffix = f" | {block.domain}" if block.domain else ""
            trunc = " (truncated)" if block.truncated else ""
            header = f"[{label}{domain_suffix}{trunc}]"
            parts.append(f"{header}\n{block.content}".strip())
        return "\n\n".join(parts).strip()

    def _inject_wisdom_context(
        self,
        messages: list[InnateWisdomMessage],
        blocks: list[WisdomBlock],
    ) -> list[InnateWisdomMessage]:
        if not blocks:
            return messages
        wisdom_text = self._format_wisdom_blocks(blocks)
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].role == "user":
                content = messages[idx].content.rstrip()
                messages[idx].content = f"{content}\n\n{wisdom_text}".strip()
                return messages
        messages.append(InnateWisdomMessage(role="user", content=wisdom_text))
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
        wisdom_style: str,
        domains: list[str],
        wisdom_focus: str,
        show_steps: bool,
        max_insights: int,
        prefix: str | None,
    ) -> str:
        show_steps_text = "yes" if show_steps else "no"
        domains_text = ", ".join(domains)
        rendered = self._render_template(
            template,
            wisdom_style,
            domains_text,
            wisdom_focus,
            show_steps_text,
            max_insights,
        )
        if prefix and prefix.strip():
            return f"{prefix.strip()}\n\n{rendered}".strip()
        return rendered.strip()

    def _render_template(
        self,
        template: str,
        wisdom_style: str,
        domains_text: str,
        wisdom_focus: str,
        show_steps_text: str,
        max_insights: int,
    ) -> str:
        had_placeholders = (
            "{wisdom_style}" in template
            or "{domains}" in template
            or "{wisdom_focus}" in template
            or "{show_steps}" in template
            or "{max_insights}" in template
        )
        rendered = template
        if "{wisdom_style}" in template:
            rendered = rendered.replace("{wisdom_style}", wisdom_style)
        if "{domains}" in template:
            rendered = rendered.replace("{domains}", domains_text)
        if "{wisdom_focus}" in template:
            rendered = rendered.replace("{wisdom_focus}", wisdom_focus)
        if "{show_steps}" in template:
            rendered = rendered.replace("{show_steps}", show_steps_text)
        if "{max_insights}" in template:
            rendered = rendered.replace("{max_insights}", str(max_insights))

        if had_placeholders:
            return rendered

        extra = (
            f"Wisdom style: {wisdom_style}\n"
            f"Domains: {domains_text}\n"
            f"Wisdom focus: {wisdom_focus}\n"
            f"Show steps: {show_steps_text}\n"
            f"Max insights: {max_insights}"
        )
        return f"{rendered.rstrip()}\n\n{extra}"

    def _normalize_messages(
        self, args: InnateWisdomArgs, system_prompt: str
    ) -> list[InnateWisdomMessage]:
        messages: list[InnateWisdomMessage] = []
        if args.messages:
            for msg in args.messages:
                role = (msg.role or "").strip()
                content = (msg.content or "").strip()
                if not content:
                    continue
                if not role:
                    role = "user"
                messages.append(InnateWisdomMessage(role=role, content=content))
        elif args.prompt and args.prompt.strip():
            messages.append(
                InnateWisdomMessage(role="user", content=args.prompt.strip())
            )
        else:
            raise ToolError("Provide prompt or messages.")

        if not messages:
            raise ToolError("No usable messages provided.")

        if system_prompt.strip():
            messages.insert(
                0,
                InnateWisdomMessage(role="system", content=system_prompt.strip()),
            )
        return messages

    def _call_llm(
        self,
        messages: list[InnateWisdomMessage],
        args: InnateWisdomArgs,
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

    def _resolve_model(self, args: InnateWisdomArgs) -> str:
        return args.llm_model or self.config.llm_model

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, InnateWisdomArgs):
            return ToolCallDisplay(summary="innate_wisdom")
        return ToolCallDisplay(
            summary="innate_wisdom",
            details={
                "wisdom_style": event.args.wisdom_style,
                "domains": event.args.domains,
                "wisdom_focus": event.args.wisdom_focus,
                "show_steps": event.args.show_steps,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, InnateWisdomResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = "Innate wisdom complete"
        if event.result.errors:
            message = "Innate wisdom finished with errors"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "answer": event.result.answer,
                "used_wisdom_style": event.result.used_wisdom_style,
                "used_domains": event.result.used_domains,
                "used_wisdom_focus": event.result.used_wisdom_focus,
                "show_steps": event.result.show_steps,
                "max_insights": event.result.max_insights,
                "template_source": event.result.template_source,
                "wisdom_blocks": len(event.result.wisdom_blocks),
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Innate wisdom"
