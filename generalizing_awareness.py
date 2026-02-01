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


DEFAULT_PROMPT_TEMPLATE = """### GENERALIZING AWARENESS MODE (OPT-IN)
Generalize different forms of awareness before answering.
- Identify common patterns across awareness forms.
- Capture transferable principles and boundaries; call out uncertainty.
- Avoid overgeneralizing; note exceptions and constraints.
- Keep internal generalization private; do not mention this mode in the response.
- If the user asks for steps or show_steps is enabled, give a concise outline.

Generalization style: {generalization_style}
Awareness forms: {awareness_forms}
Generalization focus: {generalization_focus}
Show steps: {show_steps}
Max generalizations: {max_generalizations}
"""

TOOL_PROMPT = (
    "Use `generalizing_awareness` to generalize awareness forms. "
    "Provide `prompt` or `messages`, and optionally set `generalization_style`, "
    "`awareness_forms`, `generalization_focus`, `show_steps`, and "
    "`max_generalizations`."
)


@dataclass(frozen=True)
class _AwarenessSource:
    content: str
    label: str | None
    awareness_form: str | None
    source_path: str | None


class AwarenessMessage(BaseModel):
    role: str
    content: str


class AwarenessItem(BaseModel):
    id: str | None = Field(default=None, description="Optional item id.")
    label: str | None = Field(default=None, description="Optional label.")
    awareness_form: str | None = Field(
        default=None, description="Awareness form label."
    )
    context: str | None = Field(default=None, description="Context summary.")
    content: str | None = Field(default=None, description="Inline content.")
    tags: list[str] | None = Field(default=None, description="Optional tags.")
    path: str | None = Field(default=None, description="Path to a context file.")


class GeneralizingAwarenessArgs(BaseModel):
    prompt: str | None = Field(
        default=None, description="User prompt to solve."
    )
    messages: list[AwarenessMessage] | None = Field(
        default=None, description="Optional chat messages."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prefix."
    )
    generalization_style: str | None = Field(
        default=None, description="Generalization style label."
    )
    awareness_forms: list[str] | None = Field(
        default=None, description="Awareness forms to generalize."
    )
    generalization_focus: str | None = Field(
        default=None, description="Generalization focus label."
    )
    show_steps: bool | None = Field(
        default=None, description="Whether to include a step outline."
    )
    max_generalizations: int | None = Field(
        default=None, description="Maximum generalizations in the outline."
    )
    items: list[AwarenessItem] | None = Field(
        default=None, description="Awareness context items."
    )
    item_paths: list[str] | None = Field(
        default=None, description="Additional context file paths."
    )
    max_item_chars: int | None = Field(
        default=None, description="Max chars per item."
    )
    max_item_total_chars: int | None = Field(
        default=None, description="Max total item chars."
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


class AwarenessBlock(BaseModel):
    label: str | None
    awareness_form: str | None
    source_path: str | None
    content: str
    truncated: bool


class GeneralizingAwarenessResult(BaseModel):
    answer: str
    system_prompt: str
    messages: list[AwarenessMessage]
    used_generalization_style: str
    used_awareness_forms: list[str]
    used_generalization_focus: str
    show_steps: bool
    max_generalizations: int
    template_source: str
    item_blocks: list[AwarenessBlock]
    warnings: list[str]
    errors: list[str]
    llm_model: str


class GeneralizingAwarenessConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    llm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL.",
    )
    llm_model: str = Field(
        default="gpt-oss:latest", description="Default LLM model name."
    )
    default_generalization_style: str = Field(
        default="transferable",
        description="Default generalization style.",
    )
    default_awareness_forms: list[str] = Field(
        default_factory=lambda: [
            "sensory",
            "situational",
            "social",
            "metacognitive",
            "interoceptive",
            "spatial",
        ],
        description="Default awareness forms.",
    )
    default_generalization_focus: str = Field(
        default="awareness-forms",
        description="Default generalization focus.",
    )
    default_show_steps: bool = Field(
        default=False, description="Default for show_steps."
    )
    default_max_generalizations: int = Field(
        default=6, description="Default max generalizations."
    )
    prompt_path: Path | None = Field(
        default=Path.home()
        / "mistral-vibe"
        / "vibe"
        / "core"
        / "prompts"
        / "generalizing_awareness.md",
        description="Optional path to a prompt template.",
    )
    prompt_max_chars: int = Field(
        default=8000, description="Maximum template characters to load."
    )
    max_item_chars: int = Field(
        default=4000, description="Maximum characters per item."
    )
    max_item_total_chars: int = Field(
        default=12000, description="Maximum total item characters."
    )


class GeneralizingAwarenessState(BaseToolState):
    pass


class GeneralizingAwareness(
    BaseTool[
        GeneralizingAwarenessArgs,
        GeneralizingAwarenessResult,
        GeneralizingAwarenessConfig,
        GeneralizingAwarenessState,
    ],
    ToolUIData[GeneralizingAwarenessArgs, GeneralizingAwarenessResult],
):
    description: ClassVar[str] = (
        "Generalize awareness forms using a prompt template."
    )

    @classmethod
    def get_tool_prompt(cls) -> str | None:
        return TOOL_PROMPT

    async def run(
        self, args: GeneralizingAwarenessArgs
    ) -> GeneralizingAwarenessResult:
        warnings: list[str] = []
        errors: list[str] = []

        generalization_style = (
            args.generalization_style or self.config.default_generalization_style
        ).strip()
        if not generalization_style:
            generalization_style = "transferable"

        awareness_forms = self._normalize_list(
            args.awareness_forms, self.config.default_awareness_forms
        )
        if not awareness_forms:
            raise ToolError("awareness_forms must include at least one form.")

        generalization_focus = (
            args.generalization_focus or self.config.default_generalization_focus
        ).strip()
        if not generalization_focus:
            generalization_focus = "awareness-forms"

        show_steps = (
            args.show_steps
            if args.show_steps is not None
            else self.config.default_show_steps
        )
        max_generalizations = (
            args.max_generalizations
            if args.max_generalizations is not None
            else self.config.default_max_generalizations
        )
        if max_generalizations <= 0:
            raise ToolError("max_generalizations must be positive.")

        self._validate_llm_settings(args)

        template, template_source = self._load_template(warnings)
        system_prompt = self._build_system_prompt(
            template,
            generalization_style,
            awareness_forms,
            generalization_focus,
            show_steps,
            max_generalizations,
            args.system_prompt,
        )

        item_blocks = self._collect_items(args, warnings)
        messages = self._normalize_messages(args, system_prompt)
        messages = self._inject_items(messages, item_blocks)
        answer = self._call_llm(messages, args)

        return GeneralizingAwarenessResult(
            answer=answer,
            system_prompt=system_prompt,
            messages=messages,
            used_generalization_style=generalization_style,
            used_awareness_forms=awareness_forms,
            used_generalization_focus=generalization_focus,
            show_steps=show_steps,
            max_generalizations=max_generalizations,
            template_source=template_source,
            item_blocks=item_blocks,
            warnings=warnings,
            errors=errors,
            llm_model=self._resolve_model(args),
        )

    def _normalize_list(
        self, values: list[str] | None, defaults: list[str]
    ) -> list[str]:
        raw = values or list(defaults)
        normalized: list[str] = []
        seen: set[str] = set()
        for value in raw:
            if not value or not value.strip():
                continue
            clean = value.strip()
            key = clean.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(clean)
        return normalized

    def _validate_llm_settings(self, args: GeneralizingAwarenessArgs) -> None:
        if args.llm_temperature < 0:
            raise ToolError("llm_temperature cannot be negative.")
        if args.llm_max_tokens <= 0:
            raise ToolError("llm_max_tokens must be positive.")

    def _collect_items(
        self, args: GeneralizingAwarenessArgs, warnings: list[str]
    ) -> list[AwarenessBlock]:
        max_item = (
            args.max_item_chars
            if args.max_item_chars is not None
            else self.config.max_item_chars
        )
        max_total = (
            args.max_item_total_chars
            if args.max_item_total_chars is not None
            else self.config.max_item_total_chars
        )
        if max_item <= 0 or max_total <= 0:
            raise ToolError(
                "max_item_chars and max_item_total_chars must be positive."
            )

        sources = self._resolve_sources(args, warnings)
        blocks: list[AwarenessBlock] = []
        total_chars = 0
        for source in sources:
            if total_chars >= max_total:
                warnings.append(
                    "max_item_total_chars reached; truncating context items."
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
                AwarenessBlock(
                    label=source.label,
                    awareness_form=source.awareness_form,
                    source_path=source.source_path,
                    content=content,
                    truncated=truncated,
                )
            )

        if not blocks:
            warnings.append("No awareness context provided.")
        return blocks

    def _resolve_sources(
        self, args: GeneralizingAwarenessArgs, warnings: list[str]
    ) -> list[_AwarenessSource]:
        sources: list[_AwarenessSource] = []
        if args.items:
            for item in args.items:
                sources.append(self._load_item(item))

        if args.item_paths:
            for raw_path in args.item_paths:
                path = self._resolve_path(raw_path)
                content = path.read_text("utf-8", errors="ignore")
                sources.append(
                    _AwarenessSource(
                        content=content,
                        label=path.name,
                        awareness_form=None,
                        source_path=str(path),
                    )
                )

        if not sources and not warnings:
            warnings.append("No awareness context provided.")
        return sources

    def _load_item(self, item: AwarenessItem) -> _AwarenessSource:
        if item.content and item.path:
            raise ToolError("Provide content or path per item, not both.")
        if not item.content and not item.path and not item.context:
            raise ToolError(
                "Each item must provide content, context, or path."
            )

        label = item.label or item.id or item.path
        text = self._format_item_text(item)

        if item.path:
            path = self._resolve_path(item.path)
            content = path.read_text("utf-8", errors="ignore")
            text = self._join_sections(text, "Content", content)
            return _AwarenessSource(
                content=text,
                label=label or path.name,
                awareness_form=item.awareness_form,
                source_path=str(path),
            )

        return _AwarenessSource(
            content=text,
            label=label,
            awareness_form=item.awareness_form,
            source_path=None,
        )

    def _format_item_text(self, item: AwarenessItem) -> str:
        lines: list[str] = []
        if item.label or item.id:
            lines.append(f"Label: {item.label or item.id}")
        if item.awareness_form:
            lines.append(f"Awareness form: {item.awareness_form}")
        if item.tags:
            tags = ", ".join(tag for tag in item.tags if tag)
            if tags:
                lines.append(f"Tags: {tags}")
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

    def _format_item_blocks(self, blocks: list[AwarenessBlock]) -> str:
        if not blocks:
            return ""
        parts = ["Awareness context:"]
        for block in blocks:
            label = block.label or block.source_path or "context"
            form_suffix = f" | {block.awareness_form}" if block.awareness_form else ""
            trunc = " (truncated)" if block.truncated else ""
            header = f"[{label}{form_suffix}{trunc}]"
            parts.append(f"{header}\n{block.content}".strip())
        return "\n\n".join(parts).strip()

    def _inject_items(
        self,
        messages: list[AwarenessMessage],
        blocks: list[AwarenessBlock],
    ) -> list[AwarenessMessage]:
        if not blocks:
            return messages
        context_text = self._format_item_blocks(blocks)
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].role == "user":
                content = messages[idx].content.rstrip()
                messages[idx].content = f"{content}\n\n{context_text}".strip()
                return messages
        messages.append(AwarenessMessage(role="user", content=context_text))
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
        generalization_style: str,
        awareness_forms: list[str],
        generalization_focus: str,
        show_steps: bool,
        max_generalizations: int,
        prefix: str | None,
    ) -> str:
        show_steps_text = "yes" if show_steps else "no"
        forms_text = ", ".join(awareness_forms)
        rendered = self._render_template(
            template,
            generalization_style,
            forms_text,
            generalization_focus,
            show_steps_text,
            max_generalizations,
        )
        if prefix and prefix.strip():
            return f"{prefix.strip()}\n\n{rendered}".strip()
        return rendered.strip()

    def _render_template(
        self,
        template: str,
        generalization_style: str,
        awareness_forms: str,
        generalization_focus: str,
        show_steps_text: str,
        max_generalizations: int,
    ) -> str:
        had_placeholders = (
            "{generalization_style}" in template
            or "{awareness_forms}" in template
            or "{generalization_focus}" in template
            or "{show_steps}" in template
            or "{max_generalizations}" in template
        )
        rendered = template
        if "{generalization_style}" in template:
            rendered = rendered.replace("{generalization_style}", generalization_style)
        if "{awareness_forms}" in template:
            rendered = rendered.replace("{awareness_forms}", awareness_forms)
        if "{generalization_focus}" in template:
            rendered = rendered.replace("{generalization_focus}", generalization_focus)
        if "{show_steps}" in template:
            rendered = rendered.replace("{show_steps}", show_steps_text)
        if "{max_generalizations}" in template:
            rendered = rendered.replace(
                "{max_generalizations}", str(max_generalizations)
            )

        if had_placeholders:
            return rendered

        extra = (
            f"Generalization style: {generalization_style}\n"
            f"Awareness forms: {awareness_forms}\n"
            f"Generalization focus: {generalization_focus}\n"
            f"Show steps: {show_steps_text}\n"
            f"Max generalizations: {max_generalizations}"
        )
        return f"{rendered.rstrip()}\n\n{extra}"

    def _normalize_messages(
        self, args: GeneralizingAwarenessArgs, system_prompt: str
    ) -> list[AwarenessMessage]:
        messages: list[AwarenessMessage] = []
        if args.messages:
            for msg in args.messages:
                role = (msg.role or "").strip()
                content = (msg.content or "").strip()
                if not content:
                    continue
                if not role:
                    role = "user"
                messages.append(AwarenessMessage(role=role, content=content))
        elif args.prompt and args.prompt.strip():
            messages.append(
                AwarenessMessage(role="user", content=args.prompt.strip())
            )
        else:
            raise ToolError("Provide prompt or messages.")

        if not messages:
            raise ToolError("No usable messages provided.")

        if system_prompt.strip():
            messages.insert(
                0, AwarenessMessage(role="system", content=system_prompt.strip())
            )
        return messages

    def _call_llm(
        self,
        messages: list[AwarenessMessage],
        args: GeneralizingAwarenessArgs,
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

    def _resolve_model(self, args: GeneralizingAwarenessArgs) -> str:
        return args.llm_model or self.config.llm_model

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, GeneralizingAwarenessArgs):
            return ToolCallDisplay(summary="generalizing_awareness")
        return ToolCallDisplay(
            summary="generalizing_awareness",
            details={
                "generalization_style": event.args.generalization_style,
                "awareness_forms": event.args.awareness_forms,
                "generalization_focus": event.args.generalization_focus,
                "show_steps": event.args.show_steps,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, GeneralizingAwarenessResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = "Generalizing awareness complete"
        if event.result.errors:
            message = "Generalizing awareness finished with errors"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "answer": event.result.answer,
                "used_generalization_style": event.result.used_generalization_style,
                "used_awareness_forms": event.result.used_awareness_forms,
                "used_generalization_focus": event.result.used_generalization_focus,
                "show_steps": event.result.show_steps,
                "max_generalizations": event.result.max_generalizations,
                "template_source": event.result.template_source,
                "item_blocks": len(event.result.item_blocks),
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Generalizing awareness"
