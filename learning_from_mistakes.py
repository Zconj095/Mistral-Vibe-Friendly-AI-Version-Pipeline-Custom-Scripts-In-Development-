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


DEFAULT_PROMPT_TEMPLATE = """### LEARNING FROM MISTAKES MODE (OPT-IN)
Extract lessons and corrections from mistakes before answering.
- Identify what went wrong, why, and how to avoid repeating it.
- Capture corrective actions, safeguards, and checkpoints.
- Keep internal learning private; do not mention this mode in the response.
- If the user asks for steps or show_steps is enabled, give a concise outline.

Learning style: {learning_style}
Show steps: {show_steps}
Max lessons: {max_lessons}
"""

TOOL_PROMPT = (
    "Use `learning_from_mistakes` to extract lessons and corrections. "
    "Provide `prompt` or `messages`, and optionally set `learning_style`, "
    "`show_steps`, and `max_lessons` to control the output."
)


@dataclass(frozen=True)
class _MistakeSource:
    content: str
    label: str | None
    category: str | None
    source_path: str | None


class LearningFromMistakesMessage(BaseModel):
    role: str
    content: str


class MistakeItem(BaseModel):
    id: str | None = Field(default=None, description="Optional mistake id.")
    label: str | None = Field(default=None, description="Optional label.")
    category: str | None = Field(default=None, description="Mistake category.")
    context: str | None = Field(default=None, description="Context summary.")
    mistake: str | None = Field(default=None, description="Mistake description.")
    consequence: str | None = Field(default=None, description="Consequence.")
    correction: str | None = Field(default=None, description="Correction.")
    takeaway: str | None = Field(default=None, description="Key takeaway.")
    content: str | None = Field(default=None, description="Inline content.")
    tags: list[str] | None = Field(default=None, description="Optional tags.")
    path: str | None = Field(default=None, description="Path to a mistake file.")


class LearningFromMistakesArgs(BaseModel):
    prompt: str | None = Field(
        default=None, description="User prompt to solve."
    )
    messages: list[LearningFromMistakesMessage] | None = Field(
        default=None, description="Optional chat messages."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prefix."
    )
    learning_style: str | None = Field(
        default=None, description="Learning style label."
    )
    show_steps: bool | None = Field(
        default=None, description="Whether to include a step outline."
    )
    max_lessons: int | None = Field(
        default=None, description="Maximum lessons in the outline."
    )
    mistakes: list[MistakeItem] | None = Field(
        default=None, description="Mistake items."
    )
    mistake_paths: list[str] | None = Field(
        default=None, description="Additional mistake file paths."
    )
    max_mistake_chars: int | None = Field(
        default=None, description="Max chars per mistake item."
    )
    max_mistake_total_chars: int | None = Field(
        default=None, description="Max total mistake chars."
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


class MistakeBlock(BaseModel):
    label: str | None
    category: str | None
    source_path: str | None
    content: str
    truncated: bool


class LearningFromMistakesResult(BaseModel):
    answer: str
    system_prompt: str
    messages: list[LearningFromMistakesMessage]
    used_learning_style: str
    show_steps: bool
    max_lessons: int
    template_source: str
    mistake_blocks: list[MistakeBlock]
    warnings: list[str]
    errors: list[str]
    llm_model: str


class LearningFromMistakesConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    llm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL.",
    )
    llm_model: str = Field(
        default="gpt-oss:latest", description="Default LLM model name."
    )
    default_learning_style: str = Field(
        default="corrective",
        description="Default learning style.",
    )
    default_show_steps: bool = Field(
        default=False, description="Default for show_steps."
    )
    default_max_lessons: int = Field(
        default=6, description="Default max lessons."
    )
    prompt_path: Path | None = Field(
        default=Path.home()
        / "mistral-vibe"
        / "vibe"
        / "core"
        / "prompts"
        / "learning_from_mistakes.md",
        description="Optional path to a prompt template.",
    )
    prompt_max_chars: int = Field(
        default=8000, description="Maximum template characters to load."
    )
    max_mistake_chars: int = Field(
        default=4000, description="Maximum characters per mistake item."
    )
    max_mistake_total_chars: int = Field(
        default=12000, description="Maximum total mistake characters."
    )


class LearningFromMistakesState(BaseToolState):
    pass


class LearningFromMistakes(
    BaseTool[
        LearningFromMistakesArgs,
        LearningFromMistakesResult,
        LearningFromMistakesConfig,
        LearningFromMistakesState,
    ],
    ToolUIData[LearningFromMistakesArgs, LearningFromMistakesResult],
):
    description: ClassVar[str] = (
        "Extract lessons from mistakes using a dedicated prompt template."
    )

    @classmethod
    def get_tool_prompt(cls) -> str | None:
        return TOOL_PROMPT

    async def run(
        self, args: LearningFromMistakesArgs
    ) -> LearningFromMistakesResult:
        warnings: list[str] = []
        errors: list[str] = []

        learning_style = (
            args.learning_style or self.config.default_learning_style
        ).strip()
        if not learning_style:
            learning_style = "corrective"

        show_steps = (
            args.show_steps
            if args.show_steps is not None
            else self.config.default_show_steps
        )
        max_lessons = (
            args.max_lessons
            if args.max_lessons is not None
            else self.config.default_max_lessons
        )
        if max_lessons <= 0:
            raise ToolError("max_lessons must be positive.")

        self._validate_llm_settings(args)

        template, template_source = self._load_template(warnings)
        system_prompt = self._build_system_prompt(
            template,
            learning_style,
            show_steps,
            max_lessons,
            args.system_prompt,
        )

        mistake_blocks = self._collect_mistakes(args, warnings)
        messages = self._normalize_messages(args, system_prompt)
        messages = self._inject_mistakes(messages, mistake_blocks)
        answer = self._call_llm(messages, args)

        return LearningFromMistakesResult(
            answer=answer,
            system_prompt=system_prompt,
            messages=messages,
            used_learning_style=learning_style,
            show_steps=show_steps,
            max_lessons=max_lessons,
            template_source=template_source,
            mistake_blocks=mistake_blocks,
            warnings=warnings,
            errors=errors,
            llm_model=self._resolve_model(args),
        )

    def _validate_llm_settings(self, args: LearningFromMistakesArgs) -> None:
        if args.llm_temperature < 0:
            raise ToolError("llm_temperature cannot be negative.")
        if args.llm_max_tokens <= 0:
            raise ToolError("llm_max_tokens must be positive.")

    def _collect_mistakes(
        self, args: LearningFromMistakesArgs, warnings: list[str]
    ) -> list[MistakeBlock]:
        max_item = (
            args.max_mistake_chars
            if args.max_mistake_chars is not None
            else self.config.max_mistake_chars
        )
        max_total = (
            args.max_mistake_total_chars
            if args.max_mistake_total_chars is not None
            else self.config.max_mistake_total_chars
        )
        if max_item <= 0 or max_total <= 0:
            raise ToolError(
                "max_mistake_chars and max_mistake_total_chars must be positive."
            )

        sources = self._resolve_mistake_sources(args, warnings)
        blocks: list[MistakeBlock] = []
        total_chars = 0

        for source in sources:
            if total_chars >= max_total:
                warnings.append(
                    "max_mistake_total_chars reached; truncating mistakes."
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
                MistakeBlock(
                    label=source.label,
                    category=source.category,
                    source_path=source.source_path,
                    content=content,
                    truncated=truncated,
                )
            )

        if not blocks:
            warnings.append("No mistake context provided.")
        return blocks

    def _resolve_mistake_sources(
        self, args: LearningFromMistakesArgs, warnings: list[str]
    ) -> list[_MistakeSource]:
        sources: list[_MistakeSource] = []
        if args.mistakes:
            for item in args.mistakes:
                sources.append(self._load_mistake_item(item))

        if args.mistake_paths:
            for raw_path in args.mistake_paths:
                path = self._resolve_path(raw_path)
                content = path.read_text("utf-8", errors="ignore")
                sources.append(
                    _MistakeSource(
                        content=content,
                        label=path.name,
                        category=None,
                        source_path=str(path),
                    )
                )

        if not sources and not warnings:
            warnings.append("No mistake context provided.")
        return sources

    def _load_mistake_item(self, item: MistakeItem) -> _MistakeSource:
        if item.content and item.path:
            raise ToolError("Provide content or path per mistake item, not both.")
        if not (
            item.content
            or item.path
            or item.context
            or item.mistake
            or item.consequence
            or item.correction
            or item.takeaway
        ):
            raise ToolError(
                "Each mistake item must provide content, details, or path."
            )

        label = item.label or item.id or item.path
        text = self._format_mistake_text(item)
        if item.path:
            path = self._resolve_path(item.path)
            content = path.read_text("utf-8", errors="ignore")
            text = self._join_sections(text, "Content", content)
            return _MistakeSource(
                content=text,
                label=label or path.name,
                category=item.category,
                source_path=str(path),
            )

        return _MistakeSource(
            content=text,
            label=label,
            category=item.category,
            source_path=None,
        )

    def _format_mistake_text(self, item: MistakeItem) -> str:
        lines: list[str] = []
        if item.label or item.id:
            lines.append(f"Label: {item.label or item.id}")
        if item.category:
            lines.append(f"Category: {item.category}")
        if item.tags:
            tags = ", ".join(tag for tag in item.tags if tag)
            if tags:
                lines.append(f"Tags: {tags}")
        if item.context:
            lines.append("Context:")
            lines.append(item.context)
        if item.mistake:
            lines.append("Mistake:")
            lines.append(item.mistake)
        if item.consequence:
            lines.append("Consequence:")
            lines.append(item.consequence)
        if item.correction:
            lines.append("Correction:")
            lines.append(item.correction)
        if item.takeaway:
            lines.append("Takeaway:")
            lines.append(item.takeaway)
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

    def _format_mistake_blocks(self, blocks: list[MistakeBlock]) -> str:
        if not blocks:
            return ""
        parts = ["Mistake context:"]
        for block in blocks:
            label = block.label or block.source_path or "mistake"
            category_suffix = f" | {block.category}" if block.category else ""
            trunc = " (truncated)" if block.truncated else ""
            header = f"[{label}{category_suffix}{trunc}]"
            parts.append(f"{header}\n{block.content}".strip())
        return "\n\n".join(parts).strip()

    def _inject_mistakes(
        self,
        messages: list[LearningFromMistakesMessage],
        blocks: list[MistakeBlock],
    ) -> list[LearningFromMistakesMessage]:
        if not blocks:
            return messages
        mistake_text = self._format_mistake_blocks(blocks)
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].role == "user":
                content = messages[idx].content.rstrip()
                messages[idx].content = f"{content}\n\n{mistake_text}".strip()
                return messages
        messages.append(LearningFromMistakesMessage(role="user", content=mistake_text))
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
        learning_style: str,
        show_steps: bool,
        max_lessons: int,
        prefix: str | None,
    ) -> str:
        show_steps_text = "yes" if show_steps else "no"
        rendered = self._render_template(
            template,
            learning_style,
            show_steps_text,
            max_lessons,
        )
        if prefix and prefix.strip():
            return f"{prefix.strip()}\n\n{rendered}".strip()
        return rendered.strip()

    def _render_template(
        self,
        template: str,
        learning_style: str,
        show_steps_text: str,
        max_lessons: int,
    ) -> str:
        had_placeholders = (
            "{learning_style}" in template
            or "{show_steps}" in template
            or "{max_lessons}" in template
        )
        rendered = template
        if "{learning_style}" in template:
            rendered = rendered.replace("{learning_style}", learning_style)
        if "{show_steps}" in template:
            rendered = rendered.replace("{show_steps}", show_steps_text)
        if "{max_lessons}" in template:
            rendered = rendered.replace("{max_lessons}", str(max_lessons))

        if had_placeholders:
            return rendered

        extra = (
            f"Learning style: {learning_style}\n"
            f"Show steps: {show_steps_text}\n"
            f"Max lessons: {max_lessons}"
        )
        return f"{rendered.rstrip()}\n\n{extra}"

    def _normalize_messages(
        self, args: LearningFromMistakesArgs, system_prompt: str
    ) -> list[LearningFromMistakesMessage]:
        messages: list[LearningFromMistakesMessage] = []
        if args.messages:
            for msg in args.messages:
                role = (msg.role or "").strip()
                content = (msg.content or "").strip()
                if not content:
                    continue
                if not role:
                    role = "user"
                messages.append(LearningFromMistakesMessage(role=role, content=content))
        elif args.prompt and args.prompt.strip():
            messages.append(
                LearningFromMistakesMessage(role="user", content=args.prompt.strip())
            )
        else:
            raise ToolError("Provide prompt or messages.")

        if not messages:
            raise ToolError("No usable messages provided.")

        if system_prompt.strip():
            messages.insert(
                0,
                LearningFromMistakesMessage(
                    role="system", content=system_prompt.strip()
                ),
            )
        return messages

    def _call_llm(
        self,
        messages: list[LearningFromMistakesMessage],
        args: LearningFromMistakesArgs,
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

    def _resolve_model(self, args: LearningFromMistakesArgs) -> str:
        return args.llm_model or self.config.llm_model

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, LearningFromMistakesArgs):
            return ToolCallDisplay(summary="learning_from_mistakes")
        return ToolCallDisplay(
            summary="learning_from_mistakes",
            details={
                "learning_style": event.args.learning_style,
                "show_steps": event.args.show_steps,
                "max_lessons": event.args.max_lessons,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, LearningFromMistakesResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = "Learning from mistakes complete"
        if event.result.errors:
            message = "Learning from mistakes finished with errors"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "answer": event.result.answer,
                "used_learning_style": event.result.used_learning_style,
                "show_steps": event.result.show_steps,
                "max_lessons": event.result.max_lessons,
                "template_source": event.result.template_source,
                "mistake_blocks": len(event.result.mistake_blocks),
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Learning from mistakes"
