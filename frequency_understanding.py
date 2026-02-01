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


DEFAULT_PROMPT_TEMPLATE = """### FREQUENCY UNDERSTANDING MODE (OPT-IN)
Develop an advanced understanding of different frequencies and their meaning.
- Interpret frequencies using concepts like period, wavelength, harmonics, spectra, and modulation.
- Distinguish raw signals from inferences; call out noise and uncertainty.
- Connect frequency behavior to plausible sources, mechanisms, and effects.
- Keep internal frequency reasoning private; do not mention this mode in the response.
- If the user asks for steps or show_steps is enabled, give a concise outline.

Understanding style: {understanding_style}
Domain focus: {domain_focus}
Frequency range: {frequency_range}
Show steps: {show_steps}
Max concepts: {max_concepts}
"""

TOOL_PROMPT = (
    "Use `frequency_understanding` to build advanced frequency understanding. "
    "Provide `prompt` or `messages`, and optionally set `understanding_style`, "
    "`domain_focus`, `frequency_range`, `show_steps`, and `max_concepts`."
)


@dataclass(frozen=True)
class _FrequencySource:
    content: str
    label: str | None
    signal_type: str | None
    source_path: str | None


class FrequencyUnderstandingMessage(BaseModel):
    role: str
    content: str


class FrequencyDatasetItem(BaseModel):
    id: str | None = Field(default=None, description="Optional dataset id.")
    label: str | None = Field(default=None, description="Optional label.")
    signal_type: str | None = Field(
        default=None, description="Signal type (time-series, spectrum, log, etc.)."
    )
    content: str | None = Field(default=None, description="Inline dataset content.")
    path: str | None = Field(default=None, description="Path to a dataset file.")


class FrequencyUnderstandingArgs(BaseModel):
    prompt: str | None = Field(
        default=None, description="User prompt to solve."
    )
    messages: list[FrequencyUnderstandingMessage] | None = Field(
        default=None, description="Optional chat messages."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prefix."
    )
    understanding_style: str | None = Field(
        default=None, description="Understanding style label."
    )
    domain_focus: str | None = Field(
        default=None, description="Domain focus (audio, RF, bio, finance, etc.)."
    )
    frequency_range: str | None = Field(
        default=None, description="Frequency range or band of interest."
    )
    show_steps: bool | None = Field(
        default=None, description="Whether to include a step outline."
    )
    max_concepts: int | None = Field(
        default=None, description="Maximum concepts in the outline."
    )
    datasets: list[FrequencyDatasetItem] | None = Field(
        default=None, description="Dataset items to analyze."
    )
    dataset_paths: list[str] | None = Field(
        default=None, description="Additional dataset file paths."
    )
    max_dataset_chars: int | None = Field(
        default=None, description="Max chars per dataset item."
    )
    max_dataset_total_chars: int | None = Field(
        default=None, description="Max total dataset chars."
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


class DatasetBlock(BaseModel):
    label: str | None
    signal_type: str | None
    source_path: str | None
    content: str
    truncated: bool


class FrequencyUnderstandingResult(BaseModel):
    answer: str
    system_prompt: str
    messages: list[FrequencyUnderstandingMessage]
    used_understanding_style: str
    used_domain_focus: str
    used_frequency_range: str
    show_steps: bool
    max_concepts: int
    template_source: str
    dataset_blocks: list[DatasetBlock]
    warnings: list[str]
    errors: list[str]
    llm_model: str


class FrequencyUnderstandingConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    llm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL.",
    )
    llm_model: str = Field(
        default="gpt-oss:latest", description="Default LLM model name."
    )
    default_understanding_style: str = Field(
        default="advanced",
        description="Default understanding style.",
    )
    default_domain_focus: str = Field(
        default="general",
        description="Default domain focus.",
    )
    default_frequency_range: str = Field(
        default="unspecified",
        description="Default frequency range.",
    )
    default_show_steps: bool = Field(
        default=False, description="Default for show_steps."
    )
    default_max_concepts: int = Field(
        default=6, description="Default max concepts."
    )
    prompt_path: Path | None = Field(
        default=Path.home()
        / "mistral-vibe"
        / "vibe"
        / "core"
        / "prompts"
        / "frequency_understanding.md",
        description="Optional path to a prompt template.",
    )
    prompt_max_chars: int = Field(
        default=8000, description="Maximum template characters to load."
    )
    max_dataset_chars: int = Field(
        default=4000, description="Maximum characters per dataset item."
    )
    max_dataset_total_chars: int = Field(
        default=12000, description="Maximum total dataset characters."
    )


class FrequencyUnderstandingState(BaseToolState):
    pass


class FrequencyUnderstanding(
    BaseTool[
        FrequencyUnderstandingArgs,
        FrequencyUnderstandingResult,
        FrequencyUnderstandingConfig,
        FrequencyUnderstandingState,
    ],
    ToolUIData[FrequencyUnderstandingArgs, FrequencyUnderstandingResult],
):
    description: ClassVar[str] = (
        "Form advanced understanding of frequencies using a prompt template."
    )

    @classmethod
    def get_tool_prompt(cls) -> str | None:
        return TOOL_PROMPT

    async def run(
        self, args: FrequencyUnderstandingArgs
    ) -> FrequencyUnderstandingResult:
        warnings: list[str] = []
        errors: list[str] = []

        understanding_style = (
            args.understanding_style or self.config.default_understanding_style
        ).strip()
        if not understanding_style:
            understanding_style = "advanced"

        domain_focus = (
            args.domain_focus or self.config.default_domain_focus
        ).strip()
        if not domain_focus:
            domain_focus = "general"

        frequency_range = (
            args.frequency_range or self.config.default_frequency_range
        ).strip()
        if not frequency_range:
            frequency_range = "unspecified"

        show_steps = (
            args.show_steps
            if args.show_steps is not None
            else self.config.default_show_steps
        )
        max_concepts = (
            args.max_concepts
            if args.max_concepts is not None
            else self.config.default_max_concepts
        )
        if max_concepts <= 0:
            raise ToolError("max_concepts must be positive.")

        self._validate_llm_settings(args)

        template, template_source = self._load_template(warnings)
        system_prompt = self._build_system_prompt(
            template,
            understanding_style,
            domain_focus,
            frequency_range,
            show_steps,
            max_concepts,
            args.system_prompt,
        )

        dataset_blocks = self._collect_datasets(args, warnings)
        messages = self._normalize_messages(args, system_prompt)
        messages = self._inject_datasets(messages, dataset_blocks)
        answer = self._call_llm(messages, args)

        return FrequencyUnderstandingResult(
            answer=answer,
            system_prompt=system_prompt,
            messages=messages,
            used_understanding_style=understanding_style,
            used_domain_focus=domain_focus,
            used_frequency_range=frequency_range,
            show_steps=show_steps,
            max_concepts=max_concepts,
            template_source=template_source,
            dataset_blocks=dataset_blocks,
            warnings=warnings,
            errors=errors,
            llm_model=self._resolve_model(args),
        )

    def _validate_llm_settings(self, args: FrequencyUnderstandingArgs) -> None:
        if args.llm_temperature < 0:
            raise ToolError("llm_temperature cannot be negative.")
        if args.llm_max_tokens <= 0:
            raise ToolError("llm_max_tokens must be positive.")

    def _collect_datasets(
        self, args: FrequencyUnderstandingArgs, warnings: list[str]
    ) -> list[DatasetBlock]:
        max_item = (
            args.max_dataset_chars
            if args.max_dataset_chars is not None
            else self.config.max_dataset_chars
        )
        max_total = (
            args.max_dataset_total_chars
            if args.max_dataset_total_chars is not None
            else self.config.max_dataset_total_chars
        )
        if max_item <= 0 or max_total <= 0:
            raise ToolError(
                "max_dataset_chars and max_dataset_total_chars must be positive."
            )

        sources = self._resolve_dataset_sources(args, warnings)
        blocks: list[DatasetBlock] = []
        total_chars = 0

        for source in sources:
            if total_chars >= max_total:
                warnings.append(
                    "max_dataset_total_chars reached; truncating datasets."
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
                DatasetBlock(
                    label=source.label,
                    signal_type=source.signal_type,
                    source_path=source.source_path,
                    content=content,
                    truncated=truncated,
                )
            )

        if not blocks:
            warnings.append("No dataset context provided.")
        return blocks

    def _resolve_dataset_sources(
        self, args: FrequencyUnderstandingArgs, warnings: list[str]
    ) -> list[_FrequencySource]:
        sources: list[_FrequencySource] = []
        if args.datasets:
            for item in args.datasets:
                sources.append(self._load_dataset_item(item))

        if args.dataset_paths:
            for raw_path in args.dataset_paths:
                path = self._resolve_path(raw_path)
                content = path.read_text("utf-8", errors="ignore")
                sources.append(
                    _FrequencySource(
                        content=content,
                        label=path.name,
                        signal_type=None,
                        source_path=str(path),
                    )
                )

        if not sources and not warnings:
            warnings.append("No dataset context provided.")
        return sources

    def _load_dataset_item(self, item: FrequencyDatasetItem) -> _FrequencySource:
        if item.content and item.path:
            raise ToolError("Provide content or path per dataset item, not both.")
        if not item.content and not item.path:
            raise ToolError("Each dataset item must provide content or path.")

        label = item.label or item.id or item.path
        if item.content is not None:
            return _FrequencySource(
                content=item.content,
                label=label,
                signal_type=item.signal_type,
                source_path=None,
            )

        path = self._resolve_path(item.path or "")
        content = path.read_text("utf-8", errors="ignore")
        return _FrequencySource(
            content=content,
            label=label or path.name,
            signal_type=item.signal_type,
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

    def _format_dataset_blocks(self, blocks: list[DatasetBlock]) -> str:
        if not blocks:
            return ""
        parts = ["Dataset context:"]
        for block in blocks:
            label = block.label or block.source_path or "dataset"
            type_suffix = f" | {block.signal_type}" if block.signal_type else ""
            trunc = " (truncated)" if block.truncated else ""
            header = f"[{label}{type_suffix}{trunc}]"
            parts.append(f"{header}\n{block.content}".strip())
        return "\n\n".join(parts).strip()

    def _inject_datasets(
        self,
        messages: list[FrequencyUnderstandingMessage],
        blocks: list[DatasetBlock],
    ) -> list[FrequencyUnderstandingMessage]:
        if not blocks:
            return messages
        dataset_text = self._format_dataset_blocks(blocks)
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].role == "user":
                content = messages[idx].content.rstrip()
                messages[idx].content = f"{content}\n\n{dataset_text}".strip()
                return messages
        messages.append(
            FrequencyUnderstandingMessage(role="user", content=dataset_text)
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
        understanding_style: str,
        domain_focus: str,
        frequency_range: str,
        show_steps: bool,
        max_concepts: int,
        prefix: str | None,
    ) -> str:
        show_steps_text = "yes" if show_steps else "no"
        rendered = self._render_template(
            template,
            understanding_style,
            domain_focus,
            frequency_range,
            show_steps_text,
            max_concepts,
        )
        if prefix and prefix.strip():
            return f"{prefix.strip()}\n\n{rendered}".strip()
        return rendered.strip()

    def _render_template(
        self,
        template: str,
        understanding_style: str,
        domain_focus: str,
        frequency_range: str,
        show_steps_text: str,
        max_concepts: int,
    ) -> str:
        had_placeholders = (
            "{understanding_style}" in template
            or "{domain_focus}" in template
            or "{frequency_range}" in template
            or "{show_steps}" in template
            or "{max_concepts}" in template
        )
        rendered = template
        if "{understanding_style}" in template:
            rendered = rendered.replace("{understanding_style}", understanding_style)
        if "{domain_focus}" in template:
            rendered = rendered.replace("{domain_focus}", domain_focus)
        if "{frequency_range}" in template:
            rendered = rendered.replace("{frequency_range}", frequency_range)
        if "{show_steps}" in template:
            rendered = rendered.replace("{show_steps}", show_steps_text)
        if "{max_concepts}" in template:
            rendered = rendered.replace("{max_concepts}", str(max_concepts))

        if had_placeholders:
            return rendered

        extra = (
            f"Understanding style: {understanding_style}\n"
            f"Domain focus: {domain_focus}\n"
            f"Frequency range: {frequency_range}\n"
            f"Show steps: {show_steps_text}\n"
            f"Max concepts: {max_concepts}"
        )
        return f"{rendered.rstrip()}\n\n{extra}"

    def _normalize_messages(
        self, args: FrequencyUnderstandingArgs, system_prompt: str
    ) -> list[FrequencyUnderstandingMessage]:
        messages: list[FrequencyUnderstandingMessage] = []
        if args.messages:
            for msg in args.messages:
                role = (msg.role or "").strip()
                content = (msg.content or "").strip()
                if not content:
                    continue
                if not role:
                    role = "user"
                messages.append(
                    FrequencyUnderstandingMessage(role=role, content=content)
                )
        elif args.prompt and args.prompt.strip():
            messages.append(
                FrequencyUnderstandingMessage(
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
                FrequencyUnderstandingMessage(
                    role="system", content=system_prompt.strip()
                ),
            )
        return messages

    def _call_llm(
        self,
        messages: list[FrequencyUnderstandingMessage],
        args: FrequencyUnderstandingArgs,
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

    def _resolve_model(self, args: FrequencyUnderstandingArgs) -> str:
        return args.llm_model or self.config.llm_model

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, FrequencyUnderstandingArgs):
            return ToolCallDisplay(summary="frequency_understanding")
        return ToolCallDisplay(
            summary="frequency_understanding",
            details={
                "understanding_style": event.args.understanding_style,
                "domain_focus": event.args.domain_focus,
                "frequency_range": event.args.frequency_range,
                "show_steps": event.args.show_steps,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, FrequencyUnderstandingResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = "Frequency understanding complete"
        if event.result.errors:
            message = "Frequency understanding finished with errors"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "answer": event.result.answer,
                "used_understanding_style": event.result.used_understanding_style,
                "used_domain_focus": event.result.used_domain_focus,
                "used_frequency_range": event.result.used_frequency_range,
                "show_steps": event.result.show_steps,
                "max_concepts": event.result.max_concepts,
                "template_source": event.result.template_source,
                "dataset_blocks": len(event.result.dataset_blocks),
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Frequency understanding"
