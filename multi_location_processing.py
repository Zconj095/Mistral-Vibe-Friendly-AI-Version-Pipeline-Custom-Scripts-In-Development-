from __future__ import annotations

import fnmatch
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


DEFAULT_PROMPT_TEMPLATE = """### MULTI-LOCATION PROCESSING MODE (OPT-IN)
Process large amounts of information from multiple locations before answering.
- Consolidate key facts across sources and reconcile conflicts.
- Flag missing data or mismatched formats.
- Keep internal aggregation private; do not mention this mode in the response.
- If the user asks for steps or show_steps is enabled, give a concise outline.

Style: {style}
Show steps: {show_steps}
Max locations: {max_locations}
"""

TOOL_PROMPT = (
    "Use `multi_location_processing` to consolidate information from many locations. "
    "Provide `prompt` or `messages`, and optionally set `style`, `show_steps`, "
    "and `max_locations` to control the processing behavior."
)


@dataclass(frozen=True)
class _LocationSource:
    content: str
    label: str | None
    location_type: str | None
    source_path: str | None
    truncated: bool


class MultiLocationMessage(BaseModel):
    role: str
    content: str


class LocationItem(BaseModel):
    id: str | None = Field(default=None, description="Optional location id.")
    label: str | None = Field(default=None, description="Optional label.")
    location_type: str | None = Field(
        default=None, description="Location type (file, note, log, etc.)."
    )
    content: str | None = Field(default=None, description="Inline content.")
    path: str | None = Field(default=None, description="Path to a file or directory.")


class MultiLocationProcessingArgs(BaseModel):
    prompt: str | None = Field(
        default=None, description="User prompt to solve."
    )
    messages: list[MultiLocationMessage] | None = Field(
        default=None, description="Optional chat messages."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prefix."
    )
    style: str | None = Field(
        default=None, description="Processing style label."
    )
    show_steps: bool | None = Field(
        default=None, description="Whether to include a step outline."
    )
    max_locations: int | None = Field(
        default=None, description="Maximum locations in the outline."
    )
    locations: list[LocationItem] | None = Field(
        default=None, description="Location items."
    )
    location_paths: list[str] | None = Field(
        default=None, description="Additional location paths."
    )
    include_globs: list[str] | None = Field(
        default=None, description="Glob patterns to include when scanning directories."
    )
    exclude_globs: list[str] | None = Field(
        default=None, description="Glob patterns to exclude when scanning directories."
    )
    recursive: bool | None = Field(
        default=None, description="Recursively scan directories."
    )
    max_files: int | None = Field(
        default=None, description="Maximum files to read from directories."
    )
    max_file_bytes: int | None = Field(
        default=None, description="Maximum bytes to read per file."
    )
    max_location_chars: int | None = Field(
        default=None, description="Max chars per location item."
    )
    max_location_total_chars: int | None = Field(
        default=None, description="Max total location chars."
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


class LocationBlock(BaseModel):
    label: str | None
    location_type: str | None
    source_path: str | None
    content: str
    truncated: bool


class MultiLocationProcessingResult(BaseModel):
    answer: str
    system_prompt: str
    messages: list[MultiLocationMessage]
    used_style: str
    show_steps: bool
    max_locations: int
    template_source: str
    location_blocks: list[LocationBlock]
    warnings: list[str]
    errors: list[str]
    llm_model: str


class MultiLocationProcessingConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    llm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL.",
    )
    llm_model: str = Field(
        default="gpt-oss:latest", description="Default LLM model name."
    )
    default_style: str = Field(
        default="aggregative",
        description="Default processing style.",
    )
    default_show_steps: bool = Field(
        default=False, description="Default for show_steps."
    )
    default_max_locations: int = Field(
        default=8, description="Default max locations."
    )
    prompt_path: Path | None = Field(
        default=Path.home()
        / "mistral-vibe"
        / "vibe"
        / "core"
        / "prompts"
        / "multi_location_processing.md",
        description="Optional path to a prompt template.",
    )
    prompt_max_chars: int = Field(
        default=8000, description="Maximum template characters to load."
    )
    recursive: bool = Field(
        default=True, description="Recursively scan directories."
    )
    max_files: int = Field(
        default=40, description="Maximum files to read from directories."
    )
    max_file_bytes: int = Field(
        default=400_000, description="Maximum bytes to read per file."
    )
    max_location_chars: int = Field(
        default=4000, description="Maximum characters per location item."
    )
    max_location_total_chars: int = Field(
        default=12000, description="Maximum total location characters."
    )
    include_globs: list[str] = Field(
        default_factory=list,
        description="Glob patterns to include when scanning directories.",
    )
    exclude_globs: list[str] = Field(
        default_factory=lambda: [
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.gif",
            "*.webp",
            "*.bmp",
            "*.exe",
            "*.dll",
            "*.bin",
            "*.zip",
            "*.7z",
            "*.tar",
            "*.gz",
            "*.pdf",
        ],
        description="Glob patterns to exclude when scanning directories.",
    )


class MultiLocationProcessingState(BaseToolState):
    pass


class MultiLocationProcessing(
    BaseTool[
        MultiLocationProcessingArgs,
        MultiLocationProcessingResult,
        MultiLocationProcessingConfig,
        MultiLocationProcessingState,
    ],
    ToolUIData[MultiLocationProcessingArgs, MultiLocationProcessingResult],
):
    description: ClassVar[str] = (
        "Process large amounts of information from multiple locations."
    )

    @classmethod
    def get_tool_prompt(cls) -> str | None:
        return TOOL_PROMPT

    async def run(
        self, args: MultiLocationProcessingArgs
    ) -> MultiLocationProcessingResult:
        warnings: list[str] = []
        errors: list[str] = []

        style = (args.style or self.config.default_style).strip()
        if not style:
            style = "aggregative"

        show_steps = (
            args.show_steps
            if args.show_steps is not None
            else self.config.default_show_steps
        )
        max_locations = (
            args.max_locations
            if args.max_locations is not None
            else self.config.default_max_locations
        )
        if max_locations <= 0:
            raise ToolError("max_locations must be positive.")

        self._validate_llm_settings(args)

        template, template_source = self._load_template(warnings)
        system_prompt = self._build_system_prompt(
            template, style, show_steps, max_locations, args.system_prompt
        )

        location_blocks = self._collect_locations(args, warnings)
        messages = self._normalize_messages(args, system_prompt)
        messages = self._inject_locations(messages, location_blocks)
        answer = self._call_llm(messages, args)

        return MultiLocationProcessingResult(
            answer=answer,
            system_prompt=system_prompt,
            messages=messages,
            used_style=style,
            show_steps=show_steps,
            max_locations=max_locations,
            template_source=template_source,
            location_blocks=location_blocks,
            warnings=warnings,
            errors=errors,
            llm_model=self._resolve_model(args),
        )

    def _validate_llm_settings(self, args: MultiLocationProcessingArgs) -> None:
        if args.llm_temperature < 0:
            raise ToolError("llm_temperature cannot be negative.")
        if args.llm_max_tokens <= 0:
            raise ToolError("llm_max_tokens must be positive.")

    def _collect_locations(
        self, args: MultiLocationProcessingArgs, warnings: list[str]
    ) -> list[LocationBlock]:
        max_item = (
            args.max_location_chars
            if args.max_location_chars is not None
            else self.config.max_location_chars
        )
        max_total = (
            args.max_location_total_chars
            if args.max_location_total_chars is not None
            else self.config.max_location_total_chars
        )
        if max_item <= 0 or max_total <= 0:
            raise ToolError(
                "max_location_chars and max_location_total_chars must be positive."
            )

        sources = self._resolve_sources(args, warnings)
        blocks: list[LocationBlock] = []
        total_chars = 0

        for source in sources:
            if total_chars >= max_total:
                warnings.append(
                    "max_location_total_chars reached; truncating locations."
                )
                break
            content = source.content
            truncated = source.truncated
            if len(content) > max_item:
                content = content[:max_item]
                truncated = True
            if total_chars + len(content) > max_total:
                content = content[: max_total - total_chars]
                truncated = True
            total_chars += len(content)
            blocks.append(
                LocationBlock(
                    label=source.label,
                    location_type=source.location_type,
                    source_path=source.source_path,
                    content=content,
                    truncated=truncated,
                )
            )

        if not blocks:
            warnings.append("No location context provided.")
        return blocks

    def _resolve_sources(
        self, args: MultiLocationProcessingArgs, warnings: list[str]
    ) -> list[_LocationSource]:
        sources: list[_LocationSource] = []
        recursive = (
            args.recursive
            if args.recursive is not None
            else self.config.recursive
        )
        max_files = (
            args.max_files if args.max_files is not None else self.config.max_files
        )
        max_file_bytes = (
            args.max_file_bytes
            if args.max_file_bytes is not None
            else self.config.max_file_bytes
        )
        include_globs = args.include_globs or self.config.include_globs
        exclude_globs = args.exclude_globs or self.config.exclude_globs

        if args.locations:
            for item in args.locations:
                sources.extend(
                    self._load_location_item(
                        item,
                        recursive,
                        max_files,
                        max_file_bytes,
                        include_globs,
                        exclude_globs,
                        warnings,
                    )
                )

        if args.location_paths:
            for raw_path in args.location_paths:
                sources.extend(
                    self._load_location_path(
                        raw_path,
                        None,
                        recursive,
                        max_files,
                        max_file_bytes,
                        include_globs,
                        exclude_globs,
                        warnings,
                    )
                )

        return sources

    def _load_location_item(
        self,
        item: LocationItem,
        recursive: bool,
        max_files: int,
        max_file_bytes: int,
        include_globs: list[str],
        exclude_globs: list[str],
        warnings: list[str],
    ) -> list[_LocationSource]:
        if item.content and item.path:
            raise ToolError("Provide content or path per location item, not both.")
        if not item.content and not item.path:
            raise ToolError("Each location item must provide content or path.")

        label = item.label or item.id or item.path
        if item.content is not None:
            return [
                _LocationSource(
                    content=item.content,
                    label=label,
                    location_type=item.location_type,
                    source_path=None,
                    truncated=False,
                )
            ]

        return self._load_location_path(
            item.path or "",
            item.location_type,
            recursive,
            max_files,
            max_file_bytes,
            include_globs,
            exclude_globs,
            warnings,
            label=label,
        )

    def _load_location_path(
        self,
        raw_path: str,
        location_type: str | None,
        recursive: bool,
        max_files: int,
        max_file_bytes: int,
        include_globs: list[str],
        exclude_globs: list[str],
        warnings: list[str],
        label: str | None = None,
    ) -> list[_LocationSource]:
        path = self._resolve_path(raw_path)
        if path.is_dir():
            return self._scan_directory(
                path,
                location_type,
                recursive,
                max_files,
                max_file_bytes,
                include_globs,
                exclude_globs,
                warnings,
            )

        content, truncated = self._read_file_limited(path, max_file_bytes)
        return [
            _LocationSource(
                content=content,
                label=label or path.name,
                location_type=location_type,
                source_path=str(path),
                truncated=truncated,
            )
        ]

    def _scan_directory(
        self,
        directory: Path,
        location_type: str | None,
        recursive: bool,
        max_files: int,
        max_file_bytes: int,
        include_globs: list[str],
        exclude_globs: list[str],
        warnings: list[str],
    ) -> list[_LocationSource]:
        sources: list[_LocationSource] = []
        if max_files <= 0:
            warnings.append("max_files is 0; skipping directory scan.")
            return sources

        iterator = directory.rglob("*") if recursive else directory.iterdir()
        for path in sorted(iterator):
            if not path.is_file():
                continue
            if not self._glob_allowed(path, include_globs, exclude_globs):
                continue
            content, truncated = self._read_file_limited(path, max_file_bytes)
            sources.append(
                _LocationSource(
                    content=content,
                    label=path.name,
                    location_type=location_type,
                    source_path=str(path),
                    truncated=truncated,
                )
            )
            if len(sources) >= max_files:
                warnings.append(
                    f"max_files reached; stopped scanning directory {directory}."
                )
                break

        return sources

    def _glob_allowed(
        self, path: Path, include_globs: list[str], exclude_globs: list[str]
    ) -> bool:
        path_str = path.as_posix()
        if include_globs:
            matched = any(fnmatch.fnmatch(path_str, pattern) for pattern in include_globs)
            if not matched:
                return False
        if exclude_globs:
            if any(fnmatch.fnmatch(path_str, pattern) for pattern in exclude_globs):
                return False
        return True

    def _read_file_limited(
        self, path: Path, max_file_bytes: int
    ) -> tuple[str, bool]:
        if max_file_bytes <= 0:
            max_file_bytes = 1
        truncated = False
        try:
            size = path.stat().st_size
        except OSError:
            size = None
        if size is not None and size > max_file_bytes:
            truncated = True
        try:
            with path.open("rb") as handle:
                raw = handle.read(max_file_bytes)
            text = raw.decode("utf-8", errors="ignore")
        except OSError as exc:
            raise ToolError(f"Failed to read file {path}: {exc}") from exc
        return text, truncated

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
            raise ToolError(f"Path not found: {resolved}")
        return resolved

    def _format_location_blocks(self, blocks: list[LocationBlock]) -> str:
        if not blocks:
            return ""
        parts = ["Location context:"]
        for block in blocks:
            label = block.label or block.source_path or "location"
            type_suffix = f" | {block.location_type}" if block.location_type else ""
            trunc = " (truncated)" if block.truncated else ""
            header = f"[{label}{type_suffix}{trunc}]"
            parts.append(f"{header}\n{block.content}".strip())
        return "\n\n".join(parts).strip()

    def _inject_locations(
        self,
        messages: list[MultiLocationMessage],
        blocks: list[LocationBlock],
    ) -> list[MultiLocationMessage]:
        if not blocks:
            return messages
        location_text = self._format_location_blocks(blocks)
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].role == "user":
                content = messages[idx].content.rstrip()
                messages[idx].content = f"{content}\n\n{location_text}".strip()
                return messages
        messages.append(MultiLocationMessage(role="user", content=location_text))
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
        show_steps: bool,
        max_locations: int,
        prefix: str | None,
    ) -> str:
        show_steps_text = "yes" if show_steps else "no"
        rendered = self._render_template(
            template, style, show_steps_text, max_locations
        )
        if prefix and prefix.strip():
            return f"{prefix.strip()}\n\n{rendered}".strip()
        return rendered.strip()

    def _render_template(
        self,
        template: str,
        style: str,
        show_steps_text: str,
        max_locations: int,
    ) -> str:
        had_placeholders = (
            "{style}" in template
            or "{show_steps}" in template
            or "{max_locations}" in template
        )
        rendered = template
        if "{style}" in template:
            rendered = rendered.replace("{style}", style)
        if "{show_steps}" in template:
            rendered = rendered.replace("{show_steps}", show_steps_text)
        if "{max_locations}" in template:
            rendered = rendered.replace("{max_locations}", str(max_locations))

        if had_placeholders:
            return rendered

        extra = (
            f"Style: {style}\n"
            f"Show steps: {show_steps_text}\n"
            f"Max locations: {max_locations}"
        )
        return f"{rendered.rstrip()}\n\n{extra}"

    def _normalize_messages(
        self, args: MultiLocationProcessingArgs, system_prompt: str
    ) -> list[MultiLocationMessage]:
        messages: list[MultiLocationMessage] = []
        if args.messages:
            for msg in args.messages:
                role = (msg.role or "").strip()
                content = (msg.content or "").strip()
                if not content:
                    continue
                if not role:
                    role = "user"
                messages.append(MultiLocationMessage(role=role, content=content))
        elif args.prompt and args.prompt.strip():
            messages.append(
                MultiLocationMessage(role="user", content=args.prompt.strip())
            )
        else:
            raise ToolError("Provide prompt or messages.")

        if not messages:
            raise ToolError("No usable messages provided.")

        if system_prompt.strip():
            messages.insert(
                0, MultiLocationMessage(role="system", content=system_prompt.strip())
            )
        return messages

    def _call_llm(
        self,
        messages: list[MultiLocationMessage],
        args: MultiLocationProcessingArgs,
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

    def _resolve_model(self, args: MultiLocationProcessingArgs) -> str:
        return args.llm_model or self.config.llm_model

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, MultiLocationProcessingArgs):
            return ToolCallDisplay(summary="multi_location_processing")
        return ToolCallDisplay(
            summary="multi_location_processing",
            details={
                "style": event.args.style,
                "show_steps": event.args.show_steps,
                "max_locations": event.args.max_locations,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, MultiLocationProcessingResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = "Multi-location processing complete"
        if event.result.errors:
            message = "Multi-location processing finished with errors"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "answer": event.result.answer,
                "used_style": event.result.used_style,
                "show_steps": event.result.show_steps,
                "max_locations": event.result.max_locations,
                "template_source": event.result.template_source,
                "location_blocks": len(event.result.location_blocks),
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Multi-location processing"
