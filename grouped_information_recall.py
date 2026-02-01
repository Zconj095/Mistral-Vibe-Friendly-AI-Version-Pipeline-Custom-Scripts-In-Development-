from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from datetime import datetime
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


DEFAULT_PROMPT_TEMPLATE = """### GROUPED INFORMATION RECALL MODE (OPT-IN)
Group items into multiple categories and prepare recall cues.
- Provide clear group labels and brief descriptions.
- Include recall hints so groups can be retrieved later.
- Keep internal grouping private; do not mention this mode in the response.
- If the user asks for steps or show_steps is enabled, give a concise outline.

Grouping style: {grouping_style}
Show steps: {show_steps}
Max groups: {max_groups}
"""

TOOL_PROMPT = (
    "Use `grouped_information_recall` to group many groups of information and recall them "
    "later. Provide `prompt` or `messages` and optionally `items`, `categories`, "
    "`memory_key`, and `mode`."
)


class GroupedInformationRecallMessage(BaseModel):
    role: str
    content: str


class GroupedInformationRecallArgs(BaseModel):
    mode: str | None = Field(
        default="group",
        description="Mode: group or recall.",
    )
    prompt: str | None = Field(
        default=None, description="User prompt to solve or recall query."
    )
    messages: list[GroupedInformationRecallMessage] | None = Field(
        default=None, description="Optional chat messages."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prefix."
    )
    items: list[str] | None = Field(
        default=None, description="Items to group."
    )
    categories: list[str] | None = Field(
        default=None, description="Optional categories to group by."
    )
    grouping_style: str | None = Field(
        default=None, description="Grouping style label."
    )
    show_steps: bool | None = Field(
        default=None, description="Whether to include a step outline."
    )
    max_groups: int | None = Field(
        default=None, description="Maximum groups in the outline."
    )
    memory_key: str | None = Field(
        default=None, description="Memory key for storing or recalling groups."
    )
    store_path: str | None = Field(
        default=None, description="Optional path to store or load group data."
    )
    save: bool | None = Field(
        default=True, description="Whether to save grouped data."
    )
    recall_query: str | None = Field(
        default=None, description="Query to answer using stored groups."
    )
    include_raw: bool | None = Field(
        default=False, description="Include stored payload in the result."
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
        default=800, description="LLM max tokens."
    )
    llm_stream: bool = Field(
        default=False, description="Stream LLM tokens."
    )


class GroupedInformationRecallResult(BaseModel):
    answer: str
    system_prompt: str
    messages: list[GroupedInformationRecallMessage]
    mode: str
    used_grouping_style: str
    show_steps: bool
    max_groups: int
    template_source: str
    warnings: list[str]
    errors: list[str]
    llm_model: str | None
    memory_key: str | None
    store_path: str | None
    stored_data: dict | None


class GroupedInformationRecallConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    llm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL.",
    )
    llm_model: str = Field(
        default="gpt-oss:latest", description="Default LLM model name."
    )
    default_grouping_style: str = Field(
        default="hierarchical",
        description="Default grouping style.",
    )
    default_show_steps: bool = Field(
        default=False, description="Default for show_steps."
    )
    default_max_groups: int = Field(
        default=8, description="Default max groups."
    )
    memory_dir: Path = Field(
        default=Path.home() / ".vibe" / "grouped_memory",
        description="Directory for stored grouped information.",
    )
    prompt_path: Path | None = Field(
        default=Path.home()
        / "mistral-vibe"
        / "vibe"
        / "core"
        / "prompts"
        / "grouped_information_recall.md",
        description="Optional path to a prompt template.",
    )
    prompt_max_chars: int = Field(
        default=8000, description="Maximum template characters to load."
    )


class GroupedInformationRecallState(BaseToolState):
    pass


class GroupedInformationRecall(
    BaseTool[
        GroupedInformationRecallArgs,
        GroupedInformationRecallResult,
        GroupedInformationRecallConfig,
        GroupedInformationRecallState,
    ],
    ToolUIData[GroupedInformationRecallArgs, GroupedInformationRecallResult],
):
    description: ClassVar[str] = (
        "Group information into multiple categories and enable recall from stored data."
    )

    @classmethod
    def get_tool_prompt(cls) -> str | None:
        return TOOL_PROMPT

    async def run(
        self, args: GroupedInformationRecallArgs
    ) -> GroupedInformationRecallResult:
        warnings: list[str] = []
        errors: list[str] = []

        mode = (args.mode or "group").strip().lower()
        if mode not in {"group", "recall"}:
            raise ToolError("mode must be group or recall.")

        grouping_style = (args.grouping_style or self.config.default_grouping_style).strip()
        if not grouping_style:
            grouping_style = "hierarchical"

        show_steps = (
            args.show_steps
            if args.show_steps is not None
            else self.config.default_show_steps
        )
        max_groups = (
            args.max_groups
            if args.max_groups is not None
            else self.config.default_max_groups
        )
        if max_groups <= 0:
            raise ToolError("max_groups must be positive.")

        self._validate_llm_settings(args)

        template, template_source = self._load_template(warnings)
        system_prompt = self._build_system_prompt(
            template, grouping_style, show_steps, max_groups, args.system_prompt
        )

        if mode == "group":
            messages = self._normalize_messages(args, system_prompt)
            self._append_items_message(messages, args.items, args.categories)
            answer = self._call_llm(messages, args)
            payload = self._build_payload(answer, args, grouping_style, max_groups)
            memory_key = self._resolve_memory_key(args.memory_key)
            store_path = None
            if args.save:
                store_path = self._resolve_store_path(args, memory_key)
                self._save_payload(store_path, payload, warnings)
            return GroupedInformationRecallResult(
                answer=answer,
                system_prompt=system_prompt,
                messages=messages,
                mode=mode,
                used_grouping_style=grouping_style,
                show_steps=show_steps,
                max_groups=max_groups,
                template_source=template_source,
                warnings=warnings,
                errors=errors,
                llm_model=self._resolve_model(args),
                memory_key=memory_key if args.save or args.memory_key else None,
                store_path=str(store_path) if store_path else None,
                stored_data=payload if args.include_raw else None,
            )

        store_path = self._resolve_store_path(args, args.memory_key, require=True)
        payload = self._load_payload(store_path)
        recall_query = args.recall_query or args.prompt
        messages: list[GroupedInformationRecallMessage] = []
        answer = payload.get("answer", "")
        if recall_query:
            messages = self._build_recall_messages(system_prompt, recall_query, payload)
            answer = self._call_llm(messages, args)
        if not answer:
            answer = json.dumps(payload, indent=2)
        return GroupedInformationRecallResult(
            answer=answer,
            system_prompt=system_prompt,
            messages=messages,
            mode=mode,
            used_grouping_style=grouping_style,
            show_steps=show_steps,
            max_groups=max_groups,
            template_source=template_source,
            warnings=warnings,
            errors=errors,
            llm_model=self._resolve_model(args),
            memory_key=args.memory_key,
            store_path=str(store_path),
            stored_data=payload if args.include_raw else None,
        )

    def _build_recall_messages(
        self,
        system_prompt: str,
        recall_query: str,
        payload: dict,
    ) -> list[GroupedInformationRecallMessage]:
        messages: list[GroupedInformationRecallMessage] = []
        if system_prompt.strip():
            messages.append(
                GroupedInformationRecallMessage(
                    role="system", content=system_prompt.strip()
                )
            )
        messages.append(
            GroupedInformationRecallMessage(role="user", content=recall_query.strip())
        )
        payload_text = json.dumps(payload, indent=2)
        messages.append(
            GroupedInformationRecallMessage(
                role="user",
                content=f"Stored groups:\n{payload_text}",
            )
        )
        return messages

    def _append_items_message(
        self,
        messages: list[GroupedInformationRecallMessage],
        items: list[str] | None,
        categories: list[str] | None,
    ) -> None:
        parts: list[str] = []
        if items:
            parts.append("Items:")
            parts.extend(f"- {item}" for item in items if item.strip())
        if categories:
            parts.append("Categories:")
            parts.extend(f"- {category}" for category in categories if category.strip())
        if parts:
            messages.append(
                GroupedInformationRecallMessage(role="user", content="\n".join(parts))
            )

    def _build_payload(
        self,
        answer: str,
        args: GroupedInformationRecallArgs,
        grouping_style: str,
        max_groups: int,
    ) -> dict:
        return {
            "version": 1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "grouping_style": grouping_style,
            "max_groups": max_groups,
            "items": args.items or [],
            "categories": args.categories or [],
            "prompt": (args.prompt or "").strip(),
            "answer": answer,
        }

    def _resolve_memory_key(self, key: str | None) -> str:
        if key and key.strip():
            return self._sanitize_key(key.strip())
        return datetime.now().strftime("grouped_%Y%m%d_%H%M%S")

    def _sanitize_key(self, key: str) -> str:
        sanitized = []
        for ch in key:
            if ch.isalnum() or ch in {"-", "_"}:
                sanitized.append(ch)
            else:
                sanitized.append("_")
        return "".join(sanitized).strip("_") or "grouped"

    def _resolve_store_path(
        self,
        args: GroupedInformationRecallArgs,
        memory_key: str | None,
        require: bool = False,
    ) -> Path:
        if args.store_path:
            path = Path(args.store_path).expanduser()
            if not path.is_absolute():
                path = self.config.effective_workdir / path
            return path.resolve()
        if memory_key:
            self.config.memory_dir.mkdir(parents=True, exist_ok=True)
            return (self.config.memory_dir / f"{memory_key}.json").resolve()
        if require:
            raise ToolError("Provide memory_key or store_path for recall.")
        raise ToolError("store_path or memory_key is required.")

    def _save_payload(
        self, path: Path, payload: dict, warnings: list[str]
    ) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), "utf-8")
        except OSError as exc:
            warnings.append(f"Failed to save grouped data: {exc}")

    def _load_payload(self, path: Path) -> dict:
        try:
            text = path.read_text("utf-8")
        except OSError as exc:
            raise ToolError(f"Failed to read grouped data: {exc}") from exc
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ToolError(f"Invalid grouped data JSON: {exc}") from exc

    def _validate_llm_settings(self, args: GroupedInformationRecallArgs) -> None:
        if args.llm_temperature < 0:
            raise ToolError("llm_temperature cannot be negative.")
        if args.llm_max_tokens <= 0:
            raise ToolError("llm_max_tokens must be positive.")

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
        grouping_style: str,
        show_steps: bool,
        max_groups: int,
        prefix: str | None,
    ) -> str:
        show_steps_text = "yes" if show_steps else "no"
        rendered = self._render_template(
            template, grouping_style, show_steps_text, max_groups
        )
        if prefix and prefix.strip():
            return f"{prefix.strip()}\n\n{rendered}".strip()
        return rendered.strip()

    def _render_template(
        self,
        template: str,
        grouping_style: str,
        show_steps_text: str,
        max_groups: int,
    ) -> str:
        had_placeholders = (
            "{grouping_style}" in template
            or "{show_steps}" in template
            or "{max_groups}" in template
        )
        rendered = template
        if "{grouping_style}" in template:
            rendered = rendered.replace("{grouping_style}", grouping_style)
        if "{show_steps}" in template:
            rendered = rendered.replace("{show_steps}", show_steps_text)
        if "{max_groups}" in template:
            rendered = rendered.replace("{max_groups}", str(max_groups))

        if had_placeholders:
            return rendered

        extra = (
            f"Grouping style: {grouping_style}\n"
            f"Show steps: {show_steps_text}\n"
            f"Max groups: {max_groups}"
        )
        return f"{rendered.rstrip()}\n\n{extra}"

    def _normalize_messages(
        self, args: GroupedInformationRecallArgs, system_prompt: str
    ) -> list[GroupedInformationRecallMessage]:
        messages: list[GroupedInformationRecallMessage] = []
        if args.messages:
            for msg in args.messages:
                role = (msg.role or "").strip()
                content = (msg.content or "").strip()
                if not content:
                    continue
                if not role:
                    role = "user"
                messages.append(
                    GroupedInformationRecallMessage(role=role, content=content)
                )
        elif args.prompt and args.prompt.strip():
            messages.append(
                GroupedInformationRecallMessage(
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
                GroupedInformationRecallMessage(
                    role="system", content=system_prompt.strip()
                ),
            )
        return messages

    def _call_llm(
        self,
        messages: list[GroupedInformationRecallMessage],
        args: GroupedInformationRecallArgs,
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

    def _resolve_model(self, args: GroupedInformationRecallArgs) -> str:
        return args.llm_model or self.config.llm_model

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, GroupedInformationRecallArgs):
            return ToolCallDisplay(summary="grouped_information_recall")
        return ToolCallDisplay(
            summary="grouped_information_recall",
            details={
                "mode": event.args.mode,
                "grouping_style": event.args.grouping_style,
                "memory_key": event.args.memory_key,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, GroupedInformationRecallResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = "Grouped information recall complete"
        if event.result.errors:
            message = "Grouped information recall finished with errors"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "mode": event.result.mode,
                "memory_key": event.result.memory_key,
                "store_path": event.result.store_path,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Grouped information recall"
