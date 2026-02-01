from __future__ import annotations

from pathlib import Path
import re
from typing import ClassVar, final

from pydantic import BaseModel, Field
import tomli_w

from vibe.core.config_path import AGENT_DIR, PROMPT_DIR
from vibe.core.tools.base import BaseTool, BaseToolConfig, BaseToolState, ToolError
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from vibe.core.types import ToolCallEvent, ToolResultEvent

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class CreateGptOssModelArgs(BaseModel):
    name: str = Field(description="Name of the custom GPT model profile.")
    system_prompt: str | None = Field(
        default=None,
        description="System prompt text for the custom GPT model.",
    )
    instructions: str | None = Field(
        default=None,
        description="Additional instructions appended to the system prompt.",
    )
    base_model: str | None = Field(
        default=None,
        description="Model alias to use (defaults to gpt-oss).",
    )
    overwrite: bool = Field(
        default=False,
        description="Overwrite existing agent/prompt files if they already exist.",
    )


class CreateGptOssModelResult(BaseModel):
    name: str
    agent_path: str
    prompt_path: str
    base_model: str
    instructions_set: bool
    system_prompt_bytes: int


class CreateGptOssModelConfig(BaseToolConfig):
    default_base_model: str = "gpt-oss"
    max_name_length: int = 64
    max_prompt_bytes: int = 200_000
    max_instructions_bytes: int = 50_000


class CreateGptOssModelState(BaseToolState):
    pass


class CreateGptOssModel(
    BaseTool[
        CreateGptOssModelArgs,
        CreateGptOssModelResult,
        CreateGptOssModelConfig,
        CreateGptOssModelState,
    ],
    ToolUIData[CreateGptOssModelArgs, CreateGptOssModelResult],
):
    description: ClassVar[str] = (
        "Create a custom GPT-OSS profile (agent + system prompt) in ~/.vibe."
    )

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, CreateGptOssModelArgs):
            return ToolCallDisplay(summary="create_gptoss_model")

        return ToolCallDisplay(
            summary=f"Creating GPT-OSS profile: {event.args.name}",
            details={
                "name": event.args.name,
                "base_model": event.args.base_model,
                "overwrite": event.args.overwrite,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if isinstance(event.result, CreateGptOssModelResult):
            return ToolResultDisplay(
                success=True,
                message=f"Created GPT-OSS profile: {event.result.name}",
                details={
                    "agent_path": event.result.agent_path,
                    "prompt_path": event.result.prompt_path,
                    "base_model": event.result.base_model,
                },
            )
        return ToolResultDisplay(success=True, message="Profile created")

    @classmethod
    def get_status_text(cls) -> str:
        return "Creating GPT-OSS profile"

    @final
    async def run(self, args: CreateGptOssModelArgs) -> CreateGptOssModelResult:
        name = self._normalize_name(args.name)
        base_model = (args.base_model or self.config.default_base_model).strip()

        prompt_text = self._build_prompt_text(name, args.system_prompt)
        prompt_bytes = len(prompt_text.encode("utf-8"))
        self._enforce_prompt_limits(prompt_bytes)

        instructions = self._normalize_instructions(args.instructions)
        if instructions is not None:
            self._enforce_instruction_limits(instructions)

        agent_path = AGENT_DIR.path / f"{name}.toml"
        prompt_path = PROMPT_DIR.path / f"{name}.md"

        if not args.overwrite:
            if agent_path.exists():
                raise ToolError(f"Agent profile already exists: {agent_path}")
            if prompt_path.exists():
                raise ToolError(f"Prompt file already exists: {prompt_path}")

        AGENT_DIR.path.mkdir(parents=True, exist_ok=True)
        PROMPT_DIR.path.mkdir(parents=True, exist_ok=True)

        prompt_path.write_text(prompt_text, encoding="utf-8")

        config_payload: dict[str, object] = {
            "active_model": base_model,
            "system_prompt_id": name,
        }
        if instructions:
            config_payload["instructions"] = instructions

        with agent_path.open("wb") as f:
            tomli_w.dump(config_payload, f)

        return CreateGptOssModelResult(
            name=name,
            agent_path=str(agent_path),
            prompt_path=str(prompt_path),
            base_model=base_model,
            instructions_set=bool(instructions),
            system_prompt_bytes=prompt_bytes,
        )

    def _normalize_name(self, raw_name: str) -> str:
        name = raw_name.strip()
        if not name:
            raise ToolError("Name cannot be empty.")
        if "/" in name or "\\" in name:
            raise ToolError("Name cannot contain path separators.")
        if len(name) > self.config.max_name_length:
            raise ToolError(
                f"Name exceeds {self.config.max_name_length} character limit."
            )
        if not NAME_RE.match(name):
            raise ToolError(
                "Name must start with a letter or number and only contain "
                "letters, numbers, hyphens, or underscores."
            )
        return name

    def _build_prompt_text(self, name: str, prompt: str | None) -> str:
        if prompt is None or not prompt.strip():
            prompt = (
                f"You are a custom GPT model named '{name}'.\n"
                "Follow the user's instructions carefully."
            )
        return prompt.strip() + "\n"

    def _normalize_instructions(self, instructions: str | None) -> str | None:
        if instructions is None:
            return None
        instructions = instructions.strip()
        return instructions if instructions else None

    def _enforce_prompt_limits(self, prompt_bytes: int) -> None:
        max_bytes = self.config.max_prompt_bytes
        if max_bytes > 0 and prompt_bytes > max_bytes:
            raise ToolError(
                f"System prompt is {prompt_bytes} bytes, exceeds {max_bytes} bytes."
            )

    def _enforce_instruction_limits(self, instructions: str) -> None:
        max_bytes = self.config.max_instructions_bytes
        if max_bytes <= 0:
            return
        size = len(instructions.encode("utf-8"))
        if size > max_bytes:
            raise ToolError(
                f"Instructions are {size} bytes, exceeds {max_bytes} bytes."
            )
