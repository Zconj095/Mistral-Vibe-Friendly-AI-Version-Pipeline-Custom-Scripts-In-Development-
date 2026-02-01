from __future__ import annotations

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


DEFAULT_SKILLS_DIR = Path.home() / ".vibe" / "skills"


class SkillsReadConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    skills_dir: Path = Field(
        default=DEFAULT_SKILLS_DIR,
        description="Directory containing skill files or skill folders.",
    )
    max_bytes: int = Field(
        default=64_000,
        description="Maximum number of bytes to read from a skill file.",
    )
    file_extension: str = Field(
        default=".md",
        description="File extension for root-level skill files.",
    )
    skill_filename: str = Field(
        default="SKILL.md",
        description="Skill file name for directory-based skills.",
    )
    allow_subpath: bool = Field(
        default=True,
        description="Allow reading files inside a skill directory (e.g., references).",
    )


class SkillsReadState(BaseToolState):
    pass


class SkillsReadArgs(BaseModel):
    name: str = Field(description="Skill name (directory name or file stem).")
    subpath: str | None = Field(
        default=None,
        description="Optional relative path inside the skill directory.",
    )


class SkillsReadResult(BaseModel):
    name: str
    path: str
    content: str
    truncated: bool


class SkillsRead(
    BaseTool[SkillsReadArgs, SkillsReadResult, SkillsReadConfig, SkillsReadState],
    ToolUIData[SkillsReadArgs, SkillsReadResult],
):
    description: ClassVar[str] = (
        "Read a skill by name (root file or SKILL.md in a subdirectory)."
    )

    async def run(self, args: SkillsReadArgs) -> SkillsReadResult:
        name = args.name.strip()
        if not name:
            raise ToolError("Skill name is required.")
        if name in {".", ".."}:
            raise ToolError("Invalid skill name.")
        if Path(name).name != name:
            raise ToolError("Skill name must not include path separators.")

        extension = self._normalize_extension(self.config.file_extension)
        filename = name if name.endswith(extension) else f"{name}{extension}"

        skills_dir = self.config.skills_dir.expanduser().resolve()
        if not skills_dir.is_dir():
            raise ToolError(f"Skills directory not found: {skills_dir}")

        root_path = (skills_dir / filename).resolve()
        if root_path.parent == skills_dir and root_path.is_file():
            if args.subpath:
                raise ToolError("Subpath is not valid for root-level skills.")
            return self._read_path(name, root_path)

        skill_dir = (skills_dir / name).resolve()
        if skill_dir.parent != skills_dir or not skill_dir.is_dir():
            raise ToolError(f"Skill not found: {name}")

        if args.subpath:
            if not self.config.allow_subpath:
                raise ToolError("Subpath reading is disabled for skills.")
            target = self._resolve_subpath(skill_dir, args.subpath)
        else:
            target = (skill_dir / self.config.skill_filename).resolve()

        if skill_dir not in target.parents and target != skill_dir:
            raise ToolError("Skill path resolves outside the skill directory.")
        if not target.is_file():
            raise ToolError(f"Skill file not found: {target}")

        return self._read_path(name, target)

    def _read_path(self, name: str, path: Path) -> SkillsReadResult:
        data = path.read_bytes()
        truncated = len(data) > self.config.max_bytes if self.config.max_bytes > 0 else False
        if truncated:
            data = data[: self.config.max_bytes]

        content = data.decode("utf-8", errors="ignore")
        return SkillsReadResult(
            name=name, path=str(path), content=content, truncated=truncated
        )

    def _resolve_subpath(self, skill_dir: Path, subpath: str) -> Path:
        cleaned = subpath.strip()
        if not cleaned:
            raise ToolError("Subpath cannot be empty.")
        rel = Path(cleaned)
        if rel.is_absolute() or ".." in rel.parts:
            raise ToolError("Subpath must be a relative path within the skill folder.")
        target = (skill_dir / rel).resolve()
        return target

    @classmethod
    def _normalize_extension(cls, value: str) -> str:
        cleaned = value.strip() or ".md"
        return cleaned if cleaned.startswith(".") else f".{cleaned}"

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        summary = "skills_read"
        if hasattr(event, "args") and getattr(event.args, "name", None):
            summary = f"skills_read: {event.args.name}"
            if getattr(event.args, "subpath", None):
                summary += f" ({event.args.subpath})"
        return ToolCallDisplay(summary=summary)

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, SkillsReadResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = f"Loaded skill: {event.result.name}"
        warnings = ["Skill content was truncated."] if event.result.truncated else []
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=warnings,
            details={
                "name": event.result.name,
                "path": event.result.path,
                "truncated": event.result.truncated,
                "content": event.result.content,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Reading skill"
