from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, Field

from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    ToolPermission,
)
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData

if TYPE_CHECKING:
    from vibe.core.types import ToolCallEvent, ToolResultEvent


DEFAULT_SKILLS_DIR = Path.home() / ".vibe" / "skills"


class SkillsListConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    skills_dir: Path = Field(
        default=DEFAULT_SKILLS_DIR,
        description="Directory containing skill files or skill folders.",
    )
    max_files: int = Field(
        default=0,
        description="Maximum number of skills to return. Use 0 for no limit.",
    )
    file_glob: str = Field(
        default="*.md",
        description="Glob pattern for root-level skill files.",
    )
    skill_filename: str = Field(
        default="SKILL.md",
        description="Skill file name for directory-based skills.",
    )
    include_subdirs: bool = Field(
        default=True,
        description="Whether to include skills stored as subdirectories.",
    )
    include_root_files: bool = Field(
        default=True,
        description="Whether to include root-level skill files.",
    )


class SkillsListState(BaseToolState):
    pass


class SkillsListArgs(BaseModel):
    pass


class SkillsListResult(BaseModel):
    skills: list[str]
    count: int


class SkillsList(
    BaseTool[SkillsListArgs, SkillsListResult, SkillsListConfig, SkillsListState],
    ToolUIData[SkillsListArgs, SkillsListResult],
):
    description: ClassVar[str] = (
        "List available skills from the skills folder (root files + subdirs)."
    )

    async def run(self, args: SkillsListArgs) -> SkillsListResult:
        skills_dir = self.config.skills_dir.expanduser().resolve()
        if not skills_dir.is_dir():
            return SkillsListResult(skills=[], count=0)

        skills: dict[str, Path] = {}

        if self.config.include_subdirs:
            for entry in skills_dir.iterdir():
                if not entry.is_dir():
                    continue
                skill_path = entry / self.config.skill_filename
                if skill_path.is_file():
                    skills.setdefault(entry.name, entry)

        if self.config.include_root_files:
            for path in skills_dir.glob(self.config.file_glob):
                if not path.is_file():
                    continue
                if path.name.lower() == self.config.skill_filename.lower():
                    continue
                skills.setdefault(path.stem, path)

        names = sorted(skills.keys(), key=lambda name: name.lower())
        if self.config.max_files > 0:
            names = names[: self.config.max_files]

        return SkillsListResult(skills=names, count=len(names))

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        return ToolCallDisplay(summary="skills_list")

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, SkillsListResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = f"Found {event.result.count} skills"
        return ToolResultDisplay(
            success=True,
            message=message,
            details={"skills": event.result.skills},
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Listing skills"
