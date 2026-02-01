from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from vibe.core.types import ToolCallEvent, ToolResultEvent


class LsArgs(BaseModel):
    path: str | None = Field(
        default=None, description="Directory to list (defaults to workdir)."
    )
    recursive: bool = Field(default=False, description="List files recursively.")
    max_entries: int = Field(
        default=2000, description="Maximum number of entries to return."
    )
    include_hidden: bool = Field(
        default=False, description="Include dotfiles and dotdirs."
    )


class LsEntry(BaseModel):
    name: str
    path: str
    is_dir: bool
    size: int | None = None


class LsResult(BaseModel):
    path: str
    entries: list[LsEntry]
    total_entries: int
    truncated: bool


class LsConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_entries: int = Field(default=2000, description="Maximum entries to return.")


class LsState(BaseToolState):
    pass


class Ls(BaseTool[LsArgs, LsResult, LsConfig, LsState], ToolUIData[LsArgs, LsResult]):
    description: ClassVar[str] = "List directory contents."

    async def run(self, args: LsArgs) -> LsResult:
        target = Path(args.path).expanduser() if args.path else None
        if target is None:
            target = self.config.effective_workdir
        if not target.is_absolute():
            target = (self.config.effective_workdir / target).resolve()
        if not target.exists():
            raise ToolError(f"Path not found: {target}")
        if not target.is_dir():
            raise ToolError(f"Path is not a directory: {target}")

        max_entries = args.max_entries if args.max_entries > 0 else self.config.max_entries
        if max_entries <= 0:
            max_entries = self.config.max_entries

        entries: list[LsEntry] = []
        truncated = False
        iterator = target.rglob("*") if args.recursive else target.iterdir()
        for entry in iterator:
            if not args.include_hidden and entry.name.startswith("."):
                continue
            try:
                is_dir = entry.is_dir()
            except OSError:
                is_dir = False
            size = None
            if not is_dir:
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = None
            entries.append(
                LsEntry(name=entry.name, path=str(entry), is_dir=is_dir, size=size)
            )
            if len(entries) >= max_entries:
                truncated = True
                break

        return LsResult(
            path=str(target),
            entries=entries,
            total_entries=len(entries),
            truncated=truncated,
        )

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, LsArgs):
            return ToolCallDisplay(summary="ls")
        summary = f"ls: {event.args.path or '.'}"
        return ToolCallDisplay(
            summary=summary,
            details={
                "recursive": event.args.recursive,
                "max_entries": event.args.max_entries,
                "include_hidden": event.args.include_hidden,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, LsResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = f"Listed {event.result.total_entries} entry(s)"
        if event.result.truncated:
            message += " (truncated)"
        return ToolResultDisplay(
            success=True,
            message=message,
            details={
                "path": event.result.path,
                "entries": event.result.entries,
                "total_entries": event.result.total_entries,
                "truncated": event.result.truncated,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Listing directory"
