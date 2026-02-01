from __future__ import annotations

import difflib
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


class DiffUnifiedArgs(BaseModel):
    path: str = Field(description="Base file path.")
    new_content: str | None = Field(
        default=None, description="New content to diff against."
    )
    compare_path: str | None = Field(
        default=None, description="Optional second file path to diff against."
    )
    context_lines: int = Field(
        default=3, description="Number of context lines."
    )
    max_diff_chars: int = Field(
        default=200_000, description="Maximum diff characters to return."
    )


class DiffUnifiedResult(BaseModel):
    path: str
    compare_path: str | None
    diff: str
    changed: bool
    lines_added: int
    lines_removed: int
    truncated: bool


class DiffUnifiedConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS


class DiffUnifiedState(BaseToolState):
    pass


class DiffUnified(
    BaseTool[DiffUnifiedArgs, DiffUnifiedResult, DiffUnifiedConfig, DiffUnifiedState],
    ToolUIData[DiffUnifiedArgs, DiffUnifiedResult],
):
    description: ClassVar[str] = "Generate a unified diff for a file."

    async def run(self, args: DiffUnifiedArgs) -> DiffUnifiedResult:
        base = self.config.effective_workdir.resolve()
        path = self._resolve_file(args.path, base)

        if args.compare_path:
            compare_path = self._resolve_file(args.compare_path, base)
            new_text = compare_path.read_text(encoding="utf-8", errors="ignore")
        elif args.new_content is not None:
            compare_path = None
            new_text = args.new_content
        else:
            raise ToolError("Provide new_content or compare_path.")

        original_text = path.read_text(encoding="utf-8", errors="ignore")

        original_lines = original_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)

        diff_lines = list(
            difflib.unified_diff(
                original_lines,
                new_lines,
                fromfile=str(path),
                tofile=str(compare_path) if compare_path else "new",
                lineterm="",
                n=max(0, args.context_lines),
            )
        )

        lines_added = 0
        lines_removed = 0
        for line in diff_lines:
            if line.startswith("+++ ") or line.startswith("--- "):
                continue
            if line.startswith("+"):
                lines_added += 1
            elif line.startswith("-"):
                lines_removed += 1

        diff_text = "".join(diff_lines)
        truncated = False
        if args.max_diff_chars > 0 and len(diff_text) > args.max_diff_chars:
            diff_text = diff_text[: args.max_diff_chars] + "\n...(diff truncated)"
            truncated = True

        return DiffUnifiedResult(
            path=str(path),
            compare_path=str(compare_path) if compare_path else None,
            diff=diff_text,
            changed=bool(diff_text.strip()),
            lines_added=lines_added,
            lines_removed=lines_removed,
            truncated=truncated,
        )

    def _resolve_file(self, raw: str, base: Path) -> Path:
        if not raw or not raw.strip():
            raise ToolError("Path cannot be empty.")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = base / path
        path = path.resolve()
        try:
            path.relative_to(base)
        except ValueError as exc:
            raise ToolError(f"Path must stay within project root: {base}") from exc
        if not path.exists():
            raise ToolError(f"File not found: {path}")
        if not path.is_file():
            raise ToolError(f"Path is not a file: {path}")
        return path

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, DiffUnifiedArgs):
            return ToolCallDisplay(summary="diff_unified")
        return ToolCallDisplay(
            summary=f"diff_unified: {event.args.path}",
            details={
                "compare_path": event.args.compare_path,
                "context_lines": event.args.context_lines,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, DiffUnifiedResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = "Diff generated"
        if not event.result.changed:
            message = "No changes detected"
        warnings = ["Diff truncated"] if event.result.truncated else []
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=warnings,
            details={
                "diff": event.result.diff,
                "lines_added": event.result.lines_added,
                "lines_removed": event.result.lines_removed,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Generating diff"
