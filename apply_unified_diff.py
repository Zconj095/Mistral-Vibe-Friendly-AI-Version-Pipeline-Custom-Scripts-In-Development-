from __future__ import annotations

import re
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


HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass(frozen=True)
class _Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str]


class ApplyUnifiedDiffArgs(BaseModel):
    path: str = Field(description="Target file path.")
    diff: str = Field(description="Unified diff text.")
    dry_run: bool = Field(
        default=False, description="Validate patch without writing."
    )
    allow_create: bool = Field(
        default=False, description="Allow creating a new file if missing."
    )
    create_backup: bool | None = Field(
        default=None, description="Override backup behavior."
    )


class ApplyUnifiedDiffResult(BaseModel):
    path: str
    applied: bool
    dry_run: bool
    hunks: int
    lines_added: int
    lines_removed: int
    warnings: list[str]


class ApplyUnifiedDiffConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    max_diff_bytes: int = Field(
        default=200_000, description="Maximum diff bytes to accept."
    )
    create_backup: bool = Field(
        default=False, description="Create .bak files before writing."
    )


class ApplyUnifiedDiffState(BaseToolState):
    pass


class ApplyUnifiedDiff(
    BaseTool[
        ApplyUnifiedDiffArgs,
        ApplyUnifiedDiffResult,
        ApplyUnifiedDiffConfig,
        ApplyUnifiedDiffState,
    ],
    ToolUIData[ApplyUnifiedDiffArgs, ApplyUnifiedDiffResult],
):
    description: ClassVar[str] = "Apply a unified diff to a file."

    async def run(self, args: ApplyUnifiedDiffArgs) -> ApplyUnifiedDiffResult:
        if not args.diff.strip():
            raise ToolError("Diff cannot be empty.")

        diff_bytes = len(args.diff.encode("utf-8"))
        if self.config.max_diff_bytes > 0 and diff_bytes > self.config.max_diff_bytes:
            raise ToolError("Diff exceeds max_diff_bytes limit.")

        base = self.config.effective_workdir.resolve()
        path = self._resolve_path(args.path, base)

        if path.exists():
            content = path.read_text(encoding="utf-8", errors="ignore")
            ends_with_newline = content.endswith(("\n", "\r\n"))
            newline = "\r\n" if "\r\n" in content else "\n"
        else:
            if not args.allow_create:
                raise ToolError(f"File not found: {path}")
            content = ""
            ends_with_newline = True
            newline = "\n"

        hunks, added, removed, saw_no_newline = self._parse_diff(args.diff)
        if not hunks:
            raise ToolError("No hunks found in diff.")

        if saw_no_newline:
            ends_with_newline = False

        original_lines = content.splitlines()
        new_lines = self._apply_hunks(original_lines, hunks)

        new_content = newline.join(new_lines)
        if ends_with_newline and new_content:
            new_content += newline

        applied = new_content != content
        if applied and not args.dry_run:
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
            if args.create_backup or (
                args.create_backup is None and self.config.create_backup
            ):
                self._write_backup(path, content)
            path.write_text(new_content, encoding="utf-8")

        return ApplyUnifiedDiffResult(
            path=str(path),
            applied=applied,
            dry_run=args.dry_run,
            hunks=len(hunks),
            lines_added=added,
            lines_removed=removed,
            warnings=[],
        )

    def _resolve_path(self, raw: str, base: Path) -> Path:
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
        return path

    def _parse_diff(self, diff: str) -> tuple[list[_Hunk], int, int, bool]:
        lines = diff.splitlines()
        hunks: list[_Hunk] = []
        current: list[str] | None = None
        added = 0
        removed = 0
        saw_no_newline = False
        old_start = old_count = new_start = new_count = 0

        for line in lines:
            if line.startswith("\\ No newline at end of file"):
                saw_no_newline = True
                continue
            if line.startswith("@@"):
                if current is not None:
                    hunks.append(
                        _Hunk(
                            old_start=old_start,
                            old_count=old_count,
                            new_start=new_start,
                            new_count=new_count,
                            lines=current,
                        )
                    )
                match = HUNK_RE.match(line)
                if not match:
                    raise ToolError(f"Invalid hunk header: {line}")
                old_start = int(match.group(1))
                old_count = int(match.group(2) or "1")
                new_start = int(match.group(3))
                new_count = int(match.group(4) or "1")
                current = []
                continue
            if current is None:
                continue
            if not line:
                current.append(" ")
                continue
            prefix = line[0]
            if prefix == "+":
                if not line.startswith("+++ "):
                    added += 1
            elif prefix == "-":
                if not line.startswith("--- "):
                    removed += 1
            current.append(line)

        if current is not None:
            hunks.append(
                _Hunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    lines=current,
                )
            )

        return hunks, added, removed, saw_no_newline

    def _apply_hunks(self, lines: list[str], hunks: list[_Hunk]) -> list[str]:
        new_lines: list[str] = []
        src_index = 0

        for idx, hunk in enumerate(hunks, start=1):
            hunk_start = max(hunk.old_start - 1, 0)
            if hunk_start < src_index:
                raise ToolError(
                    f"Hunk {idx} overlaps previous changes at line {hunk_start + 1}."
                )
            if hunk_start > len(lines):
                raise ToolError(
                    f"Hunk {idx} starts beyond EOF at line {hunk_start + 1}."
                )

            new_lines.extend(lines[src_index:hunk_start])
            src_index = hunk_start

            for hline in hunk.lines:
                prefix = hline[0] if hline else ""
                text = hline[1:] if len(hline) > 1 else ""
                if prefix == " ":
                    self._expect_line(lines, src_index, text, idx)
                    new_lines.append(lines[src_index])
                    src_index += 1
                elif prefix == "-":
                    self._expect_line(lines, src_index, text, idx)
                    src_index += 1
                elif prefix == "+":
                    new_lines.append(text)
                elif prefix == "\\":
                    continue
                else:
                    raise ToolError(f"Hunk {idx} has invalid line prefix: {hline}")

        new_lines.extend(lines[src_index:])
        return new_lines

    def _expect_line(self, lines: list[str], index: int, text: str, hunk: int) -> None:
        if index >= len(lines):
            raise ToolError(f"Hunk {hunk} expects line {index + 1}, got EOF.")
        if lines[index] != text:
            context = self._format_context(lines, index)
            raise ToolError(
                f"Hunk {hunk} mismatch at line {index + 1}. "
                f"Expected {text!r}, got {lines[index]!r}.\n{context}"
            )

    @staticmethod
    def _format_context(lines: list[str], index: int, radius: int = 2) -> str:
        start = max(0, index - radius)
        end = min(len(lines), index + radius + 1)
        context_lines = []
        for i in range(start, end):
            marker = ">>>" if i == index else "   "
            context_lines.append(f"{marker} {i + 1:4d}: {lines[i]}")
        return "\n".join(context_lines)

    @staticmethod
    def _write_backup(path: Path, content: str) -> None:
        backup_path = path.with_suffix(path.suffix + ".bak")
        backup_path.write_text(content, encoding="utf-8")

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ApplyUnifiedDiffArgs):
            return ToolCallDisplay(summary="apply_unified_diff")
        return ToolCallDisplay(
            summary=f"apply_unified_diff: {event.args.path}",
            details={
                "dry_run": event.args.dry_run,
                "allow_create": event.args.allow_create,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ApplyUnifiedDiffResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = "Patch applied"
        if event.result.dry_run:
            message = "Patch validated"
        if not event.result.applied:
            message = "Patch resulted in no changes"
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=event.result.warnings,
            details={
                "hunks": event.result.hunks,
                "lines_added": event.result.lines_added,
                "lines_removed": event.result.lines_removed,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Applying patch"
