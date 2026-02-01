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


class ReadFileFullConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_read_bytes: int = Field(
        default=120_000,
        description="Maximum file size (bytes) allowed for a single-chunk read.",
    )


class ReadFileFullState(BaseToolState):
    pass


class ReadFileFullArgs(BaseModel):
    path: str
    max_bytes: int | None = Field(
        default=None,
        description="Override the configured max_read_bytes for this call.",
    )


class ReadFileFullResult(BaseModel):
    path: str
    content: str
    bytes_total: int


class ReadFileFull(
    BaseTool[ReadFileFullArgs, ReadFileFullResult, ReadFileFullConfig, ReadFileFullState],
    ToolUIData[ReadFileFullArgs, ReadFileFullResult],
):
    description: ClassVar[str] = (
        "Read an entire UTF-8 file in a single chunk. Fails if the file exceeds "
        "the configured size limit."
    )

    async def run(self, args: ReadFileFullArgs) -> ReadFileFullResult:
        path = self._resolve_path(args.path)
        max_bytes = self._resolve_max_bytes(args.max_bytes)
        file_size = path.stat().st_size

        if max_bytes > 0 and file_size > max_bytes:
            raise ToolError(
                f"File is {file_size} bytes, which exceeds the single-chunk limit "
                f"of {max_bytes} bytes."
            )

        data = path.read_bytes()
        content = data.decode("utf-8", errors="ignore")
        return ReadFileFullResult(
            path=str(path),
            content=content,
            bytes_total=file_size,
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

    def _resolve_max_bytes(self, override: int | None) -> int:
        max_bytes = override if override is not None else self.config.max_read_bytes
        return max_bytes

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ReadFileFullArgs):
            return ToolCallDisplay(summary="read_file_full")

        summary = f"read_file_full: {event.args.path}"
        if event.args.max_bytes:
            summary += f" (max {event.args.max_bytes} bytes)"
        return ToolCallDisplay(
            summary=summary,
            details={
                "path": event.args.path,
                "max_bytes": event.args.max_bytes,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ReadFileFullResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        return ToolResultDisplay(
            success=True,
            message=f"Read {event.result.bytes_total} bytes from {Path(event.result.path).name}",
            details={
                "path": event.result.path,
                "bytes_total": event.result.bytes_total,
                "content": event.result.content,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Reading full file"
