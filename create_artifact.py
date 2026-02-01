from __future__ import annotations

from pathlib import Path
import re
from typing import ClassVar, final

from pydantic import BaseModel, Field, field_validator

from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from vibe.core.types import ToolCallEvent, ToolResultEvent

NAME_RE = re.compile(r"[A-Za-z0-9_-]+")
EXT_RE = re.compile(r"^[A-Za-z0-9]+$")


class CreateArtifactArgs(BaseModel):
    name: str = Field(description="Artifact name.")
    content: str = Field(description="Artifact content.")
    extension: str | None = Field(
        default=None,
        description="File extension without dot (defaults to config).",
    )
    overwrite: bool = Field(
        default=False, description="Overwrite existing artifact if it exists."
    )


class CreateArtifactResult(BaseModel):
    name: str
    path: str
    bytes_written: int
    content: str
    file_extension: str


class CreateArtifactConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    output_dir: Path = Field(default=Path.home() / ".vibe" / "artifacts")
    default_extension: str = "md"
    max_artifact_bytes: int = 200_000
    max_total_bytes: int = 2_000_000
    max_artifacts: int = 200

    @field_validator("output_dir", mode="before")
    @classmethod
    def set_default_output_dir(cls, v: Path | str) -> Path:
        if isinstance(v, Path):
            return v
        if not v or not str(v).strip():
            return Path.home() / ".vibe" / "artifacts"
        return Path(v)

    @field_validator("output_dir", mode="after")
    @classmethod
    def expand_output_dir(cls, v: Path) -> Path:
        return v.expanduser().resolve()


class CreateArtifactState(BaseToolState):
    pass


class CreateArtifact(
    BaseTool[
        CreateArtifactArgs,
        CreateArtifactResult,
        CreateArtifactConfig,
        CreateArtifactState,
    ],
    ToolUIData[CreateArtifactArgs, CreateArtifactResult],
):
    description: ClassVar[str] = (
        "Create a persistent artifact file for content like Claude artifacts."
    )

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, CreateArtifactArgs):
            return ToolCallDisplay(summary="create_artifact")

        return ToolCallDisplay(
            summary=f"Creating artifact: {event.args.name}",
            content=event.args.content,
            details={
                "name": event.args.name,
                "extension": event.args.extension,
                "overwrite": event.args.overwrite,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if isinstance(event.result, CreateArtifactResult):
            return ToolResultDisplay(
                success=True,
                message=f"Artifact created: {event.result.name}",
                details={
                    "path": event.result.path,
                    "bytes_written": event.result.bytes_written,
                    "content": event.result.content,
                    "file_extension": event.result.file_extension,
                },
            )

        return ToolResultDisplay(success=True, message="Artifact created")

    @classmethod
    def get_status_text(cls) -> str:
        return "Creating artifact"

    @final
    async def run(self, args: CreateArtifactArgs) -> CreateArtifactResult:
        name = self._normalize_name(args.name)
        extension = self._normalize_extension(args.extension)
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        path = output_dir / f"{name}.{extension}"
        if path.exists() and not args.overwrite:
            raise ToolError(f"Artifact already exists: {path}")

        content_bytes = len(args.content.encode("utf-8"))
        self._enforce_size_limits(output_dir, content_bytes, path)

        path.write_text(args.content, encoding="utf-8")
        bytes_written = path.stat().st_size

        return CreateArtifactResult(
            name=name,
            path=str(path),
            bytes_written=bytes_written,
            content=args.content,
            file_extension=extension,
        )

    def _normalize_name(self, raw_name: str) -> str:
        name = raw_name.strip()
        if not name:
            raise ToolError("Name cannot be empty.")
        if "/" in name or "\\" in name:
            raise ToolError("Name cannot include path separators.")
        safe = re.sub(r"\s+", "_", name)
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", safe)
        safe = safe.strip("_")
        if not safe or not NAME_RE.search(safe):
            raise ToolError(
                "Name must contain letters or numbers after sanitization."
            )
        return safe

    def _normalize_extension(self, ext: str | None) -> str:
        extension = (ext or self.config.default_extension).strip().lower().lstrip(".")
        if not extension:
            raise ToolError("Extension cannot be empty.")
        if not EXT_RE.match(extension):
            raise ToolError("Extension must be alphanumeric.")
        return extension

    def _enforce_size_limits(
        self, output_dir: Path, content_bytes: int, path: Path
    ) -> None:
        max_artifact_bytes = self.config.max_artifact_bytes
        if max_artifact_bytes > 0 and content_bytes > max_artifact_bytes:
            raise ToolError(
                f"Artifact is {content_bytes} bytes, exceeds {max_artifact_bytes} bytes."
            )

        max_artifacts = self.config.max_artifacts
        if max_artifacts > 0:
            existing = [p for p in output_dir.iterdir() if p.is_file()]
            if path.exists():
                existing = [p for p in existing if p.resolve() != path.resolve()]
            if len(existing) >= max_artifacts:
                raise ToolError(
                    f"Artifact limit reached ({max_artifacts} files)."
                )

        max_total_bytes = self.config.max_total_bytes
        if max_total_bytes > 0:
            total = 0
            for entry in output_dir.iterdir():
                if not entry.is_file():
                    continue
                if path.exists() and entry.resolve() == path.resolve():
                    continue
                try:
                    total += entry.stat().st_size
                except OSError:
                    continue
            if total + content_bytes > max_total_bytes:
                raise ToolError(
                    f"Total artifact bytes would exceed {max_total_bytes} bytes."
                )
