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


class ReadToolchainSingleConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_source_bytes: int = Field(
        default=5_000_000, description="Maximum size per source (bytes)."
    )
    max_total_bytes: int = Field(
        default=20_000_000, description="Maximum total size across sources (bytes)."
    )
    max_sources: int = Field(
        default=100, description="Maximum sources to read."
    )


class ReadToolchainSingleState(BaseToolState):
    pass


class ToolchainSource(BaseModel):
    path: str | None = Field(default=None, description="Path to a file.")
    content: str | None = Field(default=None, description="Inline content.")
    label: str | None = Field(default=None, description="Optional label.")


class ReadToolchainSingleArgs(BaseModel):
    kind: str = Field(description="compiler, interpreter, or transpiled.")
    sources: list[ToolchainSource]
    max_source_bytes: int | None = Field(
        default=None, description="Override max_source_bytes."
    )
    max_total_bytes: int | None = Field(
        default=None, description="Override max_total_bytes."
    )
    max_sources: int | None = Field(
        default=None, description="Override max_sources."
    )


class ToolchainEntry(BaseModel):
    label: str
    source_path: str | None
    bytes_total: int
    content: str


class ReadToolchainSingleResult(BaseModel):
    kind: str
    entries: list[ToolchainEntry]
    count: int
    total_bytes: int
    truncated: bool
    errors: list[str]


class ReadToolchainSingle(
    BaseTool[
        ReadToolchainSingleArgs,
        ReadToolchainSingleResult,
        ReadToolchainSingleConfig,
        ReadToolchainSingleState,
    ],
    ToolUIData[ReadToolchainSingleArgs, ReadToolchainSingleResult],
):
    description: ClassVar[str] = (
        "Read compiler, interpreter, or transpiled data (single kind)."
    )

    async def run(self, args: ReadToolchainSingleArgs) -> ReadToolchainSingleResult:
        kind = args.kind.strip().lower()
        if kind not in {"compiler", "interpreter", "transpiled"}:
            raise ToolError("kind must be compiler, interpreter, or transpiled.")
        if not args.sources:
            raise ToolError("Provide at least one source.")

        max_source_bytes = (
            args.max_source_bytes
            if args.max_source_bytes is not None
            else self.config.max_source_bytes
        )
        max_total_bytes = (
            args.max_total_bytes
            if args.max_total_bytes is not None
            else self.config.max_total_bytes
        )
        max_sources = (
            args.max_sources if args.max_sources is not None else self.config.max_sources
        )
        if max_source_bytes <= 0:
            raise ToolError("max_source_bytes must be a positive integer.")
        if max_total_bytes <= 0:
            raise ToolError("max_total_bytes must be a positive integer.")
        if max_sources <= 0:
            raise ToolError("max_sources must be a positive integer.")

        total_bytes = 0
        truncated = False
        errors: list[str] = []
        entries: list[ToolchainEntry] = []

        sources = args.sources
        if len(sources) > max_sources:
            sources = sources[:max_sources]
            truncated = True

        for index, source in enumerate(sources, start=1):
            label = self._source_label(kind, index, source)
            try:
                content, size_bytes, source_path = self._load_source(source)
                if size_bytes > max_source_bytes:
                    raise ToolError(
                        f"{label} exceeds max_source_bytes "
                        f"({size_bytes} > {max_source_bytes})."
                    )
                if total_bytes + size_bytes > max_total_bytes:
                    truncated = True
                    break

                total_bytes += size_bytes
                entries.append(
                    ToolchainEntry(
                        label=label,
                        source_path=source_path,
                        bytes_total=size_bytes,
                        content=content,
                    )
                )
            except ToolError as exc:
                errors.append(f"{label}: {exc}")

        return ReadToolchainSingleResult(
            kind=kind,
            entries=entries,
            count=len(entries),
            total_bytes=total_bytes,
            truncated=truncated,
            errors=errors,
        )

    def _source_label(
        self, kind: str, index: int, source: ToolchainSource
    ) -> str:
        if source.label and source.label.strip():
            return source.label.strip()
        if source.path and source.path.strip():
            return Path(source.path).name
        return f"{kind}-{index}"

    def _load_source(self, source: ToolchainSource) -> tuple[str, int, str | None]:
        if (source.path is None and source.content is None) or (
            source.path is not None and source.content is not None
        ):
            raise ToolError("Provide either path or content, but not both.")

        if source.path is not None:
            path = self._resolve_path(source.path)
            data = path.read_bytes()
            content = data.decode("utf-8", errors="ignore")
            return content, len(data), str(path)

        content = source.content or ""
        size_bytes = len(content.encode("utf-8"))
        return content, size_bytes, None

    def _resolve_path(self, raw_path: str) -> Path:
        if not raw_path.strip():
            raise ToolError("Path cannot be empty.")

        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.config.effective_workdir / path

        try:
            resolved = path.resolve()
        except ValueError as exc:
            raise ToolError("Security error: cannot resolve the provided path.") from exc

        if not resolved.exists():
            raise ToolError(f"File not found at: {resolved}")
        if resolved.is_dir():
            raise ToolError(f"Path is a directory, not a file: {resolved}")

        return resolved

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ReadToolchainSingleArgs):
            return ToolCallDisplay(summary="read_toolchain_single")

        summary = f"read_toolchain_single: {event.args.kind}"
        return ToolCallDisplay(
            summary=summary,
            details={
                "kind": event.args.kind,
                "sources": len(event.args.sources),
                "max_source_bytes": event.args.max_source_bytes,
                "max_total_bytes": event.args.max_total_bytes,
                "max_sources": event.args.max_sources,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ReadToolchainSingleResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = f"Read {event.result.count} source(s) ({event.result.kind})"
        warnings = event.result.errors[:]
        if event.result.truncated:
            warnings.append("Source list truncated by size or limits")

        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=warnings,
            details={
                "kind": event.result.kind,
                "count": event.result.count,
                "total_bytes": event.result.total_bytes,
                "truncated": event.result.truncated,
                "errors": event.result.errors,
                "entries": event.result.entries,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Reading toolchain data (single kind)"
