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


class ReadToolchainMultiConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_source_bytes: int = Field(
        default=5_000_000, description="Maximum size per source (bytes)."
    )
    max_total_bytes: int = Field(
        default=20_000_000, description="Maximum total size across sources (bytes)."
    )
    max_sources_per_kind: int = Field(
        default=50, description="Maximum sources per data kind."
    )


class ReadToolchainMultiState(BaseToolState):
    pass


class ToolchainSource(BaseModel):
    path: str | None = Field(default=None, description="Path to a file.")
    content: str | None = Field(default=None, description="Inline content.")
    label: str | None = Field(default=None, description="Optional label.")


class ReadToolchainMultiArgs(BaseModel):
    compiler_sources: list[ToolchainSource] | None = Field(
        default=None, description="Compiler data sources."
    )
    interpreter_sources: list[ToolchainSource] | None = Field(
        default=None, description="Interpreter data sources."
    )
    transpiled_sources: list[ToolchainSource] | None = Field(
        default=None, description="Transpiled data sources."
    )
    max_source_bytes: int | None = Field(
        default=None, description="Override max_source_bytes."
    )
    max_total_bytes: int | None = Field(
        default=None, description="Override max_total_bytes."
    )
    max_sources_per_kind: int | None = Field(
        default=None, description="Override max_sources_per_kind."
    )


class ToolchainEntry(BaseModel):
    kind: str
    label: str
    source_path: str | None
    bytes_total: int
    content: str


class ReadToolchainMultiResult(BaseModel):
    compiler: list[ToolchainEntry]
    interpreter: list[ToolchainEntry]
    transpiled: list[ToolchainEntry]
    count: int
    compiler_count: int
    interpreter_count: int
    transpiled_count: int
    total_bytes: int
    truncated: bool
    errors: list[str]


class ReadToolchainMulti(
    BaseTool[
        ReadToolchainMultiArgs,
        ReadToolchainMultiResult,
        ReadToolchainMultiConfig,
        ReadToolchainMultiState,
    ],
    ToolUIData[ReadToolchainMultiArgs, ReadToolchainMultiResult],
):
    description: ClassVar[str] = (
        "Read compiler, interpreter, and transpiled data together."
    )

    async def run(self, args: ReadToolchainMultiArgs) -> ReadToolchainMultiResult:
        compiler_sources = args.compiler_sources or []
        interpreter_sources = args.interpreter_sources or []
        transpiled_sources = args.transpiled_sources or []

        if not compiler_sources and not interpreter_sources and not transpiled_sources:
            raise ToolError("Provide at least one compiler/interpreter/transpiled source.")

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
        max_sources_per_kind = (
            args.max_sources_per_kind
            if args.max_sources_per_kind is not None
            else self.config.max_sources_per_kind
        )
        if max_source_bytes <= 0:
            raise ToolError("max_source_bytes must be a positive integer.")
        if max_total_bytes <= 0:
            raise ToolError("max_total_bytes must be a positive integer.")
        if max_sources_per_kind <= 0:
            raise ToolError("max_sources_per_kind must be a positive integer.")

        total_bytes = 0
        errors: list[str] = []
        truncated = False

        compiler_entries: list[ToolchainEntry] = []
        interpreter_entries: list[ToolchainEntry] = []
        transpiled_entries: list[ToolchainEntry] = []

        def load_group(
            kind: str,
            sources: list[ToolchainSource],
            current_total: int,
        ) -> tuple[list[ToolchainEntry], int, bool]:
            entries: list[ToolchainEntry] = []
            local_truncated = False

            if len(sources) > max_sources_per_kind:
                sources = sources[:max_sources_per_kind]
                local_truncated = True

            for index, source in enumerate(sources, start=1):
                label = self._source_label(kind, index, source)
                try:
                    content, size_bytes, source_path = self._load_source(source)
                    if size_bytes > max_source_bytes:
                        raise ToolError(
                            f"{label} exceeds max_source_bytes "
                            f"({size_bytes} > {max_source_bytes})."
                        )
                    if current_total + size_bytes > max_total_bytes:
                        local_truncated = True
                        break

                    current_total += size_bytes
                    entries.append(
                        ToolchainEntry(
                            kind=kind,
                            label=label,
                            source_path=source_path,
                            bytes_total=size_bytes,
                            content=content,
                        )
                    )
                except ToolError as exc:
                    errors.append(f"{label}: {exc}")

            return entries, current_total, local_truncated

        if compiler_sources:
            compiler_entries, total_bytes, truncated = load_group(
                "compiler", compiler_sources, total_bytes
            )
        if not truncated and interpreter_sources:
            interpreter_entries, total_bytes, truncated = load_group(
                "interpreter", interpreter_sources, total_bytes
            )
        if not truncated and transpiled_sources:
            transpiled_entries, total_bytes, truncated = load_group(
                "transpiled", transpiled_sources, total_bytes
            )

        return ReadToolchainMultiResult(
            compiler=compiler_entries,
            interpreter=interpreter_entries,
            transpiled=transpiled_entries,
            count=len(compiler_entries) + len(interpreter_entries) + len(transpiled_entries),
            compiler_count=len(compiler_entries),
            interpreter_count=len(interpreter_entries),
            transpiled_count=len(transpiled_entries),
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
        if not isinstance(event.args, ReadToolchainMultiArgs):
            return ToolCallDisplay(summary="read_toolchain_multi")

        compiler_count = len(event.args.compiler_sources or [])
        interpreter_count = len(event.args.interpreter_sources or [])
        transpiled_count = len(event.args.transpiled_sources or [])
        summary = (
            "read_toolchain_multi: "
            f"{compiler_count} compiler, {interpreter_count} interpreter, "
            f"{transpiled_count} transpiled"
        )
        return ToolCallDisplay(
            summary=summary,
            details={
                "compiler_sources": compiler_count,
                "interpreter_sources": interpreter_count,
                "transpiled_sources": transpiled_count,
                "max_source_bytes": event.args.max_source_bytes,
                "max_total_bytes": event.args.max_total_bytes,
                "max_sources_per_kind": event.args.max_sources_per_kind,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ReadToolchainMultiResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = (
            f"Read {event.result.count} source(s) "
            f"({event.result.compiler_count} compiler, "
            f"{event.result.interpreter_count} interpreter, "
            f"{event.result.transpiled_count} transpiled)"
        )
        warnings = event.result.errors[:]
        if event.result.truncated:
            warnings.append("Source list truncated by size or limits")

        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=warnings,
            details={
                "count": event.result.count,
                "compiler_count": event.result.compiler_count,
                "interpreter_count": event.result.interpreter_count,
                "transpiled_count": event.result.transpiled_count,
                "total_bytes": event.result.total_bytes,
                "truncated": event.result.truncated,
                "errors": event.result.errors,
                "compiler": event.result.compiler,
                "interpreter": event.result.interpreter,
                "transpiled": event.result.transpiled,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Reading toolchain data"
