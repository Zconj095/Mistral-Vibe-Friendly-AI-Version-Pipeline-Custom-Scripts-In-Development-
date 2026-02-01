from __future__ import annotations

import sys
import time
from pathlib import Path
import subprocess
from typing import TYPE_CHECKING, ClassVar
from uuid import uuid4

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


DEFAULT_SANDBOX_DIR = Path.home() / ".vibe" / "sandbox"


class CodeInterpreterConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    sandbox_dir: Path = Field(
        default=DEFAULT_SANDBOX_DIR, description="Base directory for sandbox runs."
    )
    timeout_seconds: float = Field(
        default=20.0, description="Maximum execution time."
    )
    max_output_chars: int = Field(
        default=12000, description="Maximum stdout characters to return."
    )
    max_error_chars: int = Field(
        default=8000, description="Maximum stderr characters to return."
    )
    max_files: int = Field(
        default=20, description="Maximum number of output files to return."
    )


class CodeInterpreterState(BaseToolState):
    pass


class CodeInterpreterArgs(BaseModel):
    code: str = Field(description="Python code to execute.")
    timeout_seconds: float | None = Field(
        default=None, description="Override execution timeout."
    )
    working_dir: str | None = Field(
        default=None, description="Optional subdirectory within the sandbox."
    )
    files: dict[str, str] | None = Field(
        default=None, description="Optional input files to write before execution."
    )


class CodeInterpreterResult(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
    stdout_truncated: bool
    stderr_truncated: bool
    sandbox_dir: str
    artifacts: list[str]


class CodeInterpreter(
    BaseTool[
        CodeInterpreterArgs,
        CodeInterpreterResult,
        CodeInterpreterConfig,
        CodeInterpreterState,
    ],
    ToolUIData[CodeInterpreterArgs, CodeInterpreterResult],
):
    description: ClassVar[str] = (
        "Run Python code in a local sandbox directory and return stdout/stderr."
    )

    async def run(self, args: CodeInterpreterArgs) -> CodeInterpreterResult:
        if not args.code.strip():
            raise ToolError("code cannot be empty.")

        sandbox_root = self.config.sandbox_dir.expanduser().resolve()
        sandbox_root.mkdir(parents=True, exist_ok=True)

        run_dir = self._prepare_run_dir(sandbox_root, args.working_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        input_files = self._write_input_files(run_dir, args.files)
        code_path = run_dir / "main.py"
        code_path.write_text(args.code, encoding="utf-8")

        timeout = args.timeout_seconds or self.config.timeout_seconds

        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-u", str(code_path)],
                cwd=str(run_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            exit_code = int(proc.returncode)
        except subprocess.TimeoutExpired:
            stdout = ""
            stderr = f"Execution timed out after {timeout} seconds."
            exit_code = -1

        stdout, stdout_truncated = self._truncate(
            stdout, self.config.max_output_chars
        )
        stderr, stderr_truncated = self._truncate(
            stderr, self.config.max_error_chars
        )

        artifacts = self._collect_artifacts(run_dir, input_files, code_path)

        return CodeInterpreterResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            sandbox_dir=str(run_dir),
            artifacts=artifacts,
        )

    def _prepare_run_dir(self, base: Path, working_dir: str | None) -> Path:
        if working_dir:
            candidate = Path(working_dir)
            if candidate.is_absolute():
                raise ToolError("working_dir must be a relative path.")
            run_dir = (base / candidate).resolve()
        else:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            run_dir = (base / f"run_{stamp}_{uuid4().hex}").resolve()

        if not self._is_within(run_dir, base):
            raise ToolError("working_dir escapes the sandbox directory.")
        return run_dir

    def _write_input_files(
        self, run_dir: Path, files: dict[str, str] | None
    ) -> set[Path]:
        written: set[Path] = set()
        if not files:
            return written

        for name, content in files.items():
            if not name:
                continue
            rel = Path(name)
            if rel.is_absolute():
                raise ToolError("Input file paths must be relative.")
            path = (run_dir / rel).resolve()
            if not self._is_within(path, run_dir):
                raise ToolError("Input file path escapes the sandbox directory.")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            written.add(path)
        return written

    def _collect_artifacts(
        self, run_dir: Path, inputs: set[Path], code_path: Path
    ) -> list[str]:
        artifacts: list[str] = []
        for path in run_dir.rglob("*"):
            if not path.is_file():
                continue
            if path == code_path or path in inputs:
                continue
            artifacts.append(str(path))
            if len(artifacts) >= self.config.max_files:
                break
        return artifacts

    def _truncate(self, text: str, limit: int) -> tuple[str, bool]:
        if limit <= 0:
            return "", bool(text)
        if len(text) <= limit:
            return text, False
        return text[:limit], True

    def _is_within(self, path: Path, base: Path) -> bool:
        try:
            path.relative_to(base)
            return True
        except ValueError:
            return False

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, CodeInterpreterArgs):
            return ToolCallDisplay(summary="code_interpreter")
        return ToolCallDisplay(
            summary="code_interpreter",
            details={
                "timeout_seconds": event.args.timeout_seconds,
                "working_dir": event.args.working_dir,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, CodeInterpreterResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = f"Exit code {event.result.exit_code}"
        warnings = []
        if event.result.stdout_truncated:
            warnings.append("stdout truncated")
        if event.result.stderr_truncated:
            warnings.append("stderr truncated")

        return ToolResultDisplay(
            success=event.result.exit_code == 0,
            message=message,
            warnings=warnings,
            details={
                "stdout": event.result.stdout,
                "stderr": event.result.stderr,
                "artifacts": event.result.artifacts,
                "sandbox_dir": event.result.sandbox_dir,
                "exit_code": event.result.exit_code,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Running code"
