from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os
import re
import shutil
import subprocess
import sys
import time
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


DEFAULT_BLOCKED_PATTERNS = [
    r"\brm\b",
    r"\brmdir\b",
    r"\brd\b",
    r"\bdel\b",
    r"\berase\b",
    r"\bformat\b",
    r"\bmkfs\b",
    r"\bdd\b",
    r"\bdiskpart\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bpoweroff\b",
    r"\bhalt\b",
    r"\breg\s+delete\b",
    r"\bremove-item\b",
]


@dataclass(frozen=True)
class _CommandResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_sec: float
    timed_out: bool
    truncated: bool


class RunCommandArgs(BaseModel):
    command: str = Field(description="Command to execute.")
    workdir: str | None = Field(
        default=None, description="Working directory (defaults to project root)."
    )
    timeout_sec: float | None = Field(
        default=None, description="Timeout in seconds (0 or None for default)."
    )
    shell: str | None = Field(
        default=None,
        description="Shell to use: auto, powershell, pwsh, cmd, bash, sh.",
    )
    allow_dangerous: bool = Field(
        default=False,
        description="Allow commands that match blocked patterns.",
    )
    env: dict[str, str] | None = Field(
        default=None, description="Extra environment variables."
    )


class RunCommandResult(BaseModel):
    command: str
    workdir: str
    shell: str
    exit_code: int
    stdout: str
    stderr: str
    duration_sec: float
    timed_out: bool
    output_truncated: bool
    timestamp: str


class RunCommandConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    default_timeout_sec: float = 60.0
    max_output_chars: int = 8000
    default_shell: str = "auto"
    blocked_patterns: list[str] = Field(
        default_factory=lambda: DEFAULT_BLOCKED_PATTERNS.copy()
    )


class RunCommandState(BaseToolState):
    pass


class RunCommand(
    BaseTool[RunCommandArgs, RunCommandResult, RunCommandConfig, RunCommandState],
    ToolUIData[RunCommandArgs, RunCommandResult],
):
    description: ClassVar[str] = "Run a shell command locally with guardrails."

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, RunCommandArgs):
            return ToolCallDisplay(summary="run_command")
        return ToolCallDisplay(
            summary=f"run_command: {event.args.command}",
            details={
                "workdir": event.args.workdir,
                "timeout_sec": event.args.timeout_sec,
                "shell": event.args.shell,
                "allow_dangerous": event.args.allow_dangerous,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if isinstance(event.result, RunCommandResult):
            message = f"Command exited with code {event.result.exit_code}"
            if event.result.timed_out:
                message = "Command timed out"
            warnings = []
            if event.result.output_truncated:
                warnings.append("Output truncated")
            return ToolResultDisplay(
                success=event.result.exit_code == 0 and not event.result.timed_out,
                message=message,
                warnings=warnings,
                details={
                    "stdout": event.result.stdout,
                    "stderr": event.result.stderr,
                    "duration_sec": event.result.duration_sec,
                    "workdir": event.result.workdir,
                    "shell": event.result.shell,
                },
            )
        return ToolResultDisplay(success=True, message="Command complete")

    @classmethod
    def get_status_text(cls) -> str:
        return "Running command"

    async def run(self, args: RunCommandArgs) -> RunCommandResult:
        command = args.command.strip()
        if not command:
            raise ToolError("Command cannot be empty.")

        self._check_dangerous(command, args.allow_dangerous)
        self._check_allowlist_denylist(command)

        workdir = self._resolve_workdir(args.workdir)
        shell = self._resolve_shell(args.shell)
        timeout = self._resolve_timeout(args.timeout_sec)

        start = time.monotonic()
        result = self._execute(command, workdir, shell, timeout, args.env)
        duration = time.monotonic() - start

        return RunCommandResult(
            command=command,
            workdir=str(workdir),
            shell=shell,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_sec=duration,
            timed_out=result.timed_out,
            output_truncated=result.truncated,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )

    def _resolve_workdir(self, workdir: str | None) -> Path:
        base = self.config.effective_workdir.resolve()
        if workdir:
            path = Path(workdir).expanduser()
            if not path.is_absolute():
                path = base / path
            path = path.resolve()
        else:
            path = base

        try:
            path.relative_to(base)
        except ValueError:
            raise ToolError(f"workdir must stay within project root: {base}")
        if not path.exists():
            raise ToolError(f"workdir does not exist: {path}")
        if not path.is_dir():
            raise ToolError(f"workdir is not a directory: {path}")
        return path

    def _resolve_shell(self, shell: str | None) -> str:
        value = (shell or self.config.default_shell or "auto").strip().lower()
        if value not in {"auto", "powershell", "pwsh", "cmd", "bash", "sh"}:
            raise ToolError(
                "shell must be auto, powershell, pwsh, cmd, bash, or sh."
            )
        if value == "auto":
            if sys.platform.startswith("win"):
                if shutil.which("pwsh"):
                    return "pwsh"
                if shutil.which("powershell"):
                    return "powershell"
                return "cmd"
            if shutil.which("bash"):
                return "bash"
            return "sh"
        if value in {"powershell", "pwsh"} and not shutil.which(value):
            raise ToolError(f"{value} not found on PATH.")
        if value == "bash" and not shutil.which("bash"):
            raise ToolError("bash not found on PATH.")
        if value == "sh" and not shutil.which("sh"):
            raise ToolError("sh not found on PATH.")
        return value

    def _resolve_timeout(self, timeout: float | None) -> float:
        if timeout is None or timeout <= 0:
            return float(self.config.default_timeout_sec)
        return float(timeout)

    def _execute(
        self,
        command: str,
        workdir: Path,
        shell: str,
        timeout: float,
        extra_env: dict[str, str] | None,
    ) -> _CommandResult:
        env = os.environ.copy()
        if extra_env:
            env.update({str(k): str(v) for k, v in extra_env.items()})

        cmd = self._build_shell_command(shell, command)
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            stdout, out_trunc = self._truncate(proc.stdout)
            stderr, err_trunc = self._truncate(proc.stderr)
            return _CommandResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=proc.returncode,
                duration_sec=0.0,
                timed_out=False,
                truncated=out_trunc or err_trunc,
            )
        except subprocess.TimeoutExpired as exc:
            stdout, out_trunc = self._truncate(exc.stdout or "")
            stderr, err_trunc = self._truncate(exc.stderr or "")
            return _CommandResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=-1,
                duration_sec=0.0,
                timed_out=True,
                truncated=out_trunc or err_trunc,
            )

    def _build_shell_command(self, shell: str, command: str) -> list[str]:
        if shell == "powershell":
            return ["powershell", "-NoProfile", "-Command", command]
        if shell == "pwsh":
            return ["pwsh", "-NoProfile", "-Command", command]
        if shell == "cmd":
            return ["cmd", "/c", command]
        if shell == "bash":
            return ["bash", "-lc", command]
        return ["sh", "-lc", command]

    def _truncate(self, text: str) -> tuple[str, bool]:
        max_chars = self.config.max_output_chars
        if max_chars > 0 and len(text) > max_chars:
            return text[:max_chars] + "\n...[truncated]", True
        return text, False

    def _check_dangerous(self, command: str, allow_dangerous: bool) -> None:
        if allow_dangerous:
            return
        lowered = command.lower()
        for pattern in self.config.blocked_patterns:
            if re.search(pattern, lowered):
                raise ToolError(
                    "Command blocked by safety rules. "
                    "Set allow_dangerous=true to override."
                )

    def _check_allowlist_denylist(self, command: str) -> None:
        import fnmatch

        allowlist = getattr(self.config, "allowlist", [])
        denylist = getattr(self.config, "denylist", [])

        for pattern in denylist:
            if fnmatch.fnmatch(command, pattern):
                raise ToolError("Command blocked by denylist.")

        if allowlist:
            if not any(fnmatch.fnmatch(command, pattern) for pattern in allowlist):
                raise ToolError("Command not in allowlist.")
