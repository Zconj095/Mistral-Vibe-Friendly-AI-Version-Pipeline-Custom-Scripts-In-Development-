from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, Field

try:
    from actions_lib import ActionSpec, apply_defaults, load_action_specs, validate_args
except ModuleNotFoundError:  # Fallback when tools directory is not on sys.path.
    _actions_path = Path(__file__).with_name("actions_lib.py")
    _spec = importlib.util.spec_from_file_location("actions_lib", _actions_path)
    if not _spec or not _spec.loader:
        raise
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    ActionSpec = _module.ActionSpec
    apply_defaults = _module.apply_defaults
    load_action_specs = _module.load_action_specs
    validate_args = _module.validate_args
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
class _ActionRunResult:
    status: str
    output: Any | None = None
    stdout: str | None = None
    stderr: str | None = None
    exit_code: int | None = None
    timed_out: bool | None = None
    duration_sec: float | None = None
    error: str | None = None


class ActionsExecuteArgs(BaseModel):
    name: str = Field(description="Action name to execute.")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Arguments for the action."
    )
    action_dir: str | None = Field(
        default=None, description="Override the actions directory."
    )
    dry_run: bool = Field(
        default=False, description="Validate and resolve, but do not execute."
    )
    confirm: bool = Field(
        default=False,
        description="Confirm execution when action permission is 'ask'.",
    )
    allow_dangerous: bool = Field(
        default=False, description="Allow commands that match blocked patterns."
    )


class ActionsExecuteResult(BaseModel):
    name: str
    status: str
    args: dict[str, Any]
    run_type: str
    action_path: str
    dry_run: bool
    output: Any | None = None
    stdout: str | None = None
    stderr: str | None = None
    exit_code: int | None = None
    timed_out: bool | None = None
    duration_sec: float | None = None
    error: str | None = None


class ActionsExecuteConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    actions_dir: Path = Field(
        default=Path("tools/actions"),
        description="Directory containing action spec JSON files.",
    )
    default_timeout_sec: float = Field(
        default=60.0, description="Default command timeout in seconds."
    )
    max_output_chars: int = Field(
        default=8000, description="Maximum stdout/stderr characters to return."
    )
    allow_shell_commands: bool = Field(
        default=False,
        description="Allow shell string commands (shell=True). Prefer argv lists.",
    )
    blocked_patterns: list[str] = Field(
        default_factory=lambda: DEFAULT_BLOCKED_PATTERNS.copy()
    )
    allow_actions: list[str] = Field(
        default_factory=list,
        description="Optional allowlist patterns for action names.",
    )
    deny_actions: list[str] = Field(
        default_factory=list,
        description="Optional denylist patterns for action names.",
    )


class ActionsExecuteState(BaseToolState):
    pass


class ActionsExecute(
    BaseTool[
        ActionsExecuteArgs,
        ActionsExecuteResult,
        ActionsExecuteConfig,
        ActionsExecuteState,
    ],
    ToolUIData[ActionsExecuteArgs, ActionsExecuteResult],
):
    description: ClassVar[str] = (
        "Execute a local action spec (offline Actions Library runtime)."
    )

    async def run(self, args: ActionsExecuteArgs) -> ActionsExecuteResult:
        if not args.name.strip():
            raise ToolError("Action name cannot be empty.")

        actions_dir = self._resolve_actions_dir(args.action_dir)
        actions, errors = load_action_specs(actions_dir)
        if errors:
            raise ToolError("Action specs failed to load: " + "; ".join(errors))

        action = actions.get(args.name)
        if not action:
            available = ", ".join(sorted(actions.keys())) or "none"
            raise ToolError(
                f"Unknown action '{args.name}'. Available actions: {available}"
            )

        self._check_allowlist_denylist(action.name)
        self._check_permission(action, args.confirm)

        if not isinstance(args.arguments, dict):
            raise ToolError("Arguments must be an object.")

        resolved_args = apply_defaults(action.parameters, args.arguments)
        errors = validate_args(action.parameters, resolved_args)
        if errors:
            raise ToolError("Invalid arguments: " + "; ".join(errors))

        if args.dry_run:
            return ActionsExecuteResult(
                name=action.name,
                status="dry_run",
                args=resolved_args,
                run_type=action.run.get("type", "unknown"),
                action_path=str(action.source_path),
                dry_run=True,
            )

        start = time.monotonic()
        result = await self._execute_action(action, resolved_args, args.allow_dangerous)
        duration = time.monotonic() - start

        return ActionsExecuteResult(
            name=action.name,
            status=result.status,
            args=resolved_args,
            run_type=action.run.get("type", "unknown"),
            action_path=str(action.source_path),
            dry_run=False,
            output=result.output,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            timed_out=result.timed_out,
            duration_sec=duration if result.duration_sec is None else result.duration_sec,
            error=result.error,
        )

    async def _execute_action(
        self, action: ActionSpec, args: dict[str, Any], allow_dangerous: bool
    ) -> _ActionRunResult:
        run_type = action.run.get("type")
        if run_type == "python":
            return await self._run_python_action(action, args)
        if run_type == "shell":
            return self._run_shell_action(action, args, allow_dangerous)
        raise ToolError(f"Unsupported run type: {run_type}")

    async def _run_python_action(
        self, action: ActionSpec, args: dict[str, Any]
    ) -> _ActionRunResult:
        run = action.run
        module_path = run.get("path")
        module_name = run.get("module")
        function_name = run.get("function")
        base_dir = self.config.effective_workdir.resolve()

        if module_path:
            module_file = self._resolve_module_path(module_path, base_dir)
            spec = importlib.util.spec_from_file_location(
                f"action_{action.name}", module_file
            )
            if not spec or not spec.loader:
                return _ActionRunResult(
                    status="error",
                    error=f"Unable to load module from {module_file}",
                )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(module_name)

        func = getattr(module, function_name, None)
        if not callable(func):
            return _ActionRunResult(
                status="error",
                error=f"Function '{function_name}' not found in module.",
            )

        try:
            kwargs = self._prepare_kwargs(func, args, base_dir, action.source_path)
            result = func(**kwargs)
            if inspect.isawaitable(result):
                result = await result
            output = self._normalize_output(result)
            return _ActionRunResult(status="success", output=output)
        except Exception as exc:
            return _ActionRunResult(status="error", error=str(exc))

    def _run_shell_action(
        self, action: ActionSpec, args: dict[str, Any], allow_dangerous: bool
    ) -> _ActionRunResult:
        run = action.run
        base_dir = self.config.effective_workdir.resolve()
        workdir = run.get("workdir")
        timeout = run.get("timeout_sec", self.config.default_timeout_sec)

        if workdir:
            workdir_path = self._resolve_workdir(workdir, base_dir)
        else:
            workdir_path = base_dir

        env = os.environ.copy()
        env_spec = run.get("env")
        if isinstance(env_spec, dict):
            env.update(
                {
                    str(key): self._format_template(str(value), args)
                    for key, value in env_spec.items()
                }
            )

        argv = run.get("argv")
        command = run.get("command")

        if argv is not None:
            if not isinstance(argv, list) or not all(
                isinstance(item, str) for item in argv
            ):
                return _ActionRunResult(
                    status="error", error="shell argv must be a list of strings"
                )
            resolved = [self._format_template(item, args) for item in argv]
            command_str = " ".join(resolved)
            self._check_dangerous(command_str, allow_dangerous)
            return self._execute_subprocess(
                resolved, workdir_path, env, timeout, shell=False
            )

        if not self.config.allow_shell_commands:
            return _ActionRunResult(
                status="error",
                error="shell command strings are disabled by config",
            )

        if not isinstance(command, str):
            return _ActionRunResult(
                status="error", error="shell command must be a string"
            )
        resolved_command = self._format_template(command, args)
        self._check_dangerous(resolved_command, allow_dangerous)
        return self._execute_subprocess(
            resolved_command, workdir_path, env, timeout, shell=True
        )

    def _execute_subprocess(
        self,
        command: list[str] | str,
        workdir: Path,
        env: dict[str, str],
        timeout: float,
        shell: bool,
    ) -> _ActionRunResult:
        start = time.monotonic()
        try:
            proc = subprocess.run(
                command,
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                shell=shell,
            )
            stdout, _ = self._truncate(proc.stdout)
            stderr, _ = self._truncate(proc.stderr)
            status = "success" if proc.returncode == 0 else "error"
            return _ActionRunResult(
                status=status,
                stdout=stdout,
                stderr=stderr,
                exit_code=proc.returncode,
                timed_out=False,
                duration_sec=time.monotonic() - start,
                error="Non-zero exit code" if proc.returncode != 0 else None,
            )
        except subprocess.TimeoutExpired as exc:
            stdout, _ = self._truncate(exc.stdout or "")
            stderr, _ = self._truncate(exc.stderr or "")
            return _ActionRunResult(
                status="error",
                stdout=stdout,
                stderr=stderr,
                exit_code=-1,
                timed_out=True,
                duration_sec=time.monotonic() - start,
                error="Command timed out",
            )

    def _resolve_actions_dir(self, override: str | None) -> Path:
        base = self.config.effective_workdir.resolve()
        if override:
            path = Path(override).expanduser()
            if not path.is_absolute():
                path = base / path
            path = path.resolve()
        else:
            path = (base / self.config.actions_dir).resolve()

        try:
            path.relative_to(base)
        except ValueError as exc:
            raise ToolError(
                f"actions_dir must stay within project root: {base}"
            ) from exc
        return path

    def _resolve_module_path(self, module_path: str, base: Path) -> Path:
        path = Path(module_path).expanduser()
        if not path.is_absolute():
            path = base / path
        path = path.resolve()
        try:
            path.relative_to(base)
        except ValueError as exc:
            raise ToolError(
                f"Module path must stay within project root: {base}"
            ) from exc
        if not path.exists():
            raise ToolError(f"Module path does not exist: {path}")
        return path

    def _resolve_workdir(self, workdir: str, base: Path) -> Path:
        path = Path(workdir).expanduser()
        if not path.is_absolute():
            path = base / path
        path = path.resolve()
        try:
            path.relative_to(base)
        except ValueError as exc:
            raise ToolError(
                f"workdir must stay within project root: {base}"
            ) from exc
        if not path.exists():
            raise ToolError(f"workdir does not exist: {path}")
        if not path.is_dir():
            raise ToolError(f"workdir is not a directory: {path}")
        return path

    def _prepare_kwargs(
        self,
        func: Any,
        args: dict[str, Any],
        base_dir: Path,
        action_path: Path,
    ) -> dict[str, Any]:
        sig = inspect.signature(func)
        has_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in sig.parameters.values()
        )

        kwargs = dict(args)
        if "base_dir" in sig.parameters or has_kwargs:
            kwargs["base_dir"] = base_dir
        if "action_dir" in sig.parameters or has_kwargs:
            kwargs["action_dir"] = action_path.parent
        if "action_spec_path" in sig.parameters or has_kwargs:
            kwargs["action_spec_path"] = action_path

        if not has_kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

        return kwargs

    def _normalize_output(self, value: Any) -> Any:
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        try:
            json.dumps(value)
        except TypeError:
            value = json.loads(json.dumps(value, default=str))
        if isinstance(value, str):
            value, _ = self._truncate(value)
        return value

    def _format_template(self, template: str, args: dict[str, Any]) -> str:
        mapping = {key: self._format_value(value) for key, value in args.items()}
        try:
            return template.format_map(mapping)
        except KeyError as exc:
            raise ToolError(f"Missing argument for template: {exc}") from exc
        except ValueError as exc:
            raise ToolError(f"Invalid template: {exc}") from exc

    def _format_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=True)
        return str(value)

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

    def _check_allowlist_denylist(self, name: str) -> None:
        import fnmatch

        for pattern in self.config.deny_actions:
            if fnmatch.fnmatch(name, pattern):
                raise ToolError("Action blocked by denylist.")

        if self.config.allow_actions:
            if not any(fnmatch.fnmatch(name, p) for p in self.config.allow_actions):
                raise ToolError("Action not in allowlist.")

    def _check_permission(self, action: ActionSpec, confirm: bool) -> None:
        if action.permission == "never":
            raise ToolError(f"Action '{action.name}' is disabled.")
        if action.permission == "ask" and not confirm:
            raise ToolError("Action requires confirm=true to run.")

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ActionsExecuteArgs):
            return ToolCallDisplay(summary="actions_execute")
        return ToolCallDisplay(
            summary=f"actions_execute: {event.args.name}",
            details={
                "dry_run": event.args.dry_run,
                "confirm": event.args.confirm,
                "action_dir": event.args.action_dir,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ActionsExecuteResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        success = event.result.status == "success"
        message = f"Action {event.result.name} {event.result.status}"
        if event.result.dry_run:
            message = f"Action {event.result.name} validated (dry run)"
        return ToolResultDisplay(
            success=success,
            message=message,
            details=event.result.model_dump(),
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Executing action"
