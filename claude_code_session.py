from __future__ import annotations

import fnmatch
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, Field

try:
    from actions_lib import validate_args
except ModuleNotFoundError:  # Fallback when tools directory is not on sys.path.
    import importlib.util

    _actions_path = Path(__file__).with_name("actions_lib.py")
    _spec = importlib.util.spec_from_file_location("actions_lib", _actions_path)
    if not _spec or not _spec.loader:
        raise
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
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


DEFAULT_EXCLUDE_GLOBS = [
    "**/.git/**",
    "**/.svn/**",
    "**/.hg/**",
    "**/node_modules/**",
    "**/.venv/**",
    "**/venv/**",
    "**/.idea/**",
    "**/.vscode/**",
    "**/dist/**",
    "**/build/**",
    "**/target/**",
    "**/.mypy_cache/**",
    "**/.pytest_cache/**",
    "**/.cache/**",
    "**/__pycache__/**",
]

HUNK_RE = re.compile(r"^@@ -(\\d+)(?:,(\\d+))? \\+(\\d+)(?:,(\\d+))? @@")

PATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "diff": {"type": "string"},
        "summary": {"type": "string"},
        "allow_create": {"type": "boolean"},
    },
    "required": ["path", "diff"],
    "additionalProperties": False,
}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "plan": {"type": "array", "items": {"type": "string"}},
        "patches": {"type": "array", "items": PATCH_SCHEMA},
        "commands": {"type": "array", "items": {"type": "string"}},
        "questions": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["plan", "patches"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class _Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str]


@dataclass
class _FileState:
    path: Path
    original: str
    current: str
    newline: str
    ends_with_newline: bool


@dataclass
class _ContextFile:
    info: "ContextFileInfo"
    content: str


class ClaudeCodeMessage(BaseModel):
    role: str
    content: str


class ContextFileInfo(BaseModel):
    path: str
    bytes: int
    truncated: bool
    editable: bool
    missing: bool


class PatchProposal(BaseModel):
    path: str
    diff: str
    summary: str | None = Field(default=None, description="Optional summary.")
    allow_create: bool | None = Field(
        default=None, description="Allow creating new file."
    )


class PatchResult(BaseModel):
    path: str
    diff: str
    summary: str | None
    allow_create: bool
    applied: bool
    written: bool
    valid: bool
    hunks: int
    lines_added: int
    lines_removed: int
    error: str | None


class ClaudeCodeSessionArgs(BaseModel):
    task: str = Field(description="Coding task to complete.")
    paths: list[str] | None = Field(
        default=None, description="Editable file paths."
    )
    context_paths: list[str] | None = Field(
        default=None, description="Read-only context file paths."
    )
    task_context: str | None = Field(
        default=None, description="Additional task context."
    )
    messages: list[ClaudeCodeMessage] | None = Field(
        default=None, description="Optional prior chat messages."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prefix."
    )
    include_tree: bool = Field(
        default=False, description="Include a file tree listing."
    )
    tree_root: str | None = Field(
        default=None, description="Root for tree listing."
    )
    include_globs: list[str] | None = Field(
        default=None, description="Include glob filters for tree listing."
    )
    exclude_globs: list[str] | None = Field(
        default=None, description="Exclude glob filters for tree listing."
    )
    max_depth: int | None = Field(
        default=None, description="Maximum depth for tree listing."
    )
    max_tree_files: int | None = Field(
        default=None, description="Maximum files in tree listing."
    )
    plan_only: bool = Field(
        default=False, description="Only return a plan, no diffs."
    )
    apply_patches: bool = Field(
        default=False, description="Apply patches to disk."
    )
    allow_create: bool = Field(
        default=False, description="Allow creating new files."
    )
    allow_unlisted: bool = Field(
        default=False, description="Allow patches outside of paths list."
    )
    create_backup: bool | None = Field(
        default=None, description="Override backup behavior."
    )
    llm_api_base: str | None = Field(
        default=None, description="OpenAI-compatible API base URL."
    )
    llm_model: str | None = Field(
        default=None, description="LLM model name."
    )
    llm_temperature: float = Field(default=0.2, description="LLM temperature.")
    llm_max_tokens: int = Field(default=1400, description="LLM max tokens.")
    llm_stream: bool = Field(default=False, description="Stream LLM tokens.")
    max_retries: int | None = Field(
        default=None, description="Max retries for invalid JSON."
    )
    strict_json: bool = Field(
        default=True, description="Require JSON-only output."
    )
    include_raw: bool = Field(
        default=True, description="Include raw model output."
    )
    max_file_bytes: int | None = Field(
        default=None, description="Override max file bytes."
    )
    max_total_bytes: int | None = Field(
        default=None, description="Override max total bytes."
    )
    max_diff_bytes: int | None = Field(
        default=None, description="Override max diff bytes."
    )


class ClaudeCodeSessionResult(BaseModel):
    task: str
    plan: list[str]
    patches: list[PatchResult]
    commands: list[str]
    questions: list[str]
    notes: str | None
    warnings: list[str]
    errors: list[str]
    attempts: int
    llm_model: str
    raw: str | None
    context_files: list[ContextFileInfo]
    tree: list[str]


class ClaudeCodeSessionConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    llm_api_base: str = "http://127.0.0.1:11434/v1"
    llm_model: str = "gpt-oss:latest"
    max_file_bytes: int = Field(
        default=200_000, description="Maximum bytes to read per file."
    )
    max_total_bytes: int = Field(
        default=2_000_000, description="Maximum bytes across all files."
    )
    max_diff_bytes: int = Field(
        default=200_000, description="Maximum diff bytes to accept."
    )
    default_exclude_globs: list[str] = Field(
        default=DEFAULT_EXCLUDE_GLOBS,
        description="Default glob patterns excluded during tree listing.",
    )
    default_max_tree_files: int = Field(
        default=500, description="Default max files for tree listing."
    )
    default_max_depth: int = Field(
        default=8, description="Default max depth for tree listing."
    )
    create_backup: bool = Field(
        default=False, description="Create .bak backups before writing."
    )
    max_retries: int = Field(default=2, description="Max JSON retries.")


class ClaudeCodeSessionState(BaseToolState):
    pass


class ClaudeCodeSession(
    BaseTool[
        ClaudeCodeSessionArgs,
        ClaudeCodeSessionResult,
        ClaudeCodeSessionConfig,
        ClaudeCodeSessionState,
    ],
    ToolUIData[ClaudeCodeSessionArgs, ClaudeCodeSessionResult],
):
    description: ClassVar[str] = (
        "Plan and draft unified diffs for local code changes."
    )

    async def run(self, args: ClaudeCodeSessionArgs) -> ClaudeCodeSessionResult:
        task = (args.task or "").strip()
        if not task:
            raise ToolError("task cannot be empty.")

        warnings: list[str] = []
        errors: list[str] = []
        plan_only = bool(args.plan_only)
        apply_patches = bool(args.apply_patches)
        if plan_only and apply_patches:
            warnings.append("plan_only enabled; disabling apply_patches.")
            apply_patches = False

        base = self.config.effective_workdir.resolve()
        editable_paths = self._normalize_paths(args.paths, base)
        context_paths = self._normalize_paths(args.context_paths, base)
        if not editable_paths and not context_paths:
            warnings.append("No file paths provided; using task-only context.")

        context_files = self._load_context_files(
            editable_paths,
            context_paths,
            args,
            warnings,
        )
        tree: list[str] = []
        if args.include_tree:
            tree_root = self._resolve_tree_root(args.tree_root, base)
            tree = self._build_tree(
                tree_root,
                args.include_globs,
                args.exclude_globs,
                args.max_depth,
                args.max_tree_files,
            )

        messages = self._build_messages(
            task,
            context_files,
            tree,
            args,
        )
        output, call_errors, raw, attempts = self._call_llm_json(
            OUTPUT_SCHEMA, messages, args
        )
        errors.extend(call_errors)

        if output is None:
            return ClaudeCodeSessionResult(
                task=task,
                plan=[],
                patches=[],
                commands=[],
                questions=[],
                notes=None,
                warnings=warnings,
                errors=errors,
                attempts=attempts,
                llm_model=self._resolve_model(args),
                raw=raw if args.include_raw else None,
                context_files=[item.info for item in context_files],
                tree=tree,
            )

        model_warnings = output.get("warnings", []) if isinstance(output, dict) else []
        if isinstance(model_warnings, list):
            warnings.extend([str(w) for w in model_warnings if str(w).strip()])

        plan = self._coerce_list(output.get("plan")) if isinstance(output, dict) else []
        patch_specs = output.get("patches", []) if isinstance(output, dict) else []
        commands = self._coerce_list(output.get("commands")) if isinstance(output, dict) else []
        questions = self._coerce_list(output.get("questions")) if isinstance(output, dict) else []
        notes = output.get("notes") if isinstance(output, dict) else None

        patch_results: list[PatchResult] = []
        if plan_only:
            if patch_specs:
                warnings.append("plan_only requested; ignoring patches from model.")
        else:
            patch_results, patch_errors = self._apply_patch_specs(
                patch_specs,
                editable_paths,
                context_paths,
                apply_patches,
                args,
                warnings,
            )
            errors.extend(patch_errors)

        return ClaudeCodeSessionResult(
            task=task,
            plan=plan,
            patches=patch_results,
            commands=commands,
            questions=questions,
            notes=str(notes) if notes is not None else None,
            warnings=warnings,
            errors=errors,
            attempts=attempts,
            llm_model=self._resolve_model(args),
            raw=raw if args.include_raw else None,
            context_files=[item.info for item in context_files],
            tree=tree,
        )

    def _normalize_paths(self, paths: list[str] | None, base: Path) -> list[Path]:
        resolved: list[Path] = []
        if not paths:
            return resolved
        for raw in paths:
            if not raw or not str(raw).strip():
                continue
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = base / path
            path = path.resolve()
            try:
                path.relative_to(base)
            except ValueError as exc:
                raise ToolError(
                    f"Path must stay within project root: {base}"
                ) from exc
            resolved.append(path)
        return resolved

    def _load_context_files(
        self,
        editable: list[Path],
        readonly: list[Path],
        args: ClaudeCodeSessionArgs,
        warnings: list[str],
    ) -> list[_ContextFile]:
        max_file_bytes = (
            args.max_file_bytes
            if args.max_file_bytes is not None
            else self.config.max_file_bytes
        )
        max_total_bytes = (
            args.max_total_bytes
            if args.max_total_bytes is not None
            else self.config.max_total_bytes
        )

        files: list[_ContextFile] = []
        total = 0
        seen: set[Path] = set()

        def add_file(path: Path, editable_flag: bool) -> None:
            nonlocal total
            if path in seen:
                return
            seen.add(path)

            if not path.exists():
                if editable_flag and args.allow_create:
                    info = ContextFileInfo(
                        path=self._display_path(path),
                        bytes=0,
                        truncated=False,
                        editable=True,
                        missing=True,
                    )
                    files.append(_ContextFile(info=info, content=""))
                    warnings.append(f"Editable file missing, will be created: {path}")
                    return
                if editable_flag:
                    raise ToolError(f"Editable file not found: {path}")
                warnings.append(f"Context file not found: {path}")
                return
            if path.is_dir():
                if editable_flag:
                    raise ToolError(f"Editable path is a directory: {path}")
                warnings.append(f"Context path is a directory: {path}")
                return

            content, truncated = self._read_file(path, max_file_bytes)
            size = len(content.encode("utf-8"))
            if max_total_bytes > 0 and total + size > max_total_bytes:
                warnings.append(
                    f"Total context bytes exceeded at {path}; truncating context list."
                )
                return
            total += size
            info = ContextFileInfo(
                path=self._display_path(path),
                bytes=size,
                truncated=truncated,
                editable=editable_flag,
                missing=False,
            )
            files.append(_ContextFile(info=info, content=content))
            if truncated:
                warnings.append(f"Context truncated for {path}.")

        for path in editable:
            add_file(path, True)
        for path in readonly:
            add_file(path, False)

        return files

    def _read_file(self, path: Path, max_bytes: int) -> tuple[str, bool]:
        truncated = False
        try:
            with path.open("rb") as handle:
                if max_bytes > 0:
                    data = handle.read(max_bytes + 1)
                    if len(data) > max_bytes:
                        data = data[:max_bytes]
                        truncated = True
                else:
                    data = handle.read()
        except OSError as exc:
            raise ToolError(f"Failed to read {path}: {exc}") from exc
        return data.decode("utf-8", errors="ignore"), truncated

    def _build_messages(
        self,
        task: str,
        context_files: list[_ContextFile],
        tree: list[str],
        args: ClaudeCodeSessionArgs,
    ) -> list[dict[str, str]]:
        schema_text = json.dumps(OUTPUT_SCHEMA, ensure_ascii=True)
        base_prompt = (
            "You are a local coding agent. Use the provided context to plan and "
            "draft unified diffs. Only edit files marked editable. "
            "Do not add commentary outside the JSON response. "
            "Each patch diff must be a unified diff with @@ headers and enough context. "
            "If you need more information, add a question to questions."
        )
        if args.plan_only:
            base_prompt += " Return an empty patches array."
        if args.system_prompt:
            base_prompt = f"{args.system_prompt.strip()}\\n\\n{base_prompt}"
        system_prompt = (
            f"{base_prompt}\\n\\nReply ONLY with valid JSON that conforms to the JSON "
            f"schema below. Do not include extra keys, comments, or markdown.\\n"
            f"JSON schema:\\n{schema_text}"
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        for msg in self._normalize_messages(args.messages):
            messages.append(msg)

        user_prompt = self._build_user_prompt(task, context_files, tree, args)
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _normalize_messages(
        self, messages: list[ClaudeCodeMessage] | None
    ) -> list[dict[str, str]]:
        if not messages:
            return []
        normalized: list[dict[str, str]] = []
        for msg in messages:
            role = (msg.role or "").strip()
            content = (msg.content or "").strip()
            if not role or not content:
                continue
            normalized.append({"role": role, "content": content})
        return normalized

    def _build_user_prompt(
        self,
        task: str,
        context_files: list[_ContextFile],
        tree: list[str],
        args: ClaudeCodeSessionArgs,
    ) -> str:
        lines: list[str] = ["Task:", task]
        if args.task_context:
            lines.extend(["", "Additional context:", args.task_context.strip()])

        if tree:
            lines.extend(["", "Project tree:"])
            lines.extend(tree)

        if context_files:
            lines.append("")
            lines.append("Files:")
            for item in context_files:
                tag = "editable" if item.info.editable else "readonly"
                status = "missing" if item.info.missing else "present"
                truncated = "truncated" if item.info.truncated else "full"
                lines.append(
                    f"FILE {item.info.path} [{tag}, {status}, {truncated}]"
                )
                if item.content:
                    lines.append(item.content)
                else:
                    lines.append("")
                lines.append("END FILE")
        else:
            lines.append("")
            lines.append("No file context provided.")

        if args.paths:
            lines.append("")
            lines.append("Editable paths:")
            for path in args.paths:
                if path and str(path).strip():
                    lines.append(f"- {path}")
        if args.context_paths:
            lines.append("")
            lines.append("Read-only context paths:")
            for path in args.context_paths:
                if path and str(path).strip():
                    lines.append(f"- {path}")

        if args.allow_unlisted:
            lines.append("")
            lines.append("Patches may target any file within the project root.")

        return "\\n".join(lines).strip()

    def _call_llm_json(
        self,
        schema: dict,
        messages: list[dict[str, str]],
        args: ClaudeCodeSessionArgs,
    ) -> tuple[dict | None, list[str], str | None, int]:
        errors: list[str] = []
        max_retries = (
            args.max_retries if args.max_retries is not None else self.config.max_retries
        )
        raw: str | None = None
        attempts = 0
        current_messages = list(messages)

        for attempt in range(max_retries + 1):
            attempts = attempt + 1
            raw = self._call_llm(current_messages, args)
            parsed, parse_error = self._parse_json(raw, args.strict_json)
            if parse_error:
                errors.append(parse_error)
                if attempt < max_retries:
                    current_messages = self._append_retry(
                        current_messages, raw, parse_error
                    )
                    continue
                return None, errors, raw, attempts

            validation_errors = validate_args(schema, parsed)
            if validation_errors:
                errors.extend(validation_errors)
                if attempt < max_retries:
                    current_messages = self._append_retry(
                        current_messages, raw, "; ".join(validation_errors)
                    )
                    continue
                return None, errors, raw, attempts
            return parsed, errors, raw, attempts

        return None, errors, raw, attempts

    def _append_retry(
        self, messages: list[dict[str, str]], raw: str, error: str
    ) -> list[dict[str, str]]:
        retry_prompt = (
            "Your previous output was invalid.\\n"
            f"Error: {error}\\n"
            "Reply again with ONLY valid JSON that matches the schema."
        )
        return messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": retry_prompt},
        ]

    def _call_llm(self, messages: list[dict[str, str]], args: ClaudeCodeSessionArgs) -> str:
        api_base = (args.llm_api_base or self.config.llm_api_base).rstrip("/")
        url = api_base + "/chat/completions"
        payload = {
            "model": self._resolve_model(args),
            "messages": messages,
            "temperature": args.llm_temperature,
            "max_tokens": args.llm_max_tokens,
            "stream": bool(args.llm_stream),
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=600)
        except urllib.error.URLError as exc:
            raise ToolError(f"LLM request failed: {exc}") from exc

        if not args.llm_stream:
            body = resp.read().decode("utf-8")
            parsed = json.loads(body)
            return parsed["choices"][0]["message"].get("content", "").strip()

        return self._read_streaming_response(resp)

    def _read_streaming_response(self, resp) -> str:
        parts: list[str] = []
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            if line.startswith("data:"):
                line = line[len("data:") :].strip()
            if line == "[DONE]":
                break
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            choice = chunk.get("choices", [{}])[0]
            delta = choice.get("delta") or choice.get("message") or {}
            content = delta.get("content")
            if content:
                parts.append(content)
                sys.stdout.write(content)
                sys.stdout.flush()
        if parts:
            sys.stdout.write("\\n")
            sys.stdout.flush()
        return "".join(parts).strip()

    def _parse_json(
        self, raw: str, strict: bool
    ) -> tuple[Any | None, str | None]:
        text = raw.strip()
        if not text:
            return None, "Empty output"
        if strict:
            try:
                return json.loads(text), None
            except json.JSONDecodeError as exc:
                return None, f"JSON parse error: {exc}"
        decoder = json.JSONDecoder()
        start = min(
            [i for i in (text.find("{"), text.find("[")) if i != -1],
            default=-1,
        )
        if start == -1:
            return None, "No JSON object or array found"
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError as exc:
            return None, f"JSON parse error: {exc}"
        return obj, None

    def _resolve_model(self, args: ClaudeCodeSessionArgs) -> str:
        return args.llm_model or self.config.llm_model

    def _coerce_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []

    def _apply_patch_specs(
        self,
        patch_specs: Any,
        editable_paths: list[Path],
        context_paths: list[Path],
        apply_patches: bool,
        args: ClaudeCodeSessionArgs,
        warnings: list[str],
    ) -> tuple[list[PatchResult], list[str]]:
        errors: list[str] = []
        results: list[PatchResult] = []

        if not isinstance(patch_specs, list):
            errors.append("patches must be a list.")
            return results, errors

        editable_set = {path for path in editable_paths}
        readonly_set = {path for path in context_paths}
        allow_unlisted = bool(args.allow_unlisted) or not editable_set
        if not editable_set and not args.allow_unlisted:
            warnings.append(
                "No editable paths provided; allowing patches to any file in root."
            )
            allow_unlisted = True

        max_diff_bytes = (
            args.max_diff_bytes
            if args.max_diff_bytes is not None
            else self.config.max_diff_bytes
        )
        create_backup = (
            args.create_backup
            if args.create_backup is not None
            else self.config.create_backup
        )

        file_states: dict[Path, _FileState] = {}
        backups_written: set[Path] = set()

        for idx, patch in enumerate(patch_specs, start=1):
            if not isinstance(patch, dict):
                errors.append(f"patches[{idx}] is not an object.")
                continue
            try:
                proposal = PatchProposal(**patch)
            except Exception as exc:
                errors.append(f"patches[{idx}] invalid: {exc}")
                continue

            raw_path = proposal.path.strip()
            diff_text = proposal.diff
            summary = proposal.summary
            allow_create = (
                proposal.allow_create
                if proposal.allow_create is not None
                else args.allow_create
            )

            if not raw_path:
                errors.append(f"patches[{idx}] path is empty.")
                continue
            if not diff_text.strip():
                errors.append(f"patches[{idx}] diff is empty.")
                continue

            try:
                resolved = self._resolve_patch_path(raw_path)
            except ToolError as exc:
                results.append(
                    PatchResult(
                        path=raw_path,
                        diff=diff_text,
                        summary=summary,
                        allow_create=bool(allow_create),
                        applied=False,
                        written=False,
                        valid=False,
                        hunks=0,
                        lines_added=0,
                        lines_removed=0,
                        error=str(exc),
                    )
                )
                errors.append(f"patches[{idx}]: {exc}")
                continue

            if resolved in readonly_set:
                error = f"patches[{idx}] targets read-only file: {resolved}"
                errors.append(error)
                results.append(
                    PatchResult(
                        path=str(resolved),
                        diff=diff_text,
                        summary=summary,
                        allow_create=bool(allow_create),
                        applied=False,
                        written=False,
                        valid=False,
                        hunks=0,
                        lines_added=0,
                        lines_removed=0,
                        error=error,
                    )
                )
                continue

            if not allow_unlisted and resolved not in editable_set:
                error = f"patches[{idx}] path not in editable list: {resolved}"
                errors.append(error)
                results.append(
                    PatchResult(
                        path=str(resolved),
                        diff=diff_text,
                        summary=summary,
                        allow_create=bool(allow_create),
                        applied=False,
                        written=False,
                        valid=False,
                        hunks=0,
                        lines_added=0,
                        lines_removed=0,
                        error=error,
                    )
                )
                continue

            state = file_states.get(resolved)
            if state is None:
                try:
                    state = self._load_file_state(resolved, bool(allow_create))
                except ToolError as exc:
                    error = f"patches[{idx}] failed to load file: {exc}"
                    errors.append(error)
                    results.append(
                        PatchResult(
                            path=str(resolved),
                            diff=diff_text,
                            summary=summary,
                            allow_create=bool(allow_create),
                            applied=False,
                            written=False,
                            valid=False,
                            hunks=0,
                            lines_added=0,
                            lines_removed=0,
                            error=error,
                        )
                    )
                    continue
                file_states[resolved] = state

            try:
                applied, hunks, added, removed = self._apply_diff_to_state(
                    state, diff_text, max_diff_bytes
                )
            except ToolError as exc:
                error = f"patches[{idx}] failed: {exc}"
                errors.append(error)
                results.append(
                    PatchResult(
                        path=str(resolved),
                        diff=diff_text,
                        summary=summary,
                        allow_create=bool(allow_create),
                        applied=False,
                        written=False,
                        valid=False,
                        hunks=0,
                        lines_added=0,
                        lines_removed=0,
                        error=error,
                    )
                )
                continue

            written = False
            if apply_patches and applied:
                try:
                    self._write_file(
                        state, create_backup, backups_written
                    )
                    written = True
                except ToolError as exc:
                    error = f"patches[{idx}] write failed: {exc}"
                    errors.append(error)
                    results.append(
                        PatchResult(
                            path=str(resolved),
                            diff=diff_text,
                            summary=summary,
                            allow_create=bool(allow_create),
                            applied=applied,
                            written=False,
                            valid=False,
                            hunks=hunks,
                            lines_added=added,
                            lines_removed=removed,
                            error=error,
                        )
                    )
                    continue

            results.append(
                PatchResult(
                    path=str(resolved),
                    diff=diff_text,
                    summary=summary,
                    allow_create=bool(allow_create),
                    applied=applied,
                    written=written,
                    valid=True,
                    hunks=hunks,
                    lines_added=added,
                    lines_removed=removed,
                    error=None,
                )
            )

        return results, errors

    def _resolve_patch_path(self, raw: str) -> Path:
        base = self.config.effective_workdir.resolve()
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = base / path
        path = path.resolve()
        try:
            path.relative_to(base)
        except ValueError as exc:
            raise ToolError(f"Path must stay within project root: {base}") from exc
        return path

    def _load_file_state(self, path: Path, allow_create: bool) -> _FileState:
        if path.exists():
            try:
                content = path.read_text("utf-8", errors="ignore")
            except OSError as exc:
                raise ToolError(f"Failed to read {path}: {exc}") from exc
            newline = "\\r\\n" if "\\r\\n" in content else "\\n"
            ends_with_newline = content.endswith(("\\n", "\\r\\n"))
            return _FileState(
                path=path,
                original=content,
                current=content,
                newline=newline,
                ends_with_newline=ends_with_newline,
            )
        if not allow_create:
            raise ToolError(f"File not found: {path}")
        return _FileState(
            path=path,
            original="",
            current="",
            newline="\\n",
            ends_with_newline=True,
        )

    def _apply_diff_to_state(
        self, state: _FileState, diff: str, max_diff_bytes: int
    ) -> tuple[bool, int, int, int]:
        diff_bytes = len(diff.encode("utf-8"))
        if max_diff_bytes > 0 and diff_bytes > max_diff_bytes:
            raise ToolError("Diff exceeds max_diff_bytes limit.")

        hunks, added, removed, saw_no_newline = self._parse_diff(diff)
        if not hunks:
            raise ToolError("No hunks found in diff.")

        ends_with_newline = state.ends_with_newline
        if saw_no_newline:
            ends_with_newline = False

        original_lines = state.current.splitlines()
        new_lines = self._apply_hunks(original_lines, hunks)

        new_content = state.newline.join(new_lines)
        if ends_with_newline and new_content:
            new_content += state.newline

        applied = new_content != state.current
        state.current = new_content
        state.ends_with_newline = ends_with_newline
        return applied, len(hunks), added, removed

    def _parse_diff(self, diff: str) -> tuple[list[_Hunk], int, int, bool]:
        lines = diff.splitlines()
        hunks: list[_Hunk] = []
        current: list[str] | None = None
        added = 0
        removed = 0
        saw_no_newline = False
        old_start = old_count = new_start = new_count = 0

        for line in lines:
            if line.startswith("\\\\ No newline at end of file"):
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
                elif prefix == "\\\\":
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
                f"Expected {text!r}, got {lines[index]!r}.\\n{context}"
            )

    @staticmethod
    def _format_context(lines: list[str], index: int, radius: int = 2) -> str:
        start = max(0, index - radius)
        end = min(len(lines), index + radius + 1)
        context_lines = []
        for i in range(start, end):
            marker = ">>>" if i == index else "   "
            context_lines.append(f"{marker} {i + 1:4d}: {lines[i]}")
        return "\\n".join(context_lines)

    def _write_file(
        self, state: _FileState, create_backup: bool, backups_written: set[Path]
    ) -> None:
        if not state.path.exists():
            state.path.parent.mkdir(parents=True, exist_ok=True)
        if create_backup and state.path.exists() and state.path not in backups_written:
            backup_path = state.path.with_suffix(state.path.suffix + ".bak")
            backup_path.write_text(state.original, encoding="utf-8")
            backups_written.add(state.path)
        state.path.write_text(state.current, encoding="utf-8")

    def _resolve_tree_root(self, raw: str | None, base: Path) -> Path:
        if raw:
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = base / path
            path = path.resolve()
        else:
            path = base
        try:
            path.relative_to(base)
        except ValueError as exc:
            raise ToolError(
                f"Tree root must stay within project root: {base}"
            ) from exc
        if not path.exists():
            raise ToolError(f"Tree root not found: {path}")
        if not path.is_dir():
            raise ToolError(f"Tree root is not a directory: {path}")
        return path

    def _build_tree(
        self,
        root: Path,
        include_globs: list[str] | None,
        exclude_globs: list[str] | None,
        max_depth: int | None,
        max_files: int | None,
    ) -> list[str]:
        include = include_globs or ["**/*"]
        exclude = exclude_globs or self.config.default_exclude_globs
        depth_limit = (
            max_depth
            if max_depth is not None
            else self.config.default_max_depth
        )
        file_limit = (
            max_files
            if max_files is not None
            else self.config.default_max_tree_files
        )

        results: list[str] = []
        root_depth = len(root.parts)

        for dirpath, dirnames, filenames in os.walk(root):
            depth = len(Path(dirpath).parts) - root_depth
            if depth_limit >= 0 and depth > depth_limit:
                dirnames[:] = []
                continue

            for name in filenames:
                rel_path = Path(dirpath) / name
                rel = rel_path.relative_to(root).as_posix()
                if not self._matches_any(rel, include):
                    continue
                if self._matches_any(rel, exclude):
                    continue
                results.append(rel)
                if file_limit > 0 and len(results) >= file_limit:
                    return results
        return results

    @staticmethod
    def _matches_any(path: str, patterns: list[str]) -> bool:
        for pattern in patterns:
            if fnmatch.fnmatch(path, pattern):
                return True
        return False

    def _display_path(self, path: Path) -> str:
        base = self.config.effective_workdir.resolve()
        try:
            return str(path.relative_to(base))
        except ValueError:
            return str(path)

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ClaudeCodeSessionArgs):
            return ToolCallDisplay(summary="claude_code_session")
        return ToolCallDisplay(
            summary="claude_code_session",
            details={
                "paths": len(event.args.paths or []),
                "context_paths": len(event.args.context_paths or []),
                "plan_only": event.args.plan_only,
                "apply_patches": event.args.apply_patches,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ClaudeCodeSessionResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        applied = sum(1 for patch in event.result.patches if patch.applied)
        message = f"Code session complete ({len(event.result.patches)} patches)"
        if applied:
            message = f"Code session complete ({applied} applied patches)"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "plan": event.result.plan,
                "patches": event.result.patches,
                "commands": event.result.commands,
                "questions": event.result.questions,
                "notes": event.result.notes,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Running code session"
