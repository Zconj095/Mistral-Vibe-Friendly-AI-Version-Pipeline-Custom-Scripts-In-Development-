from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
import urllib.error
import urllib.request
import uuid

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


DEFAULT_STORE_PATH = Path.home() / ".vibe" / "memory" / "batch_jobs.json"

VALID_ACTIONS = {
    "submit",
    "run",
    "list",
    "get",
    "delete",
    "clear",
    "status",
}
VALID_STATUS = {"queued", "running", "completed", "failed"}
VALID_EFFORT = {"low", "medium", "high", "auto", "custom"}
VALID_VALIDATION = {"none", "json_schema"}

DEFAULT_PROFILES: dict[str, dict[str, Any]] = {
    "low": {
        "temperature": 0.6,
        "max_tokens": 400,
        "max_retries": 0,
        "validation_mode": "none",
    },
    "medium": {
        "temperature": 0.3,
        "max_tokens": 800,
        "max_retries": 1,
        "validation_mode": "json_schema",
    },
    "high": {
        "temperature": 0.2,
        "max_tokens": 1200,
        "max_retries": 3,
        "validation_mode": "json_schema",
    },
}


class BatchMessage(BaseModel):
    role: str
    content: str


class BatchJobInput(BaseModel):
    id: str | None = Field(default=None, description="Optional job id.")
    prompt: str | None = Field(
        default=None, description="User prompt for the job."
    )
    messages: list[BatchMessage] | None = Field(
        default=None, description="Optional chat messages."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt for the job."
    )
    model: str | None = Field(default=None, description="Override model name.")
    temperature: float | None = Field(
        default=None, description="Override temperature."
    )
    max_tokens: int | None = Field(
        default=None, description="Override max tokens."
    )
    stream: bool | None = Field(default=None, description="Stream output.")
    output_schema: dict | str | None = Field(
        default=None, description="Optional JSON schema for validation."
    )
    strict_json: bool | None = Field(
        default=None, description="Require JSON-only output."
    )
    max_retries: int | None = Field(
        default=None, description="Max retries for invalid JSON."
    )
    validation_mode: str | None = Field(
        default=None, description="none or json_schema."
    )
    effort: str | None = Field(
        default=None, description="low, medium, high, auto, or custom."
    )
    metadata: dict | None = Field(
        default=None, description="Optional metadata for the job."
    )


class BatchProcessingArgs(BaseModel):
    action: str | None = Field(
        default="run", description="submit, run, list, get, delete, clear, status."
    )
    jobs: list[BatchJobInput] | None = Field(
        default=None, description="Jobs to submit or run."
    )
    job_ids: list[str] | None = Field(
        default=None, description="Job ids for run/get/delete."
    )
    batch_id: str | None = Field(
        default=None, description="Batch id to scope operations."
    )
    limit: int | None = Field(
        default=None, description="Limit number of jobs processed or listed."
    )
    include_results: bool = Field(
        default=True, description="Include response payloads in output."
    )
    include_request: bool = Field(
        default=False, description="Include request payloads in output."
    )
    shared_system_prompt: str | None = Field(
        default=None, description="Shared system prompt prefix."
    )
    shared_context: str | None = Field(
        default=None, description="Shared context block."
    )
    shared_context_path: str | None = Field(
        default=None, description="Path to shared context file."
    )
    shared_messages: list[BatchMessage] | None = Field(
        default=None, description="Shared messages appended before job input."
    )
    llm_api_base: str | None = Field(
        default=None, description="OpenAI-compatible API base URL."
    )
    llm_model: str | None = Field(
        default=None, description="Default LLM model."
    )
    temperature: float | None = Field(
        default=None, description="Default temperature for jobs."
    )
    max_tokens: int | None = Field(
        default=None, description="Default max tokens for jobs."
    )
    stream: bool | None = Field(
        default=None, description="Default streaming flag for jobs."
    )
    output_schema: dict | str | None = Field(
        default=None, description="Default JSON schema for jobs."
    )
    strict_json: bool = Field(
        default=True, description="Require JSON-only output when schema is set."
    )
    max_retries: int | None = Field(
        default=None, description="Default max retries for schema validation."
    )
    validation_mode: str | None = Field(
        default=None, description="Default validation mode."
    )
    effort: str | None = Field(
        default=None, description="Default effort profile."
    )
    task_chars: int | None = Field(
        default=None, description="Optional char estimate for auto effort."
    )


class BatchJobOutput(BaseModel):
    id: str
    batch_id: str | None
    status: str
    response: str | None
    parsed_output: Any | None
    attempts: int
    error: str | None
    metadata: dict | None
    model: str | None
    validation_errors: list[str] | None
    request: dict | None


class BatchProcessingResult(BaseModel):
    action: str
    batch_id: str | None
    jobs: list[BatchJobOutput]
    summary: dict[str, int]
    warnings: list[str]
    errors: list[str]


class BatchProcessingConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    store_path: Path = Field(
        default=DEFAULT_STORE_PATH,
        description="Path to the batch job store.",
    )
    llm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL.",
    )
    llm_model: str = Field(
        default="gpt-oss:latest", description="Default LLM model name."
    )
    temperature: float = Field(default=0.3, description="Default temperature.")
    max_tokens: int = Field(default=800, description="Default max tokens.")
    max_retries: int = Field(default=1, description="Default max retries.")
    auto_token_thresholds: list[int] = Field(
        default_factory=lambda: [200, 800],
        description="Token thresholds for auto effort (low, medium).",
    )
    auto_char_thresholds: list[int] = Field(
        default_factory=lambda: [800, 3200],
        description="Character thresholds for auto effort (low, medium).",
    )


class BatchProcessingState(BaseToolState):
    pass


class BatchProcessing(
    BaseTool[
        BatchProcessingArgs,
        BatchProcessingResult,
        BatchProcessingConfig,
        BatchProcessingState,
    ],
    ToolUIData[BatchProcessingArgs, BatchProcessingResult],
):
    description: ClassVar[str] = (
        "Queue and execute batches of LLM requests locally."
    )

    async def run(self, args: BatchProcessingArgs) -> BatchProcessingResult:
        action = (args.action or "run").strip().lower()
        if action not in VALID_ACTIONS:
            raise ToolError(
                f"action must be one of: {', '.join(sorted(VALID_ACTIONS))}"
            )

        warnings: list[str] = []
        errors: list[str] = []
        store_path = self.config.store_path

        if action in {"submit", "run", "list", "get", "delete", "clear", "status"}:
            store = self._load_store(store_path)
        else:
            store = {}

        if action == "submit":
            batch_id = args.batch_id or self._new_batch_id()
            jobs = self._resolve_jobs(args, batch_id, warnings)
            for job in jobs:
                store["jobs"][job["id"]] = job
            self._save_store(store_path, store)
            outputs = self._build_outputs(jobs, args, include_response=False)
            return self._result(action, batch_id, outputs, warnings, errors)

        if action == "run":
            batch_id = args.batch_id
            if args.jobs:
                jobs = self._resolve_jobs(args, batch_id, warnings, persist=False)
                outputs = self._run_jobs(jobs, args, warnings)
                return self._result(action, batch_id, outputs, warnings, errors)

            jobs = self._select_store_jobs(store, args, warnings)
            outputs = self._run_jobs(jobs, args, warnings, store=store)
            self._save_store(store_path, store)
            return self._result(action, batch_id, outputs, warnings, errors)

        if action == "list":
            jobs = self._filter_store_jobs(store, args, warnings)
            outputs = self._build_outputs(jobs, args, include_response=False)
            return self._result(action, args.batch_id, outputs, warnings, errors)

        if action == "status":
            jobs = self._filter_store_jobs(store, args, warnings)
            outputs = self._build_outputs(jobs, args, include_response=False)
            return self._result(action, args.batch_id, outputs, warnings, errors)

        if action == "get":
            jobs = self._filter_store_jobs(store, args, warnings)
            outputs = self._build_outputs(jobs, args, include_response=True)
            return self._result(action, args.batch_id, outputs, warnings, errors)

        if action == "delete":
            deleted = self._delete_jobs(store, args, warnings)
            self._save_store(store_path, store)
            outputs = self._build_outputs(deleted, args, include_response=False)
            return self._result(action, args.batch_id, outputs, warnings, errors)

        if action == "clear":
            cleared = self._clear_store(store, args, warnings)
            if cleared:
                self._save_store(store_path, store)
            outputs = self._build_outputs(cleared, args, include_response=False)
            return self._result(action, args.batch_id, outputs, warnings, errors)

        return self._result(action, None, [], warnings, errors)

    def _resolve_jobs(
        self,
        args: BatchProcessingArgs,
        batch_id: str | None,
        warnings: list[str],
        persist: bool = True,
    ) -> list[dict[str, Any]]:
        if not args.jobs:
            raise ToolError("jobs is required.")

        shared_context = self._resolve_shared_context(args, warnings)
        shared_system = args.shared_system_prompt or ""
        shared_messages = self._normalize_messages(args.shared_messages)
        static_hash = self._hash_text(shared_system + shared_context)

        jobs: list[dict[str, Any]] = []
        for job in args.jobs:
            job_id = (job.id or self._new_job_id()).strip()
            if not job_id:
                raise ToolError("job id cannot be empty.")
            spec = self._build_job_spec(job, args, warnings)
            spec["shared_system_prompt"] = shared_system or None
            spec["shared_context"] = shared_context or None
            spec["shared_messages"] = shared_messages or None
            spec["static_context_hash"] = (
                static_hash if shared_context or shared_system else None
            )
            record = {
                "id": job_id,
                "batch_id": batch_id,
                "status": "queued",
                "created_at": self._now_iso(),
                "updated_at": self._now_iso(),
                "request": spec,
                "response": None,
                "parsed_output": None,
                "attempts": 0,
                "error": None,
                "validation_errors": None,
                "metadata": job.metadata if isinstance(job.metadata, dict) else None,
            }
            if not persist:
                record["status"] = "queued"
            jobs.append(record)
        return jobs

    def _build_job_spec(
        self, job: BatchJobInput, args: BatchProcessingArgs, warnings: list[str]
    ) -> dict[str, Any]:
        effort = self._normalize_effort(job.effort or args.effort)
        task_chars = self._estimate_task_chars(job, args)
        if effort == "auto":
            effort = self._auto_effort(task_chars, warnings)

        profile = DEFAULT_PROFILES.get(effort or "medium", DEFAULT_PROFILES["medium"])

        schema = self._load_schema(job.output_schema or args.output_schema, warnings)
        validation_mode = (
            job.validation_mode
            or args.validation_mode
            or profile.get("validation_mode")
            or "none"
        )
        if validation_mode not in VALID_VALIDATION:
            raise ToolError("validation_mode must be none or json_schema.")
        if schema is None and validation_mode == "json_schema":
            warnings.append(
                "validation_mode json_schema requires output_schema; disabling validation."
            )
            validation_mode = "none"

        return {
            "prompt": job.prompt,
            "messages": self._normalize_messages(job.messages),
            "system_prompt": job.system_prompt,
            "model": job.model or args.llm_model or self.config.llm_model,
            "temperature": self._resolve_float(
                job.temperature,
                args.temperature,
                profile.get("temperature"),
                self.config.temperature,
            ),
            "max_tokens": self._resolve_int(
                job.max_tokens,
                args.max_tokens,
                profile.get("max_tokens"),
                self.config.max_tokens,
            ),
            "stream": bool(
                job.stream if job.stream is not None else args.stream or False
            ),
            "output_schema": schema,
            "strict_json": bool(
                job.strict_json if job.strict_json is not None else args.strict_json
            ),
            "max_retries": self._resolve_int(
                job.max_retries,
                args.max_retries,
                profile.get("max_retries"),
                self.config.max_retries,
            ),
            "validation_mode": validation_mode,
        }

    def _resolve_shared_context(
        self, args: BatchProcessingArgs, warnings: list[str]
    ) -> str:
        blocks: list[str] = []
        if args.shared_context:
            blocks.append(args.shared_context.strip())
        if args.shared_context_path:
            path = self._resolve_path(args.shared_context_path)
            try:
                blocks.append(path.read_text("utf-8", errors="ignore"))
            except OSError as exc:
                warnings.append(f"Failed to read shared context: {exc}")
        return "\n".join([b for b in blocks if b]).strip()

    def _resolve_path(self, raw_path: str) -> Path:
        if not raw_path.strip():
            raise ToolError("Path cannot be empty.")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.config.effective_workdir / path
        resolved = path.resolve()
        if not resolved.exists():
            raise ToolError(f"Path not found: {resolved}")
        if resolved.is_dir():
            raise ToolError(f"Path is a directory: {resolved}")
        return resolved

    def _normalize_messages(
        self, messages: list[BatchMessage] | None
    ) -> list[dict[str, str]] | None:
        if not messages:
            return None
        normalized = []
        for msg in messages:
            content = (msg.content or "").strip()
            if not content:
                continue
            normalized.append({"role": msg.role, "content": content})
        return normalized or None

    def _load_schema(
        self, schema: dict | str | None, warnings: list[str]
    ) -> dict | None:
        if schema is None:
            return None
        if isinstance(schema, dict):
            return schema
        if isinstance(schema, str):
            value = schema.strip()
            if not value:
                warnings.append("output_schema provided but empty.")
                return None
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                warnings.append(f"output_schema JSON parse error: {exc}")
                return None
            if isinstance(parsed, dict):
                return parsed
            warnings.append("output_schema JSON must be an object.")
            return None
        warnings.append("output_schema must be a dict or JSON string.")
        return None

    def _normalize_effort(self, effort: str | None) -> str | None:
        if effort is None:
            return None
        value = effort.strip().lower()
        if value not in VALID_EFFORT:
            raise ToolError(
                f"effort must be one of: {', '.join(sorted(VALID_EFFORT))}"
            )
        return value

    def _estimate_task_chars(
        self, job: BatchJobInput, args: BatchProcessingArgs
    ) -> int:
        if args.task_chars is not None and args.task_chars > 0:
            return args.task_chars
        total = 0
        if job.prompt:
            total += len(job.prompt)
        if job.messages:
            total += sum(len(msg.content) for msg in job.messages if msg.content)
        return total

    def _auto_effort(self, task_chars: int, warnings: list[str]) -> str:
        if task_chars <= 0:
            warnings.append(
                "auto effort requested but task size missing; using medium."
            )
            return "medium"
        thresholds = self._sorted_thresholds(self.config.auto_char_thresholds)
        if task_chars <= thresholds[0]:
            return "low"
        if task_chars <= thresholds[1]:
            return "medium"
        return "high"

    def _sorted_thresholds(self, values: list[int]) -> list[int]:
        if len(values) < 2:
            return [800, 3200]
        thresholds = [int(values[0]), int(values[1])]
        thresholds.sort()
        return thresholds

    def _resolve_int(self, value: int | None, *fallbacks: Any) -> int:
        for item in (value, *fallbacks):
            if isinstance(item, int):
                return item
        raise ToolError("Missing integer configuration.")

    def _resolve_float(self, value: float | None, *fallbacks: Any) -> float:
        for item in (value, *fallbacks):
            if isinstance(item, (int, float)):
                return float(item)
        raise ToolError("Missing float configuration.")

    def _new_batch_id(self) -> str:
        return f"batch_{uuid.uuid4().hex}"

    def _new_job_id(self) -> str:
        return f"job_{uuid.uuid4().hex}"

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load_store(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"version": 1, "updated_at": self._now_iso(), "jobs": {}}
        try:
            raw = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ToolError(f"Failed to read batch store: {exc}") from exc
        if not isinstance(raw, dict):
            raise ToolError("Invalid batch store format.")
        if "jobs" not in raw or not isinstance(raw["jobs"], dict):
            raw["jobs"] = {}
        if "version" not in raw:
            raw["version"] = 1
        return raw

    def _save_store(self, path: Path, store: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        store["updated_at"] = self._now_iso()
        payload = json.dumps(store, ensure_ascii=True, indent=2)
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(payload, "utf-8")
        temp_path.replace(path)

    def _select_store_jobs(
        self, store: dict[str, Any], args: BatchProcessingArgs, warnings: list[str]
    ) -> list[dict[str, Any]]:
        jobs = self._filter_store_jobs(store, args, warnings)
        queued = [job for job in jobs if job.get("status") == "queued"]
        limit = args.limit if args.limit is not None else len(queued)
        if limit is not None and limit > 0:
            queued = queued[:limit]
        if not queued:
            warnings.append("No queued jobs to run.")
        return queued

    def _filter_store_jobs(
        self, store: dict[str, Any], args: BatchProcessingArgs, warnings: list[str]
    ) -> list[dict[str, Any]]:
        jobs = list(store.get("jobs", {}).values())
        if args.batch_id:
            jobs = [job for job in jobs if job.get("batch_id") == args.batch_id]
        if args.job_ids:
            ids = {
                job_id.strip()
                for job_id in args.job_ids
                if job_id and job_id.strip()
            }
            jobs = [job for job in jobs if job.get("id") in ids]
        if args.limit is not None and args.limit > 0:
            jobs = jobs[: args.limit]
        if not jobs and (args.batch_id or args.job_ids):
            warnings.append("No matching jobs found.")
        return jobs

    def _delete_jobs(
        self, store: dict[str, Any], args: BatchProcessingArgs, warnings: list[str]
    ) -> list[dict[str, Any]]:
        jobs = self._filter_store_jobs(store, args, warnings)
        for job in jobs:
            store["jobs"].pop(job.get("id"), None)
        return jobs

    def _clear_store(
        self, store: dict[str, Any], args: BatchProcessingArgs, warnings: list[str]
    ) -> list[dict[str, Any]]:
        jobs = self._filter_store_jobs(store, args, warnings)
        if not jobs:
            warnings.append("No jobs to clear.")
            return []
        for job in jobs:
            store["jobs"].pop(job.get("id"), None)
        return jobs

    def _run_jobs(
        self,
        jobs: list[dict[str, Any]],
        args: BatchProcessingArgs,
        warnings: list[str],
        store: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not jobs:
            return []
        if args.stream and len(jobs) > 1:
            warnings.append(
                "Streaming enabled with multiple jobs; output will interleave."
            )

        outputs: list[dict[str, Any]] = []
        for job in jobs:
            job["status"] = "running"
            job["updated_at"] = self._now_iso()
            if store is not None:
                store["jobs"][job["id"]] = job
            result = self._process_job(job, args, warnings)
            outputs.append(result)
            if store is not None:
                store["jobs"][job["id"]] = result
        return outputs

    def _process_job(
        self, job: dict[str, Any], args: BatchProcessingArgs, warnings: list[str]
    ) -> dict[str, Any]:
        request = job.get("request") or {}
        messages = self._build_messages(request)
        schema = request.get("output_schema")
        strict_json = bool(request.get("strict_json", True))
        max_retries = int(request.get("max_retries", 0))
        validation_mode = request.get("validation_mode") or "none"
        temperature = float(request.get("temperature"))
        max_tokens = int(request.get("max_tokens"))
        stream = bool(request.get("stream"))
        model = request.get("model")

        attempts = 0
        response = ""
        parsed_output = None
        validation_errors: list[str] | None = None
        last_error: str | None = None

        for attempt in range(max_retries + 1):
            attempts = attempt + 1
            if last_error:
                messages = self._append_retry(messages, response, last_error)
            response = self._call_llm(
                messages,
                args,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )

            if validation_mode == "json_schema" and schema is not None:
                parsed_output, error = self._parse_json(response, strict_json)
                if error:
                    last_error = error
                    validation_errors = [error]
                    continue
                validation_errors = validate_args(schema, parsed_output)
                if validation_errors:
                    last_error = "; ".join(validation_errors)
                    continue
                last_error = None
                break
            break

        status = "completed"
        error_text = None
        if validation_mode == "json_schema" and schema is not None and validation_errors:
            status = "failed"
            error_text = "Validation failed after retries."

        job.update(
            {
                "status": status,
                "updated_at": self._now_iso(),
                "response": response,
                "parsed_output": parsed_output,
                "attempts": attempts,
                "error": error_text,
                "validation_errors": validation_errors,
                "request": request,
            }
        )
        return job

    def _build_messages(self, request: dict[str, Any]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        system_parts: list[str] = []

        shared_system = request.get("shared_system_prompt")
        shared_context = request.get("shared_context")
        if shared_system:
            system_parts.append(shared_system)
        if shared_context:
            system_parts.append(shared_context)
        if request.get("system_prompt"):
            system_parts.append(request["system_prompt"])

        schema = request.get("output_schema")
        validation_mode = request.get("validation_mode") or "none"
        if validation_mode == "json_schema" and schema is not None:
            schema_text = json.dumps(schema, ensure_ascii=True)
            schema_instruction = (
                "Reply ONLY with valid JSON that conforms to the JSON schema below. "
                "Do not include extra keys, comments, or markdown.\n"
                f"JSON schema:\n{schema_text}"
            )
            system_parts.append(schema_instruction)

        if system_parts:
            messages.append(
                {"role": "system", "content": "\n\n".join(system_parts).strip()}
            )

        shared_messages = request.get("shared_messages") or []
        for msg in shared_messages:
            messages.append({"role": msg["role"], "content": msg["content"]})

        for msg in request.get("messages") or []:
            messages.append({"role": msg["role"], "content": msg["content"]})

        prompt = request.get("prompt")
        if prompt:
            messages.append({"role": "user", "content": prompt})

        if not messages:
            raise ToolError("Job has no prompt or messages.")
        return messages

    def _append_retry(
        self, messages: list[dict[str, str]], raw: str, error: str
    ) -> list[dict[str, str]]:
        retry_prompt = (
            "Your previous output was invalid.\n"
            f"Error: {error}\n"
            "Reply again with ONLY valid JSON that matches the schema."
        )
        return messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": retry_prompt},
        ]

    def _call_llm(
        self,
        messages: list[dict[str, str]],
        args: BatchProcessingArgs,
        model: str,
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> str:
        api_base = (args.llm_api_base or self.config.llm_api_base).rstrip("/")
        url = api_base + "/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": bool(stream),
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

        if not stream:
            body = resp.read().decode("utf-8")
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError as exc:
                raise ToolError(f"LLM response parse failed: {exc}") from exc
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

    def _build_outputs(
        self,
        jobs: list[dict[str, Any]],
        args: BatchProcessingArgs,
        include_response: bool,
    ) -> list[BatchJobOutput]:
        outputs: list[BatchJobOutput] = []
        for job in jobs:
            response = (
                job.get("response")
                if include_response and args.include_results
                else None
            )
            parsed_output = (
                job.get("parsed_output")
                if include_response and args.include_results
                else None
            )
            request = job.get("request") if args.include_request else None
            outputs.append(
                BatchJobOutput(
                    id=job.get("id", ""),
                    batch_id=job.get("batch_id"),
                    status=job.get("status", "queued"),
                    response=response,
                    parsed_output=parsed_output,
                    attempts=int(job.get("attempts") or 0),
                    error=job.get("error"),
                    metadata=job.get("metadata"),
                    model=(job.get("request") or {}).get("model"),
                    validation_errors=job.get("validation_errors"),
                    request=request,
                )
            )
        return outputs

    def _result(
        self,
        action: str,
        batch_id: str | None,
        jobs: list[BatchJobOutput],
        warnings: list[str],
        errors: list[str],
    ) -> BatchProcessingResult:
        summary = {"queued": 0, "running": 0, "completed": 0, "failed": 0}
        for job in jobs:
            status = job.status
            if status in summary:
                summary[status] += 1
        return BatchProcessingResult(
            action=action,
            batch_id=batch_id,
            jobs=jobs,
            summary=summary,
            warnings=warnings,
            errors=errors,
        )

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, BatchProcessingArgs):
            return ToolCallDisplay(summary="batch_processing")
        return ToolCallDisplay(
            summary="batch_processing",
            details={
                "action": event.args.action,
                "batch_id": event.args.batch_id,
                "jobs": len(event.args.jobs or []),
                "job_ids": event.args.job_ids,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, BatchProcessingResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = f"Batch {event.result.action} complete"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "batch_id": event.result.batch_id,
                "summary": event.result.summary,
                "jobs": event.result.jobs,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Running batch processing"
