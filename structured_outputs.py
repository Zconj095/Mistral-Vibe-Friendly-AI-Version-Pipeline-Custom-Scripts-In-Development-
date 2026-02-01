from __future__ import annotations

import importlib.util
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, Field

try:
    from actions_lib import validate_args
except ModuleNotFoundError:  # Fallback when tools directory is not on sys.path.
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
from vibe.core.types import ToolCallEvent, ToolResultEvent

if TYPE_CHECKING:
    from jsonschema.exceptions import ValidationError


class StructuredOutputsArgs(BaseModel):
    schema: dict | str = Field(description="JSON schema dict or JSON string.")
    prompt: str | None = Field(
        default=None, description="User prompt to generate structured output."
    )
    messages: list[dict] | None = Field(
        default=None,
        description="Optional chat messages. Overrides prompt if provided.",
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prefix."
    )
    llm_api_base: str | None = Field(
        default=None, description="OpenAI-compatible API base URL."
    )
    llm_model: str | None = Field(
        default=None, description="LLM model name."
    )
    llm_temperature: float = Field(default=0.2, description="LLM temperature.")
    llm_max_tokens: int = Field(default=500, description="LLM max tokens.")
    llm_stream: bool = Field(default=False, description="Stream LLM tokens.")
    max_retries: int = Field(default=2, description="Retry count for invalid output.")
    strict_json: bool = Field(
        default=True,
        description="Require JSON-only output with no extra text.",
    )
    include_raw: bool = Field(
        default=True,
        description="Include raw model output in the result.",
    )


class StructuredOutputsResult(BaseModel):
    success: bool
    output: Any | None
    raw: str | None
    errors: list[str]
    attempts: int
    llm_model: str
    validation_backend: str


class StructuredOutputsConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    llm_api_base: str = "http://127.0.0.1:11434/v1"
    llm_model: str = "gpt-oss:latest"
    default_max_retries: int = 2
    validation_backend: str = "auto"


class StructuredOutputsState(BaseToolState):
    pass


class StructuredOutputs(
    BaseTool[
        StructuredOutputsArgs,
        StructuredOutputsResult,
        StructuredOutputsConfig,
        StructuredOutputsState,
    ],
    ToolUIData[StructuredOutputsArgs, StructuredOutputsResult],
):
    description: ClassVar[str] = (
        "Generate JSON that conforms to a schema, with validation and retries."
    )

    async def run(self, args: StructuredOutputsArgs) -> StructuredOutputsResult:
        schema = self._load_schema(args.schema)
        messages = self._build_messages(args, schema)
        max_retries = (
            args.max_retries
            if args.max_retries is not None
            else self.config.default_max_retries
        )

        errors: list[str] = []
        last_raw: str | None = None
        attempts = 0
        validation_backend = self._pick_validation_backend(schema)

        for attempt in range(max_retries + 1):
            attempts = attempt + 1
            raw = self._call_llm(messages, args)
            last_raw = raw
            parsed, parse_error = self._parse_json(raw, args.strict_json)
            if parse_error:
                errors.append(parse_error)
                if attempt < max_retries:
                    messages = self._append_retry(messages, raw, parse_error)
                    continue
                break

            validation_errors = self._validate(schema, parsed, validation_backend)
            if validation_errors:
                errors.extend(validation_errors)
                if attempt < max_retries:
                    messages = self._append_retry(
                        messages, raw, "; ".join(validation_errors)
                    )
                    continue
                break

            return StructuredOutputsResult(
                success=True,
                output=parsed,
                raw=raw if args.include_raw else None,
                errors=[],
                attempts=attempts,
                llm_model=self._resolve_model(args),
                validation_backend=validation_backend,
            )

        return StructuredOutputsResult(
            success=False,
            output=None,
            raw=last_raw if args.include_raw else None,
            errors=errors,
            attempts=attempts,
            llm_model=self._resolve_model(args),
            validation_backend=validation_backend,
        )

    def _load_schema(self, schema: dict | str) -> dict:
        if isinstance(schema, dict):
            return schema
        if isinstance(schema, str):
            value = schema.strip()
            if not value:
                raise ToolError("schema cannot be empty.")
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ToolError(f"schema JSON parse error: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ToolError("schema JSON must be an object.")
            return parsed
        raise ToolError("schema must be a dict or JSON string.")

    def _build_messages(
        self, args: StructuredOutputsArgs, schema: dict
    ) -> list[dict[str, str]]:
        base_messages = []
        if args.messages:
            if not isinstance(args.messages, list):
                raise ToolError("messages must be a list.")
            base_messages = list(args.messages)
        elif args.prompt:
            base_messages = [{"role": "user", "content": args.prompt}]
        else:
            raise ToolError("Provide either messages or prompt.")

        schema_text = json.dumps(schema, ensure_ascii=True)
        instruction = (
            "Reply ONLY with valid JSON that conforms to the JSON schema below. "
            "Do not include extra keys, comments, or markdown.\n"
            f"JSON schema:\n{schema_text}"
        )
        if args.system_prompt:
            instruction = f"{args.system_prompt}\n\n{instruction}"

        return [{"role": "system", "content": instruction}] + base_messages

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

    def _call_llm(self, messages: list[dict[str, str]], args: StructuredOutputsArgs) -> str:
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
            sys.stdout.write("\n")
            sys.stdout.flush()
        return "".join(parts).strip()

    def _parse_json(self, raw: str, strict: bool) -> tuple[Any | None, str | None]:
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

    def _validate(
        self, schema: dict, value: Any, backend: str
    ) -> list[str]:
        if backend == "jsonschema":
            return self._validate_jsonschema(schema, value)
        if backend == "basic":
            return validate_args(schema, value)
        errors = self._validate_jsonschema(schema, value)
        if not errors:
            return errors
        return errors + validate_args(schema, value)

    def _validate_jsonschema(self, schema: dict, value: Any) -> list[str]:
        try:
            import jsonschema
        except ModuleNotFoundError:
            return ["jsonschema not installed; falling back to basic validation."]
        try:
            validator = jsonschema.Draft202012Validator(schema)
        except Exception as exc:
            return [f"Invalid JSON schema: {exc}"]
        errors = []
        for err in validator.iter_errors(value):
            errors.append(self._format_jsonschema_error(err))
        return errors

    def _format_jsonschema_error(self, err: "ValidationError") -> str:
        path = ".".join(str(p) for p in err.path) if err.path else "root"
        return f"{path}: {err.message}"

    def _pick_validation_backend(self, schema: dict) -> str:
        preference = (self.config.validation_backend or "auto").lower()
        if preference in {"jsonschema", "basic"}:
            return preference
        try:
            import jsonschema  # noqa: F401
            return "jsonschema"
        except ModuleNotFoundError:
            return "basic"

    def _resolve_model(self, args: StructuredOutputsArgs) -> str:
        return args.llm_model or self.config.llm_model

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, StructuredOutputsArgs):
            return ToolCallDisplay(summary="structured_outputs")
        return ToolCallDisplay(
            summary="structured_outputs",
            details={
                "llm_model": event.args.llm_model,
                "max_retries": event.args.max_retries,
                "strict_json": event.args.strict_json,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if isinstance(event.result, StructuredOutputsResult):
            message = "Structured output complete"
            if not event.result.success:
                message = "Structured output failed"
            return ToolResultDisplay(
                success=event.result.success,
                message=message,
                details={
                    "output": event.result.output,
                    "raw": event.result.raw,
                    "errors": event.result.errors,
                    "attempts": event.result.attempts,
                },
            )
        return ToolResultDisplay(success=True, message="Structured output complete")

    @classmethod
    def get_status_text(cls) -> str:
        return "Generating structured output"
