from __future__ import annotations

import importlib.util
import json
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


VALID_MODES = {"fast", "balanced", "deep", "custom"}
VALID_MERGE = {"merge", "best_of", "first"}
VALID_VALIDATION = {"none", "json_schema"}

MODE_PROFILES: dict[str, dict[str, Any]] = {
    "fast": {
        "passes": 1,
        "reasoning_max_tokens": 400,
        "answer_max_tokens": 400,
        "reasoning_temperature": 0.4,
        "answer_temperature": 0.4,
        "max_retries": 0,
    },
    "balanced": {
        "passes": 2,
        "reasoning_max_tokens": 1200,
        "answer_max_tokens": 700,
        "reasoning_temperature": 0.3,
        "answer_temperature": 0.2,
        "max_retries": 1,
    },
    "deep": {
        "passes": 3,
        "reasoning_max_tokens": 2000,
        "answer_max_tokens": 900,
        "reasoning_temperature": 0.2,
        "answer_temperature": 0.2,
        "max_retries": 2,
    },
}

SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "best_index": {"type": "integer"},
        "rationale": {"type": "string"},
    },
    "required": ["best_index"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class _PassOutput:
    answer: str
    parsed: Any | None
    validation_errors: list[str]
    scratchpad: str | None
    attempts: int


class DeepthinkMessage(BaseModel):
    role: str
    content: str


class DeepthinkPassResult(BaseModel):
    index: int
    answer: str
    parsed_output: Any | None
    validation_errors: list[str]
    attempts: int
    scratchpad: str | None = Field(
        default=None, description="Included only when include_scratchpads is true."
    )


class DeepthinkModeArgs(BaseModel):
    prompt: str | None = Field(
        default=None, description="User prompt to solve."
    )
    messages: list[DeepthinkMessage] | None = Field(
        default=None, description="Optional chat messages."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prefix."
    )
    mode: str | None = Field(
        default="balanced", description="fast, balanced, deep, or custom."
    )
    passes: int | None = Field(
        default=None, description="Override number of reasoning passes."
    )
    merge_strategy: str | None = Field(
        default="merge", description="merge, best_of, or first."
    )
    reasoning_instructions: str | None = Field(
        default=None, description="Instructions for the hidden reasoning phase."
    )
    answer_instructions: str | None = Field(
        default=None, description="Instructions for the final answer."
    )
    use_scratchpad: bool = Field(
        default=True, description="Inject scratchpad into answer phase."
    )
    reasoning_temperature: float | None = Field(
        default=None, description="Temperature for reasoning phase."
    )
    answer_temperature: float | None = Field(
        default=None, description="Temperature for answer phase."
    )
    reasoning_max_tokens: int | None = Field(
        default=None, description="Token budget for reasoning phase."
    )
    answer_max_tokens: int | None = Field(
        default=None, description="Token budget for answer phase."
    )
    max_retries: int | None = Field(
        default=None, description="Max retries per pass when validation fails."
    )
    validation_mode: str | None = Field(
        default="none", description="none or json_schema."
    )
    output_schema: dict | str | None = Field(
        default=None, description="JSON schema for answer validation."
    )
    strict_json: bool = Field(
        default=True, description="Require JSON-only output when schema is set."
    )
    include_passes: bool = Field(
        default=False, description="Include per-pass answers in output."
    )
    include_scratchpads: bool = Field(
        default=False, description="Include scratchpads in pass output."
    )
    llm_api_base: str | None = Field(
        default=None, description="OpenAI-compatible API base URL."
    )
    llm_model: str | None = Field(
        default=None, description="LLM model name."
    )
    stream_final: bool = Field(
        default=False, description="Stream final answer tokens."
    )


class DeepthinkModeResult(BaseModel):
    answer: str
    parsed_output: Any | None
    passes: list[DeepthinkPassResult]
    attempts: int
    warnings: list[str]
    errors: list[str]
    llm_model: str
    selection_rationale: str | None


class DeepthinkModeConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    llm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL.",
    )
    llm_model: str = Field(
        default="gpt-oss:latest", description="Default LLM model name."
    )


class DeepthinkModeState(BaseToolState):
    pass


class DeepthinkMode(
    BaseTool[
        DeepthinkModeArgs,
        DeepthinkModeResult,
        DeepthinkModeConfig,
        DeepthinkModeState,
    ],
    ToolUIData[DeepthinkModeArgs, DeepthinkModeResult],
):
    description: ClassVar[str] = (
        "Run multi-pass reasoning and merge answers for a deep-think response."
    )

    async def run(self, args: DeepthinkModeArgs) -> DeepthinkModeResult:
        warnings: list[str] = []
        errors: list[str] = []

        mode = self._normalize_mode(args.mode)
        merge_strategy = self._normalize_merge_strategy(args.merge_strategy)
        validation_mode = self._normalize_validation_mode(args.validation_mode)
        output_schema = self._load_schema(args.output_schema, warnings)
        if output_schema and validation_mode == "none":
            validation_mode = "json_schema"
            warnings.append("output_schema provided; enabling json_schema validation.")
        if validation_mode == "json_schema" and output_schema is None:
            raise ToolError("validation_mode=json_schema requires output_schema.")

        profile = self._resolve_profile(mode, args, warnings)
        pass_count = max(1, int(args.passes or profile["passes"]))

        base_messages = self._build_base_messages(args)
        question = self._extract_question(args, base_messages)

        pass_outputs: list[_PassOutput] = []
        total_attempts = 0
        stream_final = bool(args.stream_final)
        if stream_final and pass_count > 1 and merge_strategy != "merge":
            warnings.append(
                "stream_final ignored when merge_strategy is not 'merge' with multiple passes."
            )
            stream_final = False

        for pass_index in range(pass_count):
            stream_answer = bool(stream_final and pass_count == 1 and pass_index == 0)
            output = self._run_pass(
                base_messages,
                args,
                profile,
                output_schema,
                validation_mode,
                warnings,
                stream_answer,
            )
            pass_outputs.append(output)
            total_attempts += output.attempts

        selection_rationale: str | None = None
        final_answer = pass_outputs[0].answer
        parsed_output = pass_outputs[0].parsed

        if pass_count > 1:
            if merge_strategy == "first":
                selection_rationale = "Selected first pass."
            elif merge_strategy == "best_of":
                best, rationale, selection_attempts = self._select_best(
                    question,
                    pass_outputs,
                    args,
                    output_schema,
                    validation_mode,
                )
                total_attempts += selection_attempts
                final_answer = best.answer
                parsed_output = best.parsed
                selection_rationale = rationale
            else:
                merged, parsed, rationale, merge_attempts, merge_errors = self._merge_answers(
                    question,
                    pass_outputs,
                    args,
                    output_schema,
                    validation_mode,
                    stream_final,
                    profile,
                )
                total_attempts += merge_attempts
                errors.extend(merge_errors)
                if merged is None:
                    fallback, rationale, selection_attempts = self._select_best(
                        question,
                        pass_outputs,
                        args,
                        output_schema,
                        validation_mode,
                    )
                    total_attempts += selection_attempts
                    final_answer = fallback.answer
                    parsed_output = fallback.parsed
                    selection_rationale = (
                        rationale or "Merge failed; using best candidate."
                    )
                else:
                    final_answer = merged
                    parsed_output = parsed
                    selection_rationale = rationale

        pass_results: list[DeepthinkPassResult] = []
        if args.include_passes:
            for idx, output in enumerate(pass_outputs, start=1):
                pass_results.append(
                    DeepthinkPassResult(
                        index=idx,
                        answer=output.answer,
                        parsed_output=output.parsed,
                        validation_errors=output.validation_errors,
                        attempts=output.attempts,
                        scratchpad=output.scratchpad if args.include_scratchpads else None,
                    )
                )

        return DeepthinkModeResult(
            answer=final_answer,
            parsed_output=parsed_output,
            passes=pass_results,
            attempts=total_attempts,
            warnings=warnings,
            errors=errors,
            llm_model=self._resolve_model(args),
            selection_rationale=selection_rationale,
        )

    def _normalize_mode(self, value: str | None) -> str:
        mode = (value or "balanced").strip().lower()
        if mode not in VALID_MODES:
            raise ToolError(
                f"mode must be one of: {', '.join(sorted(VALID_MODES))}"
            )
        return mode

    def _normalize_merge_strategy(self, value: str | None) -> str:
        strategy = (value or "merge").strip().lower()
        if strategy not in VALID_MERGE:
            raise ToolError(
                f"merge_strategy must be one of: {', '.join(sorted(VALID_MERGE))}"
            )
        return strategy

    def _normalize_validation_mode(self, value: str | None) -> str:
        mode = (value or "none").strip().lower()
        if mode not in VALID_VALIDATION:
            raise ToolError(
                f"validation_mode must be one of: {', '.join(sorted(VALID_VALIDATION))}"
            )
        return mode

    def _resolve_profile(
        self, mode: str, args: DeepthinkModeArgs, warnings: list[str]
    ) -> dict[str, Any]:
        profile = MODE_PROFILES.get(mode, MODE_PROFILES["balanced"]).copy()
        if mode == "custom" and not any(
            [
                args.passes,
                args.reasoning_max_tokens,
                args.answer_max_tokens,
                args.reasoning_temperature,
                args.answer_temperature,
                args.max_retries,
            ]
        ):
            warnings.append("custom mode requested with no overrides; using balanced profile.")
        if args.passes is not None:
            profile["passes"] = int(args.passes)
        if args.reasoning_max_tokens is not None:
            profile["reasoning_max_tokens"] = int(args.reasoning_max_tokens)
        if args.answer_max_tokens is not None:
            profile["answer_max_tokens"] = int(args.answer_max_tokens)
        if args.reasoning_temperature is not None:
            profile["reasoning_temperature"] = float(args.reasoning_temperature)
        if args.answer_temperature is not None:
            profile["answer_temperature"] = float(args.answer_temperature)
        if args.max_retries is not None:
            profile["max_retries"] = int(args.max_retries)
        return profile

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

    def _build_base_messages(self, args: DeepthinkModeArgs) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if args.system_prompt and args.system_prompt.strip():
            messages.append(
                {"role": "system", "content": args.system_prompt.strip()}
            )
        if args.messages:
            for msg in args.messages:
                content = (msg.content or "").strip()
                if not content:
                    continue
                messages.append({"role": msg.role, "content": content})
        if args.prompt and args.prompt.strip():
            messages.append({"role": "user", "content": args.prompt.strip()})
        if not messages:
            raise ToolError("Provide prompt or messages.")
        return messages

    def _extract_question(
        self, args: DeepthinkModeArgs, messages: list[dict[str, str]]
    ) -> str:
        if args.prompt and args.prompt.strip():
            return args.prompt.strip()
        for msg in reversed(messages):
            if msg.get("role") == "user" and msg.get("content"):
                return str(msg["content"]).strip()
        return "Request"

    def _run_pass(
        self,
        base_messages: list[dict[str, str]],
        args: DeepthinkModeArgs,
        profile: dict[str, Any],
        output_schema: dict | None,
        validation_mode: str,
        warnings: list[str],
        stream_answer: bool,
    ) -> _PassOutput:
        max_retries = max(0, int(profile.get("max_retries", 0)))
        attempts = 0
        last_error: str | None = None
        last_answer: str | None = None
        validation_errors: list[str] = []
        parsed: Any | None = None
        scratchpad: str | None = None
        answer = ""

        for attempt in range(max_retries + 1):
            attempts = attempt + 1
            attempt_messages = list(base_messages)
            if last_error:
                if last_answer:
                    attempt_messages.append(
                        {"role": "assistant", "content": last_answer}
                    )
                attempt_messages.append(
                    {"role": "user", "content": self._retry_prompt(last_error)}
                )

            scratchpad = None
            if args.use_scratchpad:
                reasoning_messages = self._build_reasoning_messages(
                    attempt_messages, args
                )
                scratchpad = self._call_llm(
                    reasoning_messages,
                    args,
                    temperature=profile.get("reasoning_temperature", 0.3),
                    max_tokens=profile.get("reasoning_max_tokens", 1200),
                    stream=False,
                )
            else:
                if attempt == 0:
                    warnings.append(
                        "use_scratchpad=false; reasoning phase skipped."
                    )

            answer_messages = self._build_answer_messages(
                attempt_messages,
                args,
                output_schema,
                scratchpad if args.use_scratchpad else None,
            )
            answer = self._call_llm(
                answer_messages,
                args,
                temperature=profile.get("answer_temperature", 0.2),
                max_tokens=profile.get("answer_max_tokens", 700),
                stream=stream_answer,
            )

            parsed, validation_errors = self._validate_answer(
                answer,
                output_schema,
                validation_mode,
                args,
            )
            if not validation_errors:
                break

            last_error = "; ".join(validation_errors) or "Validation failed."
            last_answer = answer

        return _PassOutput(
            answer=answer,
            parsed=parsed,
            validation_errors=validation_errors,
            scratchpad=scratchpad,
            attempts=attempts,
        )

    def _build_reasoning_messages(
        self, base_messages: list[dict[str, str]], args: DeepthinkModeArgs
    ) -> list[dict[str, str]]:
        instructions = (
            args.reasoning_instructions
            or "Think step-by-step and output your scratchpad only. Do not answer."
        )
        messages = list(base_messages)
        messages.append({"role": "user", "content": instructions.strip()})
        return messages

    def _build_answer_messages(
        self,
        base_messages: list[dict[str, str]],
        args: DeepthinkModeArgs,
        schema: dict | None,
        scratchpad: str | None,
    ) -> list[dict[str, str]]:
        instructions = self._compose_answer_instructions(args, schema)
        messages = list(base_messages)

        if scratchpad:
            messages.append(
                {
                    "role": "assistant",
                    "content": f"<scratchpad>\n{scratchpad}\n</scratchpad>",
                }
            )
            instructions = (
                "Use the scratchpad above for internal guidance. "
                "Do not reveal the scratchpad.\n"
                + instructions
            )

        messages.append({"role": "user", "content": instructions})
        return messages

    def _compose_answer_instructions(
        self, args: DeepthinkModeArgs, schema: dict | None
    ) -> str:
        instruction = (
            args.answer_instructions
            or "Provide the final answer to the user request above."
        )
        lines = [instruction.strip(), "Do not reveal hidden reasoning."]

        if schema is not None:
            schema_text = json.dumps(schema, ensure_ascii=True)
            lines.append("Output must match the JSON schema below:")
            lines.append(schema_text)
            if args.strict_json:
                lines.append("Return JSON only. No extra text.")

        return "\n".join(lines).strip()

    def _validate_answer(
        self,
        answer: str,
        schema: dict | None,
        mode: str,
        args: DeepthinkModeArgs,
    ) -> tuple[Any | None, list[str]]:
        if mode == "none" or schema is None:
            return None, []

        parsed, parse_error = self._parse_json(answer, args.strict_json)
        if parse_error:
            return None, [parse_error]

        validation_errors = validate_args(schema, parsed)
        return parsed, validation_errors

    def _retry_prompt(self, error: str) -> str:
        return (
            "Your previous answer did not pass validation.\n"
            f"Issue: {error}\n"
            "Please provide a corrected answer."
        )

    def _select_best(
        self,
        question: str,
        pass_outputs: list[_PassOutput],
        args: DeepthinkModeArgs,
        output_schema: dict | None,
        validation_mode: str,
    ) -> tuple[_PassOutput, str | None, int]:
        if len(pass_outputs) == 1:
            return pass_outputs[0], None, 0

        attempts = 0
        max_retries = 1
        schema_text = json.dumps(SELECTION_SCHEMA, ensure_ascii=True)
        system_prompt = (
            "You are a strict evaluator. Choose the best candidate response "
            "for the user question. Reply ONLY with JSON."
        )
        user_lines = [
            "Select the best candidate answer.",
            "Return JSON with best_index (1-based) and optional rationale.",
            f"JSON schema:\n{schema_text}",
            f"Question:\n{question}",
        ]
        for idx, output in enumerate(pass_outputs, start=1):
            error_note = (
                "; ".join(output.validation_errors)
                if output.validation_errors
                else "none"
            )
            user_lines.append(
                f"\nCandidate {idx} (validation errors: {error_note}):\n{output.answer}"
            )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(user_lines).strip()},
        ]

        for attempt in range(max_retries + 1):
            attempts = attempt + 1
            raw = self._call_llm(
                messages,
                args,
                temperature=0.0,
                max_tokens=300,
                stream=False,
            )
            parsed, parse_error = self._parse_json(raw, strict=True)
            if parse_error:
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": self._retry_prompt(parse_error),
                    },
                ]
                continue
            if not isinstance(parsed, dict):
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": self._retry_prompt("Selection output must be an object."),
                    },
                ]
                continue
            validation_errors = validate_args(SELECTION_SCHEMA, parsed)
            if validation_errors:
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": self._retry_prompt("; ".join(validation_errors)),
                    },
                ]
                continue

            best_index = parsed.get("best_index")
            rationale = parsed.get("rationale")
            if isinstance(rationale, str):
                rationale = rationale.strip() or None
            else:
                rationale = None

            resolved = None
            if isinstance(best_index, int):
                if 1 <= best_index <= len(pass_outputs):
                    resolved = best_index - 1
                elif 0 <= best_index < len(pass_outputs):
                    resolved = best_index
            if resolved is not None:
                return pass_outputs[resolved], rationale, attempts

            messages = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": self._retry_prompt("best_index out of range."),
                },
            ]

        fallback = self._fallback_best(pass_outputs)
        return fallback, None, attempts

    def _fallback_best(self, pass_outputs: list[_PassOutput]) -> _PassOutput:
        return sorted(
            pass_outputs,
            key=lambda item: (len(item.validation_errors), -len(item.answer)),
        )[0]

    def _merge_answers(
        self,
        question: str,
        pass_outputs: list[_PassOutput],
        args: DeepthinkModeArgs,
        output_schema: dict | None,
        validation_mode: str,
        stream_final: bool,
        profile: dict[str, Any],
    ) -> tuple[str | None, Any | None, str | None, int, list[str]]:
        errors: list[str] = []
        attempts = 0
        max_retries = max(0, int(args.max_retries if args.max_retries is not None else 1))

        schema_text = json.dumps(output_schema, ensure_ascii=True) if output_schema else None
        system_prompt = (
            "You are a synthesis assistant. Combine the best elements of the candidate answers "
            "into a single response. Prioritize correctness and clarity."
        )
        if output_schema:
            system_prompt += (
                "\nReturn ONLY valid JSON that matches the provided schema."
            )

        user_lines = [
            f"Question:\n{question}",
            "Candidates:",
        ]
        for idx, output in enumerate(pass_outputs, start=1):
            error_note = (
                "; ".join(output.validation_errors)
                if output.validation_errors
                else "none"
            )
            user_lines.append(
                f"\nCandidate {idx} (validation errors: {error_note}):\n{output.answer}"
            )
        if output_schema and schema_text:
            user_lines.append(f"\nJSON schema:\n{schema_text}")
            if args.strict_json:
                user_lines.append("Return JSON only. No extra text.")

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(user_lines).strip()},
        ]

        merged_answer: str | None = None
        parsed_output: Any | None = None
        rationale = f"Merged from {len(pass_outputs)} passes."

        for attempt in range(max_retries + 1):
            attempts = attempt + 1
            raw = self._call_llm(
                messages,
                args,
                temperature=profile.get("answer_temperature", 0.2),
                max_tokens=profile.get("answer_max_tokens", 700),
                stream=bool(stream_final),
            )
            parsed_output, validation_errors = self._validate_answer(
                raw,
                output_schema,
                validation_mode,
                args,
            )
            if not validation_errors:
                merged_answer = raw
                return merged_answer, parsed_output, rationale, attempts, errors

            error_text = "; ".join(validation_errors) or "Validation failed."
            errors.append(f"Merge attempt {attempts} failed validation: {error_text}")
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": self._retry_prompt(error_text)},
            ]

        return None, None, None, attempts, errors

    def _call_llm(
        self,
        messages: list[dict[str, str]],
        args: DeepthinkModeArgs,
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> str:
        api_base = (args.llm_api_base or self.config.llm_api_base).rstrip("/")
        url = api_base + "/chat/completions"
        payload = {
            "model": self._resolve_model(args),
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

    def _resolve_model(self, args: DeepthinkModeArgs) -> str:
        return args.llm_model or self.config.llm_model

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, DeepthinkModeArgs):
            return ToolCallDisplay(summary="deepthink_mode")
        return ToolCallDisplay(
            summary="deepthink_mode",
            details={
                "mode": event.args.mode,
                "passes": event.args.passes,
                "merge_strategy": event.args.merge_strategy,
                "validation_mode": event.args.validation_mode,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, DeepthinkModeResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = "Deepthink mode complete"
        if event.result.errors:
            message = "Deepthink mode finished with errors"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "answer": event.result.answer,
                "parsed_output": event.result.parsed_output,
                "attempts": event.result.attempts,
                "selection_rationale": event.result.selection_rationale,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Running deepthink mode"
