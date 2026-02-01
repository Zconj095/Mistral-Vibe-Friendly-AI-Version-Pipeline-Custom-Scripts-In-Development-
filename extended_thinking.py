from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
import urllib.error
import urllib.request

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


VALID_VALIDATION = {"none", "json_schema", "llm"}

VALIDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "valid": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "score": {"type": "number"},
    },
    "required": ["valid"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class _ValidationResult:
    valid: bool
    issues: list[str]
    score: float | None
    raw: str | None


class PromptMessage(BaseModel):
    role: str
    content: str


class ExtendedThinkingArgs(BaseModel):
    prompt: str | None = Field(
        default=None, description="User prompt to solve."
    )
    messages: list[PromptMessage] | None = Field(
        default=None, description="Optional chat messages."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prefix."
    )
    include_resonance_quotes: bool = Field(
        default=True,
        description="Include resonance quote bank in system prompts.",
    )
    resonance_quotes_path: str | None = Field(
        default=None, description="Optional path to resonance quotes."
    )
    resonance_quotes_max_chars: int | None = Field(
        default=None, description="Max resonance quote chars to include."
    )
    reasoning_instructions: str | None = Field(
        default=None, description="Instructions for the hidden reasoning phase."
    )
    answer_instructions: str | None = Field(
        default=None, description="Instructions for the final answer."
    )
    use_scratchpad: bool = Field(
        default=True, description="Inject reasoning scratchpad into answer phase."
    )
    scratchpad_max_chars: int | None = Field(
        default=None, description="Max scratchpad chars to carry forward."
    )
    include_reasoning_summary: bool = Field(
        default=False, description="Summarize reasoning for optional storage."
    )
    reasoning_summary_prompt: str | None = Field(
        default=None, description="Prompt to summarize reasoning."
    )
    reasoning_temperature: float = Field(
        default=0.2, description="Temperature for reasoning phase."
    )
    answer_temperature: float = Field(
        default=0.2, description="Temperature for answer phase."
    )
    reasoning_max_tokens: int = Field(
        default=1200, description="Token budget for reasoning phase."
    )
    answer_max_tokens: int = Field(
        default=600, description="Token budget for answer phase."
    )
    stream_final: bool = Field(
        default=False, description="Stream final answer tokens."
    )
    llm_api_base: str | None = Field(
        default=None, description="OpenAI-compatible API base URL."
    )
    llm_model: str | None = Field(
        default=None, description="LLM model name."
    )
    max_retries: int = Field(
        default=1, description="Max retries when validation fails."
    )
    validation_mode: str | None = Field(
        default="none", description="none, json_schema, or llm."
    )
    output_schema: dict | str | None = Field(
        default=None, description="JSON schema for answer validation."
    )
    strict_json: bool = Field(
        default=True, description="Require JSON-only output when schema is set."
    )
    validator_prompt: str | None = Field(
        default=None, description="Custom validator prompt."
    )
    validator_system_prompt: str | None = Field(
        default=None, description="Custom validator system prompt."
    )
    validator_temperature: float = Field(
        default=0.0, description="Temperature for validator model."
    )
    validator_max_tokens: int = Field(
        default=400, description="Max tokens for validator response."
    )


class ExtendedThinkingResult(BaseModel):
    answer: str
    parsed_output: Any | None
    validation: dict | None
    attempts: int
    reasoning_summary: str | None
    warnings: list[str]
    errors: list[str]


class ExtendedThinkingConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    llm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL.",
    )
    llm_model: str = Field(
        default="gpt-oss:latest", description="Default LLM model name."
    )
    scratchpad_max_chars: int = Field(
        default=8000, description="Max scratchpad chars to carry forward."
    )
    resonance_quotes_path: Path = Field(
        default=Path.home()
        / ".vibe"
        / "libraries"
        / "mistral_intelligence"
        / "resonance_quotes.txt",
        description="Default resonance quotes path.",
    )
    resonance_quotes_max_chars: int = Field(
        default=12000, description="Max resonance quote chars to include."
    )


class ExtendedThinkingState(BaseToolState):
    pass


class ExtendedThinking(
    BaseTool[
        ExtendedThinkingArgs,
        ExtendedThinkingResult,
        ExtendedThinkingConfig,
        ExtendedThinkingState,
    ],
    ToolUIData[ExtendedThinkingArgs, ExtendedThinkingResult],
):
    description: ClassVar[str] = (
        "Run hidden reasoning followed by a validated final answer."
    )

    async def run(
        self, args: ExtendedThinkingArgs
    ) -> ExtendedThinkingResult:
        warnings: list[str] = []
        errors: list[str] = []

        validation_mode = self._normalize_validation_mode(args)
        output_schema = self._load_schema(args.output_schema, warnings)
        if output_schema and validation_mode == "none":
            validation_mode = "json_schema"
            warnings.append(
                "output_schema provided; enabling json_schema validation."
            )
        if validation_mode == "json_schema" and output_schema is None:
            raise ToolError("validation_mode=json_schema requires output_schema.")

        max_retries = max(0, int(args.max_retries))
        self._validate_token_limits(args)

        resonance_quotes = self._load_resonance_quotes(args, warnings)
        base_messages = self._build_base_messages(args, resonance_quotes)
        question = self._extract_question(args, base_messages)

        attempts = 0
        answer = ""
        parsed_output: Any | None = None
        validation_payload: dict | None = None
        reasoning_summary: str | None = None
        last_error: str | None = None
        last_answer: str | None = None
        final_scratchpad: str | None = None

        for attempt in range(max_retries + 1):
            attempts = attempt + 1
            attempt_messages = list(base_messages)
            if last_error:
                if last_answer:
                    attempt_messages.append(
                        {"role": "assistant", "content": last_answer}
                    )
                attempt_messages.append(
                    {
                        "role": "user",
                        "content": self._retry_prompt(last_error),
                    }
                )

            scratchpad = ""
            if self._should_reason(args):
                reasoning_messages = self._build_reasoning_messages(
                    attempt_messages, args
                )
                scratchpad = self._call_llm(
                    reasoning_messages,
                    args,
                    temperature=args.reasoning_temperature,
                    max_tokens=args.reasoning_max_tokens,
                    stream=False,
                )
                scratchpad = self._truncate_text(
                    scratchpad,
                    args.scratchpad_max_chars or self.config.scratchpad_max_chars,
                )
            else:
                warnings.append(
                    "Reasoning phase skipped (use_scratchpad and include_reasoning_summary are false)."
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
                temperature=args.answer_temperature,
                max_tokens=args.answer_max_tokens,
                stream=args.stream_final,
            )

            parsed_output, validation = self._validate_output(
                answer,
                validation_mode,
                output_schema,
                args,
                question,
            )
            validation_payload = (
                self._format_validation(validation_mode, validation)
                if validation
                else None
            )

            if validation_mode == "none" or (validation and validation.valid):
                final_scratchpad = scratchpad
                break

            issue_text = "; ".join(validation.issues) if validation else "Validation failed."
            last_error = issue_text or "Validation failed."
            last_answer = answer
            warnings.append(
                f"Attempt {attempts} failed validation: {last_error}"
            )
            final_scratchpad = scratchpad

        if validation_mode != "none" and validation_payload and not validation_payload.get(
            "valid", True
        ):
            errors.append(
                "Validation failed after retries."
                + (f" Issues: {', '.join(validation_payload.get('issues') or [])}" if validation_payload.get("issues") else "")
            )

        if args.include_reasoning_summary and final_scratchpad:
            reasoning_summary = self._summarize_reasoning(
                final_scratchpad, args
            )
            if reasoning_summary is None:
                warnings.append("Reasoning summary failed.")

        return ExtendedThinkingResult(
            answer=answer,
            parsed_output=parsed_output,
            validation=validation_payload,
            attempts=attempts,
            reasoning_summary=reasoning_summary,
            warnings=warnings,
            errors=errors,
        )

    def _normalize_validation_mode(self, args: ExtendedThinkingArgs) -> str:
        mode = (args.validation_mode or "none").strip().lower()
        if mode not in VALID_VALIDATION:
            raise ToolError(
                f"validation_mode must be one of: {', '.join(sorted(VALID_VALIDATION))}"
            )
        return mode

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

    def _validate_token_limits(self, args: ExtendedThinkingArgs) -> None:
        if args.reasoning_max_tokens <= 0:
            raise ToolError("reasoning_max_tokens must be positive.")
        if args.answer_max_tokens <= 0:
            raise ToolError("answer_max_tokens must be positive.")
        if args.reasoning_temperature < 0:
            raise ToolError("reasoning_temperature cannot be negative.")
        if args.answer_temperature < 0:
            raise ToolError("answer_temperature cannot be negative.")

    def _load_resonance_quotes(
        self, args: ExtendedThinkingArgs, warnings: list[str]
    ) -> str | None:
        if not args.include_resonance_quotes:
            return None
        path_value = args.resonance_quotes_path or self.config.resonance_quotes_path
        if not path_value:
            return None
        path = Path(path_value).expanduser() if isinstance(path_value, (str, Path)) else None
        if path is None:
            return None
        if not path.is_absolute():
            path = self.config.effective_workdir / path
        path = path.resolve()
        if not path.exists():
            warnings.append(f"Resonance quotes file not found: {path}")
            return None
        if path.is_dir():
            warnings.append(f"Resonance quotes path is a directory: {path}")
            return None
        try:
            text = path.read_text("utf-8", errors="ignore").strip()
        except OSError as exc:
            warnings.append(f"Failed to read resonance quotes: {exc}")
            return None
        if not text:
            return None
        max_chars = (
            args.resonance_quotes_max_chars
            if args.resonance_quotes_max_chars is not None
            else self.config.resonance_quotes_max_chars
        )
        if max_chars > 0 and len(text) > max_chars:
            warnings.append("Resonance quotes truncated to max chars.")
            text = text[:max_chars].rstrip()
        return text

    def _format_resonance_prompt(self, quotes: str) -> str:
        return (
            "Resonance ethos statements (tone guidance only; do not quote unless asked):\n"
            + quotes.strip()
        )

    def _build_base_messages(
        self, args: ExtendedThinkingArgs, resonance_quotes: str | None
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []

        if resonance_quotes:
            messages.append(
                {
                    "role": "system",
                    "content": self._format_resonance_prompt(resonance_quotes),
                }
            )

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
        self,
        args: ExtendedThinkingArgs,
        messages: list[dict[str, str]],
    ) -> str:
        if args.prompt and args.prompt.strip():
            return args.prompt.strip()
        for msg in reversed(messages):
            if msg.get("role") == "user" and msg.get("content"):
                return str(msg["content"]).strip()
        return "Request"

    def _build_reasoning_messages(
        self, base_messages: list[dict[str, str]], args: ExtendedThinkingArgs
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
        args: ExtendedThinkingArgs,
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
        self, args: ExtendedThinkingArgs, schema: dict | None
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

    def _retry_prompt(self, error: str) -> str:
        return (
            "Your previous answer did not pass validation.\n"
            f"Issue: {error}\n"
            "Please provide a corrected answer."
        )

    def _should_reason(self, args: ExtendedThinkingArgs) -> bool:
        return bool(args.use_scratchpad or args.include_reasoning_summary)

    def _call_llm(
        self,
        messages: list[dict[str, str]],
        args: ExtendedThinkingArgs,
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> str:
        api_base = (args.llm_api_base or self.config.llm_api_base).rstrip("/")
        url = api_base + "/chat/completions"
        payload = {
            "model": args.llm_model or self.config.llm_model,
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

    def _validate_output(
        self,
        answer: str,
        mode: str,
        schema: dict | None,
        args: ExtendedThinkingArgs,
        question: str,
    ) -> tuple[Any | None, _ValidationResult | None]:
        issues: list[str] = []
        parsed_output: Any | None = None
        valid = True

        if schema is not None:
            parsed_output, parse_error = self._parse_json(answer, args.strict_json)
            if parse_error:
                issues.append(parse_error)
                valid = False
            else:
                validation_errors = validate_args(schema, parsed_output)
                if validation_errors:
                    issues.extend(validation_errors)
                    valid = False

        if mode == "llm":
            llm_validation = self._validate_with_llm(
                question, answer, schema, args
            )
            if llm_validation.issues:
                issues.extend(llm_validation.issues)
            if not llm_validation.valid:
                valid = False
            return parsed_output, _ValidationResult(
                valid=valid,
                issues=issues,
                score=llm_validation.score,
                raw=llm_validation.raw,
            )

        if mode == "json_schema":
            return parsed_output, _ValidationResult(
                valid=valid,
                issues=issues,
                score=None,
                raw=None,
            )

        return None, None

    def _validate_with_llm(
        self,
        question: str,
        answer: str,
        schema: dict | None,
        args: ExtendedThinkingArgs,
    ) -> _ValidationResult:
        system_prompt = (
            args.validator_system_prompt
            or "You are a strict validator. Reply only with JSON."
        )
        schema_text = json.dumps(VALIDATION_SCHEMA, ensure_ascii=True)
        system_prompt = (
            f"{system_prompt}\nJSON schema for your reply:\n{schema_text}"
        )

        if args.validator_prompt and args.validator_prompt.strip():
            user_prompt = args.validator_prompt.strip()
        else:
            lines = [
                "Validate the assistant answer for correctness and completeness.",
                "Return JSON with: valid (boolean), issues (array of strings), score (number 0-1 optional).",
                f"User request:\n{question}",
                f"Assistant answer:\n{answer}",
            ]
            if schema is not None:
                lines.append(
                    f"Expected JSON schema:\n{json.dumps(schema, ensure_ascii=True)}"
                )
            user_prompt = "\n\n".join(lines).strip()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        raw = self._call_llm(
            messages,
            args,
            temperature=args.validator_temperature,
            max_tokens=args.validator_max_tokens,
            stream=False,
        )
        parsed, parse_error = self._parse_json(raw, strict=True)
        if parse_error:
            return _ValidationResult(
                valid=False,
                issues=[f"Validator JSON parse error: {parse_error}"],
                score=None,
                raw=raw,
            )
        if not isinstance(parsed, dict):
            return _ValidationResult(
                valid=False,
                issues=["Validator output was not an object."],
                score=None,
                raw=raw,
            )
        validation_errors = validate_args(VALIDATION_SCHEMA, parsed)
        if validation_errors:
            return _ValidationResult(
                valid=False,
                issues=validation_errors,
                score=None,
                raw=raw,
            )

        valid = bool(parsed.get("valid"))
        issues = [
            str(item)
            for item in (parsed.get("issues") or [])
            if isinstance(item, str) and item.strip()
        ]
        score_value = parsed.get("score")
        score = float(score_value) if isinstance(score_value, (int, float)) else None
        return _ValidationResult(
            valid=valid,
            issues=issues,
            score=score,
            raw=raw,
        )

    def _format_validation(
        self, mode: str, result: _ValidationResult | None
    ) -> dict | None:
        if result is None:
            return None
        payload = {
            "mode": mode,
            "valid": result.valid,
            "issues": result.issues,
        }
        if result.score is not None:
            payload["score"] = result.score
        if result.raw is not None:
            payload["raw"] = result.raw
        return payload

    def _summarize_reasoning(
        self, scratchpad: str, args: ExtendedThinkingArgs
    ) -> str | None:
        prompt = (
            args.reasoning_summary_prompt
            or "Summarize the reasoning into 3-5 bullet points focusing on key decisions."
        )
        messages = [
            {"role": "system", "content": "Summarize internal reasoning for memory. Do not include chain-of-thought."},
            {"role": "user", "content": f"{prompt}\n\n{scratchpad}"},
        ]
        try:
            return self._call_llm(
                messages,
                args,
                temperature=0.2,
                max_tokens=min(300, args.answer_max_tokens),
                stream=False,
            )
        except ToolError:
            return None

    def _truncate_text(self, text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return text
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "..."

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ExtendedThinkingArgs):
            return ToolCallDisplay(summary="extended_thinking")
        return ToolCallDisplay(
            summary="extended_thinking",
            details={
                "messages": len(event.args.messages or []),
                "validation_mode": event.args.validation_mode,
                "max_retries": event.args.max_retries,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ExtendedThinkingResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = "Extended thinking complete"
        if event.result.errors:
            message = "Extended thinking finished with errors"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "answer": event.result.answer,
                "parsed_output": event.result.parsed_output,
                "validation": event.result.validation,
                "attempts": event.result.attempts,
                "reasoning_summary": event.result.reasoning_summary,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Running extended thinking"
