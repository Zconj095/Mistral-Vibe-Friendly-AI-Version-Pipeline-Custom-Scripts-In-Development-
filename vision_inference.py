from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from urllib import request
from urllib.parse import urlparse

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


DEFAULT_MODEL_NAME = "llava-phi3"
VALID_VALIDATION = {"none", "json_schema"}


class VisionInferenceConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    backend: str = Field(
        default="ollama", description="Backend: ollama (local VLM server)."
    )
    model_name: str = Field(
        default=DEFAULT_MODEL_NAME,
        description="Local vision model name (for example: llava-phi3).",
    )
    ollama_url: str = Field(
        default="http://127.0.0.1:11434",
        description="Base URL for the local Ollama server.",
    )
    request_timeout_seconds: float = Field(
        default=120.0, description="HTTP timeout for the backend request."
    )
    max_images: int = Field(default=4, description="Maximum images per request.")
    max_image_bytes: int = Field(
        default=10_000_000, description="Maximum bytes per image."
    )
    max_total_image_bytes: int = Field(
        default=40_000_000, description="Maximum total bytes across images."
    )
    allow_remote_urls: bool = Field(
        default=False, description="Allow fetching remote image URLs."
    )
    temperature: float = Field(default=0.2, description="Decoding temperature.")
    max_tokens: int = Field(
        default=512, description="Maximum tokens to generate."
    )
    num_gpu: int | None = Field(
        default=None,
        description="Number of GPU layers to use (0 forces CPU).",
    )


class VisionInferenceState(BaseToolState):
    pass


class VisionImageInput(BaseModel):
    path: str | None = Field(default=None, description="Path to a local image.")
    image_base64: str | None = Field(
        default=None,
        description="Base64-encoded image data (optionally data:...;base64,...).",
    )
    image_url: str | None = Field(
        default=None, description="Image URL (remote if allowed)."
    )


class VisionMessage(BaseModel):
    role: str
    content: str | None = Field(default=None, description="Message content.")
    images: list[VisionImageInput] | None = Field(
        default=None, description="Images attached to the message."
    )


class VisionInferenceArgs(BaseModel):
    prompt: str | None = Field(
        default=None, description="User prompt/question about the image."
    )
    messages: list[VisionMessage] | None = Field(
        default=None, description="Optional chat messages with images."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prefix."
    )
    images: list[VisionImageInput] | None = Field(
        default=None, description="List of images to analyze."
    )
    path: str | None = Field(
        default=None, description="Single image path (shortcut)."
    )
    image_base64: str | None = Field(
        default=None, description="Single image base64 (shortcut)."
    )
    image_url: str | None = Field(
        default=None, description="Single image URL (shortcut)."
    )
    model: str | None = Field(
        default=None, description="Override model name."
    )
    backend: str | None = Field(
        default=None, description="Override backend."
    )
    temperature: float | None = Field(
        default=None, description="Override temperature."
    )
    max_tokens: int | None = Field(
        default=None, description="Override max tokens."
    )
    num_gpu: int | None = Field(
        default=None, description="Override num_gpu option (0 forces CPU)."
    )
    allow_remote_urls: bool | None = Field(
        default=None, description="Override remote URL allowance."
    )
    output_schema: dict | str | None = Field(
        default=None, description="JSON schema for optional output validation."
    )
    validation_mode: str | None = Field(
        default="none", description="none or json_schema."
    )
    strict_json: bool = Field(
        default=True, description="Require JSON-only output when schema is set."
    )
    max_retries: int = Field(
        default=1, description="Max retries for schema validation failures."
    )
    stream: bool = Field(
        default=False, description="Stream model output."
    )


class VisionInferenceResult(BaseModel):
    text: str
    parsed_output: Any | None = None
    model: str
    backend: str
    image_count: int
    attempts: int = 1
    validation_errors: list[str] | None = None


class VisionInference(
    BaseTool[
        VisionInferenceArgs,
        VisionInferenceResult,
        VisionInferenceConfig,
        VisionInferenceState,
    ],
    ToolUIData[VisionInferenceArgs, VisionInferenceResult],
):
    description: ClassVar[str] = (
        "Analyze local images with a local vision-language model (via Ollama)."
    )

    async def run(self, args: VisionInferenceArgs) -> VisionInferenceResult:
        if args.messages:
            if any([args.prompt, args.images, args.path, args.image_base64, args.image_url]):
                raise ToolError(
                    "Provide messages or prompt/images, not both."
                )
        else:
            if not args.prompt or not args.prompt.strip():
                raise ToolError("prompt cannot be empty.")

        backend = (args.backend or self.config.backend).strip().lower()
        if backend != "ollama":
            raise ToolError("Only the ollama backend is supported.")

        allow_remote = (
            args.allow_remote_urls
            if args.allow_remote_urls is not None
            else self.config.allow_remote_urls
        )

        schema = self._load_schema(args.output_schema)
        validation_mode = self._normalize_validation_mode(args)
        if schema is None and validation_mode == "json_schema":
            raise ToolError("validation_mode json_schema requires output_schema.")

        messages, image_count = self._build_messages(
            args, schema, validation_mode, allow_remote
        )
        if image_count == 0:
            raise ToolError("At least one image is required.")

        model = args.model or self.config.model_name
        temperature = (
            args.temperature
            if args.temperature is not None
            else self.config.temperature
        )
        max_tokens = (
            args.max_tokens
            if args.max_tokens is not None
            else self.config.max_tokens
        )
        num_gpu = (
            args.num_gpu
            if args.num_gpu is not None
            else self.config.num_gpu
        )

        attempts = 0
        text = ""
        parsed_output: Any | None = None
        validation_errors: list[str] | None = None
        last_error: str | None = None

        max_retries = max(0, int(args.max_retries))
        for attempt in range(max_retries + 1):
            attempts = attempt + 1
            if last_error:
                messages = self._append_retry(messages, text, last_error)
            text = self._call_ollama(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                num_gpu=num_gpu,
                stream=args.stream,
            )

            if validation_mode == "json_schema" and schema is not None:
                parsed_output, error = self._parse_json(text, args.strict_json)
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

        return VisionInferenceResult(
            text=text,
            parsed_output=parsed_output,
            model=model,
            backend=backend,
            image_count=image_count,
            attempts=attempts,
            validation_errors=validation_errors,
        )

    def _collect_images(self, args: VisionInferenceArgs) -> list[VisionImageInput]:
        if args.images and any([args.path, args.image_base64, args.image_url]):
            raise ToolError(
                "Provide either images list or a single image shortcut, not both."
            )

        if args.images:
            return args.images

        if args.path or args.image_base64 or args.image_url:
            return [
                VisionImageInput(
                    path=args.path,
                    image_base64=args.image_base64,
                    image_url=args.image_url,
                )
            ]

        return []

    def _normalize_validation_mode(self, args: VisionInferenceArgs) -> str:
        mode = (args.validation_mode or "none").strip().lower()
        if mode not in VALID_VALIDATION:
            raise ToolError("validation_mode must be none or json_schema.")
        return mode

    def _load_schema(self, schema: dict | str | None) -> dict | None:
        if schema is None:
            return None
        if isinstance(schema, dict):
            return schema
        if isinstance(schema, str):
            value = schema.strip()
            if not value:
                return None
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ToolError(f"output_schema JSON parse error: {exc}") from exc
            if isinstance(parsed, dict):
                return parsed
            raise ToolError("output_schema JSON must be an object.")
        raise ToolError("output_schema must be a dict or JSON string.")

    def _build_messages(
        self,
        args: VisionInferenceArgs,
        schema: dict | None,
        validation_mode: str,
        allow_remote: bool,
    ) -> tuple[list[dict[str, Any]], int]:
        messages: list[dict[str, Any]] = []
        system_parts: list[str] = []
        total_bytes = 0
        image_count = 0

        if args.system_prompt and args.system_prompt.strip():
            system_parts.append(args.system_prompt.strip())

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

        if args.messages:
            for msg in args.messages:
                content = (msg.content or "").strip()
                encoded_images, total_bytes, image_count = self._encode_images(
                    msg.images or [],
                    allow_remote,
                    total_bytes,
                    image_count,
                )
                if not content and not encoded_images:
                    continue
                payload: dict[str, Any] = {"role": msg.role, "content": content}
                if encoded_images:
                    payload["images"] = encoded_images
                messages.append(payload)
            return messages, image_count

        prompt = (args.prompt or "").strip()
        images = self._collect_images(args)
        if not prompt:
            raise ToolError("prompt cannot be empty.")
        encoded_images, total_bytes, image_count = self._encode_images(
            images,
            allow_remote,
            total_bytes,
            image_count,
        )
        if not encoded_images:
            raise ToolError("At least one image is required.")
        messages.append(
            {"role": "user", "content": prompt, "images": encoded_images}
        )
        return messages, image_count

    def _encode_images(
        self,
        images: list[VisionImageInput],
        allow_remote: bool,
        total_bytes: int,
        image_count: int,
    ) -> tuple[list[str], int, int]:
        encoded: list[str] = []
        for image in images:
            raw = self._resolve_image_bytes(image, allow_remote)
            size = len(raw)
            if size > self.config.max_image_bytes:
                raise ToolError("Image exceeds max_image_bytes.")
            total_bytes += size
            if total_bytes > self.config.max_total_image_bytes:
                raise ToolError("Images exceed max_total_image_bytes.")
            image_count += 1
            if image_count > self.config.max_images:
                raise ToolError(
                    f"Too many images: {image_count} exceeds max_images."
                )
            encoded.append(base64.b64encode(raw).decode("ascii"))
        return encoded, total_bytes, image_count

    def _load_images(
        self, images: list[VisionImageInput], allow_remote: bool
    ) -> list[str]:
        total_bytes = 0
        encoded: list[str] = []

        for image in images:
            raw = self._resolve_image_bytes(image, allow_remote)
            size = len(raw)
            if size > self.config.max_image_bytes:
                raise ToolError("Image exceeds max_image_bytes.")
            total_bytes += size
            if total_bytes > self.config.max_total_image_bytes:
                raise ToolError("Images exceed max_total_image_bytes.")
            encoded.append(base64.b64encode(raw).decode("ascii"))

        return encoded

    def _resolve_image_bytes(
        self, image: VisionImageInput, allow_remote: bool
    ) -> bytes:
        sources = [image.path, image.image_base64, image.image_url]
        if sum(1 for source in sources if source) != 1:
            raise ToolError(
                "Each image must include exactly one of path, image_base64, or image_url."
            )

        if image.path:
            return self._read_local_path(image.path)
        if image.image_base64:
            return self._decode_image_base64(image.image_base64)
        if image.image_url:
            return self._fetch_image_url(image.image_url, allow_remote)
        raise ToolError("Invalid image input.")

    def _read_local_path(self, raw_path: str) -> bytes:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.config.effective_workdir / path

        try:
            resolved = path.resolve()
        except OSError as exc:
            raise ToolError(f"Invalid path: {exc}") from exc

        if not resolved.exists():
            raise ToolError(f"Image not found at: {resolved}")
        if resolved.is_dir():
            raise ToolError(f"Path is a directory, not a file: {resolved}")

        return resolved.read_bytes()

    def _decode_image_base64(self, data: str) -> bytes:
        payload = data.strip()
        if payload.startswith("data:"):
            parts = payload.split(",", 1)
            payload = parts[1] if len(parts) > 1 else ""
        try:
            return base64.b64decode(payload, validate=False)
        except Exception as exc:
            raise ToolError(f"Invalid base64 image data: {exc}") from exc

    def _fetch_image_url(self, url: str, allow_remote: bool) -> bytes:
        parsed = urlparse(url)
        if parsed.scheme in {"", "file"}:
            path = parsed.path or url
            if path.startswith("/") and len(path) > 2 and path[2] == ":":
                path = path.lstrip("/")
            return self._read_local_path(path)

        if parsed.scheme not in {"http", "https"}:
            raise ToolError("image_url must be http(s) or file.")
        if not allow_remote:
            raise ToolError("Remote image URLs are disabled in config.")

        try:
            with request.urlopen(url, timeout=30) as resp:
                return resp.read()
        except Exception as exc:
            raise ToolError(f"Failed to fetch image_url: {exc}") from exc

    def _append_retry(
        self, messages: list[dict[str, Any]], raw: str, error: str
    ) -> list[dict[str, Any]]:
        retry_prompt = (
            "Your previous output was invalid.\n"
            f"Error: {error}\n"
            "Reply again with ONLY valid JSON that matches the schema."
        )
        return messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": retry_prompt},
        ]

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

    def _call_ollama(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        num_gpu: int | None,
        stream: bool,
    ) -> str:
        payload: dict[str, object] = {
            "model": model,
            "messages": messages,
            "stream": bool(stream),
        }
        options: dict[str, object] = {}
        if temperature is not None:
            options["temperature"] = float(temperature)
        if max_tokens is not None:
            options["num_predict"] = int(max_tokens)
        if num_gpu is not None:
            options["num_gpu"] = int(num_gpu)
        if options:
            payload["options"] = options

        data = json.dumps(payload).encode("utf-8")
        url = self.config.ollama_url.rstrip("/") + "/api/chat"
        req = request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            resp = request.urlopen(req, timeout=self.config.request_timeout_seconds)
        except Exception as exc:
            raise ToolError(f"Ollama vision call failed: {exc}") from exc

        if not stream:
            try:
                response = json.loads(resp.read().decode("utf-8"))
            except Exception as exc:
                raise ToolError(f"Ollama response parse failed: {exc}") from exc
            if isinstance(response, dict) and response.get("error"):
                raise ToolError(f"Ollama error: {response.get('error')}")
            message = response.get("message") if isinstance(response, dict) else None
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"].strip()
            raise ToolError("Unexpected response from Ollama vision endpoint.")

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
            if isinstance(chunk, dict) and chunk.get("error"):
                raise ToolError(f"Ollama error: {chunk.get('error')}")
            message = chunk.get("message") if isinstance(chunk, dict) else None
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                parts.append(message["content"])
            if isinstance(chunk, dict) and chunk.get("done") is True:
                break
        return "".join(parts).strip()

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, VisionInferenceArgs):
            return ToolCallDisplay(summary="vision_inference")

        return ToolCallDisplay(
            summary="vision_inference",
            details={
                "prompt": event.args.prompt,
                "model": event.args.model,
                "image_count": cls._display_image_count(event.args),
                "messages": len(event.args.messages or []),
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, VisionInferenceResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        return ToolResultDisplay(
            success=True,
            message="Vision analysis complete",
            details={
                "text": event.result.text,
                "parsed_output": event.result.parsed_output,
                "model": event.result.model,
                "backend": event.result.backend,
                "image_count": event.result.image_count,
                "attempts": event.result.attempts,
                "validation_errors": event.result.validation_errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Analyzing image(s)"

    @staticmethod
    def _display_image_count(args: VisionInferenceArgs) -> int:
        if args.images:
            return len(args.images)
        if args.messages:
            total = 0
            for msg in args.messages:
                total += len(msg.images or [])
            return total
        return 1 if (args.path or args.image_base64 or args.image_url) else 0
