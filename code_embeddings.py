from __future__ import annotations

import json
from typing import TYPE_CHECKING, ClassVar
from urllib import request

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


DEFAULT_MODEL_NAME = "nomic-embed-text"


class CodeEmbeddingsConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    backend: str = Field(
        default="ollama", description="Backend: ollama (local embeddings server)."
    )
    model_name: str = Field(
        default=DEFAULT_MODEL_NAME,
        description="Local embeddings model name (for example: nomic-embed-text).",
    )
    ollama_url: str = Field(
        default="http://127.0.0.1:11434",
        description="Base URL for the local Ollama server.",
    )
    request_timeout_seconds: float = Field(
        default=60.0, description="HTTP timeout for the backend request."
    )
    max_inputs: int = Field(default=128, description="Maximum inputs per call.")
    max_input_chars: int = Field(
        default=50_000, description="Maximum characters per input."
    )
    default_output_dtype: str = Field(
        default="float", description="Default output dtype."
    )
    default_output_dimension: int = Field(
        default=0, description="Default output dimension (0 uses full)."
    )
    max_output_dimension: int = Field(
        default=3072, description="Maximum allowed output dimension."
    )
    normalize: bool = Field(
        default=False, description="L2 normalize embeddings."
    )


class CodeEmbeddingsState(BaseToolState):
    pass


class CodeEmbeddingsArgs(BaseModel):
    inputs: list[str] | None = Field(
        default=None, description="Input texts to embed."
    )
    input: str | None = Field(
        default=None, description="Single input text to embed."
    )
    model: str | None = Field(
        default=None, description="Override model name."
    )
    output_dtype: str | None = Field(
        default=None,
        description="float, int8, uint8, binary, or ubinary.",
    )
    output_dimension: int | None = Field(
        default=None, description="Override output dimension."
    )
    backend: str | None = Field(
        default=None, description="Override backend."
    )
    normalize: bool | None = Field(
        default=None, description="Override normalization."
    )


class EmbeddingItem(BaseModel):
    index: int
    embedding: list[float] | list[int]
    dimension: int


class CodeEmbeddingsResult(BaseModel):
    model: str
    output_dtype: str
    output_dimension: int
    embeddings: list[EmbeddingItem]


class CodeEmbeddings(
    BaseTool[
        CodeEmbeddingsArgs,
        CodeEmbeddingsResult,
        CodeEmbeddingsConfig,
        CodeEmbeddingsState,
    ],
    ToolUIData[CodeEmbeddingsArgs, CodeEmbeddingsResult],
):
    description: ClassVar[str] = (
        "Generate embeddings for code or text using a local embeddings backend."
    )

    async def run(self, args: CodeEmbeddingsArgs) -> CodeEmbeddingsResult:
        inputs = self._resolve_inputs(args)
        model = (args.model or self.config.model_name).strip()
        backend = (args.backend or self.config.backend).strip().lower()
        output_dtype = (
            args.output_dtype or self.config.default_output_dtype
        ).strip().lower()
        output_dimension = self._resolve_output_dimension(args.output_dimension)
        normalize = (
            args.normalize
            if args.normalize is not None
            else self.config.normalize
        )

        if backend != "ollama":
            raise ToolError("Only the ollama backend is supported.")

        embeddings = self._embed_with_ollama(model, inputs)
        processed: list[EmbeddingItem] = []

        for index, vector in enumerate(embeddings):
            trimmed = self._apply_output_dimension(vector, output_dimension)
            if normalize:
                trimmed = self._l2_normalize(trimmed)
            converted = self._apply_output_dtype(trimmed, output_dtype)
            processed.append(
                EmbeddingItem(
                    index=index,
                    embedding=converted,
                    dimension=self._effective_dimension(converted, output_dtype),
                )
            )

        effective_dimension = (
            self._effective_dimension(processed[0].embedding, output_dtype)
            if processed
            else 0
        )

        return CodeEmbeddingsResult(
            model=model,
            output_dtype=output_dtype,
            output_dimension=effective_dimension,
            embeddings=processed,
        )

    def _resolve_inputs(self, args: CodeEmbeddingsArgs) -> list[str]:
        if args.inputs and args.input:
            raise ToolError("Provide either inputs or input, not both.")
        if args.inputs is None and args.input is None:
            raise ToolError("Provide at least one input.")

        inputs = args.inputs if args.inputs is not None else [args.input or ""]
        if not inputs:
            raise ToolError("inputs cannot be empty.")
        if len(inputs) > self.config.max_inputs:
            raise ToolError("inputs exceeds max_inputs.")

        cleaned: list[str] = []
        for text in inputs:
            if not text.strip():
                raise ToolError("inputs cannot contain empty strings.")
            if len(text) > self.config.max_input_chars:
                raise ToolError("input exceeds max_input_chars.")
            cleaned.append(text)
        return cleaned

    def _resolve_output_dimension(self, override: int | None) -> int | None:
        if override is not None:
            if override <= 0:
                raise ToolError("output_dimension must be positive.")
            if override > self.config.max_output_dimension:
                raise ToolError("output_dimension exceeds max_output_dimension.")
            return override

        default_dim = self.config.default_output_dimension
        if default_dim and default_dim > 0:
            if default_dim > self.config.max_output_dimension:
                raise ToolError("default_output_dimension exceeds max_output_dimension.")
            return default_dim
        return None

    def _embed_with_ollama(self, model: str, inputs: list[str]) -> list[list[float]]:
        url = self.config.ollama_url.rstrip("/") + "/api/embeddings"
        embeddings: list[list[float]] = []
        for text in inputs:
            payload = json.dumps({"model": model, "prompt": text}).encode("utf-8")
            req = request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with request.urlopen(
                    req, timeout=self.config.request_timeout_seconds
                ) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except Exception as exc:
                raise ToolError(f"Ollama embeddings failed: {exc}") from exc

            if isinstance(data, dict) and data.get("error"):
                raise ToolError(f"Ollama error: {data.get('error')}")

            embedding = data.get("embedding") if isinstance(data, dict) else None
            if not isinstance(embedding, list):
                raise ToolError("Invalid embeddings response from Ollama.")
            embeddings.append([float(x) for x in embedding])

        return embeddings

    def _apply_output_dimension(
        self, vector: list[float], output_dimension: int | None
    ) -> list[float]:
        if output_dimension is None:
            return list(vector)
        if output_dimension > len(vector):
            raise ToolError("output_dimension exceeds embedding size.")
        return list(vector[:output_dimension])

    def _apply_output_dtype(
        self, vector: list[float], output_dtype: str
    ) -> list[float] | list[int]:
        if output_dtype == "float":
            return [float(x) for x in vector]
        if output_dtype == "int8":
            return self._quantize_int8(vector)
        if output_dtype == "uint8":
            return self._quantize_uint8(vector)
        if output_dtype in {"binary", "ubinary"}:
            return self._quantize_binary(vector, output_dtype == "binary")
        raise ToolError("output_dtype must be float, int8, uint8, binary, or ubinary.")

    def _quantize_int8(self, vector: list[float]) -> list[int]:
        max_abs = max((abs(x) for x in vector), default=0.0)
        if max_abs == 0:
            return [0 for _ in vector]
        scale = 127.0 / max_abs
        quantized: list[int] = []
        for value in vector:
            q = int(round(value * scale))
            if q < -128:
                q = -128
            elif q > 127:
                q = 127
            quantized.append(q)
        return quantized

    def _quantize_uint8(self, vector: list[float]) -> list[int]:
        max_abs = max((abs(x) for x in vector), default=0.0)
        if max_abs == 0:
            return [128 for _ in vector]
        scale = 127.0 / max_abs
        quantized: list[int] = []
        for value in vector:
            q = int(round(value * scale))
            if q < -127:
                q = -127
            elif q > 127:
                q = 127
            quantized.append(q + 128)
        return quantized

    def _quantize_binary(self, vector: list[float], signed: bool) -> list[int]:
        if len(vector) % 8 != 0:
            raise ToolError("binary outputs require output_dimension divisible by 8.")
        bits = [1 if value >= 0 else 0 for value in vector]
        packed: list[int] = []
        for i in range(0, len(bits), 8):
            byte = 0
            for bit_index in range(8):
                byte |= (bits[i + bit_index] & 1) << bit_index
            if signed and byte > 127:
                byte -= 256
            packed.append(byte)
        return packed

    def _l2_normalize(self, vector: list[float]) -> list[float]:
        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def _effective_dimension(
        self, embedding: list[float] | list[int], output_dtype: str
    ) -> int:
        if output_dtype in {"binary", "ubinary"}:
            return len(embedding) * 8
        return len(embedding)

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, CodeEmbeddingsArgs):
            return ToolCallDisplay(summary="code_embeddings")

        count = len(event.args.inputs or []) if event.args.inputs else 1
        return ToolCallDisplay(
            summary=f"code_embeddings ({count} input(s))",
            details={
                "model": event.args.model,
                "output_dtype": event.args.output_dtype,
                "output_dimension": event.args.output_dimension,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, CodeEmbeddingsResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=True,
            message=f"Generated {len(event.result.embeddings)} embedding(s)",
            details={
                "model": event.result.model,
                "output_dtype": event.result.output_dtype,
                "output_dimension": event.result.output_dimension,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Generating embeddings"
