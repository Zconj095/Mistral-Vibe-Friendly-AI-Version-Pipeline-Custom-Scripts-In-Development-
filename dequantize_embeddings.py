from __future__ import annotations

from math import sqrt
from typing import TYPE_CHECKING, ClassVar

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


class DequantizeEmbeddingsConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    default_input_dtype: str = Field(
        default="int8", description="Default input dtype."
    )
    default_output_dimension: int = Field(
        default=0, description="Default output dimension for binary inputs."
    )
    normalize: bool = Field(
        default=False, description="L2 normalize the output vectors."
    )


class DequantizeEmbeddingsState(BaseToolState):
    pass


class DequantizeEmbeddingsArgs(BaseModel):
    embeddings: list[list[int]] | None = Field(
        default=None, description="List of quantized embeddings."
    )
    embedding: list[int] | None = Field(
        default=None, description="Single quantized embedding."
    )
    input_dtype: str | None = Field(
        default=None, description="int8, uint8, binary, or ubinary."
    )
    output_dimension: int | None = Field(
        default=None, description="Original dimension for binary/ubinary."
    )
    scale: float | list[float] | None = Field(
        default=None, description="Optional scale for int8/uint8."
    )
    zero_point: int | None = Field(
        default=None, description="Zero point for uint8."
    )
    normalize: bool | None = Field(
        default=None, description="Override normalization."
    )


class DequantizeEmbeddingsResult(BaseModel):
    input_dtype: str
    output_dimension: int
    normalized: bool
    embeddings: list[list[float]]


class DequantizeEmbeddings(
    BaseTool[
        DequantizeEmbeddingsArgs,
        DequantizeEmbeddingsResult,
        DequantizeEmbeddingsConfig,
        DequantizeEmbeddingsState,
    ],
    ToolUIData[DequantizeEmbeddingsArgs, DequantizeEmbeddingsResult],
):
    description: ClassVar[str] = (
        "Dequantize int8/uint8/binary embeddings back to float vectors."
    )

    async def run(
        self, args: DequantizeEmbeddingsArgs
    ) -> DequantizeEmbeddingsResult:
        inputs = self._resolve_inputs(args)
        input_dtype = (
            args.input_dtype or self.config.default_input_dtype
        ).strip().lower()
        output_dimension = self._resolve_output_dimension(args.output_dimension)
        normalize = (
            args.normalize
            if args.normalize is not None
            else self.config.normalize
        )

        if input_dtype not in {"int8", "uint8", "binary", "ubinary"}:
            raise ToolError("input_dtype must be int8, uint8, binary, or ubinary.")

        scales = self._resolve_scales(args.scale, len(inputs))
        zero_point = self._resolve_zero_point(args.zero_point, input_dtype)

        decoded: list[list[float]] = []
        for index, embedding in enumerate(inputs):
            if input_dtype in {"binary", "ubinary"}:
                vector = self._decode_binary(embedding, input_dtype, output_dimension)
            else:
                scale = scales[index] if scales is not None else 1.0 / 127.0
                vector = self._decode_int(embedding, input_dtype, scale, zero_point)
            if normalize:
                vector = self._l2_normalize(vector)
            decoded.append(vector)

        final_dimension = len(decoded[0]) if decoded else 0
        return DequantizeEmbeddingsResult(
            input_dtype=input_dtype,
            output_dimension=final_dimension,
            normalized=normalize,
            embeddings=decoded,
        )

    def _resolve_inputs(self, args: DequantizeEmbeddingsArgs) -> list[list[int]]:
        if args.embeddings and args.embedding:
            raise ToolError("Provide embeddings or embedding, not both.")
        if args.embeddings is None and args.embedding is None:
            raise ToolError("Provide at least one embedding.")
        if args.embeddings is None:
            return [args.embedding or []]
        if not args.embeddings:
            raise ToolError("embeddings cannot be empty.")
        return args.embeddings

    def _resolve_output_dimension(self, value: int | None) -> int | None:
        if value is None:
            default_dim = self.config.default_output_dimension
            return default_dim if default_dim > 0 else None
        if value <= 0:
            raise ToolError("output_dimension must be positive.")
        if value % 8 != 0:
            raise ToolError("output_dimension must be divisible by 8.")
        return value

    def _resolve_scales(
        self, scale: float | list[float] | None, count: int
    ) -> list[float] | None:
        if scale is None:
            return None
        if isinstance(scale, list):
            if len(scale) != count:
                raise ToolError("scale list length must match embeddings.")
            return [float(x) for x in scale]
        return [float(scale) for _ in range(count)]

    def _resolve_zero_point(self, value: int | None, dtype: str) -> int:
        if dtype == "uint8":
            return 128 if value is None else int(value)
        return 0 if value is None else int(value)

    def _decode_int(
        self, embedding: list[int], dtype: str, scale: float, zero_point: int
    ) -> list[float]:
        if dtype == "int8":
            return [float(value) * scale for value in embedding]
        if dtype == "uint8":
            return [float(value - zero_point) * scale for value in embedding]
        raise ToolError("Invalid dtype for int decoding.")

    def _decode_binary(
        self, embedding: list[int], dtype: str, output_dimension: int | None
    ) -> list[float]:
        total_bits = len(embedding) * 8
        dimension = output_dimension or total_bits
        if dimension > total_bits:
            raise ToolError("output_dimension exceeds available bits.")
        if dimension % 8 != 0:
            raise ToolError("output_dimension must be divisible by 8.")

        values: list[float] = []
        bytes_needed = dimension // 8
        for index in range(bytes_needed):
            byte = embedding[index]
            if dtype == "binary" and byte < 0:
                byte = byte & 0xFF
            for bit_index in range(8):
                bit = (byte >> bit_index) & 1
                values.append(1.0 if bit == 1 else -1.0)
        return values

    def _l2_normalize(self, vector: list[float]) -> list[float]:
        norm = sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, DequantizeEmbeddingsArgs):
            return ToolCallDisplay(summary="dequantize_embeddings")

        count = len(event.args.embeddings or []) if event.args.embeddings else 1
        return ToolCallDisplay(
            summary=f"dequantize_embeddings ({count} input(s))",
            details={
                "input_dtype": event.args.input_dtype,
                "output_dimension": event.args.output_dimension,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, DequantizeEmbeddingsResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=True,
            message=f"Dequantized {len(event.result.embeddings)} embedding(s)",
            details={
                "input_dtype": event.result.input_dtype,
                "output_dimension": event.result.output_dimension,
                "normalized": event.result.normalized,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Dequantizing embeddings"
