from __future__ import annotations

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


class ContextAttentionSpanConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    round_digits: int = Field(
        default=4, description="Decimal places for derived values."
    )


class ContextAttentionSpanState(BaseToolState):
    pass


class ContextAttentionSpanArgs(BaseModel):
    t_focus: float = Field(description="Maximum focus time based on environment.")
    distractions: float = Field(description="Distractions in the environment (D).")
    lambda_value: float = Field(
        description="Scaling constant for susceptibility (lambda)."
    )
    unit: str | None = Field(
        default=None, description="Optional time unit label."
    )
    label: str | None = Field(
        default=None, description="Optional scenario label."
    )


class ContextAttentionSpanResult(BaseModel):
    attention_span: float
    t_focus: float
    distractions: float
    lambda_value: float
    denominator: float
    distraction_load: float
    focus_ratio: float
    percent_of_focus: float
    unit: str | None
    label: str | None
    warnings: list[str]


class ContextAttentionSpan(
    BaseTool[
        ContextAttentionSpanArgs,
        ContextAttentionSpanResult,
        ContextAttentionSpanConfig,
        ContextAttentionSpanState,
    ],
    ToolUIData[ContextAttentionSpanArgs, ContextAttentionSpanResult],
):
    description: ClassVar[str] = (
        "Compute attention span from focus time and distractions."
    )

    async def run(
        self, args: ContextAttentionSpanArgs
    ) -> ContextAttentionSpanResult:
        t_focus = args.t_focus
        distractions = args.distractions
        lambda_value = args.lambda_value

        if t_focus <= 0:
            raise ToolError("t_focus must be positive.")
        if distractions < 0:
            raise ToolError("distractions must be non-negative.")
        if lambda_value < 0:
            raise ToolError("lambda_value must be non-negative.")

        distraction_load = lambda_value * distractions
        denominator = 1.0 + distraction_load
        if denominator <= 0:
            raise ToolError("1 + lambda_value * distractions must be positive.")

        attention_span = t_focus / denominator
        focus_ratio = attention_span / t_focus
        percent_of_focus = focus_ratio * 100.0

        attention_span = self._round(attention_span)
        denominator = self._round(denominator)
        distraction_load = self._round(distraction_load)
        focus_ratio = self._round(focus_ratio)
        percent_of_focus = self._round(percent_of_focus)

        unit = args.unit.strip() if args.unit and args.unit.strip() else None
        label = args.label.strip() if args.label and args.label.strip() else None

        return ContextAttentionSpanResult(
            attention_span=attention_span,
            t_focus=t_focus,
            distractions=distractions,
            lambda_value=lambda_value,
            denominator=denominator,
            distraction_load=distraction_load,
            focus_ratio=focus_ratio,
            percent_of_focus=percent_of_focus,
            unit=unit,
            label=label,
            warnings=[],
        )

    def _round(self, value: float) -> float:
        digits = self.config.round_digits
        if digits < 0:
            return value
        return round(value, digits)

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextAttentionSpanArgs):
            return ToolCallDisplay(summary="context_attention_span")
        return ToolCallDisplay(
            summary="context_attention_span",
            details={
                "t_focus": event.args.t_focus,
                "distractions": event.args.distractions,
                "lambda_value": event.args.lambda_value,
                "unit": event.args.unit,
                "label": event.args.label,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextAttentionSpanResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        unit = f" {event.result.unit}" if event.result.unit else ""
        message = f"Computed attention span {event.result.attention_span}{unit}"
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=event.result.warnings,
            details={
                "attention_span": event.result.attention_span,
                "focus_ratio": event.result.focus_ratio,
                "percent_of_focus": event.result.percent_of_focus,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Calculating attention span"
