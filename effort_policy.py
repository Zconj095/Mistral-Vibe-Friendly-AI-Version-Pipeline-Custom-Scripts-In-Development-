from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar

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


VALID_EFFORT = {"low", "medium", "high", "auto", "custom"}
VALID_TARGETS = {"extended_thinking", "structured_outputs", "none"}
VALID_VALIDATION = {"none", "json_schema", "llm"}

PROFILE_FIELDS = {
    "reasoning_max_tokens",
    "answer_max_tokens",
    "reasoning_temperature",
    "answer_temperature",
    "max_retries",
    "validation_mode",
    "use_scratchpad",
    "include_reasoning_summary",
}

DEFAULT_PROFILES: dict[str, dict[str, Any]] = {
    "low": {
        "reasoning_max_tokens": 256,
        "answer_max_tokens": 400,
        "reasoning_temperature": 0.6,
        "answer_temperature": 0.6,
        "max_retries": 0,
        "validation_mode": "none",
        "use_scratchpad": False,
        "include_reasoning_summary": False,
    },
    "medium": {
        "reasoning_max_tokens": 1200,
        "answer_max_tokens": 800,
        "reasoning_temperature": 0.3,
        "answer_temperature": 0.3,
        "max_retries": 1,
        "validation_mode": "json_schema",
        "use_scratchpad": True,
        "include_reasoning_summary": False,
    },
    "high": {
        "reasoning_max_tokens": 3000,
        "answer_max_tokens": 1200,
        "reasoning_temperature": 0.2,
        "answer_temperature": 0.2,
        "max_retries": 3,
        "validation_mode": "llm",
        "use_scratchpad": True,
        "include_reasoning_summary": True,
    },
}


class EffortProfile(BaseModel):
    name: str
    reasoning_max_tokens: int
    answer_max_tokens: int
    reasoning_temperature: float
    answer_temperature: float
    max_retries: int
    validation_mode: str
    use_scratchpad: bool
    include_reasoning_summary: bool


class EffortPolicyArgs(BaseModel):
    action: str | None = Field(
        default="select",
        description="select or list",
    )
    effort: str | None = Field(
        default="medium",
        description="low, medium, high, auto, or custom",
    )
    target: str | None = Field(
        default="extended_thinking",
        description="Target tool name to map settings to.",
    )
    task: str | None = Field(
        default=None,
        description="Optional task text for auto effort selection.",
    )
    task_tokens: int | None = Field(
        default=None,
        description="Optional task token estimate for auto effort.",
    )
    task_chars: int | None = Field(
        default=None,
        description="Optional task character count for auto effort.",
    )
    base_args: dict | None = Field(
        default=None,
        description="Base args to merge with the effort profile.",
    )
    overrides: dict | None = Field(
        default=None,
        description="Override profile values.",
    )
    override: bool = Field(
        default=True,
        description="Override base args with profile values.",
    )


class EffortPolicyResult(BaseModel):
    action: str
    effort: str | None
    target: str | None
    profile: EffortProfile | None
    profiles: list[EffortProfile] | None
    applied_args: dict | None
    estimated_tokens: int | None
    warnings: list[str]
    errors: list[str]


class EffortPolicyConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    auto_token_thresholds: list[int] = Field(
        default_factory=lambda: [200, 800],
        description="Token thresholds for auto effort (low, medium).",
    )
    auto_char_thresholds: list[int] = Field(
        default_factory=lambda: [800, 3200],
        description="Character thresholds for auto effort (low, medium).",
    )


class EffortPolicyState(BaseToolState):
    pass


class EffortPolicy(
    BaseTool[
        EffortPolicyArgs,
        EffortPolicyResult,
        EffortPolicyConfig,
        EffortPolicyState,
    ],
    ToolUIData[EffortPolicyArgs, EffortPolicyResult],
):
    description: ClassVar[str] = (
        "Select an effort profile and map it to tool arguments."
    )

    async def run(self, args: EffortPolicyArgs) -> EffortPolicyResult:
        action = (args.action or "select").strip().lower()
        if action not in {"select", "list"}:
            raise ToolError("action must be select or list.")

        warnings: list[str] = []
        errors: list[str] = []

        if action == "list":
            profiles = self._list_profiles()
            return EffortPolicyResult(
                action=action,
                effort=None,
                target=None,
                profile=None,
                profiles=profiles,
                applied_args=None,
                estimated_tokens=None,
                warnings=warnings,
                errors=errors,
            )

        effort = self._normalize_effort(args.effort)
        estimated_tokens = self._estimate_tokens(args)
        if effort == "auto":
            effort = self._auto_effort(args, estimated_tokens, warnings)

        if effort == "custom" and not args.overrides:
            warnings.append("custom effort provided without overrides; using medium defaults.")
            effort = "medium"

        profile = self._build_profile(effort, args.overrides, warnings)
        self._validate_profile(profile)

        target = self._normalize_target(args.target)
        applied_args = self._apply_profile(
            profile,
            target,
            args.base_args,
            args.override,
            warnings,
        )

        return EffortPolicyResult(
            action=action,
            effort=profile.name,
            target=target,
            profile=profile,
            profiles=None,
            applied_args=applied_args,
            estimated_tokens=estimated_tokens,
            warnings=warnings,
            errors=errors,
        )

    def _normalize_effort(self, value: str | None) -> str:
        effort = (value or "medium").strip().lower()
        if effort not in VALID_EFFORT:
            raise ToolError(
                f"effort must be one of: {', '.join(sorted(VALID_EFFORT))}"
            )
        return effort

    def _normalize_target(self, value: str | None) -> str:
        target = (value or "extended_thinking").strip().lower()
        if target not in VALID_TARGETS:
            raise ToolError(
                f"target must be one of: {', '.join(sorted(VALID_TARGETS))}"
            )
        return target

    def _estimate_tokens(self, args: EffortPolicyArgs) -> int | None:
        if args.task_tokens is not None and args.task_tokens > 0:
            return args.task_tokens
        if args.task_chars is not None and args.task_chars > 0:
            return max(1, args.task_chars // 4)
        if args.task and args.task.strip():
            return max(1, len(args.task.strip()) // 4)
        return None

    def _auto_effort(
        self,
        args: EffortPolicyArgs,
        estimated_tokens: int | None,
        warnings: list[str],
    ) -> str:
        token_thresholds = self._sorted_thresholds(self.config.auto_token_thresholds)
        char_thresholds = self._sorted_thresholds(self.config.auto_char_thresholds)

        if estimated_tokens is not None:
            if estimated_tokens <= token_thresholds[0]:
                return "low"
            if estimated_tokens <= token_thresholds[1]:
                return "medium"
            return "high"

        if args.task_chars is not None and args.task_chars > 0:
            if args.task_chars <= char_thresholds[0]:
                return "low"
            if args.task_chars <= char_thresholds[1]:
                return "medium"
            return "high"

        warnings.append("auto effort requested but no task size provided; using medium.")
        return "medium"

    def _sorted_thresholds(self, values: list[int]) -> list[int]:
        if len(values) < 2:
            return [200, 800]
        thresholds = [int(values[0]), int(values[1])]
        thresholds.sort()
        return thresholds

    def _build_profile(
        self,
        effort: str,
        overrides: dict | None,
        warnings: list[str],
    ) -> EffortProfile:
        base = DEFAULT_PROFILES.get(effort, DEFAULT_PROFILES["medium"])
        profile_data = dict(base)
        if overrides:
            if not isinstance(overrides, dict):
                raise ToolError("overrides must be an object.")
            for key, value in overrides.items():
                if key not in PROFILE_FIELDS:
                    warnings.append(f"Unknown override field ignored: {key}")
                    continue
                profile_data[key] = value

        return EffortProfile(name=effort, **profile_data)

    def _validate_profile(self, profile: EffortProfile) -> None:
        if profile.reasoning_max_tokens <= 0:
            raise ToolError("reasoning_max_tokens must be positive.")
        if profile.answer_max_tokens <= 0:
            raise ToolError("answer_max_tokens must be positive.")
        if profile.reasoning_temperature < 0:
            raise ToolError("reasoning_temperature cannot be negative.")
        if profile.answer_temperature < 0:
            raise ToolError("answer_temperature cannot be negative.")
        if profile.max_retries < 0:
            raise ToolError("max_retries must be >= 0.")
        if profile.validation_mode not in VALID_VALIDATION:
            raise ToolError(
                f"validation_mode must be one of: {', '.join(sorted(VALID_VALIDATION))}"
            )

    def _apply_profile(
        self,
        profile: EffortProfile,
        target: str,
        base_args: dict | None,
        override: bool,
        warnings: list[str],
    ) -> dict | None:
        if target == "none":
            return base_args or {}
        if base_args is not None and not isinstance(base_args, dict):
            raise ToolError("base_args must be an object.")

        merged = dict(base_args or {})

        if target == "extended_thinking":
            mapping = {
                "reasoning_max_tokens": profile.reasoning_max_tokens,
                "answer_max_tokens": profile.answer_max_tokens,
                "reasoning_temperature": profile.reasoning_temperature,
                "answer_temperature": profile.answer_temperature,
                "max_retries": profile.max_retries,
                "validation_mode": profile.validation_mode,
                "use_scratchpad": profile.use_scratchpad,
                "include_reasoning_summary": profile.include_reasoning_summary,
            }
        elif target == "structured_outputs":
            mapping = {
                "llm_temperature": profile.answer_temperature,
                "llm_max_tokens": profile.answer_max_tokens,
                "max_retries": profile.max_retries,
            }
            if profile.validation_mode != "none":
                mapping["strict_json"] = True
            if profile.validation_mode == "none":
                mapping["max_retries"] = 0
        else:
            raise ToolError("Unsupported target.")

        merged = self._merge_args(merged, mapping, override)

        if target == "extended_thinking":
            if merged.get("validation_mode") == "json_schema":
                if not merged.get("output_schema"):
                    warnings.append(
                        "validation_mode json_schema requires output_schema; setting to none."
                    )
                    merged["validation_mode"] = "none"

        return merged

    def _merge_args(
        self, base: dict, mapping: dict, override: bool
    ) -> dict:
        if override:
            base.update(mapping)
            return base
        for key, value in mapping.items():
            if key not in base:
                base[key] = value
        return base

    def _list_profiles(self) -> list[EffortProfile]:
        profiles = []
        for name in ("low", "medium", "high"):
            data = DEFAULT_PROFILES[name]
            profiles.append(EffortProfile(name=name, **data))
        return profiles

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, EffortPolicyArgs):
            return ToolCallDisplay(summary="effort_policy")
        return ToolCallDisplay(
            summary="effort_policy",
            details={
                "action": event.args.action,
                "effort": event.args.effort,
                "target": event.args.target,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, EffortPolicyResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = "Effort policy selected"
        if event.result.action == "list":
            message = "Effort profiles listed"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "effort": event.result.effort,
                "target": event.result.target,
                "profile": event.result.profile,
                "applied_args": event.result.applied_args,
                "profiles": event.result.profiles,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Selecting effort policy"
