from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, Field

import importlib.util

try:
    from actions_lib import ActionSpec, load_action_specs
except ModuleNotFoundError:  # Fallback when tools directory is not on sys.path.
    _actions_path = Path(__file__).with_name("actions_lib.py")
    _spec = importlib.util.spec_from_file_location("actions_lib", _actions_path)
    if not _spec or not _spec.loader:
        raise
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    ActionSpec = _module.ActionSpec
    load_action_specs = _module.load_action_specs
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


class ActionsRegistryArgs(BaseModel):
    name: str | None = Field(
        default=None, description="Optional action name to fetch."
    )
    action_dir: str | None = Field(
        default=None, description="Override the actions directory."
    )
    include_schema: bool = Field(
        default=False, description="Include parameters and run specs."
    )


class ActionInfo(BaseModel):
    name: str
    description: str
    version: str | None
    permission: str
    path: str
    parameters: dict | None = None
    run: dict | None = None


class ActionsRegistryResult(BaseModel):
    actions: list[ActionInfo]
    count: int
    errors: list[str] = Field(default_factory=list)


class ActionsRegistryConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    actions_dir: Path = Field(
        default=Path("tools/actions"),
        description="Directory containing action spec JSON files.",
    )


class ActionsRegistryState(BaseToolState):
    pass


class ActionsRegistry(
    BaseTool[
        ActionsRegistryArgs,
        ActionsRegistryResult,
        ActionsRegistryConfig,
        ActionsRegistryState,
    ],
    ToolUIData[ActionsRegistryArgs, ActionsRegistryResult],
):
    description: ClassVar[str] = "List locally installed Actions Library specs."

    async def run(self, args: ActionsRegistryArgs) -> ActionsRegistryResult:
        actions_dir = self._resolve_actions_dir(args.action_dir)
        actions, errors = load_action_specs(actions_dir)

        selected = self._select_actions(actions, args.name)
        info = [self._to_info(action, args.include_schema) for action in selected]

        return ActionsRegistryResult(actions=info, count=len(info), errors=errors)

    def _resolve_actions_dir(self, override: str | None) -> Path:
        base = self.config.effective_workdir.resolve()
        if override:
            path = Path(override).expanduser()
            if not path.is_absolute():
                path = base / path
            path = path.resolve()
        else:
            path = (base / self.config.actions_dir).resolve()

        try:
            path.relative_to(base)
        except ValueError as exc:
            raise ToolError(
                f"actions_dir must stay within project root: {base}"
            ) from exc
        return path

    def _select_actions(
        self, actions: dict[str, ActionSpec], name: str | None
    ) -> list[ActionSpec]:
        if name:
            action = actions.get(name)
            if not action:
                available = ", ".join(sorted(actions.keys())) or "none"
                raise ToolError(
                    f"Unknown action '{name}'. Available actions: {available}"
                )
            return [action]
        return sorted(actions.values(), key=lambda spec: spec.name.lower())

    @staticmethod
    def _to_info(action: ActionSpec, include_schema: bool) -> ActionInfo:
        return ActionInfo(
            name=action.name,
            description=action.description,
            version=action.version,
            permission=action.permission,
            path=str(action.source_path),
            parameters=action.parameters if include_schema else None,
            run=action.run if include_schema else None,
        )

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ActionsRegistryArgs):
            return ToolCallDisplay(summary="actions_registry")
        summary = "actions_registry"
        if event.args.name:
            summary = f"actions_registry: {event.args.name}"
        return ToolCallDisplay(
            summary=summary,
            details={
                "action_dir": event.args.action_dir,
                "include_schema": event.args.include_schema,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ActionsRegistryResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = f"Found {event.result.count} action{'' if event.result.count == 1 else 's'}"
        warnings = ["Some action specs failed to load"] if event.result.errors else []
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=warnings,
            details={
                "actions": [action.model_dump() for action in event.result.actions],
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Loading actions"
