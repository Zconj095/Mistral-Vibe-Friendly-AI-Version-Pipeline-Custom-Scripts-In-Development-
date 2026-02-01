from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import ClassVar


_module_path = Path(__file__).with_name("cache_prompting_loss_functions.py")
_spec = importlib.util.spec_from_file_location(
    "cache_prompting_loss_functions", _module_path
)
if not _spec or not _spec.loader:
    raise RuntimeError("Failed to load cache_prompting_loss_functions module.")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


class CachePromptingLossFunctionChoice(_module.CachePromptingLossFunctions):
    description: ClassVar[str] = (
        "Cache prompting with selectable loss function choice."
    )

    @classmethod
    def get_tool_prompt(cls) -> str | None:
        return (
            "Use `cache_prompting_loss_function_choice` to choose a loss "
            "function for cache prompting updates."
        )
