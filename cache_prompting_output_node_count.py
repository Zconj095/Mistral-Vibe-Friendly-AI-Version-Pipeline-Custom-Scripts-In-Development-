from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import ClassVar


_module_path = Path(__file__).with_name("cache_prompting_output_nodes.py")
_spec = importlib.util.spec_from_file_location(
    "cache_prompting_output_nodes", _module_path
)
if not _spec or not _spec.loader:
    raise RuntimeError("Failed to load cache_prompting_output_nodes module.")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


class CachePromptingOutputNodeCount(_module.CachePromptingOutputNodes):
    description: ClassVar[str] = (
        "Cache prompting with configurable output node count."
    )

    @classmethod
    def get_tool_prompt(cls) -> str | None:
        return (
            "Use `cache_prompting_output_node_count` to select the number of "
            "output nodes for cache prompting."
        )
