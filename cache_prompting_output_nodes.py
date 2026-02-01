from __future__ import annotations

import difflib
import hashlib
import json
import math
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
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


DEFAULT_PROMPT_TEMPLATE = """### CACHE PROMPTING: OUTPUT NODES (OPT-IN)
Choose the number of output nodes to classify cache usage.
- Score cache candidates using similarity, length match, and recency.
- Use the selected output node to decide cache usage.
- Keep cache routing private; do not mention this mode in the response.
- If the user asks for steps or show_steps is enabled, provide a short outline.

Cache mode: {cache_mode}
Similarity threshold: {cache_similarity}
Decision threshold: {decision_threshold}
Output nodes: {output_nodes}
Output activation: {output_activation}
Cache node index: {cache_node_index}
Show steps: {show_steps}
"""

TOOL_PROMPT = (
    "Use `cache_prompting_output_nodes` for cache prompting with a selectable "
    "number of output nodes."
)

FEATURE_NAMES = ("similarity", "length_match", "recency")
VALID_CACHE_MODES = {"cache-first", "cache-only", "skip"}
VALID_ACTIVATIONS = {"softmax", "sigmoid", "linear"}


class _CacheEntry(BaseModel):
    entry_id: str
    prompt: str
    response: str
    created_at: float
    last_used_at: float | None = None
    usage_count: int = 0
    model: str | None = None
    prompt_hash: str | None = None


@dataclass(frozen=True)
class _CacheCandidate:
    entry: _CacheEntry
    similarity: float
    raw_scores: list[float]
    values: list[float]
    cache_value: float
    features: list[float]


class CachePromptingOutputMessage(BaseModel):
    role: str
    content: str


class CachePromptingOutputArgs(BaseModel):
    prompt: str | None = Field(
        default=None, description="User prompt to solve."
    )
    messages: list[CachePromptingOutputMessage] | None = Field(
        default=None, description="Optional chat messages."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prefix."
    )
    show_steps: bool | None = Field(
        default=None, description="Whether to include a step outline."
    )
    cache_mode: str | None = Field(
        default=None, description="cache-first, cache-only, or skip."
    )
    cache_enabled: bool | None = Field(
        default=None, description="Enable cache prompting."
    )
    cache_path: str | None = Field(
        default=None, description="Override cache storage path."
    )
    cache_store: bool | None = Field(
        default=None, description="Store new responses in the cache."
    )
    cache_similarity_threshold: float | None = Field(
        default=None, description="Minimum similarity for cache hits."
    )
    cache_max_items: int | None = Field(
        default=None, description="Maximum cache entries to keep."
    )
    output_nodes: int | None = Field(
        default=None, description="Number of output nodes."
    )
    cache_node_index: int | None = Field(
        default=None, description="Index of the cache output node."
    )
    output_activation: str | None = Field(
        default=None, description="softmax, sigmoid, or linear."
    )
    output_weights: list[list[float]] | None = Field(
        default=None, description="Override output weights."
    )
    output_biases: list[float] | None = Field(
        default=None, description="Override output biases."
    )
    weights_path: str | None = Field(
        default=None, description="Override weights storage path."
    )
    decision_threshold: float | None = Field(
        default=None, description="Decision threshold for cache hit."
    )
    llm_api_base: str | None = Field(
        default=None, description="OpenAI-compatible API base URL."
    )
    llm_model: str | None = Field(
        default=None, description="LLM model name."
    )
    llm_temperature: float = Field(
        default=0.2, description="LLM temperature."
    )
    llm_max_tokens: int = Field(
        default=700, description="LLM max tokens."
    )
    llm_stream: bool = Field(
        default=False, description="Stream LLM tokens."
    )


class CachePromptingOutputResult(BaseModel):
    answer: str
    system_prompt: str
    messages: list[CachePromptingOutputMessage]
    cache_mode: str
    cache_hit: bool
    cache_used: bool
    cache_similarity: float | None
    cache_score: float | None
    cache_entry_id: str | None
    cache_features: dict[str, float] | None
    output_nodes: int
    cache_node_index: int
    output_activation: str
    output_scores: list[float] | None
    output_values: list[float] | None
    output_winner_index: int | None
    decision_threshold: float
    template_source: str
    warnings: list[str]
    errors: list[str]
    llm_model: str | None


class CachePromptingOutputConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    llm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL.",
    )
    llm_model: str = Field(
        default="gpt-oss:latest", description="Default LLM model name."
    )
    default_cache_mode: str = Field(
        default="cache-first", description="Default cache mode."
    )
    default_cache_enabled: bool = Field(
        default=True, description="Enable cache prompting by default."
    )
    default_cache_store: bool = Field(
        default=True, description="Store new responses by default."
    )
    default_cache_similarity: float = Field(
        default=0.86, description="Default cache similarity threshold."
    )
    default_cache_max_items: int = Field(
        default=200, description="Default cache size."
    )
    default_output_nodes: int = Field(
        default=2, description="Default output nodes."
    )
    default_cache_node_index: int = Field(
        default=0, description="Default cache node index."
    )
    default_output_activation: str = Field(
        default="softmax", description="Default output activation."
    )
    default_cache_weights: list[float] = Field(
        default_factory=lambda: [0.7, 0.2, 0.1],
        description="Default cache node weights.",
    )
    default_cache_bias: float = Field(
        default=0.0, description="Default cache node bias."
    )
    default_other_weights: list[float] = Field(
        default_factory=lambda: [-0.4, -0.15, -0.05],
        description="Default non-cache node weights.",
    )
    default_other_bias: float = Field(
        default=-0.2, description="Default non-cache node bias."
    )
    default_decision_threshold: float = Field(
        default=0.55, description="Default decision threshold."
    )
    default_show_steps: bool = Field(
        default=False, description="Default for show_steps."
    )
    cache_path: Path = Field(
        default=Path.home()
        / ".vibe"
        / "cache"
        / "cache_prompting_output_nodes.json",
        description="Default cache storage path.",
    )
    weights_path: Path = Field(
        default=Path.home()
        / ".vibe"
        / "cache"
        / "cache_prompting_output_weights.json",
        description="Default weights storage path.",
    )
    prompt_path: Path | None = Field(
        default=Path.home()
        / "mistral-vibe"
        / "vibe"
        / "core"
        / "prompts"
        / "cache_prompting_output_nodes.md",
        description="Optional path to a prompt template.",
    )
    prompt_max_chars: int = Field(
        default=8000, description="Maximum template characters to load."
    )


class CachePromptingOutputState(BaseToolState):
    pass


class CachePromptingOutputNodes(
    BaseTool[
        CachePromptingOutputArgs,
        CachePromptingOutputResult,
        CachePromptingOutputConfig,
        CachePromptingOutputState,
    ],
    ToolUIData[CachePromptingOutputArgs, CachePromptingOutputResult],
):
    description: ClassVar[str] = (
        "Cache prompting with configurable output nodes."
    )

    @classmethod
    def get_tool_prompt(cls) -> str | None:
        return TOOL_PROMPT

    async def run(
        self, args: CachePromptingOutputArgs
    ) -> CachePromptingOutputResult:
        warnings: list[str] = []
        errors: list[str] = []

        cache_mode = (args.cache_mode or self.config.default_cache_mode).strip().lower()
        if cache_mode not in VALID_CACHE_MODES:
            raise ToolError(
                f"cache_mode must be one of {sorted(VALID_CACHE_MODES)}."
            )

        cache_enabled = (
            args.cache_enabled
            if args.cache_enabled is not None
            else self.config.default_cache_enabled
        )
        cache_store = (
            args.cache_store
            if args.cache_store is not None
            else self.config.default_cache_store
        )
        cache_similarity = (
            args.cache_similarity_threshold
            if args.cache_similarity_threshold is not None
            else self.config.default_cache_similarity
        )
        cache_max_items = (
            args.cache_max_items
            if args.cache_max_items is not None
            else self.config.default_cache_max_items
        )
        output_nodes = (
            args.output_nodes
            if args.output_nodes is not None
            else self.config.default_output_nodes
        )
        cache_node_index = (
            args.cache_node_index
            if args.cache_node_index is not None
            else self.config.default_cache_node_index
        )
        output_activation = (
            args.output_activation or self.config.default_output_activation
        ).strip().lower()
        decision_threshold = (
            args.decision_threshold
            if args.decision_threshold is not None
            else self._default_threshold(output_activation)
        )
        show_steps = (
            args.show_steps
            if args.show_steps is not None
            else self.config.default_show_steps
        )

        if cache_mode == "skip":
            cache_enabled = False

        self._validate_llm_settings(args)
        self._validate_cache_settings(cache_similarity, cache_max_items)
        self._validate_output_settings(
            output_nodes, cache_node_index, output_activation, decision_threshold
        )

        weights_path = self._resolve_weights_path(args.weights_path)
        weights, biases = self._load_weights(weights_path, output_nodes, warnings)
        if args.output_weights is not None:
            weights = args.output_weights
        if args.output_biases is not None:
            biases = args.output_biases
        weights, biases = self._align_weights(
            weights, biases, output_nodes, warnings
        )

        template, template_source = self._load_template(warnings)
        system_prompt = self._build_system_prompt(
            template,
            cache_mode,
            cache_similarity,
            decision_threshold,
            output_nodes,
            output_activation,
            cache_node_index,
            show_steps,
            args.system_prompt,
        )
        messages = self._normalize_messages(args, system_prompt)
        prompt_text = self._build_prompt_text(messages)

        cache_hit = False
        cache_used = False
        cache_similarity_score: float | None = None
        cache_score: float | None = None
        cache_entry_id: str | None = None
        cache_features: dict[str, float] | None = None
        output_scores: list[float] | None = None
        output_values: list[float] | None = None
        output_winner_index: int | None = None

        cache_entries: list[_CacheEntry] = []
        cache_candidate: _CacheCandidate | None = None
        cache_dirty = False
        cache_path = self._resolve_cache_path(args.cache_path)

        if cache_enabled:
            cache_entries = self._load_cache_entries(cache_path, warnings)
            cache_candidate = self._find_cache_candidate(
                cache_entries,
                prompt_text,
                weights,
                biases,
                output_activation,
                cache_node_index,
            )

        if cache_candidate:
            cache_similarity_score = cache_candidate.similarity
            cache_score = cache_candidate.cache_value
            cache_entry_id = cache_candidate.entry.entry_id
            cache_features = dict(zip(FEATURE_NAMES, cache_candidate.features))
            output_scores = cache_candidate.raw_scores
            output_values = cache_candidate.values
            output_winner_index = self._winner_index(
                output_activation, cache_candidate.raw_scores, cache_candidate.values
            )
            cache_hit = self._cache_decision(
                cache_candidate,
                cache_similarity,
                decision_threshold,
                output_activation,
                cache_node_index,
                output_nodes,
            )

        llm_used = False
        if cache_mode == "cache-only":
            if not cache_hit:
                raise ToolError("cache-only mode requested but no cache hit found.")
            answer = cache_candidate.entry.response if cache_candidate else ""
            cache_used = True
        elif cache_mode == "cache-first" and cache_hit:
            answer = cache_candidate.entry.response if cache_candidate else ""
            cache_used = True
        else:
            answer = self._call_llm(messages, args)
            llm_used = True

        now = time.time()
        if cache_candidate and cache_hit:
            cache_candidate.entry.usage_count += 1
            cache_candidate.entry.last_used_at = now
            cache_dirty = True

        if llm_used and cache_store:
            cache_dirty |= self._upsert_cache_entry(
                cache_entries,
                prompt_text,
                answer,
                self._resolve_model(args),
                now,
            )

        if cache_dirty and cache_enabled:
            self._trim_cache(cache_entries, cache_max_items)
            self._save_cache_entries(cache_path, cache_entries, warnings)

        return CachePromptingOutputResult(
            answer=answer,
            system_prompt=system_prompt,
            messages=messages,
            cache_mode=cache_mode,
            cache_hit=cache_hit,
            cache_used=cache_used,
            cache_similarity=cache_similarity_score,
            cache_score=cache_score,
            cache_entry_id=cache_entry_id,
            cache_features=cache_features,
            output_nodes=output_nodes,
            cache_node_index=cache_node_index,
            output_activation=output_activation,
            output_scores=output_scores,
            output_values=output_values,
            output_winner_index=output_winner_index,
            decision_threshold=decision_threshold,
            template_source=template_source,
            warnings=warnings,
            errors=errors,
            llm_model=self._resolve_model(args),
        )

    def _default_threshold(self, output_activation: str) -> float:
        if output_activation == "linear":
            return 0.0
        if output_activation == "sigmoid":
            return 0.6
        return self.config.default_decision_threshold

    def _validate_llm_settings(self, args: CachePromptingOutputArgs) -> None:
        if args.llm_temperature < 0:
            raise ToolError("llm_temperature cannot be negative.")
        if args.llm_max_tokens <= 0:
            raise ToolError("llm_max_tokens must be positive.")

    def _validate_cache_settings(
        self, cache_similarity: float, cache_max_items: int
    ) -> None:
        if cache_similarity <= 0 or cache_similarity >= 1:
            raise ToolError("cache_similarity_threshold must be between 0 and 1.")
        if cache_max_items <= 0:
            raise ToolError("cache_max_items must be positive.")

    def _validate_output_settings(
        self,
        output_nodes: int,
        cache_node_index: int,
        output_activation: str,
        decision_threshold: float,
    ) -> None:
        if output_nodes <= 0:
            raise ToolError("output_nodes must be positive.")
        if not (0 <= cache_node_index < output_nodes):
            raise ToolError("cache_node_index must be within output_nodes.")
        if output_activation not in VALID_ACTIVATIONS:
            raise ToolError(
                f"output_activation must be one of {sorted(VALID_ACTIVATIONS)}."
            )
        if output_nodes == 1 and output_activation == "softmax":
            raise ToolError("softmax requires output_nodes > 1.")
        if output_nodes > 1 and output_activation == "sigmoid":
            raise ToolError("sigmoid is only supported for output_nodes == 1.")
        if output_activation in {"sigmoid", "softmax"} and not (
            0 <= decision_threshold <= 1
        ):
            raise ToolError(
                "decision_threshold must be between 0 and 1 for this activation."
            )

    def _load_template(self, warnings: list[str]) -> tuple[str, str]:
        template = DEFAULT_PROMPT_TEMPLATE
        source = "embedded"

        if not self.config.prompt_path:
            return self._truncate_template(template, warnings), source

        path = self._resolve_prompt_path(self.config.prompt_path)
        if not path.exists():
            warnings.append(f"Prompt template not found: {path}")
            return self._truncate_template(template, warnings), source
        if path.is_dir():
            warnings.append(f"Prompt template is a directory: {path}")
            return self._truncate_template(template, warnings), source

        try:
            text = path.read_text("utf-8", errors="ignore").strip()
        except OSError as exc:
            warnings.append(f"Failed to read prompt template: {exc}")
            return self._truncate_template(template, warnings), source

        if not text:
            warnings.append(f"Prompt template empty: {path}")
            return self._truncate_template(template, warnings), source

        template = text
        source = str(path)
        return self._truncate_template(template, warnings), source

    def _resolve_prompt_path(self, raw_path: Path | str) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.config.effective_workdir / path
        return path.resolve()

    def _truncate_template(self, template: str, warnings: list[str]) -> str:
        max_chars = self.config.prompt_max_chars
        if max_chars > 0 and len(template) > max_chars:
            warnings.append("Prompt template truncated to prompt_max_chars.")
            return template[:max_chars].rstrip()
        return template

    def _build_system_prompt(
        self,
        template: str,
        cache_mode: str,
        cache_similarity: float,
        decision_threshold: float,
        output_nodes: int,
        output_activation: str,
        cache_node_index: int,
        show_steps: bool,
        prefix: str | None,
    ) -> str:
        rendered = self._render_template(
            template,
            cache_mode,
            cache_similarity,
            decision_threshold,
            output_nodes,
            output_activation,
            cache_node_index,
            show_steps,
        )
        if prefix and prefix.strip():
            return f"{prefix.strip()}\n\n{rendered}".strip()
        return rendered.strip()

    def _render_template(
        self,
        template: str,
        cache_mode: str,
        cache_similarity: float,
        decision_threshold: float,
        output_nodes: int,
        output_activation: str,
        cache_node_index: int,
        show_steps: bool,
    ) -> str:
        show_steps_text = "yes" if show_steps else "no"
        rendered = template
        rendered = rendered.replace("{cache_mode}", cache_mode)
        rendered = rendered.replace("{cache_similarity}", f"{cache_similarity:.3f}")
        rendered = rendered.replace(
            "{decision_threshold}", f"{decision_threshold:.3f}"
        )
        rendered = rendered.replace("{output_nodes}", str(output_nodes))
        rendered = rendered.replace("{output_activation}", output_activation)
        rendered = rendered.replace("{cache_node_index}", str(cache_node_index))
        rendered = rendered.replace("{show_steps}", show_steps_text)
        return rendered

    def _normalize_messages(
        self, args: CachePromptingOutputArgs, system_prompt: str
    ) -> list[CachePromptingOutputMessage]:
        messages: list[CachePromptingOutputMessage] = []
        if args.messages:
            for msg in args.messages:
                role = (msg.role or "").strip()
                content = (msg.content or "").strip()
                if not content:
                    continue
                if not role:
                    role = "user"
                messages.append(
                    CachePromptingOutputMessage(role=role, content=content)
                )
        elif args.prompt and args.prompt.strip():
            messages.append(
                CachePromptingOutputMessage(
                    role="user", content=args.prompt.strip()
                )
            )
        else:
            raise ToolError("Provide prompt or messages.")

        if not messages:
            raise ToolError("No usable messages provided.")

        if system_prompt.strip():
            messages.insert(
                0,
                CachePromptingOutputMessage(
                    role="system", content=system_prompt.strip()
                ),
            )
        return messages

    def _build_prompt_text(
        self, messages: list[CachePromptingOutputMessage]
    ) -> str:
        parts = [msg.content for msg in messages if msg.role != "system"]
        return "\n".join(parts).strip()

    def _resolve_cache_path(self, override: str | None) -> Path:
        if override:
            path = Path(override).expanduser()
        else:
            path = self.config.cache_path
        if not path.is_absolute():
            path = self.config.effective_workdir / path
        return path.resolve()

    def _resolve_weights_path(self, override: str | None) -> Path:
        if override:
            path = Path(override).expanduser()
        else:
            path = self.config.weights_path
        if not path.is_absolute():
            path = self.config.effective_workdir / path
        return path.resolve()

    def _load_cache_entries(
        self, path: Path, warnings: list[str]
    ) -> list[_CacheEntry]:
        if not path.exists():
            return []
        try:
            raw = path.read_text("utf-8")
        except OSError as exc:
            warnings.append(f"Failed to read cache: {exc}")
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            warnings.append(f"Failed to parse cache: {exc}")
            return []
        if not isinstance(data, list):
            warnings.append("Cache content is not a list.")
            return []
        entries: list[_CacheEntry] = []
        for item in data:
            try:
                entries.append(_CacheEntry.model_validate(item))
            except Exception:
                warnings.append("Skipped invalid cache entry.")
        return entries

    def _save_cache_entries(
        self, path: Path, entries: list[_CacheEntry], warnings: list[str]
    ) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = [entry.model_dump() for entry in entries]
            path.write_text(
                json.dumps(payload, ensure_ascii=True, indent=2), "utf-8"
            )
        except OSError as exc:
            warnings.append(f"Failed to save cache: {exc}")

    def _trim_cache(self, entries: list[_CacheEntry], max_items: int) -> None:
        if len(entries) <= max_items:
            return
        entries.sort(
            key=lambda entry: entry.last_used_at or entry.created_at
        )
        del entries[:-max_items]

    def _load_weights(
        self,
        path: Path,
        output_nodes: int,
        warnings: list[str],
    ) -> tuple[list[list[float]], list[float]]:
        if not path.exists():
            return self._default_weights(output_nodes)
        try:
            raw = path.read_text("utf-8")
            data = json.loads(raw)
        except Exception as exc:
            warnings.append(f"Failed to read weights: {exc}")
            return self._default_weights(output_nodes)
        weights = data.get("weights")
        biases = data.get("biases")
        if not isinstance(weights, list) or not isinstance(biases, list):
            warnings.append("Weights file invalid; using defaults.")
            return self._default_weights(output_nodes)
        return weights, biases

    def _default_weights(
        self, output_nodes: int
    ) -> tuple[list[list[float]], list[float]]:
        weights: list[list[float]] = []
        biases: list[float] = []
        for idx in range(output_nodes):
            if idx == 0:
                weights.append(list(self.config.default_cache_weights))
                biases.append(self.config.default_cache_bias)
            else:
                weights.append(list(self.config.default_other_weights))
                biases.append(self.config.default_other_bias)
        return weights, biases

    def _align_weights(
        self,
        weights: list[list[float]],
        biases: list[float],
        output_nodes: int,
        warnings: list[str],
    ) -> tuple[list[list[float]], list[float]]:
        aligned_weights = list(weights)
        aligned_biases = list(biases)

        if len(aligned_weights) < output_nodes:
            extra_weights, extra_biases = self._default_weights(
                output_nodes - len(aligned_weights)
            )
            aligned_weights.extend(extra_weights)
            aligned_biases.extend(extra_biases)
            warnings.append("Output weights padded for missing nodes.")
        elif len(aligned_weights) > output_nodes:
            aligned_weights = aligned_weights[:output_nodes]
            aligned_biases = aligned_biases[:output_nodes]
            warnings.append("Output weights trimmed to output_nodes.")

        if len(aligned_biases) < output_nodes:
            aligned_biases.extend([0.0] * (output_nodes - len(aligned_biases)))
            warnings.append("Output biases padded to output_nodes.")
        elif len(aligned_biases) > output_nodes:
            aligned_biases = aligned_biases[:output_nodes]
            warnings.append("Output biases trimmed to output_nodes.")

        for idx, weight in enumerate(aligned_weights):
            if len(weight) < len(FEATURE_NAMES):
                weight = weight + [0.0] * (len(FEATURE_NAMES) - len(weight))
                aligned_weights[idx] = weight
                warnings.append("Output weight padded to match feature count.")
            elif len(weight) > len(FEATURE_NAMES):
                aligned_weights[idx] = weight[: len(FEATURE_NAMES)]
                warnings.append("Output weight trimmed to match feature count.")

        return aligned_weights, aligned_biases

    def _find_cache_candidate(
        self,
        entries: list[_CacheEntry],
        prompt_text: str,
        weights: list[list[float]],
        biases: list[float],
        output_activation: str,
        cache_node_index: int,
    ) -> _CacheCandidate | None:
        best: _CacheCandidate | None = None
        for entry in entries:
            similarity = self._compute_similarity(prompt_text, entry.prompt)
            features = self._build_features(similarity, prompt_text, entry)
            raw_scores = [
                sum(w * x for w, x in zip(weight, features)) + bias
                for weight, bias in zip(weights, biases)
            ]
            values = self._apply_output_activation(raw_scores, output_activation)
            cache_value = values[cache_node_index]
            if best is None or cache_value > best.cache_value:
                best = _CacheCandidate(
                    entry=entry,
                    similarity=similarity,
                    raw_scores=raw_scores,
                    values=values,
                    cache_value=cache_value,
                    features=features,
                )
        return best

    def _apply_output_activation(
        self, raw_scores: list[float], output_activation: str
    ) -> list[float]:
        if output_activation == "softmax":
            return self._softmax(raw_scores)
        if output_activation == "sigmoid":
            return [self._sigmoid(score) for score in raw_scores]
        return list(raw_scores)

    def _softmax(self, scores: list[float]) -> list[float]:
        if not scores:
            return []
        max_score = max(scores)
        exp_scores = [math.exp(score - max_score) for score in scores]
        total = sum(exp_scores) or 1.0
        return [score / total for score in exp_scores]

    def _sigmoid(self, value: float) -> float:
        if value >= 0:
            exp_neg = math.exp(-value)
            return 1.0 / (1.0 + exp_neg)
        exp_pos = math.exp(value)
        return exp_pos / (1.0 + exp_pos)

    def _winner_index(
        self,
        output_activation: str,
        raw_scores: list[float],
        values: list[float],
    ) -> int:
        if output_activation == "linear":
            return max(range(len(raw_scores)), key=raw_scores.__getitem__)
        return max(range(len(values)), key=values.__getitem__)

    def _cache_decision(
        self,
        candidate: _CacheCandidate,
        similarity_threshold: float,
        decision_threshold: float,
        output_activation: str,
        cache_node_index: int,
        output_nodes: int,
    ) -> bool:
        if candidate.similarity < similarity_threshold:
            return False
        if output_nodes == 1:
            return candidate.values[0] >= decision_threshold
        winner = self._winner_index(
            output_activation, candidate.raw_scores, candidate.values
        )
        if winner != cache_node_index:
            return False
        if output_activation == "linear":
            return candidate.raw_scores[cache_node_index] >= decision_threshold
        return candidate.values[cache_node_index] >= decision_threshold

    def _compute_similarity(self, left: str, right: str) -> float:
        return difflib.SequenceMatcher(None, left, right).ratio()

    def _build_features(
        self, similarity: float, prompt_text: str, entry: _CacheEntry
    ) -> list[float]:
        length_score = self._length_similarity(len(prompt_text), len(entry.prompt))
        recency_score = self._recency_score(entry.created_at)
        return [similarity, length_score, recency_score]

    def _length_similarity(self, left: int, right: int) -> float:
        denom = max(left, right, 1)
        return 1.0 - (abs(left - right) / denom)

    def _recency_score(self, created_at: float) -> float:
        age_hours = max(0.0, (time.time() - created_at) / 3600.0)
        return 1.0 / (1.0 + age_hours / 24.0)

    def _upsert_cache_entry(
        self,
        entries: list[_CacheEntry],
        prompt_text: str,
        response: str,
        model: str,
        now: float,
    ) -> bool:
        prompt_hash = self._hash_text(prompt_text)
        for entry in entries:
            if entry.prompt_hash == prompt_hash:
                entry.prompt = prompt_text
                entry.response = response
                entry.model = model
                entry.prompt_hash = prompt_hash
                return True
        entry = _CacheEntry(
            entry_id=prompt_hash[:12],
            prompt=prompt_text,
            response=response,
            created_at=now,
            last_used_at=None,
            usage_count=0,
            model=model,
            prompt_hash=prompt_hash,
        )
        entries.append(entry)
        return True

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _call_llm(
        self,
        messages: list[CachePromptingOutputMessage],
        args: CachePromptingOutputArgs,
    ) -> str:
        api_base = (args.llm_api_base or self.config.llm_api_base).rstrip("/")
        url = api_base + "/chat/completions"
        payload = {
            "model": self._resolve_model(args),
            "messages": [msg.model_dump() for msg in messages],
            "temperature": args.llm_temperature,
            "max_tokens": args.llm_max_tokens,
            "stream": bool(args.llm_stream),
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

        if not args.llm_stream:
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

    def _resolve_model(self, args: CachePromptingOutputArgs) -> str:
        return args.llm_model or self.config.llm_model

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, CachePromptingOutputArgs):
            return ToolCallDisplay(summary="cache_prompting_output_nodes")
        return ToolCallDisplay(
            summary="cache_prompting_output_nodes",
            details={
                "cache_mode": event.args.cache_mode,
                "output_nodes": event.args.output_nodes,
                "output_activation": event.args.output_activation,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, CachePromptingOutputResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = "Cache prompting (output nodes) complete"
        if event.result.errors:
            message = "Cache prompting (output nodes) finished with errors"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "cache_mode": event.result.cache_mode,
                "cache_hit": event.result.cache_hit,
                "cache_used": event.result.cache_used,
                "cache_similarity": event.result.cache_similarity,
                "cache_score": event.result.cache_score,
                "cache_entry_id": event.result.cache_entry_id,
                "template_source": event.result.template_source,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Cache prompting (output nodes)"
