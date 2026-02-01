from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from pydantic import BaseModel, Field

from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData

try:
    import cupy as cp
except Exception:  # pragma: no cover - optional
    cp = None

if TYPE_CHECKING:
    from vibe.core.types import ToolCallEvent, ToolResultEvent


DEFAULT_PROMPT_TEMPLATE = """### VIRTUAL NPU MODE (OPT-IN)
Use the virtual neural processing unit (NPU) to accelerate routing, caching,
and signal profiling for LLM responses.
- Use NPU signal profiles for routing and caching decisions.
- Keep internal routing private; do not mention this mode in the response.
- If the user asks for steps or show_steps is enabled, give a concise outline.

NPU style: {npu_style}
Routing strategy: {routing_strategy}
Show steps: {show_steps}
"""

TOOL_PROMPT = (
    "Use `virtual_npu` to route and accelerate LLM/llama.cpp responses with a "
    "GPU-backed virtual NPU and optional NEAT routing. Provide `prompt` or "
    "`messages`, and optionally set routing and cache controls."
)


@dataclass(frozen=True)
class _CacheEntry:
    prompt: str
    response: str
    vector: np.ndarray
    route: str
    model: str
    created_at: float


class VirtualNPUMessage(BaseModel):
    role: str
    content: str


class VirtualNPUArgs(BaseModel):
    prompt: str | None = Field(
        default=None, description="User prompt to solve."
    )
    messages: list[VirtualNPUMessage] | None = Field(
        default=None, description="Optional chat messages."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prefix."
    )
    route: str | None = Field(
        default=None,
        description="Routing mode: auto, llm, llamacpp, both, cache-only.",
    )
    routing_strategy: str | None = Field(
        default=None, description="Routing strategy label."
    )
    npu_style: str | None = Field(
        default=None, description="NPU style label."
    )
    npu_vector_dim: int | None = Field(
        default=None, description="Signal vector dimension."
    )
    use_gpu: bool | None = Field(
        default=True, description="Use cupy for GPU acceleration when available."
    )
    cache_enabled: bool | None = Field(
        default=True, description="Enable NPU cache."
    )
    cache_similarity_threshold: float | None = Field(
        default=None, description="Cosine similarity threshold for cache hit."
    )
    cache_max_items: int | None = Field(
        default=None, description="Maximum cache entries."
    )
    auto_llamacpp_max_chars: int | None = Field(
        default=None, description="Prompt length cutoff for llama.cpp in auto mode."
    )
    worker_threads: int | None = Field(
        default=None,
        description="Thread count for NPU routing (defaults to logical cores).",
    )
    show_steps: bool | None = Field(
        default=None, description="Whether to include a step outline."
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
    llamacpp_api_base: str | None = Field(
        default=None, description="llama.cpp OpenAI-compatible API base URL."
    )
    llamacpp_model: str | None = Field(
        default=None, description="llama.cpp model name."
    )
    llamacpp_temperature: float = Field(
        default=0.2, description="llama.cpp temperature."
    )
    llamacpp_max_tokens: int = Field(
        default=700, description="llama.cpp max tokens."
    )
    llamacpp_stream: bool = Field(
        default=False, description="Stream llama.cpp tokens."
    )
    neat_enabled: bool | None = Field(
        default=None, description="Enable NEAT-based routing if configured."
    )
    neat_config_path: str | None = Field(
        default=None, description="Path to NEAT config file."
    )
    neat_genome_path: str | None = Field(
        default=None, description="Path to a pickled NEAT genome."
    )
    neat_output_threshold: float | None = Field(
        default=None, description="Threshold for NEAT routing decision."
    )


class VirtualNPUResult(BaseModel):
    answer: str
    system_prompt: str
    messages: list[VirtualNPUMessage]
    route_used: str
    cache_hit: bool
    cache_similarity: float | None
    template_source: str
    npu_profile: dict[str, float]
    warnings: list[str]
    errors: list[str]
    llm_model: str | None
    llamacpp_model: str | None


class VirtualNPUConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    llm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL.",
    )
    llm_model: str = Field(
        default="gpt-oss:latest", description="Default LLM model name."
    )
    llamacpp_api_base: str = Field(
        default="http://127.0.0.1:8080/v1",
        description="llama.cpp OpenAI-compatible API base URL.",
    )
    llamacpp_model: str = Field(
        default="llama.cpp", description="Default llama.cpp model name."
    )
    default_route: str = Field(
        default="auto", description="Default routing mode."
    )
    default_routing_strategy: str = Field(
        default="cache-then-model",
        description="Default routing strategy.",
    )
    default_npu_style: str = Field(
        default="gpu-profiled",
        description="Default NPU style label.",
    )
    default_vector_dim: int = Field(
        default=256, description="Default signal vector dimension."
    )
    default_cache_similarity: float = Field(
        default=0.92, description="Default cache similarity threshold."
    )
    default_cache_max_items: int = Field(
        default=128, description="Default cache size."
    )
    default_auto_llamacpp_max_chars: int = Field(
        default=800, description="Default auto mode prompt cutoff."
    )
    default_show_steps: bool = Field(
        default=False, description="Default for show_steps."
    )
    default_neat_output_threshold: float = Field(
        default=0.55, description="Default NEAT routing threshold."
    )
    prompt_path: Path | None = Field(
        default=Path.home()
        / "mistral-vibe"
        / "vibe"
        / "core"
        / "prompts"
        / "virtual_npu.md",
        description="Optional path to a prompt template.",
    )
    prompt_max_chars: int = Field(
        default=8000, description="Maximum template characters to load."
    )


class VirtualNPUState(BaseToolState):
    pass


class VirtualNPU(
    BaseTool[
        VirtualNPUArgs,
        VirtualNPUResult,
        VirtualNPUConfig,
        VirtualNPUState,
    ],
    ToolUIData[VirtualNPUArgs, VirtualNPUResult],
):
    description: ClassVar[str] = (
        "Virtual neural processing unit that accelerates routing and caching."
    )

    @classmethod
    def get_tool_prompt(cls) -> str | None:
        return TOOL_PROMPT

    async def run(self, args: VirtualNPUArgs) -> VirtualNPUResult:
        warnings: list[str] = []
        errors: list[str] = []

        route = (args.route or self.config.default_route).strip().lower()
        if not route:
            route = "auto"

        routing_strategy = (
            args.routing_strategy or self.config.default_routing_strategy
        ).strip()
        if not routing_strategy:
            routing_strategy = "cache-then-model"

        npu_style = (args.npu_style or self.config.default_npu_style).strip()
        if not npu_style:
            npu_style = "gpu-profiled"

        vector_dim = args.npu_vector_dim or self.config.default_vector_dim
        if vector_dim <= 0:
            raise ToolError("npu_vector_dim must be positive.")

        cache_enabled = args.cache_enabled if args.cache_enabled is not None else True
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
        if cache_similarity <= 0 or cache_similarity >= 1:
            raise ToolError("cache_similarity_threshold must be between 0 and 1.")
        if cache_max_items <= 0:
            raise ToolError("cache_max_items must be positive.")

        auto_llamacpp_max_chars = (
            args.auto_llamacpp_max_chars
            if args.auto_llamacpp_max_chars is not None
            else self.config.default_auto_llamacpp_max_chars
        )
        if auto_llamacpp_max_chars <= 0:
            raise ToolError("auto_llamacpp_max_chars must be positive.")

        use_gpu = bool(args.use_gpu)
        if use_gpu and cp is None:
            warnings.append("cupy unavailable; falling back to numpy.")
            use_gpu = False

        worker_threads = (
            args.worker_threads if args.worker_threads is not None else os.cpu_count()
        )
        if not worker_threads or worker_threads <= 0:
            worker_threads = 1

        show_steps = (
            args.show_steps
            if args.show_steps is not None
            else self.config.default_show_steps
        )

        self._validate_llm_settings(args)

        template, template_source = self._load_template(warnings)
        base_system_prompt = self._build_system_prompt(
            template, npu_style, routing_strategy, show_steps, args.system_prompt
        )

        messages = self._normalize_messages(args, base_system_prompt)
        prompt_text = self._extract_prompt_text(messages)

        cache_hit = False
        cache_similarity_score: float | None = None
        cache_entry: _CacheEntry | None = None

        self._ensure_cache()
        vector = self._build_signal_vector(prompt_text, vector_dim, use_gpu)
        npu_profile = self._build_profile(vector)

        if cache_enabled and self._cache:
            cache_entry, cache_similarity_score = self._find_cache_match(
                vector, use_gpu
            )
            if cache_entry and cache_similarity_score >= cache_similarity:
                cache_hit = True

        route_used = route
        if cache_hit:
            route_used = "cache"
            answer = cache_entry.response if cache_entry else ""
        else:
            route_used = self._select_route(
                route,
                prompt_text,
                auto_llamacpp_max_chars,
                args,
                cache_similarity_score,
                warnings,
            )
            answer = self._execute_route(
                route_used,
                messages,
                args,
                worker_threads,
                warnings,
            )

        if not cache_hit and cache_enabled and answer:
            self._update_cache(
                prompt_text,
                answer,
                vector,
                route_used,
                self._resolve_model_for_route(route_used, args),
                cache_max_items,
            )

        return VirtualNPUResult(
            answer=answer,
            system_prompt=base_system_prompt,
            messages=messages,
            route_used=route_used,
            cache_hit=cache_hit,
            cache_similarity=cache_similarity_score,
            template_source=template_source,
            npu_profile=npu_profile,
            warnings=warnings,
            errors=errors,
            llm_model=self._resolve_llm_model(args),
            llamacpp_model=self._resolve_llamacpp_model(args),
        )

    def _ensure_cache(self) -> None:
        if not hasattr(self, "_cache"):
            self._cache: list[_CacheEntry] = []

    def _build_signal_vector(
        self, text: str, dim: int, use_gpu: bool
    ) -> np.ndarray:
        digest = hashlib.blake2b(
            text.encode("utf-8", errors="ignore"), digest_size=64
        ).digest()
        raw = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
        vector = np.resize(raw, dim)
        vector = (vector - vector.mean()) / (vector.std() + 1e-6)
        if use_gpu and cp is not None:
            vec_gpu = cp.asarray(vector)
            vec_gpu = cp.tanh(vec_gpu)
            vector = cp.asnumpy(vec_gpu)
        return vector.astype(np.float32)

    def _build_profile(self, vector: np.ndarray) -> dict[str, float]:
        norm = float(np.linalg.norm(vector))
        mean = float(vector.mean())
        std = float(vector.std())
        max_abs = float(np.max(np.abs(vector))) if vector.size else 0.0
        return {
            "norm": norm,
            "mean": mean,
            "std": std,
            "max_abs": max_abs,
        }

    def _find_cache_match(
        self, vector: np.ndarray, use_gpu: bool
    ) -> tuple[_CacheEntry | None, float | None]:
        if not self._cache:
            return None, None
        vectors = np.vstack([entry.vector for entry in self._cache])
        if use_gpu and cp is not None:
            mat = cp.asarray(vectors)
            vec = cp.asarray(vector)
            dot = mat @ vec
            denom = cp.linalg.norm(mat, axis=1) * (cp.linalg.norm(vec) + 1e-8)
            scores = dot / denom
            scores = cp.asnumpy(scores)
        else:
            dot = vectors @ vector
            denom = np.linalg.norm(vectors, axis=1) * (np.linalg.norm(vector) + 1e-8)
            scores = dot / denom
        best_idx = int(np.argmax(scores))
        return self._cache[best_idx], float(scores[best_idx])

    def _select_route(
        self,
        route: str,
        prompt_text: str,
        auto_llamacpp_max_chars: int,
        args: VirtualNPUArgs,
        cache_similarity: float | None,
        warnings: list[str],
    ) -> str:
        if route in {"llm", "llamacpp", "both"}:
            return route
        if route == "cache-only":
            return "cache-only"

        if self._maybe_route_with_neat(prompt_text, cache_similarity, args, warnings):
            return "llamacpp"

        if len(prompt_text) <= auto_llamacpp_max_chars:
            return "llamacpp"
        return "llm"

    def _maybe_route_with_neat(
        self,
        prompt_text: str,
        cache_similarity: float | None,
        args: VirtualNPUArgs,
        warnings: list[str],
    ) -> bool:
        if args.neat_enabled is False:
            return False
        if not args.neat_config_path or not args.neat_genome_path:
            return False
        try:
            import neat
        except Exception as exc:  # pragma: no cover - optional
            warnings.append(f"NEAT unavailable: {exc}")
            return False

        config_path = self._resolve_path(args.neat_config_path)
        genome_path = self._resolve_path(args.neat_genome_path)
        try:
            config = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                str(config_path),
            )
        except Exception as exc:
            warnings.append(f"Failed to load NEAT config: {exc}")
            return False
        try:
            with open(genome_path, "rb") as handle:
                genome = pickle.load(handle)
        except Exception as exc:
            warnings.append(f"Failed to load NEAT genome: {exc}")
            return False

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        length_norm = min(len(prompt_text) / 4000.0, 1.0)
        similarity = cache_similarity if cache_similarity is not None else 0.0
        cache_ratio = len(self._cache) / max(1.0, float(self.config.default_cache_max_items))
        features = [length_norm, similarity, cache_ratio]
        output = net.activate(features)[0]
        threshold = (
            args.neat_output_threshold
            if args.neat_output_threshold is not None
            else self.config.default_neat_output_threshold
        )
        return output >= threshold

    def _execute_route(
        self,
        route: str,
        messages: list[VirtualNPUMessage],
        args: VirtualNPUArgs,
        worker_threads: int,
        warnings: list[str],
    ) -> str:
        if route == "cache-only":
            warnings.append("cache-only route selected with no cache hit.")
            return ""
        if route == "llm":
            return self._call_openai(
                api_base=self._resolve_llm_base(args),
                model=self._resolve_llm_model(args),
                messages=messages,
                temperature=args.llm_temperature,
                max_tokens=args.llm_max_tokens,
                stream=args.llm_stream,
            )
        if route == "llamacpp":
            return self._call_openai(
                api_base=self._resolve_llamacpp_base(args),
                model=self._resolve_llamacpp_model(args),
                messages=messages,
                temperature=args.llamacpp_temperature,
                max_tokens=args.llamacpp_max_tokens,
                stream=args.llamacpp_stream,
            )
        if route == "both":
            if args.llm_stream or args.llamacpp_stream:
                warnings.append("Streaming disabled for route=both.")
            with ThreadPoolExecutor(max_workers=worker_threads) as executor:
                futures = {
                    executor.submit(
                        self._call_openai,
                        api_base=self._resolve_llm_base(args),
                        model=self._resolve_llm_model(args),
                        messages=messages,
                        temperature=args.llm_temperature,
                        max_tokens=args.llm_max_tokens,
                        stream=False,
                    ): "llm",
                    executor.submit(
                        self._call_openai,
                        api_base=self._resolve_llamacpp_base(args),
                        model=self._resolve_llamacpp_model(args),
                        messages=messages,
                        temperature=args.llamacpp_temperature,
                        max_tokens=args.llamacpp_max_tokens,
                        stream=False,
                    ): "llamacpp",
                }
                for future in as_completed(futures):
                    try:
                        return future.result()
                    except Exception as exc:
                        warnings.append(f"Route {futures[future]} failed: {exc}")
            return ""
        return ""

    def _update_cache(
        self,
        prompt: str,
        response: str,
        vector: np.ndarray,
        route: str,
        model: str,
        max_items: int,
    ) -> None:
        entry = _CacheEntry(
            prompt=prompt,
            response=response,
            vector=vector,
            route=route,
            model=model,
            created_at=time.time(),
        )
        self._cache.append(entry)
        if len(self._cache) > max_items:
            self._cache = self._cache[-max_items:]

    def _resolve_model_for_route(self, route: str, args: VirtualNPUArgs) -> str:
        if route == "llamacpp":
            return self._resolve_llamacpp_model(args)
        return self._resolve_llm_model(args)

    def _resolve_llm_base(self, args: VirtualNPUArgs) -> str:
        return (args.llm_api_base or self.config.llm_api_base).rstrip("/")

    def _resolve_llm_model(self, args: VirtualNPUArgs) -> str:
        return args.llm_model or self.config.llm_model

    def _resolve_llamacpp_base(self, args: VirtualNPUArgs) -> str:
        return (args.llamacpp_api_base or self.config.llamacpp_api_base).rstrip("/")

    def _resolve_llamacpp_model(self, args: VirtualNPUArgs) -> str:
        return args.llamacpp_model or self.config.llamacpp_model

    def _validate_llm_settings(self, args: VirtualNPUArgs) -> None:
        if args.llm_temperature < 0 or args.llamacpp_temperature < 0:
            raise ToolError("Temperatures cannot be negative.")
        if args.llm_max_tokens <= 0 or args.llamacpp_max_tokens <= 0:
            raise ToolError("Max tokens must be positive.")

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

    def _resolve_path(self, raw_path: str) -> Path:
        if not raw_path.strip():
            raise ToolError("Path cannot be empty.")
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
        npu_style: str,
        routing_strategy: str,
        show_steps: bool,
        prefix: str | None,
    ) -> str:
        show_steps_text = "yes" if show_steps else "no"
        rendered = self._render_template(
            template, npu_style, routing_strategy, show_steps_text
        )
        if prefix and prefix.strip():
            return f"{prefix.strip()}\n\n{rendered}".strip()
        return rendered.strip()

    def _render_template(
        self,
        template: str,
        npu_style: str,
        routing_strategy: str,
        show_steps_text: str,
    ) -> str:
        rendered = template
        rendered = rendered.replace("{npu_style}", npu_style)
        rendered = rendered.replace("{routing_strategy}", routing_strategy)
        rendered = rendered.replace("{show_steps}", show_steps_text)
        return rendered

    def _normalize_messages(
        self, args: VirtualNPUArgs, system_prompt: str
    ) -> list[VirtualNPUMessage]:
        messages: list[VirtualNPUMessage] = []
        if args.messages:
            for msg in args.messages:
                role = (msg.role or "").strip()
                content = (msg.content or "").strip()
                if not content:
                    continue
                if not role:
                    role = "user"
                messages.append(VirtualNPUMessage(role=role, content=content))
        elif args.prompt and args.prompt.strip():
            messages.append(
                VirtualNPUMessage(role="user", content=args.prompt.strip())
            )
        else:
            raise ToolError("Provide prompt or messages.")

        if not messages:
            raise ToolError("No usable messages provided.")

        if system_prompt.strip():
            messages.insert(
                0, VirtualNPUMessage(role="system", content=system_prompt.strip())
            )
        return messages

    def _extract_prompt_text(self, messages: list[VirtualNPUMessage]) -> str:
        parts = [msg.content for msg in messages if msg.role == "user"]
        return "\n\n".join(parts).strip()

    def _call_openai(
        self,
        *,
        api_base: str,
        model: str,
        messages: list[VirtualNPUMessage],
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> str:
        url = api_base.rstrip("/") + "/chat/completions"
        payload = {
            "model": model,
            "messages": [msg.model_dump() for msg in messages],
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

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, VirtualNPUArgs):
            return ToolCallDisplay(summary="virtual_npu")
        return ToolCallDisplay(
            summary="virtual_npu",
            details={
                "route": event.args.route,
                "routing_strategy": event.args.routing_strategy,
                "npu_style": event.args.npu_style,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, VirtualNPUResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = "Virtual NPU complete"
        if event.result.errors:
            message = "Virtual NPU finished with errors"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "answer": event.result.answer,
                "route_used": event.result.route_used,
                "cache_hit": event.result.cache_hit,
                "cache_similarity": event.result.cache_similarity,
                "npu_profile": event.result.npu_profile,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Virtual NPU"
