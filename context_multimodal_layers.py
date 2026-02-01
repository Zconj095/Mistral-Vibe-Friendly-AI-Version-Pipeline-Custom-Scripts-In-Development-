from __future__ import annotations

import json
from pathlib import Path
import re
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


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}
CODE_EXTS = {
    ".py",
    ".pyi",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".cs",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".go",
    ".rs",
    ".php",
    ".rb",
    ".swift",
    ".kt",
    ".m",
    ".mm",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
}


class ContextMultimodalLayersConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    ollama_url: str = Field(
        default="http://127.0.0.1:11434",
        description="Base URL for the Ollama/GPT-OSS server.",
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        description="Embedding model to use with Ollama/GPT-OSS.",
    )
    max_items: int = Field(default=200, description="Maximum items to process.")
    max_source_bytes: int = Field(
        default=5_000_000, description="Maximum size per file (bytes)."
    )
    max_total_bytes: int = Field(
        default=20_000_000, description="Maximum total bytes across files."
    )
    max_embedding_bytes: int = Field(
        default=200_000, description="Maximum bytes sent to embeddings."
    )
    preview_chars: int = Field(default=400, description="Preview length per item.")
    min_similarity: float = Field(
        default=0.1, description="Minimum similarity for item connections."
    )
    max_item_neighbors: int = Field(
        default=6, description="Maximum neighbors per item."
    )
    min_group_similarity: float = Field(
        default=0.15, description="Minimum similarity for group connections."
    )
    max_group_neighbors: int = Field(
        default=6, description="Maximum neighbors per group."
    )
    max_tokens_per_item: int = Field(
        default=500, description="Maximum tokens stored per item."
    )
    min_token_length: int = Field(
        default=3, description="Minimum token length."
    )
    max_shared_tokens: int = Field(
        default=8, description="Maximum shared tokens returned per connection."
    )
    default_code_extensions: list[str] = Field(
        default=sorted(CODE_EXTS),
        description="Extensions treated as code for auto detection.",
    )


class ContextMultimodalLayersState(BaseToolState):
    pass


class MultimodalItem(BaseModel):
    id: str | None = Field(default=None, description="Optional item identifier.")
    kind: str | None = Field(
        default=None,
        description="text, code, image, audio, video, table, or auto.",
    )
    layer: str = Field(description="Layer name.")
    sector: str | None = Field(default=None, description="Sector name.")
    subsector: str | None = Field(default=None, description="Subsector name.")
    content: str | None = Field(default=None, description="Inline text content.")
    path: str | None = Field(default=None, description="Path to a local file.")
    caption: str | None = Field(
        default=None, description="Caption or transcript for non-text modalities."
    )


class ContextMultimodalLayersArgs(BaseModel):
    items: list[MultimodalItem] = Field(description="Items to embed and connect.")
    embedding_model: str | None = Field(
        default=None, description="Override embedding model."
    )
    max_items: int | None = Field(
        default=None, description="Override maximum items."
    )
    max_source_bytes: int | None = Field(
        default=None, description="Override max_source_bytes."
    )
    max_total_bytes: int | None = Field(
        default=None, description="Override max_total_bytes."
    )
    max_embedding_bytes: int | None = Field(
        default=None, description="Override max_embedding_bytes."
    )
    preview_chars: int | None = Field(
        default=None, description="Override preview length."
    )
    min_similarity: float | None = Field(
        default=None, description="Override minimum item similarity."
    )
    max_item_neighbors: int | None = Field(
        default=None, description="Override max item neighbors."
    )
    min_group_similarity: float | None = Field(
        default=None, description="Override minimum group similarity."
    )
    max_group_neighbors: int | None = Field(
        default=None, description="Override max group neighbors."
    )
    max_shared_tokens: int | None = Field(
        default=None, description="Override max shared tokens."
    )


class ItemNode(BaseModel):
    index: int
    id: str | None
    kind: str
    layer: str
    sector: str | None
    subsector: str | None
    source: str | None
    preview: str


class ItemConnection(BaseModel):
    source_index: int
    target_index: int
    score: float
    shared_tokens: list[str]


class LayerNode(BaseModel):
    layer: str
    item_count: int


class SectorNode(BaseModel):
    layer: str
    sector: str
    item_count: int


class SubsectorNode(BaseModel):
    layer: str
    sector: str
    subsector: str
    item_count: int


class LayerConnection(BaseModel):
    source_layer: str
    target_layer: str
    score: float


class SectorConnection(BaseModel):
    source_layer: str
    source_sector: str
    target_layer: str
    target_sector: str
    score: float


class SubsectorConnection(BaseModel):
    source_layer: str
    source_sector: str
    source_subsector: str
    target_layer: str
    target_sector: str
    target_subsector: str
    score: float


class ContextMultimodalLayersResult(BaseModel):
    items: list[ItemNode]
    item_connections: list[ItemConnection]
    layers: list[LayerNode]
    sectors: list[SectorNode]
    subsectors: list[SubsectorNode]
    layer_connections: list[LayerConnection]
    sector_connections: list[SectorConnection]
    subsector_connections: list[SubsectorConnection]
    item_count: int
    connection_count: int
    truncated: bool
    errors: list[str]


class ContextMultimodalLayers(
    BaseTool[
        ContextMultimodalLayersArgs,
        ContextMultimodalLayersResult,
        ContextMultimodalLayersConfig,
        ContextMultimodalLayersState,
    ],
    ToolUIData[ContextMultimodalLayersArgs, ContextMultimodalLayersResult],
):
    description: ClassVar[str] = (
        "Build layered multimodal connections using embeddings across items."
    )

    async def run(
        self, args: ContextMultimodalLayersArgs
    ) -> ContextMultimodalLayersResult:
        if not args.items:
            raise ToolError("items is required.")

        max_items = args.max_items if args.max_items is not None else self.config.max_items
        if max_items <= 0:
            raise ToolError("max_items must be a positive integer.")
        if len(args.items) > max_items:
            raise ToolError(f"items exceeds max_items ({len(args.items)} > {max_items}).")

        embedding_model = args.embedding_model or self.config.embedding_model
        max_source_bytes = (
            args.max_source_bytes
            if args.max_source_bytes is not None
            else self.config.max_source_bytes
        )
        max_total_bytes = (
            args.max_total_bytes
            if args.max_total_bytes is not None
            else self.config.max_total_bytes
        )
        max_embedding_bytes = (
            args.max_embedding_bytes
            if args.max_embedding_bytes is not None
            else self.config.max_embedding_bytes
        )
        preview_chars = (
            args.preview_chars
            if args.preview_chars is not None
            else self.config.preview_chars
        )
        min_similarity = (
            args.min_similarity
            if args.min_similarity is not None
            else self.config.min_similarity
        )
        max_item_neighbors = (
            args.max_item_neighbors
            if args.max_item_neighbors is not None
            else self.config.max_item_neighbors
        )
        min_group_similarity = (
            args.min_group_similarity
            if args.min_group_similarity is not None
            else self.config.min_group_similarity
        )
        max_group_neighbors = (
            args.max_group_neighbors
            if args.max_group_neighbors is not None
            else self.config.max_group_neighbors
        )
        max_shared_tokens = (
            args.max_shared_tokens
            if args.max_shared_tokens is not None
            else self.config.max_shared_tokens
        )

        if max_source_bytes <= 0:
            raise ToolError("max_source_bytes must be a positive integer.")
        if max_total_bytes <= 0:
            raise ToolError("max_total_bytes must be a positive integer.")
        if max_embedding_bytes <= 0:
            raise ToolError("max_embedding_bytes must be a positive integer.")
        if preview_chars < 0:
            raise ToolError("preview_chars must be >= 0.")
        if min_similarity < 0:
            raise ToolError("min_similarity must be >= 0.")
        if max_item_neighbors < 0:
            raise ToolError("max_item_neighbors must be >= 0.")
        if min_group_similarity < 0:
            raise ToolError("min_group_similarity must be >= 0.")
        if max_group_neighbors < 0:
            raise ToolError("max_group_neighbors must be >= 0.")
        if max_shared_tokens < 0:
            raise ToolError("max_shared_tokens must be >= 0.")

        items: list[ItemNode] = []
        embeddings: list[list[float]] = []
        token_sets: list[set[str]] = []
        errors: list[str] = []
        total_bytes = 0
        truncated = False
        embedding_cache: dict[str, list[float]] = {}

        for idx, item in enumerate(args.items, start=1):
            try:
                if not item.layer or not item.layer.strip():
                    raise ToolError("layer is required for each item.")

                kind = self._resolve_kind(item)
                content, source, size_bytes = self._load_item_content(item, max_source_bytes)
                if size_bytes is not None:
                    if total_bytes + size_bytes > max_total_bytes:
                        truncated = True
                        break
                    total_bytes += size_bytes

                embedding_text = self._build_embedding_text(item, content)
                if not embedding_text:
                    raise ToolError("Item has no text or caption to embed.")

                embedding_text = self._trim_text(embedding_text, max_embedding_bytes)
                embedding = embedding_cache.get(embedding_text)
                if embedding is None:
                    embedding = self._embed_text(embedding_model, embedding_text)
                    embedding_cache[embedding_text] = embedding

                embeddings.append(embedding)
                token_sets.append(self._extract_tokens(embedding_text))
                node_index = len(items) + 1
                items.append(
                    ItemNode(
                        index=node_index,
                        id=item.id,
                        kind=kind,
                        layer=item.layer.strip(),
                        sector=item.sector.strip() if item.sector else None,
                        subsector=item.subsector.strip() if item.subsector else None,
                        source=source,
                        preview=self._preview_text(embedding_text, preview_chars),
                    )
                )
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not items:
            raise ToolError("No valid items to process.")

        item_connections = self._build_item_connections(
            embeddings,
            token_sets,
            min_similarity,
            max_item_neighbors,
            max_shared_tokens,
        )

        layer_nodes, layer_embeddings = self._build_layer_nodes(items, embeddings)
        sector_nodes, sector_embeddings = self._build_sector_nodes(items, embeddings)
        subsector_nodes, subsector_embeddings = self._build_subsector_nodes(items, embeddings)

        layer_connections = self._build_group_connections(
            layer_embeddings,
            min_group_similarity,
            max_group_neighbors,
            level="layer",
        )
        sector_connections = self._build_group_connections(
            sector_embeddings,
            min_group_similarity,
            max_group_neighbors,
            level="sector",
        )
        subsector_connections = self._build_group_connections(
            subsector_embeddings,
            min_group_similarity,
            max_group_neighbors,
            level="subsector",
        )

        return ContextMultimodalLayersResult(
            items=items,
            item_connections=item_connections,
            layers=layer_nodes,
            sectors=sector_nodes,
            subsectors=subsector_nodes,
            layer_connections=layer_connections,
            sector_connections=sector_connections,
            subsector_connections=subsector_connections,
            item_count=len(items),
            connection_count=len(item_connections),
            truncated=truncated,
            errors=errors,
        )

    def _resolve_kind(self, item: MultimodalItem) -> str:
        kind = (item.kind or "auto").strip().lower()
        if kind not in {
            "auto",
            "text",
            "code",
            "image",
            "audio",
            "video",
            "table",
            "document",
        }:
            return "text"

        if kind != "auto":
            if kind == "document":
                return "text"
            return kind

        if item.path:
            ext = Path(item.path).suffix.lower()
            if ext in set(self.config.default_code_extensions):
                return "code"
        return "text"

    def _load_item_content(
        self, item: MultimodalItem, max_source_bytes: int
    ) -> tuple[str | None, str | None, int | None]:
        if item.content and item.path:
            raise ToolError("Provide content or path, not both.")

        if item.path:
            path = Path(item.path).expanduser()
            if not path.is_absolute():
                path = self.config.effective_workdir / path
            path = path.resolve()
            if not path.exists():
                raise ToolError(f"Path not found: {path}")
            size = path.stat().st_size
            if size > max_source_bytes:
                raise ToolError(
                    f"{path} exceeds max_source_bytes ({size} > {max_source_bytes})."
                )
            return path.read_text("utf-8", errors="ignore"), str(path), size

        if item.content:
            size = len(item.content.encode("utf-8"))
            return item.content, None, size

        return None, None, None

    def _build_embedding_text(self, item: MultimodalItem, content: str | None) -> str:
        parts: list[str] = []
        if item.caption:
            parts.append(item.caption)
        if content:
            parts.append(content)
        elif item.content:
            parts.append(item.content)
        return "\n\n".join(part for part in parts if part)

    def _trim_text(self, text: str, max_bytes: int) -> str:
        data = text.encode("utf-8")
        if len(data) <= max_bytes:
            return text
        head_bytes = max_bytes // 2
        tail_bytes = max_bytes - head_bytes
        head = data[:head_bytes].decode("utf-8", errors="ignore")
        tail = data[-tail_bytes:].decode("utf-8", errors="ignore")
        return f"{head}\n...\n{tail}"

    def _preview_text(self, text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def _embed_text(self, model: str, text: str) -> list[float]:
        payload = json.dumps({"model": model, "prompt": text}).encode("utf-8")
        url = self.config.ollama_url.rstrip("/") + "/api/embeddings"
        req = request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise ToolError(f"Ollama/GPT-OSS embeddings failed: {exc}") from exc

        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            raise ToolError("Invalid embeddings response from Ollama/GPT-OSS.")

        return self._normalize_embedding([float(x) for x in embedding])

    def _normalize_embedding(self, embedding: list[float]) -> list[float]:
        norm = sum(x * x for x in embedding) ** 0.5
        if norm == 0:
            return embedding
        return [x / norm for x in embedding]

    def _extract_tokens(self, content: str) -> set[str]:
        min_len = self.config.min_token_length
        tokens: dict[str, int] = {}
        for match in TOKEN_RE.findall(content.lower()):
            if len(match) < min_len:
                continue
            if match.isdigit():
                continue
            if match in STOPWORDS:
                continue
            tokens[match] = tokens.get(match, 0) + 1

        if not tokens:
            return set()

        sorted_tokens = sorted(tokens.items(), key=lambda item: (-item[1], item[0]))
        max_tokens = self.config.max_tokens_per_item
        if max_tokens > 0:
            sorted_tokens = sorted_tokens[:max_tokens]
        return {token for token, _ in sorted_tokens}

    def _build_item_connections(
        self,
        embeddings: list[list[float]],
        token_sets: list[set[str]],
        min_similarity: float,
        max_neighbors: int,
        max_shared_tokens: int,
    ) -> list[ItemConnection]:
        count = len(embeddings)
        if count < 2:
            return []
        neighbors: list[list[tuple[int, float]]] = [[] for _ in range(count)]
        for i in range(count):
            for j in range(i + 1, count):
                sim = self._dot(embeddings[i], embeddings[j])
                if sim < min_similarity:
                    continue
                neighbors[i].append((j, sim))
                neighbors[j].append((i, sim))

        for idx in range(count):
            neighbors[idx].sort(key=lambda item: (-item[1], item[0]))
            if max_neighbors > 0:
                neighbors[idx] = neighbors[idx][:max_neighbors]

        connections: list[ItemConnection] = []
        for i, items in enumerate(neighbors):
            for j, score in items:
                if i >= j:
                    continue
                shared = sorted(token_sets[i] & token_sets[j])
                if max_shared_tokens > 0 and len(shared) > max_shared_tokens:
                    shared = shared[:max_shared_tokens]
                connections.append(
                    ItemConnection(
                        source_index=i + 1,
                        target_index=j + 1,
                        score=round(score, 6),
                        shared_tokens=shared,
                    )
                )
        return connections

    def _dot(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ToolError("Embedding dimensions do not match.")
        return sum(a * b for a, b in zip(left, right))

    def _average_embeddings(self, vectors: list[list[float]]) -> list[float]:
        if not vectors:
            return []
        dim = len(vectors[0])
        totals = [0.0] * dim
        for vec in vectors:
            if len(vec) != dim:
                raise ToolError("Embedding dimensions do not match.")
            for idx, value in enumerate(vec):
                totals[idx] += value
        return self._normalize_embedding(totals)

    def _build_layer_nodes(
        self, items: list[ItemNode], embeddings: list[list[float]]
    ) -> tuple[list[LayerNode], dict[str, list[float]]]:
        groups: dict[str, list[int]] = {}
        for idx, item in enumerate(items):
            groups.setdefault(item.layer, []).append(idx)

        nodes: list[LayerNode] = []
        embeddings_map: dict[str, list[float]] = {}
        for layer, indices in groups.items():
            vectors = [embeddings[idx] for idx in indices]
            embeddings_map[layer] = self._average_embeddings(vectors)
            nodes.append(LayerNode(layer=layer, item_count=len(indices)))
        return nodes, embeddings_map

    def _build_sector_nodes(
        self, items: list[ItemNode], embeddings: list[list[float]]
    ) -> tuple[list[SectorNode], dict[tuple[str, str], list[float]]]:
        groups: dict[tuple[str, str], list[int]] = {}
        for idx, item in enumerate(items):
            sector = item.sector or "unassigned"
            key = (item.layer, sector)
            groups.setdefault(key, []).append(idx)

        nodes: list[SectorNode] = []
        embeddings_map: dict[tuple[str, str], list[float]] = {}
        for (layer, sector), indices in groups.items():
            vectors = [embeddings[idx] for idx in indices]
            embeddings_map[(layer, sector)] = self._average_embeddings(vectors)
            nodes.append(SectorNode(layer=layer, sector=sector, item_count=len(indices)))
        return nodes, embeddings_map

    def _build_subsector_nodes(
        self, items: list[ItemNode], embeddings: list[list[float]]
    ) -> tuple[list[SubsectorNode], dict[tuple[str, str, str], list[float]]]:
        groups: dict[tuple[str, str, str], list[int]] = {}
        for idx, item in enumerate(items):
            sector = item.sector or "unassigned"
            subsector = item.subsector or "unassigned"
            key = (item.layer, sector, subsector)
            groups.setdefault(key, []).append(idx)

        nodes: list[SubsectorNode] = []
        embeddings_map: dict[tuple[str, str, str], list[float]] = {}
        for (layer, sector, subsector), indices in groups.items():
            vectors = [embeddings[idx] for idx in indices]
            embeddings_map[(layer, sector, subsector)] = self._average_embeddings(vectors)
            nodes.append(
                SubsectorNode(
                    layer=layer,
                    sector=sector,
                    subsector=subsector,
                    item_count=len(indices),
                )
            )
        return nodes, embeddings_map

    def _build_group_connections(
        self,
        embeddings_map: dict,
        min_similarity: float,
        max_neighbors: int,
        level: str,
    ):
        keys = list(embeddings_map.keys())
        count = len(keys)
        if count < 2:
            return []

        neighbors: list[list[tuple[int, float]]] = [[] for _ in range(count)]
        for i in range(count):
            for j in range(i + 1, count):
                sim = self._dot(embeddings_map[keys[i]], embeddings_map[keys[j]])
                if sim < min_similarity:
                    continue
                neighbors[i].append((j, sim))
                neighbors[j].append((i, sim))

        for idx in range(count):
            neighbors[idx].sort(key=lambda item: (-item[1], item[0]))
            if max_neighbors > 0:
                neighbors[idx] = neighbors[idx][:max_neighbors]

        if level == "layer":
            return self._connections_to_layers(keys, neighbors)
        if level == "sector":
            return self._connections_to_sectors(keys, neighbors)
        return self._connections_to_subsectors(keys, neighbors)

    def _connections_to_layers(
        self, keys: list[str], neighbors: list[list[tuple[int, float]]]
    ) -> list[LayerConnection]:
        connections: list[LayerConnection] = []
        for i, items in enumerate(neighbors):
            for j, score in items:
                if i >= j:
                    continue
                connections.append(
                    LayerConnection(
                        source_layer=str(keys[i]),
                        target_layer=str(keys[j]),
                        score=round(score, 6),
                    )
                )
        return connections

    def _connections_to_sectors(
        self,
        keys: list[tuple[str, str]],
        neighbors: list[list[tuple[int, float]]],
    ) -> list[SectorConnection]:
        connections: list[SectorConnection] = []
        for i, items in enumerate(neighbors):
            for j, score in items:
                if i >= j:
                    continue
                source_layer, source_sector = keys[i]
                target_layer, target_sector = keys[j]
                connections.append(
                    SectorConnection(
                        source_layer=source_layer,
                        source_sector=source_sector,
                        target_layer=target_layer,
                        target_sector=target_sector,
                        score=round(score, 6),
                    )
                )
        return connections

    def _connections_to_subsectors(
        self,
        keys: list[tuple[str, str, str]],
        neighbors: list[list[tuple[int, float]]],
    ) -> list[SubsectorConnection]:
        connections: list[SubsectorConnection] = []
        for i, items in enumerate(neighbors):
            for j, score in items:
                if i >= j:
                    continue
                source_layer, source_sector, source_subsector = keys[i]
                target_layer, target_sector, target_subsector = keys[j]
                connections.append(
                    SubsectorConnection(
                        source_layer=source_layer,
                        source_sector=source_sector,
                        source_subsector=source_subsector,
                        target_layer=target_layer,
                        target_sector=target_sector,
                        target_subsector=target_subsector,
                        score=round(score, 6),
                    )
                )
        return connections

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextMultimodalLayersArgs):
            return ToolCallDisplay(summary="context_multimodal_layers")

        summary = f"context_multimodal_layers: {len(event.args.items)} item(s)"
        return ToolCallDisplay(
            summary=summary,
            details={
                "item_count": len(event.args.items),
                "embedding_model": event.args.embedding_model,
                "min_similarity": event.args.min_similarity,
                "max_item_neighbors": event.args.max_item_neighbors,
                "min_group_similarity": event.args.min_group_similarity,
                "max_group_neighbors": event.args.max_group_neighbors,
                "max_shared_tokens": event.args.max_shared_tokens,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextMultimodalLayersResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = (
            f"Processed {event.result.item_count} item(s) with "
            f"{event.result.connection_count} connection(s)"
        )
        warnings = event.result.errors[:]
        if event.result.truncated:
            warnings.append("Output truncated by size or limits")

        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=warnings,
            details={
                "item_count": event.result.item_count,
                "connection_count": event.result.connection_count,
                "layer_count": len(event.result.layers),
                "sector_count": len(event.result.sectors),
                "subsector_count": len(event.result.subsectors),
                "truncated": event.result.truncated,
                "errors": event.result.errors,
                "items": event.result.items,
                "item_connections": event.result.item_connections,
                "layers": event.result.layers,
                "sectors": event.result.sectors,
                "subsectors": event.result.subsectors,
                "layer_connections": event.result.layer_connections,
                "sector_connections": event.result.sector_connections,
                "subsector_connections": event.result.subsector_connections,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Building multimodal layer connections"
