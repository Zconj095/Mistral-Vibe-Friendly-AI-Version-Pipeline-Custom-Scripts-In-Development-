from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from array import array
from pathlib import Path
import json
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

TIME_UNIT_NS = {
    "ns": 1,
    "nanosecond": 1,
    "nanoseconds": 1,
    "us": 1_000,
    "microsecond": 1_000,
    "microseconds": 1_000,
    "ms": 1_000_000,
    "millisecond": 1_000_000,
    "milliseconds": 1_000_000,
    "s": 1_000_000_000,
    "sec": 1_000_000_000,
    "secs": 1_000_000_000,
    "second": 1_000_000_000,
    "seconds": 1_000_000_000,
    "m": 60 * 1_000_000_000,
    "min": 60 * 1_000_000_000,
    "mins": 60 * 1_000_000_000,
    "minute": 60 * 1_000_000_000,
    "minutes": 60 * 1_000_000_000,
    "h": 3600 * 1_000_000_000,
    "hr": 3600 * 1_000_000_000,
    "hour": 3600 * 1_000_000_000,
    "hours": 3600 * 1_000_000_000,
    "d": 86400 * 1_000_000_000,
    "day": 86400 * 1_000_000_000,
    "days": 86400 * 1_000_000_000,
    "w": 7 * 86400 * 1_000_000_000,
    "week": 7 * 86400 * 1_000_000_000,
    "weeks": 7 * 86400 * 1_000_000_000,
    "mo": 30 * 86400 * 1_000_000_000,
    "month": 30 * 86400 * 1_000_000_000,
    "months": 30 * 86400 * 1_000_000_000,
    "y": 365 * 86400 * 1_000_000_000,
    "yr": 365 * 86400 * 1_000_000_000,
    "year": 365 * 86400 * 1_000_000_000,
    "years": 365 * 86400 * 1_000_000_000,
    "decade": 10 * 365 * 86400 * 1_000_000_000,
    "decades": 10 * 365 * 86400 * 1_000_000_000,
}


@dataclass
class _MemoryEntry:
    item_id: str
    timestamp_ns: int
    source: str | None
    embedding: list[float]
    embedding_dim: int
    embedding_model: str
    token_counts: dict[str, int]


class ContextMemoryConsolidationConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    profile_path: Path = Field(
        default=Path.home() / ".vibe" / "memory" / "memory_consolidation.json",
        description="Path to the consolidation profile.",
    )
    ollama_url: str = Field(
        default="http://127.0.0.1:11434",
        description="Base URL for the Ollama/GPT-OSS server.",
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        description="Default embedding model to use with Ollama/GPT-OSS.",
    )
    max_items: int = Field(default=200, description="Maximum items to process.")
    max_profile_entries: int = Field(
        default=5000, description="Maximum entries stored in the profile."
    )
    max_source_bytes: int = Field(
        default=5_000_000, description="Maximum size per item (bytes)."
    )
    max_total_bytes: int = Field(
        default=20_000_000, description="Maximum total bytes across items."
    )
    preview_chars: int = Field(default=200, description="Preview length per item.")
    default_time_unit: str = Field(
        default="s", description="Default time unit for numeric timestamps."
    )
    consolidation_half_life: str = Field(
        default="7d", description="Half-life for consolidation growth."
    )
    base_weight: float = Field(
        default=0.0, description="Base weight added to maturity."
    )
    min_word_length: int = Field(default=3, description="Minimum token length.")
    max_keywords: int = Field(default=20, description="Maximum keywords per summary.")
    max_items_summary: int = Field(
        default=50, description="Maximum items summarized."
    )


class ContextMemoryConsolidationState(BaseToolState):
    pass


class MemoryItem(BaseModel):
    id: str | None = Field(default=None, description="Optional item id.")
    content: str | None = Field(default=None, description="Inline text content.")
    path: str | None = Field(default=None, description="Path to a text file.")
    embedding: list[float] | None = Field(
        default=None, description="Optional embedding vector."
    )
    embedding_model: str | None = Field(
        default=None, description="Embedding model used for the embedding."
    )
    timestamp: float | int | str = Field(description="Timestamp value or ISO string.")
    time_unit: str | None = Field(default=None, description="Unit for numeric timestamps.")
    source: str | None = Field(default=None, description="Source description.")


class ContextMemoryConsolidationArgs(BaseModel):
    action: str | None = Field(
        default="consolidate", description="consolidate or update."
    )
    items: list[MemoryItem] | None = Field(
        default=None, description="Items to store in the profile."
    )
    profile_path: str | None = Field(
        default=None, description="Override profile path."
    )
    now: float | int | str | None = Field(
        default=None, description="Override the current time for consolidation."
    )
    include_embeddings: bool | None = Field(
        default=None, description="Include consolidated embeddings in output."
    )
    consolidation_half_life: str | float | int | None = Field(
        default=None, description="Override consolidation half-life."
    )
    base_weight: float | None = Field(
        default=None, description="Override base weight."
    )
    default_time_unit: str | None = Field(
        default=None, description="Override default time unit."
    )
    max_items: int | None = Field(default=None, description="Override max_items.")
    max_profile_entries: int | None = Field(
        default=None, description="Override max_profile_entries."
    )
    max_source_bytes: int | None = Field(
        default=None, description="Override max_source_bytes."
    )
    max_total_bytes: int | None = Field(
        default=None, description="Override max_total_bytes."
    )
    preview_chars: int | None = Field(default=None, description="Override preview_chars.")
    min_word_length: int | None = Field(
        default=None, description="Override min_word_length."
    )
    max_keywords: int | None = Field(
        default=None, description="Override max_keywords."
    )
    max_items_summary: int | None = Field(
        default=None, description="Override max_items_summary."
    )


class ConsolidatedEmbedding(BaseModel):
    embedding_dim: int
    item_count: int
    average_maturity: float
    embedding: list[float] | None


class MemorySummary(BaseModel):
    total_items: int
    earliest_ns: int
    latest_ns: int
    average_maturity: float
    keywords: list[str]


class MemoryItemSummary(BaseModel):
    item_id: str
    timestamp_ns: int
    age_ns: int
    maturity: float
    source: str | None
    preview: str
    keywords: list[str]


class ContextMemoryConsolidationResult(BaseModel):
    summary: MemorySummary
    items: list[MemoryItemSummary]
    consolidated_embeddings: list[ConsolidatedEmbedding]
    profile_path: str
    updated_items: int
    truncated: bool
    errors: list[str]


class ContextMemoryConsolidation(
    BaseTool[
        ContextMemoryConsolidationArgs,
        ContextMemoryConsolidationResult,
        ContextMemoryConsolidationConfig,
        ContextMemoryConsolidationState,
    ],
    ToolUIData[ContextMemoryConsolidationArgs, ContextMemoryConsolidationResult],
):
    description: ClassVar[str] = (
        "Consolidate stored embeddings over time without active usage."
    )

    async def run(
        self, args: ContextMemoryConsolidationArgs
    ) -> ContextMemoryConsolidationResult:
        action = (args.action or "consolidate").strip().lower()
        if action not in {"consolidate", "update"}:
            raise ToolError("action must be consolidate or update.")

        max_items = args.max_items if args.max_items is not None else self.config.max_items
        max_profile_entries = (
            args.max_profile_entries
            if args.max_profile_entries is not None
            else self.config.max_profile_entries
        )
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
        preview_chars = (
            args.preview_chars if args.preview_chars is not None else self.config.preview_chars
        )
        min_word_length = (
            args.min_word_length
            if args.min_word_length is not None
            else self.config.min_word_length
        )
        max_keywords = (
            args.max_keywords if args.max_keywords is not None else self.config.max_keywords
        )
        max_items_summary = (
            args.max_items_summary
            if args.max_items_summary is not None
            else self.config.max_items_summary
        )
        default_time_unit = (
            args.default_time_unit
            if args.default_time_unit is not None
            else self.config.default_time_unit
        )
        half_life_value = (
            args.consolidation_half_life
            if args.consolidation_half_life is not None
            else self.config.consolidation_half_life
        )
        base_weight = args.base_weight if args.base_weight is not None else self.config.base_weight
        include_embeddings = args.include_embeddings if args.include_embeddings is not None else False

        if max_items <= 0:
            raise ToolError("max_items must be a positive integer.")
        if max_profile_entries <= 0:
            raise ToolError("max_profile_entries must be a positive integer.")
        if max_source_bytes <= 0:
            raise ToolError("max_source_bytes must be a positive integer.")
        if max_total_bytes <= 0:
            raise ToolError("max_total_bytes must be a positive integer.")
        if preview_chars < 0:
            raise ToolError("preview_chars must be >= 0.")
        if min_word_length <= 0:
            raise ToolError("min_word_length must be a positive integer.")
        if max_keywords < 0:
            raise ToolError("max_keywords must be >= 0.")
        if max_items_summary < 0:
            raise ToolError("max_items_summary must be >= 0.")
        if base_weight < 0:
            raise ToolError("base_weight must be >= 0.")

        half_life_ns = self._parse_duration(half_life_value, default_time_unit)
        if half_life_ns <= 0:
            raise ToolError("consolidation_half_life must be a positive duration.")

        profile_path = (
            Path(args.profile_path).expanduser()
            if args.profile_path
            else self.config.profile_path
        )

        errors: list[str] = []
        truncated = False
        updated_items = 0

        entries = self._load_profile(profile_path, errors)

        if action == "update":
            if not args.items:
                raise ToolError("items is required for update.")
            if len(args.items) > max_items:
                raise ToolError(f"items exceeds max_items ({len(args.items)} > {max_items}).")

            total_bytes = 0
            used_ids = {entry.item_id for entry in entries}
            for idx, item in enumerate(args.items, start=1):
                try:
                    content, size_bytes = self._load_item_content(item, max_source_bytes)
                    if size_bytes is not None:
                        if total_bytes + size_bytes > max_total_bytes:
                            truncated = True
                            break
                        total_bytes += size_bytes

                    embedding = item.embedding
                    embedding_model = item.embedding_model or self.config.embedding_model
                    if embedding is None:
                        if content is None:
                            raise ToolError("Item must include content or embedding.")
                        embedding = self._embed_text(embedding_model, content)
                    if not embedding:
                        raise ToolError("Embedding is empty.")

                    timestamp_ns = self._parse_timestamp(
                        item.timestamp, item.time_unit or default_time_unit
                    )
                    token_counts = (
                        self._extract_token_counts(content or "", min_word_length)
                        if content is not None
                        else {}
                    )
                    item_id = self._resolve_item_id(item, idx, used_ids)

                    entry = _MemoryEntry(
                        item_id=item_id,
                        timestamp_ns=timestamp_ns,
                        source=item.source,
                        embedding=[float(x) for x in embedding],
                        embedding_dim=len(embedding),
                        embedding_model=embedding_model or "provided",
                        token_counts=token_counts,
                    )
                    entries = [e for e in entries if e.item_id != item_id]
                    entries.append(entry)
                    updated_items += 1
                except ToolError as exc:
                    errors.append(f"item[{idx}]: {exc}")
                except Exception as exc:
                    errors.append(f"item[{idx}]: {exc}")

            entries.sort(key=lambda item: item.timestamp_ns)
            if len(entries) > max_profile_entries:
                truncated = True
                entries = entries[-max_profile_entries:]

            self._save_profile(profile_path, entries, errors)

        if not entries:
            raise ToolError("No memory entries available for consolidation.")

        now_ns = self._resolve_now(args.now, default_time_unit)
        earliest_ns = min(entry.timestamp_ns for entry in entries)
        latest_ns = max(entry.timestamp_ns for entry in entries)

        total_maturity = 0.0
        consolidated: dict[int, dict[str, object]] = {}
        global_counts: dict[str, float] = {}
        items_summary: list[MemoryItemSummary] = []

        for entry in entries:
            age_ns = max(0, now_ns - entry.timestamp_ns)
            maturity = self._maturity(age_ns, half_life_ns)
            total_maturity += maturity
            weight = maturity + base_weight
            if weight > 0:
                group = consolidated.setdefault(
                    entry.embedding_dim,
                    {
                        "sum": [0.0] * entry.embedding_dim,
                        "count": 0,
                        "maturity_sum": 0.0,
                    },
                )
                summed = group["sum"]
                for idx, value in enumerate(entry.embedding):
                    summed[idx] += value * weight
                group["count"] = group["count"] + 1
                group["maturity_sum"] = group["maturity_sum"] + maturity

            for token, count in entry.token_counts.items():
                global_counts[token] = global_counts.get(token, 0.0) + (count * maturity)

            preview = self._preview_text(
                self._keywords_to_text(entry.token_counts, max_keywords),
                preview_chars,
            )
            keywords = self._select_keywords(entry.token_counts, max_keywords)
            if max_items_summary <= 0 or len(items_summary) < max_items_summary:
                items_summary.append(
                    MemoryItemSummary(
                        item_id=entry.item_id,
                        timestamp_ns=entry.timestamp_ns,
                        age_ns=age_ns,
                        maturity=round(maturity, 6),
                        source=entry.source,
                        preview=preview,
                        keywords=keywords,
                    )
                )

        average_maturity = total_maturity / len(entries) if entries else 0.0
        summary_keywords = self._select_keywords_float(global_counts, max_keywords)

        consolidated_embeddings: list[ConsolidatedEmbedding] = []
        for dim, data in consolidated.items():
            count = int(data["count"])
            maturity_sum = float(data["maturity_sum"])
            embedding = self._normalize_embedding(data["sum"])
            consolidated_embeddings.append(
                ConsolidatedEmbedding(
                    embedding_dim=dim,
                    item_count=count,
                    average_maturity=round(maturity_sum / count, 6) if count else 0.0,
                    embedding=embedding if include_embeddings else None,
                )
            )

        consolidated_embeddings.sort(key=lambda item: (-item.item_count, item.embedding_dim))
        items_summary.sort(key=lambda item: item.timestamp_ns, reverse=True)

        summary = MemorySummary(
            total_items=len(entries),
            earliest_ns=earliest_ns,
            latest_ns=latest_ns,
            average_maturity=round(average_maturity, 6),
            keywords=summary_keywords,
        )

        return ContextMemoryConsolidationResult(
            summary=summary,
            items=items_summary,
            consolidated_embeddings=consolidated_embeddings,
            profile_path=str(profile_path),
            updated_items=updated_items,
            truncated=truncated,
            errors=errors,
        )

    def _load_item_content(
        self, item: MemoryItem, max_source_bytes: int
    ) -> tuple[str | None, int | None]:
        if item.content and item.path:
            raise ToolError("Provide content or path, not both.")
        if item.path:
            path = Path(item.path).expanduser()
            if not path.is_absolute():
                path = self.config.effective_workdir / path
            path = path.resolve()
            if not path.exists():
                raise ToolError(f"Path not found: {path}")
            if path.is_dir():
                raise ToolError(f"Path is a directory, not a file: {path}")
            size = path.stat().st_size
            if size > max_source_bytes:
                raise ToolError(
                    f"{path} exceeds max_source_bytes ({size} > {max_source_bytes})."
                )
            return path.read_text("utf-8", errors="ignore"), size
        if item.content is not None:
            size = len(item.content.encode("utf-8"))
            if size > max_source_bytes:
                raise ToolError(
                    f"content exceeds max_source_bytes ({size} > {max_source_bytes})."
                )
            return item.content, size
        return None, None

    def _resolve_item_id(
        self, item: MemoryItem, index: int, used_ids: set[str]
    ) -> str:
        raw = item.id or item.source or item.path or f"item{index}"
        normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(raw).strip())
        if not normalized:
            normalized = f"item{index}"
        candidate = normalized
        counter = 2
        while candidate in used_ids:
            candidate = f"{normalized}_{counter}"
            counter += 1
        used_ids.add(candidate)
        return candidate

    def _extract_token_counts(self, text: str, min_word_length: int) -> dict[str, int]:
        counts: dict[str, int] = {}
        for token in TOKEN_RE.findall(text):
            lowered = token.lower()
            if len(lowered) < min_word_length:
                continue
            if lowered in STOPWORDS:
                continue
            counts[lowered] = counts.get(lowered, 0) + 1
        return counts

    def _select_keywords(self, counts: dict[str, int], limit: int) -> list[str]:
        if limit <= 0:
            return []
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [token for token, _ in ordered[:limit]]

    def _select_keywords_float(self, counts: dict[str, float], limit: int) -> list[str]:
        if limit <= 0:
            return []
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [token for token, _ in ordered[:limit]]

    def _keywords_to_text(self, counts: dict[str, int], max_keywords: int) -> str:
        keywords = self._select_keywords(counts, max_keywords)
        return " ".join(keywords)

    def _preview_text(self, text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def _maturity(self, age_ns: int, half_life_ns: int) -> float:
        if half_life_ns <= 0:
            return 0.0
        if age_ns <= 0:
            return 0.0
        return 1.0 - (0.5 ** (age_ns / half_life_ns))

    def _resolve_now(self, now: float | int | str | None, default_unit: str) -> int:
        if now is None:
            return int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)
        return self._parse_timestamp(now, default_unit)

    def _parse_duration(self, value: str | float | int, default_unit: str) -> int:
        if isinstance(value, (int, float)):
            unit_ns = self._resolve_time_unit(default_unit)
            return int(round(float(value) * unit_ns))
        if not isinstance(value, str):
            raise ToolError("Duration value must be a string or number.")
        text = value.strip()
        if not text:
            raise ToolError("Duration value cannot be empty.")
        if self._is_numeric(text):
            unit_ns = self._resolve_time_unit(default_unit)
            return int(round(float(text) * unit_ns))
        match = re.match(r"^\s*([-+]?[0-9]*\.?[0-9]+)\s*([A-Za-z]+)\s*$", text)
        if not match:
            raise ToolError(f"Invalid duration format: {value}")
        amount = float(match.group(1))
        unit = match.group(2).lower()
        unit_ns = self._resolve_time_unit(unit)
        return int(round(amount * unit_ns))

    def _parse_timestamp(self, value: float | int | str, time_unit: str) -> int:
        if isinstance(value, (int, float)):
            unit_ns = self._resolve_time_unit(time_unit)
            return int(round(float(value) * unit_ns))
        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise ToolError("timestamp cannot be empty.")
            if self._is_numeric(text):
                unit_ns = self._resolve_time_unit(time_unit)
                return int(round(float(text) * unit_ns))
            try:
                if text.endswith("Z"):
                    text = text[:-1] + "+00:00"
                dt = datetime.fromisoformat(text)
            except ValueError as exc:
                raise ToolError(f"Invalid timestamp string: {value}") from exc
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1_000_000_000)
        raise ToolError("timestamp must be a number or string.")

    def _resolve_time_unit(self, unit: str | None) -> int:
        key = (unit or "").strip().lower()
        if not key:
            raise ToolError("time_unit cannot be empty.")
        if key not in TIME_UNIT_NS:
            raise ToolError(f"Unsupported time unit: {unit}")
        return TIME_UNIT_NS[key]

    def _is_numeric(self, value: str) -> bool:
        return bool(re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value))

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
        norm = sum(value * value for value in embedding) ** 0.5
        if norm == 0:
            return embedding
        return [value / norm for value in embedding]

    def _load_profile(self, path: Path, errors: list[str]) -> list[_MemoryEntry]:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"Failed to read profile: {exc}")
            return []
        entries_raw = data.get("entries", []) if isinstance(data, dict) else data
        entries: list[_MemoryEntry] = []
        if isinstance(entries_raw, list):
            for entry in entries_raw:
                try:
                    embedding = [float(x) for x in entry.get("embedding", [])]
                    entries.append(
                        _MemoryEntry(
                            item_id=str(entry.get("item_id")),
                            timestamp_ns=int(entry.get("timestamp_ns", 0)),
                            source=entry.get("source"),
                            embedding=embedding,
                            embedding_dim=int(entry.get("embedding_dim", len(embedding))),
                            embedding_model=str(entry.get("embedding_model") or "unknown"),
                            token_counts={
                                str(k): int(v)
                                for k, v in (entry.get("token_counts") or {}).items()
                            },
                        )
                    )
                except Exception as exc:
                    errors.append(f"Invalid profile entry: {exc}")
        return entries

    def _save_profile(
        self, path: Path, entries: list[_MemoryEntry], errors: list[str]
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "entries": [
                {
                    "item_id": entry.item_id,
                    "timestamp_ns": entry.timestamp_ns,
                    "source": entry.source,
                    "embedding": entry.embedding,
                    "embedding_dim": entry.embedding_dim,
                    "embedding_model": entry.embedding_model,
                    "token_counts": entry.token_counts,
                }
                for entry in entries
            ],
        }
        try:
            path.write_text(json.dumps(payload, indent=2), "utf-8")
        except OSError as exc:
            errors.append(f"Failed to write profile: {exc}")

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextMemoryConsolidationArgs):
            return ToolCallDisplay(summary="context_memory_consolidation")

        item_count = len(event.args.items or [])
        summary = f"context_memory_consolidation: {item_count} item(s)"
        return ToolCallDisplay(
            summary=summary,
            details={
                "action": event.args.action,
                "item_count": item_count,
                "profile_path": event.args.profile_path,
                "include_embeddings": event.args.include_embeddings,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextMemoryConsolidationResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = (
            f"Consolidated {event.result.summary.total_items} item(s)"
        )
        warnings = event.result.errors[:]
        if event.result.truncated:
            warnings.append("Output truncated by limits")

        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=warnings,
            details={
                "summary": event.result.summary,
                "items": event.result.items,
                "consolidated_embeddings": event.result.consolidated_embeddings,
                "profile_path": event.result.profile_path,
                "updated_items": event.result.updated_items,
                "truncated": event.result.truncated,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Consolidating memory embeddings"
