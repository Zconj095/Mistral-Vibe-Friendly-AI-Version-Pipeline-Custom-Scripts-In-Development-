from __future__ import annotations

import hashlib
import json
import sqlite3
from array import array
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
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


DEFAULT_DB_PATH = Path.home() / ".vibe" / "cache" / "context_cache.sqlite"
VALID_ACTIONS = {
    "static_hash",
    "static_store",
    "static_get",
    "static_list",
    "static_delete",
    "semantic_store",
    "semantic_search",
    "semantic_list",
    "semantic_delete",
    "stats",
}


@dataclass(frozen=True)
class _ContentItem:
    content: str
    metadata: dict | None


class ContextCacheConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    db_path: Path = Field(
        default=DEFAULT_DB_PATH,
        description="Path to the cache database file.",
    )
    default_scope: str = Field(
        default="default",
        description="Default cache scope/namespace.",
    )
    ollama_url: str = Field(
        default="http://127.0.0.1:11434",
        description="Base URL for the local Ollama server.",
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        description="Embedding model name for semantic cache.",
    )
    max_source_bytes: int = Field(
        default=5_000_000,
        description="Maximum bytes per input item.",
    )
    max_total_bytes: int = Field(
        default=20_000_000,
        description="Maximum bytes across all inputs.",
    )
    max_results: int = Field(
        default=5, description="Default max results to return."
    )
    max_content_chars: int = Field(
        default=4000, description="Maximum cached content chars to return."
    )
    max_response_chars: int = Field(
        default=2000, description="Maximum cached response chars to return."
    )
    normalize: bool = Field(
        default=True,
        description="Normalize content before hashing.",
    )


class ContextCacheState(BaseToolState):
    pass


class ContextCacheArgs(BaseModel):
    action: str | None = Field(
        default="static_hash",
        description="Cache action.",
    )
    db_path: str | None = Field(
        default=None, description="Override cache database path."
    )
    scope: str | None = Field(
        default=None, description="Cache scope/namespace."
    )
    all_scopes: bool = Field(
        default=False, description="Ignore scope filter."
    )
    key: str | None = Field(default=None, description="Single cache key.")
    keys: list[str] | None = Field(
        default=None, description="Multiple cache keys."
    )
    id: int | None = Field(
        default=None, description="Semantic cache entry id."
    )
    ids: list[int] | None = Field(
        default=None, description="Semantic cache entry ids."
    )
    content: str | None = Field(
        default=None, description="Inline content."
    )
    contents: list[str] | None = Field(
        default=None, description="Inline contents."
    )
    path: str | None = Field(
        default=None, description="Path to a file."
    )
    paths: list[str] | None = Field(
        default=None, description="Paths to files."
    )
    normalize: bool | None = Field(
        default=None, description="Normalize content before hashing."
    )
    store_content: bool = Field(
        default=True, description="Store content in the cache."
    )
    include_content: bool = Field(
        default=False, description="Include cached content in outputs."
    )
    include_metadata: bool = Field(
        default=True, description="Include metadata in outputs."
    )
    metadata: dict | None = Field(
        default=None, description="Metadata to store."
    )
    limit: int | None = Field(
        default=None, description="Limit list results (0 means no limit)."
    )
    max_source_bytes: int | None = Field(
        default=None, description="Override max bytes per input."
    )
    max_total_bytes: int | None = Field(
        default=None, description="Override max bytes across inputs."
    )
    max_content_chars: int | None = Field(
        default=None, description="Override max content chars in output."
    )
    query: str | None = Field(
        default=None, description="Semantic cache query."
    )
    response: str | None = Field(
        default=None, description="Semantic cache response."
    )
    context_key: str | None = Field(
        default=None,
        description="Optional context key for semantic cache entries.",
    )
    embedding_model: str | None = Field(
        default=None, description="Override embedding model."
    )
    top_k: int | None = Field(
        default=None, description="Number of semantic results to return."
    )
    min_score: float | None = Field(
        default=None, description="Minimum semantic similarity score."
    )
    max_response_chars: int | None = Field(
        default=None, description="Override max response chars in output."
    )
    include_response: bool = Field(
        default=True, description="Include cached responses in output."
    )
    max_age_days: float | None = Field(
        default=None, description="Filter results newer than this many days."
    )


class StaticCacheEntry(BaseModel):
    key: str
    scope: str
    content: str | None
    content_bytes: int | None
    char_count: int | None
    line_count: int | None
    token_estimate: int | None
    metadata: dict | None
    created_at: str | None
    updated_at: str | None


class SemanticCacheEntry(BaseModel):
    id: int
    scope: str
    query: str
    response: str | None
    score: float | None
    embedding_model: str
    context_key: str | None
    metadata: dict | None
    created_at: str | None
    updated_at: str | None


class ContextCacheResult(BaseModel):
    action: str
    db_path: str
    hashes: list[str] = Field(default_factory=list)
    static_entries: list[StaticCacheEntry] = Field(default_factory=list)
    semantic_entries: list[SemanticCacheEntry] = Field(default_factory=list)
    deleted: int = 0
    stats: dict[str, int] | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ContextCache(
    BaseTool[
        ContextCacheArgs,
        ContextCacheResult,
        ContextCacheConfig,
        ContextCacheState,
    ],
    ToolUIData[ContextCacheArgs, ContextCacheResult],
):
    description: ClassVar[str] = (
        "Cache static context hashes and semantic query responses locally."
    )

    async def run(self, args: ContextCacheArgs) -> ContextCacheResult:
        action = (args.action or "static_hash").strip().lower()
        if action not in VALID_ACTIONS:
            raise ToolError(
                f"action must be one of: {', '.join(sorted(VALID_ACTIONS))}"
            )

        db_path = Path(args.db_path).expanduser() if args.db_path else self.config.db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)

        warnings: list[str] = []
        errors: list[str] = []

        with sqlite3.connect(str(db_path)) as conn:
            self._init_db(conn)

            if action == "static_hash":
                entries = self._build_static_entries(args, warnings)
                return ContextCacheResult(
                    action=action,
                    db_path=str(db_path),
                    hashes=[entry.key for entry in entries],
                    static_entries=entries,
                    warnings=warnings,
                    errors=errors,
                )

            if action == "static_store":
                entries = self._store_static_entries(conn, args, warnings)
                return ContextCacheResult(
                    action=action,
                    db_path=str(db_path),
                    hashes=[entry.key for entry in entries],
                    static_entries=entries,
                    warnings=warnings,
                    errors=errors,
                )

            if action == "static_get":
                entries = self._get_static_entries(conn, args)
                return ContextCacheResult(
                    action=action,
                    db_path=str(db_path),
                    static_entries=entries,
                    warnings=warnings,
                    errors=errors,
                )

            if action == "static_list":
                entries = self._list_static_entries(conn, args)
                return ContextCacheResult(
                    action=action,
                    db_path=str(db_path),
                    static_entries=entries,
                    warnings=warnings,
                    errors=errors,
                )

            if action == "static_delete":
                deleted = self._delete_static_entries(conn, args)
                return ContextCacheResult(
                    action=action,
                    db_path=str(db_path),
                    deleted=deleted,
                    warnings=warnings,
                    errors=errors,
                )

            if action == "semantic_store":
                entry = self._store_semantic_entry(conn, args)
                return ContextCacheResult(
                    action=action,
                    db_path=str(db_path),
                    semantic_entries=[entry],
                    warnings=warnings,
                    errors=errors,
                )

            if action == "semantic_search":
                entries = self._search_semantic_entries(conn, args)
                return ContextCacheResult(
                    action=action,
                    db_path=str(db_path),
                    semantic_entries=entries,
                    warnings=warnings,
                    errors=errors,
                )

            if action == "semantic_list":
                entries = self._list_semantic_entries(conn, args)
                return ContextCacheResult(
                    action=action,
                    db_path=str(db_path),
                    semantic_entries=entries,
                    warnings=warnings,
                    errors=errors,
                )

            if action == "semantic_delete":
                deleted = self._delete_semantic_entries(conn, args)
                return ContextCacheResult(
                    action=action,
                    db_path=str(db_path),
                    deleted=deleted,
                    warnings=warnings,
                    errors=errors,
                )

            stats = self._cache_stats(conn, args)
            return ContextCacheResult(
                action=action,
                db_path=str(db_path),
                stats=stats,
                warnings=warnings,
                errors=errors,
            )

    def _resolve_scope(self, args: ContextCacheArgs) -> str | None:
        if args.all_scopes:
            return None
        scope = (args.scope or self.config.default_scope or "default").strip()
        if not scope:
            raise ToolError("scope cannot be empty.")
        return scope

    def _resolve_limits(self, args: ContextCacheArgs) -> tuple[int | None, int]:
        limit = args.limit if args.limit is not None else self.config.max_results
        if limit is not None and limit < 0:
            raise ToolError("limit must be >= 0.")
        max_chars = args.max_content_chars or self.config.max_content_chars
        if max_chars <= 0:
            raise ToolError("max_content_chars must be positive.")
        return limit, max_chars

    def _resolve_response_limit(self, args: ContextCacheArgs) -> int:
        max_chars = args.max_response_chars or self.config.max_response_chars
        if max_chars <= 0:
            raise ToolError("max_response_chars must be positive.")
        return max_chars

    def _normalize_text(self, text: str, normalize: bool) -> str:
        if not normalize:
            return text
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.rstrip() for line in normalized.split("\n")]
        return "\n".join(lines)

    def _build_static_entries(
        self, args: ContextCacheArgs, warnings: list[str]
    ) -> list[StaticCacheEntry]:
        items = self._collect_inputs(args, warnings)
        scope = self._resolve_scope(args) or "default"
        normalize = (
            args.normalize if args.normalize is not None else self.config.normalize
        )

        entries: list[StaticCacheEntry] = []
        for item in items:
            content = self._normalize_text(item.content, normalize)
            key = self._hash_content(content)
            stats = self._content_stats(content)
            entries.append(
                StaticCacheEntry(
                    key=key,
                    scope=scope,
                    content=content if args.include_content else None,
                    content_bytes=stats["content_bytes"],
                    char_count=stats["char_count"],
                    line_count=stats["line_count"],
                    token_estimate=stats["token_estimate"],
                    metadata=item.metadata if args.include_metadata else None,
                    created_at=None,
                    updated_at=None,
                )
            )
        return entries

    def _store_static_entries(
        self, conn: sqlite3.Connection, args: ContextCacheArgs, warnings: list[str]
    ) -> list[StaticCacheEntry]:
        items = self._collect_inputs(args, warnings)
        scope = self._resolve_scope(args) or "default"
        normalize = (
            args.normalize if args.normalize is not None else self.config.normalize
        )
        include_content = bool(args.store_content)
        include_metadata = args.include_metadata
        entries: list[StaticCacheEntry] = []
        now = datetime.utcnow().isoformat()

        for item in items:
            content = self._normalize_text(item.content, normalize)
            key = self._hash_content(content)
            stats = self._content_stats(content)
            metadata_json = self._encode_metadata(item.metadata)
            stored_content = content if include_content else None
            self._upsert_static_entry(
                conn,
                scope,
                key,
                stored_content,
                stats,
                metadata_json,
                now,
                update=True,
            )
            entry = self._fetch_static_entry(
                conn,
                scope,
                key,
                args.include_content,
                include_metadata,
                args.max_content_chars or self.config.max_content_chars,
            )
            if entry is not None:
                entries.append(entry)
        return entries

    def _get_static_entries(
        self, conn: sqlite3.Connection, args: ContextCacheArgs
    ) -> list[StaticCacheEntry]:
        keys = self._resolve_keys(args)
        if not keys:
            raise ToolError("key or keys required for static_get.")
        scope = self._resolve_scope(args)
        include_metadata = args.include_metadata
        max_chars = args.max_content_chars or self.config.max_content_chars

        entries: list[StaticCacheEntry] = []
        for key in keys:
            entries.extend(
                self._fetch_static_entries_by_key(
                    conn,
                    scope,
                    key,
                    args.include_content,
                    include_metadata,
                    max_chars,
                )
            )
        return entries

    def _list_static_entries(
        self, conn: sqlite3.Connection, args: ContextCacheArgs
    ) -> list[StaticCacheEntry]:
        scope = self._resolve_scope(args)
        limit, max_chars = self._resolve_limits(args)
        include_metadata = args.include_metadata

        params: list[Any] = []
        query = (
            "SELECT scope, key, content, content_bytes, char_count, line_count, "
            "token_estimate, metadata, created_at, updated_at FROM static_entries"
        )
        if scope:
            query += " WHERE scope = ?"
            params.append(scope)
        query += " ORDER BY updated_at DESC"
        if limit and limit > 0:
            query += " LIMIT ?"
            params.append(limit)

        entries: list[StaticCacheEntry] = []
        for row in conn.execute(query, params):
            entries.append(
                self._row_to_static_entry(
                    row,
                    args.include_content,
                    include_metadata,
                    max_chars,
                )
            )
        return entries

    def _delete_static_entries(
        self, conn: sqlite3.Connection, args: ContextCacheArgs
    ) -> int:
        keys = self._resolve_keys(args)
        scope = self._resolve_scope(args)

        if not keys and scope is None:
            raise ToolError("Provide key(s) or scope (or set all_scopes).")

        deleted = 0
        if keys:
            for key in keys:
                if scope:
                    cursor = conn.execute(
                        "DELETE FROM static_entries WHERE scope = ? AND key = ?",
                        (scope, key),
                    )
                else:
                    cursor = conn.execute(
                        "DELETE FROM static_entries WHERE key = ?", (key,)
                    )
                deleted += cursor.rowcount
            return deleted

        if scope:
            cursor = conn.execute(
                "DELETE FROM static_entries WHERE scope = ?", (scope,)
            )
            return cursor.rowcount

        cursor = conn.execute("DELETE FROM static_entries")
        return cursor.rowcount

    def _store_semantic_entry(
        self, conn: sqlite3.Connection, args: ContextCacheArgs
    ) -> SemanticCacheEntry:
        if not args.query or not args.query.strip():
            raise ToolError("query is required for semantic_store.")
        if args.response is None:
            raise ToolError("response is required for semantic_store.")

        scope = self._resolve_scope(args) or "default"
        model = args.embedding_model or self.config.embedding_model
        query = args.query.strip()
        response = args.response
        context_key = args.context_key
        metadata_json = self._encode_metadata(args.metadata)
        now = datetime.utcnow().isoformat()

        embedding = self._embed_text(model, query)
        entry_hash = self._semantic_hash(scope, model, query, response, context_key)
        embedding_blob = self._pack_embedding(embedding)

        existing_id = self._find_semantic_entry(conn, entry_hash)
        if existing_id is None:
            cursor = conn.execute(
                """
                INSERT INTO semantic_entries (
                    scope,
                    entry_hash,
                    query,
                    response,
                    context_key,
                    embedding_model,
                    embedding,
                    embedding_dim,
                    metadata,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scope,
                    entry_hash,
                    query,
                    response,
                    context_key,
                    model,
                    sqlite3.Binary(embedding_blob),
                    len(embedding),
                    metadata_json,
                    now,
                    now,
                ),
            )
            entry_id = cursor.lastrowid
        else:
            conn.execute(
                """
                UPDATE semantic_entries
                SET response = ?, metadata = ?, embedding = ?, embedding_dim = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    response,
                    metadata_json,
                    sqlite3.Binary(embedding_blob),
                    len(embedding),
                    now,
                    existing_id,
                ),
            )
            entry_id = existing_id

        entry = self._fetch_semantic_entry(
            conn,
            entry_id,
            include_response=args.include_response,
            max_chars=self._resolve_response_limit(args),
        )
        if entry is None:
            raise ToolError("Failed to read stored semantic entry.")
        return entry

    def _search_semantic_entries(
        self, conn: sqlite3.Connection, args: ContextCacheArgs
    ) -> list[SemanticCacheEntry]:
        if not args.query or not args.query.strip():
            raise ToolError("query is required for semantic_search.")

        scope = self._resolve_scope(args)
        model = args.embedding_model or self.config.embedding_model
        top_k = args.top_k if args.top_k is not None else self.config.max_results
        if top_k <= 0:
            raise ToolError("top_k must be positive.")

        min_score = args.min_score
        if min_score is not None and (min_score < -1.0 or min_score > 1.0):
            raise ToolError("min_score must be between -1.0 and 1.0.")

        max_age = self._resolve_max_age(args)
        max_chars = self._resolve_response_limit(args)
        query_vec = self._embed_text(model, args.query.strip())

        candidates = self._semantic_candidate_rows(
            conn, scope, model, args.context_key
        )
        scored: list[tuple[float, tuple[Any, ...]]] = []
        for row in candidates:
            if max_age is not None and not self._is_recent(row[10], max_age):
                continue
            embedding = self._unpack_embedding(row[7])
            score = self._dot(query_vec, embedding)
            if min_score is not None and score < min_score:
                continue
            if len(scored) < top_k:
                scored.append((score, row))
                scored.sort(key=lambda item: item[0])
            else:
                if score > scored[0][0]:
                    scored[0] = (score, row)
                    scored.sort(key=lambda item: item[0])

        results: list[SemanticCacheEntry] = []
        for score, row in sorted(scored, key=lambda item: item[0], reverse=True):
            results.append(
                self._row_to_semantic_entry(
                    row,
                    score=score,
                    include_response=args.include_response,
                    max_chars=max_chars,
                )
            )
        return results

    def _list_semantic_entries(
        self, conn: sqlite3.Connection, args: ContextCacheArgs
    ) -> list[SemanticCacheEntry]:
        scope = self._resolve_scope(args)
        limit, _ = self._resolve_limits(args)
        max_chars = self._resolve_response_limit(args)
        max_age = self._resolve_max_age(args)

        params: list[Any] = []
        query = (
            "SELECT id, scope, query, response, context_key, embedding_model, "
            "embedding, embedding_dim, metadata, created_at, updated_at "
            "FROM semantic_entries"
        )
        if scope:
            query += " WHERE scope = ?"
            params.append(scope)
        query += " ORDER BY updated_at DESC"
        if limit and limit > 0:
            query += " LIMIT ?"
            params.append(limit)

        entries: list[SemanticCacheEntry] = []
        for row in conn.execute(query, params):
            if max_age is not None and not self._is_recent(row[10], max_age):
                continue
            entries.append(
                self._row_to_semantic_entry(
                    row,
                    score=None,
                    include_response=args.include_response,
                    max_chars=max_chars,
                )
            )
        return entries

    def _delete_semantic_entries(
        self, conn: sqlite3.Connection, args: ContextCacheArgs
    ) -> int:
        scope = self._resolve_scope(args)
        if args.key or args.keys:
            raise ToolError("Use id or ids for semantic_delete.")

        ids = self._resolve_ids(args)
        if ids:
            deleted = 0
            for item_id in ids:
                cursor = conn.execute(
                    "DELETE FROM semantic_entries WHERE id = ?", (item_id,)
                )
                deleted += cursor.rowcount
            return deleted

        if scope:
            cursor = conn.execute(
                "DELETE FROM semantic_entries WHERE scope = ?", (scope,)
            )
            return cursor.rowcount

        cursor = conn.execute("DELETE FROM semantic_entries")
        return cursor.rowcount

    def _cache_stats(
        self, conn: sqlite3.Connection, args: ContextCacheArgs
    ) -> dict[str, int]:
        scope = self._resolve_scope(args)
        params: list[Any] = []
        static_query = "SELECT COUNT(*) FROM static_entries"
        semantic_query = "SELECT COUNT(*) FROM semantic_entries"
        if scope:
            static_query += " WHERE scope = ?"
            semantic_query += " WHERE scope = ?"
            params = [scope]

        static_count = conn.execute(static_query, params).fetchone()[0]
        semantic_count = conn.execute(semantic_query, params).fetchone()[0]
        return {"static_entries": static_count, "semantic_entries": semantic_count}

    def _resolve_keys(self, args: ContextCacheArgs) -> list[str]:
        if args.key and args.keys:
            raise ToolError("Provide key or keys, not both.")
        keys = args.keys or ([args.key] if args.key else [])
        return [key.strip() for key in keys if key and key.strip()]

    def _resolve_ids(self, args: ContextCacheArgs) -> list[int]:
        ids: list[int] = []
        raw_ids = list(args.ids or [])
        if args.id is not None:
            raw_ids.append(args.id)
        for value in raw_ids:
            if not isinstance(value, int):
                raise ToolError("ids must be integers.")
            ids.append(value)
        return ids

    def _resolve_max_age(self, args: ContextCacheArgs) -> timedelta | None:
        if args.max_age_days is None:
            return None
        if args.max_age_days <= 0:
            raise ToolError("max_age_days must be positive.")
        return timedelta(days=float(args.max_age_days))

    def _collect_inputs(
        self, args: ContextCacheArgs, warnings: list[str]
    ) -> list[_ContentItem]:
        if args.content and args.contents:
            raise ToolError("Provide content or contents, not both.")
        if args.path and args.paths:
            raise ToolError("Provide path or paths, not both.")

        max_source = (
            args.max_source_bytes
            if args.max_source_bytes is not None
            else self.config.max_source_bytes
        )
        max_total = (
            args.max_total_bytes
            if args.max_total_bytes is not None
            else self.config.max_total_bytes
        )
        if max_source <= 0:
            raise ToolError("max_source_bytes must be positive.")
        if max_total <= 0:
            raise ToolError("max_total_bytes must be positive.")

        items: list[_ContentItem] = []
        total_bytes = 0
        base_metadata = self._validate_metadata(args.metadata)

        inline_contents: list[str] = []
        if args.content is not None:
            inline_contents = [args.content]
        elif args.contents:
            inline_contents = list(args.contents)

        for text in inline_contents:
            content_bytes = len(text.encode("utf-8"))
            if content_bytes > max_source:
                raise ToolError(
                    f"Content exceeds max_source_bytes ({content_bytes} > {max_source})."
                )
            if total_bytes + content_bytes > max_total:
                warnings.append("max_total_bytes reached; skipping remaining inputs.")
                break
            total_bytes += content_bytes
            items.append(_ContentItem(content=text, metadata=base_metadata))

        file_paths: list[str] = []
        if args.path is not None:
            file_paths = [args.path]
        elif args.paths:
            file_paths = list(args.paths)

        for raw_path in file_paths:
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = self.config.effective_workdir / path
            path = path.resolve()
            if not path.exists():
                raise ToolError(f"Path not found: {path}")
            if path.is_dir():
                raise ToolError(f"Path is a directory: {path}")

            content = path.read_text("utf-8", errors="ignore")
            content_bytes = len(content.encode("utf-8"))
            if content_bytes > max_source:
                raise ToolError(
                    f"{path} exceeds max_source_bytes ({content_bytes} > {max_source})."
                )
            if total_bytes + content_bytes > max_total:
                warnings.append("max_total_bytes reached; skipping remaining inputs.")
                break
            total_bytes += content_bytes
            metadata = self._merge_metadata(base_metadata, {"source_path": str(path)})
            items.append(_ContentItem(content=content, metadata=metadata))

        if not items:
            raise ToolError("No inputs provided for caching.")
        return items

    def _merge_metadata(self, base: dict | None, extra: dict | None) -> dict | None:
        metadata = dict(base) if base else {}
        if extra:
            metadata.update(extra)
        return metadata or None

    def _validate_metadata(self, metadata: dict | None) -> dict | None:
        if metadata is None:
            return None
        if not isinstance(metadata, dict):
            raise ToolError("metadata must be an object.")
        try:
            json.dumps(metadata, ensure_ascii=True, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ToolError(f"metadata is not JSON-serializable: {exc}") from exc
        return dict(metadata)

    def _encode_metadata(self, metadata: dict | None) -> str | None:
        if metadata is None:
            return None
        return json.dumps(metadata, ensure_ascii=True, sort_keys=True)

    def _decode_metadata(self, raw: str | None) -> dict | None:
        if raw is None:
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _hash_content(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _semantic_hash(
        self,
        scope: str,
        model: str,
        query: str,
        response: str,
        context_key: str | None,
    ) -> str:
        seed = json.dumps(
            {
                "scope": scope,
                "model": model,
                "query": query,
                "response": response,
                "context_key": context_key,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()

    def _content_stats(self, content: str) -> dict[str, int]:
        content_bytes = len(content.encode("utf-8"))
        char_count = len(content)
        line_count = content.count("\n") + 1 if content else 0
        token_estimate = 0 if not content else max(1, char_count // 4)
        return {
            "content_bytes": content_bytes,
            "char_count": char_count,
            "line_count": line_count,
            "token_estimate": token_estimate,
        }

    def _init_db(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS static_entries (
                scope TEXT NOT NULL,
                key TEXT NOT NULL,
                content TEXT,
                content_bytes INTEGER,
                char_count INTEGER,
                line_count INTEGER,
                token_estimate INTEGER,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (scope, key)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_entries (
                id INTEGER PRIMARY KEY,
                scope TEXT NOT NULL,
                entry_hash TEXT NOT NULL,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                context_key TEXT,
                embedding_model TEXT NOT NULL,
                embedding BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_semantic_hash ON semantic_entries(entry_hash)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_static_scope ON static_entries(scope)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_semantic_scope ON semantic_entries(scope)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_semantic_context ON semantic_entries(context_key)"
        )

    def _upsert_static_entry(
        self,
        conn: sqlite3.Connection,
        scope: str,
        key: str,
        content: str | None,
        stats: dict[str, int],
        metadata_json: str | None,
        timestamp: str,
        update: bool,
    ) -> None:
        if update:
            conn.execute(
                """
                INSERT INTO static_entries (
                    scope,
                    key,
                    content,
                    content_bytes,
                    char_count,
                    line_count,
                    token_estimate,
                    metadata,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope, key) DO UPDATE SET
                    content = COALESCE(excluded.content, static_entries.content),
                    content_bytes = excluded.content_bytes,
                    char_count = excluded.char_count,
                    line_count = excluded.line_count,
                    token_estimate = excluded.token_estimate,
                    metadata = COALESCE(excluded.metadata, static_entries.metadata),
                    updated_at = excluded.updated_at
                """,
                (
                    scope,
                    key,
                    content,
                    stats["content_bytes"],
                    stats["char_count"],
                    stats["line_count"],
                    stats["token_estimate"],
                    metadata_json,
                    timestamp,
                    timestamp,
                ),
            )
        else:
            conn.execute(
                """
                INSERT OR IGNORE INTO static_entries (
                    scope,
                    key,
                    content,
                    content_bytes,
                    char_count,
                    line_count,
                    token_estimate,
                    metadata,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scope,
                    key,
                    content,
                    stats["content_bytes"],
                    stats["char_count"],
                    stats["line_count"],
                    stats["token_estimate"],
                    metadata_json,
                    timestamp,
                    timestamp,
                ),
            )

    def _fetch_static_entry(
        self,
        conn: sqlite3.Connection,
        scope: str,
        key: str,
        include_content: bool,
        include_metadata: bool,
        max_chars: int,
    ) -> StaticCacheEntry | None:
        row = conn.execute(
            """
            SELECT scope, key, content, content_bytes, char_count, line_count,
                   token_estimate, metadata, created_at, updated_at
            FROM static_entries WHERE scope = ? AND key = ?
            """,
            (scope, key),
        ).fetchone()
        if not row:
            return None
        return self._row_to_static_entry(row, include_content, include_metadata, max_chars)

    def _fetch_static_entries_by_key(
        self,
        conn: sqlite3.Connection,
        scope: str | None,
        key: str,
        include_content: bool,
        include_metadata: bool,
        max_chars: int,
    ) -> list[StaticCacheEntry]:
        params: list[Any] = [key]
        query = (
            "SELECT scope, key, content, content_bytes, char_count, line_count, "
            "token_estimate, metadata, created_at, updated_at FROM static_entries "
            "WHERE key = ?"
        )
        if scope:
            query += " AND scope = ?"
            params.append(scope)

        entries: list[StaticCacheEntry] = []
        for row in conn.execute(query, params):
            entries.append(
                self._row_to_static_entry(row, include_content, include_metadata, max_chars)
            )
        return entries

    def _row_to_static_entry(
        self,
        row: tuple[Any, ...],
        include_content: bool,
        include_metadata: bool,
        max_chars: int,
    ) -> StaticCacheEntry:
        content = row[2] if include_content else None
        if content and max_chars > 0:
            content = content[:max_chars]
        metadata = self._decode_metadata(row[7]) if include_metadata else None
        return StaticCacheEntry(
            scope=row[0],
            key=row[1],
            content=content,
            content_bytes=row[3],
            char_count=row[4],
            line_count=row[5],
            token_estimate=row[6],
            metadata=metadata,
            created_at=row[8],
            updated_at=row[9],
        )

    def _find_semantic_entry(
        self, conn: sqlite3.Connection, entry_hash: str
    ) -> int | None:
        row = conn.execute(
            "SELECT id FROM semantic_entries WHERE entry_hash = ?",
            (entry_hash,),
        ).fetchone()
        if not row:
            return None
        return int(row[0])

    def _fetch_semantic_entry(
        self,
        conn: sqlite3.Connection,
        entry_id: int,
        include_response: bool,
        max_chars: int,
    ) -> SemanticCacheEntry | None:
        row = conn.execute(
            """
            SELECT id, scope, query, response, context_key, embedding_model,
                   embedding, embedding_dim, metadata, created_at, updated_at
            FROM semantic_entries WHERE id = ?
            """,
            (entry_id,),
        ).fetchone()
        if not row:
            return None
        return self._row_to_semantic_entry(
            row, score=None, include_response=include_response, max_chars=max_chars
        )

    def _semantic_candidate_rows(
        self,
        conn: sqlite3.Connection,
        scope: str | None,
        model: str,
        context_key: str | None,
    ) -> list[tuple[Any, ...]]:
        params: list[Any] = [model]
        query = (
            "SELECT id, scope, query, response, context_key, embedding_model, "
            "embedding, embedding_dim, metadata, created_at, updated_at "
            "FROM semantic_entries WHERE embedding_model = ?"
        )
        if scope:
            query += " AND scope = ?"
            params.append(scope)
        if context_key:
            query += " AND context_key = ?"
            params.append(context_key)

        return list(conn.execute(query, params))

    def _row_to_semantic_entry(
        self,
        row: tuple[Any, ...],
        score: float | None,
        include_response: bool,
        max_chars: int,
    ) -> SemanticCacheEntry:
        response = row[3] if include_response else None
        if response and max_chars > 0:
            response = response[:max_chars]
        metadata = self._decode_metadata(row[8])
        return SemanticCacheEntry(
            id=int(row[0]),
            scope=row[1],
            query=row[2],
            response=response,
            score=score,
            embedding_model=row[5],
            context_key=row[4],
            metadata=metadata,
            created_at=row[9],
            updated_at=row[10],
        )

    def _is_recent(self, timestamp: str, max_age: timedelta) -> bool:
        try:
            parsed = datetime.fromisoformat(timestamp)
        except ValueError:
            return False
        return datetime.utcnow() - parsed <= max_age

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
            with request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise ToolError(f"Ollama embeddings failed: {exc}") from exc

        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            raise ToolError("Invalid embeddings response from Ollama.")
        return self._normalize_embedding([float(x) for x in embedding])

    def _normalize_embedding(self, embedding: list[float]) -> list[float]:
        norm = sum(x * x for x in embedding) ** 0.5
        if norm == 0:
            return embedding
        return [x / norm for x in embedding]

    def _pack_embedding(self, embedding: list[float]) -> bytes:
        arr = array("f", embedding)
        return arr.tobytes()

    def _unpack_embedding(self, blob: bytes) -> list[float]:
        arr = array("f")
        arr.frombytes(blob)
        return list(arr)

    def _dot(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            return 0.0
        return sum(l * r for l, r in zip(left, right))

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextCacheArgs):
            return ToolCallDisplay(summary="context_cache")
        return ToolCallDisplay(
            summary=f"context_cache ({event.args.action})",
            details={
                "scope": event.args.scope,
                "action": event.args.action,
                "key": event.args.key,
                "query": event.args.query,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextCacheResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        action = event.result.action
        message = f"context_cache {action}"
        if action.startswith("static"):
            message += f" ({len(event.result.static_entries)} item(s))"
        elif action.startswith("semantic"):
            message += f" ({len(event.result.semantic_entries)} item(s))"
        elif action == "stats":
            message += " (stats)"

        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "hashes": event.result.hashes,
                "deleted": event.result.deleted,
                "stats": event.result.stats,
                "static_entries": event.result.static_entries,
                "semantic_entries": event.result.semantic_entries,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Caching context"
