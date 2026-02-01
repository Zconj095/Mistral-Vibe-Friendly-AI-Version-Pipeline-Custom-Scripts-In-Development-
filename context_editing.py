from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
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


DEFAULT_STORE_PATH = Path.home() / ".vibe" / "memory" / "context_editing.json"
SUPPORTED_ACTIONS = {
    "init",
    "list_sessions",
    "get",
    "upsert",
    "delete",
    "reorder",
    "assemble",
    "stats",
    "clear",
}
VALID_ROLES = {"system", "user", "assistant", "tool", "context"}


class ContextEditingConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    store_path: Path = Field(
        default=DEFAULT_STORE_PATH,
        description="Path to the context editing store.",
    )
    max_content_chars: int = Field(
        default=4000, description="Maximum characters per segment in output."
    )
    max_prompt_chars: int = Field(
        default=16000, description="Maximum characters in assembled prompt."
    )


class ContextEditingState(BaseToolState):
    pass


class ContextSegmentInput(BaseModel):
    id: str = Field(description="Segment id.")
    content: str | None = Field(default=None, description="Segment content.")
    path: str | None = Field(default=None, description="Path to content file.")
    role: str | None = Field(default=None, description="Segment role.")
    tags: list[str] | None = Field(default=None, description="Optional tags.")
    metadata: dict | None = Field(default=None, description="Optional metadata.")
    index: int | None = Field(default=None, description="Insert index (0-based).")


class ContextEditingArgs(BaseModel):
    action: str | None = Field(default="get", description="Action to run.")
    session_id: str | None = Field(
        default=None, description="Context session id."
    )
    segments: list[ContextSegmentInput] | None = Field(
        default=None, description="Segments to insert or update."
    )
    ids: list[str] | None = Field(
        default=None, description="Segment ids for get/delete."
    )
    tags: list[str] | None = Field(
        default=None, description="Segment tags filter."
    )
    exclude_ids: list[str] | None = Field(
        default=None, description="Segment ids to exclude."
    )
    exclude_tags: list[str] | None = Field(
        default=None, description="Segment tags to exclude."
    )
    order: list[str] | None = Field(
        default=None, description="Segment id order for reorder."
    )
    overwrite: bool = Field(
        default=False, description="Overwrite session on init."
    )
    include_content: bool = Field(
        default=True, description="Include content in output."
    )
    include_metadata: bool = Field(
        default=True, description="Include metadata in output."
    )
    max_content_chars: int | None = Field(
        default=None, description="Override max content characters."
    )
    include_role_headers: bool = Field(
        default=False, description="Include role headers in prompt."
    )
    merge_same_role: bool = Field(
        default=True, description="Merge adjacent segments with same role."
    )
    prompt_separator: str | None = Field(
        default=None, description="Separator between segments in prompt."
    )
    max_prompt_chars: int | None = Field(
        default=None, description="Override max prompt chars."
    )


class ContextSegmentOutput(BaseModel):
    id: str
    role: str
    content: str | None
    tags: list[str]
    metadata: dict | None
    created_at: str
    updated_at: str
    index: int
    char_count: int
    token_estimate: int
    content_hash: str


class ContextEditingResult(BaseModel):
    action: str
    session_id: str
    segments: list[ContextSegmentOutput]
    messages: list[dict[str, str]] | None
    prompt: str | None
    stats: dict[str, Any] | None
    sessions: list[dict[str, Any]] | None
    warnings: list[str]
    errors: list[str]


class ContextEditing(
    BaseTool[
        ContextEditingArgs,
        ContextEditingResult,
        ContextEditingConfig,
        ContextEditingState,
    ],
    ToolUIData[ContextEditingArgs, ContextEditingResult],
):
    description: ClassVar[str] = (
        "Edit and assemble segmented context for long-running sessions."
    )

    async def run(self, args: ContextEditingArgs) -> ContextEditingResult:
        action = (args.action or "get").strip().lower()
        if action not in SUPPORTED_ACTIONS:
            raise ToolError(
                f"action must be one of: {', '.join(sorted(SUPPORTED_ACTIONS))}"
            )

        warnings: list[str] = []
        errors: list[str] = []

        store_path = self.config.store_path
        store = self._load_store(store_path)

        if action == "list_sessions":
            sessions = self._list_sessions(store)
            return ContextEditingResult(
                action=action,
                session_id=args.session_id or "",
                segments=[],
                messages=None,
                prompt=None,
                stats=None,
                sessions=sessions,
                warnings=warnings,
                errors=errors,
            )

        session_id = self._require_session_id(args.session_id)

        if action == "init":
            session = self._init_session(store, session_id, args, warnings)
            self._save_store(store_path, store)
            segments_out = self._serialize_segments(
                session["segments"], args, warnings
            )
            return ContextEditingResult(
                action=action,
                session_id=session_id,
                segments=segments_out,
                messages=None,
                prompt=None,
                stats=self._session_stats(session["segments"]),
                sessions=None,
                warnings=warnings,
                errors=errors,
            )

        if action == "clear":
            removed = self._clear_session(store, session_id, warnings)
            if removed:
                self._save_store(store_path, store)
            return ContextEditingResult(
                action=action,
                session_id=session_id,
                segments=[],
                messages=None,
                prompt=None,
                stats=None,
                sessions=None,
                warnings=warnings,
                errors=errors,
            )

        session = self._get_session(store, session_id)

        if action == "upsert":
            self._upsert_segments(session, args, warnings)
            self._touch_session(session)
            self._save_store(store_path, store)
            segments_out = self._serialize_segments(
                session["segments"], args, warnings
            )
            return ContextEditingResult(
                action=action,
                session_id=session_id,
                segments=segments_out,
                messages=None,
                prompt=None,
                stats=self._session_stats(session["segments"]),
                sessions=None,
                warnings=warnings,
                errors=errors,
            )

        if action == "delete":
            deleted = self._delete_segments(session, args, warnings)
            if deleted:
                self._touch_session(session)
                self._save_store(store_path, store)
            segments_out = self._serialize_segments(
                session["segments"], args, warnings
            )
            return ContextEditingResult(
                action=action,
                session_id=session_id,
                segments=segments_out,
                messages=None,
                prompt=None,
                stats=self._session_stats(session["segments"]),
                sessions=None,
                warnings=warnings,
                errors=errors,
            )

        if action == "reorder":
            self._reorder_segments(session, args, warnings)
            self._touch_session(session)
            self._save_store(store_path, store)
            segments_out = self._serialize_segments(
                session["segments"], args, warnings
            )
            return ContextEditingResult(
                action=action,
                session_id=session_id,
                segments=segments_out,
                messages=None,
                prompt=None,
                stats=self._session_stats(session["segments"]),
                sessions=None,
                warnings=warnings,
                errors=errors,
            )

        if action == "assemble":
            segments = self._filter_segments(session["segments"], args, warnings)
            messages, prompt = self._assemble_prompt(segments, args, warnings)
            segments_out = self._serialize_segments(segments, args, warnings)
            return ContextEditingResult(
                action=action,
                session_id=session_id,
                segments=segments_out,
                messages=messages,
                prompt=prompt,
                stats=self._session_stats(segments),
                sessions=None,
                warnings=warnings,
                errors=errors,
            )

        if action == "stats":
            stats = self._session_stats(session["segments"])
            segments_out = self._serialize_segments(
                session["segments"], args, warnings
            )
            return ContextEditingResult(
                action=action,
                session_id=session_id,
                segments=segments_out,
                messages=None,
                prompt=None,
                stats=stats,
                sessions=None,
                warnings=warnings,
                errors=errors,
            )

        segments = self._filter_segments(session["segments"], args, warnings)
        segments_out = self._serialize_segments(segments, args, warnings)
        return ContextEditingResult(
            action=action,
            session_id=session_id,
            segments=segments_out,
            messages=None,
            prompt=None,
            stats=self._session_stats(segments),
            sessions=None,
            warnings=warnings,
            errors=errors,
        )

    def _require_session_id(self, value: str | None) -> str:
        if not value or not value.strip():
            raise ToolError("session_id is required.")
        return value.strip()

    def _load_store(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"version": 1, "updated_at": self._now_iso(), "sessions": {}}
        try:
            raw = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ToolError(f"Failed to read store: {exc}") from exc

        if not isinstance(raw, dict):
            raise ToolError("Invalid store format.")
        if "sessions" not in raw or not isinstance(raw["sessions"], dict):
            raw["sessions"] = {}
        if "version" not in raw:
            raw["version"] = 1
        return raw

    def _save_store(self, path: Path, store: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        store["updated_at"] = self._now_iso()
        payload = json.dumps(store, ensure_ascii=True, indent=2)
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(payload, "utf-8")
        temp_path.replace(path)

    def _list_sessions(self, store: dict[str, Any]) -> list[dict[str, Any]]:
        sessions = []
        for session_id, session in store.get("sessions", {}).items():
            segments = session.get("segments", [])
            sessions.append(
                {
                    "session_id": session_id,
                    "segment_count": len(segments),
                    "created_at": session.get("created_at"),
                    "updated_at": session.get("updated_at"),
                }
            )
        sessions.sort(key=lambda item: item["session_id"])
        return sessions

    def _get_session(self, store: dict[str, Any], session_id: str) -> dict[str, Any]:
        sessions = store.get("sessions", {})
        session = sessions.get(session_id)
        if session is None:
            raise ToolError(f"Session not found: {session_id}")
        if "segments" not in session or not isinstance(session["segments"], list):
            session["segments"] = []
        return session

    def _init_session(
        self,
        store: dict[str, Any],
        session_id: str,
        args: ContextEditingArgs,
        warnings: list[str],
    ) -> dict[str, Any]:
        sessions = store.setdefault("sessions", {})
        if session_id in sessions and not args.overwrite:
            raise ToolError("Session already exists. Set overwrite=true to replace.")

        now = self._now_iso()
        segments = []
        if args.segments:
            seen: set[str] = set()
            for segment in args.segments:
                segment_id = segment.id.strip()
                if segment_id in seen:
                    raise ToolError(f"Duplicate segment id: {segment_id}")
                seen.add(segment_id)
                segments.append(self._build_segment(segment, now))

        session = {
            "created_at": now,
            "updated_at": now,
            "segments": segments,
        }
        sessions[session_id] = session
        return session

    def _clear_session(
        self,
        store: dict[str, Any],
        session_id: str,
        warnings: list[str],
    ) -> bool:
        sessions = store.get("sessions", {})
        if session_id not in sessions:
            warnings.append("Session did not exist.")
            return False
        del sessions[session_id]
        return True

    def _touch_session(self, session: dict[str, Any]) -> None:
        session["updated_at"] = self._now_iso()

    def _upsert_segments(
        self, session: dict[str, Any], args: ContextEditingArgs, warnings: list[str]
    ) -> None:
        if not args.segments:
            raise ToolError("segments is required for upsert.")

        segments = session.get("segments", [])
        index_by_id = {seg["id"]: idx for idx, seg in enumerate(segments)}
        now = self._now_iso()

        for item in args.segments:
            segment_id = item.id.strip()
            if not segment_id:
                raise ToolError("Segment id cannot be empty.")
            existing_idx = index_by_id.get(segment_id)
            if existing_idx is None:
                new_seg = self._build_segment(item, now)
                insert_idx = self._resolve_insert_index(item.index, len(segments))
                segments.insert(insert_idx, new_seg)
                index_by_id = {seg["id"]: idx for idx, seg in enumerate(segments)}
            else:
                seg = segments[existing_idx]
                updated = self._update_segment(seg, item, now, warnings)
                if updated:
                    segments[existing_idx] = updated
                if item.index is not None:
                    new_index = self._resolve_insert_index(item.index, len(segments))
                    moved = segments.pop(existing_idx)
                    segments.insert(new_index, moved)
                    index_by_id = {seg["id"]: idx for idx, seg in enumerate(segments)}

        session["segments"] = segments

    def _delete_segments(
        self, session: dict[str, Any], args: ContextEditingArgs, warnings: list[str]
    ) -> int:
        ids = self._normalize_ids(args.ids)
        if not ids:
            raise ToolError("ids is required for delete.")

        segments = session.get("segments", [])
        removed = 0
        remaining = []
        for seg in segments:
            if seg["id"] in ids:
                removed += 1
            else:
                remaining.append(seg)
        if removed == 0:
            warnings.append("No matching segments to delete.")
        session["segments"] = remaining
        return removed

    def _reorder_segments(
        self, session: dict[str, Any], args: ContextEditingArgs, warnings: list[str]
    ) -> None:
        order = [seg_id.strip() for seg_id in (args.order or []) if seg_id and seg_id.strip()]
        if not order:
            raise ToolError("order is required for reorder.")

        segments = session.get("segments", [])
        segment_map = {seg["id"]: seg for seg in segments}
        ordered: list[dict[str, Any]] = []
        seen: set[str] = set()

        for seg_id in order:
            seg = segment_map.get(seg_id)
            if not seg:
                warnings.append(f"Segment id not found: {seg_id}")
                continue
            ordered.append(seg)
            seen.add(seg_id)

        for seg in segments:
            if seg["id"] not in seen:
                ordered.append(seg)

        session["segments"] = ordered

    def _filter_segments(
        self,
        segments: list[dict[str, Any]],
        args: ContextEditingArgs,
        warnings: list[str],
    ) -> list[dict[str, Any]]:
        include_ids = set(self._normalize_ids(args.ids))
        include_tags = self._normalize_tags(args.tags)
        exclude_ids = set(self._normalize_ids(args.exclude_ids))
        exclude_tags = self._normalize_tags(args.exclude_tags)

        filtered: list[dict[str, Any]] = []
        for seg in segments:
            seg_id = seg.get("id")
            if include_ids and seg_id not in include_ids:
                continue
            if include_tags and not (set(seg.get("tags", [])) & set(include_tags)):
                continue
            if exclude_ids and seg_id in exclude_ids:
                continue
            if exclude_tags and (set(seg.get("tags", [])) & set(exclude_tags)):
                continue
            filtered.append(seg)

        return filtered

    def _assemble_prompt(
        self,
        segments: list[dict[str, Any]],
        args: ContextEditingArgs,
        warnings: list[str],
    ) -> tuple[list[dict[str, str]], str]:
        separator = args.prompt_separator if args.prompt_separator is not None else "\n\n"
        max_prompt = (
            args.max_prompt_chars
            if args.max_prompt_chars is not None
            else self.config.max_prompt_chars
        )

        messages: list[dict[str, str]] = []
        prompt_parts: list[str] = []
        last_role: str | None = None

        for seg in segments:
            role = seg.get("role") or "user"
            content = seg.get("content") or ""
            if not content:
                continue

            if args.merge_same_role and messages and last_role == role:
                messages[-1]["content"] = messages[-1]["content"] + "\n" + content
            else:
                messages.append({"role": role, "content": content})
                last_role = role

            if args.include_role_headers:
                prompt_parts.append(f"{role}: {content}")
            else:
                prompt_parts.append(content)

        prompt = separator.join(prompt_parts).strip()
        if max_prompt > 0 and len(prompt) > max_prompt:
            prompt = prompt[:max_prompt].rstrip() + "..."
            warnings.append("Prompt truncated to max_prompt_chars.")

        return messages, prompt

    def _build_segment(
        self, item: ContextSegmentInput, now: str
    ) -> dict[str, Any]:
        content = self._resolve_content(item)
        if content is None or not content.strip():
            raise ToolError("Segment content cannot be empty.")
        role = self._normalize_role(item.role)
        tags = self._normalize_tags(item.tags)
        metadata = self._validate_metadata(item.metadata)
        return {
            "id": item.id.strip(),
            "role": role,
            "content": content,
            "tags": tags,
            "metadata": metadata,
            "created_at": now,
            "updated_at": now,
        }

    def _update_segment(
        self,
        segment: dict[str, Any],
        item: ContextSegmentInput,
        now: str,
        warnings: list[str],
    ) -> dict[str, Any]:
        updated = dict(segment)
        if item.content is not None or item.path is not None:
            content = self._resolve_content(item)
            if content is None or not content.strip():
                warnings.append("Updated content was empty; keeping existing content.")
            else:
                updated["content"] = content
        if item.role:
            updated["role"] = self._normalize_role(item.role)
        if item.tags is not None:
            updated["tags"] = self._normalize_tags(item.tags)
        if item.metadata is not None:
            updated["metadata"] = self._validate_metadata(item.metadata)
        updated["updated_at"] = now
        return updated

    def _resolve_content(self, item: ContextSegmentInput) -> str | None:
        if item.content is not None and item.path is not None:
            raise ToolError("Provide content or path per segment, not both.")
        if item.content is not None:
            return item.content
        if item.path is None:
            return None
        path = self._resolve_path(item.path)
        return path.read_text("utf-8", errors="ignore")

    def _resolve_path(self, raw_path: str) -> Path:
        if not raw_path.strip():
            raise ToolError("Path cannot be empty.")

        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.config.effective_workdir / path
        try:
            resolved = path.resolve()
        except ValueError as exc:
            raise ToolError("Security error: cannot resolve the provided path.") from exc
        if not resolved.exists():
            raise ToolError(f"File not found: {resolved}")
        if resolved.is_dir():
            raise ToolError(f"Path is a directory: {resolved}")
        return resolved

    def _serialize_segments(
        self,
        segments: list[dict[str, Any]],
        args: ContextEditingArgs,
        warnings: list[str],
    ) -> list[ContextSegmentOutput]:
        max_chars = (
            args.max_content_chars
            if args.max_content_chars is not None
            else self.config.max_content_chars
        )
        if max_chars <= 0:
            raise ToolError("max_content_chars must be positive.")

        output: list[ContextSegmentOutput] = []
        for idx, seg in enumerate(segments):
            content = seg.get("content") or ""
            content_out = content if args.include_content else None
            if content_out is not None and len(content_out) > max_chars:
                content_out = content_out[:max_chars].rstrip() + "..."
                warnings.append(f"Segment {seg.get('id')} truncated to max_content_chars.")
            output.append(
                ContextSegmentOutput(
                    id=seg.get("id", ""),
                    role=seg.get("role", "user"),
                    content=content_out,
                    tags=list(seg.get("tags") or []),
                    metadata=seg.get("metadata") if args.include_metadata else None,
                    created_at=seg.get("created_at", ""),
                    updated_at=seg.get("updated_at", ""),
                    index=idx,
                    char_count=len(content),
                    token_estimate=self._token_estimate(content),
                    content_hash=self._hash_content(content),
                )
            )
        return output

    def _session_stats(self, segments: list[dict[str, Any]]) -> dict[str, Any]:
        stats = {"segments": len(segments), "chars": 0, "roles": {}}
        for seg in segments:
            content = seg.get("content") or ""
            stats["chars"] += len(content)
            role = seg.get("role", "user")
            stats["roles"][role] = stats["roles"].get(role, 0) + 1
        return stats

    def _normalize_role(self, role: str | None) -> str:
        value = (role or "user").strip().lower()
        if value not in VALID_ROLES:
            raise ToolError(f"role must be one of: {', '.join(sorted(VALID_ROLES))}")
        return value

    def _normalize_tags(self, tags: list[str] | None) -> list[str]:
        if not tags:
            return []
        normalized = []
        seen = set()
        for tag in tags:
            value = (tag or "").strip()
            if not value:
                continue
            if value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        return normalized

    def _normalize_ids(self, ids: list[str] | None) -> list[str]:
        if not ids:
            return []
        cleaned = []
        for value in ids:
            if not value:
                continue
            trimmed = value.strip()
            if trimmed:
                cleaned.append(trimmed)
        return cleaned

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

    def _resolve_insert_index(self, index: int | None, length: int) -> int:
        if index is None:
            return length
        if index < 0:
            return 0
        if index > length:
            return length
        return index

    def _hash_content(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _token_estimate(self, content: str) -> int:
        if not content:
            return 0
        return max(1, len(content) // 4)

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextEditingArgs):
            return ToolCallDisplay(summary="context_editing")
        return ToolCallDisplay(
            summary=f"context_editing ({event.args.action})",
            details={
                "session_id": event.args.session_id,
                "segments": len(event.args.segments or []),
                "ids": event.args.ids,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextEditingResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = f"context_editing {event.result.action}"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "segments": event.result.segments,
                "prompt": event.result.prompt,
                "stats": event.result.stats,
                "sessions": event.result.sessions,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Editing context"
