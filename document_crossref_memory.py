from __future__ import annotations

import fnmatch
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
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


DEFAULT_DB_PATH = Path.home() / ".vibe" / "memory" / "document_crossrefs.sqlite"
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
FILENAME_RE = re.compile(r"[A-Za-z0-9_.-]+\\.[A-Za-z0-9]{1,10}")
LINK_RE = re.compile(r"\\[[^\\]]+\\]\\(([^)]+)\\)")
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


@dataclass
class _DocInfo:
    path: Path
    name: str
    title: str | None
    tokens: list[str]
    content: str
    size: int
    mtime: int


class DocumentCrossrefMemoryConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    db_path: Path = Field(
        default=DEFAULT_DB_PATH,
        description="Path to the sqlite database file.",
    )
    max_source_bytes: int = Field(
        default=5_000_000, description="Maximum size per file (bytes)."
    )
    max_total_bytes: int = Field(
        default=20_000_000, description="Maximum total bytes across files."
    )
    max_files: int = Field(
        default=500, description="Maximum files to process."
    )
    max_tokens_per_doc: int = Field(
        default=2000, description="Maximum tokens stored per document."
    )
    min_token_length: int = Field(
        default=3, description="Minimum token length."
    )
    max_results: int = Field(
        default=10, description="Maximum cross-reference results."
    )
    default_extensions: list[str] = Field(
        default=[
            ".md",
            ".txt",
            ".rst",
            ".log",
            ".csv",
            ".tsv",
        ],
        description="Default document extensions.",
    )
    default_exclude_globs: list[str] = Field(
        default=[
            "**/.git/**",
            "**/.svn/**",
            "**/.hg/**",
            "**/node_modules/**",
            "**/.venv/**",
            "**/venv/**",
            "**/.idea/**",
            "**/.vscode/**",
            "**/dist/**",
            "**/build/**",
            "**/target/**",
            "**/.mypy_cache/**",
            "**/.pytest_cache/**",
            "**/.cache/**",
            "**/__pycache__/**",
        ],
        description="Default glob patterns excluded during auto-discovery.",
    )


class DocumentCrossrefMemoryState(BaseToolState):
    pass


class DocumentCrossrefMemoryArgs(BaseModel):
    action: str = Field(
        description="index, query, stats, or clear."
    )
    paths: list[str] | None = Field(
        default=None, description="Root directories or files to scan."
    )
    extension: str | None = Field(
        default=None, description="Single extension to match (e.g. .md)."
    )
    extensions: list[str] | None = Field(
        default=None, description="Extensions to match (e.g. ['.md', '.txt'])."
    )
    include_globs: list[str] | None = Field(
        default=None, description="Optional include globs."
    )
    exclude_globs: list[str] | None = Field(
        default=None, description="Optional exclude globs."
    )
    replace_existing: bool = Field(
        default=False, description="Rebuild entries for existing documents."
    )
    query_text: str | None = Field(
        default=None, description="Query text for cross-references."
    )
    path: str | None = Field(
        default=None, description="Document path to query from memory."
    )
    top_k: int | None = Field(
        default=None, description="Override max results."
    )
    max_files: int | None = Field(
        default=None, description="Override max files limit."
    )
    max_source_bytes: int | None = Field(
        default=None, description="Override max_source_bytes."
    )
    max_total_bytes: int | None = Field(
        default=None, description="Override max_total_bytes."
    )
    db_path: str | None = Field(
        default=None, description="Override database path."
    )


class DocumentCrossrefMatch(BaseModel):
    path: str
    title: str | None
    overlap: int
    explicit_refs: int
    score: int


class DocumentCrossrefMemoryResult(BaseModel):
    action: str
    indexed_files: int
    skipped_files: int
    reference_count: int
    document_count: int
    matches: list[DocumentCrossrefMatch]
    errors: list[str]
    db_path: str


class DocumentCrossrefMemory(
    BaseTool[
        DocumentCrossrefMemoryArgs,
        DocumentCrossrefMemoryResult,
        DocumentCrossrefMemoryConfig,
        DocumentCrossrefMemoryState,
    ],
    ToolUIData[DocumentCrossrefMemoryArgs, DocumentCrossrefMemoryResult],
):
    description: ClassVar[str] = (
        "Cross-reference documents using a persistent memory store."
    )

    async def run(self, args: DocumentCrossrefMemoryArgs) -> DocumentCrossrefMemoryResult:
        action = (args.action or "").strip().lower()
        if action not in {"index", "query", "stats", "clear"}:
            raise ToolError("action must be index, query, stats, or clear.")

        db_path = Path(args.db_path).expanduser() if args.db_path else self.config.db_path
        db_path = db_path.resolve()

        if action == "clear":
            removed = 0
            if db_path.exists():
                db_path.unlink()
                removed = 1
            return DocumentCrossrefMemoryResult(
                action=action,
                indexed_files=0,
                skipped_files=0,
                reference_count=0,
                document_count=0,
                matches=[],
                errors=[],
                db_path=str(db_path),
            )

        if action in {"query", "stats"} and not db_path.exists():
            raise ToolError("Document memory store not found. Run action=index first.")

        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        try:
            self._init_db(conn)

            if action == "stats":
                doc_count = self._count_rows(conn, "documents")
                ref_count = self._count_rows(conn, "references")
                return DocumentCrossrefMemoryResult(
                    action=action,
                    indexed_files=0,
                    skipped_files=0,
                    reference_count=ref_count,
                    document_count=doc_count,
                    matches=[],
                    errors=[],
                    db_path=str(db_path),
                )

            if action == "index":
                return self._index_documents(conn, args, db_path)

            return self._query_documents(conn, args, db_path)
        finally:
            conn.close()

    def _count_rows(self, conn: sqlite3.Connection, table: str) -> int:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0]) if row else 0

    def _index_documents(
        self,
        conn: sqlite3.Connection,
        args: DocumentCrossrefMemoryArgs,
        db_path: Path,
    ) -> DocumentCrossrefMemoryResult:
        if not args.paths:
            raise ToolError("paths is required for action=index.")

        max_files = args.max_files if args.max_files is not None else self.config.max_files
        if max_files <= 0:
            raise ToolError("max_files must be a positive integer.")

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
        if max_source_bytes <= 0:
            raise ToolError("max_source_bytes must be a positive integer.")
        if max_total_bytes <= 0:
            raise ToolError("max_total_bytes must be a positive integer.")

        include_globs = self._resolve_include_globs(args)
        exclude_globs = self._normalize_globs(
            args.exclude_globs, self.config.default_exclude_globs
        )

        files = self._gather_files(args.paths, include_globs, exclude_globs, max_files)
        if not files:
            raise ToolError("No files matched the provided paths/patterns.")

        errors: list[str] = []
        indexed_files = 0
        skipped_files = 0
        reference_count = 0
        total_bytes = 0
        docs: list[_DocInfo] = []

        for file_path in files:
            try:
                stat = file_path.stat()
                if stat.st_size > max_source_bytes:
                    raise ToolError(
                        f"{file_path} exceeds max_source_bytes ({stat.st_size} > {max_source_bytes})."
                    )
                if total_bytes + stat.st_size > max_total_bytes:
                    break

                if not args.replace_existing and self._is_up_to_date(
                    conn, file_path, stat.st_mtime_ns, stat.st_size
                ):
                    skipped_files += 1
                    continue

                content = file_path.read_text("utf-8", errors="ignore")
                tokens = self._extract_tokens(content)
                title = self._extract_title(content, file_path)
                docs.append(
                    _DocInfo(
                        path=file_path,
                        name=file_path.name.lower(),
                        title=title,
                        tokens=tokens,
                        content=content,
                        size=stat.st_size,
                        mtime=stat.st_mtime_ns,
                    )
                )
                total_bytes += stat.st_size
            except ToolError as exc:
                errors.append(str(exc))
            except Exception as exc:
                errors.append(f"{file_path}: {exc}")

        for doc in docs:
            self._upsert_document(conn, doc)
            indexed_files += 1

        conn.commit()

        name_map = self._load_name_map(conn)
        for doc in docs:
            source_id = self._get_document_id(conn, doc.path)
            if source_id is None:
                continue
            conn.execute("DELETE FROM references WHERE source_id = ?", (source_id,))
            refs = self._extract_references(doc, name_map)
            for target_id, weight in refs.items():
                if target_id == source_id:
                    continue
                self._upsert_reference(conn, source_id, target_id, weight)
                reference_count += 1

        conn.commit()

        doc_count = self._count_rows(conn, "documents")
        return DocumentCrossrefMemoryResult(
            action="index",
            indexed_files=indexed_files,
            skipped_files=skipped_files,
            reference_count=reference_count,
            document_count=doc_count,
            matches=[],
            errors=errors,
            db_path=str(db_path),
        )

    def _query_documents(
        self,
        conn: sqlite3.Connection,
        args: DocumentCrossrefMemoryArgs,
        db_path: Path,
    ) -> DocumentCrossrefMemoryResult:
        if not args.query_text and not args.path:
            raise ToolError("query_text or path is required for action=query.")

        top_k = args.top_k if args.top_k is not None else self.config.max_results
        if top_k <= 0:
            raise ToolError("top_k must be a positive integer.")

        query_tokens: set[str] = set()
        explicit_refs: dict[int, int] = {}
        source_id: int | None = None

        if args.path:
            source_path = Path(args.path).expanduser()
            if not source_path.is_absolute():
                source_path = self.config.effective_workdir / source_path
            source_path = source_path.resolve()

            source_id = self._get_document_id(conn, source_path)
            if source_id is None:
                raise ToolError(f"Document not found in memory: {source_path}")

            stored_tokens = self._get_document_tokens(conn, source_id)
            query_tokens.update(stored_tokens)
            explicit_refs = self._get_explicit_refs(conn, source_id)

        if args.query_text:
            query_tokens.update(self._extract_tokens(args.query_text))

        if not query_tokens and not explicit_refs:
            raise ToolError("No usable tokens to query cross-references.")

        matches: list[DocumentCrossrefMatch] = []
        rows = conn.execute(
            "SELECT id, path, title, tokens FROM documents"
        ).fetchall()
        for doc_id, path, title, tokens_json in rows:
            if source_id is not None and doc_id == source_id:
                continue
            tokens = set(json.loads(tokens_json)) if tokens_json else set()
            overlap = len(query_tokens & tokens)
            explicit_weight = explicit_refs.get(doc_id, 0)
            if overlap == 0 and explicit_weight == 0:
                continue
            score = overlap + explicit_weight * 5
            matches.append(
                DocumentCrossrefMatch(
                    path=path,
                    title=title,
                    overlap=overlap,
                    explicit_refs=explicit_weight,
                    score=score,
                )
            )

        matches.sort(
            key=lambda item: (-int(item.explicit_refs > 0), -item.explicit_refs, -item.overlap, item.path)
        )
        matches = matches[:top_k]

        doc_count = self._count_rows(conn, "documents")
        return DocumentCrossrefMemoryResult(
            action="query",
            indexed_files=0,
            skipped_files=0,
            reference_count=len(explicit_refs),
            document_count=doc_count,
            matches=matches,
            errors=[],
            db_path=str(db_path),
        )

    def _init_db(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                title TEXT,
                tokens TEXT NOT NULL,
                size INTEGER NOT NULL,
                mtime INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS references (
                source_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                weight INTEGER NOT NULL,
                PRIMARY KEY (source_id, target_id)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_references_source ON references(source_id)"
        )

    def _upsert_document(self, conn: sqlite3.Connection, doc: _DocInfo) -> None:
        conn.execute(
            """
            INSERT INTO documents (
                path, name, title, tokens, size, mtime, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                name = excluded.name,
                title = excluded.title,
                tokens = excluded.tokens,
                size = excluded.size,
                mtime = excluded.mtime,
                updated_at = excluded.updated_at
            """,
            (
                str(doc.path),
                doc.name,
                doc.title,
                json.dumps(doc.tokens),
                doc.size,
                doc.mtime,
                datetime.utcnow().isoformat(),
            ),
        )

    def _is_up_to_date(
        self, conn: sqlite3.Connection, path: Path, mtime: int, size: int
    ) -> bool:
        row = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE path = ? AND mtime = ? AND size = ?",
            (str(path), mtime, size),
        ).fetchone()
        return bool(row and row[0] > 0)

    def _get_document_id(
        self, conn: sqlite3.Connection, path: Path
    ) -> int | None:
        row = conn.execute(
            "SELECT id FROM documents WHERE path = ?", (str(path),)
        ).fetchone()
        return int(row[0]) if row else None

    def _get_document_tokens(
        self, conn: sqlite3.Connection, doc_id: int
    ) -> list[str]:
        row = conn.execute(
            "SELECT tokens FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        if not row:
            return []
        return json.loads(row[0]) if row[0] else []

    def _load_name_map(self, conn: sqlite3.Connection) -> dict[str, list[int]]:
        name_map: dict[str, list[int]] = {}
        rows = conn.execute("SELECT id, name FROM documents").fetchall()
        for doc_id, name in rows:
            key = (name or "").lower()
            if not key:
                continue
            name_map.setdefault(key, []).append(int(doc_id))
        return name_map

    def _upsert_reference(
        self, conn: sqlite3.Connection, source_id: int, target_id: int, weight: int
    ) -> None:
        conn.execute(
            """
            INSERT INTO references (source_id, target_id, weight)
            VALUES (?, ?, ?)
            ON CONFLICT(source_id, target_id) DO UPDATE SET
                weight = weight + excluded.weight
            """,
            (source_id, target_id, weight),
        )

    def _get_explicit_refs(
        self, conn: sqlite3.Connection, source_id: int
    ) -> dict[int, int]:
        rows = conn.execute(
            "SELECT target_id, weight FROM references WHERE source_id = ?",
            (source_id,),
        ).fetchall()
        return {int(target_id): int(weight) for target_id, weight in rows}

    def _extract_tokens(self, content: str) -> list[str]:
        min_len = self.config.min_token_length
        tokens: dict[str, int] = {}
        for match in TOKEN_RE.findall(content.lower()):
            if len(match) < min_len:
                continue
            if match in STOPWORDS:
                continue
            tokens[match] = tokens.get(match, 0) + 1

        if not tokens:
            return []

        sorted_tokens = sorted(tokens.items(), key=lambda item: (-item[1], item[0]))
        max_tokens = self.config.max_tokens_per_doc
        return [token for token, _ in sorted_tokens[:max_tokens]]

    def _extract_title(self, content: str, path: Path) -> str | None:
        lines = content.splitlines()
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip() or path.stem
            if stripped.lower().startswith("title:"):
                return stripped.split(":", 1)[-1].strip() or path.stem
            return stripped[:120]
        return path.stem if path.stem else None

    def _extract_references(
        self, doc: _DocInfo, name_map: dict[str, list[int]]
    ) -> dict[int, int]:
        refs: dict[int, int] = {}
        content = doc.content
        lower = content.lower()

        for target in LINK_RE.findall(content):
            name = self._normalize_link_target(target)
            if not name:
                continue
            key = Path(name).name.lower()
            for target_id in name_map.get(key, []):
                refs[target_id] = refs.get(target_id, 0) + 2

        for match in FILENAME_RE.findall(lower):
            key = match.lower()
            for target_id in name_map.get(key, []):
                refs[target_id] = refs.get(target_id, 0) + 1

        return refs

    def _normalize_link_target(self, target: str) -> str | None:
        value = target.strip().strip("<>")
        if not value:
            return None
        if value.startswith("http://") or value.startswith("https://"):
            return None
        value = value.split("#", 1)[0]
        value = value.split("?", 1)[0]
        return value.strip() or None

    def _resolve_include_globs(
        self, args: DocumentCrossrefMemoryArgs
    ) -> list[str]:
        if args.include_globs:
            return self._normalize_globs(args.include_globs, [])

        if args.extension and args.extensions:
            raise ToolError("Provide extension or extensions, not both.")

        ext_list: list[str] = []
        if args.extension:
            ext_list = [args.extension]
        elif args.extensions:
            ext_list = args.extensions
        else:
            ext_list = self.config.default_extensions

        normalized = self._normalize_extensions(ext_list)
        if not normalized:
            raise ToolError("Provide extension(s) or include_globs.")

        return [f"**/*{ext}" for ext in normalized]

    def _normalize_extensions(self, ext_list: list[str]) -> list[str]:
        normalized: list[str] = []
        for ext in ext_list:
            value = (ext or "").strip()
            if not value:
                continue
            if not value.startswith("."):
                value = f".{value}"
            normalized.append(value.lower())
        return sorted(set(normalized))

    def _normalize_globs(
        self, value: list[str] | None, defaults: list[str]
    ) -> list[str]:
        globs = value if value is not None else defaults
        globs = [g.strip() for g in globs if g and g.strip()]
        if not globs:
            return []
        return globs

    def _gather_files(
        self,
        paths: list[str],
        include_globs: list[str],
        exclude_globs: list[str],
        max_files: int,
    ) -> list[Path]:
        discovered: list[Path] = []
        seen: set[str] = set()

        for raw in paths:
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = self.config.effective_workdir / path
            path = path.resolve()

            if path.is_dir():
                for pattern in include_globs:
                    for file_path in path.glob(pattern):
                        if not file_path.is_file():
                            continue
                        rel = file_path.relative_to(path).as_posix()
                        if self._is_excluded(rel, exclude_globs):
                            continue
                        key = str(file_path)
                        if key in seen:
                            continue
                        seen.add(key)
                        discovered.append(file_path)
                        if len(discovered) >= max_files:
                            return discovered
            elif path.is_file():
                key = str(path)
                if key not in seen:
                    if not self._is_excluded(path.name, exclude_globs):
                        seen.add(key)
                        discovered.append(path)
                        if len(discovered) >= max_files:
                            return discovered
            else:
                raise ToolError(f"Path not found: {path}")

        return discovered

    def _is_excluded(self, rel_path: str, exclude_globs: list[str]) -> bool:
        if not exclude_globs:
            return False
        return any(fnmatch.fnmatch(rel_path, pattern) for pattern in exclude_globs)

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, DocumentCrossrefMemoryArgs):
            return ToolCallDisplay(summary="document_crossref_memory")

        summary = f"document_crossref_memory: {event.args.action}"
        return ToolCallDisplay(
            summary=summary,
            details={
                "action": event.args.action,
                "paths": event.args.paths,
                "extension": event.args.extension,
                "extensions": event.args.extensions,
                "include_globs": event.args.include_globs,
                "exclude_globs": event.args.exclude_globs,
                "replace_existing": event.args.replace_existing,
                "query_text": event.args.query_text,
                "path": event.args.path,
                "top_k": event.args.top_k,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, DocumentCrossrefMemoryResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        result = event.result
        if result.action == "index":
            message = (
                f"Indexed {result.indexed_files} file(s), "
                f"{result.reference_count} reference(s)"
            )
            if result.skipped_files:
                message += f", skipped {result.skipped_files}"
        elif result.action == "query":
            message = f"Found {len(result.matches)} cross-reference(s)"
        elif result.action == "stats":
            message = (
                f"Memory store has {result.document_count} document(s), "
                f"{result.reference_count} reference(s)"
            )
        else:
            message = "Cleared document memory store"

        warnings = result.errors[:]
        return ToolResultDisplay(
            success=not bool(result.errors),
            message=message,
            warnings=warnings,
            details={
                "action": result.action,
                "indexed_files": result.indexed_files,
                "skipped_files": result.skipped_files,
                "reference_count": result.reference_count,
                "document_count": result.document_count,
                "matches": result.matches,
                "errors": result.errors,
                "db_path": result.db_path,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Cross-referencing documents"
