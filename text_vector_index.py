from __future__ import annotations

import fnmatch
import json
from array import array
from datetime import datetime
from pathlib import Path
import sqlite3
from typing import TYPE_CHECKING, ClassVar, Iterable
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


DEFAULT_DB_PATH = Path.home() / ".vibe" / "vectorstores" / "pdf_vectors.sqlite"


class TextVectorIndexConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    db_path: Path = Field(
        default=DEFAULT_DB_PATH,
        description="Path to the sqlite database file.",
    )
    ollama_url: str = Field(
        default="http://127.0.0.1:11434",
        description="Base URL for the Ollama/GPT-OSS server.",
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        description="Embedding model to use with Ollama/GPT-OSS.",
    )
    max_source_bytes: int = Field(
        default=5_000_000, description="Maximum text bytes per file."
    )
    max_total_bytes: int = Field(
        default=20_000_000, description="Maximum total bytes across files."
    )
    max_chunk_bytes: int = Field(
        default=200_000, description="Maximum bytes per chunk."
    )
    chunk_unit: str = Field(default="lines", description="chars or lines.")
    chunk_size: int = Field(default=200, description="Chunk size in units.")
    overlap: int = Field(default=20, description="Overlap in units.")
    max_chunks_per_doc: int = Field(
        default=500, description="Maximum chunks per file."
    )
    max_files: int = Field(default=1000, description="Maximum files to index.")
    default_extensions: list[str] = Field(
        default=[
            ".txt",
            ".md",
            ".rst",
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".csv",
            ".html",
            ".css",
            ".ini",
            ".cfg",
        ],
        description="Default extensions to index.",
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


class TextVectorIndexState(BaseToolState):
    pass


class TextVectorIndexArgs(BaseModel):
    paths: list[str] = Field(description="List of files or directories.")
    db_path: str | None = Field(default=None, description="Override database path.")
    embedding_model: str | None = Field(
        default=None, description="Override embedding model."
    )
    chunk_unit: str | None = Field(default=None, description="chars or lines.")
    chunk_size: int | None = Field(default=None, description="Chunk size in units.")
    overlap: int | None = Field(default=None, description="Overlap in units.")
    max_chunks_per_doc: int | None = Field(
        default=None, description="Override max chunks per file."
    )
    max_files: int | None = Field(default=None, description="Override max files.")
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
    replace_existing: bool = Field(default=False, description="Rebuild existing rows.")


class TextVectorIndexResult(BaseModel):
    indexed_files: int
    indexed_chunks: int
    skipped_files: int
    errors: list[str]
    db_path: str


class TextVectorIndex(
    BaseTool[
        TextVectorIndexArgs,
        TextVectorIndexResult,
        TextVectorIndexConfig,
        TextVectorIndexState,
    ],
    ToolUIData[TextVectorIndexArgs, TextVectorIndexResult],
):
    description: ClassVar[str] = (
        "Index text files into a local vector database using Ollama/GPT-OSS embeddings."
    )

    async def run(self, args: TextVectorIndexArgs) -> TextVectorIndexResult:
        if not args.paths:
            raise ToolError("At least one path is required.")

        db_path = Path(args.db_path).expanduser() if args.db_path else self.config.db_path
        embedding_model = args.embedding_model or self.config.embedding_model
        chunk_unit = (args.chunk_unit or self.config.chunk_unit).strip().lower()
        chunk_size = args.chunk_size or self.config.chunk_size
        overlap = args.overlap if args.overlap is not None else self.config.overlap
        max_chunks_per_doc = (
            args.max_chunks_per_doc
            if args.max_chunks_per_doc is not None
            else self.config.max_chunks_per_doc
        )
        max_files = args.max_files if args.max_files is not None else self.config.max_files

        self._validate_chunking(chunk_unit, chunk_size, overlap)
        if max_chunks_per_doc <= 0:
            raise ToolError("max_chunks_per_doc must be a positive integer.")
        if max_files <= 0:
            raise ToolError("max_files must be a positive integer.")

        include_globs = self._resolve_include_globs(args)
        exclude_globs = self._normalize_globs(
            args.exclude_globs, self.config.default_exclude_globs
        )

        text_files = self._gather_files(args.paths, include_globs, exclude_globs, max_files)
        if not text_files:
            raise ToolError("No text files found.")

        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        try:
            self._init_db(conn)
            indexed_files = 0
            indexed_chunks = 0
            skipped_files = 0
            errors: list[str] = []
            total_bytes = 0

            for file_path in text_files:
                try:
                    stat = file_path.stat()
                    if not args.replace_existing and self._is_up_to_date(
                        conn, file_path, stat.st_mtime_ns, stat.st_size
                    ):
                        skipped_files += 1
                        continue

                    content = file_path.read_text("utf-8", errors="ignore")
                    if not content.strip():
                        errors.append(f"{file_path}: empty file")
                        continue

                    text_bytes = len(content.encode("utf-8"))
                    if text_bytes > self.config.max_source_bytes:
                        raise ToolError(
                            f"{file_path} exceeds max_source_bytes ({text_bytes} > {self.config.max_source_bytes})."
                        )
                    if total_bytes + text_bytes > self.config.max_total_bytes:
                        errors.append("max_total_bytes reached; stopping early.")
                        break

                    if args.replace_existing:
                        self._delete_existing(conn, file_path)

                    total_bytes += text_bytes
                    chunks = self._chunk_text(
                        content,
                        chunk_unit,
                        chunk_size,
                        overlap,
                        max_chunks_per_doc,
                    )

                    for chunk_index, (chunk_text, start, end, unit) in enumerate(
                        chunks, start=1
                    ):
                        embedding = self._embed_text(embedding_model, chunk_text)
                        embedding_blob = self._pack_embedding(embedding)
                        self._insert_chunk(
                            conn,
                            file_path,
                            chunk_index,
                            unit,
                            start,
                            end,
                            chunk_text,
                            embedding_blob,
                            len(embedding),
                            stat.st_mtime_ns,
                            stat.st_size,
                        )
                        indexed_chunks += 1

                    indexed_files += 1
                    conn.commit()
                except ToolError as exc:
                    errors.append(str(exc))
                except Exception as exc:
                    errors.append(f"{file_path}: {exc}")

            return TextVectorIndexResult(
                indexed_files=indexed_files,
                indexed_chunks=indexed_chunks,
                skipped_files=skipped_files,
                errors=errors,
                db_path=str(db_path),
            )
        finally:
            conn.close()

    def _validate_chunking(self, unit: str, size: int, overlap: int) -> None:
        if unit not in {"chars", "lines"}:
            raise ToolError("chunk_unit must be 'chars' or 'lines'.")
        if size <= 0:
            raise ToolError("chunk_size must be a positive integer.")
        if overlap < 0:
            raise ToolError("overlap must be a non-negative integer.")
        if overlap >= size:
            raise ToolError("overlap must be smaller than chunk_size.")

    def _resolve_include_globs(self, args: TextVectorIndexArgs) -> list[str]:
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
            ext_list = list(self.config.default_extensions)

        normalized = self._normalize_extensions(ext_list)
        if not normalized:
            raise ToolError("No extensions provided.")
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

    def _init_db(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pdf_chunks (
                id INTEGER PRIMARY KEY,
                source_path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                unit TEXT NOT NULL,
                start_index INTEGER,
                end_index INTEGER,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL,
                source_mtime INTEGER NOT NULL,
                source_size INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pdf_chunks_source ON pdf_chunks(source_path)"
        )

    def _is_up_to_date(
        self, conn: sqlite3.Connection, path: Path, mtime: int, size: int
    ) -> bool:
        row = conn.execute(
            "SELECT COUNT(*) FROM pdf_chunks WHERE source_path = ? AND source_mtime = ? AND source_size = ?",
            (str(path), mtime, size),
        ).fetchone()
        return bool(row and row[0] > 0)

    def _delete_existing(self, conn: sqlite3.Connection, path: Path) -> None:
        conn.execute("DELETE FROM pdf_chunks WHERE source_path = ?", (str(path),))

    def _insert_chunk(
        self,
        conn: sqlite3.Connection,
        path: Path,
        chunk_index: int,
        unit: str,
        start: int | None,
        end: int | None,
        content: str,
        embedding_blob: bytes,
        embedding_dim: int,
        mtime: int,
        size: int,
    ) -> None:
        conn.execute(
            """
            INSERT INTO pdf_chunks (
                source_path,
                chunk_index,
                unit,
                start_index,
                end_index,
                content,
                embedding,
                embedding_dim,
                source_mtime,
                source_size,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(path),
                chunk_index,
                unit,
                start,
                end,
                content,
                sqlite3.Binary(embedding_blob),
                embedding_dim,
                mtime,
                size,
                datetime.utcnow().isoformat(),
            ),
        )

    def _chunk_text(
        self,
        content: str,
        unit: str,
        size: int,
        overlap: int,
        max_chunks: int,
    ) -> Iterable[tuple[str, int | None, int | None, str]]:
        if unit == "chars":
            step = size - overlap
            index = 0
            chunk_index = 0
            length = len(content)
            while index < length and chunk_index < max_chunks:
                chunk_text = content[index : index + size]
                if not chunk_text:
                    break
                self._check_chunk_bytes(chunk_text)
                start = index
                end = index + len(chunk_text) - 1
                chunk_index += 1
                yield chunk_text, start, end, "chars"
                index += step
            return

        lines = content.splitlines()
        step = size - overlap
        index = 0
        chunk_index = 0
        while index < len(lines) and chunk_index < max_chunks:
            subset = lines[index : index + size]
            if not subset:
                break
            chunk_text = "\n".join(subset)
            self._check_chunk_bytes(chunk_text)
            start = index + 1
            end = start + len(subset) - 1
            chunk_index += 1
            yield chunk_text, start, end, "lines"
            index += step

    def _check_chunk_bytes(self, content: str) -> None:
        size = len(content.encode("utf-8"))
        if size > self.config.max_chunk_bytes:
            raise ToolError(
                f"Chunk exceeds max_chunk_bytes ({size} > {self.config.max_chunk_bytes})."
            )

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

    def _pack_embedding(self, embedding: list[float]) -> bytes:
        arr = array("f", embedding)
        return arr.tobytes()

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, TextVectorIndexArgs):
            return ToolCallDisplay(summary="text_vector_index")

        summary = f"text_vector_index: {len(event.args.paths)} path(s)"
        return ToolCallDisplay(
            summary=summary,
            details={
                "paths": event.args.paths,
                "chunk_unit": event.args.chunk_unit,
                "chunk_size": event.args.chunk_size,
                "overlap": event.args.overlap,
                "max_chunks_per_doc": event.args.max_chunks_per_doc,
                "replace_existing": event.args.replace_existing,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, TextVectorIndexResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = (
            f"Indexed {event.result.indexed_files} file(s), "
            f"{event.result.indexed_chunks} chunk(s)"
        )
        if event.result.skipped_files:
            message += f", skipped {event.result.skipped_files}"

        warnings = event.result.errors[:]
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=warnings,
            details={
                "db_path": event.result.db_path,
                "indexed_files": event.result.indexed_files,
                "indexed_chunks": event.result.indexed_chunks,
                "skipped_files": event.result.skipped_files,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Indexing text vectors"
