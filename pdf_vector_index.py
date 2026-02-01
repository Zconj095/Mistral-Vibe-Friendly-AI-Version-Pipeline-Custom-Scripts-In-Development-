from __future__ import annotations

import json
import os
from array import array
from datetime import datetime
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
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


class PdfVectorIndexConfig(BaseToolConfig):
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
        default=5_000_000, description="Maximum text bytes per PDF."
    )
    max_total_bytes: int = Field(
        default=20_000_000, description="Maximum total bytes across PDFs."
    )
    max_chunk_bytes: int = Field(
        default=200_000, description="Maximum bytes per chunk."
    )
    chunk_unit: str = Field(default="chars", description="chars or lines.")
    chunk_size: int = Field(default=2000, description="Chunk size in units.")
    overlap: int = Field(default=200, description="Overlap in units.")
    max_chunks_per_doc: int = Field(
        default=500, description="Maximum chunks per PDF."
    )


class PdfVectorIndexState(BaseToolState):
    pass


class PdfVectorIndexArgs(BaseModel):
    paths: list[str] = Field(description="List of PDF files or directories.")
    db_path: str | None = Field(default=None, description="Override database path.")
    embedding_model: str | None = Field(default=None, description="Override embedding model.")
    chunk_unit: str | None = Field(default=None, description="chars or lines.")
    chunk_size: int | None = Field(default=None, description="Chunk size in units.")
    overlap: int | None = Field(default=None, description="Overlap in units.")
    max_chunks_per_doc: int | None = Field(default=None, description="Override max chunks per PDF.")
    max_pages: int | None = Field(default=None, description="Optional maximum pages to extract.")
    replace_existing: bool = Field(default=False, description="Rebuild entries for existing PDFs.")


class PdfVectorIndexResult(BaseModel):
    indexed_files: int
    indexed_chunks: int
    skipped_files: int
    errors: list[str]
    db_path: str


class PdfVectorIndex(
    BaseTool[PdfVectorIndexArgs, PdfVectorIndexResult, PdfVectorIndexConfig, PdfVectorIndexState],
    ToolUIData[PdfVectorIndexArgs, PdfVectorIndexResult],
):
    description: ClassVar[str] = (
        "Index PDF files into a local vector database using Ollama/GPT-OSS embeddings."
    )

    async def run(self, args: PdfVectorIndexArgs) -> PdfVectorIndexResult:
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

        self._validate_chunking(chunk_unit, chunk_size, overlap)
        if max_chunks_per_doc <= 0:
            raise ToolError("max_chunks_per_doc must be a positive integer.")

        pdf_files = self._gather_pdf_files(args.paths)
        if not pdf_files:
            raise ToolError("No PDF files found.")

        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        try:
            self._init_db(conn)
            indexed_files = 0
            indexed_chunks = 0
            skipped_files = 0
            errors: list[str] = []
            total_bytes = 0

            for pdf_path in pdf_files:
                try:
                    stat = pdf_path.stat()
                    if not args.replace_existing and self._is_up_to_date(
                        conn, pdf_path, stat.st_mtime_ns, stat.st_size
                    ):
                        skipped_files += 1
                        continue

                    text = self._extract_text(pdf_path, args.max_pages)
                    text_bytes = len(text.encode("utf-8"))
                    if text_bytes > self.config.max_source_bytes:
                        raise ToolError(
                            f"{pdf_path} exceeds max_source_bytes ({text_bytes} > {self.config.max_source_bytes})."
                        )

                    if total_bytes + text_bytes > self.config.max_total_bytes:
                        break

                    if args.replace_existing:
                        self._delete_existing(conn, pdf_path)

                    total_bytes += text_bytes
                    chunks = self._chunk_text(
                        text,
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
                            pdf_path,
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
                    errors.append(f"{pdf_path}: {exc}")

            return PdfVectorIndexResult(
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

    def _gather_pdf_files(self, paths: list[str]) -> list[Path]:
        pdfs: list[Path] = []
        for raw in paths:
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = self.config.effective_workdir / path
            path = path.resolve()
            if path.is_dir():
                pdfs.extend(sorted(path.rglob("*.pdf")))
            elif path.is_file() and path.suffix.lower() == ".pdf":
                pdfs.append(path)
        return pdfs

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

    def _extract_text(self, path: Path, max_pages: int | None) -> str:
        text = self._extract_text_with_pypdf(path, max_pages)
        if text is not None:
            return text

        text = self._extract_text_with_pdftotext(path, max_pages)
        if text is not None:
            return text

        raise ToolError(
            "No PDF text extractor available. Install 'pypdf' or add 'pdftotext' to PATH."
        )

    def _extract_text_with_pypdf(self, path: Path, max_pages: int | None) -> str | None:
        try:
            import pypdf
        except ModuleNotFoundError:
            return None

        try:
            reader = pypdf.PdfReader(str(path))
        except Exception as exc:
            raise ToolError(f"Failed to open PDF: {exc}") from exc

        pages = reader.pages
        limit = len(pages) if max_pages is None else min(max_pages, len(pages))
        chunks: list[str] = []
        for page_index in range(limit):
            chunks.append(pages[page_index].extract_text() or "")
        return "\n".join(chunks)

    def _extract_text_with_pdftotext(self, path: Path, max_pages: int | None) -> str | None:
        if not shutil.which("pdftotext"):
            return None

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                out_path = Path(tmp_dir) / "out.txt"
                cmd = ["pdftotext", "-layout"]
                if max_pages is not None:
                    cmd.extend(["-f", "1", "-l", str(max_pages)])
                cmd.extend([str(path), str(out_path)])
                proc = subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if proc.stderr:
                    stderr = proc.stderr.strip()
                    if stderr:
                        raise ToolError(f"pdftotext error: {stderr}")
                return out_path.read_text("utf-8", errors="ignore")
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else "pdftotext failed."
            raise ToolError(stderr) from exc
        except OSError as exc:
            raise ToolError(f"pdftotext failed: {exc}") from exc

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
        if not isinstance(event.args, PdfVectorIndexArgs):
            return ToolCallDisplay(summary="pdf_vector_index")

        summary = f"pdf_vector_index: {len(event.args.paths)} path(s)"
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
        if not isinstance(event.result, PdfVectorIndexResult):
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
        return "Indexing PDF vectors"
