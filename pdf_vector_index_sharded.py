from __future__ import annotations

import json
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


DEFAULT_ROOT_DIR = Path.home() / ".vibe" / "vectorstores" / "pdf_shards"
DEFAULT_MANIFEST_PATH = Path.home() / ".vibe" / "vectorstores" / "pdf_vectors_manifest.json"


class PdfVectorIndexShardedConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    root_dir: Path = Field(
        default=DEFAULT_ROOT_DIR,
        description="Directory that stores shard databases.",
    )
    manifest_path: Path = Field(
        default=DEFAULT_MANIFEST_PATH,
        description="Path to the manifest JSON file.",
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
        default=700_000_000_000, description="Maximum total bytes across PDFs."
    )
    max_shard_bytes: int = Field(
        default=50_000_000_000, description="Maximum bytes per shard database."
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


class PdfVectorIndexShardedState(BaseToolState):
    pass


class PdfVectorIndexShardedArgs(BaseModel):
    paths: list[str] = Field(description="List of PDF files or directories.")
    root_dir: str | None = Field(default=None, description="Override shard root directory.")
    manifest_path: str | None = Field(default=None, description="Override manifest path.")
    embedding_model: str | None = Field(default=None, description="Override embedding model.")
    chunk_unit: str | None = Field(default=None, description="chars or lines.")
    chunk_size: int | None = Field(default=None, description="Chunk size in units.")
    overlap: int | None = Field(default=None, description="Overlap in units.")
    max_chunks_per_doc: int | None = Field(default=None, description="Override max chunks per PDF.")
    max_pages: int | None = Field(default=None, description="Optional maximum pages to extract.")
    replace_existing: bool = Field(default=False, description="Rebuild entries for existing PDFs.")


class PdfVectorIndexShardedResult(BaseModel):
    indexed_files: int
    indexed_chunks: int
    skipped_files: int
    shard_count: int
    truncated: bool
    errors: list[str]
    root_dir: str
    manifest_path: str


class PdfVectorIndexSharded(
    BaseTool[
        PdfVectorIndexShardedArgs,
        PdfVectorIndexShardedResult,
        PdfVectorIndexShardedConfig,
        PdfVectorIndexShardedState,
    ],
    ToolUIData[PdfVectorIndexShardedArgs, PdfVectorIndexShardedResult],
):
    description: ClassVar[str] = (
        "Index PDF files into a sharded vector database using Ollama/GPT-OSS embeddings."
    )

    async def run(self, args: PdfVectorIndexShardedArgs) -> PdfVectorIndexShardedResult:
        if not args.paths:
            raise ToolError("At least one path is required.")

        root_dir = (
            Path(args.root_dir).expanduser()
            if args.root_dir
            else self.config.root_dir
        )
        manifest_path = (
            Path(args.manifest_path).expanduser()
            if args.manifest_path
            else self.config.manifest_path
        )
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

        root_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        manifest = self._load_manifest(manifest_path)
        manifest = self._ensure_manifest(manifest, embedding_model)

        indexed_files = 0
        indexed_chunks = 0
        skipped_files = 0
        errors: list[str] = []
        truncated = False

        current_shard = self._select_shard_path(
            manifest, root_dir, self.config.max_shard_bytes
        )
        conn = self._open_shard(current_shard)
        try:
            for pdf_path in pdf_files:
                try:
                    stat = pdf_path.stat()
                    if args.replace_existing:
                        self._delete_existing(manifest, pdf_path)
                    else:
                        if self._is_up_to_date(manifest, pdf_path, stat.st_mtime_ns, stat.st_size):
                            skipped_files += 1
                            continue

                    text = self._extract_text(pdf_path, args.max_pages)
                    text_bytes = len(text.encode("utf-8"))
                    if text_bytes > self.config.max_source_bytes:
                        raise ToolError(
                            f"{pdf_path} exceeds max_source_bytes ({text_bytes} > {self.config.max_source_bytes})."
                        )

                    if self._total_shard_bytes(manifest) + text_bytes > self.config.max_total_bytes:
                        truncated = True
                        break

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
                        if self._shard_bytes(current_shard) >= self.config.max_shard_bytes:
                            conn.commit()
                            conn.close()
                            current_shard = self._create_new_shard(
                                manifest, root_dir
                            )
                            conn = self._open_shard(current_shard)

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

            conn.commit()
        finally:
            conn.close()

        self._refresh_manifest(manifest, manifest_path, embedding_model)
        shard_count = len(manifest.get("shards", []))

        return PdfVectorIndexShardedResult(
            indexed_files=indexed_files,
            indexed_chunks=indexed_chunks,
            skipped_files=skipped_files,
            shard_count=shard_count,
            truncated=truncated,
            errors=errors,
            root_dir=str(root_dir),
            manifest_path=str(manifest_path),
        )

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

    def _load_manifest(self, path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _ensure_manifest(self, manifest: dict, embedding_model: str) -> dict:
        if not manifest:
            return {"version": 1, "embedding_model": embedding_model, "shards": []}
        if manifest.get("embedding_model") and manifest.get("embedding_model") != embedding_model:
            raise ToolError(
                "Embedding model mismatch with manifest. "
                "Use the same embedding_model or create a new manifest."
            )
        manifest.setdefault("version", 1)
        manifest.setdefault("embedding_model", embedding_model)
        manifest.setdefault("shards", [])
        return manifest

    def _refresh_manifest(self, manifest: dict, path: Path, embedding_model: str) -> None:
        shards = manifest.get("shards", [])
        refreshed = []
        for shard in shards:
            shard_path = Path(shard["path"])
            if not shard_path.exists():
                continue
            bytes_size = shard_path.stat().st_size
            chunk_count, embedding_dim = self._count_chunks(shard_path)
            refreshed.append(
                {
                    "path": str(shard_path),
                    "bytes": bytes_size,
                    "chunks": chunk_count,
                    "embedding_dim": embedding_dim,
                    "updated_at": datetime.utcnow().isoformat(),
                }
            )
        manifest["embedding_model"] = embedding_model
        manifest["shards"] = refreshed
        path.write_text(json.dumps(manifest, indent=2), "utf-8")

    def _count_chunks(self, shard_path: Path) -> tuple[int, int]:
        with sqlite3.connect(str(shard_path)) as conn:
            row = conn.execute("SELECT COUNT(*), MAX(embedding_dim) FROM pdf_chunks").fetchone()
            if not row:
                return 0, 0
            return int(row[0] or 0), int(row[1] or 0)

    def _select_shard_path(
        self, manifest: dict, root_dir: Path, max_shard_bytes: int
    ) -> Path:
        shards = manifest.get("shards", [])
        if shards:
            last_path = Path(shards[-1]["path"])
            if last_path.exists() and last_path.stat().st_size < max_shard_bytes:
                return last_path
        return self._create_new_shard(manifest, root_dir)

    def _create_new_shard(self, manifest: dict, root_dir: Path) -> Path:
        shards = manifest.get("shards", [])
        index = len(shards) + 1
        name = f"shard_{index:04d}.sqlite"
        path = root_dir / name
        shards.append({"path": str(path)})
        manifest["shards"] = shards
        return path

    def _open_shard(self, path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(str(path))
        self._init_db(conn)
        return conn

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

    def _total_shard_bytes(self, manifest: dict) -> int:
        total = 0
        for shard in manifest.get("shards", []):
            path = Path(shard["path"])
            if path.exists():
                total += path.stat().st_size
        return total

    def _shard_bytes(self, path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:
            return 0

    def _delete_existing(self, manifest: dict, path: Path) -> None:
        for shard in manifest.get("shards", []):
            shard_path = Path(shard["path"])
            if not shard_path.exists():
                continue
            with sqlite3.connect(str(shard_path)) as conn:
                conn.execute(
                    "DELETE FROM pdf_chunks WHERE source_path = ?", (str(path),)
                )
                conn.commit()

    def _is_up_to_date(
        self, manifest: dict, path: Path, mtime: int, size: int
    ) -> bool:
        for shard in manifest.get("shards", []):
            shard_path = Path(shard["path"])
            if not shard_path.exists():
                continue
            with sqlite3.connect(str(shard_path)) as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM pdf_chunks WHERE source_path = ? AND source_mtime = ? AND source_size = ?",
                    (str(path), mtime, size),
                ).fetchone()
                if row and row[0] > 0:
                    return True
        return False

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
        if not isinstance(event.args, PdfVectorIndexShardedArgs):
            return ToolCallDisplay(summary="pdf_vector_index_sharded")

        summary = f"pdf_vector_index_sharded: {len(event.args.paths)} path(s)"
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
        if not isinstance(event.result, PdfVectorIndexShardedResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = (
            f"Indexed {event.result.indexed_files} file(s), "
            f"{event.result.indexed_chunks} chunk(s)"
        )
        if event.result.skipped_files:
            message += f", skipped {event.result.skipped_files}"
        if event.result.truncated:
            message += " (truncated)"

        warnings = event.result.errors[:]
        if event.result.truncated:
            warnings.append("Indexing stopped due to max_total_bytes limit")

        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=warnings,
            details={
                "root_dir": event.result.root_dir,
                "manifest_path": event.result.manifest_path,
                "indexed_files": event.result.indexed_files,
                "indexed_chunks": event.result.indexed_chunks,
                "skipped_files": event.result.skipped_files,
                "shard_count": event.result.shard_count,
                "truncated": event.result.truncated,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Indexing sharded PDF vectors"
