from __future__ import annotations

from pathlib import Path
from typing import Any


def list_directory(
    path: str = ".",
    max_entries: int = 200,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    base = _resolve_base_dir(base_dir)
    target = _resolve_path(path, base)
    if not target.exists():
        raise ValueError(f"Directory not found: {target}")
    if not target.is_dir():
        raise ValueError(f"Path is not a directory: {target}")

    entries = sorted(p.name for p in target.iterdir())
    truncated = False
    if max_entries > 0 and len(entries) > max_entries:
        entries = entries[:max_entries]
        truncated = True

    return {
        "path": str(target),
        "entries": entries,
        "count": len(entries),
        "truncated": truncated,
    }


def read_text(
    path: str,
    max_bytes: int = 64000,
    encoding: str = "utf-8",
    base_dir: Path | None = None,
) -> dict[str, Any]:
    base = _resolve_base_dir(base_dir)
    target = _resolve_path(path, base)
    if not target.exists():
        raise ValueError(f"File not found: {target}")
    if target.is_dir():
        raise ValueError(f"Path is a directory: {target}")

    if max_bytes is None or max_bytes <= 0:
        max_bytes = 0

    truncated = False
    read_size = max_bytes + 1 if max_bytes else -1
    with target.open("rb") as handle:
        data = handle.read(read_size)
    if max_bytes and len(data) > max_bytes:
        data = data[:max_bytes]
        truncated = True

    text = data.decode(encoding, errors="replace")
    return {
        "path": str(target),
        "content": text,
        "bytes_read": len(data),
        "truncated": truncated,
    }


def _resolve_base_dir(base_dir: Path | None) -> Path:
    if base_dir is None:
        return Path.cwd().resolve()
    return Path(base_dir).resolve()


def _resolve_path(path: str, base: Path) -> Path:
    if not path:
        raise ValueError("Path cannot be empty.")
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = base / target
    target = target.resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise ValueError(
            f"Path must stay within project root: {base}"
        ) from exc
    return target
