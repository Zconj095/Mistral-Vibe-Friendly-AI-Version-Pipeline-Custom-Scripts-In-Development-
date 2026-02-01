
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
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


ATX_RE = re.compile(r"^\s*(#{1,6})\s*(.+?)\s*#*\s*$")
SETEXT_RE = re.compile(r"^\s*(=+|-+)\s*$")
NUMBERED_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:[.)])?\s+(.+)$")
FENCE_RE = re.compile(r"^\s*(```+|~~~+)")


@dataclass
class _HeadingEntry:
    title: str
    depth: int
    start_line: int
    end_line: int | None = None
    parent_index: int | None = None


@dataclass
class _Group:
    root_index: int
    heading_indices: list[int]
    heading_titles: list[str]
    start_line: int
    end_line: int
    line_span: int
    max_depth: int
    density: float
    density_per_100_lines: float
    content: str


class ChunkTextStructuresConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_input_bytes: int = Field(
        default=5_000_000, description="Maximum input size in bytes."
    )
    max_chunk_bytes: int = Field(
        default=500_000, description="Maximum structure chunk size in bytes."
    )
    max_chunks: int = Field(
        default=200, description="Maximum number of structure chunks to return."
    )
    include_nested: bool = Field(
        default=True, description="Include nested headings in results."
    )
    include_singletons: bool = Field(
        default=True,
        description="Include sections with only a single heading (or none).",
    )
    include_markdown_headings: bool = Field(
        default=True, description="Detect Markdown # style headings."
    )
    include_setext_headings: bool = Field(
        default=True, description="Detect setext-style headings (---/===)."
    )
    include_numbered_headings: bool = Field(
        default=True, description="Detect numbered headings (e.g., 1.2 Title)."
    )


class ChunkTextStructuresState(BaseToolState):
    pass


class ChunkTextStructuresArgs(BaseModel):
    content: str | None = Field(default=None, description="Raw text to chunk.")
    path: str | None = Field(default=None, description="Path to a text file.")
    include_nested: bool | None = Field(
        default=None, description="Include nested headings."
    )
    include_singletons: bool | None = Field(
        default=None, description="Include single-heading sections."
    )
    include_markdown_headings: bool | None = Field(
        default=None, description="Detect Markdown # headings."
    )
    include_setext_headings: bool | None = Field(
        default=None, description="Detect setext headings."
    )
    include_numbered_headings: bool | None = Field(
        default=None, description="Detect numbered headings."
    )
    max_chunks: int | None = Field(
        default=None, description="Override the configured max chunks limit."
    )


class HeadingNode(BaseModel):
    index: int
    title: str
    depth: int
    relative_depth: int
    start_line: int
    end_line: int
    parent_index: int | None
    group_index: int


class TextStructureChunk(BaseModel):
    index: int
    heading_titles: list[str]
    heading_count: int
    start_line: int
    end_line: int
    line_span: int
    max_depth: int
    density: float
    density_per_100_lines: float
    content: str


class ChunkTextStructuresResult(BaseModel):
    include_nested: bool
    include_singletons: bool
    chunks: list[TextStructureChunk]
    headings: list[HeadingNode]
    count: int
    truncated: bool

class ChunkTextStructures(
    BaseTool[
        ChunkTextStructuresArgs,
        ChunkTextStructuresResult,
        ChunkTextStructuresConfig,
        ChunkTextStructuresState,
    ],
    ToolUIData[ChunkTextStructuresArgs, ChunkTextStructuresResult],
):
    description: ClassVar[str] = (
        "Chunk text by heading structures with depth and density metrics."
    )

    async def run(self, args: ChunkTextStructuresArgs) -> ChunkTextStructuresResult:
        content = self._load_content(args)
        include_nested = (
            args.include_nested
            if args.include_nested is not None
            else self.config.include_nested
        )
        include_singletons = (
            args.include_singletons
            if args.include_singletons is not None
            else self.config.include_singletons
        )
        include_markdown = (
            args.include_markdown_headings
            if args.include_markdown_headings is not None
            else self.config.include_markdown_headings
        )
        include_setext = (
            args.include_setext_headings
            if args.include_setext_headings is not None
            else self.config.include_setext_headings
        )
        include_numbered = (
            args.include_numbered_headings
            if args.include_numbered_headings is not None
            else self.config.include_numbered_headings
        )

        if not content:
            return ChunkTextStructuresResult(
                include_nested=include_nested,
                include_singletons=include_singletons,
                chunks=[],
                headings=[],
                count=0,
                truncated=False,
            )

        max_chunks = args.max_chunks if args.max_chunks is not None else self.config.max_chunks
        if max_chunks <= 0:
            raise ToolError("max_chunks must be a positive integer.")

        lines = content.splitlines()
        headings = self._extract_headings(
            lines,
            include_markdown=include_markdown,
            include_setext=include_setext,
            include_numbered=include_numbered,
        )

        if not headings:
            if include_singletons:
                chunk = self._build_singleton_chunk(lines)
                self._validate_chunk_sizes([chunk])
                return ChunkTextStructuresResult(
                    include_nested=include_nested,
                    include_singletons=include_singletons,
                    chunks=[chunk],
                    headings=[],
                    count=1,
                    truncated=False,
                )

            return ChunkTextStructuresResult(
                include_nested=include_nested,
                include_singletons=include_singletons,
                chunks=[],
                headings=[],
                count=0,
                truncated=False,
            )

        last_line = len(lines)
        self._assign_parents_and_ends(headings, last_line)
        root_depth = min(entry.depth for entry in headings)

        groups = self._build_groups(
            headings,
            lines,
            root_depth,
            include_nested=include_nested,
            include_singletons=include_singletons,
        )
        groups = sorted(groups, key=lambda item: (item.start_line, item.end_line))

        truncated = len(groups) > max_chunks
        if truncated:
            groups = groups[:max_chunks]

        chunks, nodes = self._finalize_groups(groups, headings, root_depth)
        self._validate_chunk_sizes(chunks)

        return ChunkTextStructuresResult(
            include_nested=include_nested,
            include_singletons=include_singletons,
            chunks=chunks,
            headings=nodes,
            count=len(chunks),
            truncated=truncated,
        )

    def _load_content(self, args: ChunkTextStructuresArgs) -> str:
        if args.content and args.path:
            raise ToolError("Provide content or path, not both.")
        if args.content is None and args.path is None:
            raise ToolError("Provide content or path.")

        if args.content is not None:
            data = args.content.encode("utf-8")
            self._validate_input_size(len(data))
            return args.content

        path = self._resolve_path(args.path or "")
        size = path.stat().st_size
        self._validate_input_size(size)
        return path.read_text("utf-8", errors="ignore")

    def _validate_input_size(self, size: int) -> None:
        if size > self.config.max_input_bytes:
            raise ToolError(
                f"Input is {size} bytes, which exceeds max_input_bytes "
                f"({self.config.max_input_bytes})."
            )

    def _resolve_path(self, raw_path: str) -> Path:
        if not raw_path.strip():
            raise ToolError("Path cannot be empty.")

        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.config.effective_workdir / path

        try:
            resolved = path.resolve()
        except OSError as exc:
            raise ToolError(f"Failed to resolve path: {exc}") from exc

        if not resolved.exists():
            raise ToolError(f"Path not found: {resolved}")
        if resolved.is_dir():
            raise ToolError(f"Path is a directory, not a file: {resolved}")
        return resolved

    def _validate_chunk_sizes(self, chunks: list[TextStructureChunk]) -> None:
        max_bytes = self.config.max_chunk_bytes
        for chunk in chunks:
            size = len(chunk.content.encode("utf-8"))
            if size > max_bytes:
                raise ToolError(
                    f"Chunk {chunk.index} exceeds max_chunk_bytes ({size} > {max_bytes})."
                )

    def _build_singleton_chunk(self, lines: list[str]) -> TextStructureChunk:
        if not lines:
            return TextStructureChunk(
                index=1,
                heading_titles=[],
                heading_count=0,
                start_line=1,
                end_line=1,
                line_span=1,
                max_depth=0,
                density=0.0,
                density_per_100_lines=0.0,
                content="",
            )

        start_line = 1
        end_line = len(lines)
        line_span = max(end_line - start_line + 1, 1)
        content = "\n".join(lines)
        return TextStructureChunk(
            index=1,
            heading_titles=[],
            heading_count=0,
            start_line=start_line,
            end_line=end_line,
            line_span=line_span,
            max_depth=0,
            density=0.0,
            density_per_100_lines=0.0,
            content=content,
        )

    def _extract_headings(
        self,
        lines: list[str],
        *,
        include_markdown: bool,
        include_setext: bool,
        include_numbered: bool,
    ) -> list[_HeadingEntry]:
        skip_lines = self._mark_fence_lines(lines)
        headings: list[_HeadingEntry] = []
        heading_lines: set[int] = set()

        for idx, line in enumerate(lines):
            if skip_lines[idx]:
                continue

            raw = line.rstrip("\r\n")
            if include_setext:
                if match := SETEXT_RE.match(raw):
                    prev_index = idx - 1
                    if prev_index >= 0 and not skip_lines[prev_index]:
                        title = lines[prev_index].strip()
                        if title and (prev_index + 1) not in heading_lines:
                            depth = 1 if match.group(1).startswith("=") else 2
                            headings.append(
                                _HeadingEntry(
                                    title=title,
                                    depth=depth,
                                    start_line=prev_index + 1,
                                )
                            )
                            heading_lines.add(prev_index + 1)
                    continue

            if include_markdown:
                if match := ATX_RE.match(raw):
                    title = match.group(2).strip()
                    if title and (idx + 1) not in heading_lines:
                        depth = len(match.group(1))
                        headings.append(
                            _HeadingEntry(title=title, depth=depth, start_line=idx + 1)
                        )
                        heading_lines.add(idx + 1)
                    continue

            if include_numbered:
                if match := NUMBERED_RE.match(raw):
                    title = match.group(2).strip()
                    if title and (idx + 1) not in heading_lines:
                        depth = len(match.group(1).split("."))
                        headings.append(
                            _HeadingEntry(title=title, depth=depth, start_line=idx + 1)
                        )
                        heading_lines.add(idx + 1)

        headings.sort(key=lambda item: item.start_line)
        return headings

    def _mark_fence_lines(self, lines: list[str]) -> list[bool]:
        skip = [False] * len(lines)
        in_fence = False
        fence_char: str | None = None

        for idx, line in enumerate(lines):
            if match := FENCE_RE.match(line):
                skip[idx] = True
                marker = match.group(1)
                marker_char = marker[0]
                if not in_fence:
                    in_fence = True
                    fence_char = marker_char
                elif fence_char == marker_char:
                    in_fence = False
                    fence_char = None
                continue

            if in_fence:
                skip[idx] = True

        return skip

    def _assign_parents_and_ends(self, headings: list[_HeadingEntry], last_line: int) -> None:
        stack: list[int] = []
        for idx, heading in enumerate(headings):
            while stack and heading.depth <= headings[stack[-1]].depth:
                prev = stack.pop()
                headings[prev].end_line = heading.start_line - 1

            heading.parent_index = stack[-1] if stack else None
            stack.append(idx)

        while stack:
            prev = stack.pop()
            headings[prev].end_line = last_line

    def _build_groups(
        self,
        headings: list[_HeadingEntry],
        lines: list[str],
        root_depth: int,
        *,
        include_nested: bool,
        include_singletons: bool,
    ) -> list[_Group]:
        groups: list[_Group] = []
        last_line = len(lines)
        root_indices = [
            idx for idx, entry in enumerate(headings) if entry.depth == root_depth
        ]

        for root_idx in root_indices:
            root = headings[root_idx]
            start_line = root.start_line
            end_line = root.end_line or last_line
            heading_indices = [
                idx
                for idx, entry in enumerate(headings)
                if start_line <= entry.start_line <= end_line
                and (include_nested or entry.depth == root_depth)
            ]

            if not include_singletons and len(heading_indices) < 2:
                continue

            heading_titles = [headings[idx].title for idx in heading_indices]
            line_span = max(end_line - start_line + 1, 1)

            if include_nested and heading_indices:
                max_depth = max(
                    headings[idx].depth - root_depth + 1 for idx in heading_indices
                )
            elif heading_indices:
                max_depth = 1
            else:
                max_depth = 0

            density = len(heading_indices) / line_span
            density_per_100 = density * 100.0
            content = "\n".join(lines[start_line - 1 : end_line])

            groups.append(
                _Group(
                    root_index=root_idx,
                    heading_indices=heading_indices,
                    heading_titles=heading_titles,
                    start_line=start_line,
                    end_line=end_line,
                    line_span=line_span,
                    max_depth=max_depth,
                    density=density,
                    density_per_100_lines=density_per_100,
                    content=content,
                )
            )

        return groups

    def _finalize_groups(
        self,
        groups: list[_Group],
        headings: list[_HeadingEntry],
        root_depth: int,
    ) -> tuple[list[TextStructureChunk], list[HeadingNode]]:
        chunks: list[TextStructureChunk] = []
        heading_group: dict[int, int] = {}
        included_indices: list[int] = []

        for group_index, group in enumerate(groups, start=1):
            chunks.append(
                TextStructureChunk(
                    index=group_index,
                    heading_titles=group.heading_titles,
                    heading_count=len(group.heading_titles),
                    start_line=group.start_line,
                    end_line=group.end_line,
                    line_span=group.line_span,
                    max_depth=group.max_depth,
                    density=group.density,
                    density_per_100_lines=group.density_per_100_lines,
                    content=group.content,
                )
            )
            for idx in group.heading_indices:
                heading_group[idx] = group_index
            included_indices.extend(group.heading_indices)

        unique_indices = sorted(set(included_indices), key=lambda idx: headings[idx].start_line)
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_indices, start=1)}

        nodes: list[HeadingNode] = []
        for old_idx in unique_indices:
            entry = headings[old_idx]
            parent_old = entry.parent_index
            parent_index = index_map.get(parent_old) if parent_old in index_map else None
            group_index = heading_group.get(old_idx)
            if group_index is None:
                continue

            relative_depth = entry.depth - root_depth + 1
            nodes.append(
                HeadingNode(
                    index=index_map[old_idx],
                    title=entry.title,
                    depth=entry.depth,
                    relative_depth=relative_depth,
                    start_line=entry.start_line,
                    end_line=entry.end_line or entry.start_line,
                    parent_index=parent_index,
                    group_index=group_index,
                )
            )

        return chunks, nodes

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ChunkTextStructuresArgs):
            return ToolCallDisplay(summary="chunk_text_structures")

        return ToolCallDisplay(
            summary="chunk_text_structures",
            details={
                "path": event.args.path,
                "include_nested": event.args.include_nested,
                "include_singletons": event.args.include_singletons,
                "include_markdown_headings": event.args.include_markdown_headings,
                "include_setext_headings": event.args.include_setext_headings,
                "include_numbered_headings": event.args.include_numbered_headings,
                "max_chunks": event.args.max_chunks,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ChunkTextStructuresResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = f"Found {event.result.count} text structure chunk(s)"
        warnings: list[str] = []
        if event.result.truncated:
            warnings.append("Chunk list truncated by max_chunks limit")

        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=warnings,
            details={
                "include_nested": event.result.include_nested,
                "include_singletons": event.result.include_singletons,
                "count": event.result.count,
                "truncated": event.result.truncated,
                "chunks": event.result.chunks,
                "headings": event.result.headings,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Chunking text structures"
