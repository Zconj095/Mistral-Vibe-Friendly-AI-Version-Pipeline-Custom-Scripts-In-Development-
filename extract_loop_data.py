from __future__ import annotations

import ast
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


INDENT_LANGS = {"python", "py"}
LOOP_HEADER_RE = re.compile(r"\b(foreach|for|while)\b\s*\(([^)]*)\)")
DO_RE = re.compile(r"\bdo\b")
ASSIGN_RE = re.compile(
    r"\b([A-Za-z_]\w*)\s*(?:=|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=)"
)
INC_RE = re.compile(r"\b([A-Za-z_]\w*)\s*(?:\+\+|--)")
ATTR_ACCESS_RE = re.compile(r"\b([A-Za-z_]\w*)(?:\.|->)([A-Za-z_]\w*)")
ATTR_ASSIGN_RE = re.compile(
    r"\b([A-Za-z_]\w*)(?:\.|->)([A-Za-z_]\w*)\s*"
    r"(?:=|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=)"
)
SUBSCRIPT_ACCESS_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\[")
SUBSCRIPT_ASSIGN_RE = re.compile(
    r"\b([A-Za-z_]\w*)\s*\[[^\]]*\]\s*"
    r"(?:=|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=)"
)
IDENT_RE = re.compile(r"\b[A-Za-z_]\w*\b")
KEYWORDS = {
    "if",
    "for",
    "foreach",
    "while",
    "switch",
    "case",
    "break",
    "continue",
    "return",
    "else",
    "do",
    "try",
    "catch",
    "finally",
    "throw",
    "new",
    "class",
    "struct",
    "interface",
    "enum",
    "public",
    "private",
    "protected",
    "static",
    "const",
    "let",
    "var",
    "val",
    "function",
    "func",
    "fn",
    "def",
    "async",
    "await",
    "yield",
    "in",
    "of",
    "true",
    "false",
    "null",
    "nil",
    "this",
    "self",
    "super",
}


@dataclass
class _ScanState:
    in_block_comment: bool = False
    in_string: str | None = None


@dataclass
class _LoopEntry:
    loop_id: int
    kind: str
    start_line: int
    end_line: int | None
    depth: int
    parent_id: int | None
    target_vars: set[str]
    assigned_vars: set[str]
    read_vars: set[str]
    data_reads: set[str]
    data_writes: set[str]
    start_depth: int | None = None


class ExtractLoopDataConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_input_bytes: int = Field(
        default=5_000_000, description="Maximum input size in bytes."
    )
    max_loops: int = Field(
        default=200, description="Maximum number of loops to return."
    )
    max_shared_vars: int = Field(
        default=200, description="Maximum shared variables to return."
    )
    max_depth: int = Field(
        default=12, description="Maximum loop depth to include."
    )


class ExtractLoopDataState(BaseToolState):
    pass


class ExtractLoopDataArgs(BaseModel):
    content: str | None = Field(default=None, description="Raw code to analyze.")
    path: str | None = Field(default=None, description="Path to a code file.")
    language: str | None = Field(default=None, description="Language hint.")
    max_depth: int | None = Field(default=None, description="Override max loop depth.")
    max_loops: int | None = Field(
        default=None, description="Override the configured max loops limit."
    )
    max_shared_vars: int | None = Field(
        default=None, description="Override the configured max shared vars limit."
    )


class LoopData(BaseModel):
    index: int
    kind: str
    start_line: int
    end_line: int
    depth: int
    parent_index: int | None
    target_vars: list[str]
    assigned_vars: list[str]
    read_vars: list[str]
    external_vars: list[str]
    data_reads: list[str]
    data_writes: list[str]


class LoopVariableLink(BaseModel):
    name: str
    loops: list[int]


class ExtractLoopDataResult(BaseModel):
    mode: str
    loops: list[LoopData]
    shared_variables: list[LoopVariableLink]
    count: int
    shared_count: int
    truncated: bool
    shared_truncated: bool


class ExtractLoopData(
    BaseTool[
        ExtractLoopDataArgs,
        ExtractLoopDataResult,
        ExtractLoopDataConfig,
        ExtractLoopDataState,
    ],
    ToolUIData[ExtractLoopDataArgs, ExtractLoopDataResult],
):
    description: ClassVar[str] = (
        "Extract loop-scoped data to understand nested data inside and across loops."
    )

    async def run(self, args: ExtractLoopDataArgs) -> ExtractLoopDataResult:
        content = self._load_content(args)
        if not content:
            return ExtractLoopDataResult(
                mode="auto",
                loops=[],
                shared_variables=[],
                count=0,
                shared_count=0,
                truncated=False,
                shared_truncated=False,
            )

        max_depth = args.max_depth if args.max_depth is not None else self.config.max_depth
        if max_depth <= 0:
            raise ToolError("max_depth must be a positive integer.")

        max_loops = args.max_loops if args.max_loops is not None else self.config.max_loops
        if max_loops <= 0:
            raise ToolError("max_loops must be a positive integer.")

        max_shared = (
            args.max_shared_vars
            if args.max_shared_vars is not None
            else self.config.max_shared_vars
        )
        if max_shared <= 0:
            raise ToolError("max_shared_vars must be a positive integer.")

        language = (args.language or "").strip().lower()
        if language in INDENT_LANGS:
            mode = "python"
            loops = self._parse_python(content)
        else:
            mode = "generic"
            loops = self._parse_generic(content)

        loops = [loop for loop in loops if loop.depth <= max_depth]
        loops = self._normalize_loops(loops)

        truncated = len(loops) > max_loops
        if truncated:
            loops = loops[:max_loops]

        shared_variables, shared_truncated = self._build_shared_variables(
            loops, max_shared
        )

        return ExtractLoopDataResult(
            mode=mode,
            loops=loops,
            shared_variables=shared_variables,
            count=len(loops),
            shared_count=len(shared_variables),
            truncated=truncated,
            shared_truncated=shared_truncated,
        )

    def _load_content(self, args: ExtractLoopDataArgs) -> str:
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
        except ValueError as exc:
            raise ToolError("Security error: cannot resolve the provided path.") from exc

        if not resolved.exists():
            raise ToolError(f"File not found at: {resolved}")
        if resolved.is_dir():
            raise ToolError(f"Path is a directory, not a file: {resolved}")
        return resolved

    def _parse_python(self, content: str) -> list[_LoopEntry]:
        try:
            tree = ast.parse(content)
        except SyntaxError as exc:
            raise ToolError(f"Python parse error: {exc}") from exc

        loops: list[_LoopEntry] = []
        loop_stack: list[_LoopEntry] = []
        loop_id = 0

        def push_loop(kind: str, node: ast.AST) -> _LoopEntry:
            nonlocal loop_id
            loop_id += 1
            entry = _LoopEntry(
                loop_id=loop_id,
                kind=kind,
                start_line=getattr(node, "lineno", 0),
                end_line=None,
                depth=len(loop_stack) + 1,
                parent_id=loop_stack[-1].loop_id if loop_stack else None,
                target_vars=set(),
                assigned_vars=set(),
                read_vars=set(),
                data_reads=set(),
                data_writes=set(),
            )
            loops.append(entry)
            loop_stack.append(entry)
            return entry

        def pop_loop(entry: _LoopEntry, node: ast.AST) -> None:
            entry.end_line = getattr(node, "end_lineno", entry.start_line)
            if loop_stack:
                loop_stack.pop()

        def record_targets(target: ast.AST, entry: _LoopEntry) -> None:
            for name in self._extract_target_names(target):
                entry.target_vars.add(name)

        class Visitor(ast.NodeVisitor):
            def visit_For(self, node: ast.For) -> None:
                entry = push_loop("for", node)
                record_targets(node.target, entry)
                self.generic_visit(node)
                pop_loop(entry, node)

            def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
                entry = push_loop("async_for", node)
                record_targets(node.target, entry)
                self.generic_visit(node)
                pop_loop(entry, node)

            def visit_While(self, node: ast.While) -> None:
                entry = push_loop("while", node)
                self.generic_visit(node)
                pop_loop(entry, node)

            def visit_Name(self, node: ast.Name) -> None:
                if loop_stack:
                    for entry in loop_stack:
                        if isinstance(node.ctx, ast.Load):
                            entry.read_vars.add(node.id)
                        else:
                            entry.assigned_vars.add(node.id)
                self.generic_visit(node)

            def visit_Attribute(self, node: ast.Attribute) -> None:
                if loop_stack:
                    path = self._format_access(node)
                    if path:
                        for entry in loop_stack:
                            if isinstance(node.ctx, ast.Load):
                                entry.data_reads.add(path)
                            else:
                                entry.data_writes.add(path)
                self.generic_visit(node)

            def visit_Subscript(self, node: ast.Subscript) -> None:
                if loop_stack:
                    path = self._format_access(node)
                    if path:
                        for entry in loop_stack:
                            if isinstance(node.ctx, ast.Load):
                                entry.data_reads.add(path)
                            else:
                                entry.data_writes.add(path)
                self.generic_visit(node)

        Visitor().visit(tree)
        last_line = content.count("\n") + 1
        for entry in loops:
            if entry.end_line is None:
                entry.end_line = last_line
        return loops

    def _parse_generic(self, content: str) -> list[_LoopEntry]:
        loops: list[_LoopEntry] = []
        loop_id = 0
        stack: list[_LoopEntry] = []
        pending: _LoopEntry | None = None
        brace_depth = 0
        state = _ScanState()
        lines = content.splitlines()

        for idx, line in enumerate(lines, start=1):
            sanitized, state = self._scan_line(line, state, hash_comments=True)
            stripped = sanitized.strip()

            if stripped and not pending:
                if match := LOOP_HEADER_RE.search(stripped):
                    kind = match.group(1).lower()
                    header = match.group(2)
                    loop_id += 1
                    pending = _LoopEntry(
                        loop_id=loop_id,
                        kind=kind,
                        start_line=idx,
                        end_line=None,
                        depth=len(stack) + 1,
                        parent_id=stack[-1].loop_id if stack else None,
                        target_vars=set(),
                        assigned_vars=set(),
                        read_vars=set(),
                        data_reads=set(),
                        data_writes=set(),
                    )
                    pending.target_vars.update(self._extract_header_targets(header))
                    pending.read_vars.update(
                        self._extract_header_reads(header, pending.target_vars)
                    )
                elif DO_RE.match(stripped):
                    loop_id += 1
                    pending = _LoopEntry(
                        loop_id=loop_id,
                        kind="do",
                        start_line=idx,
                        end_line=None,
                        depth=len(stack) + 1,
                        parent_id=stack[-1].loop_id if stack else None,
                        target_vars=set(),
                        assigned_vars=set(),
                        read_vars=set(),
                        data_reads=set(),
                        data_writes=set(),
                    )

            active = list(stack)
            if pending and "{" in sanitized:
                active.append(pending)
            if pending and stripped.endswith(";") and "(" in sanitized:
                active.append(pending)

            if active:
                assigned, read_vars, data_reads, data_writes = self._extract_line_data(
                    sanitized
                )
                for entry in active:
                    entry.assigned_vars.update(assigned)
                    entry.read_vars.update(read_vars)
                    entry.data_reads.update(data_reads)
                    entry.data_writes.update(data_writes)

            for ch in sanitized:
                if ch == "{":
                    brace_depth += 1
                    if pending:
                        pending.start_depth = brace_depth
                        stack.append(pending)
                        pending = None
                        continue
                elif ch == "}":
                    brace_depth = max(brace_depth - 1, 0)
                    while stack and stack[-1].start_depth is not None:
                        if brace_depth < (stack[-1].start_depth or 0):
                            entry = stack.pop()
                            entry.end_line = idx
                            loops.append(entry)
                            continue
                        break

            if pending and "{" not in sanitized and stripped.endswith(";"):
                pending.end_line = idx
                loops.append(pending)
                pending = None

        last_line = len(lines)
        if pending:
            pending.end_line = last_line
            loops.append(pending)
        while stack:
            entry = stack.pop()
            entry.end_line = last_line
            loops.append(entry)

        return loops

    def _extract_target_names(self, node: ast.AST) -> set[str]:
        if isinstance(node, ast.Name):
            return {node.id}
        if isinstance(node, (ast.Tuple, ast.List)):
            names: set[str] = set()
            for elt in node.elts:
                names.update(self._extract_target_names(elt))
            return names
        return set()

    def _format_access(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._format_access(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        if isinstance(node, ast.Subscript):
            base = self._format_access(node.value)
            return f"{base}[]" if base else "[]"
        if isinstance(node, ast.Call):
            return self._format_access(node.func)
        return None

    def _extract_header_targets(self, header: str) -> set[str]:
        left = header
        if ":" in header:
            left = header.split(":", 1)[0]
        elif " in " in header:
            left = header.split(" in ", 1)[0]
        elif " of " in header:
            left = header.split(" of ", 1)[0]
        elif ";" in header:
            left = header.split(";", 1)[0]

        targets = {match.group(1) for match in ASSIGN_RE.finditer(left)}
        targets.update(match.group(1) for match in INC_RE.finditer(left))
        if targets:
            return {name for name in targets if name.lower() not in KEYWORDS}

        idents = [
            name
            for name in IDENT_RE.findall(left)
            if name.lower() not in KEYWORDS
        ]
        return {idents[-1]} if idents else set()

    def _extract_header_reads(self, header: str, targets: set[str]) -> set[str]:
        idents = {
            name
            for name in IDENT_RE.findall(header)
            if name.lower() not in KEYWORDS
        }
        return {name for name in idents if name not in targets}

    def _extract_line_data(
        self, line: str
    ) -> tuple[set[str], set[str], set[str], set[str]]:
        assigned = {match.group(1) for match in ASSIGN_RE.finditer(line)}
        assigned.update(match.group(1) for match in INC_RE.finditer(line))
        assigned = {name for name in assigned if name.lower() not in KEYWORDS}

        names = {
            name for name in IDENT_RE.findall(line) if name.lower() not in KEYWORDS
        }
        read_vars = {name for name in names if name not in assigned}

        data_reads = {
            f"{match.group(1)}.{match.group(2)}"
            for match in ATTR_ACCESS_RE.finditer(line)
        }
        data_writes = {
            f"{match.group(1)}.{match.group(2)}"
            for match in ATTR_ASSIGN_RE.finditer(line)
        }
        data_reads.update(
            f"{match.group(1)}[]" for match in SUBSCRIPT_ACCESS_RE.finditer(line)
        )
        data_writes.update(
            f"{match.group(1)}[]" for match in SUBSCRIPT_ASSIGN_RE.finditer(line)
        )

        return assigned, read_vars, data_reads, data_writes

    def _normalize_loops(self, loops: list[_LoopEntry]) -> list[LoopData]:
        ordered = sorted(loops, key=lambda item: (item.start_line, item.end_line or 0))
        id_map: dict[int, int] = {}
        for idx, entry in enumerate(ordered, start=1):
            id_map[entry.loop_id] = idx

        normalized: list[LoopData] = []
        for entry in ordered:
            parent_index = id_map.get(entry.parent_id) if entry.parent_id else None
            read_vars = sorted(entry.read_vars)
            assigned_vars = sorted(entry.assigned_vars)
            target_vars = sorted(entry.target_vars)
            external_vars = sorted(
                set(read_vars) - set(assigned_vars) - set(target_vars)
            )
            normalized.append(
                LoopData(
                    index=id_map[entry.loop_id],
                    kind=entry.kind,
                    start_line=entry.start_line,
                    end_line=entry.end_line or entry.start_line,
                    depth=entry.depth,
                    parent_index=parent_index,
                    target_vars=target_vars,
                    assigned_vars=assigned_vars,
                    read_vars=read_vars,
                    external_vars=external_vars,
                    data_reads=sorted(entry.data_reads),
                    data_writes=sorted(entry.data_writes),
                )
            )
        return normalized

    def _build_shared_variables(
        self, loops: list[LoopData], max_shared: int
    ) -> tuple[list[LoopVariableLink], bool]:
        usage: dict[str, set[int]] = {}
        for loop in loops:
            combined = set(loop.target_vars) | set(loop.assigned_vars) | set(loop.read_vars)
            for name in combined:
                usage.setdefault(name, set()).add(loop.index)

        links = [
            LoopVariableLink(name=name, loops=sorted(list(indices)))
            for name, indices in usage.items()
            if len(indices) > 1
        ]
        links.sort(key=lambda item: (-len(item.loops), item.name))
        truncated = len(links) > max_shared
        if truncated:
            links = links[:max_shared]
        return links, truncated

    def _scan_line(
        self, line: str, state: _ScanState, *, hash_comments: bool
    ) -> tuple[str, _ScanState]:
        result: list[str] = []
        i = 0
        length = len(line)
        while i < length:
            ch = line[i]
            next_two = line[i : i + 2]

            if state.in_block_comment:
                if next_two == "*/":
                    state.in_block_comment = False
                    i += 2
                    continue
                i += 1
                continue

            if state.in_string:
                if ch == "\\":
                    i += 2
                    continue
                if ch == state.in_string:
                    state.in_string = None
                i += 1
                continue

            if next_two == "/*":
                state.in_block_comment = True
                i += 2
                continue
            if next_two == "//":
                break
            if hash_comments and ch == "#":
                break
            if ch in {"'", '"', "`"}:
                state.in_string = ch
                i += 1
                continue

            result.append(ch)
            i += 1

        return "".join(result), state

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ExtractLoopDataArgs):
            return ToolCallDisplay(summary="extract_loop_data")

        return ToolCallDisplay(
            summary="extract_loop_data",
            details={
                "path": event.args.path,
                "language": event.args.language,
                "max_depth": event.args.max_depth,
                "max_loops": event.args.max_loops,
                "max_shared_vars": event.args.max_shared_vars,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ExtractLoopDataResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = f"Extracted {event.result.count} loop(s)"
        warnings: list[str] = []
        if event.result.truncated:
            warnings.append("Loop list truncated by max_loops limit")
        if event.result.shared_truncated:
            warnings.append("Shared variables truncated by max_shared_vars limit")

        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=warnings,
            details={
                "mode": event.result.mode,
                "count": event.result.count,
                "shared_count": event.result.shared_count,
                "truncated": event.result.truncated,
                "shared_truncated": event.result.shared_truncated,
                "loops": event.result.loops,
                "shared_variables": event.result.shared_variables,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Extracting loop data"
