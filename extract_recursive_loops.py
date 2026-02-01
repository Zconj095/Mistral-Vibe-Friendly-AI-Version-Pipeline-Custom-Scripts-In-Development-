
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


INDENT_EXTS = {".py", ".pyi"}
INDENT_LANGS = {"python", "py"}
LOOP_HEADER_RE = re.compile(r"\b(foreach|for|while)\b\s*\(([^)]*)\)")
DO_RE = re.compile(r"\bdo\b")
FUNC_KW_RE = re.compile(r"\b(func|fn|function|def)\s+([A-Za-z_]\w*)")
FUNC_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\([^;{]*\)\s*(?:\{|$)")
CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
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
class _FunctionEntry:
    func_id: int
    name: str
    simple_name: str
    start_line: int
    end_line: int | None
    calls: set[str]
    start_depth: int | None = None


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
    called_functions: set[str]
    start_depth: int | None = None

class ExtractRecursiveLoopsConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_input_bytes: int = Field(
        default=5_000_000, description="Maximum input size in bytes."
    )
    max_loops: int = Field(default=200, description="Maximum loops to return.")
    max_functions: int = Field(default=200, description="Maximum functions to return.")
    max_recursive_groups: int = Field(
        default=100, description="Maximum recursive groups to return."
    )
    max_depth: int = Field(default=12, description="Maximum loop depth to include.")


class ExtractRecursiveLoopsState(BaseToolState):
    pass


class ExtractRecursiveLoopsArgs(BaseModel):
    content: str | None = Field(default=None, description="Raw code to analyze.")
    path: str | None = Field(default=None, description="Path to a code file.")
    language: str | None = Field(default=None, description="Language hint.")
    max_depth: int | None = Field(default=None, description="Override max loop depth.")
    max_loops: int | None = Field(default=None, description="Override max loops.")
    max_functions: int | None = Field(
        default=None, description="Override max functions."
    )
    max_recursive_groups: int | None = Field(
        default=None, description="Override max recursive groups."
    )

class FunctionInfo(BaseModel):
    index: int
    name: str
    start_line: int
    end_line: int
    recursive: bool
    group_index: int | None
    calls: list[str]


class RecursiveGroup(BaseModel):
    index: int
    functions: list[str]
    size: int


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
    called_functions: list[str]
    recursive_functions: list[str]
    recursive_group_indexes: list[int]


class LoopRecursionLink(BaseModel):
    group_index: int
    loop_indexes: list[int]


class ExtractRecursiveLoopsResult(BaseModel):
    mode: str
    loops: list[LoopData]
    functions: list[FunctionInfo]
    recursive_groups: list[RecursiveGroup]
    loop_recursion_links: list[LoopRecursionLink]
    count: int
    function_count: int
    recursive_group_count: int
    truncated: bool
    function_truncated: bool
    recursion_truncated: bool


class ExtractRecursiveLoops(
    BaseTool[
        ExtractRecursiveLoopsArgs,
        ExtractRecursiveLoopsResult,
        ExtractRecursiveLoopsConfig,
        ExtractRecursiveLoopsState,
    ],
    ToolUIData[ExtractRecursiveLoopsArgs, ExtractRecursiveLoopsResult],
):
    description: ClassVar[str] = (
        "Analyze nested loops and intertwined recursion groups with loop data access."
    )

    async def run(self, args: ExtractRecursiveLoopsArgs) -> ExtractRecursiveLoopsResult:
        content = self._load_content(args)
        if not content:
            return ExtractRecursiveLoopsResult(
                mode="auto",
                loops=[],
                functions=[],
                recursive_groups=[],
                loop_recursion_links=[],
                count=0,
                function_count=0,
                recursive_group_count=0,
                truncated=False,
                function_truncated=False,
                recursion_truncated=False,
            )

        max_depth = args.max_depth if args.max_depth is not None else self.config.max_depth
        if max_depth <= 0:
            raise ToolError("max_depth must be a positive integer.")

        max_loops = args.max_loops if args.max_loops is not None else self.config.max_loops
        if max_loops <= 0:
            raise ToolError("max_loops must be a positive integer.")

        max_functions = (
            args.max_functions if args.max_functions is not None else self.config.max_functions
        )
        if max_functions <= 0:
            raise ToolError("max_functions must be a positive integer.")

        max_groups = (
            args.max_recursive_groups
            if args.max_recursive_groups is not None
            else self.config.max_recursive_groups
        )
        if max_groups <= 0:
            raise ToolError("max_recursive_groups must be a positive integer.")

        mode = self._resolve_mode(args)
        if mode == "python":
            loops, functions = self._parse_python(content)
        else:
            loops, functions = self._parse_generic(content)

        loops = [loop for loop in loops if loop.depth <= max_depth]
        functions_sorted = self._normalize_functions(functions, content)
        loops_sorted = self._normalize_loops(loops)

        function_truncated = len(functions_sorted) > max_functions
        if function_truncated:
            functions_sorted = functions_sorted[:max_functions]

        recursive_groups, recursion_map, recursion_truncated = self._build_recursive_groups(
            functions_sorted, max_groups
        )

        recursion_name_map = self._build_recursive_name_map(functions_sorted, recursion_map)

        truncated = len(loops_sorted) > max_loops
        if truncated:
            loops_sorted = loops_sorted[:max_loops]

        loops_enriched = self._annotate_loops_with_recursion(
            loops_sorted, recursion_name_map
        )
        loop_links = self._build_loop_recursion_links(loops_enriched)

        return ExtractRecursiveLoopsResult(
            mode=mode,
            loops=loops_enriched,
            functions=functions_sorted,
            recursive_groups=recursive_groups,
            loop_recursion_links=loop_links,
            count=len(loops_enriched),
            function_count=len(functions_sorted),
            recursive_group_count=len(recursive_groups),
            truncated=truncated,
            function_truncated=function_truncated,
            recursion_truncated=recursion_truncated,
        )

    def _load_content(self, args: ExtractRecursiveLoopsArgs) -> str:
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

    def _resolve_mode(self, args: ExtractRecursiveLoopsArgs) -> str:
        language = (args.language or "").strip().lower()
        if language in INDENT_LANGS:
            return "python"

        if args.path:
            ext = Path(args.path).suffix.lower()
            if ext in INDENT_EXTS:
                return "python"

        return "generic"

    def _parse_python(self, content: str) -> tuple[list[_LoopEntry], list[_FunctionEntry]]:
        try:
            tree = ast.parse(content)
        except SyntaxError as exc:
            raise ToolError(f"Python parse error: {exc}") from exc

        loops: list[_LoopEntry] = []
        functions: list[_FunctionEntry] = []
        loop_stack: list[_LoopEntry] = []
        function_stack: list[_FunctionEntry] = []
        loop_id = 0
        func_id = 0

        def push_loop(kind: str, node: ast.AST, parent: _LoopEntry | None) -> _LoopEntry:
            nonlocal loop_id
            loop_id += 1
            entry = _LoopEntry(
                loop_id=loop_id,
                kind=kind,
                start_line=getattr(node, "lineno", 0),
                end_line=None,
                depth=len(loop_stack) + 1,
                parent_id=parent.loop_id if parent else None,
                target_vars=set(),
                assigned_vars=set(),
                read_vars=set(),
                data_reads=set(),
                data_writes=set(),
                called_functions=set(),
            )
            loops.append(entry)
            loop_stack.append(entry)
            return entry

        def pop_loop(entry: _LoopEntry, node: ast.AST) -> None:
            entry.end_line = getattr(node, "end_lineno", entry.start_line)
            if loop_stack:
                loop_stack.pop()

        def push_function(name: str, node: ast.AST) -> _FunctionEntry:
            nonlocal func_id
            func_id += 1
            scoped = self._scoped_name(name, function_stack)
            entry = _FunctionEntry(
                func_id=func_id,
                name=scoped,
                simple_name=name,
                start_line=getattr(node, "lineno", 0),
                end_line=None,
                calls=set(),
            )
            functions.append(entry)
            function_stack.append(entry)
            return entry

        def pop_function(entry: _FunctionEntry, node: ast.AST) -> None:
            entry.end_line = getattr(node, "end_lineno", entry.start_line)
            if function_stack:
                function_stack.pop()

        def record_targets(target: ast.AST, entry: _LoopEntry) -> None:
            for name in self._extract_target_names(target):
                entry.target_vars.add(name)

        class Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                entry = push_function(node.name, node)
                self.generic_visit(node)
                pop_function(entry, node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                entry = push_function(node.name, node)
                self.generic_visit(node)
                pop_function(entry, node)

            def visit_For(self, node: ast.For) -> None:
                parent = loop_stack[-1] if loop_stack else None
                entry = push_loop("for", node, parent)
                record_targets(node.target, entry)
                self.generic_visit(node)
                pop_loop(entry, node)

            def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
                parent = loop_stack[-1] if loop_stack else None
                entry = push_loop("async_for", node, parent)
                record_targets(node.target, entry)
                self.generic_visit(node)
                pop_loop(entry, node)

            def visit_While(self, node: ast.While) -> None:
                parent = loop_stack[-1] if loop_stack else None
                entry = push_loop("while", node, parent)
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

            def visit_Call(self, node: ast.Call) -> None:
                name = self._call_name(node.func)
                if name and function_stack:
                    function_stack[-1].calls.add(name)
                if name and loop_stack:
                    for entry in loop_stack:
                        entry.called_functions.add(name)
                self.generic_visit(node)

        Visitor().visit(tree)

        last_line = content.count("\n") + 1
        for entry in loops:
            if entry.end_line is None:
                entry.end_line = last_line
        for entry in functions:
            if entry.end_line is None:
                entry.end_line = last_line

        return loops, functions

    def _parse_generic(self, content: str) -> tuple[list[_LoopEntry], list[_FunctionEntry]]:
        loops: list[_LoopEntry] = []
        functions: list[_FunctionEntry] = []
        loop_stack: list[_LoopEntry] = []
        function_stack: list[_FunctionEntry] = []
        pending_loop: _LoopEntry | None = None
        pending_function: _FunctionEntry | None = None
        brace_depth = 0
        state = _ScanState()
        loop_id = 0
        func_id = 0
        lines = content.splitlines()

        for idx, line in enumerate(lines, start=1):
            sanitized, state = self._scan_line(line, state, hash_comments=True)
            stripped = sanitized.strip()
            if not stripped:
                continue

            func_match = FUNC_KW_RE.search(stripped)
            func_name = None
            if func_match:
                func_name = func_match.group(2)
            else:
                func_match = FUNC_RE.search(stripped)
                if func_match:
                    candidate = func_match.group(1)
                    if candidate.lower() not in KEYWORDS:
                        func_name = candidate

            if func_name:
                func_id += 1
                scoped = self._scoped_name(func_name, function_stack)
                pending_function = _FunctionEntry(
                    func_id=func_id,
                    name=scoped,
                    simple_name=func_name,
                    start_line=idx,
                    end_line=None,
                    calls=set(),
                )
                functions.append(pending_function)

            if match := LOOP_HEADER_RE.search(stripped):
                loop_id += 1
                kind = match.group(1).lower()
                header = match.group(2)
                pending_loop = _LoopEntry(
                    loop_id=loop_id,
                    kind=kind,
                    start_line=idx,
                    end_line=None,
                    depth=len(loop_stack) + 1,
                    parent_id=loop_stack[-1].loop_id if loop_stack else None,
                    target_vars=set(),
                    assigned_vars=set(),
                    read_vars=set(),
                    data_reads=set(),
                    data_writes=set(),
                    called_functions=set(),
                )
                pending_loop.target_vars.update(self._extract_header_targets(header))
                pending_loop.read_vars.update(
                    self._extract_header_reads(header, pending_loop.target_vars)
                )
            elif DO_RE.match(stripped):
                loop_id += 1
                pending_loop = _LoopEntry(
                    loop_id=loop_id,
                    kind="do",
                    start_line=idx,
                    end_line=None,
                    depth=len(loop_stack) + 1,
                    parent_id=loop_stack[-1].loop_id if loop_stack else None,
                    target_vars=set(),
                    assigned_vars=set(),
                    read_vars=set(),
                    data_reads=set(),
                    data_writes=set(),
                    called_functions=set(),
                )

            active_loops = list(loop_stack)
            if pending_loop and "{" in sanitized:
                active_loops.append(pending_loop)
            if pending_loop and stripped.endswith(";") and "(" in sanitized:
                active_loops.append(pending_loop)

            active_function = function_stack[-1] if function_stack else None
            if pending_function and "{" in sanitized:
                active_function = pending_function

            calls = self._extract_calls(sanitized, func_name)
            if active_function and calls:
                active_function.calls.update(calls)
            if active_loops and calls:
                for loop_entry in active_loops:
                    loop_entry.called_functions.update(calls)

            if active_loops:
                assigned, read_vars, data_reads, data_writes = self._extract_line_data(
                    sanitized
                )
                for loop_entry in active_loops:
                    loop_entry.assigned_vars.update(assigned)
                    loop_entry.read_vars.update(read_vars)
                    loop_entry.data_reads.update(data_reads)
                    loop_entry.data_writes.update(data_writes)

            for ch in sanitized:
                if ch == "{":
                    brace_depth += 1
                    if pending_function:
                        pending_function.start_depth = brace_depth
                        function_stack.append(pending_function)
                        pending_function = None
                        continue
                    if pending_loop:
                        pending_loop.start_depth = brace_depth
                        loop_stack.append(pending_loop)
                        pending_loop = None
                        continue
                elif ch == "}":
                    brace_depth = max(brace_depth - 1, 0)
                    while function_stack and function_stack[-1].start_depth is not None:
                        if brace_depth < (function_stack[-1].start_depth or 0):
                            entry = function_stack.pop()
                            entry.end_line = idx
                            continue
                        break
                    while loop_stack and loop_stack[-1].start_depth is not None:
                        if brace_depth < (loop_stack[-1].start_depth or 0):
                            entry = loop_stack.pop()
                            entry.end_line = idx
                            continue
                        break

            if pending_loop and "{" not in sanitized and stripped.endswith(";"):
                pending_loop.end_line = idx
                loops.append(pending_loop)
                pending_loop = None

            if pending_function and "{" not in sanitized and stripped.endswith(";"):
                pending_function.end_line = idx
                pending_function = None

        last_line = len(lines)
        if pending_loop:
            pending_loop.end_line = last_line
            loops.append(pending_loop)
        while loop_stack:
            entry = loop_stack.pop()
            entry.end_line = last_line
            loops.append(entry)

        if pending_function:
            pending_function.end_line = last_line
        while function_stack:
            entry = function_stack.pop()
            entry.end_line = last_line

        return loops, functions

    def _normalize_functions(
        self, functions: list[_FunctionEntry], content: str
    ) -> list[FunctionInfo]:
        last_line = content.count("\n") + 1
        ordered = sorted(functions, key=lambda item: (item.start_line, item.end_line or 0))
        return [
            FunctionInfo(
                index=idx + 1,
                name=entry.name,
                start_line=entry.start_line,
                end_line=entry.end_line or last_line,
                recursive=False,
                group_index=None,
                calls=sorted(entry.calls),
            )
            for idx, entry in enumerate(ordered)
        ]

    def _normalize_loops(self, loops: list[_LoopEntry]) -> list[LoopData]:
        ordered = sorted(loops, key=lambda item: (item.start_line, item.end_line or 0))
        id_map = {entry.loop_id: idx + 1 for idx, entry in enumerate(ordered)}
        normalized: list[LoopData] = []
        for entry in ordered:
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
                    parent_index=id_map.get(entry.parent_id) if entry.parent_id else None,
                    target_vars=target_vars,
                    assigned_vars=assigned_vars,
                    read_vars=read_vars,
                    external_vars=external_vars,
                    data_reads=sorted(entry.data_reads),
                    data_writes=sorted(entry.data_writes),
                    called_functions=sorted(entry.called_functions),
                    recursive_functions=[],
                    recursive_group_indexes=[],
                )
            )
        return normalized

    def _build_recursive_groups(
        self, functions: list[FunctionInfo], max_groups: int
    ) -> tuple[list[RecursiveGroup], dict[int, int], bool]:
        if not functions:
            return [], {}, False

        name_map = {func.name: idx for idx, func in enumerate(functions)}
        simple_map: dict[str, list[int]] = {}
        for idx, func in enumerate(functions):
            simple = func.name.split(".")[-1]
            simple_map.setdefault(simple, []).append(idx)

        edges: dict[int, set[int]] = {idx: set() for idx in range(len(functions))}
        for idx, func in enumerate(functions):
            for call in func.calls:
                base = call.split(".")[-1]
                targets = set(simple_map.get(base, []))
                if call in name_map:
                    targets.add(name_map[call])
                edges[idx].update(targets)

        sccs = self._strongly_connected_components(edges)
        recursive_groups: list[RecursiveGroup] = []
        function_group: dict[int, int] = {}
        for group in sccs:
            if len(group) > 1:
                recursive_groups.append(
                    RecursiveGroup(
                        index=len(recursive_groups) + 1,
                        functions=[functions[i].name for i in group],
                        size=len(group),
                    )
                )
                for node in group:
                    function_group[node] = recursive_groups[-1].index
            elif len(group) == 1:
                node = group[0]
                if node in edges.get(node, set()):
                    recursive_groups.append(
                        RecursiveGroup(
                            index=len(recursive_groups) + 1,
                            functions=[functions[node].name],
                            size=1,
                        )
                    )
                    function_group[node] = recursive_groups[-1].index

        truncated = len(recursive_groups) > max_groups
        if truncated:
            recursive_groups = recursive_groups[:max_groups]

        allowed = {group.index for group in recursive_groups}
        function_group = {
            idx: group for idx, group in function_group.items() if group in allowed
        }

        for idx, func in enumerate(functions):
            group_index = function_group.get(idx)
            func.recursive = group_index is not None
            func.group_index = group_index

        return recursive_groups, function_group, truncated

    def _build_recursive_name_map(
        self, functions: list[FunctionInfo], function_group: dict[int, int]
    ) -> dict[str, set[int]]:
        name_map: dict[str, set[int]] = {}
        for idx, func in enumerate(functions):
            if func.group_index is None:
                continue
            simple = func.name.split(".")[-1]
            name_map.setdefault(simple, set()).add(func.group_index)
        return name_map

    def _annotate_loops_with_recursion(
        self, loops: list[LoopData], recursion_name_map: dict[str, set[int]]
    ) -> list[LoopData]:
        for loop in loops:
            recursive_functions: set[str] = set()
            recursive_groups: set[int] = set()
            for call in loop.called_functions:
                base = call.split(".")[-1]
                if base in recursion_name_map:
                    recursive_functions.add(base)
                    recursive_groups.update(recursion_name_map[base])
            loop.recursive_functions = sorted(recursive_functions)
            loop.recursive_group_indexes = sorted(recursive_groups)
        return loops

    def _build_loop_recursion_links(
        self, loops: list[LoopData]
    ) -> list[LoopRecursionLink]:
        group_map: dict[int, set[int]] = {}
        for loop in loops:
            for group_index in loop.recursive_group_indexes:
                group_map.setdefault(group_index, set()).add(loop.index)
        links = [
            LoopRecursionLink(group_index=group, loop_indexes=sorted(list(indexes)))
            for group, indexes in sorted(group_map.items())
        ]
        return links

    def _strongly_connected_components(
        self, edges: dict[int, set[int]]
    ) -> list[list[int]]:
        index = 0
        stack: list[int] = []
        indices: dict[int, int] = {}
        lowlinks: dict[int, int] = {}
        on_stack: set[int] = set()
        result: list[list[int]] = []

        def strongconnect(node: int) -> None:
            nonlocal index
            indices[node] = index
            lowlinks[node] = index
            index += 1
            stack.append(node)
            on_stack.add(node)

            for neighbor in edges.get(node, set()):
                if neighbor not in indices:
                    strongconnect(neighbor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
                elif neighbor in on_stack:
                    lowlinks[node] = min(lowlinks[node], indices[neighbor])

            if lowlinks[node] == indices[node]:
                component: list[int] = []
                while stack:
                    w = stack.pop()
                    on_stack.remove(w)
                    component.append(w)
                    if w == node:
                        break
                result.append(component)

        for node in edges:
            if node not in indices:
                strongconnect(node)

        return result

    def _scoped_name(self, name: str, stack: list[_FunctionEntry]) -> str:
        if not stack:
            return name
        parts = [entry.simple_name for entry in stack] + [name]
        return ".".join(parts)

    def _extract_target_names(self, node: ast.AST) -> set[str]:
        if isinstance(node, ast.Name):
            return {node.id}
        if isinstance(node, (ast.Tuple, ast.List)):
            names: set[str] = set()
            for elt in node.elts:
                names.update(self._extract_target_names(elt))
            return names
        return set()

    def _call_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return self._format_access(node)
        return None

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
            name for name in IDENT_RE.findall(left) if name.lower() not in KEYWORDS
        ]
        return {idents[-1]} if idents else set()

    def _extract_header_reads(self, header: str, targets: set[str]) -> set[str]:
        idents = {
            name for name in IDENT_RE.findall(header) if name.lower() not in KEYWORDS
        }
        return {name for name in idents if name not in targets}

    def _extract_line_data(
        self, line: str
    ) -> tuple[set[str], set[str], set[str], set[str]]:
        assigned = {match.group(1) for match in ASSIGN_RE.finditer(line)}
        assigned.update(match.group(1) for match in INC_RE.finditer(line))
        assigned = {name for name in assigned if name.lower() not in KEYWORDS}

        names = {name for name in IDENT_RE.findall(line) if name.lower() not in KEYWORDS}
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

    def _extract_calls(self, line: str, defined_name: str | None) -> set[str]:
        calls = {match.group(1) for match in CALL_RE.finditer(line)}
        calls = {name for name in calls if name.lower() not in KEYWORDS}
        if defined_name:
            calls.discard(defined_name)
        return calls

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
        if not isinstance(event.args, ExtractRecursiveLoopsArgs):
            return ToolCallDisplay(summary="extract_recursive_loops")

        return ToolCallDisplay(
            summary="extract_recursive_loops",
            details={
                "path": event.args.path,
                "language": event.args.language,
                "max_depth": event.args.max_depth,
                "max_loops": event.args.max_loops,
                "max_functions": event.args.max_functions,
                "max_recursive_groups": event.args.max_recursive_groups,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ExtractRecursiveLoopsResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = (
            f"Found {event.result.count} loop(s) and "
            f"{event.result.function_count} function(s)"
        )
        warnings: list[str] = []
        if event.result.truncated:
            warnings.append("Loop list truncated by max_loops limit")
        if event.result.function_truncated:
            warnings.append("Function list truncated by max_functions limit")
        if event.result.recursion_truncated:
            warnings.append("Recursive groups truncated by max_recursive_groups limit")

        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=warnings,
            details={
                "mode": event.result.mode,
                "count": event.result.count,
                "function_count": event.result.function_count,
                "recursive_group_count": event.result.recursive_group_count,
                "truncated": event.result.truncated,
                "function_truncated": event.result.function_truncated,
                "recursion_truncated": event.result.recursion_truncated,
                "loops": event.result.loops,
                "functions": event.result.functions,
                "recursive_groups": event.result.recursive_groups,
                "loop_recursion_links": event.result.loop_recursion_links,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Analyzing recursive loops"
