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


CONTROL_KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "foreach",
    "else",
    "do",
    "try",
}

CLASS_RE = re.compile(r"\b(class|struct|interface|enum)\s+([A-Za-z_]\w*)")
FUNC_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\([^;{]*\)\s*(?:\{|$)")
FUNC_KW_RE = re.compile(r"\b(func|fn|function)\s+([A-Za-z_]\w*)")
VAR_KEYWORD_RE = re.compile(r"\b(let|const|var|auto|final|val|var)\s+([A-Za-z_]\w*)")
TYPE_VAR_RE = re.compile(
    r"^\s*(?:[A-Za-z_][\w:<>\[\],*&\s]+)\s+([A-Za-z_]\w*)\s*(?:=|;|,)"
)
ATTR_RE = re.compile(r"\b(this|self|cls)\s*(?:\.|->)\s*([A-Za-z_]\w*)")
SENTENCE_STRIP_RE = re.compile(r"(['\"`]).*?\1")


@dataclass
class Scope:
    name: str
    kind: str
    brace_depth: int


class ExtractNestedVarsConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_snippets: int = Field(
        default=50, description="Maximum number of snippets to process."
    )
    max_snippet_bytes: int = Field(
        default=250_000, description="Maximum size per snippet."
    )
    max_total_bytes: int = Field(
        default=1_000_000, description="Maximum total size across all snippets."
    )
    max_depth: int = Field(
        default=8, description="Maximum scope depth to include in results."
    )


class ExtractNestedVarsState(BaseToolState):
    pass


class SnippetInput(BaseModel):
    id: str | None = Field(default=None, description="Optional snippet identifier.")
    language: str | None = Field(default=None, description="Language hint (e.g., python).")
    content: str | None = Field(default=None, description="Snippet content.")
    path: str | None = Field(default=None, description="Path to a file with code.")


class ExtractNestedVarsArgs(BaseModel):
    snippets: list[SnippetInput]
    include_params: bool = Field(default=True, description="Include function parameters.")
    include_locals: bool = Field(default=True, description="Include local variables.")
    include_fields: bool = Field(default=True, description="Include class/struct fields.")
    include_globals: bool = Field(default=True, description="Include module-level variables.")
    max_depth: int | None = Field(default=None, description="Override max scope depth.")


class NestedSymbol(BaseModel):
    name: str
    kind: str
    scope: str
    scope_path: list[str]
    snippet_id: str
    line: int
    column: int | None
    language: str | None
    owner: str | None = None


class ExtractNestedVarsResult(BaseModel):
    symbols: list[NestedSymbol]
    count: int
    snippet_count: int
    errors: list[str]


class ExtractNestedVars(
    BaseTool[
        ExtractNestedVarsArgs,
        ExtractNestedVarsResult,
        ExtractNestedVarsConfig,
        ExtractNestedVarsState,
    ],
    ToolUIData[ExtractNestedVarsArgs, ExtractNestedVarsResult],
):
    description: ClassVar[str] = (
        "Extract nested variables across classes/structs/functions from multiple snippets."
    )

    async def run(self, args: ExtractNestedVarsArgs) -> ExtractNestedVarsResult:
        if not args.snippets:
            raise ToolError("At least one snippet is required.")
        if len(args.snippets) > self.config.max_snippets:
            raise ToolError(
                f"Too many snippets: {len(args.snippets)} > {self.config.max_snippets}."
            )

        include_params = args.include_params
        include_locals = args.include_locals
        include_fields = args.include_fields
        include_globals = args.include_globals
        max_depth = args.max_depth if args.max_depth is not None else self.config.max_depth
        if max_depth <= 0:
            raise ToolError("max_depth must be a positive integer.")

        total_bytes = 0
        symbols: list[NestedSymbol] = []
        errors: list[str] = []

        for index, snippet in enumerate(args.snippets, start=1):
            snippet_id, language, content = self._load_snippet(snippet, index)
            content_bytes = len(content.encode("utf-8"))
            if content_bytes > self.config.max_snippet_bytes:
                raise ToolError(
                    f"Snippet '{snippet_id}' exceeds max_snippet_bytes "
                    f"({content_bytes} > {self.config.max_snippet_bytes})."
                )

            total_bytes += content_bytes
            if total_bytes > self.config.max_total_bytes:
                raise ToolError(
                    f"Total snippet size exceeds max_total_bytes "
                    f"({total_bytes} > {self.config.max_total_bytes})."
                )

            try:
                if self._is_python(language):
                    symbols.extend(
                        self._parse_python(
                            content,
                            snippet_id,
                            language,
                            include_params,
                            include_locals,
                            include_fields,
                            include_globals,
                            max_depth,
                        )
                    )
                else:
                    symbols.extend(
                        self._parse_generic(
                            content,
                            snippet_id,
                            language,
                            include_locals,
                            include_fields,
                            include_globals,
                            max_depth,
                        )
                    )
            except ToolError:
                raise
            except Exception as exc:
                errors.append(f"{snippet_id}: {exc}")

        return ExtractNestedVarsResult(
            symbols=symbols,
            count=len(symbols),
            snippet_count=len(args.snippets),
            errors=errors,
        )

    def _load_snippet(self, snippet: SnippetInput, index: int) -> tuple[str, str | None, str]:
        if snippet.content and snippet.path:
            raise ToolError("Each snippet must provide content or path, not both.")
        if not snippet.content and not snippet.path:
            raise ToolError("Each snippet must provide content or path.")

        language = snippet.language.strip().lower() if snippet.language else None
        if snippet.content is not None:
            snippet_id = snippet.id or f"snippet-{index}"
            return snippet_id, language, snippet.content

        path = self._resolve_path(snippet.path or "")
        snippet_id = snippet.id or str(path)
        return snippet_id, language, path.read_text("utf-8", errors="ignore")

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

    def _is_python(self, language: str | None) -> bool:
        if language is None:
            return False
        return language in {"python", "py"}

    def _parse_python(
        self,
        content: str,
        snippet_id: str,
        language: str | None,
        include_params: bool,
        include_locals: bool,
        include_fields: bool,
        include_globals: bool,
        max_depth: int,
    ) -> list[NestedSymbol]:
        try:
            tree = ast.parse(content)
        except SyntaxError as exc:
            raise ToolError(f"{snippet_id}: Python parse error: {exc}") from exc

        symbols: list[NestedSymbol] = []
        scope_stack: list[tuple[str, str]] = []

        def add_symbol(
            name: str,
            kind: str,
            node: ast.AST,
            owner: str | None = None,
        ) -> None:
            scope_path = [item[0] for item in scope_stack]
            if scope_path and max_depth > 0 and len(scope_path) > max_depth:
                scope_path = scope_path[-max_depth:]
            scope = ".".join(scope_path) if scope_path else "<module>"
            symbols.append(
                NestedSymbol(
                    name=name,
                    kind=kind,
                    scope=scope,
                    scope_path=scope_path,
                    snippet_id=snippet_id,
                    line=getattr(node, "lineno", 0),
                    column=getattr(node, "col_offset", None),
                    language=language,
                    owner=owner,
                )
            )

        def current_class() -> str | None:
            for scope_name, scope_kind in reversed(scope_stack):
                if scope_kind == "class":
                    return scope_name
            return None

        class Visitor(ast.NodeVisitor):
            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                add_symbol(node.name, "class", node)
                scope_stack.append((node.name, "class"))
                self.generic_visit(node)
                scope_stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                add_symbol(node.name, "function", node)
                scope_stack.append((node.name, "function"))
                if include_params:
                    for arg in node.args.args + node.args.kwonlyargs:
                        add_symbol(arg.arg, "param", arg)
                    if node.args.vararg:
                        add_symbol(node.args.vararg.arg, "param", node.args.vararg)
                    if node.args.kwarg:
                        add_symbol(node.args.kwarg.arg, "param", node.args.kwarg)
                self.generic_visit(node)
                scope_stack.pop()

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self.visit_FunctionDef(node)

            def visit_Assign(self, node: ast.Assign) -> None:
                self._handle_assignment(node.targets, node)
                self.generic_visit(node)

            def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
                targets = [node.target]
                self._handle_assignment(targets, node)
                self.generic_visit(node)

            def visit_AugAssign(self, node: ast.AugAssign) -> None:
                targets = [node.target]
                self._handle_assignment(targets, node)
                self.generic_visit(node)

            def visit_For(self, node: ast.For) -> None:
                self._handle_target(node.target, node)
                self.generic_visit(node)

            def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
                self.visit_For(node)

            def visit_With(self, node: ast.With) -> None:
                for item in node.items:
                    if item.optional_vars is not None:
                        self._handle_target(item.optional_vars, node)
                self.generic_visit(node)

            def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
                if include_locals and node.name:
                    add_symbol(node.name, "local", node)
                self.generic_visit(node)

            def _handle_assignment(self, targets: list[ast.expr], node: ast.AST) -> None:
                for target in targets:
                    self._handle_target(target, node)

            def _handle_target(self, target: ast.expr, node: ast.AST) -> None:
                if isinstance(target, ast.Name):
                    if scope_stack:
                        if include_locals:
                            add_symbol(target.id, "local", node)
                    else:
                        if include_globals:
                            add_symbol(target.id, "global", node)
                    return

                if isinstance(target, (ast.Tuple, ast.List)):
                    for elt in target.elts:
                        self._handle_target(elt, node)
                    return

                if isinstance(target, ast.Attribute) and isinstance(
                    target.value, ast.Name
                ):
                    if not include_fields:
                        return
                    owner = current_class()
                    if target.value.id in {"self", "cls"} and owner:
                        add_symbol(target.attr, "field", node, owner=owner)

        Visitor().visit(tree)
        return symbols

    def _parse_generic(
        self,
        content: str,
        snippet_id: str,
        language: str | None,
        include_locals: bool,
        include_fields: bool,
        include_globals: bool,
        max_depth: int,
    ) -> list[NestedSymbol]:
        symbols: list[NestedSymbol] = []
        scope_stack: list[Scope] = []
        brace_depth = 0
        pending_scope: tuple[str, str] | None = None

        def add_symbol(
            name: str,
            kind: str,
            line_no: int,
            column: int | None,
            owner: str | None = None,
        ) -> None:
            scope_path = [scope.name for scope in scope_stack]
            if scope_path and max_depth > 0 and len(scope_path) > max_depth:
                scope_path = scope_path[-max_depth:]
            scope = ".".join(scope_path) if scope_path else "<module>"
            symbols.append(
                NestedSymbol(
                    name=name,
                    kind=kind,
                    scope=scope,
                    scope_path=scope_path,
                    snippet_id=snippet_id,
                    line=line_no,
                    column=column,
                    language=language,
                    owner=owner,
                )
            )

        def current_class() -> str | None:
            for scope in reversed(scope_stack):
                if scope.kind in {"class", "struct"}:
                    return scope.name
            return None

        def current_function() -> str | None:
            for scope in reversed(scope_stack):
                if scope.kind == "function":
                    return scope.name
            return None

        lines = content.splitlines()
        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            sanitized = self._sanitize_line(line)
            if not sanitized:
                continue

            if pending_scope and "{" in sanitized:
                brace_depth += sanitized.count("{")
                scope_stack.append(
                    Scope(pending_scope[0], pending_scope[1], brace_depth)
                )
                pending_scope = None
            else:
                open_count = sanitized.count("{")
                close_count = sanitized.count("}")
                if open_count:
                    brace_depth += open_count
                if close_count:
                    brace_depth -= close_count
                    if brace_depth < 0:
                        brace_depth = 0

            class_match = CLASS_RE.search(sanitized)
            if class_match:
                kind = class_match.group(1)
                name = class_match.group(2)
                add_symbol(name, kind, idx, class_match.start(2))
                if "{" in sanitized:
                    scope_stack.append(Scope(name, kind, brace_depth))
                else:
                    pending_scope = (name, kind)
                continue

            func_kw_match = FUNC_KW_RE.search(sanitized)
            func_match = func_kw_match or FUNC_RE.search(sanitized)
            if func_match:
                name = func_match.group(2) if func_kw_match else func_match.group(1)
                if name not in CONTROL_KEYWORDS:
                    add_symbol(name, "function", idx, func_match.start(1))
                    if "{" in sanitized:
                        scope_stack.append(Scope(name, "function", brace_depth))
                    else:
                        pending_scope = (name, "function")

            attr_match = ATTR_RE.search(sanitized)
            if attr_match and include_fields:
                owner = current_class()
                if owner:
                    add_symbol(attr_match.group(2), "field", idx, attr_match.start(2), owner=owner)

            var_match = VAR_KEYWORD_RE.search(sanitized)
            if var_match:
                name = var_match.group(2)
                if current_function():
                    if include_locals:
                        add_symbol(name, "local", idx, var_match.start(2))
                else:
                    if include_globals:
                        add_symbol(name, "global", idx, var_match.start(2))

            if TYPE_VAR_RE.match(sanitized) and "(" not in sanitized:
                name_match = TYPE_VAR_RE.match(sanitized)
                if name_match:
                    name = name_match.group(1)
                    if current_class() and not current_function():
                        if include_fields:
                            add_symbol(name, "field", idx, name_match.start(1), owner=current_class())
                    elif current_function():
                        if include_locals:
                            add_symbol(name, "local", idx, name_match.start(1))
                    else:
                        if include_globals:
                            add_symbol(name, "global", idx, name_match.start(1))

            while scope_stack and brace_depth < scope_stack[-1].brace_depth:
                scope_stack.pop()

        return symbols

    def _sanitize_line(self, line: str) -> str:
        text = SENTENCE_STRIP_RE.sub("", line)
        text = re.sub(r"//.*$", "", text)
        text = re.sub(r"#.*$", "", text)
        return text.strip()

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ExtractNestedVarsArgs):
            return ToolCallDisplay(summary="extract_nested_vars")

        summary = f"extract_nested_vars: {len(event.args.snippets)} snippet(s)"
        return ToolCallDisplay(
            summary=summary,
            details={
                "snippets": len(event.args.snippets),
                "include_params": event.args.include_params,
                "include_locals": event.args.include_locals,
                "include_fields": event.args.include_fields,
                "include_globals": event.args.include_globals,
                "max_depth": event.args.max_depth,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ExtractNestedVarsResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = f"Extracted {event.result.count} symbols from {event.result.snippet_count} snippet(s)"
        warnings = event.result.errors[:]
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=warnings,
            details={
                "count": event.result.count,
                "snippet_count": event.result.snippet_count,
                "errors": event.result.errors,
                "symbols": event.result.symbols,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Extracting nested variables"
