from __future__ import annotations

import fnmatch
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
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


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
CLASS_RE = re.compile(r"\b(class|struct|interface|record)\s+[A-Za-z_][A-Za-z0-9_]*")
FUNC_RE = re.compile(r"\b(def|function|fn|func)\s+[A-Za-z_][A-Za-z0-9_]*")
IMPORT_PATTERNS = [
    re.compile(r"^\s*from\s+([A-Za-z0-9_.]+)\s+import\s+", re.M),
    re.compile(r"^\s*import\s+([A-Za-z0-9_.]+)", re.M),
    re.compile(r"^\s*#include\s+[<\"]([^>\"]+)[>\"]", re.M),
    re.compile(r"^\s*using\s+([A-Za-z0-9_.]+)\s*;", re.M),
    re.compile(r"^\s*require\(\s*[\"']([^\"']+)[\"']\s*\)", re.M),
]

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

CODE_KEYWORDS = {
    "abstract",
    "async",
    "await",
    "bool",
    "break",
    "case",
    "catch",
    "class",
    "const",
    "continue",
    "def",
    "default",
    "delete",
    "do",
    "elif",
    "else",
    "enum",
    "export",
    "extends",
    "false",
    "finally",
    "for",
    "foreach",
    "from",
    "function",
    "if",
    "import",
    "in",
    "interface",
    "let",
    "match",
    "namespace",
    "new",
    "null",
    "public",
    "private",
    "protected",
    "raise",
    "record",
    "return",
    "self",
    "static",
    "struct",
    "super",
    "switch",
    "this",
    "throw",
    "true",
    "try",
    "typedef",
    "var",
    "void",
    "while",
    "with",
}

IGNORED_TOKENS = STOPWORDS | CODE_KEYWORDS


@dataclass
class _FileEntry:
    root: Path
    path: Path
    rel_path: str


class ContextSpeechExplainCodebaseConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_file_bytes: int = Field(
        default=2_000_000, description="Maximum bytes per file."
    )
    max_total_bytes: int = Field(
        default=50_000_000, description="Maximum total bytes across files."
    )
    max_files: int = Field(default=1000, description="Maximum files to scan.")
    max_sections: int = Field(
        default=200, description="Maximum speech sections (0 for unlimited)."
    )
    max_keywords: int = Field(default=12, description="Maximum keywords per section.")
    max_imports: int = Field(default=8, description="Maximum imports per section.")
    min_token_length: int = Field(default=3, description="Minimum token length.")
    default_group_by: str = Field(
        default="directory", description="directory, top_level, or file."
    )
    default_include_globs: list[str] = Field(
        default=[
            "**/*.py",
            "**/*.pyi",
            "**/*.js",
            "**/*.ts",
            "**/*.tsx",
            "**/*.jsx",
            "**/*.java",
            "**/*.cs",
            "**/*.cpp",
            "**/*.c",
            "**/*.h",
            "**/*.hpp",
            "**/*.go",
            "**/*.rs",
            "**/*.php",
            "**/*.rb",
            "**/*.swift",
            "**/*.kt",
            "**/*.m",
            "**/*.mm",
            "**/*.json",
            "**/*.yaml",
            "**/*.yml",
            "**/*.toml",
            "**/*.md",
        ],
        description="Default include globs for codebase discovery.",
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
        description="Default glob patterns excluded during discovery.",
    )


class ContextSpeechExplainCodebaseState(BaseToolState):
    pass


class ContextSpeechExplainCodebaseArgs(BaseModel):
    paths: list[str] = Field(description="Root directories or files to scan.")
    include_globs: list[str] | None = Field(
        default=None, description="Optional include globs."
    )
    exclude_globs: list[str] | None = Field(
        default=None, description="Optional exclude globs."
    )
    group_by: str | None = Field(
        default=None, description="directory, top_level, or file."
    )
    max_files: int | None = Field(default=None, description="Override max_files.")
    max_total_bytes: int | None = Field(
        default=None, description="Override max_total_bytes."
    )
    max_file_bytes: int | None = Field(
        default=None, description="Override max_file_bytes."
    )
    max_sections: int | None = Field(
        default=None, description="Override max_sections (0 for unlimited)."
    )
    max_keywords: int | None = Field(
        default=None, description="Override max_keywords."
    )
    max_imports: int | None = Field(
        default=None, description="Override max_imports."
    )


class CodebaseFileStat(BaseModel):
    path: str
    rel_path: str
    language: str | None
    bytes_total: int
    line_count: int
    function_count: int
    class_count: int
    top_keywords: list[str]
    top_imports: list[str]


class CodebaseGroupStat(BaseModel):
    group: str
    file_count: int
    line_count: int
    languages: dict[str, int]
    top_files: list[str]
    top_keywords: list[str]
    top_imports: list[str]


class SpeechSection(BaseModel):
    index: int
    title: str
    summary: str
    files: list[str]


class ContextSpeechExplainCodebaseResult(BaseModel):
    total_files: int
    total_bytes: int
    total_lines: int
    languages: dict[str, int]
    groups: list[CodebaseGroupStat]
    sections: list[SpeechSection]
    section_count: int
    overall_keywords: list[str]
    overall_imports: list[str]
    speech_opening: str
    speech_closing: str
    truncated: bool
    warnings: list[str]


class ContextSpeechExplainCodebase(
    BaseTool[
        ContextSpeechExplainCodebaseArgs,
        ContextSpeechExplainCodebaseResult,
        ContextSpeechExplainCodebaseConfig,
        ContextSpeechExplainCodebaseState,
    ],
    ToolUIData[
        ContextSpeechExplainCodebaseArgs,
        ContextSpeechExplainCodebaseResult,
    ],
):
    description: ClassVar[str] = (
        "Scan a codebase and prepare a speech outline describing its structure."
    )

    async def run(
        self, args: ContextSpeechExplainCodebaseArgs
    ) -> ContextSpeechExplainCodebaseResult:
        if not args.paths:
            raise ToolError("At least one path is required.")

        group_by = (args.group_by or self.config.default_group_by).strip().lower()
        if group_by not in {"directory", "top_level", "file"}:
            raise ToolError("group_by must be directory, top_level, or file.")

        max_files = args.max_files if args.max_files is not None else self.config.max_files
        max_total_bytes = (
            args.max_total_bytes
            if args.max_total_bytes is not None
            else self.config.max_total_bytes
        )
        max_file_bytes = (
            args.max_file_bytes
            if args.max_file_bytes is not None
            else self.config.max_file_bytes
        )
        max_sections = (
            args.max_sections
            if args.max_sections is not None
            else self.config.max_sections
        )
        if max_files <= 0:
            raise ToolError("max_files must be positive.")
        if max_total_bytes <= 0:
            raise ToolError("max_total_bytes must be positive.")
        if max_file_bytes <= 0:
            raise ToolError("max_file_bytes must be positive.")
        if max_sections <= 0:
            max_sections = None

        max_keywords = (
            args.max_keywords
            if args.max_keywords is not None
            else self.config.max_keywords
        )
        max_imports = (
            args.max_imports
            if args.max_imports is not None
            else self.config.max_imports
        )
        if max_keywords <= 0:
            raise ToolError("max_keywords must be positive.")
        if max_imports <= 0:
            raise ToolError("max_imports must be positive.")

        include_globs = self._normalize_globs(
            args.include_globs, self.config.default_include_globs
        )
        exclude_globs = self._normalize_globs(
            args.exclude_globs, self.config.default_exclude_globs
        )
        if not include_globs:
            raise ToolError("No include globs configured.")

        files = self._gather_files(args.paths, include_globs, exclude_globs, max_files)
        if not files:
            raise ToolError("No files matched the provided paths/patterns.")

        warnings: list[str] = []
        total_bytes = 0
        total_lines = 0
        total_files = 0
        truncated = False

        language_counts: Counter[str] = Counter()
        overall_tokens: Counter[str] = Counter()
        overall_imports: Counter[str] = Counter()
        file_stats: list[CodebaseFileStat] = []

        for entry in files:
            if truncated:
                break
            try:
                size_bytes = entry.path.stat().st_size
                if size_bytes > max_file_bytes:
                    warnings.append(
                        f"Skipped {entry.rel_path}: exceeds max_file_bytes."
                    )
                    continue
                if total_bytes + size_bytes > max_total_bytes:
                    truncated = True
                    warnings.append("Total byte limit reached; output truncated.")
                    break

                content = entry.path.read_text("utf-8", errors="ignore")
                total_bytes += size_bytes
                total_files += 1
                line_count = content.count("\n") + (1 if content else 0)
                total_lines += line_count

                language = self._resolve_language(entry.path)
                if language:
                    language_counts[language] += 1

                tokens = self._tokenize(content, self.config.min_token_length)
                token_counts = Counter(tokens)
                overall_tokens.update(token_counts)

                imports = self._extract_imports(content)
                import_counts = Counter(imports)
                overall_imports.update(import_counts)

                function_count = len(FUNC_RE.findall(content))
                class_count = len(CLASS_RE.findall(content))

                file_stats.append(
                    CodebaseFileStat(
                        path=str(entry.path),
                        rel_path=entry.rel_path,
                        language=language,
                        bytes_total=size_bytes,
                        line_count=line_count,
                        function_count=function_count,
                        class_count=class_count,
                        top_keywords=[
                            word for word, _ in token_counts.most_common(max_keywords)
                        ],
                        top_imports=[
                            name for name, _ in import_counts.most_common(max_imports)
                        ],
                    )
                )
            except ToolError as exc:
                warnings.append(str(exc))
            except Exception as exc:
                warnings.append(f"{entry.rel_path}: {exc}")

        groups = self._build_groups(file_stats, group_by, max_keywords, max_imports)
        sections = self._build_sections(groups, max_sections)

        overall_keywords = [
            word for word, _ in overall_tokens.most_common(max_keywords)
        ]
        overall_import_list = [
            name for name, _ in overall_imports.most_common(max_imports)
        ]

        speech_opening = self._speech_opening(
            total_files, total_lines, language_counts, groups
        )
        speech_closing = self._speech_closing(groups, overall_import_list)

        return ContextSpeechExplainCodebaseResult(
            total_files=total_files,
            total_bytes=total_bytes,
            total_lines=total_lines,
            languages=dict(language_counts),
            groups=groups,
            sections=sections,
            section_count=len(sections),
            overall_keywords=overall_keywords,
            overall_imports=overall_import_list,
            speech_opening=speech_opening,
            speech_closing=speech_closing,
            truncated=truncated,
            warnings=warnings,
        )

    def _normalize_globs(
        self, value: list[str] | None, defaults: list[str]
    ) -> list[str]:
        globs = value if value is not None else defaults
        globs = [g.strip() for g in globs if g and g.strip()]
        return globs

    def _gather_files(
        self,
        paths: list[str],
        include_globs: list[str],
        exclude_globs: list[str],
        max_files: int,
    ) -> list[_FileEntry]:
        discovered: list[_FileEntry] = []
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
                        discovered.append(
                            _FileEntry(root=path, path=file_path, rel_path=rel)
                        )
                        if len(discovered) >= max_files:
                            return discovered
            elif path.is_file():
                rel = path.name
                if self._is_excluded(rel, exclude_globs):
                    continue
                key = str(path)
                if key not in seen:
                    seen.add(key)
                    discovered.append(
                        _FileEntry(root=path.parent, path=path, rel_path=rel)
                    )
                    if len(discovered) >= max_files:
                        return discovered
            else:
                raise ToolError(f"Path not found: {path}")

        return discovered

    def _is_excluded(self, rel_path: str, exclude_globs: list[str]) -> bool:
        if not exclude_globs:
            return False
        return any(fnmatch.fnmatch(rel_path, pattern) for pattern in exclude_globs)

    def _resolve_language(self, path: Path) -> str | None:
        ext = path.suffix.lower().lstrip(".")
        return ext if ext else None

    def _tokenize(self, text: str, min_len: int) -> list[str]:
        tokens = []
        for token in TOKEN_RE.findall(text):
            lower = token.lower()
            if len(lower) < min_len:
                continue
            if lower in IGNORED_TOKENS:
                continue
            tokens.append(lower)
        return tokens

    def _extract_imports(self, text: str) -> list[str]:
        imports: list[str] = []
        for pattern in IMPORT_PATTERNS:
            for match in pattern.findall(text):
                if isinstance(match, tuple):
                    value = match[0]
                else:
                    value = match
                value = value.strip()
                if value:
                    imports.append(value)
        return imports

    def _build_groups(
        self,
        files: list[CodebaseFileStat],
        group_by: str,
        max_keywords: int,
        max_imports: int,
    ) -> list[CodebaseGroupStat]:
        group_files: dict[str, list[CodebaseFileStat]] = defaultdict(list)
        for stat in files:
            group = self._group_key(stat.rel_path, group_by)
            group_files[group].append(stat)

        groups: list[CodebaseGroupStat] = []
        for group, items in group_files.items():
            lang_counter: Counter[str] = Counter()
            keyword_counter: Counter[str] = Counter()
            import_counter: Counter[str] = Counter()
            line_count = 0
            for item in items:
                line_count += item.line_count
                if item.language:
                    lang_counter[item.language] += 1
                keyword_counter.update(item.top_keywords)
                import_counter.update(item.top_imports)

            top_files = [
                entry.rel_path
                for entry in sorted(items, key=lambda s: s.line_count, reverse=True)[:5]
            ]
            groups.append(
                CodebaseGroupStat(
                    group=group,
                    file_count=len(items),
                    line_count=line_count,
                    languages=dict(lang_counter),
                    top_files=top_files,
                    top_keywords=[
                        word for word, _ in keyword_counter.most_common(max_keywords)
                    ],
                    top_imports=[
                        name for name, _ in import_counter.most_common(max_imports)
                    ],
                )
            )

        groups.sort(key=lambda item: item.line_count, reverse=True)
        return groups

    def _group_key(self, rel_path: str, group_by: str) -> str:
        parts = rel_path.split("/")
        if group_by == "file":
            return rel_path
        if group_by == "top_level":
            return parts[0] if parts else rel_path
        if len(parts) <= 1:
            return "."
        return "/".join(parts[:-1])

    def _build_sections(
        self, groups: list[CodebaseGroupStat], max_sections: int | None
    ) -> list[SpeechSection]:
        sections: list[SpeechSection] = []
        for idx, group in enumerate(groups, start=1):
            if max_sections is not None and idx > max_sections:
                break
            lang_summary = self._format_top_items(group.languages, 3)
            keywords = ", ".join(group.top_keywords[:5])
            imports = ", ".join(group.top_imports[:4])
            summary_parts = [
                f"{group.file_count} files",
                f"{group.line_count} lines",
            ]
            if lang_summary:
                summary_parts.append(f"languages: {lang_summary}")
            if keywords:
                summary_parts.append(f"keywords: {keywords}")
            if imports:
                summary_parts.append(f"imports: {imports}")
            summary = "; ".join(summary_parts) + "."
            sections.append(
                SpeechSection(
                    index=idx,
                    title=f"{group.group}",
                    summary=summary,
                    files=group.top_files,
                )
            )
        return sections

    def _format_top_items(self, items: dict[str, int], max_items: int) -> str:
        if not items:
            return ""
        pairs = sorted(items.items(), key=lambda item: item[1], reverse=True)
        return ", ".join(f"{name} ({count})" for name, count in pairs[:max_items])

    def _speech_opening(
        self,
        total_files: int,
        total_lines: int,
        languages: Counter[str],
        groups: list[CodebaseGroupStat],
    ) -> str:
        lang_summary = self._format_top_items(dict(languages), 4)
        top_groups = ", ".join(group.group for group in groups[:3])
        parts = [
            f"Overview: {total_files} files, {total_lines} lines.",
        ]
        if lang_summary:
            parts.append(f"Main languages: {lang_summary}.")
        if top_groups:
            parts.append(f"Primary areas: {top_groups}.")
        return " ".join(parts)

    def _speech_closing(
        self, groups: list[CodebaseGroupStat], imports: list[str]
    ) -> str:
        if not groups:
            return "Summarize the codebase and highlight next steps."
        largest = groups[0]
        parts = [
            f"Wrap up by revisiting {largest.group} and its main responsibilities."
        ]
        if imports:
            parts.append(f"Key dependencies include: {', '.join(imports[:5])}.")
        return " ".join(parts)

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextSpeechExplainCodebaseArgs):
            return ToolCallDisplay(summary="context_speech_explain_codebase")
        return ToolCallDisplay(
            summary="context_speech_explain_codebase",
            details={
                "paths": event.args.paths,
                "group_by": event.args.group_by,
                "include_globs": event.args.include_globs,
                "exclude_globs": event.args.exclude_globs,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpeechExplainCodebaseResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = (
            f"Prepared {event.result.section_count} section(s) from "
            f"{event.result.total_files} file(s)"
        )
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=event.result.warnings,
            details={
                "section_count": event.result.section_count,
                "total_files": event.result.total_files,
                "total_lines": event.result.total_lines,
                "truncated": event.result.truncated,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Preparing speech outline for codebase"
