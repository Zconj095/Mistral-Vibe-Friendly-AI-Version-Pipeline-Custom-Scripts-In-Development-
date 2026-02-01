from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import ClassVar

from pydantic import BaseModel, Field, field_validator

from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from vibe.core.types import ToolCallEvent, ToolResultEvent

INLINE_PAREN_RE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
DISPLAY_BRACKET_RE = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
MATH_ENV_RE = re.compile(
    r"\\begin\{(?P<env>equation\*?|align\*?|alignat\*?|gather\*?|multline\*?|eqnarray\*?|displaymath)\}"
    r"(?P<content>.*?)\\end\{(?P=env)\}",
    re.DOTALL,
)


@dataclass(frozen=True)
class MathRange:
    start: int
    end: int
    kind: str
    env: str | None
    content: str


class LatexBlock(BaseModel):
    index: int
    kind: str
    env: str | None = None
    preview: str


class ProcessLatexArgs(BaseModel):
    content: str | None = Field(
        default=None, description="LaTeX source text."
    )
    path: str | None = Field(
        default=None, description="Path to a LaTeX source file."
    )
    extract_math: bool = Field(
        default=True, description="Extract math blocks from the source."
    )
    compile_pdf: bool = Field(
        default=False, description="Compile the LaTeX source to PDF."
    )
    engine: str = Field(
        default="auto", description="Compile engine: auto, tectonic, latexmk, pdflatex."
    )
    wrap_document: bool = Field(
        default=True,
        description="Wrap the source in a minimal document if needed.",
    )
    output_name: str | None = Field(
        default=None, description="Base name for compiled output."
    )


class ProcessLatexResult(BaseModel):
    source: str
    source_bytes: int
    math_blocks: list[LatexBlock]
    math_block_count: int
    math_block_truncated: bool
    cleaned_text: str
    cleaned_text_truncated: bool
    compiled_path: str | None = None
    compile_engine: str | None = None
    compile_success: bool = False
    compile_error: str | None = None


class ProcessLatexConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    max_source_bytes: int = 1_000_000
    max_blocks: int = 200
    max_block_chars: int = 2_000
    max_clean_text_chars: int = 20_000
    output_dir: Path = Field(default=Path.home() / ".vibe" / "latex")
    compile_timeout_seconds: float = 20.0
    max_output_bytes: int = 10_000_000

    @field_validator("output_dir", mode="before")
    @classmethod
    def set_default_output_dir(cls, v: Path | str) -> Path:
        if isinstance(v, Path):
            return v
        if not v or not str(v).strip():
            return Path.home() / ".vibe" / "latex"
        return Path(v)

    @field_validator("output_dir", mode="after")
    @classmethod
    def expand_output_dir(cls, v: Path) -> Path:
        return v.expanduser().resolve()


class ProcessLatexState(BaseToolState):
    pass


class ProcessLatex(
    BaseTool[
        ProcessLatexArgs,
        ProcessLatexResult,
        ProcessLatexConfig,
        ProcessLatexState,
    ],
    ToolUIData[ProcessLatexArgs, ProcessLatexResult],
):
    description: ClassVar[str] = (
        "Parse complex LaTeX, extract math blocks, and optionally compile to PDF."
    )

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ProcessLatexArgs):
            return ToolCallDisplay(summary="process_latex")

        return ToolCallDisplay(
            summary="Processing LaTeX",
            details={
                "path": event.args.path,
                "compile_pdf": event.args.compile_pdf,
                "engine": event.args.engine,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if isinstance(event.result, ProcessLatexResult):
            return ToolResultDisplay(
                success=True,
                message="LaTeX processed",
                details={
                    "source": event.result.source,
                    "math_block_count": event.result.math_block_count,
                    "math_block_truncated": event.result.math_block_truncated,
                    "cleaned_text": event.result.cleaned_text,
                    "compiled_path": event.result.compiled_path,
                    "compile_engine": event.result.compile_engine,
                    "compile_success": event.result.compile_success,
                    "compile_error": event.result.compile_error,
                },
            )
        return ToolResultDisplay(success=True, message="LaTeX processed")

    @classmethod
    def get_status_text(cls) -> str:
        return "Processing LaTeX"

    async def run(self, args: ProcessLatexArgs) -> ProcessLatexResult:
        content, source = self._load_source(args)
        source_bytes = len(content.encode("utf-8"))

        blocks: list[MathRange] = []
        if args.extract_math:
            blocks = self._extract_math_blocks(content)

        latex_blocks, truncated = self._build_block_summaries(blocks)
        cleaned_text, cleaned_truncated = self._build_cleaned_text(content, blocks)

        compiled_path = None
        compile_engine = None
        compile_success = False
        compile_error = None
        if args.compile_pdf:
            try:
                compiled_path, compile_engine = self._compile_pdf(
                    content, source, args
                )
                compile_success = True
            except ToolError as exc:
                compile_error = str(exc)

        return ProcessLatexResult(
            source=source,
            source_bytes=source_bytes,
            math_blocks=latex_blocks,
            math_block_count=len(blocks),
            math_block_truncated=truncated,
            cleaned_text=cleaned_text,
            cleaned_text_truncated=cleaned_truncated,
            compiled_path=compiled_path,
            compile_engine=compile_engine,
            compile_success=compile_success,
            compile_error=compile_error,
        )

    def _load_source(self, args: ProcessLatexArgs) -> tuple[str, str]:
        if args.content and args.content.strip():
            return args.content, "inline"

        if args.path is None or not args.path.strip():
            raise ToolError("Provide either content or path.")

        path = Path(args.path).expanduser()
        if not path.is_absolute():
            path = self.config.effective_workdir / path

        if not path.exists():
            raise ToolError(f"File not found: {path}")
        if path.is_dir():
            raise ToolError(f"Path is a directory: {path}")

        data = path.read_bytes()
        max_bytes = self.config.max_source_bytes
        if max_bytes > 0 and len(data) > max_bytes:
            raise ToolError(
                f"Source is {len(data)} bytes, exceeds {max_bytes} bytes."
            )

        content = data.decode("utf-8", errors="ignore")
        return content, str(path)

    def _extract_math_blocks(self, content: str) -> list[MathRange]:
        cleaned = self._strip_comments(content)
        ranges: list[MathRange] = []

        ranges.extend(self._extract_env_blocks(cleaned))
        ranges.extend(self._extract_bracket_blocks(cleaned))
        ranges.extend(self._extract_dollar_blocks(cleaned))

        ranges.sort(key=lambda item: (item.start, item.end))
        deduped: list[MathRange] = []
        for item in ranges:
            if any(self._overlaps(item, existing) for existing in deduped):
                continue
            deduped.append(item)

        return deduped

    def _strip_comments(self, content: str) -> str:
        lines = []
        for line in content.splitlines():
            cutoff = None
            escaped = False
            for idx, char in enumerate(line):
                if char == "\\":
                    escaped = not escaped
                    continue
                if char == "%" and not escaped:
                    cutoff = idx
                    break
                escaped = False
            if cutoff is not None:
                line = line[:cutoff]
            lines.append(line)
        return "\n".join(lines)

    def _extract_env_blocks(self, content: str) -> list[MathRange]:
        blocks = []
        for match in MATH_ENV_RE.finditer(content):
            blocks.append(
                MathRange(
                    start=match.start(),
                    end=match.end(),
                    kind="environment",
                    env=match.group("env"),
                    content=match.group("content"),
                )
            )
        return blocks

    def _extract_bracket_blocks(self, content: str) -> list[MathRange]:
        blocks = []
        for match in DISPLAY_BRACKET_RE.finditer(content):
            blocks.append(
                MathRange(
                    start=match.start(),
                    end=match.end(),
                    kind="display",
                    env="bracket",
                    content=match.group(1),
                )
            )
        for match in INLINE_PAREN_RE.finditer(content):
            blocks.append(
                MathRange(
                    start=match.start(),
                    end=match.end(),
                    kind="inline",
                    env="paren",
                    content=match.group(1),
                )
            )
        return blocks

    def _extract_dollar_blocks(self, content: str) -> list[MathRange]:
        blocks = []
        i = 0
        start = None
        delim = None
        while i < len(content):
            char = content[i]
            if char == "\\":
                i += 2
                continue
            if char == "$":
                if i + 1 < len(content) and content[i + 1] == "$":
                    token = "$$"
                    if delim == token and start is not None:
                        blocks.append(
                            MathRange(
                                start=start - len(token),
                                end=i + 2,
                                kind="display",
                                env="dollar",
                                content=content[start:i],
                            )
                        )
                        start = None
                        delim = None
                    elif delim is None:
                        delim = token
                        start = i + 2
                    i += 2
                    continue

                token = "$"
                if delim == token and start is not None:
                    blocks.append(
                        MathRange(
                            start=start - len(token),
                            end=i + 1,
                            kind="inline",
                            env="dollar",
                            content=content[start:i],
                        )
                    )
                    start = None
                    delim = None
                elif delim is None:
                    delim = token
                    start = i + 1
                i += 1
                continue
            i += 1
        return blocks

    def _overlaps(self, left: MathRange, right: MathRange) -> bool:
        return left.start < right.end and right.start < left.end

    def _build_block_summaries(
        self, blocks: list[MathRange]
    ) -> tuple[list[LatexBlock], bool]:
        max_blocks = self.config.max_blocks
        truncated = max_blocks > 0 and len(blocks) > max_blocks
        trimmed = blocks[:max_blocks] if max_blocks > 0 else blocks
        max_chars = self.config.max_block_chars

        summaries = []
        for index, block in enumerate(trimmed, start=1):
            preview = block.content.strip()
            if max_chars > 0 and len(preview) > max_chars:
                preview = preview[:max_chars] + "..."
            summaries.append(
                LatexBlock(
                    index=index,
                    kind=block.kind,
                    env=block.env,
                    preview=preview,
                )
            )
        return summaries, truncated

    def _build_cleaned_text(
        self, content: str, blocks: list[MathRange]
    ) -> tuple[str, bool]:
        if not blocks:
            cleaned = content
        else:
            cleaned = content
            for index, block in enumerate(
                sorted(blocks, key=lambda b: b.start), start=1
            ):
                placeholder = f"[MATH_{index}]"
                cleaned = (
                    cleaned[: block.start]
                    + placeholder
                    + cleaned[block.end :]
                )

        max_chars = self.config.max_clean_text_chars
        if max_chars > 0 and len(cleaned) > max_chars:
            return cleaned[:max_chars] + "...", True
        return cleaned, False

    def _compile_pdf(
        self, content: str, source: str, args: ProcessLatexArgs
    ) -> tuple[str, str]:
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        name = self._resolve_output_name(args, source)
        tex_content = self._maybe_wrap_document(content, args.wrap_document)

        engine = self._resolve_engine(args.engine)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tex_path = tmp_path / f"{name}.tex"
            tex_path.write_text(tex_content, encoding="utf-8")

            out_dir = tmp_path / "out"
            out_dir.mkdir(parents=True, exist_ok=True)

            self._run_engine(engine, tex_path, out_dir)

            pdf_path = out_dir / f"{name}.pdf"
            if not pdf_path.exists():
                raise ToolError("PDF output not found after compilation.")

            if (
                self.config.max_output_bytes > 0
                and pdf_path.stat().st_size > self.config.max_output_bytes
            ):
                raise ToolError("Compiled PDF exceeds size limit.")

            final_path = output_dir / pdf_path.name
            shutil.copy2(pdf_path, final_path)
            return str(final_path), engine

    def _resolve_output_name(self, args: ProcessLatexArgs, source: str) -> str:
        if args.output_name:
            name = args.output_name.strip()
        elif source != "inline":
            name = Path(source).stem
        else:
            name = "latex_output"

        name = re.sub(r"[^A-Za-z0-9_-]", "_", name)
        return name or "latex_output"

    def _maybe_wrap_document(self, content: str, wrap: bool) -> str:
        if not wrap:
            return content
        if "\\documentclass" in content:
            return content
        return (
            "\\documentclass{article}\n"
            "\\usepackage{amsmath,amssymb}\n"
            "\\begin{document}\n"
            + content
            + "\n\\end{document}\n"
        )

    def _resolve_engine(self, engine: str) -> str:
        engine = engine.lower().strip()
        if engine != "auto":
            return engine

        for candidate in ("tectonic", "latexmk", "pdflatex"):
            if shutil.which(candidate):
                return candidate
        raise ToolError("No LaTeX engine found (tectonic, latexmk, pdflatex).")

    def _run_engine(self, engine: str, tex_path: Path, out_dir: Path) -> None:
        timeout = self.config.compile_timeout_seconds
        if engine == "tectonic":
            cmd = [engine, "--outdir", str(out_dir), str(tex_path)]
        elif engine == "latexmk":
            cmd = [
                engine,
                "-pdf",
                "-interaction=nonstopmode",
                "-outdir=" + str(out_dir),
                str(tex_path),
            ]
        elif engine == "pdflatex":
            cmd = [
                engine,
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                str(out_dir),
                str(tex_path),
            ]
        else:
            raise ToolError(f"Unsupported engine: {engine}")

        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise ToolError(f"{engine} is not installed.") from exc
        except subprocess.TimeoutExpired as exc:
            raise ToolError("LaTeX compilation timed out.") from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            message = stderr or "LaTeX compilation failed."
            raise ToolError(message) from exc
