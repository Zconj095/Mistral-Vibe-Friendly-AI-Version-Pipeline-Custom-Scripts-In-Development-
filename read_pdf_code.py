from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import tempfile
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


CODE_KEYWORDS = re.compile(
    r"^\s*(def|class|import|from|if|elif|else|for|while|try|except|finally|return|with|"
    r"async|await|function|var|let|const|public|private|protected|static|void|int|float|"
    r"double|string|bool|boolean|struct|enum|switch|case|break|continue|using|namespace|"
    r"package|interface|extends|implements|yield|include)\b"
)

CODE_SYMBOLS = set("{}[]();<>:=#/*+-_|&^%$~`")


class ReadPdfCodeConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_read_bytes: int = Field(
        default=120_000,
        description="Maximum number of bytes allowed for a single-chunk code extract.",
    )


class ReadPdfCodeState(BaseToolState):
    pass


class ReadPdfCodeArgs(BaseModel):
    path: str
    max_bytes: int | None = Field(
        default=None, description="Override the configured max_read_bytes for this call."
    )
    max_pages: int | None = Field(
        default=None, description="Optional maximum number of pages to extract."
    )


class ReadPdfCodeResult(BaseModel):
    path: str
    content: str
    lines_total: int
    bytes_total: int


class ReadPdfCode(
    BaseTool[ReadPdfCodeArgs, ReadPdfCodeResult, ReadPdfCodeConfig, ReadPdfCodeState],
    ToolUIData[ReadPdfCodeArgs, ReadPdfCodeResult],
):
    description: ClassVar[str] = (
        "Extract programming code from a PDF into a single chunk. "
        "Fails if the code exceeds the configured size limit."
    )

    async def run(self, args: ReadPdfCodeArgs) -> ReadPdfCodeResult:
        path = self._resolve_path(args.path)
        max_bytes = self._resolve_max_bytes(args.max_bytes)
        max_pages = self._resolve_max_pages(args.max_pages)
        text = self._extract_text(path, max_pages)
        code_lines = self._extract_code_lines(text)

        if not code_lines:
            raise ToolError("No code-like lines were found in the PDF.")

        content = "\n".join(code_lines).rstrip()
        content_bytes = content.encode("utf-8")
        if max_bytes > 0 and len(content_bytes) > max_bytes:
            raise ToolError(
                f"Extracted code is {len(content_bytes)} bytes, which exceeds the "
                f"single-chunk limit of {max_bytes} bytes."
            )

        return ReadPdfCodeResult(
            path=str(path),
            content=content,
            lines_total=len(code_lines),
            bytes_total=len(content_bytes),
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
        if resolved.suffix.lower() != ".pdf":
            raise ToolError("Expected a .pdf file.")

        return resolved

    def _resolve_max_bytes(self, override: int | None) -> int:
        max_bytes = override if override is not None else self.config.max_read_bytes
        return max_bytes

    def _resolve_max_pages(self, value: int | None) -> int | None:
        if value is None:
            return None
        if value <= 0:
            raise ToolError("max_pages must be a positive integer.")
        return value

    def _extract_text(self, path: Path, max_pages: int | None) -> str:
        text = self._extract_text_with_pypdf(path, max_pages)
        if text is not None:
            return text

        text = self._extract_text_with_pdftotext(path, max_pages)
        if text is not None:
            return text

        raise ToolError(
            "No PDF text extractor available. Install 'pypdf' or add 'pdftotext' "
            "to your PATH."
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
            page_text = pages[page_index].extract_text() or ""
            chunks.append(page_text)

        return "\n".join(chunks)

    def _extract_text_with_pdftotext(
        self, path: Path, max_pages: int | None
    ) -> str | None:
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

    def _extract_code_lines(self, text: str) -> list[str]:
        lines = text.splitlines()
        code_lines: list[str] = []
        for line in lines:
            candidate = self._strip_line_number(line)
            if self._is_code_line(candidate):
                code_lines.append(candidate.rstrip())
        return code_lines

    def _strip_line_number(self, line: str) -> str:
        return re.sub(r"^\s*\d+\s+", "", line)

    def _is_code_line(self, line: str) -> bool:
        trimmed = line.strip()
        if not trimmed:
            return False

        if trimmed in {"{", "}", "};", ")", "]"}:
            return True

        if trimmed.startswith(("#", "//", "/*", "*/")):
            return True

        if trimmed.startswith("@") and re.match(r"@[A-Za-z_][\w\.]*", trimmed):
            return True

        if CODE_KEYWORDS.match(trimmed):
            return True

        symbol_count = sum(1 for ch in trimmed if ch in CODE_SYMBOLS)
        word_count = len(re.findall(r"[A-Za-z]+", trimmed))
        digit_count = sum(1 for ch in trimmed if ch.isdigit())
        indent = len(line) - len(line.lstrip())

        score = 0
        if indent >= 4:
            score += 1
        if ";" in trimmed:
            score += 2
        if "->" in trimmed or "=>" in trimmed:
            score += 2
        if "::" in trimmed:
            score += 1
        if "(" in trimmed and ")" in trimmed:
            score += 1
        if "=" in trimmed:
            score += 1
        if any(op in trimmed for op in ("==", "!=", ">=", "<=", "&&", "||")):
            score += 1
        if any(op in trimmed for op in ("+=", "-=", "*=", "/=", "%=")):
            score += 1
        if symbol_count >= 3:
            score += 1
        if symbol_count >= 6:
            score += 1
        if digit_count and symbol_count:
            score += 1

        if word_count >= 10 and symbol_count <= 1:
            score -= 2
        if trimmed.endswith((".", "?", "!")) and word_count >= 6:
            score -= 2

        return score >= 2

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ReadPdfCodeArgs):
            return ToolCallDisplay(summary="read_pdf_code")

        summary = f"read_pdf_code: {event.args.path}"
        if event.args.max_pages:
            summary += f" (pages: {event.args.max_pages})"
        if event.args.max_bytes:
            summary += f" (max {event.args.max_bytes} bytes)"
        return ToolCallDisplay(
            summary=summary,
            details={
                "path": event.args.path,
                "max_pages": event.args.max_pages,
                "max_bytes": event.args.max_bytes,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ReadPdfCodeResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = (
            f"Extracted {event.result.lines_total} code lines "
            f"({event.result.bytes_total} bytes)."
        )
        return ToolResultDisplay(
            success=True,
            message=message,
            details={
                "path": event.result.path,
                "lines_total": event.result.lines_total,
                "bytes_total": event.result.bytes_total,
                "content": event.result.content,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Extracting code from PDF"
