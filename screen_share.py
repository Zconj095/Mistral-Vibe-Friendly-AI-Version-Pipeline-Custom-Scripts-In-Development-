from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from typing import ClassVar, final

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

IMAGE_FORMATS = {"png": "Png", "jpg": "Jpeg", "jpeg": "Jpeg"}


@dataclass(frozen=True)
class CaptureResult:
    width: int | None
    height: int | None


class ScreenShareArgs(BaseModel):
    path: str | None = Field(
        default=None,
        description="Optional output path for the screenshot.",
    )
    format: str | None = Field(
        default=None,
        description="Image format: png or jpg. Defaults to config.",
    )
    left: int | None = Field(default=None, description="Region left X coordinate.")
    top: int | None = Field(default=None, description="Region top Y coordinate.")
    width: int | None = Field(default=None, description="Region width in pixels.")
    height: int | None = Field(default=None, description="Region height in pixels.")
    ocr: bool = Field(default=True, description="Run OCR on the screenshot.")
    ocr_engine: str = Field(
        default="auto",
        description="OCR engine: auto, tesseract, easyocr, or none.",
    )
    ocr_language: str | None = Field(
        default=None,
        description="OCR language (defaults to config).",
    )


class ScreenShareResult(BaseModel):
    path: str
    bytes_written: int
    width: int | None
    height: int | None
    ocr_text: str
    ocr_engine: str
    ocr_success: bool
    ocr_truncated: bool
    ocr_error: str | None = None


class ScreenShareConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    output_dir: Path = Field(default=Path.home() / ".vibe" / "screenshots")
    default_format: str = "png"
    file_prefix: str = "screen"
    max_image_bytes: int = 5_000_000
    max_ocr_chars: int = 10_000
    ocr_language: str = "eng"

    @field_validator("output_dir", mode="before")
    @classmethod
    def set_default_output_dir(cls, v: Path | str) -> Path:
        if isinstance(v, Path):
            return v
        if not v or not str(v).strip():
            return Path.home() / ".vibe" / "screenshots"
        return Path(v)

    @field_validator("output_dir", mode="after")
    @classmethod
    def expand_output_dir(cls, v: Path) -> Path:
        return v.expanduser().resolve()


class ScreenShareState(BaseToolState):
    pass


class ScreenShare(
    BaseTool[ScreenShareArgs, ScreenShareResult, ScreenShareConfig, ScreenShareState],
    ToolUIData[ScreenShareArgs, ScreenShareResult],
):
    description: ClassVar[str] = (
        "Capture a screenshot and optionally run OCR to extract text."
    )

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ScreenShareArgs):
            return ToolCallDisplay(summary="screen_share")

        return ToolCallDisplay(
            summary="Capturing screenshot",
            details={
                "path": event.args.path,
                "format": event.args.format,
                "ocr": event.args.ocr,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if isinstance(event.result, ScreenShareResult):
            return ToolResultDisplay(
                success=True,
                message=f"Screenshot saved to {Path(event.result.path).name}",
                details={
                    "path": event.result.path,
                    "bytes_written": event.result.bytes_written,
                    "width": event.result.width,
                    "height": event.result.height,
                    "ocr_engine": event.result.ocr_engine,
                    "ocr_success": event.result.ocr_success,
                    "ocr_error": event.result.ocr_error,
                    "ocr_text": event.result.ocr_text,
                },
            )
        return ToolResultDisplay(success=True, message="Screenshot captured")

    @classmethod
    def get_status_text(cls) -> str:
        return "Capturing screenshot"

    @final
    async def run(self, args: ScreenShareArgs) -> ScreenShareResult:
        format_name = self._resolve_format(args)
        region = self._resolve_region(args)
        output_path = self._resolve_output_path(args, format_name)

        capture = self._capture_screen(output_path, region, format_name)
        bytes_written = output_path.stat().st_size
        self._enforce_image_limit(output_path, bytes_written)

        ocr_text, ocr_engine, ocr_success, ocr_truncated, ocr_error = (
            self._run_ocr(output_path, args)
        )

        return ScreenShareResult(
            path=str(output_path),
            bytes_written=bytes_written,
            width=capture.width,
            height=capture.height,
            ocr_text=ocr_text,
            ocr_engine=ocr_engine,
            ocr_success=ocr_success,
            ocr_truncated=ocr_truncated,
            ocr_error=ocr_error,
        )

    def _resolve_format(self, args: ScreenShareArgs) -> str:
        raw = (args.format or self.config.default_format).strip().lower().lstrip(".")
        if raw not in IMAGE_FORMATS:
            raise ToolError("Format must be png or jpg.")
        return raw

    def _resolve_region(self, args: ScreenShareArgs) -> tuple[int, int, int, int] | None:
        values = [args.left, args.top, args.width, args.height]
        if all(v is None for v in values):
            return None
        if any(v is None for v in values):
            raise ToolError("Region requires left, top, width, and height.")
        left = int(args.left or 0)
        top = int(args.top or 0)
        width = int(args.width or 0)
        height = int(args.height or 0)
        if width <= 0 or height <= 0:
            raise ToolError("Region width and height must be positive.")
        return left, top, width, height

    def _resolve_output_path(self, args: ScreenShareArgs, fmt: str) -> Path:
        if args.path:
            path = Path(args.path).expanduser()
            if not path.is_absolute():
                path = self.config.effective_workdir / path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config.file_prefix}_{timestamp}.{fmt}"
            path = self.config.output_dir / filename

        if args.format and path.suffix.lower() != f".{fmt}":
            path = path.with_suffix(f".{fmt}")
        elif not path.suffix:
            path = path.with_suffix(f".{fmt}")

        path.parent.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    def _capture_screen(
        self, path: Path, region: tuple[int, int, int, int] | None, fmt: str
    ) -> CaptureResult:
        errors: list[str] = []
        for backend in (
            self._capture_with_pillow,
            self._capture_with_mss,
            self._capture_with_powershell,
        ):
            try:
                result = backend(path, region, fmt)
            except ToolError as exc:
                errors.append(str(exc))
                continue
            if result is not None:
                return result

        if errors:
            raise ToolError("Screen capture failed: " + "; ".join(errors))
        raise ToolError("No screen capture backend available.")

    def _capture_with_pillow(
        self, path: Path, region: tuple[int, int, int, int] | None, fmt: str
    ) -> CaptureResult | None:
        try:
            from PIL import ImageGrab
        except ModuleNotFoundError:
            return None

        try:
            bbox = None
            if region is not None:
                left, top, width, height = region
                bbox = (left, top, left + width, top + height)
            image = ImageGrab.grab(bbox=bbox, all_screens=True)
            image.save(path, format=IMAGE_FORMATS[fmt])
            width, height = image.size
            return CaptureResult(width=width, height=height)
        except Exception as exc:
            raise ToolError(f"Pillow capture failed: {exc}") from exc

    def _capture_with_mss(
        self, path: Path, region: tuple[int, int, int, int] | None, fmt: str
    ) -> CaptureResult | None:
        try:
            import mss
            import mss.tools
        except ModuleNotFoundError:
            return None

        if fmt != "png":
            raise ToolError("mss backend only supports png output.")

        try:
            with mss.mss() as sct:
                if region is not None:
                    left, top, width, height = region
                    monitor = {
                        "left": left,
                        "top": top,
                        "width": width,
                        "height": height,
                    }
                else:
                    monitor = sct.monitors[0]
                shot = sct.grab(monitor)
                mss.tools.to_png(shot.rgb, shot.size, output=str(path))
                return CaptureResult(width=shot.width, height=shot.height)
        except Exception as exc:
            raise ToolError(f"mss capture failed: {exc}") from exc

    def _capture_with_powershell(
        self, path: Path, region: tuple[int, int, int, int] | None, fmt: str
    ) -> CaptureResult | None:
        if sys.platform != "win32":
            return None

        format_name = IMAGE_FORMATS[fmt]
        left, top, width, height = (region or (0, 0, 0, 0))
        use_region = region is not None
        script = f"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$path = {self._powershell_quote(str(path))}
$useRegion = {'$true' if use_region else '$false'}
$left = {left}
$top = {top}
$width = {width}
$height = {height}
if ($useRegion) {{
  $bounds = New-Object System.Drawing.Rectangle($left, $top, $width, $height)
}} else {{
  $vs = [System.Windows.Forms.SystemInformation]::VirtualScreen
  $bounds = New-Object System.Drawing.Rectangle($vs.Left, $vs.Top, $vs.Width, $vs.Height)
}}
$bitmap = New-Object System.Drawing.Bitmap($bounds.Width, $bounds.Height)
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.CopyFromScreen($bounds.Left, $bounds.Top, 0, 0, $bounds.Size)
$bitmap.Save($path, [System.Drawing.Imaging.ImageFormat]::{format_name})
$graphics.Dispose()
$bitmap.Dispose()
Write-Output \"$($bounds.Width) $($bounds.Height)\"
""".strip()

        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            message = stderr or "PowerShell capture failed."
            raise ToolError(message) from exc

        width_out, height_out = self._parse_dimensions(proc.stdout)
        return CaptureResult(width=width_out, height=height_out)

    def _parse_dimensions(self, output: str) -> tuple[int | None, int | None]:
        tokens = output.strip().split()
        if len(tokens) >= 2 and tokens[0].isdigit() and tokens[1].isdigit():
            return int(tokens[0]), int(tokens[1])
        return None, None

    def _powershell_quote(self, value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    def _enforce_image_limit(self, path: Path, size: int) -> None:
        max_bytes = self.config.max_image_bytes
        if max_bytes > 0 and size > max_bytes:
            try:
                path.unlink()
            except OSError:
                pass
            raise ToolError(
                f"Screenshot is {size} bytes, exceeds {max_bytes} bytes."
            )

    def _run_ocr(
        self, path: Path, args: ScreenShareArgs
    ) -> tuple[str, str, bool, bool, str | None]:
        if not args.ocr or args.ocr_engine.lower() == "none":
            return "", "none", False, False, None

        engine = args.ocr_engine.lower()
        language = args.ocr_language or self.config.ocr_language
        max_chars = self.config.max_ocr_chars

        if engine in {"auto", "tesseract"}:
            text, error = self._run_tesseract(path, language)
            if text is not None:
                text, truncated = self._truncate_text(text, max_chars)
                return text, "tesseract", True, truncated, None
            if engine == "tesseract":
                return "", "tesseract", False, False, error

        if engine in {"auto", "easyocr"}:
            text, error = self._run_easyocr(path, language)
            if text is not None:
                text, truncated = self._truncate_text(text, max_chars)
                return text, "easyocr", True, truncated, None
            return "", "easyocr", False, False, error

        return "", engine, False, False, "Unsupported OCR engine."

    def _run_tesseract(
        self, path: Path, language: str
    ) -> tuple[str | None, str | None]:
        try:
            import pytesseract
        except ModuleNotFoundError:
            return None, "pytesseract is not installed."

        try:
            text = pytesseract.image_to_string(str(path), lang=language)
            return text.strip(), None
        except Exception as exc:
            return None, f"tesseract failed: {exc}"

    def _run_easyocr(
        self, path: Path, language: str
    ) -> tuple[str | None, str | None]:
        try:
            import easyocr
        except ModuleNotFoundError:
            return None, "easyocr is not installed."

        try:
            lang = "en" if language.startswith("en") else language
            reader = easyocr.Reader([lang], gpu=False)
            lines = reader.readtext(str(path), detail=0, paragraph=True)
            text = "\n".join(line.strip() for line in lines if line.strip())
            return text.strip(), None
        except Exception as exc:
            return None, f"easyocr failed: {exc}"

    def _truncate_text(self, text: str, max_chars: int) -> tuple[str, bool]:
        if max_chars > 0 and len(text) > max_chars:
            return text[:max_chars], True
        return text, False
