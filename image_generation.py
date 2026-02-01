from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar
from urllib import request
from uuid import uuid4

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


DEFAULT_OUTPUT_DIR = Path.home() / ".vibe" / "images"


class ImageGenerationConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    endpoint_url: str = Field(
        default="",
        description="Local image generation endpoint URL (HTTP POST).",
    )
    output_dir: Path = Field(
        default=DEFAULT_OUTPUT_DIR, description="Directory to write images."
    )
    request_timeout_seconds: float = Field(
        default=120.0, description="HTTP timeout."
    )
    max_image_bytes: int = Field(
        default=10_000_000, description="Maximum bytes per image."
    )


class ImageGenerationState(BaseToolState):
    pass


class ImageGenerationArgs(BaseModel):
    prompt: str = Field(description="Image prompt.")
    negative_prompt: str | None = Field(default=None, description="Negative prompt.")
    width: int = Field(default=512, description="Image width.")
    height: int = Field(default=512, description="Image height.")
    steps: int = Field(default=20, description="Number of diffusion steps.")
    seed: int | None = Field(default=None, description="Random seed.")
    num_images: int = Field(default=1, description="Number of images to generate.")


class ImageGenerationResultItem(BaseModel):
    path: str
    size_bytes: int


class ImageGenerationResult(BaseModel):
    images: list[ImageGenerationResultItem]
    count: int


class ImageGeneration(
    BaseTool[
        ImageGenerationArgs,
        ImageGenerationResult,
        ImageGenerationConfig,
        ImageGenerationState,
    ],
    ToolUIData[ImageGenerationArgs, ImageGenerationResult],
):
    description: ClassVar[str] = (
        "Generate images using a local HTTP endpoint and save them to disk."
    )

    async def run(self, args: ImageGenerationArgs) -> ImageGenerationResult:
        if not args.prompt.strip():
            raise ToolError("prompt cannot be empty.")
        if not self.config.endpoint_url:
            raise ToolError("endpoint_url is not configured.")
        if args.width <= 0 or args.height <= 0:
            raise ToolError("width and height must be positive.")
        if args.num_images <= 0:
            raise ToolError("num_images must be a positive integer.")

        payload = {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "seed": args.seed,
            "num_images": args.num_images,
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.config.endpoint_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.config.request_timeout_seconds) as resp:
                response = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise ToolError(f"Image endpoint failed: {exc}") from exc

        images = self._extract_images(response)
        if not images:
            raise ToolError("No images returned by the endpoint.")

        output_dir = self.config.output_dir.expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        saved: list[ImageGenerationResultItem] = []
        for index, image_data in enumerate(images[: args.num_images]):
            path, size_bytes = self._save_image(output_dir, image_data, index)
            saved.append(ImageGenerationResultItem(path=str(path), size_bytes=size_bytes))

        return ImageGenerationResult(images=saved, count=len(saved))

    def _extract_images(self, response: dict) -> list[str]:
        if isinstance(response, dict):
            if "images" in response:
                return list(response["images"] or [])
            if "image" in response:
                return [response["image"]]
            if "image_paths" in response:
                return list(response["image_paths"] or [])
            if "image_path" in response:
                return [response["image_path"]]
        return []

    def _save_image(self, output_dir: Path, image_data: str, index: int) -> tuple[Path, int]:
        if image_data.startswith("data:"):
            _, image_data = image_data.split(",", 1)

        if Path(image_data).exists():
            path = Path(image_data)
            size_bytes = path.stat().st_size
            return path, size_bytes

        raw = base64.b64decode(image_data)
        if len(raw) > self.config.max_image_bytes:
            raise ToolError("Image exceeds max_image_bytes.")

        stamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"image_{stamp}_{index}_{uuid4().hex}.png"
        path = output_dir / filename
        path.write_bytes(raw)
        return path, len(raw)

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ImageGenerationArgs):
            return ToolCallDisplay(summary="image_generation")
        return ToolCallDisplay(
            summary="image_generation",
            details={
                "prompt": event.args.prompt,
                "width": event.args.width,
                "height": event.args.height,
                "steps": event.args.steps,
                "num_images": event.args.num_images,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ImageGenerationResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = f"Generated {event.result.count} image(s)"
        return ToolResultDisplay(
            success=True,
            message=message,
            details={"images": event.result.images, "count": event.result.count},
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Generating image"
