from __future__ import annotations

import re
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


SENTENCE_RE = re.compile(r"[^.!?]+[.!?]*", re.S)
WORD_RE = re.compile(r"[A-Za-z0-9_']+")


class ContextSpokenVocalAttenuationConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_source_bytes: int = Field(
        default=5_000_000, description="Maximum bytes per content."
    )
    max_segments: int = Field(default=200, description="Maximum segments to return.")
    preview_chars: int = Field(default=200, description="Preview snippet length.")
    default_segment_by: str = Field(
        default="sentences", description="sentences, lines, or paragraphs."
    )
    min_segment_chars: int = Field(default=12, description="Minimum segment length.")
    default_mode: str = Field(default="gentle", description="gentle, strong, dynamic.")
    base_volume: str = Field(default="medium", description="soft, medium, loud.")
    min_volume: str = Field(default="soft", description="soft, medium, loud.")
    max_volume: str = Field(default="medium", description="soft, medium, loud.")
    gentle_db: int = Field(default=6, description="Attenuation for gentle mode.")
    strong_db: int = Field(default=10, description="Attenuation for strong mode.")
    dynamic_db: int = Field(default=8, description="Attenuation for dynamic mode.")
    max_db: int = Field(default=18, description="Maximum attenuation.")


class ContextSpokenVocalAttenuationState(BaseToolState):
    pass


class ContextSpokenVocalAttenuationArgs(BaseModel):
    content: str | None = Field(default=None, description="Content to analyze.")
    path: str | None = Field(default=None, description="Path to content.")
    segment_by: str | None = Field(
        default=None, description="sentences, lines, or paragraphs."
    )
    max_segments: int | None = Field(
        default=None, description="Override max segments."
    )
    attenuation_mode: str | None = Field(
        default=None, description="gentle, strong, dynamic."
    )
    base_volume: str | None = Field(
        default=None, description="soft, medium, loud."
    )
    min_volume: str | None = Field(
        default=None, description="soft, medium, loud."
    )
    max_volume: str | None = Field(
        default=None, description="soft, medium, loud."
    )


class VocalAttenuationSegment(BaseModel):
    index: int
    text: str
    word_count: int
    attenuation_db: int
    target_volume: str
    mode: str
    cue: str
    preview: str


class ContextSpokenVocalAttenuationResult(BaseModel):
    segments: list[VocalAttenuationSegment]
    segment_count: int
    average_attenuation_db: float
    speech_opening: str
    speech_closing: str
    warnings: list[str]


class ContextSpokenVocalAttenuation(
    BaseTool[
        ContextSpokenVocalAttenuationArgs,
        ContextSpokenVocalAttenuationResult,
        ContextSpokenVocalAttenuationConfig,
        ContextSpokenVocalAttenuationState,
    ],
    ToolUIData[
        ContextSpokenVocalAttenuationArgs, ContextSpokenVocalAttenuationResult
    ],
):
    description: ClassVar[str] = (
        "Plan vocal attenuation cues for speaking segments."
    )

    async def run(
        self, args: ContextSpokenVocalAttenuationArgs
    ) -> ContextSpokenVocalAttenuationResult:
        content = self._load_content(args)
        segment_by = (args.segment_by or self.config.default_segment_by).strip().lower()
        if segment_by not in {"sentences", "lines", "paragraphs"}:
            raise ToolError("segment_by must be sentences, lines, or paragraphs.")

        max_segments = (
            args.max_segments if args.max_segments is not None else self.config.max_segments
        )
        if max_segments <= 0:
            raise ToolError("max_segments must be positive.")

        mode = (args.attenuation_mode or self.config.default_mode).strip().lower()
        if mode not in {"gentle", "strong", "dynamic"}:
            raise ToolError("attenuation_mode must be gentle, strong, or dynamic.")

        base_volume = (args.base_volume or self.config.base_volume).strip().lower()
        min_volume = (args.min_volume or self.config.min_volume).strip().lower()
        max_volume = (args.max_volume or self.config.max_volume).strip().lower()
        for label in (base_volume, min_volume, max_volume):
            if label not in {"soft", "medium", "loud"}:
                raise ToolError("volume values must be soft, medium, or loud.")

        segments_raw = self._split_segments(content, segment_by)
        segments: list[VocalAttenuationSegment] = []
        warnings: list[str] = []
        total_db = 0

        for raw in segments_raw:
            if len(segments) >= max_segments:
                warnings.append("Segment limit reached; output truncated.")
                break
            if len(raw) < self.config.min_segment_chars:
                continue
            segment = self._build_segment(
                len(segments) + 1,
                raw,
                mode,
                base_volume,
                min_volume,
                max_volume,
            )
            segments.append(segment)
            total_db += segment.attenuation_db

        if not segments:
            raise ToolError("No segments generated.")

        average_db = total_db / len(segments)
        speech_opening = f"Use {mode} vocal attenuation with {base_volume} baseline."
        speech_closing = "Maintain attenuation cues consistently across segments."

        return ContextSpokenVocalAttenuationResult(
            segments=segments,
            segment_count=len(segments),
            average_attenuation_db=average_db,
            speech_opening=speech_opening,
            speech_closing=speech_closing,
            warnings=warnings,
        )

    def _load_content(self, args: ContextSpokenVocalAttenuationArgs) -> str:
        if args.content and args.path:
            raise ToolError("Provide content or path, not both.")
        if args.content is not None:
            data = args.content.encode("utf-8")
            if len(data) > self.config.max_source_bytes:
                raise ToolError(
                    f"content exceeds max_source_bytes ({len(data)} > {self.config.max_source_bytes})."
                )
            return args.content
        if not args.path:
            raise ToolError("Provide content or path.")
        path = self._resolve_path(args.path)
        size = path.stat().st_size
        if size > self.config.max_source_bytes:
            raise ToolError(
                f"{path} exceeds max_source_bytes ({size} > {self.config.max_source_bytes})."
            )
        return path.read_text("utf-8", errors="ignore")

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.config.effective_workdir / path
        path = path.resolve()
        if not path.exists():
            raise ToolError(f"Path not found: {path}")
        if path.is_dir():
            raise ToolError(f"Path is a directory: {path}")
        return path

    def _split_segments(self, text: str, mode: str) -> list[str]:
        if mode == "lines":
            return [line for line in text.splitlines() if line.strip()]
        if mode == "paragraphs":
            return [chunk for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
        return [seg.strip() for seg in SENTENCE_RE.findall(text) if seg.strip()]

    def _build_segment(
        self,
        index: int,
        text: str,
        mode: str,
        base_volume: str,
        min_volume: str,
        max_volume: str,
    ) -> VocalAttenuationSegment:
        stripped = text.strip()
        word_count = len(WORD_RE.findall(stripped))
        attenuation_db = self._base_db(mode)
        cue = f"{mode} attenuation at {attenuation_db}dB."

        if stripped.endswith("!"):
            attenuation_db = max(0, attenuation_db - 2)
            cue = f"{mode} attenuation; hold energy on emphasis."
        elif stripped.endswith("?"):
            attenuation_db = max(0, attenuation_db - 1)
            cue = f"{mode} attenuation; keep question inflection audible."

        if word_count > 18:
            attenuation_db = min(self.config.max_db, attenuation_db + 2)
            cue = f"{mode} attenuation; fade slightly on longer phrase."
        elif word_count < 8:
            attenuation_db = max(0, attenuation_db - 1)

        target_volume = self._volume_for_db(
            attenuation_db, base_volume, min_volume, max_volume
        )
        cue = f"{cue} Target volume {target_volume}."

        return VocalAttenuationSegment(
            index=index,
            text=stripped,
            word_count=word_count,
            attenuation_db=attenuation_db,
            target_volume=target_volume,
            mode=mode,
            cue=cue,
            preview=self._preview(stripped),
        )

    def _base_db(self, mode: str) -> int:
        if mode == "strong":
            return self.config.strong_db
        if mode == "dynamic":
            return self.config.dynamic_db
        return self.config.gentle_db

    def _volume_for_db(
        self, attenuation_db: int, base_volume: str, min_volume: str, max_volume: str
    ) -> str:
        if attenuation_db <= 4:
            return max_volume
        if attenuation_db <= 8:
            return base_volume
        return min_volume

    def _preview(self, text: str) -> str:
        max_chars = self.config.preview_chars
        if max_chars <= 0:
            return ""
        return text if len(text) <= max_chars else text[:max_chars]

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextSpokenVocalAttenuationArgs):
            return ToolCallDisplay(summary="context_spoken_vocal_attenuation")
        return ToolCallDisplay(
            summary="context_spoken_vocal_attenuation",
            details={
                "path": event.args.path,
                "attenuation_mode": event.args.attenuation_mode,
                "segment_by": event.args.segment_by,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpokenVocalAttenuationResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = (
            f"Prepared {event.result.segment_count} attenuation segment(s)"
        )
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=event.result.warnings,
            details={"segment_count": event.result.segment_count},
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Preparing vocal attenuation cues"
