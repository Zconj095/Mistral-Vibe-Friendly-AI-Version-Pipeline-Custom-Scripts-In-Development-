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


class ContextSpokenVocalActuationConfig(BaseToolConfig):
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
    default_actuation_mode: str = Field(
        default="smooth", description="smooth, crisp, emphatic."
    )
    onset_ms: int = Field(default=80, description="Default onset duration (ms).")
    release_ms: int = Field(default=120, description="Default release duration (ms).")
    sustain_level: str = Field(
        default="medium", description="low, medium, high."
    )


class ContextSpokenVocalActuationState(BaseToolState):
    pass


class ContextSpokenVocalActuationArgs(BaseModel):
    content: str | None = Field(default=None, description="Content to analyze.")
    path: str | None = Field(default=None, description="Path to content.")
    segment_by: str | None = Field(
        default=None, description="sentences, lines, or paragraphs."
    )
    max_segments: int | None = Field(
        default=None, description="Override max segments."
    )
    actuation_mode: str | None = Field(
        default=None, description="smooth, crisp, emphatic."
    )
    onset_ms: int | None = Field(default=None, description="Override onset duration.")
    release_ms: int | None = Field(default=None, description="Override release duration.")
    sustain_level: str | None = Field(
        default=None, description="low, medium, high."
    )


class VocalActuationSegment(BaseModel):
    index: int
    text: str
    word_count: int
    onset_ms: int
    release_ms: int
    sustain_level: str
    actuation_mode: str
    cue: str
    preview: str


class ContextSpokenVocalActuationResult(BaseModel):
    segments: list[VocalActuationSegment]
    segment_count: int
    speech_opening: str
    speech_closing: str
    warnings: list[str]


class ContextSpokenVocalActuation(
    BaseTool[
        ContextSpokenVocalActuationArgs,
        ContextSpokenVocalActuationResult,
        ContextSpokenVocalActuationConfig,
        ContextSpokenVocalActuationState,
    ],
    ToolUIData[
        ContextSpokenVocalActuationArgs,
        ContextSpokenVocalActuationResult,
    ],
):
    description: ClassVar[str] = (
        "Plan vocal actuation cues for speaking segments."
    )

    async def run(
        self, args: ContextSpokenVocalActuationArgs
    ) -> ContextSpokenVocalActuationResult:
        content = self._load_content(args)
        segment_by = (args.segment_by or self.config.default_segment_by).strip().lower()
        if segment_by not in {"sentences", "lines", "paragraphs"}:
            raise ToolError("segment_by must be sentences, lines, or paragraphs.")

        max_segments = (
            args.max_segments if args.max_segments is not None else self.config.max_segments
        )
        if max_segments <= 0:
            raise ToolError("max_segments must be positive.")

        mode = (args.actuation_mode or self.config.default_actuation_mode).strip().lower()
        if mode not in {"smooth", "crisp", "emphatic"}:
            raise ToolError("actuation_mode must be smooth, crisp, or emphatic.")

        onset_ms = args.onset_ms if args.onset_ms is not None else self.config.onset_ms
        release_ms = (
            args.release_ms if args.release_ms is not None else self.config.release_ms
        )
        sustain_level = (
            args.sustain_level if args.sustain_level is not None else self.config.sustain_level
        )
        if onset_ms < 0 or release_ms < 0:
            raise ToolError("onset_ms and release_ms must be non-negative.")
        if sustain_level not in {"low", "medium", "high"}:
            raise ToolError("sustain_level must be low, medium, or high.")

        segments_raw = self._split_segments(content, segment_by)
        segments: list[VocalActuationSegment] = []
        warnings: list[str] = []

        for raw in segments_raw:
            if len(segments) >= max_segments:
                warnings.append("Segment limit reached; output truncated.")
                break
            if len(raw) < self.config.min_segment_chars:
                continue
            adjusted = self._adjust_for_text(raw, mode, onset_ms, release_ms, sustain_level)
            segments.append(
                VocalActuationSegment(
                    index=len(segments) + 1,
                    text=raw.strip(),
                    word_count=len(WORD_RE.findall(raw)),
                    onset_ms=adjusted["onset_ms"],
                    release_ms=adjusted["release_ms"],
                    sustain_level=adjusted["sustain_level"],
                    actuation_mode=adjusted["mode"],
                    cue=adjusted["cue"],
                    preview=self._preview(raw),
                )
            )

        if not segments:
            raise ToolError("No segments generated.")

        speech_opening = f"Use {mode} vocal actuation with {sustain_level} sustain."
        speech_closing = "Keep actuation cues consistent across the response."

        return ContextSpokenVocalActuationResult(
            segments=segments,
            segment_count=len(segments),
            speech_opening=speech_opening,
            speech_closing=speech_closing,
            warnings=warnings,
        )

    def _load_content(self, args: ContextSpokenVocalActuationArgs) -> str:
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

    def _adjust_for_text(
        self, text: str, mode: str, onset_ms: int, release_ms: int, sustain_level: str
    ) -> dict[str, object]:
        stripped = text.strip()
        cue = f"{mode} actuation with {sustain_level} sustain."
        if stripped.endswith("?"):
            cue = f"{mode} actuation; add light lift at the end."
            if sustain_level == "low":
                sustain_level = "medium"
        elif stripped.endswith("!"):
            cue = f"{mode} actuation; emphasize final phrase."
            if mode == "smooth":
                mode = "emphatic"
        if len(WORD_RE.findall(stripped)) > 18:
            onset_ms = max(onset_ms, 120)
            release_ms = max(release_ms, 150)
            cue = f"{mode} actuation; extend sustain for longer phrase."
        return {
            "onset_ms": onset_ms,
            "release_ms": release_ms,
            "sustain_level": sustain_level,
            "mode": mode,
            "cue": cue,
        }

    def _preview(self, text: str) -> str:
        max_chars = self.config.preview_chars
        if max_chars <= 0:
            return ""
        return text if len(text) <= max_chars else text[:max_chars]

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextSpokenVocalActuationArgs):
            return ToolCallDisplay(summary="context_spoken_vocal_actuation")
        return ToolCallDisplay(
            summary="context_spoken_vocal_actuation",
            details={
                "path": event.args.path,
                "actuation_mode": event.args.actuation_mode,
                "segment_by": event.args.segment_by,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpokenVocalActuationResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = f"Prepared {event.result.segment_count} actuation segment(s)"
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=event.result.warnings,
            details={"segment_count": event.result.segment_count},
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Preparing vocal actuation cues"
