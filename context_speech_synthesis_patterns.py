from __future__ import annotations

import re
from collections import Counter
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


class PatternPreset(BaseModel):
    name: str
    rate: str
    pitch: str
    volume: str
    pause_ms: int
    description: str = ""


class ContextSpeechSynthesisPatternsConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_source_bytes: int = Field(
        default=5_000_000, description="Maximum bytes per content."
    )
    max_segments: int = Field(default=200, description="Maximum segments to return.")
    preview_chars: int = Field(default=240, description="Preview snippet length.")
    default_segment_by: str = Field(
        default="sentences", description="sentences, lines, or paragraphs."
    )
    min_segment_chars: int = Field(default=12, description="Minimum segment length.")
    min_token_length: int = Field(default=2, description="Minimum token length.")
    max_keywords: int = Field(default=6, description="Max keywords per segment.")
    default_pattern: str = Field(default="neutral", description="Default pattern name.")
    pattern_presets: list[PatternPreset] = Field(
        default_factory=lambda: [
            PatternPreset(
                name="neutral",
                rate="medium",
                pitch="medium",
                volume="medium",
                pause_ms=250,
                description="Balanced, clear delivery.",
            ),
            PatternPreset(
                name="calm",
                rate="slow",
                pitch="low",
                volume="soft",
                pause_ms=500,
                description="Calm and steady pacing.",
            ),
            PatternPreset(
                name="energetic",
                rate="fast",
                pitch="high",
                volume="loud",
                pause_ms=200,
                description="High energy and upbeat.",
            ),
            PatternPreset(
                name="serious",
                rate="medium",
                pitch="low",
                volume="medium",
                pause_ms=350,
                description="Measured and focused.",
            ),
            PatternPreset(
                name="instructional",
                rate="medium",
                pitch="medium",
                volume="medium",
                pause_ms=400,
                description="Clear, step-by-step delivery.",
            ),
        ],
        description="Named vocal pattern presets for synthesis.",
    )


class ContextSpeechSynthesisPatternsState(BaseToolState):
    pass


class ContextSpeechSynthesisPatternsArgs(BaseModel):
    content: str | None = Field(default=None, description="Content to synthesize.")
    path: str | None = Field(default=None, description="Path to content.")
    segment_by: str | None = Field(
        default=None, description="sentences, lines, or paragraphs."
    )
    max_segments: int | None = Field(
        default=None, description="Override max segments."
    )
    pattern_name: str | None = Field(
        default=None, description="Single pattern name to apply."
    )
    pattern_sequence: list[str] | None = Field(
        default=None, description="Pattern names to cycle by segment."
    )
    rate: str | None = Field(default=None, description="Override prosody rate.")
    pitch: str | None = Field(default=None, description="Override prosody pitch.")
    volume: str | None = Field(default=None, description="Override prosody volume.")
    pause_ms: int | None = Field(default=None, description="Override pause duration.")


class SynthesisSegment(BaseModel):
    index: int
    text: str
    pattern: str
    rate: str
    pitch: str
    volume: str
    pause_ms: int
    keywords: list[str]
    cue: str
    ssml: str


class ContextSpeechSynthesisPatternsResult(BaseModel):
    segments: list[SynthesisSegment]
    segment_count: int
    ssml: str
    speech_opening: str
    speech_closing: str
    warnings: list[str]


class ContextSpeechSynthesisPatterns(
    BaseTool[
        ContextSpeechSynthesisPatternsArgs,
        ContextSpeechSynthesisPatternsResult,
        ContextSpeechSynthesisPatternsConfig,
        ContextSpeechSynthesisPatternsState,
    ],
    ToolUIData[
        ContextSpeechSynthesisPatternsArgs,
        ContextSpeechSynthesisPatternsResult,
    ],
):
    description: ClassVar[str] = (
        "Generate speech synthesis markup for vocal patterns."
    )

    async def run(
        self, args: ContextSpeechSynthesisPatternsArgs
    ) -> ContextSpeechSynthesisPatternsResult:
        content = self._load_content(args)
        segment_by = (args.segment_by or self.config.default_segment_by).strip().lower()
        if segment_by not in {"sentences", "lines", "paragraphs"}:
            raise ToolError("segment_by must be sentences, lines, or paragraphs.")

        max_segments = (
            args.max_segments if args.max_segments is not None else self.config.max_segments
        )
        if max_segments <= 0:
            raise ToolError("max_segments must be positive.")

        presets = {preset.name.lower(): preset for preset in self.config.pattern_presets}
        warnings: list[str] = []
        default_pattern = self._resolve_pattern_name(args.pattern_name, presets, warnings)
        pattern_sequence = self._resolve_pattern_sequence(
            args.pattern_sequence, presets, warnings
        )

        segments_raw = self._split_segments(content, segment_by)
        segments: list[SynthesisSegment] = []

        for idx, text in enumerate(segments_raw, start=1):
            if len(segments) >= max_segments:
                warnings.append("Segment limit reached; output truncated.")
                break
            if len(text) < self.config.min_segment_chars:
                continue
            pattern_name = self._pattern_for_index(
                idx, default_pattern, pattern_sequence, presets
            )
            preset = presets.get(pattern_name.lower())
            if preset is None:
                preset = presets[self.config.default_pattern.lower()]
            rate = args.rate or preset.rate
            pitch = args.pitch or preset.pitch
            volume = args.volume or preset.volume
            pause_ms = args.pause_ms if args.pause_ms is not None else preset.pause_ms
            keywords = self._extract_keywords(text, self.config.max_keywords)
            cue = self._build_cue(pattern_name, rate, pitch, volume, keywords)
            ssml = self._segment_ssml(text, rate, pitch, volume, pause_ms)
            segments.append(
                SynthesisSegment(
                    index=len(segments) + 1,
                    text=text.strip(),
                    pattern=pattern_name,
                    rate=rate,
                    pitch=pitch,
                    volume=volume,
                    pause_ms=pause_ms,
                    keywords=keywords,
                    cue=cue,
                    ssml=ssml,
                )
            )

        if not segments:
            raise ToolError("No segments generated.")

        ssml = self._build_ssml(segments)
        speech_opening = f"Use vocal pattern: {segments[0].pattern}."
        speech_closing = "Keep the selected vocal pattern consistent."

        return ContextSpeechSynthesisPatternsResult(
            segments=segments,
            segment_count=len(segments),
            ssml=ssml,
            speech_opening=speech_opening,
            speech_closing=speech_closing,
            warnings=warnings,
        )

    def _load_content(self, args: ContextSpeechSynthesisPatternsArgs) -> str:
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

    def _resolve_pattern_name(
        self,
        pattern_name: str | None,
        presets: dict[str, PatternPreset],
        warnings: list[str],
    ) -> str:
        if not pattern_name:
            return self.config.default_pattern
        key = pattern_name.strip().lower()
        if key not in presets:
            warnings.append(f"Unknown pattern '{pattern_name}', using default.")
            return self.config.default_pattern
        return presets[key].name

    def _resolve_pattern_sequence(
        self,
        sequence: list[str] | None,
        presets: dict[str, PatternPreset],
        warnings: list[str],
    ) -> list[str]:
        if not sequence:
            return []
        resolved: list[str] = []
        for entry in sequence:
            key = entry.strip().lower()
            if key in presets:
                resolved.append(presets[key].name)
            else:
                warnings.append(f"Unknown pattern '{entry}' in sequence.")
        return resolved

    def _pattern_for_index(
        self,
        index: int,
        default_pattern: str,
        sequence: list[str],
        presets: dict[str, PatternPreset],
    ) -> str:
        if sequence:
            return sequence[(index - 1) % len(sequence)]
        if default_pattern.lower() in presets:
            return presets[default_pattern.lower()].name
        return self.config.default_pattern

    def _extract_keywords(self, text: str, max_items: int) -> list[str]:
        tokens = [
            token.lower()
            for token in WORD_RE.findall(text)
            if len(token) >= self.config.min_token_length
        ]
        return [word for word, _ in Counter(tokens).most_common(max_items)]

    def _build_cue(
        self, pattern: str, rate: str, pitch: str, volume: str, keywords: list[str]
    ) -> str:
        parts = [
            f"pattern {pattern}",
            f"rate {rate}",
            f"pitch {pitch}",
            f"volume {volume}",
        ]
        if keywords:
            parts.append(f"emphasize {', '.join(keywords[:5])}")
        return "; ".join(parts) + "."

    def _escape_ssml(self, text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    def _segment_ssml(
        self, text: str, rate: str, pitch: str, volume: str, pause_ms: int
    ) -> str:
        escaped = self._escape_ssml(text.strip())
        return (
            f"<prosody rate=\"{rate}\" pitch=\"{pitch}\" volume=\"{volume}\">"
            f"{escaped}</prosody><break time=\"{pause_ms}ms\"/>"
        )

    def _build_ssml(self, segments: list[SynthesisSegment]) -> str:
        parts = ["<speak>"]
        for segment in segments:
            parts.append(segment.ssml)
        parts.append("</speak>")
        return "\n".join(parts)

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextSpeechSynthesisPatternsArgs):
            return ToolCallDisplay(summary="context_speech_synthesis_patterns")
        return ToolCallDisplay(
            summary="context_speech_synthesis_patterns",
            details={
                "path": event.args.path,
                "pattern_name": event.args.pattern_name,
                "pattern_sequence": event.args.pattern_sequence,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpeechSynthesisPatternsResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = f"Prepared {event.result.segment_count} synthesis segment(s)"
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=event.result.warnings,
            details={"segment_count": event.result.segment_count},
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Preparing speech synthesis patterns"
