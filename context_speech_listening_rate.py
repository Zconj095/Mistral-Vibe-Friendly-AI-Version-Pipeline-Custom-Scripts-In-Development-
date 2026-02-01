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


LINE_TS_RE = re.compile(
    r"^\s*(?:\\[|\\()?(?P<ts>(?:\\d{1,2}:)?\\d{1,2}:\\d{2}(?:\\.\\d{1,3})?)"
    r"(?:\\]|\\))?\\s*"
)
WORD_RE = re.compile(r"[A-Za-z0-9_']+")


class ContextSpeechListeningRateConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_source_bytes: int = Field(
        default=50_000_000, description="Maximum bytes per transcript."
    )
    preview_chars: int = Field(default=200, description="Preview snippet length.")
    default_wpm: float = Field(
        default=160.0, description="Default words-per-minute for estimates."
    )
    slow_wpm: float = Field(default=120.0, description="Slow speech threshold.")
    fast_wpm: float = Field(default=180.0, description="Fast speech threshold.")
    very_fast_wpm: float = Field(default=220.0, description="Very fast threshold.")


class ContextSpeechListeningRateState(BaseToolState):
    pass


class ContextSpeechListeningRateArgs(BaseModel):
    content: str | None = Field(default=None, description="Transcript content.")
    path: str | None = Field(default=None, description="Path to transcript.")
    duration_seconds: float | None = Field(
        default=None, description="Total duration in seconds."
    )
    target_wpm: float | None = Field(
        default=None, description="Target listening rate for playback."
    )


class ListeningSegment(BaseModel):
    index: int
    start_time: float
    end_time: float | None
    duration_seconds: float | None
    word_count: int
    words_per_minute: float | None
    preview: str
    estimated: bool


class ContextSpeechListeningRateResult(BaseModel):
    total_words: int
    duration_seconds: float | None
    words_per_minute: float | None
    words_per_second: float | None
    rate_label: str
    playback_speed: float | None
    estimated: bool
    segments: list[ListeningSegment]
    warnings: list[str]


class ContextSpeechListeningRate(
    BaseTool[
        ContextSpeechListeningRateArgs,
        ContextSpeechListeningRateResult,
        ContextSpeechListeningRateConfig,
        ContextSpeechListeningRateState,
    ],
    ToolUIData[ContextSpeechListeningRateArgs, ContextSpeechListeningRateResult],
):
    description: ClassVar[str] = (
        "Estimate speech listening rate from a transcript and timing."
    )

    async def run(
        self, args: ContextSpeechListeningRateArgs
    ) -> ContextSpeechListeningRateResult:
        content = self._load_content(args)
        lines = content.splitlines()
        warnings: list[str] = []

        segments = self._build_segments(lines, content)
        total_words = sum(seg.word_count for seg in segments)

        duration_seconds, estimated = self._resolve_total_duration(
            segments, args.duration_seconds
        )
        if duration_seconds is None or duration_seconds <= 0:
            warnings.append("Unable to determine duration; rate unavailable.")
            duration_seconds = None

        words_per_minute = None
        words_per_second = None
        if duration_seconds and duration_seconds > 0:
            words_per_second = total_words / duration_seconds
            words_per_minute = words_per_second * 60.0

        rate_label = self._rate_label(words_per_minute)
        playback_speed = None
        if words_per_minute and args.target_wpm:
            if args.target_wpm <= 0:
                raise ToolError("target_wpm must be positive.")
            playback_speed = args.target_wpm / words_per_minute

        if estimated:
            warnings.append("Duration estimated from default_wpm.")

        return ContextSpeechListeningRateResult(
            total_words=total_words,
            duration_seconds=duration_seconds,
            words_per_minute=words_per_minute,
            words_per_second=words_per_second,
            rate_label=rate_label,
            playback_speed=playback_speed,
            estimated=estimated,
            segments=segments,
            warnings=warnings,
        )

    def _load_content(self, args: ContextSpeechListeningRateArgs) -> str:
        if args.content and args.path:
            raise ToolError("Provide content or path, not both.")
        if args.content is None and args.path is None:
            raise ToolError("Provide content or path.")

        if args.content is not None:
            data = args.content.encode("utf-8")
            if len(data) > self.config.max_source_bytes:
                raise ToolError(
                    f"content exceeds max_source_bytes ({len(data)} > {self.config.max_source_bytes})."
                )
            return args.content

        path = self._resolve_path(args.path or "")
        size = path.stat().st_size
        if size > self.config.max_source_bytes:
            raise ToolError(
                f"{path} exceeds max_source_bytes ({size} > {self.config.max_source_bytes})."
            )
        return path.read_text("utf-8", errors="ignore")

    def _resolve_path(self, raw_path: str) -> Path:
        if not raw_path.strip():
            raise ToolError("Path cannot be empty.")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.config.effective_workdir / path
        path = path.resolve()
        if not path.exists():
            raise ToolError(f"Path not found: {path}")
        if path.is_dir():
            raise ToolError(f"Path is a directory: {path}")
        return path

    def _build_segments(self, lines: list[str], content: str) -> list[ListeningSegment]:
        segments: list[ListeningSegment] = []
        current_ts: float | None = None
        buffer: list[str] = []

        def flush(next_ts: float | None) -> None:
            nonlocal current_ts, buffer, segments
            if current_ts is None:
                return
            text = "\n".join(buffer).strip()
            segment = self._segment_from_text(
                len(segments) + 1, current_ts, next_ts, text, estimated=False
            )
            segments.append(segment)

        for line in lines:
            match = LINE_TS_RE.match(line)
            if match:
                ts = self._parse_timestamp(match.group("ts"))
                flush(ts)
                current_ts = ts
                buffer = [line[match.end() :]]
            else:
                buffer.append(line)

        if current_ts is not None:
            flush(None)

        if not segments:
            text = content.strip()
            segments.append(
                self._segment_from_text(
                    1, 0.0, None, text, estimated=True
                )
            )

        return segments

    def _segment_from_text(
        self,
        index: int,
        start_time: float,
        end_time: float | None,
        text: str,
        *,
        estimated: bool,
    ) -> ListeningSegment:
        words = WORD_RE.findall(text)
        word_count = len(words)
        duration_seconds = None
        words_per_minute = None
        if end_time is not None and end_time > start_time:
            duration_seconds = end_time - start_time
            words_per_minute = (word_count / duration_seconds) * 60.0
        preview = self._preview(text)
        return ListeningSegment(
            index=index,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration_seconds,
            word_count=word_count,
            words_per_minute=words_per_minute,
            preview=preview,
            estimated=estimated,
        )

    def _resolve_total_duration(
        self, segments: list[ListeningSegment], duration_seconds: float | None
    ) -> tuple[float | None, bool]:
        estimated = False
        if duration_seconds is not None:
            if duration_seconds <= 0:
                raise ToolError("duration_seconds must be positive.")
            self._apply_end_time(segments, duration_seconds)
            return duration_seconds, estimated

        if len(segments) >= 2:
            last_end = self._apply_estimated_end(segments)
            return last_end, True

        if segments:
            duration_seconds = self._estimate_duration(segments[0].word_count)
            self._apply_end_time(segments, duration_seconds)
            return duration_seconds, True

        return None, estimated

    def _apply_end_time(self, segments: list[ListeningSegment], end_time: float) -> None:
        if not segments:
            return
        last = segments[-1]
        if last.end_time is None or last.end_time < end_time:
            last.end_time = end_time
            last.duration_seconds = end_time - last.start_time
            if last.duration_seconds > 0:
                last.words_per_minute = (
                    last.word_count / last.duration_seconds * 60.0
                )

    def _apply_estimated_end(self, segments: list[ListeningSegment]) -> float:
        if not segments:
            return 0.0
        for idx, segment in enumerate(segments[:-1]):
            next_start = segments[idx + 1].start_time
            segment.end_time = next_start
            segment.duration_seconds = next_start - segment.start_time
            if segment.duration_seconds > 0:
                segment.words_per_minute = (
                    segment.word_count / segment.duration_seconds * 60.0
                )
        last = segments[-1]
        estimated_duration = self._estimate_duration(last.word_count)
        last.end_time = last.start_time + estimated_duration
        last.duration_seconds = estimated_duration
        last.words_per_minute = (
            last.word_count / estimated_duration * 60.0
            if estimated_duration > 0
            else None
        )
        last.estimated = True
        return last.end_time

    def _estimate_duration(self, word_count: int) -> float:
        if self.config.default_wpm <= 0:
            raise ToolError("default_wpm must be positive.")
        return (word_count / self.config.default_wpm) * 60.0

    def _parse_timestamp(self, raw: str) -> float:
        parts = raw.split(":")
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        raise ToolError(f"Invalid timestamp: {raw}")

    def _rate_label(self, words_per_minute: float | None) -> str:
        if words_per_minute is None:
            return "unknown"
        if words_per_minute <= self.config.slow_wpm:
            return "slow"
        if words_per_minute <= self.config.fast_wpm:
            return "standard"
        if words_per_minute <= self.config.very_fast_wpm:
            return "fast"
        return "very_fast"

    def _preview(self, text: str) -> str:
        max_chars = self.config.preview_chars
        if max_chars <= 0:
            return ""
        return text if len(text) <= max_chars else text[:max_chars]

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextSpeechListeningRateArgs):
            return ToolCallDisplay(summary="context_speech_listening_rate")
        return ToolCallDisplay(
            summary="context_speech_listening_rate",
            details={
                "path": event.args.path,
                "duration_seconds": event.args.duration_seconds,
                "target_wpm": event.args.target_wpm,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpeechListeningRateResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = "Computed speech listening rate"
        if event.result.words_per_minute:
            message = (
                f"Listening rate {event.result.words_per_minute:.1f} wpm "
                f"({event.result.rate_label})"
            )
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=event.result.warnings,
            details={
                "words_per_minute": event.result.words_per_minute,
                "duration_seconds": event.result.duration_seconds,
                "segments": event.result.segments,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Calculating speech listening rate"
