from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import ClassVar, TYPE_CHECKING

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


WORD_RE = re.compile(r"[A-Za-z0-9_']+")
SPEAKER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _.'-]{0,40})\s*:\s*(.*)$")
SENTENCE_RE = re.compile(r"[.!?]+")


class ContextSpokenDecibelAdjustmentConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_items: int = Field(default=40, description="Maximum dialogues to evaluate.")
    max_source_bytes: int = Field(default=3_000_000, description="Maximum bytes per item.")
    max_total_bytes: int = Field(default=20_000_000, description="Max bytes across items.")
    preview_chars: int = Field(default=400, description="Preview snippet length.")
    max_segments: int = Field(default=200, description="Maximum segments per dialogue.")
    min_segment_chars: int = Field(default=20, description="Minimum segment length.")
    min_token_length: int = Field(default=2, description="Minimum token length.")
    max_word_levels: int = Field(default=30, description="Maximum word level entries.")
    max_char_emphasis: int = Field(default=20, description="Maximum char emphasis entries.")
    default_db: float = Field(default=-18.0, description="Default baseline dBFS.")
    min_db: float = Field(default=-30.0, description="Minimum target dBFS.")
    max_db: float = Field(default=-3.0, description="Maximum target dBFS.")
    max_gain_db: float = Field(default=12.0, description="Maximum gain adjustment in dB.")
    max_reduction_db: float = Field(default=6.0, description="Maximum reduction in dB.")
    short_segment_tokens: int = Field(default=6, description="Tokens considered short.")
    long_token_length: int = Field(default=8, description="Length considered long.")
    uppercase_ratio_boost: float = Field(default=0.12, description="Uppercase ratio for boost.")
    filler_ratio_reduce: float = Field(default=0.2, description="Filler ratio for reduction.")
    short_segment_boost_db: float = Field(default=2.0, description="Boost for short segments.")
    long_word_boost_db: float = Field(default=1.5, description="Boost for long words.")
    uppercase_boost_db: float = Field(default=2.0, description="Boost for uppercase emphasis.")
    punctuation_boost_db: float = Field(default=1.5, description="Boost for punctuation emphasis.")
    filler_reduce_db: float = Field(default=1.5, description="Reduction for filler-heavy segments.")
    repeated_char_boost_db: float = Field(default=1.0, description="Boost for repeated characters.")
    emphasis_words: list[str] = Field(
        default_factory=lambda: ["important", "warning", "note", "listen", "please"],
        description="Words to emphasize with a boost.",
    )
    filler_words: list[str] = Field(
        default_factory=lambda: [
            "um",
            "uh",
            "hmm",
            "like",
            "okay",
            "ok",
            "yeah",
            "yep",
            "nope",
            "well",
        ],
        description="Words that reduce gain for filler-heavy speech.",
    )


class ContextSpokenDecibelAdjustmentState(BaseToolState):
    pass


class SpokenDecibelItem(BaseModel):
    id: str | None = Field(default=None)
    name: str | None = Field(default=None)
    dialogue_id: str | None = Field(default=None)
    content: str | None = Field(default=None)
    path: str | None = Field(default=None)
    source: str | None = Field(default=None)
    gain_bias_db: float | None = Field(
        default=None, description="Manual gain bias in dB."
    )


class ContextSpokenDecibelAdjustmentArgs(BaseModel):
    items: list[SpokenDecibelItem] = Field(description="Dialogues to evaluate.")


class WordLevel(BaseModel):
    word: str
    index: int
    db_delta: float
    reasons: list[str]


class TokenCharEmphasis(BaseModel):
    word: str
    index: int
    chars: list[str]
    db_delta: float
    reason: str


class SegmentDecibelPlan(BaseModel):
    index: int
    speaker: str | None
    text: str
    token_count: int
    base_db: float
    target_db: float
    gain_db: float
    reasons: list[str]
    word_levels: list[WordLevel]
    char_emphasis: list[TokenCharEmphasis]


class SpokenDecibelInsight(BaseModel):
    index: int
    id: str | None
    name: str | None
    dialogue_id: str
    source_path: str | None
    preview: str
    segments: list[SegmentDecibelPlan]
    segment_count: int


class ContextSpokenDecibelAdjustmentResult(BaseModel):
    items: list[SpokenDecibelInsight]
    item_count: int
    truncated: bool
    warnings: list[str]
    errors: list[str]


class ContextSpokenDecibelAdjustment(
    BaseTool[
        ContextSpokenDecibelAdjustmentArgs,
        ContextSpokenDecibelAdjustmentResult,
        ContextSpokenDecibelAdjustmentConfig,
        ContextSpokenDecibelAdjustmentState,
    ],
    ToolUIData[
        ContextSpokenDecibelAdjustmentArgs,
        ContextSpokenDecibelAdjustmentResult,
    ],
):
    description: ClassVar[str] = (
        "Adjust spoken decibel targets across words and token characters."
    )

    async def run(
        self, args: ContextSpokenDecibelAdjustmentArgs
    ) -> ContextSpokenDecibelAdjustmentResult:
        items = list(args.items)
        if not items:
            raise ToolError("items is required.")

        errors: list[str] = []
        warnings: list[str] = []
        total_bytes = 0
        truncated = False
        insights: list[SpokenDecibelInsight] = []

        if len(items) > self.config.max_items:
            warnings.append("Item limit reached; truncating input list.")
            items = items[: self.config.max_items]

        for idx, item in enumerate(items, start=1):
            try:
                content, source_path, size_bytes = self._load_item(item)
                if size_bytes is None:
                    raise ToolError("Item has no content.")
                if total_bytes + size_bytes > self.config.max_total_bytes:
                    truncated = True
                    warnings.append("Budget exceeded; stopping evaluation.")
                    break
                total_bytes += size_bytes

                segments = self._segment_dialogue(content)
                if not segments:
                    raise ToolError("No segments found.")
                if len(segments) > self.config.max_segments:
                    segments = segments[: self.config.max_segments]
                    warnings.append("Segment limit reached; truncating dialogue.")

                gain_bias = item.gain_bias_db or 0.0
                plans = [
                    self._build_plan(segment, gain_bias) for segment in segments
                ]

                insights.append(
                    SpokenDecibelInsight(
                        index=len(insights) + 1,
                        id=item.id,
                        name=item.name,
                        dialogue_id=self._dialogue_id(item, idx),
                        source_path=source_path,
                        preview=self._preview(content),
                        segments=plans,
                        segment_count=len(plans),
                    )
                )
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not insights:
            raise ToolError("No insights generated.")

        return ContextSpokenDecibelAdjustmentResult(
            items=insights,
            item_count=len(insights),
            truncated=truncated,
            warnings=warnings,
            errors=errors,
        )

    def _dialogue_id(self, item: SpokenDecibelItem, idx: int) -> str:
        if item.dialogue_id:
            return item.dialogue_id
        if item.id:
            return item.id
        if item.name:
            return item.name
        return f"dialogue_{idx}"

    def _segment_dialogue(self, text: str) -> list[SegmentDecibelPlan]:
        segments: list[SegmentDecibelPlan] = []
        buffer: list[str] = []
        current_speaker: str | None = None
        segment_start: int | None = None
        segment_end: int | None = None
        pos = 0

        def flush() -> None:
            nonlocal buffer, segment_start, segment_end, current_speaker
            if not buffer:
                return
            combined = " ".join(buffer).strip()
            if len(combined) >= self.config.min_segment_chars:
                segments.append(
                    SegmentDecibelPlan(
                        index=len(segments) + 1,
                        speaker=current_speaker,
                        text=combined,
                        token_count=len(self._tokenize(combined)),
                        base_db=self.config.default_db,
                        target_db=self.config.default_db,
                        gain_db=0.0,
                        reasons=[],
                        word_levels=[],
                        char_emphasis=[],
                    )
                )
            buffer = []
            segment_start = None
            segment_end = None

        for raw_line in text.splitlines(True):
            line = raw_line.rstrip("\r\n")
            line_start = pos
            pos += len(raw_line)
            segment_end = pos
            match = SPEAKER_RE.match(line)
            if match:
                flush()
                current_speaker = match.group(1).strip()
                line_text = match.group(2).strip()
                segment_start = line_start
                if line_text:
                    buffer.append(line_text)
                continue
            if not line.strip():
                flush()
                continue
            if segment_start is None:
                segment_start = line_start
            buffer.append(line.strip())

        flush()
        if segments:
            return segments

        chunks = [chunk for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
        for chunk in chunks:
            seg_text = chunk.strip()
            if len(seg_text) < self.config.min_segment_chars:
                continue
            segments.append(
                SegmentDecibelPlan(
                    index=len(segments) + 1,
                    speaker=None,
                    text=seg_text,
                    token_count=len(self._tokenize(seg_text)),
                    base_db=self.config.default_db,
                    target_db=self.config.default_db,
                    gain_db=0.0,
                    reasons=[],
                    word_levels=[],
                    char_emphasis=[],
                )
            )

        return segments[: self.config.max_segments]

    def _build_plan(self, segment: SegmentDecibelPlan, gain_bias: float) -> SegmentDecibelPlan:
        tokens = self._tokenize(segment.text)
        token_count = len(tokens)
        uppercase_ratio = self._uppercase_ratio(tokens)
        filler_ratio = self._filler_ratio(tokens)
        punctuation_hits = self._punctuation_hits(segment.text)
        avg_token_len = self._avg_token_length(tokens)

        adjustment = 0.0
        reasons: list[str] = []
        if token_count and token_count <= self.config.short_segment_tokens:
            adjustment += self.config.short_segment_boost_db
            reasons.append("short_segment_boost")
        if avg_token_len >= self.config.long_token_length:
            adjustment += self.config.long_word_boost_db
            reasons.append("long_word_boost")
        if uppercase_ratio >= self.config.uppercase_ratio_boost:
            adjustment += self.config.uppercase_boost_db
            reasons.append("uppercase_emphasis")
        if punctuation_hits > 0:
            adjustment += self.config.punctuation_boost_db
            reasons.append("punctuation_emphasis")
        if filler_ratio >= self.config.filler_ratio_reduce:
            adjustment -= self.config.filler_reduce_db
            reasons.append("filler_reduce")

        adjustment += gain_bias
        target_db = self._clamp_db(self.config.default_db + adjustment)
        gain_db = target_db - self.config.default_db
        gain_db = self._clamp(gain_db, -self.config.max_reduction_db, self.config.max_gain_db)

        word_levels, char_emphasis = self._word_levels(tokens)
        return segment.model_copy(
            update={
                "token_count": token_count,
                "base_db": round(self.config.default_db, 2),
                "target_db": round(self.config.default_db + gain_db, 2),
                "gain_db": round(gain_db, 2),
                "reasons": reasons,
                "word_levels": word_levels,
                "char_emphasis": char_emphasis,
            }
        )

    def _word_levels(self, tokens: list[str]) -> tuple[list[WordLevel], list[TokenCharEmphasis]]:
        emphasis_words = {word.lower() for word in self.config.emphasis_words}
        word_levels: list[WordLevel] = []
        char_emphasis: list[TokenCharEmphasis] = []
        for idx, token in enumerate(tokens[: self.config.max_word_levels], start=1):
            reasons: list[str] = []
            db_delta = 0.0
            lower = token.lower()
            if token.isupper() and len(token) > 2:
                db_delta += self.config.uppercase_boost_db
                reasons.append("uppercase_word")
            if lower in emphasis_words:
                db_delta += self.config.punctuation_boost_db
                reasons.append("emphasis_word")
            repeated_chars = self._repeated_chars(token)
            if repeated_chars:
                db_delta += self.config.repeated_char_boost_db
                reasons.append("repeated_chars")
                if len(char_emphasis) < self.config.max_char_emphasis:
                    char_emphasis.append(
                        TokenCharEmphasis(
                            word=token,
                            index=idx,
                            chars=repeated_chars,
                            db_delta=round(self.config.repeated_char_boost_db, 2),
                            reason="repeated_chars",
                        )
                    )

            if reasons:
                word_levels.append(
                    WordLevel(
                        word=token,
                        index=idx,
                        db_delta=round(db_delta, 2),
                        reasons=reasons,
                    )
                )

        return word_levels, char_emphasis

    def _uppercase_ratio(self, tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        count = sum(1 for token in tokens if token.isupper() and len(token) > 1)
        return count / len(tokens)

    def _filler_ratio(self, tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        fillers = {word.lower() for word in self.config.filler_words}
        count = sum(1 for token in tokens if token in fillers)
        return count / len(tokens)

    def _punctuation_hits(self, text: str) -> int:
        return text.count("!") + text.count("?")

    def _avg_token_length(self, tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        return sum(len(token) for token in tokens) / len(tokens)

    def _repeated_chars(self, token: str) -> list[str]:
        repeats: list[str] = []
        last = ""
        streak = 1
        for char in token:
            if char == last:
                streak += 1
            else:
                if streak >= 3 and last:
                    repeats.append(last)
                last = char
                streak = 1
        if streak >= 3 and last:
            repeats.append(last)
        return repeats

    def _tokenize(self, text: str) -> list[str]:
        return [
            token
            for token in WORD_RE.findall(text)
            if len(token) >= self.config.min_token_length
        ]

    def _clamp_db(self, value: float) -> float:
        return self._clamp(value, self.config.min_db, self.config.max_db)

    def _clamp(self, value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    def _load_item(self, item: SpokenDecibelItem) -> tuple[str, str | None, int | None]:
        if item.content and item.path:
            raise ToolError("Provide content or path, not both.")
        if item.path:
            path = Path(item.path).expanduser()
            if not path.is_absolute():
                path = self.config.effective_workdir / path
            path = path.resolve()
            if not path.exists():
                raise ToolError(f"Path not found: {path}")
            if path.is_dir():
                raise ToolError(f"Path is a directory: {path}")
            size = path.stat().st_size
            if size > self.config.max_source_bytes:
                raise ToolError(
                    f"{path} exceeds max_source_bytes ({size} > {self.config.max_source_bytes})."
                )
            return path.read_text("utf-8", errors="ignore"), str(path), size
        if item.content is not None:
            size = len(item.content.encode("utf-8"))
            if size > self.config.max_source_bytes:
                raise ToolError(
                    f"content exceeds max_source_bytes ({size} > {self.config.max_source_bytes})."
                )
            return item.content, None, size
        return "", None, 0

    def _preview(self, text: str) -> str:
        max_chars = self.config.preview_chars
        if max_chars <= 0:
            return ""
        return text if len(text) <= max_chars else text[:max_chars]

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextSpokenDecibelAdjustmentArgs):
            return ToolCallDisplay(summary="context_spoken_decibel_adjustment")
        if not event.args.items:
            return ToolCallDisplay(summary="context_spoken_decibel_adjustment")
        first = event.args.items[0]
        return ToolCallDisplay(
            summary="context_spoken_decibel_adjustment",
            details={
                "item_count": len(event.args.items),
                "id": first.id,
                "name": first.name,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpokenDecibelAdjustmentResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=(
                f"Adjusted decibel targets for {event.result.item_count} dialogue(s)"
            ),
            warnings=event.result.warnings,
            details={
                "item_count": event.result.item_count,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Adjusting spoken decibel targets"
