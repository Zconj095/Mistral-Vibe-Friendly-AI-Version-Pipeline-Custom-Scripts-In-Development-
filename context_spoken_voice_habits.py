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


class VoiceHabit(BaseModel):
    name: str
    description: str
    cue: str


class ContextSpokenVoiceHabitsConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_source_bytes: int = Field(
        default=5_000_000, description="Maximum bytes per content."
    )
    preview_chars: int = Field(default=240, description="Preview snippet length.")
    max_habits: int = Field(default=8, description="Maximum habits to return.")
    min_token_length: int = Field(default=2, description="Minimum token length.")
    default_habits: list[VoiceHabit] = Field(
        default_factory=lambda: [
            VoiceHabit(
                name="steady_pace",
                description="Maintain a steady vocal pace across sentences.",
                cue="Keep the pace steady and avoid rushing key phrases.",
            ),
            VoiceHabit(
                name="clear_articulation",
                description="Articulate consonants and vowels with clarity.",
                cue="Use crisp diction on important terms.",
            ),
            VoiceHabit(
                name="controlled_emphasis",
                description="Emphasize key words without over-stressing fillers.",
                cue="Highlight the main nouns and verbs.",
            ),
            VoiceHabit(
                name="pause_cadence",
                description="Insert short pauses at commas and long pauses at sentence ends.",
                cue="Use brief pauses for commas and longer pauses for sentence endings.",
            ),
            VoiceHabit(
                name="pitch_variation",
                description="Vary vocal pitch slightly to avoid monotone delivery.",
                cue="Add light pitch changes on transitions.",
            ),
            VoiceHabit(
                name="volume_balance",
                description="Keep vocal volume balanced and consistent.",
                cue="Avoid sudden volume spikes across consecutive lines.",
            ),
            VoiceHabit(
                name="question_lift",
                description="Lift vocal inflection on questions.",
                cue="Slightly raise pitch at the end of questions.",
            ),
            VoiceHabit(
                name="transition_signposts",
                description="Use short vocal signposts when changing topics.",
                cue="Add a brief vocal marker before new sections.",
            ),
        ],
        description="Default voice habits to use when no analysis is provided.",
    )


class ContextSpokenVoiceHabitsState(BaseToolState):
    pass


class ContextSpokenVoiceHabitsArgs(BaseModel):
    content: str | None = Field(default=None, description="Content to analyze.")
    path: str | None = Field(default=None, description="Path to content.")
    style: str | None = Field(
        default=None, description="Optional style name for habits."
    )
    max_habits: int | None = Field(
        default=None, description="Override max habits."
    )
    include_analysis: bool = Field(
        default=True, description="Derive habits from content analysis."
    )


class VoiceHabitInsight(BaseModel):
    habits: list[VoiceHabit]
    summary_keywords: list[str]
    analysis_notes: list[str]
    speech_opening: str
    speech_closing: str
    warnings: list[str]


class ContextSpokenVoiceHabits(
    BaseTool[
        ContextSpokenVoiceHabitsArgs,
        VoiceHabitInsight,
        ContextSpokenVoiceHabitsConfig,
        ContextSpokenVoiceHabitsState,
    ],
    ToolUIData[ContextSpokenVoiceHabitsArgs, VoiceHabitInsight],
):
    description: ClassVar[str] = "Generate voice-only speaking habits and cues."

    async def run(self, args: ContextSpokenVoiceHabitsArgs) -> VoiceHabitInsight:
        warnings: list[str] = []
        analysis_notes: list[str] = []

        max_habits = args.max_habits if args.max_habits is not None else self.config.max_habits
        if max_habits <= 0:
            raise ToolError("max_habits must be positive.")

        habits = list(self.config.default_habits)
        summary_keywords: list[str] = []

        if args.include_analysis and (args.content or args.path):
            content = self._load_content(args)
            summary_keywords = self._top_keywords(content, max_items=6)
            analysis_notes.extend(self._analyze_voice(content))
            habits = self._adjust_habits(habits, content)
        elif args.include_analysis:
            warnings.append("No content provided; using default voice habits.")

        if args.style:
            habits = self._apply_style(habits, args.style)

        habits = habits[:max_habits]
        speech_opening = self._speech_opening(args.style, summary_keywords)
        speech_closing = "Keep vocal habits consistent through the full response."

        return VoiceHabitInsight(
            habits=habits,
            summary_keywords=summary_keywords,
            analysis_notes=analysis_notes,
            speech_opening=speech_opening,
            speech_closing=speech_closing,
            warnings=warnings,
        )

    def _load_content(self, args: ContextSpokenVoiceHabitsArgs) -> str:
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

    def _top_keywords(self, text: str, max_items: int) -> list[str]:
        tokens = [
            token.lower()
            for token in WORD_RE.findall(text)
            if len(token) >= self.config.min_token_length
            and token.lower() not in STOPWORDS
        ]
        return [word for word, _ in Counter(tokens).most_common(max_items)]

    def _analyze_voice(self, text: str) -> list[str]:
        sentences = [seg.strip() for seg in SENTENCE_RE.findall(text) if seg.strip()]
        if not sentences:
            return ["No sentence structure detected."]
        avg_len = sum(len(WORD_RE.findall(s)) for s in sentences) / len(sentences)
        notes = [f"Average sentence length: {avg_len:.1f} words."]
        question_count = sum(1 for s in sentences if s.endswith("?"))
        if question_count:
            notes.append(f"Questions detected: {question_count}.")
        comma_count = text.count(",")
        if comma_count:
            notes.append(f"Comma count: {comma_count}.")
        return notes

    def _adjust_habits(self, habits: list[VoiceHabit], text: str) -> list[VoiceHabit]:
        avg_len = self._average_sentence_len(text)
        if avg_len > 20:
            habits.insert(
                0,
                VoiceHabit(
                    name="phrase_breaks",
                    description="Insert additional phrase breaks in longer sentences.",
                    cue="Add short pauses mid-sentence to keep clarity.",
                ),
            )
        if text.count("?") > 0:
            habits.append(
                VoiceHabit(
                    name="question_focus",
                    description="Apply upward inflection on question endings.",
                    cue="Lift pitch at the end of questions.",
                )
            )
        return habits

    def _average_sentence_len(self, text: str) -> float:
        sentences = [seg.strip() for seg in SENTENCE_RE.findall(text) if seg.strip()]
        if not sentences:
            return 0.0
        return sum(len(WORD_RE.findall(s)) for s in sentences) / len(sentences)

    def _apply_style(self, habits: list[VoiceHabit], style: str) -> list[VoiceHabit]:
        style_key = style.strip().lower()
        if style_key == "formal":
            habits.insert(
                0,
                VoiceHabit(
                    name="formal_pacing",
                    description="Use measured pacing with careful articulation.",
                    cue="Keep a measured pace and crisp diction.",
                ),
            )
        elif style_key == "casual":
            habits.insert(
                0,
                VoiceHabit(
                    name="casual_flow",
                    description="Keep the voice relaxed and conversational.",
                    cue="Use a relaxed pace with friendly inflection.",
                ),
            )
        elif style_key == "instructional":
            habits.insert(
                0,
                VoiceHabit(
                    name="instructional_steps",
                    description="Use clear vocal signposts between steps.",
                    cue="Pause briefly before each step.",
                ),
            )
        return habits

    def _speech_opening(self, style: str | None, keywords: list[str]) -> str:
        base = "Start with clear vocal habits."
        if style:
            base = f"Start with {style} voice habits."
        if keywords:
            base += f" Key topics: {', '.join(keywords[:4])}."
        return base

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextSpokenVoiceHabitsArgs):
            return ToolCallDisplay(summary="context_spoken_voice_habits")
        return ToolCallDisplay(
            summary="context_spoken_voice_habits",
            details={
                "path": event.args.path,
                "style": event.args.style,
                "include_analysis": event.args.include_analysis,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, VoiceHabitInsight):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = f"Prepared {len(event.result.habits)} voice habit(s)"
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=event.result.warnings,
            details={"habits": [habit.name for habit in event.result.habits]},
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Preparing voice habits"
