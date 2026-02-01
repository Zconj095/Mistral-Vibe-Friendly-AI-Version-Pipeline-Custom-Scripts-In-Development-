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


SPEAKER_LINE_RE = re.compile(
    r"^\s*(?P<speaker>[A-Za-z0-9 ._\\-]{2,40})\s*:\s*(?P<text>.+)$"
)
WORD_RE = re.compile(r"[A-Za-z0-9_']+")
QUESTION_STARTS = {"why", "what", "how", "when", "where", "who", "which"}

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


class ContextSpeechDialogueConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_source_bytes: int = Field(
        default=5_000_000, description="Maximum bytes per transcript."
    )
    max_turns: int = Field(
        default=200, description="Maximum number of turns to return."
    )
    preview_chars: int = Field(default=200, description="Preview snippet length.")
    default_turn_count: int = Field(
        default=8, description="Default turns when planning dialogue."
    )
    min_token_length: int = Field(default=2, description="Minimum token length.")
    max_keywords: int = Field(default=8, description="Max keywords per turn.")


class ContextSpeechDialogueState(BaseToolState):
    pass


class ContextSpeechDialogueArgs(BaseModel):
    content: str | None = Field(default=None, description="Transcript content.")
    path: str | None = Field(default=None, description="Path to transcript.")
    speakers: list[str] | None = Field(
        default=None, description="Speaker names in order."
    )
    topic: str | None = Field(default=None, description="Dialogue topic.")
    dialogue_goal: str | None = Field(
        default=None, description="Goal for the dialogue."
    )
    turn_count: int | None = Field(
        default=None, description="Turns to plan when no transcript is provided."
    )


class DialogueTurn(BaseModel):
    index: int
    speaker: str
    text: str
    word_count: int
    keywords: list[str]
    intent: str
    cue: str


class SpeakerProfile(BaseModel):
    speaker: str
    turn_count: int
    word_count: int
    share: float
    top_keywords: list[str]


class ContextSpeechDialogueResult(BaseModel):
    turns: list[DialogueTurn]
    turn_count: int
    speaker_profiles: list[SpeakerProfile]
    speech_opening: str
    speech_closing: str
    warnings: list[str]
    parsed_transcript: bool


class ContextSpeechDialogue(
    BaseTool[
        ContextSpeechDialogueArgs,
        ContextSpeechDialogueResult,
        ContextSpeechDialogueConfig,
        ContextSpeechDialogueState,
    ],
    ToolUIData[ContextSpeechDialogueArgs, ContextSpeechDialogueResult],
):
    description: ClassVar[str] = (
        "Build or analyze a speech dialogue with turn-by-turn cues."
    )

    async def run(self, args: ContextSpeechDialogueArgs) -> ContextSpeechDialogueResult:
        warnings: list[str] = []
        parsed_transcript = False

        if args.content or args.path:
            content = self._load_content(args)
            turns = self._parse_turns(content)
            if turns:
                parsed_transcript = True
            else:
                warnings.append("No speaker markers found; planning dialogue instead.")
                turns = self._plan_turns(args)
        else:
            turns = self._plan_turns(args)

        if not turns:
            raise ToolError("No dialogue turns generated.")

        if len(turns) > self.config.max_turns:
            warnings.append("Turn limit reached; output truncated.")
            turns = turns[: self.config.max_turns]

        speaker_profiles = self._speaker_profiles(turns)
        speech_opening = self._speech_opening(args, speaker_profiles)
        speech_closing = self._speech_closing(args)

        return ContextSpeechDialogueResult(
            turns=turns,
            turn_count=len(turns),
            speaker_profiles=speaker_profiles,
            speech_opening=speech_opening,
            speech_closing=speech_closing,
            warnings=warnings,
            parsed_transcript=parsed_transcript,
        )

    def _load_content(self, args: ContextSpeechDialogueArgs) -> str:
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

    def _parse_turns(self, content: str) -> list[DialogueTurn]:
        turns: list[DialogueTurn] = []
        for line in content.splitlines():
            match = SPEAKER_LINE_RE.match(line)
            if not match:
                continue
            speaker = match.group("speaker").strip()
            text = match.group("text").strip()
            if not text:
                continue
            turns.append(self._build_turn(len(turns) + 1, speaker, text))
        return turns

    def _plan_turns(self, args: ContextSpeechDialogueArgs) -> list[DialogueTurn]:
        speakers = args.speakers or ["Speaker A", "Speaker B"]
        turn_count = (
            args.turn_count
            if args.turn_count is not None
            else self.config.default_turn_count
        )
        if turn_count <= 0:
            raise ToolError("turn_count must be positive.")

        topic = args.topic or "the current topic"
        goal = args.dialogue_goal or "reach shared understanding"
        intents = [
            "open",
            "clarify",
            "expand",
            "question",
            "summarize",
            "confirm",
        ]
        turns: list[DialogueTurn] = []
        for idx in range(turn_count):
            speaker = speakers[idx % len(speakers)]
            intent = intents[idx % len(intents)]
            text = f"{speaker} should {intent} about {topic} to {goal}."
            turns.append(self._build_turn(idx + 1, speaker, text))
        return turns

    def _build_turn(self, index: int, speaker: str, text: str) -> DialogueTurn:
        keywords = self._extract_keywords(text, self.config.max_keywords)
        intent = self._intent_from_text(text)
        cue = self._build_cue(speaker, intent, keywords)
        return DialogueTurn(
            index=index,
            speaker=speaker,
            text=text,
            word_count=len(WORD_RE.findall(text)),
            keywords=keywords,
            intent=intent,
            cue=cue,
        )

    def _extract_keywords(self, text: str, max_items: int) -> list[str]:
        tokens = []
        for token in WORD_RE.findall(text):
            lower = token.lower()
            if len(lower) < self.config.min_token_length:
                continue
            if lower in STOPWORDS:
                continue
            tokens.append(lower)
        return [word for word, _ in Counter(tokens).most_common(max_items)]

    def _intent_from_text(self, text: str) -> str:
        stripped = text.strip()
        lower = stripped.lower()
        if stripped.endswith("?"):
            return "question"
        if any(lower.startswith(q) for q in QUESTION_STARTS):
            return "question"
        if "please" in lower or "could you" in lower:
            return "request"
        if "summar" in lower or "recap" in lower:
            return "summary"
        return "statement"

    def _build_cue(self, speaker: str, intent: str, keywords: list[str]) -> str:
        if keywords:
            return f"{speaker} {intent}: focus on {', '.join(keywords[:5])}."
        return f"{speaker} {intent}: focus on the next point."

    def _speaker_profiles(self, turns: list[DialogueTurn]) -> list[SpeakerProfile]:
        totals: Counter[str] = Counter()
        words: Counter[str] = Counter()
        keywords: dict[str, Counter[str]] = {}
        for turn in turns:
            totals[turn.speaker] += 1
            words[turn.speaker] += turn.word_count
            if turn.speaker not in keywords:
                keywords[turn.speaker] = Counter()
            keywords[turn.speaker].update(turn.keywords)

        total_words = sum(words.values()) or 1
        profiles: list[SpeakerProfile] = []
        for speaker, count in totals.most_common():
            profile = SpeakerProfile(
                speaker=speaker,
                turn_count=count,
                word_count=words[speaker],
                share=words[speaker] / total_words,
                top_keywords=[
                    word for word, _ in keywords[speaker].most_common(self.config.max_keywords)
                ],
            )
            profiles.append(profile)
        return profiles

    def _speech_opening(
        self, args: ContextSpeechDialogueArgs, profiles: list[SpeakerProfile]
    ) -> str:
        speaker_names = ", ".join(profile.speaker for profile in profiles)
        topic = args.topic or "the topic at hand"
        return f"Begin a dialogue between {speaker_names} about {topic}."

    def _speech_closing(self, args: ContextSpeechDialogueArgs) -> str:
        goal = args.dialogue_goal or "summarize and confirm next steps"
        return f"Close the dialogue by {goal}."

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextSpeechDialogueArgs):
            return ToolCallDisplay(summary="context_speech_dialogue")
        return ToolCallDisplay(
            summary="context_speech_dialogue",
            details={
                "path": event.args.path,
                "topic": event.args.topic,
                "turn_count": event.args.turn_count,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpeechDialogueResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = f"Prepared {event.result.turn_count} dialogue turn(s)"
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=event.result.warnings,
            details={
                "turn_count": event.result.turn_count,
                "parsed_transcript": event.result.parsed_transcript,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Preparing speech dialogue"
