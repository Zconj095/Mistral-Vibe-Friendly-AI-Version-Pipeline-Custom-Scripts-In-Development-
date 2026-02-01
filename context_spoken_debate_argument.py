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
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]*", re.S)

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


class DebateRole(BaseModel):
    name: str
    description: str
    cue: str


class ContextSpokenDebateArgumentConfig(BaseToolConfig):
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
    min_token_length: int = Field(default=2, description="Minimum token length.")
    max_points: int = Field(default=8, description="Maximum argument points.")
    default_debate_roles: list[DebateRole] = Field(
        default_factory=lambda: [
            DebateRole(
                name="Moderator",
                description="Keeps the debate focused and fair.",
                cue="Introduce the topic and keep time balanced.",
            ),
            DebateRole(
                name="Affirmative",
                description="Argues in favor of the topic or claim.",
                cue="State the thesis, give evidence, and emphasize benefits.",
            ),
            DebateRole(
                name="Negative",
                description="Argues against the topic or claim.",
                cue="State the counterpoint, risks, and limitations.",
            ),
            DebateRole(
                name="Rebuttal",
                description="Responds to the opposing side with counters.",
                cue="Directly address the opposing claims with evidence.",
            ),
            DebateRole(
                name="Closing",
                description="Summarizes the strongest points.",
                cue="Summarize key evidence and restate the conclusion.",
            ),
        ],
        description="Default debate roles.",
    )


class ContextSpokenDebateArgumentState(BaseToolState):
    pass


class ContextSpokenDebateArgumentArgs(BaseModel):
    content: str | None = Field(default=None, description="Content to analyze.")
    path: str | None = Field(default=None, description="Path to content.")
    topic: str | None = Field(default=None, description="Debate topic.")
    claim: str | None = Field(default=None, description="Argument claim.")
    debate_roles: list[str] | None = Field(
        default=None, description="Debate roles to use."
    )
    role_sequence: list[str] | None = Field(
        default=None, description="Explicit role sequence to cycle per segment."
    )
    segment_by: str | None = Field(
        default=None, description="sentences, lines, or paragraphs."
    )
    max_segments: int | None = Field(
        default=None, description="Override max segments."
    )
    max_points: int | None = Field(default=None, description="Override max points.")
    include_debate: bool = Field(default=True, description="Include debate cues.")
    include_argument: bool = Field(default=True, description="Include argument cues.")
    include_comparison: bool = Field(
        default=True, description="Include debate vs argument comparison."
    )


class DebateTurn(BaseModel):
    index: int
    role: str
    text: str
    cue: str
    word_count: int
    preview: str


class ArgumentPoint(BaseModel):
    index: int
    focus: str
    keywords: list[str]
    cue: str


class DebateArgumentComparison(BaseModel):
    debate_traits: list[str]
    argument_traits: list[str]
    differences: list[str]


class ContextSpokenDebateArgumentResult(BaseModel):
    debate_roles: list[DebateRole]
    role_sequence: list[str]
    debate_turns: list[DebateTurn]
    debate_turn_count: int
    debate_prompt: str
    debate_opening: str
    debate_closing: str
    argument_points: list[ArgumentPoint]
    argument_prompt: str
    argument_opening: str
    argument_closing: str
    comparison: DebateArgumentComparison | None
    warnings: list[str]


class ContextSpokenDebateArgument(
    BaseTool[
        ContextSpokenDebateArgumentArgs,
        ContextSpokenDebateArgumentResult,
        ContextSpokenDebateArgumentConfig,
        ContextSpokenDebateArgumentState,
    ],
    ToolUIData[ContextSpokenDebateArgumentArgs, ContextSpokenDebateArgumentResult],
):
    description: ClassVar[str] = (
        "Build spoken debate and argument cues plus differences."
    )

    async def run(
        self, args: ContextSpokenDebateArgumentArgs
    ) -> ContextSpokenDebateArgumentResult:
        warnings: list[str] = []
        content = None
        if args.content or args.path:
            content = self._load_content(args)

        roles = self._resolve_roles(args, warnings)
        role_sequence = self._resolve_role_sequence(args, roles, warnings)

        debate_turns, debate_turn_count = self._build_debate_turns(
            content, args, roles, role_sequence, warnings
        )

        argument_points = self._build_argument_points(content, args, warnings)

        debate_prompt = self._build_debate_prompt(roles)
        debate_opening = self._debate_opening(args.topic, roles)
        debate_closing = "Close with balanced summaries and key takeaways."

        argument_prompt = self._build_argument_prompt(argument_points)
        argument_opening = self._argument_opening(args.claim)
        argument_closing = "Close with a concise restatement of the claim."

        comparison = self._build_comparison() if args.include_comparison else None

        return ContextSpokenDebateArgumentResult(
            debate_roles=roles,
            role_sequence=role_sequence,
            debate_turns=debate_turns,
            debate_turn_count=debate_turn_count,
            debate_prompt=debate_prompt,
            debate_opening=debate_opening,
            debate_closing=debate_closing,
            argument_points=argument_points,
            argument_prompt=argument_prompt,
            argument_opening=argument_opening,
            argument_closing=argument_closing,
            comparison=comparison,
            warnings=warnings,
        )

    def _load_content(self, args: ContextSpokenDebateArgumentArgs) -> str:
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

    def _resolve_roles(
        self,
        args: ContextSpokenDebateArgumentArgs,
        warnings: list[str],
    ) -> list[DebateRole]:
        roles = list(self.config.default_debate_roles)
        if not args.debate_roles:
            return roles

        lookup = {role.name.lower(): role for role in roles}
        selected: list[DebateRole] = []
        for name in args.debate_roles:
            key = name.strip().lower()
            if key in lookup:
                selected.append(lookup[key])
            else:
                warnings.append(f"Unknown debate role: {name}")
                selected.append(
                    DebateRole(
                        name=name.strip(),
                        description="Custom debate role.",
                        cue=f"{name.strip()} role: keep statements concise.",
                    )
                )
        return selected if selected else roles

    def _resolve_role_sequence(
        self,
        args: ContextSpokenDebateArgumentArgs,
        roles: list[DebateRole],
        warnings: list[str],
    ) -> list[str]:
        if args.role_sequence:
            lookup = {role.name.lower(): role for role in roles}
            sequence: list[str] = []
            for name in args.role_sequence:
                key = name.strip().lower()
                if key in lookup:
                    sequence.append(lookup[key].name)
                else:
                    warnings.append(f"Unknown role in sequence: {name}")
            return sequence if sequence else [role.name for role in roles]
        return [role.name for role in roles]

    def _build_debate_turns(
        self,
        content: str | None,
        args: ContextSpokenDebateArgumentArgs,
        roles: list[DebateRole],
        role_sequence: list[str],
        warnings: list[str],
    ) -> tuple[list[DebateTurn], int]:
        if not args.include_debate or not content:
            if args.include_debate and not content:
                warnings.append("No content provided; debate turns skipped.")
            return [], 0

        segment_by = (args.segment_by or self.config.default_segment_by).strip().lower()
        if segment_by not in {"sentences", "lines", "paragraphs"}:
            raise ToolError("segment_by must be sentences, lines, or paragraphs.")
        max_segments = (
            args.max_segments if args.max_segments is not None else self.config.max_segments
        )
        if max_segments <= 0:
            raise ToolError("max_segments must be positive.")

        lookup = {role.name.lower(): role for role in roles}
        segments_raw = self._split_segments(content, segment_by)
        turns: list[DebateTurn] = []
        for idx, text in enumerate(segments_raw, start=1):
            if len(turns) >= max_segments:
                warnings.append("Segment limit reached; debate turns truncated.")
                break
            if len(text) < self.config.min_segment_chars:
                continue
            role_name = role_sequence[(idx - 1) % len(role_sequence)]
            role = lookup.get(role_name.lower())
            cue = role.cue if role else f"{role_name} role: respond clearly."
            turns.append(
                DebateTurn(
                    index=len(turns) + 1,
                    role=role_name,
                    text=text.strip(),
                    cue=cue,
                    word_count=len(WORD_RE.findall(text)),
                    preview=self._preview(text),
                )
            )
        return turns, len(turns)

    def _build_argument_points(
        self,
        content: str | None,
        args: ContextSpokenDebateArgumentArgs,
        warnings: list[str],
    ) -> list[ArgumentPoint]:
        if not args.include_argument:
            return []
        max_points = args.max_points if args.max_points is not None else self.config.max_points
        if max_points <= 0:
            raise ToolError("max_points must be positive.")
        if not content:
            warnings.append("No content provided; using default argument points.")
            return [
                ArgumentPoint(
                    index=1,
                    focus=args.claim or "the main claim",
                    keywords=[],
                    cue="State the claim and provide one clear reason.",
                )
            ]

        keywords = self._extract_keywords(content, max_points)
        points: list[ArgumentPoint] = []
        for idx, keyword in enumerate(keywords, start=1):
            points.append(
                ArgumentPoint(
                    index=idx,
                    focus=keyword,
                    keywords=[keyword],
                    cue=f"Support the claim by focusing on {keyword}.",
                )
            )
        return points

    def _extract_keywords(self, text: str, max_items: int) -> list[str]:
        tokens = [
            token.lower()
            for token in WORD_RE.findall(text)
            if len(token) >= self.config.min_token_length
            and token.lower() not in STOPWORDS
        ]
        return [word for word, _ in Counter(tokens).most_common(max_items)]

    def _build_debate_prompt(self, roles: list[DebateRole]) -> str:
        lines = ["Debate roles:"]
        for role in roles:
            lines.append(f"- {role.name}: {role.description}")
        lines.append("Keep turns balanced and evidence-driven.")
        return "\n".join(lines)

    def _debate_opening(self, topic: str | None, roles: list[DebateRole]) -> str:
        topic_text = topic or "the topic"
        role_names = ", ".join(role.name for role in roles)
        return f"Begin a debate on {topic_text} with roles: {role_names}."

    def _build_argument_prompt(self, points: list[ArgumentPoint]) -> str:
        if not points:
            return "Argument outline: state a claim, support it, and conclude."
        lines = ["Argument outline:"]
        for point in points:
            lines.append(f"- Focus on {point.focus}")
        lines.append("Keep the argument linear and persuasive.")
        return "\n".join(lines)

    def _argument_opening(self, claim: str | None) -> str:
        claim_text = claim or "the main claim"
        return f"State {claim_text} and establish the core reasoning."

    def _build_comparison(self) -> DebateArgumentComparison:
        debate_traits = [
            "multi-party with turn-taking",
            "balanced views and rebuttals",
            "moderation and time control",
            "evidence plus counter-evidence",
        ]
        argument_traits = [
            "single speaker perspective",
            "linear persuasion",
            "focused on one claim",
            "less formal turn structure",
        ]
        differences = [
            "Debates present opposing sides; arguments present one side.",
            "Debates include rebuttals; arguments emphasize persuasion.",
            "Debates require role balance; arguments follow one voice.",
        ]
        return DebateArgumentComparison(
            debate_traits=debate_traits,
            argument_traits=argument_traits,
            differences=differences,
        )

    def _split_segments(self, text: str, mode: str) -> list[str]:
        if mode == "lines":
            return [line for line in text.splitlines() if line.strip()]
        if mode == "paragraphs":
            return [chunk for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
        return [seg.strip() for seg in SENTENCE_RE.findall(text) if seg.strip()]

    def _preview(self, text: str) -> str:
        max_chars = self.config.preview_chars
        if max_chars <= 0:
            return ""
        stripped = text.strip()
        return stripped if len(stripped) <= max_chars else stripped[:max_chars]

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextSpokenDebateArgumentArgs):
            return ToolCallDisplay(summary="context_spoken_debate_argument")
        return ToolCallDisplay(
            summary="context_spoken_debate_argument",
            details={
                "path": event.args.path,
                "topic": event.args.topic,
                "claim": event.args.claim,
                "debate_roles": event.args.debate_roles,
                "role_sequence": event.args.role_sequence,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpokenDebateArgumentResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = (
            f"Prepared {event.result.debate_turn_count} debate turn(s) "
            f"and {len(event.result.argument_points)} argument point(s)"
        )
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=event.result.warnings,
            details={
                "debate_turn_count": event.result.debate_turn_count,
                "argument_points": len(event.result.argument_points),
                "has_comparison": event.result.comparison is not None,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Preparing debate and argument cues"
