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

DEFAULT_TOPICS: list[dict[str, object]] = [
    {
        "term": "Overview",
        "description": "Summarize the big picture and key highlights.",
        "markers": ["overview", "summary", "highlight", "outline", "scope"],
    },
    {
        "term": "Instruction",
        "description": "Provide step-by-step guidance and procedures.",
        "markers": ["step", "steps", "guide", "instruction", "procedure", "how"],
    },
    {
        "term": "Troubleshooting",
        "description": "Diagnose issues and propose fixes.",
        "markers": ["error", "issue", "problem", "fix", "resolve", "debug", "failed"],
    },
    {
        "term": "Status Update",
        "description": "Report progress, blockers, and current state.",
        "markers": ["status", "update", "progress", "current", "blocked", "milestone"],
    },
    {
        "term": "Planning",
        "description": "Describe next steps, timelines, and goals.",
        "markers": ["plan", "roadmap", "timeline", "schedule", "goal", "next"],
    },
    {
        "term": "Analysis",
        "description": "Explain reasoning, causes, and implications.",
        "markers": ["analysis", "analyze", "cause", "impact", "evidence", "evaluate"],
    },
    {
        "term": "Brainstorming",
        "description": "Generate options, ideas, and possibilities.",
        "markers": ["idea", "ideas", "brainstorm", "option", "explore", "possibility"],
    },
    {
        "term": "Q&A",
        "description": "Answer questions directly and concisely.",
        "markers": ["question", "answer", "ask", "faq"],
    },
    {
        "term": "Narrative",
        "description": "Tell a story or describe events in sequence.",
        "markers": ["story", "narrative", "scene", "character", "plot", "journey"],
    },
    {
        "term": "Warning",
        "description": "Call out risks, cautions, or urgent alerts.",
        "markers": ["warning", "alert", "risk", "caution", "urgent", "critical"],
    },
    {
        "term": "Reflection",
        "description": "Discuss lessons learned and insights.",
        "markers": ["reflect", "reflection", "lesson", "insight", "learned"],
    },
]


class ContextSpokenTopicConfig(BaseToolConfig):
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
    max_keywords: int = Field(default=6, description="Max keywords per segment.")
    max_topics: int = Field(default=12, description="Maximum topics to score.")
    max_understanding_keywords: int = Field(
        default=10, description="Maximum keywords in topic understanding."
    )
    max_understanding_sentences: int = Field(
        default=3, description="Maximum evidence sentences in understanding."
    )
    max_topic_crossrefs: int = Field(
        default=6, description="Maximum topic cross-reference entries."
    )
    default_assign_mode: str = Field(
        default="single", description="single, per_segment, or sequence."
    )


class ContextSpokenTopicState(BaseToolState):
    pass


class ContextSpokenTopicArgs(BaseModel):
    content: str | None = Field(default=None, description="Content to analyze.")
    path: str | None = Field(default=None, description="Path to content.")
    topic: str | None = Field(default=None, description="Explicit speaking topic.")
    topics: list[str] | None = Field(
        default=None, description="Topic options to choose from."
    )
    topic_sequence: list[str] | None = Field(
        default=None, description="Explicit topic sequence to cycle."
    )
    assign_mode: str | None = Field(
        default=None, description="single, per_segment, or sequence."
    )
    segment_by: str | None = Field(
        default=None, description="sentences, lines, or paragraphs."
    )
    max_segments: int | None = Field(
        default=None, description="Override max segments."
    )
    include_understanding: bool = Field(
        default=True, description="Include topic understanding summary."
    )
    max_topic_crossrefs: int | None = Field(
        default=None, description="Override max topic crossrefs."
    )
    include_opening: bool = Field(
        default=True, description="Include opening cue."
    )
    include_closing: bool = Field(
        default=True, description="Include closing cue."
    )


class TopicDefinition(BaseModel):
    term: str
    normalized_term: str
    description: str
    markers: list[str]


class TopicScore(BaseModel):
    term: str
    score: int
    matched_markers: list[str]
    description: str


class TopicInterpretation(BaseModel):
    term: str
    description: str
    matched_markers: list[str]
    score: int
    rationale: list[str]


class TopicUnderstanding(BaseModel):
    topic: str
    summary: str
    keywords: list[str]
    evidence: list[str]


class TopicCrossref(BaseModel):
    topic_a: str
    topic_b: str
    shared_markers: list[str]
    shared_keywords: list[str]
    overlap_score: float
    rationale: list[str]


class SpokenTopicSegment(BaseModel):
    index: int
    text: str
    topic: str
    word_count: int
    keywords: list[str]
    cue: str


class ContextSpokenTopicResult(BaseModel):
    topic: str
    topic_description: str
    topic_prompt: str
    assign_mode: str
    topic_sequence: list[str]
    topic_interpretation: TopicInterpretation
    topic_interpretations: list[TopicInterpretation]
    topic_understanding: TopicUnderstanding | None
    topic_crossrefs: list[TopicCrossref]
    crossref_summary: str
    segments: list[SpokenTopicSegment]
    segment_count: int
    topic_scores: list[TopicScore]
    speech_opening: str
    speech_closing: str
    warnings: list[str]


class ContextSpokenTopic(
    BaseTool[
        ContextSpokenTopicArgs,
        ContextSpokenTopicResult,
        ContextSpokenTopicConfig,
        ContextSpokenTopicState,
    ],
    ToolUIData[ContextSpokenTopicArgs, ContextSpokenTopicResult],
):
    description: ClassVar[str] = "Select a speaking topic and build topic cues."

    async def run(self, args: ContextSpokenTopicArgs) -> ContextSpokenTopicResult:
        definitions = self._load_definitions(args)
        if not definitions:
            raise ToolError("No speaking topics available.")

        assign_mode = (
            args.assign_mode or self.config.default_assign_mode
        ).strip().lower()
        if args.topic_sequence:
            assign_mode = "sequence"
        if assign_mode not in {"single", "per_segment", "sequence"}:
            raise ToolError("assign_mode must be single, per_segment, or sequence.")

        content = None
        if args.content or args.path:
            content = self._load_content(args)

        selected, scores, warnings = self._select_topic(
            args, definitions, content
        )
        topic_sequence = self._resolve_topic_sequence(args, definitions, selected, warnings)
        segments = self._build_segments(
            args,
            selected,
            definitions,
            topic_sequence,
            assign_mode,
            content,
            warnings,
        )

        if assign_mode == "single":
            topic_prompt = self._build_topic_prompt(selected)
        else:
            topic_prompt = self._build_topic_prompt_multi(topic_sequence)

        interpretations = self._build_topic_interpretations(
            selected,
            topic_sequence,
            scores,
            content,
            assign_mode,
        )
        primary_interpretation = next(
            (item for item in interpretations if item.term == selected.term),
            TopicInterpretation(
                term=selected.term,
                description=selected.description,
                matched_markers=[],
                score=0,
                rationale=["Defaulted to selected topic."],
            ),
        )

        topic_understanding = None
        if args.include_understanding and content:
            topic_understanding = self._build_topic_understanding(
                content,
                selected,
                scores,
            )
        elif args.include_understanding and not content:
            warnings.append("No content provided; topic understanding skipped.")

        topic_crossrefs, crossref_summary = self._build_topic_crossrefs(
            scores,
            args.max_topic_crossrefs,
            content,
        )

        speech_opening = self._speech_opening(args, selected, topic_sequence, assign_mode)
        speech_closing = self._speech_closing(args)

        return ContextSpokenTopicResult(
            topic=selected.term,
            topic_description=selected.description,
            topic_prompt=topic_prompt,
            assign_mode=assign_mode,
            topic_sequence=[definition.term for definition in topic_sequence],
            topic_interpretation=primary_interpretation,
            topic_interpretations=interpretations,
            topic_understanding=topic_understanding,
            topic_crossrefs=topic_crossrefs,
            crossref_summary=crossref_summary,
            segments=segments,
            segment_count=len(segments),
            topic_scores=scores,
            speech_opening=speech_opening,
            speech_closing=speech_closing,
            warnings=warnings,
        )

    def _load_definitions(self, args: ContextSpokenTopicArgs) -> list[TopicDefinition]:
        definitions: list[TopicDefinition] = []
        for entry in DEFAULT_TOPICS:
            term = self._normalize_term(str(entry.get("term", "")))
            description = self._normalize_description(str(entry.get("description", "")))
            markers = [
                self._normalize_term(str(marker)).lower()
                for marker in entry.get("markers", [])
                if marker
            ]
            if not term:
                continue
            definitions.append(
                TopicDefinition(
                    term=term,
                    normalized_term=term.lower(),
                    description=description or term,
                    markers=markers,
                )
            )

        if args.topics:
            definitions = self._filter_topics(args.topics, definitions)

        max_topics = self.config.max_topics
        if max_topics > 0 and len(definitions) > max_topics:
            definitions = definitions[:max_topics]
        return definitions

    def _filter_topics(
        self, names: list[str], definitions: list[TopicDefinition]
    ) -> list[TopicDefinition]:
        lookup = {definition.normalized_term: definition for definition in definitions}
        filtered: list[TopicDefinition] = []
        for name in names:
            normalized = self._normalize_term(name).lower()
            if normalized in lookup:
                filtered.append(lookup[normalized])
                continue
            filtered.append(self._custom_topic(name))
        return filtered

    def _custom_topic(self, name: str) -> TopicDefinition:
        normalized = self._normalize_term(name)
        markers = [token.lower() for token in WORD_RE.findall(normalized)]
        return TopicDefinition(
            term=normalized,
            normalized_term=normalized.lower(),
            description="Custom speaking topic.",
            markers=markers,
        )

    def _select_topic(
        self,
        args: ContextSpokenTopicArgs,
        definitions: list[TopicDefinition],
        content: str | None,
    ) -> tuple[TopicDefinition, list[TopicScore], list[str]]:
        warnings: list[str] = []
        if args.topic:
            normalized = self._normalize_term(args.topic).lower()
            for definition in definitions:
                if definition.normalized_term == normalized:
                    return definition, [], warnings
            return (
                self._custom_topic(args.topic),
                [],
                warnings,
            )

        scores: list[TopicScore] = []
        if content:
            scores = self._score_topics(content, definitions)
            if scores:
                best = max(scores, key=lambda item: item.score)
                if best.score > 0:
                    selected = next(
                        definition
                        for definition in definitions
                        if definition.term == best.term
                    )
                    return selected, scores, warnings
                warnings.append("No topic markers matched; using default topic.")

        selected = definitions[0]
        return selected, scores, warnings

    def _resolve_topic_sequence(
        self,
        args: ContextSpokenTopicArgs,
        definitions: list[TopicDefinition],
        fallback: TopicDefinition,
        warnings: list[str],
    ) -> list[TopicDefinition]:
        if args.topic_sequence:
            return self._sequence_from_names(args.topic_sequence, definitions, warnings)
        if args.topics:
            return self._sequence_from_names(args.topics, definitions, warnings)
        if definitions:
            return definitions
        warnings.append("No topic sequence available; using fallback.")
        return [fallback]

    def _sequence_from_names(
        self,
        names: list[str],
        definitions: list[TopicDefinition],
        warnings: list[str],
    ) -> list[TopicDefinition]:
        lookup = {definition.normalized_term: definition for definition in definitions}
        sequence: list[TopicDefinition] = []
        for name in names:
            normalized = self._normalize_term(name).lower()
            definition = lookup.get(normalized)
            if definition is None:
                warnings.append(f"Unknown topic in sequence: {name}")
                definition = self._custom_topic(name)
            sequence.append(definition)
        return sequence

    def _score_topics(
        self, content: str, definitions: list[TopicDefinition]
    ) -> list[TopicScore]:
        tokens = [
            token.lower()
            for token in WORD_RE.findall(content)
            if len(token) >= self.config.min_token_length
        ]
        counts = Counter(tokens)
        scores: list[TopicScore] = []
        for definition in definitions:
            matched: list[str] = []
            score = 0
            for marker in definition.markers:
                hits = counts.get(marker, 0)
                if hits:
                    matched.append(marker)
                    score += hits
            scores.append(
                TopicScore(
                    term=definition.term,
                    score=score,
                    matched_markers=matched,
                    description=definition.description,
                )
            )
        scores.sort(key=lambda item: (-item.score, item.term))
        return scores

    def _build_segments(
        self,
        args: ContextSpokenTopicArgs,
        topic: TopicDefinition,
        definitions: list[TopicDefinition],
        topic_sequence: list[TopicDefinition],
        assign_mode: str,
        content: str | None,
        warnings: list[str],
    ) -> list[SpokenTopicSegment]:
        if not content:
            if assign_mode != "single":
                warnings.append("No content provided; no segments generated.")
            return []
        segment_by = (args.segment_by or self.config.default_segment_by).strip().lower()
        if segment_by not in {"sentences", "lines", "paragraphs"}:
            raise ToolError("segment_by must be sentences, lines, or paragraphs.")
        max_segments = (
            args.max_segments if args.max_segments is not None else self.config.max_segments
        )
        if max_segments <= 0:
            raise ToolError("max_segments must be positive.")

        segments_raw = self._split_segments(content, segment_by)
        segments: list[SpokenTopicSegment] = []
        if assign_mode == "sequence" and not topic_sequence:
            warnings.append("Topic sequence missing; falling back to single topic.")
            assign_mode = "single"

        for idx, text in enumerate(segments_raw, start=1):
            if len(segments) >= max_segments:
                break
            if len(text) < self.config.min_segment_chars:
                continue
            if assign_mode == "sequence":
                assigned = topic_sequence[(idx - 1) % len(topic_sequence)]
            elif assign_mode == "per_segment":
                assigned = self._select_segment_topic(text, definitions, topic)
            else:
                assigned = topic
            keywords = self._extract_keywords(text, self.config.max_keywords)
            cue = self._build_segment_cue(assigned, keywords)
            segments.append(
                SpokenTopicSegment(
                    index=len(segments) + 1,
                    text=text.strip(),
                    topic=assigned.term,
                    word_count=len(WORD_RE.findall(text)),
                    keywords=keywords,
                    cue=cue,
                )
            )
        return segments

    def _select_segment_topic(
        self,
        text: str,
        definitions: list[TopicDefinition],
        fallback: TopicDefinition,
    ) -> TopicDefinition:
        scores = self._score_topics(text, definitions)
        if not scores:
            return fallback
        best = max(scores, key=lambda item: item.score)
        if best.score <= 0:
            return fallback
        for definition in definitions:
            if definition.term == best.term:
                return definition
        return fallback

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

    def _split_segments(self, text: str, mode: str) -> list[str]:
        if mode == "lines":
            return [line for line in text.splitlines() if line.strip()]
        if mode == "paragraphs":
            return [chunk for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
        return [seg.strip() for seg in SENTENCE_RE.findall(text) if seg.strip()]

    def _build_topic_prompt(self, topic: TopicDefinition) -> str:
        lines = [
            f"Speaking topic: {topic.term}.",
            f"Topic guidance: {topic.description}",
            "Keep phrasing aligned to the topic throughout.",
        ]
        return " ".join(lines)

    def _build_topic_prompt_multi(self, topics: list[TopicDefinition]) -> str:
        if not topics:
            return "Speaking topics: Overview. Keep phrasing consistent per topic."
        lines = ["Speaking topics:"]
        for topic in topics:
            lines.append(f"- {topic.term}: {topic.description}")
        lines.append("Keep phrasing consistent within each topic.")
        return " ".join(lines)

    def _build_topic_interpretations(
        self,
        selected: TopicDefinition,
        topic_sequence: list[TopicDefinition],
        scores: list[TopicScore],
        content: str | None,
        assign_mode: str,
    ) -> list[TopicInterpretation]:
        score_map = {score.term: score for score in scores}
        interpretations: list[TopicInterpretation] = []
        targets = topic_sequence or [selected]
        for definition in targets:
            score_info = score_map.get(definition.term)
            matched = score_info.matched_markers if score_info else []
            score_value = score_info.score if score_info else 0
            rationale: list[str] = []
            if content is None:
                rationale.append("No content provided; interpreted by definition.")
            elif score_value > 0:
                rationale.append(
                    "Matched markers: " + ", ".join(matched) if matched else "Matched markers present."
                )
            else:
                rationale.append("No markers matched; defaulted to topic.")

            if assign_mode == "sequence":
                rationale.append("Sequence mode applies topics by order.")
            elif assign_mode == "per_segment":
                rationale.append("Segment mode selects topics per segment.")

            interpretations.append(
                TopicInterpretation(
                    term=definition.term,
                    description=definition.description,
                    matched_markers=matched,
                    score=score_value,
                    rationale=rationale,
                )
            )
        return interpretations

    def _build_segment_cue(
        self, topic: TopicDefinition, keywords: list[str]
    ) -> str:
        if keywords:
            return (
                f"{topic.term} topic; emphasize {', '.join(keywords[:5])}."
            )
        return f"{topic.term} topic; keep phrasing consistent."

    def _build_topic_understanding(
        self,
        content: str,
        topic: TopicDefinition,
        scores: list[TopicScore],
    ) -> TopicUnderstanding:
        keywords = self._extract_keywords(content, self.config.max_understanding_keywords)
        matched_markers = []
        for score in scores:
            if score.term == topic.term:
                matched_markers = score.matched_markers
                break
        if not matched_markers:
            matched_markers = topic.markers

        evidence = self._select_evidence_sentences(
            content,
            matched_markers,
            keywords,
            self.config.max_understanding_sentences,
        )

        summary_parts = [f"Topic: {topic.term}."]
        if matched_markers:
            summary_parts.append(
                f"Matched markers: {', '.join(matched_markers[:5])}."
            )
        if keywords:
            summary_parts.append(
                f"Key terms: {', '.join(keywords[:5])}."
            )
        summary = " ".join(summary_parts)

        return TopicUnderstanding(
            topic=topic.term,
            summary=summary,
            keywords=keywords,
            evidence=evidence,
        )

    def _build_topic_crossrefs(
        self,
        scores: list[TopicScore],
        max_topic_crossrefs: int | None,
        content: str | None,
    ) -> tuple[list[TopicCrossref], str]:
        limit = (
            max_topic_crossrefs
            if max_topic_crossrefs is not None
            else self.config.max_topic_crossrefs
        )
        if limit is None or limit <= 0:
            limit = 0

        if not scores or len(scores) < 2:
            return [], "Topic crossrefs: none available."

        sorted_scores = sorted(scores, key=lambda item: (-item.score, item.term))
        marker_map = {score.term: score.matched_markers for score in sorted_scores}
        crossrefs: list[TopicCrossref] = []

        for i in range(len(sorted_scores)):
            for j in range(i + 1, len(sorted_scores)):
                if limit and len(crossrefs) >= limit:
                    break
                left = sorted_scores[i]
                right = sorted_scores[j]
                left_markers = marker_map.get(left.term, [])
                right_markers = marker_map.get(right.term, [])
                shared_markers = sorted(set(left_markers) & set(right_markers))
                shared_keywords = sorted(set(left_markers) | set(right_markers))
                if content is None and not shared_markers:
                    continue
                overlap_score = 0.0
                if left.score + right.score > 0:
                    overlap_score = (min(left.score, right.score) / (left.score + right.score))

                rationale: list[str] = []
                if shared_markers:
                    rationale.append("Shared markers indicate a blended topic.")
                if content is None:
                    rationale.append("No content; crossref based on marker overlap.")
                if not shared_markers:
                    rationale.append("Minimal marker overlap; topics may be distinct.")

                crossrefs.append(
                    TopicCrossref(
                        topic_a=left.term,
                        topic_b=right.term,
                        shared_markers=shared_markers,
                        shared_keywords=shared_keywords[: self.config.max_keywords],
                        overlap_score=round(overlap_score, 4),
                        rationale=rationale,
                    )
                )
            if limit and len(crossrefs) >= limit:
                break

        if not crossrefs:
            summary = "Topic crossrefs: no overlapping topics detected."
        else:
            summary = "Topic crossrefs: " + ", ".join(
                f"{item.topic_a}+{item.topic_b}" for item in crossrefs
            ) + "."
        return crossrefs, summary

    def _select_evidence_sentences(
        self,
        content: str,
        markers: list[str],
        keywords: list[str],
        max_sentences: int,
    ) -> list[str]:
        if max_sentences <= 0:
            return []
        sentences = [seg.strip() for seg in SENTENCE_RE.findall(content) if seg.strip()]
        if not sentences:
            return []
        marker_set = {marker.lower() for marker in markers if marker}
        keyword_set = {word.lower() for word in keywords if word}
        scored: list[tuple[int, str]] = []
        for sentence in sentences:
            lower = sentence.lower()
            score = 0
            for marker in marker_set:
                if marker in lower:
                    score += 2
            for keyword in keyword_set:
                if keyword in lower:
                    score += 1
            scored.append((score, sentence))
        scored.sort(key=lambda item: (-item[0], item[1]))
        evidence = [
            self._clip_text(sentence, self.config.preview_chars)
            for score, sentence in scored
            if score > 0
        ][:max_sentences]
        if not evidence:
            evidence = [
                self._clip_text(sentence, self.config.preview_chars)
                for _, sentence in scored[:max_sentences]
            ]
        return evidence

    def _speech_opening(
        self,
        args: ContextSpokenTopicArgs,
        topic: TopicDefinition,
        topic_sequence: list[TopicDefinition],
        assign_mode: str,
    ) -> str:
        if not args.include_opening:
            return ""
        if assign_mode == "single":
            return f"Begin speaking about {topic.term}."
        names = ", ".join(item.term for item in topic_sequence if item.term)
        if not names:
            names = topic.term
        return f"Begin speaking across these topics: {names}."

    def _speech_closing(self, args: ContextSpokenTopicArgs) -> str:
        if not args.include_closing:
            return ""
        return "Close by keeping the speaking topic consistent."

    def _load_content(self, args: ContextSpokenTopicArgs) -> str:
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

    def _normalize_term(self, term: str) -> str:
        return " ".join(term.strip().split())

    def _normalize_description(self, description: str) -> str:
        return " ".join(description.strip().split())

    def _clip_text(self, text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        trimmed = " ".join(text.strip().split())
        if len(trimmed) <= max_chars:
            return trimmed
        return trimmed[:max_chars]

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextSpokenTopicArgs):
            return ToolCallDisplay(summary="context_spoken_topic")
        return ToolCallDisplay(
            summary="context_spoken_topic",
            details={
                "path": event.args.path,
                "topic": event.args.topic,
                "topics": event.args.topics,
                "assign_mode": event.args.assign_mode,
                "topic_sequence": event.args.topic_sequence,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpokenTopicResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=True,
            message=(
                f"Selected {event.result.topic} topic with "
                f"{event.result.segment_count} segment(s)"
            ),
            warnings=event.result.warnings,
            details={
                "topic": event.result.topic,
                "segment_count": event.result.segment_count,
                "assign_mode": event.result.assign_mode,
                "topic_sequence": event.result.topic_sequence,
                "has_understanding": event.result.topic_understanding is not None,
                "crossref_summary": event.result.crossref_summary,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Selecting speaking topic"
