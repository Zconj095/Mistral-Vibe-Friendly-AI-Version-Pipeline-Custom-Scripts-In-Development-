
from __future__ import annotations

import re
from collections import Counter, defaultdict
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
SPEAKER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _.'-]{0,40})\s*:\s*(.*)$")
NAME_SEQ_RE = re.compile(r"\b(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)+)\b")
NAME_WORD_RE = re.compile(r"\b[A-Z][A-Za-z0-9]{2,}\b")

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

QUESTION_WORDS = {
    "what",
    "why",
    "how",
    "who",
    "when",
    "where",
    "which",
}

REQUEST_TOKENS = {
    "please",
    "need",
    "needs",
    "should",
    "must",
    "request",
    "ask",
}
REQUEST_PHRASES = (
    "could you",
    "can you",
    "would you",
    "please",
    "need you",
)

ACTION_TOKENS = {
    "todo",
    "task",
    "action",
    "followup",
    "follow",
    "assign",
    "implement",
    "build",
    "fix",
}
ACTION_PHRASES = ("follow up", "action item", "next steps")

DECISION_TOKENS = {
    "decide",
    "decision",
    "decided",
    "choose",
    "chosen",
    "approve",
    "approved",
    "agreed",
}
DECISION_PHRASES = ("we decided", "i decided", "agreed to", "approved")

COMMITMENT_TOKENS = {
    "commit",
    "committed",
    "promise",
    "plan",
    "planning",
}
COMMITMENT_PHRASES = ("i will", "we will", "i'll", "we'll", "commit to", "promise to")

STATUS_TOKENS = {
    "status",
    "update",
    "progress",
    "current",
    "blocked",
    "blocker",
    "done",
    "complete",
    "completed",
}

SUMMARY_TOKENS = {"summary", "recap", "overview", "overall"}
SUMMARY_PHRASES = ("in short", "tl;dr", "to summarize")

AGREEMENT_TOKENS = {"yes", "agree", "agreed", "correct", "right", "ok", "okay", "sure"}
DISAGREEMENT_TOKENS = {"no", "disagree", "disagreed", "wrong", "not", "never", "cannot", "can't"}

CLARIFICATION_TOKENS = {
    "clarify",
    "clarification",
    "explain",
    "meaning",
    "rephrase",
    "detail",
    "details",
}

REFERENCE_TOKENS = {
    "refer",
    "reference",
    "see",
    "link",
    "doc",
    "documentation",
    "mentioned",
    "above",
    "below",
}

ISSUE_TOKENS = {"issue", "problem", "bug", "error", "risk", "concern", "failure"}
PLAN_TOKENS = {"plan", "planning", "roadmap", "schedule", "timeline", "next"}
IDEA_TOKENS = {"idea", "suggest", "suggestion", "option", "brainstorm", "thought"}

class ContextSpeechConversationCatalogConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_source_bytes: int = Field(
        default=5_000_000, description="Maximum bytes per content."
    )
    max_total_bytes: int = Field(
        default=20_000_000, description="Maximum bytes across utterances."
    )
    max_utterances: int = Field(default=500, description="Maximum utterances to process.")
    preview_chars: int = Field(default=240, description="Preview snippet length.")
    min_token_length: int = Field(default=3, description="Minimum token length.")
    max_keywords: int = Field(default=10, description="Max keywords per utterance.")
    max_named_terms: int = Field(default=8, description="Max named terms per utterance.")
    max_categories_per_utterance: int = Field(
        default=4, description="Max categories per utterance."
    )
    max_category_terms: int = Field(
        default=10, description="Max terms per category bucket."
    )
    max_topics: int = Field(default=20, description="Max topics to include.")
    max_category_segments: int = Field(
        default=10, description="Max category speech segments."
    )
    max_topic_segments: int = Field(default=10, description="Max topic speech segments.")
    max_speaker_segments: int = Field(
        default=6, description="Max speaker speech segments."
    )
    max_utterance_segments: int = Field(
        default=10, description="Max utterance speech segments."
    )
    max_speech_segments: int = Field(
        default=40, description="Max total speech segments."
    )
    default_split_mode: str = Field(
        default="lines", description="lines or sentences."
    )
    default_speaker_mode: str = Field(
        default="self", description="all, only, exclude, or self."
    )
    include_unlabeled_as_self: bool = Field(
        default=True, description="Treat unlabeled utterances as self."
    )
    self_speaker_labels: list[str] = Field(
        default_factory=lambda: [
            "assistant",
            "ai",
            "model",
            "bot",
            "mistral",
            "vibe",
            "system",
        ],
        description="Speaker labels treated as self.",
    )


class ContextSpeechConversationCatalogState(BaseToolState):
    pass


class ConversationUtterance(BaseModel):
    text: str = Field(description="Utterance text.")
    speaker: str | None = Field(default=None, description="Speaker label.")
    timestamp: str | None = Field(default=None, description="Optional timestamp.")


class ContextSpeechConversationCatalogArgs(BaseModel):
    content: str | None = Field(default=None, description="Conversation content.")
    path: str | None = Field(default=None, description="Path to transcript file.")
    utterances: list[ConversationUtterance] | None = Field(
        default=None, description="Explicit utterance list."
    )
    split_mode: str | None = Field(
        default=None, description="lines or sentences."
    )
    speaker_mode: str | None = Field(
        default=None, description="all, only, exclude, or self."
    )
    speakers: list[str] | None = Field(
        default=None, description="Speaker filter list."
    )
    max_source_bytes: int | None = Field(
        default=None, description="Override max_source_bytes."
    )
    max_total_bytes: int | None = Field(
        default=None, description="Override max_total_bytes."
    )
    max_utterances: int | None = Field(
        default=None, description="Override max_utterances."
    )
    min_token_length: int | None = Field(
        default=None, description="Override min_token_length."
    )
    max_keywords: int | None = Field(
        default=None, description="Override max_keywords."
    )
    max_named_terms: int | None = Field(
        default=None, description="Override max_named_terms."
    )
    max_categories_per_utterance: int | None = Field(
        default=None, description="Override max_categories_per_utterance."
    )
    max_category_terms: int | None = Field(
        default=None, description="Override max_category_terms."
    )
    max_topics: int | None = Field(default=None, description="Override max_topics.")
    max_category_segments: int | None = Field(
        default=None, description="Override max_category_segments."
    )
    max_topic_segments: int | None = Field(
        default=None, description="Override max_topic_segments."
    )
    max_speaker_segments: int | None = Field(
        default=None, description="Override max_speaker_segments."
    )
    max_utterance_segments: int | None = Field(
        default=None, description="Override max_utterance_segments."
    )
    max_speech_segments: int | None = Field(
        default=None, description="Override max_speech_segments."
    )
    include_speaker_summary: bool = Field(
        default=True, description="Include speaker summaries."
    )
    include_opening: bool = Field(
        default=True, description="Include speech opening."
    )
    include_closing: bool = Field(
        default=True, description="Include speech closing."
    )


class CatalogUtterance(BaseModel):
    index: int
    speaker: str | None
    text: str
    preview: str
    categories: list[str]
    keywords: list[str]
    named_terms: list[str]


class CategoryBucket(BaseModel):
    category: str
    count: int
    utterance_indices: list[int]
    top_terms: list[str]


class TopicBucket(BaseModel):
    term: str
    count: int
    utterance_indices: list[int]
    utterance_count: int


class CategoryCount(BaseModel):
    category: str
    count: int


class SpeakerSummary(BaseModel):
    speaker: str
    utterance_count: int
    utterance_indices: list[int]
    categories: list[CategoryCount]
    top_terms: list[str]


class SpeechSegment(BaseModel):
    index: int
    kind: str
    cue: str
    utterance_indices: list[int]
    categories: list[str]
    topics: list[str]
    speaker: str | None = None


class ContextSpeechConversationCatalogResult(BaseModel):
    utterances: list[CatalogUtterance]
    input_utterance_count: int
    utterance_count: int
    excluded_utterance_count: int
    category_buckets: list[CategoryBucket]
    category_count: int
    topic_buckets: list[TopicBucket]
    topic_count: int
    speaker_summaries: list[SpeakerSummary]
    speaker_count: int
    speaker_mode: str
    speakers: list[str]
    speech_opening: str
    speech_segments: list[SpeechSegment]
    speech_closing: str
    truncated: bool
    warnings: list[str]

class ContextSpeechConversationCatalog(
    BaseTool[
        ContextSpeechConversationCatalogArgs,
        ContextSpeechConversationCatalogResult,
        ContextSpeechConversationCatalogConfig,
        ContextSpeechConversationCatalogState,
    ],
    ToolUIData[
        ContextSpeechConversationCatalogArgs,
        ContextSpeechConversationCatalogResult,
    ],
):
    description: ClassVar[str] = (
        "Classify and organize what the assistant has said in a conversation."
    )

    async def run(
        self, args: ContextSpeechConversationCatalogArgs
    ) -> ContextSpeechConversationCatalogResult:
        max_source_bytes = args.max_source_bytes or self.config.max_source_bytes
        max_total_bytes = args.max_total_bytes or self.config.max_total_bytes
        max_utterances = (
            args.max_utterances if args.max_utterances is not None else self.config.max_utterances
        )
        min_token_length = args.min_token_length or self.config.min_token_length
        max_keywords = args.max_keywords or self.config.max_keywords
        max_named_terms = args.max_named_terms or self.config.max_named_terms
        max_categories = (
            args.max_categories_per_utterance
            if args.max_categories_per_utterance is not None
            else self.config.max_categories_per_utterance
        )
        max_category_terms = (
            args.max_category_terms
            if args.max_category_terms is not None
            else self.config.max_category_terms
        )
        max_topics = args.max_topics if args.max_topics is not None else self.config.max_topics
        max_category_segments = (
            args.max_category_segments
            if args.max_category_segments is not None
            else self.config.max_category_segments
        )
        max_topic_segments = (
            args.max_topic_segments
            if args.max_topic_segments is not None
            else self.config.max_topic_segments
        )
        max_speaker_segments = (
            args.max_speaker_segments
            if args.max_speaker_segments is not None
            else self.config.max_speaker_segments
        )
        max_utterance_segments = (
            args.max_utterance_segments
            if args.max_utterance_segments is not None
            else self.config.max_utterance_segments
        )
        max_speech_segments = (
            args.max_speech_segments
            if args.max_speech_segments is not None
            else self.config.max_speech_segments
        )

        split_mode = (args.split_mode or self.config.default_split_mode).strip().lower()
        if split_mode not in {"lines", "sentences"}:
            raise ToolError("split_mode must be lines or sentences.")

        speaker_mode = (args.speaker_mode or self.config.default_speaker_mode).strip().lower()
        if speaker_mode not in {"all", "only", "exclude", "self"}:
            raise ToolError("speaker_mode must be all, only, exclude, or self.")

        speakers = self._normalize_speakers(args.speakers or [])
        if speaker_mode in {"only", "exclude"} and not speakers:
            raise ToolError("speakers is required for speaker_mode only/exclude.")

        warnings: list[str] = []
        truncated = False

        utterances, input_count, bytes_truncated = self._load_utterances(
            args, max_source_bytes, max_total_bytes, split_mode
        )
        if bytes_truncated:
            warnings.append("Utterances truncated by byte budget.")
            truncated = True

        if max_utterances is not None and max_utterances > 0 and len(utterances) > max_utterances:
            warnings.append("Utterance limit reached; truncating list.")
            utterances = utterances[:max_utterances]
            truncated = True

        filtered = self._filter_utterances(
            utterances, speaker_mode, speakers
        )
        excluded_count = len(utterances) - len(filtered)
        if excluded_count:
            warnings.append("Some utterances excluded by speaker filters.")
        if not filtered:
            raise ToolError("No utterances available after filtering.")

        utterance_tokens: list[list[str]] = []
        utterance_categories: list[list[str]] = []
        catalog_entries: list[CatalogUtterance] = []

        for idx, utt in enumerate(filtered, start=1):
            tokens = self._tokenize(utt.text, min_token_length)
            categories = self._categorize(utt.text, tokens, max_categories)
            keywords = self._keywords(tokens, max_keywords)
            named_terms = self._named_terms(utt.text, max_named_terms)

            utterance_tokens.append(tokens)
            utterance_categories.append(categories)

            catalog_entries.append(
                CatalogUtterance(
                    index=idx,
                    speaker=utt.speaker,
                    text=utt.text,
                    preview=self._preview(utt.text),
                    categories=categories,
                    keywords=keywords,
                    named_terms=named_terms,
                )
            )

        category_buckets = self._build_category_buckets(
            utterance_categories,
            utterance_tokens,
            max_category_terms,
        )
        topic_buckets = self._build_topic_buckets(
            utterance_tokens,
            max_topics,
        )
        speaker_summaries = (
            self._build_speaker_summaries(
                catalog_entries,
                utterance_tokens,
                utterance_categories,
                max_category_terms,
            )
            if args.include_speaker_summary
            else []
        )

        speech_opening = self._speech_opening(
            args, speaker_mode, speakers, len(filtered)
        )
        speech_segments, segments_truncated = self._speech_segments(
            category_buckets,
            topic_buckets,
            speaker_summaries,
            catalog_entries,
            max_category_segments,
            max_topic_segments,
            max_speaker_segments,
            max_utterance_segments,
            max_speech_segments,
        )
        if segments_truncated:
            warnings.append("Speech segments truncated by limits.")
            truncated = True
        speech_closing = self._speech_closing(args)

        return ContextSpeechConversationCatalogResult(
            utterances=catalog_entries,
            input_utterance_count=input_count,
            utterance_count=len(filtered),
            excluded_utterance_count=excluded_count,
            category_buckets=category_buckets,
            category_count=len(category_buckets),
            topic_buckets=topic_buckets,
            topic_count=len(topic_buckets),
            speaker_summaries=speaker_summaries,
            speaker_count=len(speaker_summaries),
            speaker_mode=speaker_mode,
            speakers=speakers,
            speech_opening=speech_opening,
            speech_segments=speech_segments,
            speech_closing=speech_closing,
            truncated=truncated,
            warnings=warnings,
        )

    def _load_utterances(
        self,
        args: ContextSpeechConversationCatalogArgs,
        max_source_bytes: int,
        max_total_bytes: int,
        split_mode: str,
    ) -> tuple[list[ConversationUtterance], int, bool]:
        if args.utterances:
            utterances = list(args.utterances)
            total_bytes = 0
            limited: list[ConversationUtterance] = []
            truncated = False
            for utt in utterances:
                data = utt.text.encode("utf-8")
                if len(data) > max_source_bytes:
                    raise ToolError(
                        f"utterance exceeds max_source_bytes ({len(data)} > {max_source_bytes})."
                    )
                if total_bytes + len(data) > max_total_bytes:
                    truncated = True
                    break
                total_bytes += len(data)
                limited.append(utt)
            return limited, len(utterances), truncated

        if args.content and args.path:
            raise ToolError("Provide content or path, not both.")

        if args.content is None and args.path is None:
            return [], 0, False

        if args.content is not None:
            data = args.content.encode("utf-8")
            if len(data) > max_source_bytes:
                raise ToolError(
                    f"content exceeds max_source_bytes ({len(data)} > {max_source_bytes})."
                )
            content = args.content
        else:
            path = self._resolve_path(args.path or "")
            if path.is_dir():
                raise ToolError(f"Path is a directory: {path}")
            size = path.stat().st_size
            if size > max_source_bytes:
                raise ToolError(
                    f"{path} exceeds max_source_bytes ({size} > {max_source_bytes})."
                )
            content = path.read_text("utf-8", errors="ignore")

        utterances: list[ConversationUtterance] = []
        if split_mode == "sentences":
            for sentence in SENTENCE_RE.findall(content):
                text = sentence.strip()
                if text:
                    utterances.append(ConversationUtterance(text=text))
            return utterances, len(utterances), False

        for line in content.splitlines():
            text = line.strip()
            if not text:
                continue
            speaker = None
            match = SPEAKER_RE.match(text)
            if match:
                speaker = match.group(1).strip()
                text = match.group(2).strip()
            utterances.append(ConversationUtterance(text=text, speaker=speaker))
        return utterances, len(utterances), False

    def _resolve_path(self, raw_path: str) -> Path:
        if not raw_path.strip():
            raise ToolError("path cannot be empty.")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.config.effective_workdir / path
        return path.resolve()

    def _tokenize(self, text: str, min_token_length: int) -> list[str]:
        return [
            token.lower()
            for token in WORD_RE.findall(text)
            if len(token) >= min_token_length
        ]

    def _keywords(self, tokens: list[str], max_keywords: int) -> list[str]:
        candidates = [token for token in tokens if token not in STOPWORDS]
        counter = Counter(candidates)
        most_common = counter.most_common(max_keywords)
        return [token for token, _ in most_common]

    def _named_terms(self, text: str, max_named_terms: int) -> list[str]:
        found: list[str] = []
        found.extend(match.group(0).strip() for match in NAME_SEQ_RE.finditer(text))
        found.extend(match.group(0).strip() for match in NAME_WORD_RE.finditer(text))
        seen: set[str] = set()
        results: list[str] = []
        for term in found:
            key = term.lower()
            if key in seen:
                continue
            if key in STOPWORDS:
                continue
            seen.add(key)
            results.append(term)
            if max_named_terms and len(results) >= max_named_terms:
                break
        return results

    def _categorize(
        self, text: str, tokens: list[str], max_categories: int
    ) -> list[str]:
        lowered = text.lower()
        categories: list[str] = []

        def add(cat: str) -> None:
            if cat not in categories:
                categories.append(cat)

        if "?" in text or any(word in tokens for word in QUESTION_WORDS):
            add("question")
        if any(word in tokens for word in REQUEST_TOKENS) or any(
            phrase in lowered for phrase in REQUEST_PHRASES
        ):
            add("request")
        if any(word in tokens for word in ACTION_TOKENS) or any(
            phrase in lowered for phrase in ACTION_PHRASES
        ):
            add("action_item")
        if any(word in tokens for word in DECISION_TOKENS) or any(
            phrase in lowered for phrase in DECISION_PHRASES
        ):
            add("decision")
        if any(word in tokens for word in COMMITMENT_TOKENS) or any(
            phrase in lowered for phrase in COMMITMENT_PHRASES
        ):
            add("commitment")
        if any(word in tokens for word in STATUS_TOKENS):
            add("status")
        if any(word in tokens for word in SUMMARY_TOKENS) or any(
            phrase in lowered for phrase in SUMMARY_PHRASES
        ):
            add("summary")
        if any(word in tokens for word in PLAN_TOKENS):
            add("plan")
        if any(word in tokens for word in IDEA_TOKENS):
            add("idea")
        if any(word in tokens for word in AGREEMENT_TOKENS):
            add("agreement")
        if any(word in tokens for word in DISAGREEMENT_TOKENS):
            add("disagreement")
        if any(word in tokens for word in CLARIFICATION_TOKENS):
            add("clarification")
        if any(word in tokens for word in REFERENCE_TOKENS):
            add("reference")
        if any(word in tokens for word in ISSUE_TOKENS):
            add("issue")

        if not categories:
            categories.append("statement")

        if max_categories is not None and max_categories > 0:
            return categories[:max_categories]
        return categories

    def _normalize_speakers(self, speakers: list[str]) -> list[str]:
        normalized = []
        for speaker in speakers:
            name = self._normalize_speaker(speaker)
            if name:
                normalized.append(name)
        return sorted(set(normalized))

    def _normalize_speaker(self, speaker: str | None) -> str:
        if not speaker:
            return ""
        return " ".join(speaker.strip().lower().split())

    def _filter_utterances(
        self,
        utterances: list[ConversationUtterance],
        speaker_mode: str,
        speakers: list[str],
    ) -> list[ConversationUtterance]:
        if speaker_mode == "all":
            return list(utterances)

        if speaker_mode == "self":
            allowed = {self._normalize_speaker(label) for label in self.config.self_speaker_labels}
            filtered: list[ConversationUtterance] = []
            for utt in utterances:
                speaker = self._normalize_speaker(utt.speaker)
                if speaker and speaker in allowed:
                    filtered.append(utt)
                elif not speaker and self.config.include_unlabeled_as_self:
                    filtered.append(utt)
            return filtered

        filtered = []
        speaker_set = set(speakers)
        for utt in utterances:
            speaker = self._normalize_speaker(utt.speaker)
            in_set = speaker in speaker_set if speaker else False
            if speaker_mode == "only" and in_set:
                filtered.append(utt)
            elif speaker_mode == "exclude" and not in_set:
                filtered.append(utt)
        return filtered

    def _build_category_buckets(
        self,
        utterance_categories: list[list[str]],
        utterance_tokens: list[list[str]],
        max_category_terms: int,
    ) -> list[CategoryBucket]:
        category_indices: dict[str, list[int]] = defaultdict(list)
        category_terms: dict[str, Counter] = defaultdict(Counter)
        for idx, categories in enumerate(utterance_categories, start=1):
            tokens = [token for token in utterance_tokens[idx - 1] if token not in STOPWORDS]
            for category in categories:
                category_indices[category].append(idx)
                category_terms[category].update(tokens)

        buckets: list[CategoryBucket] = []
        for category, indices in category_indices.items():
            top_terms = [
                term for term, _ in category_terms[category].most_common(max_category_terms)
            ]
            buckets.append(
                CategoryBucket(
                    category=category,
                    count=len(indices),
                    utterance_indices=indices,
                    top_terms=top_terms,
                )
            )
        buckets.sort(key=lambda item: (-item.count, item.category))
        return buckets

    def _build_topic_buckets(
        self,
        utterance_tokens: list[list[str]],
        max_topics: int,
    ) -> list[TopicBucket]:
        token_counter: Counter = Counter()
        term_indices: dict[str, list[int]] = defaultdict(list)
        for idx, tokens in enumerate(utterance_tokens, start=1):
            filtered = [token for token in tokens if token not in STOPWORDS]
            token_counter.update(filtered)
            for token in set(filtered):
                term_indices[token].append(idx)

        buckets: list[TopicBucket] = []
        for term, count in token_counter.most_common(max_topics):
            indices = term_indices.get(term, [])
            buckets.append(
                TopicBucket(
                    term=term,
                    count=count,
                    utterance_indices=indices,
                    utterance_count=len(indices),
                )
            )
        return buckets

    def _build_speaker_summaries(
        self,
        utterances: list[CatalogUtterance],
        utterance_tokens: list[list[str]],
        utterance_categories: list[list[str]],
        max_terms: int,
    ) -> list[SpeakerSummary]:
        speaker_terms: dict[str, Counter] = defaultdict(Counter)
        speaker_counts: dict[str, Counter] = defaultdict(Counter)
        speaker_indices: dict[str, list[int]] = defaultdict(list)
        for idx, utterance in enumerate(utterances, start=1):
            speaker = utterance.speaker or "Unknown"
            speaker_indices[speaker].append(idx)
            speaker_terms[speaker].update(
                token for token in utterance_tokens[idx - 1] if token not in STOPWORDS
            )
            speaker_counts[speaker].update(utterance_categories[idx - 1])

        summaries: list[SpeakerSummary] = []
        for speaker, indices in speaker_indices.items():
            category_counts = [
                CategoryCount(category=cat, count=count)
                for cat, count in speaker_counts[speaker].most_common()
            ]
            top_terms = [term for term, _ in speaker_terms[speaker].most_common(max_terms)]
            summaries.append(
                SpeakerSummary(
                    speaker=speaker,
                    utterance_count=len(indices),
                    utterance_indices=indices,
                    categories=category_counts,
                    top_terms=top_terms,
                )
            )
        summaries.sort(key=lambda item: (-item.utterance_count, item.speaker))
        return summaries

    def _speech_opening(
        self,
        args: ContextSpeechConversationCatalogArgs,
        speaker_mode: str,
        speakers: list[str],
        utterance_count: int,
    ) -> str:
        if not args.include_opening:
            return ""
        parts = [f"Catalog {utterance_count} utterance(s)"]
        if speaker_mode == "self":
            parts.append("spoken by the assistant.")
        elif speaker_mode == "only" and speakers:
            parts.append(f"from {', '.join(speakers)}.")
        elif speaker_mode == "exclude" and speakers:
            parts.append(f"excluding {', '.join(speakers)}.")
        else:
            parts.append("from the conversation.")
        parts.append("Organize by categories and topics for quick recall.")
        return " ".join(parts)

    def _speech_segments(
        self,
        category_buckets: list[CategoryBucket],
        topic_buckets: list[TopicBucket],
        speaker_summaries: list[SpeakerSummary],
        utterances: list[CatalogUtterance],
        max_category_segments: int,
        max_topic_segments: int,
        max_speaker_segments: int,
        max_utterance_segments: int,
        max_speech_segments: int,
    ) -> tuple[list[SpeechSegment], bool]:
        segments: list[SpeechSegment] = []

        if max_category_segments and category_buckets:
            for bucket in category_buckets[:max_category_segments]:
                terms = ", ".join(bucket.top_terms[:5])
                cue = f"Review {bucket.category} items {bucket.utterance_indices}."
                if terms:
                    cue = f"{cue} Key terms: {terms}."
                segments.append(
                    SpeechSegment(
                        index=len(segments) + 1,
                        kind="category",
                        cue=cue,
                        utterance_indices=bucket.utterance_indices,
                        categories=[bucket.category],
                        topics=[],
                    )
                )

        if max_topic_segments and topic_buckets:
            for topic in topic_buckets[:max_topic_segments]:
                cue = f"Connect topic '{topic.term}' across utterances {topic.utterance_indices}."
                segments.append(
                    SpeechSegment(
                        index=len(segments) + 1,
                        kind="topic",
                        cue=cue,
                        utterance_indices=topic.utterance_indices,
                        categories=[],
                        topics=[topic.term],
                    )
                )

        if max_speaker_segments and speaker_summaries:
            for summary in speaker_summaries[:max_speaker_segments]:
                category_labels = ", ".join(
                    item.category for item in summary.categories[:4]
                )
                cue_parts = [f"Summarize {summary.speaker} contributions."]
                if category_labels:
                    cue_parts.append(f"Focus on {category_labels}.")
                if summary.top_terms:
                    cue_parts.append(
                        f"Key terms: {', '.join(summary.top_terms[:5])}."
                    )
                segments.append(
                    SpeechSegment(
                        index=len(segments) + 1,
                        kind="speaker",
                        cue=" ".join(cue_parts).strip(),
                        utterance_indices=summary.utterance_indices,
                        categories=[item.category for item in summary.categories[:4]],
                        topics=summary.top_terms[:5],
                        speaker=summary.speaker,
                    )
                )

        if max_utterance_segments and utterances:
            ranked = sorted(
                utterances,
                key=lambda utt: (-len(utt.categories), -len(utt.keywords), utt.index),
            )
            for utt in ranked[:max_utterance_segments]:
                cue = f"Highlight utterance {utt.index}: {utt.preview}"
                segments.append(
                    SpeechSegment(
                        index=len(segments) + 1,
                        kind="utterance",
                        cue=cue,
                        utterance_indices=[utt.index],
                        categories=utt.categories,
                        topics=utt.keywords[:5],
                        speaker=utt.speaker,
                    )
                )

        truncated = False
        if max_speech_segments and len(segments) > max_speech_segments:
            segments = segments[:max_speech_segments]
            truncated = True

        for idx, segment in enumerate(segments, start=1):
            segment.index = idx
        return segments, truncated

    def _speech_closing(self, args: ContextSpeechConversationCatalogArgs) -> str:
        if not args.include_closing:
            return ""
        return "Close by summarizing the most frequent categories and topics."

    def _preview(self, text: str) -> str:
        limit = self.config.preview_chars
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "..."

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextSpeechConversationCatalogArgs):
            return ToolCallDisplay(summary="context_speech_conversation_catalog")
        utterance_count = len(event.args.utterances or [])
        return ToolCallDisplay(
            summary="context_speech_conversation_catalog",
            details={
                "path": event.args.path,
                "utterance_count": utterance_count,
                "speaker_mode": event.args.speaker_mode,
                "speakers": event.args.speakers,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpeechConversationCatalogResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        warnings = event.result.warnings[:]
        if event.result.truncated:
            warnings.append("Output truncated by size or limits")
        return ToolResultDisplay(
            success=True,
            message=(
                f"Cataloged {event.result.utterance_count} utterance(s) into "
                f"{event.result.category_count} category(ies)"
            ),
            warnings=warnings,
            details={
                "utterance_count": event.result.utterance_count,
                "category_count": event.result.category_count,
                "topic_count": event.result.topic_count,
                "speaker_count": event.result.speaker_count,
                "speaker_mode": event.result.speaker_mode,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Cataloging conversation for speech"
