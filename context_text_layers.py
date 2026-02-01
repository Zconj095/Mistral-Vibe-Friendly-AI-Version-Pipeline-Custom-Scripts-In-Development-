from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
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


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
SENTENCE_RE = re.compile(r"[.!?]+")
CITATION_BRACKET_RE = re.compile(r"\[[0-9]{1,4}\]")
CITATION_PAREN_RE = re.compile(r"\([A-Za-z][A-Za-z\s,\.]{0,60}\d{4}\)")
NUMBER_RE = re.compile(r"\b\d+\b")

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

REQUEST_PHRASES = [
    "please",
    "could you",
    "would you",
    "can you",
    "i need",
    "i want",
    "need you",
    "help me",
]

INSTRUCT_VERBS = {
    "add",
    "apply",
    "build",
    "click",
    "compile",
    "create",
    "delete",
    "deploy",
    "execute",
    "install",
    "make",
    "open",
    "remove",
    "run",
    "save",
    "set",
    "test",
    "update",
    "use",
    "write",
}

PERSUADE_MARKERS = [
    "should",
    "must",
    "need to",
    "ought",
    "recommend",
    "suggest",
    "advise",
    "encourage",
]

REFLECT_MARKERS = [
    "i think",
    "i feel",
    "i believe",
    "in my opinion",
    "personally",
    "for me",
]

NARRATIVE_MARKERS = [
    "then",
    "after",
    "before",
    "yesterday",
    "today",
    "later",
    "earlier",
    "once",
    "when",
    "while",
]

WARN_MARKERS = [
    "warning",
    "beware",
    "risk",
    "danger",
    "avoid",
    "do not",
    "don't",
    "never",
]

SPECULATE_MARKERS = [
    "maybe",
    "might",
    "could",
    "possibly",
    "likely",
    "perhaps",
    "seems",
    "appears",
]

PLAN_MARKERS = [
    "plan",
    "roadmap",
    "next",
    "will",
    "going to",
    "intend",
    "future",
    "milestone",
    "todo",
]

PRONOUNS_FIRST = {"i", "me", "my", "mine", "we", "our", "us", "ours"}
PRONOUNS_SECOND = {"you", "your", "yours", "u"}
PRONOUNS_THIRD = {
    "he",
    "she",
    "they",
    "them",
    "his",
    "her",
    "hers",
    "their",
    "theirs",
    "it",
    "its",
}

POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "positive",
    "success",
    "benefit",
    "improve",
    "clear",
    "helpful",
    "strong",
    "effective",
}

NEGATIVE_WORDS = {
    "bad",
    "poor",
    "negative",
    "risk",
    "fail",
    "failure",
    "problem",
    "issue",
    "weak",
    "wrong",
    "error",
}

HEDGE_MARKERS = {
    "maybe",
    "might",
    "could",
    "possibly",
    "likely",
    "perhaps",
    "seems",
    "appears",
    "unclear",
}

ASSERTIVE_MARKERS = {
    "definitely",
    "certainly",
    "clearly",
    "must",
    "will",
    "always",
    "never",
    "prove",
    "shows",
}

NEGATION_WORDS = {"not", "no", "never", "without", "cannot", "can't", "dont", "don't"}

RATIONALE_MARKERS = ["because", "so that", "in order to", "due to", "since", "therefore", "thus"]


@dataclass
class _LayerSignals:
    tone: str
    dominant: str
    negation_ratio: float


class ContextTextLayersConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_items: int = Field(default=200, description="Maximum items to process.")
    max_source_bytes: int = Field(
        default=5_000_000, description="Maximum size per file (bytes)."
    )
    max_total_bytes: int = Field(
        default=20_000_000, description="Maximum total bytes across files."
    )
    preview_chars: int = Field(default=400, description="Preview length per item.")
    max_tokens_per_layer: int = Field(
        default=500, description="Maximum tokens stored per layer."
    )
    max_keywords: int = Field(default=20, description="Maximum keywords per layer.")
    min_token_length: int = Field(default=3, description="Minimum token length.")
    max_shared_tokens: int = Field(
        default=8, description="Maximum shared tokens per relationship."
    )
    min_overlap_tokens: int = Field(
        default=2, description="Minimum shared tokens to compare layers."
    )
    min_similarity: float = Field(
        default=0.1, description="Minimum similarity for reinforcing/contrast links."
    )
    similarity_mode: str = Field(
        default="jaccard", description="jaccard or overlap."
    )
    max_neighbors: int = Field(
        default=6, description="Maximum neighbors per layer."
    )
    max_relationships: int = Field(
        default=500, description="Maximum relationships returned."
    )
    max_rationale_sentences: int = Field(
        default=3, description="Maximum rationale sentences per layer."
    )


class ContextTextLayersState(BaseToolState):
    pass


class LayeredTextItem(BaseModel):
    id: str | None = Field(default=None, description="Optional item id.")
    layer: str | None = Field(default=None, description="Layer name.")
    depth: int | None = Field(default=None, description="Layer depth level.")
    content: str | None = Field(default=None, description="Inline content.")
    path: str | None = Field(default=None, description="Path to a text file.")
    source: str | None = Field(default=None, description="Source description.")
    audience: str | None = Field(default=None, description="Intended audience.")
    context: str | None = Field(default=None, description="Context/purpose provided.")
    intent_hint: str | None = Field(default=None, description="Optional intent hint.")
    perspective_hint: str | None = Field(
        default=None, description="Optional perspective hint."
    )
    origin: str | None = Field(default=None, description="Origin metadata.")
    tags: list[str] | None = Field(default=None, description="Optional tags.")


class ContextTextLayersArgs(BaseModel):
    items: list[LayeredTextItem] = Field(description="Layered text items.")
    max_items: int | None = Field(default=None, description="Override max_items.")
    max_source_bytes: int | None = Field(
        default=None, description="Override max_source_bytes."
    )
    max_total_bytes: int | None = Field(
        default=None, description="Override max_total_bytes."
    )
    preview_chars: int | None = Field(default=None, description="Override preview length.")
    max_tokens_per_layer: int | None = Field(
        default=None, description="Override max tokens per layer."
    )
    max_keywords: int | None = Field(default=None, description="Override max keywords.")
    min_token_length: int | None = Field(
        default=None, description="Override min token length."
    )
    max_shared_tokens: int | None = Field(
        default=None, description="Override max shared tokens."
    )
    min_overlap_tokens: int | None = Field(
        default=None, description="Override minimum overlap tokens."
    )
    min_similarity: float | None = Field(
        default=None, description="Override minimum similarity."
    )
    similarity_mode: str | None = Field(
        default=None, description="Override similarity mode."
    )
    max_neighbors: int | None = Field(
        default=None, description="Override max neighbors per layer."
    )
    max_relationships: int | None = Field(
        default=None, description="Override max relationships."
    )
    max_rationale_sentences: int | None = Field(
        default=None, description="Override max rationale sentences."
    )


class IntentScore(BaseModel):
    label: str
    score: float
    count: int


class IntentProfile(BaseModel):
    primary: str
    scores: list[IntentScore]
    total_signals: int
    confidence: float


class PerspectiveProfile(BaseModel):
    dominant: str
    first_person: int
    second_person: int
    third_person: int
    tone: str
    positive: int
    negative: int
    hedge: int
    assertive: int
    negations: int
    certainty: float


class EvidenceSignals(BaseModel):
    quote_lines: int
    citations: int
    numbers: int
    question_marks: int


class LayerAnalysis(BaseModel):
    index: int
    id: str | None
    layer: str
    depth: int
    source: str | None
    audience: str | None
    context: str | None
    intent_hint: str | None
    perspective_hint: str | None
    origin: str | None
    tags: list[str]
    content_preview: str
    token_count: int
    sentence_count: int
    paragraph_count: int
    keywords: list[str]
    intent_profile: IntentProfile
    perspective_profile: PerspectiveProfile
    inferred_purpose: str
    rationale_sentences: list[str]
    evidence: EvidenceSignals


class LayerRelationship(BaseModel):
    source_index: int
    target_index: int
    similarity: float
    overlap: int
    relationship: str
    shared_tokens: list[str]
    perspective_shift: list[str]


class ContextTextLayersResult(BaseModel):
    layers: list[LayerAnalysis]
    relationships: list[LayerRelationship]
    layer_count: int
    relationship_count: int
    truncated: bool
    errors: list[str]

class ContextTextLayers(
    BaseTool[
        ContextTextLayersArgs,
        ContextTextLayersResult,
        ContextTextLayersConfig,
        ContextTextLayersState,
    ],
    ToolUIData[ContextTextLayersArgs, ContextTextLayersResult],
):
    description: ClassVar[str] = (
        "Analyze layered text for meaning, intent, and perspective signals."
    )

    async def run(self, args: ContextTextLayersArgs) -> ContextTextLayersResult:
        if not args.items:
            raise ToolError("items is required.")

        max_items = args.max_items if args.max_items is not None else self.config.max_items
        if max_items <= 0:
            raise ToolError("max_items must be a positive integer.")
        if len(args.items) > max_items:
            raise ToolError(f"items exceeds max_items ({len(args.items)} > {max_items}).")

        max_source_bytes = (
            args.max_source_bytes
            if args.max_source_bytes is not None
            else self.config.max_source_bytes
        )
        max_total_bytes = (
            args.max_total_bytes
            if args.max_total_bytes is not None
            else self.config.max_total_bytes
        )
        preview_chars = (
            args.preview_chars
            if args.preview_chars is not None
            else self.config.preview_chars
        )
        max_tokens_per_layer = (
            args.max_tokens_per_layer
            if args.max_tokens_per_layer is not None
            else self.config.max_tokens_per_layer
        )
        max_keywords = (
            args.max_keywords
            if args.max_keywords is not None
            else self.config.max_keywords
        )
        min_token_length = (
            args.min_token_length
            if args.min_token_length is not None
            else self.config.min_token_length
        )
        max_shared_tokens = (
            args.max_shared_tokens
            if args.max_shared_tokens is not None
            else self.config.max_shared_tokens
        )
        min_overlap_tokens = (
            args.min_overlap_tokens
            if args.min_overlap_tokens is not None
            else self.config.min_overlap_tokens
        )
        min_similarity = (
            args.min_similarity
            if args.min_similarity is not None
            else self.config.min_similarity
        )
        similarity_mode = (
            args.similarity_mode
            if args.similarity_mode is not None
            else self.config.similarity_mode
        ).strip().lower()
        max_neighbors = (
            args.max_neighbors
            if args.max_neighbors is not None
            else self.config.max_neighbors
        )
        max_relationships = (
            args.max_relationships
            if args.max_relationships is not None
            else self.config.max_relationships
        )
        max_rationales = (
            args.max_rationale_sentences
            if args.max_rationale_sentences is not None
            else self.config.max_rationale_sentences
        )

        if max_source_bytes <= 0:
            raise ToolError("max_source_bytes must be a positive integer.")
        if max_total_bytes <= 0:
            raise ToolError("max_total_bytes must be a positive integer.")
        if preview_chars < 0:
            raise ToolError("preview_chars must be >= 0.")
        if max_tokens_per_layer < 0:
            raise ToolError("max_tokens_per_layer must be >= 0.")
        if max_keywords < 0:
            raise ToolError("max_keywords must be >= 0.")
        if min_token_length < 0:
            raise ToolError("min_token_length must be >= 0.")
        if max_shared_tokens < 0:
            raise ToolError("max_shared_tokens must be >= 0.")
        if min_overlap_tokens < 0:
            raise ToolError("min_overlap_tokens must be >= 0.")
        if min_similarity < 0:
            raise ToolError("min_similarity must be >= 0.")
        if similarity_mode not in {"jaccard", "overlap"}:
            raise ToolError("similarity_mode must be jaccard or overlap.")
        if max_neighbors < 0:
            raise ToolError("max_neighbors must be >= 0.")
        if max_relationships < 0:
            raise ToolError("max_relationships must be >= 0.")
        if max_rationales < 0:
            raise ToolError("max_rationale_sentences must be >= 0.")

        layers: list[LayerAnalysis] = []
        token_sets: list[set[str]] = []
        signals: list[_LayerSignals] = []
        errors: list[str] = []
        total_bytes = 0
        truncated = False

        for idx, item in enumerate(args.items, start=1):
            try:
                layer_name = (item.layer or "").strip()
                if not layer_name:
                    raise ToolError("layer is required for each item.")

                depth = item.depth if item.depth is not None else 1
                content, source_path, size_bytes = self._load_item_content(
                    item, max_source_bytes
                )
                if content is None:
                    raise ToolError("Item has no content to analyze.")
                if size_bytes is not None:
                    if total_bytes + size_bytes > max_total_bytes:
                        truncated = True
                        break
                    total_bytes += size_bytes

                analysis, token_set, layer_signal = self._analyze_item(
                    item,
                    content,
                    source_path,
                    layer_name,
                    depth,
                    preview_chars,
                    max_tokens_per_layer,
                    max_keywords,
                    min_token_length,
                    max_rationales,
                )
                analysis.index = len(layers) + 1
                layers.append(analysis)
                token_sets.append(token_set)
                signals.append(layer_signal)
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not layers:
            raise ToolError("No valid items to process.")

        relationships, rel_truncated = self._build_relationships(
            token_sets,
            signals,
            min_overlap_tokens=min_overlap_tokens,
            min_similarity=min_similarity,
            max_shared_tokens=max_shared_tokens,
            similarity_mode=similarity_mode,
            max_neighbors=max_neighbors,
            max_relationships=max_relationships,
        )
        if rel_truncated:
            truncated = True

        return ContextTextLayersResult(
            layers=layers,
            relationships=relationships,
            layer_count=len(layers),
            relationship_count=len(relationships),
            truncated=truncated,
            errors=errors,
        )

    def _load_item_content(
        self, item: LayeredTextItem, max_source_bytes: int
    ) -> tuple[str | None, str | None, int | None]:
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
                raise ToolError(f"Path is a directory, not a file: {path}")
            size = path.stat().st_size
            if size > max_source_bytes:
                raise ToolError(
                    f"{path} exceeds max_source_bytes ({size} > {max_source_bytes})."
                )
            return path.read_text("utf-8", errors="ignore"), str(path), size

        if item.content is not None:
            size = len(item.content.encode("utf-8"))
            if size > max_source_bytes:
                raise ToolError(
                    f"content exceeds max_source_bytes ({size} > {max_source_bytes})."
                )
            return item.content, None, size

        return None, None, None

    def _analyze_item(
        self,
        item: LayeredTextItem,
        content: str,
        source_path: str | None,
        layer_name: str,
        depth: int,
        preview_chars: int,
        max_tokens_per_layer: int,
        max_keywords: int,
        min_token_length: int,
        max_rationales: int,
    ) -> tuple[LayerAnalysis, set[str], _LayerSignals]:
        lines = content.splitlines()
        sentences = self._split_sentences(content)
        sentence_count = len(sentences)
        paragraph_count = self._count_paragraphs(lines)

        token_counts = self._extract_token_counts(content, min_token_length)
        token_count = sum(token_counts.values())
        keywords = self._select_keywords(token_counts, max_keywords)
        token_set = self._select_token_set(token_counts, max_tokens_per_layer)

        intent_profile = self._build_intent_profile(content, sentences, lines)
        perspective_profile = self._build_perspective_profile(token_counts)
        evidence = self._build_evidence_signals(content, lines)
        rationale_sentences = self._extract_rationales(sentences, max_rationales)
        inferred_purpose = self._infer_purpose(intent_profile.primary)
        preview = self._preview_text(content, preview_chars)

        tags = [tag for tag in (item.tags or []) if tag]
        source_value = item.source or source_path

        analysis = LayerAnalysis(
            index=0,
            id=item.id,
            layer=layer_name,
            depth=depth,
            source=source_value,
            audience=item.audience,
            context=item.context,
            intent_hint=item.intent_hint,
            perspective_hint=item.perspective_hint,
            origin=item.origin,
            tags=tags,
            content_preview=preview,
            token_count=token_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            keywords=keywords,
            intent_profile=intent_profile,
            perspective_profile=perspective_profile,
            inferred_purpose=inferred_purpose,
            rationale_sentences=rationale_sentences,
            evidence=evidence,
        )
        neg_ratio = 0.0
        if sentence_count > 0:
            neg_ratio = perspective_profile.negations / sentence_count

        signals = _LayerSignals(
            tone=perspective_profile.tone,
            dominant=perspective_profile.dominant,
            negation_ratio=neg_ratio,
        )
        return analysis, token_set, signals

    def _preview_text(self, text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def _split_sentences(self, text: str) -> list[str]:
        parts = SENTENCE_RE.split(text)
        return [part.strip() for part in parts if part.strip()]

    def _count_paragraphs(self, lines: list[str]) -> int:
        count = 0
        in_paragraph = False
        for line in lines:
            if line.strip():
                if not in_paragraph:
                    count += 1
                    in_paragraph = True
            else:
                in_paragraph = False
        return count

    def _extract_token_counts(self, text: str, min_len: int) -> dict[str, int]:
        tokens: dict[str, int] = {}
        for match in TOKEN_RE.findall(text.lower()):
            if len(match) < min_len:
                continue
            if match.isdigit():
                continue
            if match in STOPWORDS:
                continue
            tokens[match] = tokens.get(match, 0) + 1
        return tokens

    def _select_keywords(self, token_counts: dict[str, int], max_keywords: int) -> list[str]:
        if not token_counts:
            return []
        ordered = sorted(token_counts.items(), key=lambda item: (-item[1], item[0]))
        if max_keywords > 0:
            ordered = ordered[:max_keywords]
        return [token for token, _ in ordered]

    def _select_token_set(
        self, token_counts: dict[str, int], max_tokens: int
    ) -> set[str]:
        if not token_counts:
            return set()
        ordered = sorted(token_counts.items(), key=lambda item: (-item[1], item[0]))
        if max_tokens > 0:
            ordered = ordered[:max_tokens]
        return {token for token, _ in ordered}

    def _build_intent_profile(
        self, content: str, sentences: list[str], lines: list[str]
    ) -> IntentProfile:
        text = content.lower()
        counts: dict[str, int] = {
            "question": text.count("?") + self._count_interrogative_starts(sentences),
            "request": self._count_phrase_hits(text, REQUEST_PHRASES),
            "instruct": self._count_imperatives(lines),
            "persuade": self._count_phrase_hits(text, PERSUADE_MARKERS),
            "reflect": self._count_phrase_hits(text, REFLECT_MARKERS),
            "narrate": self._count_phrase_hits(text, NARRATIVE_MARKERS),
            "warn": self._count_phrase_hits(text, WARN_MARKERS),
            "speculate": self._count_phrase_hits(text, SPECULATE_MARKERS),
            "plan": self._count_phrase_hits(text, PLAN_MARKERS),
        }

        sentence_count = max(len(sentences), 1)
        other_total = sum(counts.values())
        if other_total == 0:
            counts["inform"] = sentence_count
        else:
            counts["inform"] = max(sentence_count - other_total, 1)

        total_signals = sum(counts.values())
        if total_signals <= 0:
            total_signals = 1

        scores = [
            IntentScore(
                label=label,
                score=round(count / total_signals, 6),
                count=count,
            )
            for label, count in counts.items()
        ]
        scores.sort(key=lambda item: (-item.score, item.label))
        primary = scores[0].label if scores else "inform"
        confidence = min(1.0, total_signals / sentence_count)

        return IntentProfile(
            primary=primary,
            scores=scores,
            total_signals=total_signals,
            confidence=round(confidence, 4),
        )

    def _build_perspective_profile(
        self, token_counts: dict[str, int]
    ) -> PerspectiveProfile:
        first = sum(token_counts.get(token, 0) for token in PRONOUNS_FIRST)
        second = sum(token_counts.get(token, 0) for token in PRONOUNS_SECOND)
        third = sum(token_counts.get(token, 0) for token in PRONOUNS_THIRD)

        positive = sum(token_counts.get(token, 0) for token in POSITIVE_WORDS)
        negative = sum(token_counts.get(token, 0) for token in NEGATIVE_WORDS)
        hedge = sum(token_counts.get(token, 0) for token in HEDGE_MARKERS)
        assertive = sum(token_counts.get(token, 0) for token in ASSERTIVE_MARKERS)
        negations = sum(token_counts.get(token, 0) for token in NEGATION_WORDS)

        dominant = self._dominant_pronoun(first, second, third)
        tone = self._tone_label(positive, negative)
        certainty = assertive / max(hedge + assertive, 1)

        return PerspectiveProfile(
            dominant=dominant,
            first_person=first,
            second_person=second,
            third_person=third,
            tone=tone,
            positive=positive,
            negative=negative,
            hedge=hedge,
            assertive=assertive,
            negations=negations,
            certainty=round(certainty, 4),
        )

    def _build_evidence_signals(
        self, content: str, lines: list[str]
    ) -> EvidenceSignals:
        quote_lines = sum(1 for line in lines if line.lstrip().startswith(">"))
        citations = len(CITATION_BRACKET_RE.findall(content)) + len(
            CITATION_PAREN_RE.findall(content)
        )
        numbers = len(NUMBER_RE.findall(content))
        question_marks = content.count("?")
        return EvidenceSignals(
            quote_lines=quote_lines,
            citations=citations,
            numbers=numbers,
            question_marks=question_marks,
        )

    def _extract_rationales(
        self, sentences: list[str], max_rationales: int
    ) -> list[str]:
        if max_rationales <= 0:
            return []
        collected: list[str] = []
        for sentence in sentences:
            lower = sentence.lower()
            if any(marker in lower for marker in RATIONALE_MARKERS):
                collected.append(sentence.strip())
            if len(collected) >= max_rationales:
                break
        return collected

    def _infer_purpose(self, primary_intent: str) -> str:
        mapping = {
            "request": "request action or assistance",
            "instruct": "provide instructions",
            "persuade": "influence decision",
            "reflect": "share perspective or reflection",
            "narrate": "share events or history",
            "warn": "alert to risks",
            "speculate": "explore possibilities",
            "plan": "outline next steps",
            "question": "seek information",
            "inform": "share information",
        }
        return mapping.get(primary_intent, "share information")

    def _count_phrase_hits(self, text: str, phrases: list[str]) -> int:
        count = 0
        for phrase in phrases:
            if not phrase:
                continue
            count += text.count(phrase)
        return count

    def _count_interrogative_starts(self, sentences: list[str]) -> int:
        count = 0
        for sentence in sentences:
            tokens = TOKEN_RE.findall(sentence.lower())
            if tokens and tokens[0] in QUESTION_WORDS:
                count += 1
        return count

    def _count_imperatives(self, lines: list[str]) -> int:
        count = 0
        for line in lines:
            tokens = TOKEN_RE.findall(line.lower())
            if tokens and tokens[0] in INSTRUCT_VERBS:
                count += 1
        return count

    def _dominant_pronoun(self, first: int, second: int, third: int) -> str:
        if first == 0 and second == 0 and third == 0:
            return "unknown"
        values = [(first, "first"), (second, "second"), (third, "third")]
        values.sort(reverse=True)
        if len(values) >= 2 and values[0][0] == values[1][0]:
            return "mixed"
        return values[0][1]

    def _tone_label(self, positive: int, negative: int) -> str:
        if positive == 0 and negative == 0:
            return "neutral"
        if positive > negative * 1.2:
            return "positive"
        if negative > positive * 1.2:
            return "negative"
        return "neutral"

    def _build_relationships(
        self,
        token_sets: list[set[str]],
        signals: list[_LayerSignals],
        *,
        min_overlap_tokens: int,
        min_similarity: float,
        max_shared_tokens: int,
        similarity_mode: str,
        max_neighbors: int,
        max_relationships: int,
    ) -> tuple[list[LayerRelationship], bool]:
        if len(token_sets) < 2:
            return [], False

        token_map: dict[str, list[int]] = {}
        for idx, tokens in enumerate(token_sets):
            for token in tokens:
                token_map.setdefault(token, []).append(idx)

        overlap_counts: dict[tuple[int, int], int] = {}
        for indices in token_map.values():
            if len(indices) < 2:
                continue
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    left = indices[i]
                    right = indices[j]
                    if right < left:
                        left, right = right, left
                    key = (left, right)
                    overlap_counts[key] = overlap_counts.get(key, 0) + 1

        candidates: list[tuple[float, int, int, int, str, list[str], list[str]]] = []
        for (left, right), overlap in overlap_counts.items():
            if overlap < min_overlap_tokens:
                continue

            left_tokens = token_sets[left]
            right_tokens = token_sets[right]
            if similarity_mode == "overlap":
                similarity = float(overlap)
                threshold = max(float(min_overlap_tokens), min_similarity)
                meets_similarity = similarity >= threshold
            else:
                union = len(left_tokens) + len(right_tokens) - overlap
                similarity = overlap / union if union else 0.0
                meets_similarity = similarity >= min_similarity

            shared = sorted(left_tokens & right_tokens)
            if max_shared_tokens > 0 and len(shared) > max_shared_tokens:
                shared = shared[:max_shared_tokens]

            relationship, shifts = self._classify_relationship(
                signals[left], signals[right], meets_similarity
            )
            if not meets_similarity:
                relationship = "adjacent"

            candidates.append(
                (similarity, overlap, left, right, relationship, shifts, shared)
            )

        candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))

        relationships: list[LayerRelationship] = []
        neighbor_counts = [0] * len(token_sets)
        truncated = False

        for similarity, overlap, left, right, relation, shifts, shared in candidates:
            if max_relationships > 0 and len(relationships) >= max_relationships:
                truncated = True
                break
            if max_neighbors > 0:
                if neighbor_counts[left] >= max_neighbors:
                    continue
                if neighbor_counts[right] >= max_neighbors:
                    continue

            relationships.append(
                LayerRelationship(
                    source_index=left + 1,
                    target_index=right + 1,
                    similarity=round(similarity, 6),
                    overlap=overlap,
                    relationship=relation,
                    shared_tokens=shared,
                    perspective_shift=shifts,
                )
            )
            neighbor_counts[left] += 1
            neighbor_counts[right] += 1

        if len(relationships) < len(candidates) and (
            max_relationships > 0 or max_neighbors > 0
        ):
            truncated = True

        return relationships, truncated

    def _classify_relationship(
        self, left: _LayerSignals, right: _LayerSignals, meets_similarity: bool
    ) -> tuple[str, list[str]]:
        shifts: list[str] = []
        if left.tone != right.tone:
            if left.tone != "neutral" and right.tone != "neutral":
                shifts.append(f"tone:{left.tone}->{right.tone}")
        if left.dominant != right.dominant:
            if left.dominant != "unknown" and right.dominant != "unknown":
                shifts.append(f"stance:{left.dominant}->{right.dominant}")
        if abs(left.negation_ratio - right.negation_ratio) >= 0.3:
            shifts.append("negation-shift")

        if meets_similarity and shifts:
            return "contrast", shifts
        if meets_similarity:
            return "reinforcing", shifts
        return "adjacent", shifts

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextTextLayersArgs):
            return ToolCallDisplay(summary="context_text_layers")

        summary = f"context_text_layers: {len(event.args.items)} item(s)"
        return ToolCallDisplay(
            summary=summary,
            details={
                "item_count": len(event.args.items),
                "max_items": event.args.max_items,
                "max_total_bytes": event.args.max_total_bytes,
                "min_similarity": event.args.min_similarity,
                "similarity_mode": event.args.similarity_mode,
                "max_neighbors": event.args.max_neighbors,
                "max_relationships": event.args.max_relationships,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextTextLayersResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = (
            f"Processed {event.result.layer_count} layer(s) with "
            f"{event.result.relationship_count} relationship(s)"
        )
        warnings = event.result.errors[:]
        if event.result.truncated:
            warnings.append("Output truncated by limits")

        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=warnings,
            details={
                "layer_count": event.result.layer_count,
                "relationship_count": event.result.relationship_count,
                "truncated": event.result.truncated,
                "errors": event.result.errors,
                "layers": event.result.layers,
                "relationships": event.result.relationships,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Analyzing text layers"
