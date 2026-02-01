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
NUMBER_RE = re.compile(r"\b\d+\b")
TIMESTAMP_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\b")
DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
LOG_LEVEL_RE = re.compile(r"\b(INFO|WARN|WARNING|ERROR|DEBUG|TRACE|FATAL)\b")
BULLET_RE = re.compile(r"^\s*([-*+]|\d+\.|\d+\)|[a-zA-Z]\.)\s+")
SPEAKER_RE = re.compile(r"^\s*[A-Z][A-Za-z0-9_\- ]{0,24}:\s+")
EMAIL_HEADER_RE = re.compile(r"^\s*(from|to|subject|cc|bcc|date):", re.IGNORECASE)
QA_RE = re.compile(r"^\s*(q|a)[:\-]\s+", re.IGNORECASE)
CODE_FENCE_RE = re.compile(r"```")

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

TECH_MARKERS = {
    "api",
    "function",
    "method",
    "module",
    "class",
    "parameter",
    "config",
    "runtime",
    "pipeline",
    "framework",
    "algorithm",
    "dataset",
    "schema",
    "cache",
    "compile",
    "deploy",
    "integration",
    "version",
    "interface",
}

ACADEMIC_MARKERS = {
    "abstract",
    "method",
    "methods",
    "results",
    "conclusion",
    "study",
    "experiment",
    "hypothesis",
    "analysis",
    "dataset",
    "citation",
    "evidence",
}

LEGAL_MARKERS = {
    "shall",
    "hereby",
    "pursuant",
    "whereas",
    "herein",
    "thereof",
    "witnesseth",
    "liability",
    "agreement",
    "party",
    "terms",
}

EMAIL_SIGNOFFS = [
    "regards",
    "sincerely",
    "best",
    "thanks",
    "thank you",
]

FORM_LABELS = [
    "narrative",
    "dialogue",
    "chat",
    "instruction",
    "list",
    "log",
    "technical",
    "academic",
    "legal",
    "persuasive",
    "qa",
    "poetic",
    "email",
    "data",
    "general",
]


@dataclass
class _ContextSignals:
    tone: str
    dominant: str
    primary_intent: str
    primary_form: str


class ContextTextFormsMultiConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_items: int = Field(default=200, description="Maximum items to process.")
    max_source_bytes: int = Field(
        default=5_000_000, description="Maximum size per file (bytes)."
    )
    max_total_bytes: int = Field(
        default=20_000_000, description="Maximum total bytes across files."
    )
    preview_chars: int = Field(default=400, description="Preview length per item.")
    max_tokens_per_item: int = Field(
        default=500, description="Maximum tokens stored per item."
    )
    max_keywords: int = Field(default=20, description="Maximum keywords per item.")
    min_token_length: int = Field(default=3, description="Minimum token length.")
    max_shared_tokens: int = Field(
        default=8, description="Maximum shared tokens per relationship."
    )
    min_overlap_tokens: int = Field(
        default=2, description="Minimum shared tokens to compare items."
    )
    min_similarity: float = Field(
        default=0.1, description="Minimum similarity for item links."
    )
    similarity_mode: str = Field(
        default="jaccard", description="jaccard or overlap."
    )
    max_neighbors: int = Field(
        default=6, description="Maximum neighbors per item."
    )
    max_relationships: int = Field(
        default=500, description="Maximum relationships returned."
    )
    min_form_similarity: float = Field(
        default=0.1, description="Minimum similarity for form-level links."
    )
    max_form_connections: int = Field(
        default=200, description="Maximum form-level connections returned."
    )
    max_forms_per_item: int = Field(
        default=5, description="Maximum form labels returned per item."
    )
    form_hint_boost: int = Field(
        default=3, description="Boost for matching form_hint labels."
    )


class ContextTextFormsMultiState(BaseToolState):
    pass

class TextFormItem(BaseModel):
    id: str | None = Field(default=None, description="Optional item id.")
    name: str | None = Field(default=None, description="Optional item name.")
    content: str | None = Field(default=None, description="Inline text content.")
    path: str | None = Field(default=None, description="Path to a text file.")
    form_hint: str | None = Field(default=None, description="Optional form hint.")
    source: str | None = Field(default=None, description="Source description.")
    audience: str | None = Field(default=None, description="Intended audience.")
    context: str | None = Field(default=None, description="Context/purpose notes.")
    intent_hint: str | None = Field(default=None, description="Optional intent hint.")
    perspective_hint: str | None = Field(
        default=None, description="Optional perspective hint."
    )
    origin: str | None = Field(default=None, description="Origin metadata.")
    tags: list[str] | None = Field(default=None, description="Optional tags.")


class ContextTextFormsMultiArgs(BaseModel):
    items: list[TextFormItem] = Field(description="Text items to analyze.")
    max_items: int | None = Field(default=None, description="Override max_items.")
    max_source_bytes: int | None = Field(
        default=None, description="Override max_source_bytes."
    )
    max_total_bytes: int | None = Field(
        default=None, description="Override max_total_bytes."
    )
    preview_chars: int | None = Field(default=None, description="Override preview length.")
    max_tokens_per_item: int | None = Field(
        default=None, description="Override max tokens per item."
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
        default=None, description="Override max neighbors per item."
    )
    max_relationships: int | None = Field(
        default=None, description="Override max relationships."
    )
    min_form_similarity: float | None = Field(
        default=None, description="Override minimum form similarity."
    )
    max_form_connections: int | None = Field(
        default=None, description="Override max form connections."
    )
    max_forms_per_item: int | None = Field(
        default=None, description="Override max forms per item."
    )
    form_hint_boost: int | None = Field(
        default=None, description="Override form_hint boost."
    )


class FormScore(BaseModel):
    label: str
    score: float
    count: int


class FormProfile(BaseModel):
    primary: str
    scores: list[FormScore]
    total_signals: int
    confidence: float


class FormSignals(BaseModel):
    line_count: int
    non_empty_lines: int
    bullet_lines: int
    speaker_lines: int
    qa_lines: int
    question_lines: int
    answer_lines: int
    log_lines: int
    timestamp_lines: int
    log_level_lines: int
    email_header_lines: int
    signoff_lines: int
    tabular_lines: int
    short_lines: int
    quote_lines: int
    quote_marks: int
    code_fence_lines: int
    numeric_ratio: float
    bullet_ratio: float
    qa_ratio: float
    short_line_ratio: float
    tabular_ratio: float


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


class TextFormAnalysis(BaseModel):
    index: int
    id: str | None
    name: str | None
    primary_form: str
    form_profile: FormProfile
    form_signals: FormSignals
    source: str | None
    audience: str | None
    context: str | None
    intent_hint: str | None
    perspective_hint: str | None
    origin: str | None
    tags: list[str]
    content_preview: str
    word_count: int
    token_count: int
    sentence_count: int
    paragraph_count: int
    keywords: list[str]
    intent_profile: IntentProfile
    perspective_profile: PerspectiveProfile


class FormRelationship(BaseModel):
    source_index: int
    target_index: int
    similarity: float
    overlap: int
    relationship: str
    shared_tokens: list[str]
    shifts: list[str]


class FormSummary(BaseModel):
    form: str
    item_count: int
    keywords: list[str]
    intents: list[IntentScore]
    tones: list[str]


class FormConnection(BaseModel):
    source_form: str
    target_form: str
    similarity: float
    overlap: int
    shared_tokens: list[str]


class ContextTextFormsMultiResult(BaseModel):
    items: list[TextFormAnalysis]
    relationships: list[FormRelationship]
    form_summaries: list[FormSummary]
    form_connections: list[FormConnection]
    item_count: int
    relationship_count: int
    form_count: int
    form_connection_count: int
    truncated: bool
    errors: list[str]

class ContextTextFormsMulti(
    BaseTool[
        ContextTextFormsMultiArgs,
        ContextTextFormsMultiResult,
        ContextTextFormsMultiConfig,
        ContextTextFormsMultiState,
    ],
    ToolUIData[ContextTextFormsMultiArgs, ContextTextFormsMultiResult],
):
    description: ClassVar[str] = (
        "Multi-context reasoning across different forms of text."
    )

    async def run(
        self, args: ContextTextFormsMultiArgs
    ) -> ContextTextFormsMultiResult:
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
        max_tokens_per_item = (
            args.max_tokens_per_item
            if args.max_tokens_per_item is not None
            else self.config.max_tokens_per_item
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
        min_form_similarity = (
            args.min_form_similarity
            if args.min_form_similarity is not None
            else self.config.min_form_similarity
        )
        max_form_connections = (
            args.max_form_connections
            if args.max_form_connections is not None
            else self.config.max_form_connections
        )
        max_forms_per_item = (
            args.max_forms_per_item
            if args.max_forms_per_item is not None
            else self.config.max_forms_per_item
        )
        form_hint_boost = (
            args.form_hint_boost
            if args.form_hint_boost is not None
            else self.config.form_hint_boost
        )

        if max_source_bytes <= 0:
            raise ToolError("max_source_bytes must be a positive integer.")
        if max_total_bytes <= 0:
            raise ToolError("max_total_bytes must be a positive integer.")
        if preview_chars < 0:
            raise ToolError("preview_chars must be >= 0.")
        if max_tokens_per_item < 0:
            raise ToolError("max_tokens_per_item must be >= 0.")
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
        if min_form_similarity < 0:
            raise ToolError("min_form_similarity must be >= 0.")
        if max_form_connections < 0:
            raise ToolError("max_form_connections must be >= 0.")
        if max_forms_per_item < 0:
            raise ToolError("max_forms_per_item must be >= 0.")
        if form_hint_boost < 0:
            raise ToolError("form_hint_boost must be >= 0.")

        items: list[TextFormAnalysis] = []
        token_sets: list[set[str]] = []
        token_counts: list[dict[str, int]] = []
        signals: list[_ContextSignals] = []
        errors: list[str] = []
        total_bytes = 0
        truncated = False

        for idx, item in enumerate(args.items, start=1):
            try:
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

                analysis, tokens, token_count_map, signal = self._analyze_item(
                    item,
                    content,
                    source_path,
                    preview_chars,
                    max_tokens_per_item,
                    max_keywords,
                    min_token_length,
                    max_forms_per_item,
                    form_hint_boost,
                )
                analysis.index = len(items) + 1
                items.append(analysis)
                token_sets.append(tokens)
                token_counts.append(token_count_map)
                signals.append(signal)
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not items:
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

        form_summaries, form_tokens = self._build_form_summaries(
            items, token_counts, max_keywords
        )
        form_connections = self._build_form_connections(
            form_tokens,
            min_form_similarity=min_form_similarity,
            similarity_mode=similarity_mode,
            max_shared_tokens=max_shared_tokens,
            max_form_connections=max_form_connections,
        )

        return ContextTextFormsMultiResult(
            items=items,
            relationships=relationships,
            form_summaries=form_summaries,
            form_connections=form_connections,
            item_count=len(items),
            relationship_count=len(relationships),
            form_count=len(form_summaries),
            form_connection_count=len(form_connections),
            truncated=truncated,
            errors=errors,
        )

    def _load_item_content(
        self, item: TextFormItem, max_source_bytes: int
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
        item: TextFormItem,
        content: str,
        source_path: str | None,
        preview_chars: int,
        max_tokens_per_item: int,
        max_keywords: int,
        min_token_length: int,
        max_forms_per_item: int,
        form_hint_boost: int,
    ) -> tuple[TextFormAnalysis, set[str], dict[str, int], _ContextSignals]:
        lines = content.splitlines()
        sentences = self._split_sentences(content)
        sentence_count = len(sentences)
        paragraph_count = self._count_paragraphs(lines)

        word_tokens = TOKEN_RE.findall(content)
        word_count = len(word_tokens)
        number_count = sum(1 for token in word_tokens if token.isdigit())

        token_count_map = self._extract_token_counts(content, min_token_length)
        token_count = sum(token_count_map.values())
        keywords = self._select_keywords(token_count_map, max_keywords)
        token_set = self._select_token_set(token_count_map, max_tokens_per_item)

        form_signals = self._build_form_signals(
            content, lines, word_count, number_count
        )
        form_profile = self._build_form_profile(
            content,
            word_tokens,
            token_count_map,
            form_signals,
            item.form_hint,
            max_forms_per_item,
            form_hint_boost,
        )
        intent_profile = self._build_intent_profile(content, sentences, lines)
        perspective_profile = self._build_perspective_profile(token_count_map)
        preview = self._preview_text(content, preview_chars)

        tags = [tag for tag in (item.tags or []) if tag]
        source_value = item.source or source_path

        analysis = TextFormAnalysis(
            index=0,
            id=item.id,
            name=item.name,
            primary_form=form_profile.primary,
            form_profile=form_profile,
            form_signals=form_signals,
            source=source_value,
            audience=item.audience,
            context=item.context,
            intent_hint=item.intent_hint,
            perspective_hint=item.perspective_hint,
            origin=item.origin,
            tags=tags,
            content_preview=preview,
            word_count=word_count,
            token_count=token_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            keywords=keywords,
            intent_profile=intent_profile,
            perspective_profile=perspective_profile,
        )

        signals = _ContextSignals(
            tone=perspective_profile.tone,
            dominant=perspective_profile.dominant,
            primary_intent=intent_profile.primary,
            primary_form=form_profile.primary,
        )
        return analysis, token_set, token_count_map, signals

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

    def _build_form_signals(
        self,
        content: str,
        lines: list[str],
        word_count: int,
        number_count: int,
    ) -> FormSignals:
        line_count = len(lines) if lines else 1
        non_empty_lines = 0
        bullet_lines = 0
        speaker_lines = 0
        qa_lines = 0
        question_lines = 0
        answer_lines = 0
        log_lines = 0
        timestamp_lines = 0
        log_level_lines = 0
        email_header_lines = 0
        signoff_lines = 0
        tabular_lines = 0
        short_lines = 0
        quote_lines = 0
        code_fence_lines = 0

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            non_empty_lines += 1
            if BULLET_RE.match(stripped):
                bullet_lines += 1
            if SPEAKER_RE.match(stripped):
                speaker_lines += 1
            if QA_RE.match(stripped):
                qa_lines += 1
                if stripped.lower().startswith("q"):
                    question_lines += 1
                else:
                    answer_lines += 1
            if "?" in stripped:
                question_lines += 1
            if stripped.startswith(">"):
                quote_lines += 1
            if CODE_FENCE_RE.search(stripped):
                code_fence_lines += 1

            has_timestamp = bool(TIMESTAMP_RE.search(stripped)) or (
                DATE_RE.search(stripped) and TIME_RE.search(stripped)
            )
            has_log_level = bool(LOG_LEVEL_RE.search(stripped))
            if has_timestamp:
                timestamp_lines += 1
            if has_log_level:
                log_level_lines += 1
            if has_timestamp or has_log_level:
                log_lines += 1

            if EMAIL_HEADER_RE.match(stripped):
                email_header_lines += 1
            if any(marker in stripped.lower() for marker in EMAIL_SIGNOFFS):
                signoff_lines += 1

            if "\t" in stripped or stripped.count("|") >= 2 or stripped.count(",") >= 2:
                tabular_lines += 1

            tokens = TOKEN_RE.findall(stripped)
            if tokens and len(tokens) <= 6:
                short_lines += 1

        non_empty = max(non_empty_lines, 1)
        numeric_ratio = number_count / max(word_count, 1)
        bullet_ratio = bullet_lines / non_empty
        qa_ratio = qa_lines / non_empty
        short_line_ratio = short_lines / non_empty
        tabular_ratio = tabular_lines / non_empty
        quote_marks = content.count('"')

        return FormSignals(
            line_count=line_count,
            non_empty_lines=non_empty_lines,
            bullet_lines=bullet_lines,
            speaker_lines=speaker_lines,
            qa_lines=qa_lines,
            question_lines=question_lines,
            answer_lines=answer_lines,
            log_lines=log_lines,
            timestamp_lines=timestamp_lines,
            log_level_lines=log_level_lines,
            email_header_lines=email_header_lines,
            signoff_lines=signoff_lines,
            tabular_lines=tabular_lines,
            short_lines=short_lines,
            quote_lines=quote_lines,
            quote_marks=quote_marks,
            code_fence_lines=code_fence_lines,
            numeric_ratio=round(numeric_ratio, 6),
            bullet_ratio=round(bullet_ratio, 6),
            qa_ratio=round(qa_ratio, 6),
            short_line_ratio=round(short_line_ratio, 6),
            tabular_ratio=round(tabular_ratio, 6),
        )

    def _build_form_profile(
        self,
        content: str,
        word_tokens: list[str],
        token_counts: dict[str, int],
        signals: FormSignals,
        form_hint: str | None,
        max_forms_per_item: int,
        form_hint_boost: int,
    ) -> FormProfile:
        text = content.lower()
        scores: dict[str, int] = {label: 0 for label in FORM_LABELS}

        narrative_hits = self._count_phrase_hits(text, NARRATIVE_MARKERS)
        persuade_hits = self._count_phrase_hits(text, PERSUADE_MARKERS)
        technical_hits = sum(token_counts.get(token, 0) for token in TECH_MARKERS)
        academic_hits = sum(token_counts.get(token, 0) for token in ACADEMIC_MARKERS)
        legal_hits = sum(token_counts.get(token, 0) for token in LEGAL_MARKERS)

        citation_hits = len(NUMBER_RE.findall(text)) if "[" in text or "(" in text else 0
        past_tense_hits = sum(1 for token in word_tokens if token.endswith("ed"))

        if signals.bullet_ratio >= 0.2:
            scores["list"] += 3
        scores["list"] += min(signals.bullet_lines, 6)

        if signals.speaker_lines >= 2:
            scores["dialogue"] += 2
        scores["dialogue"] += min(signals.speaker_lines, 6)
        if signals.quote_lines >= 2 or signals.quote_marks >= 4:
            scores["dialogue"] += 1

        if signals.speaker_lines > 0 and signals.short_line_ratio >= 0.5:
            scores["chat"] += min(signals.speaker_lines, 6) + 1
        if signals.qa_ratio >= 0.2:
            scores["chat"] += 1

        imperative_lines = self._count_imperatives(content.splitlines())
        if imperative_lines > 0:
            scores["instruction"] += min(imperative_lines, 6) + 1
        if "step" in text or "steps" in text or "instruction" in text:
            scores["instruction"] += 2
        if signals.bullet_ratio >= 0.2:
            scores["instruction"] += 1

        if signals.log_lines > 0:
            scores["log"] += min(signals.log_lines, 6) + 1
        if signals.timestamp_lines > 0:
            scores["log"] += 2
        if signals.log_level_lines > 0:
            scores["log"] += 1

        if technical_hits > 0:
            scores["technical"] += min(technical_hits, 6)
        if signals.code_fence_lines > 0:
            scores["technical"] += 1
        if signals.numeric_ratio >= 0.1:
            scores["technical"] += 1

        if academic_hits > 0:
            scores["academic"] += min(academic_hits, 6)
        if "abstract" in text or "method" in text or "results" in text:
            scores["academic"] += 2
        if citation_hits > 0:
            scores["academic"] += 1

        if legal_hits > 0:
            scores["legal"] += min(legal_hits, 6) + 1
        if "hereby" in text or "whereas" in text:
            scores["legal"] += 2

        if persuade_hits > 0:
            scores["persuasive"] += min(persuade_hits, 6)
        if "recommend" in text or "suggest" in text:
            scores["persuasive"] += 1

        if narrative_hits > 0:
            scores["narrative"] += min(narrative_hits, 6)
        if past_tense_hits > 0:
            scores["narrative"] += min(past_tense_hits, 3)

        if signals.qa_lines > 0:
            scores["qa"] += min(signals.qa_lines, 6) + 2
        if signals.question_lines > 0 and signals.qa_ratio >= 0.1:
            scores["qa"] += 1

        if signals.short_line_ratio >= 0.5 and signals.line_count >= 4:
            scores["poetic"] += 3
        if signals.short_line_ratio >= 0.7:
            scores["poetic"] += 1

        if signals.email_header_lines > 0:
            scores["email"] += min(signals.email_header_lines, 4) + 1
        if signals.signoff_lines > 0:
            scores["email"] += 2

        if signals.tabular_ratio >= 0.3:
            scores["data"] += 3
        if signals.numeric_ratio >= 0.3:
            scores["data"] += 2

        if form_hint:
            for hint in self._parse_form_hint(form_hint):
                if hint in scores:
                    scores[hint] += form_hint_boost

        total_signals = sum(scores.values())
        if total_signals <= 0:
            scores["general"] = max(1, signals.non_empty_lines)
            total_signals = scores["general"]

        form_scores = [
            FormScore(label=label, score=round(count / total_signals, 6), count=count)
            for label, count in scores.items()
            if count > 0
        ]
        form_scores.sort(key=lambda item: (-item.score, item.label))
        if max_forms_per_item > 0:
            form_scores = form_scores[:max_forms_per_item]

        primary = form_scores[0].label if form_scores else "general"
        confidence = form_scores[0].score if form_scores else 0.0

        return FormProfile(
            primary=primary,
            scores=form_scores,
            total_signals=total_signals,
            confidence=round(confidence, 4),
        )

    def _parse_form_hint(self, hint: str) -> list[str]:
        parts = re.split(r"[;,/|]+", hint.lower())
        return [part.strip() for part in parts if part.strip()]

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
        signals: list[_ContextSignals],
        *,
        min_overlap_tokens: int,
        min_similarity: float,
        max_shared_tokens: int,
        similarity_mode: str,
        max_neighbors: int,
        max_relationships: int,
    ) -> tuple[list[FormRelationship], bool]:
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

        relationships: list[FormRelationship] = []
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
                FormRelationship(
                    source_index=left + 1,
                    target_index=right + 1,
                    similarity=round(similarity, 6),
                    overlap=overlap,
                    relationship=relation,
                    shared_tokens=shared,
                    shifts=shifts,
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
        self, left: _ContextSignals, right: _ContextSignals, meets_similarity: bool
    ) -> tuple[str, list[str]]:
        shifts: list[str] = []
        if left.primary_form != right.primary_form:
            shifts.append(f"form:{left.primary_form}->{right.primary_form}")
        if left.primary_intent != right.primary_intent:
            shifts.append(f"intent:{left.primary_intent}->{right.primary_intent}")
        if left.tone != right.tone:
            if left.tone != "neutral" and right.tone != "neutral":
                shifts.append(f"tone:{left.tone}->{right.tone}")
        if left.dominant != right.dominant:
            if left.dominant != "unknown" and right.dominant != "unknown":
                shifts.append(f"stance:{left.dominant}->{right.dominant}")

        if meets_similarity:
            if left.primary_form != right.primary_form:
                return "cross-form", shifts
            return "same-form", shifts
        return "adjacent", shifts

    def _build_form_summaries(
        self,
        items: list[TextFormAnalysis],
        token_counts: list[dict[str, int]],
        max_keywords: int,
    ) -> tuple[list[FormSummary], dict[str, set[str]]]:
        form_buckets: dict[str, dict[str, int]] = {}
        intent_buckets: dict[str, dict[str, int]] = {}
        tone_buckets: dict[str, dict[str, int]] = {}
        form_tokens: dict[str, set[str]] = {}

        for idx, item in enumerate(items):
            form = item.primary_form
            form_bucket = form_buckets.setdefault(form, {})
            for token, count in token_counts[idx].items():
                form_bucket[token] = form_bucket.get(token, 0) + count

            intents = intent_buckets.setdefault(form, {})
            intents[item.intent_profile.primary] = intents.get(item.intent_profile.primary, 0) + 1
            tones = tone_buckets.setdefault(form, {})
            tones[item.perspective_profile.tone] = tones.get(item.perspective_profile.tone, 0) + 1
            form_tokens.setdefault(form, set()).update(token_counts[idx].keys())

        summaries: list[FormSummary] = []
        for form, tokens in sorted(form_buckets.items()):
            keywords = self._select_keywords(tokens, max_keywords)
            intent_scores = self._bucket_to_intents(intent_buckets.get(form, {}))
            tones = self._bucket_to_values(tone_buckets.get(form, {}))
            summaries.append(
                FormSummary(
                    form=form,
                    item_count=sum(1 for item in items if item.primary_form == form),
                    keywords=keywords,
                    intents=intent_scores,
                    tones=[value.label for value in tones],
                )
            )

        return summaries, form_tokens

    def _build_form_connections(
        self,
        form_tokens: dict[str, set[str]],
        *,
        min_form_similarity: float,
        similarity_mode: str,
        max_shared_tokens: int,
        max_form_connections: int,
    ) -> list[FormConnection]:
        forms = sorted(form_tokens.keys())
        connections: list[FormConnection] = []
        for i in range(len(forms)):
            for j in range(i + 1, len(forms)):
                left = forms[i]
                right = forms[j]
                left_tokens = form_tokens[left]
                right_tokens = form_tokens[right]
                shared = left_tokens & right_tokens
                overlap = len(shared)
                if similarity_mode == "overlap":
                    similarity = float(overlap)
                    meets = similarity >= min_form_similarity
                else:
                    union = len(left_tokens | right_tokens)
                    similarity = overlap / union if union else 0.0
                    meets = similarity >= min_form_similarity
                if not meets:
                    continue
                shared_tokens = sorted(shared)
                if max_shared_tokens > 0 and len(shared_tokens) > max_shared_tokens:
                    shared_tokens = shared_tokens[:max_shared_tokens]
                connections.append(
                    FormConnection(
                        source_form=left,
                        target_form=right,
                        similarity=round(similarity, 6),
                        overlap=overlap,
                        shared_tokens=shared_tokens,
                    )
                )

        connections.sort(key=lambda item: (-item.similarity, -item.overlap, item.source_form))
        if max_form_connections > 0:
            connections = connections[:max_form_connections]
        return connections

    def _bucket_to_intents(self, bucket: dict[str, int]) -> list[IntentScore]:
        total = sum(bucket.values()) or 1
        scores = [
            IntentScore(label=label, score=round(count / total, 6), count=count)
            for label, count in bucket.items()
        ]
        scores.sort(key=lambda item: (-item.score, item.label))
        return scores

    def _bucket_to_values(self, bucket: dict[str, int]) -> list[FormScore]:
        total = sum(bucket.values()) or 1
        values = [
            FormScore(label=label, score=round(count / total, 6), count=count)
            for label, count in bucket.items()
        ]
        values.sort(key=lambda item: (-item.score, item.label))
        return values

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextTextFormsMultiArgs):
            return ToolCallDisplay(summary="context_text_forms_multi")

        summary = f"context_text_forms_multi: {len(event.args.items)} item(s)"
        return ToolCallDisplay(
            summary=summary,
            details={
                "item_count": len(event.args.items),
                "min_similarity": event.args.min_similarity,
                "max_relationships": event.args.max_relationships,
                "min_form_similarity": event.args.min_form_similarity,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextTextFormsMultiResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = (
            f"Processed {event.result.item_count} item(s) with "
            f"{event.result.relationship_count} relationship(s)"
        )
        warnings = event.result.errors[:]
        if event.result.truncated:
            warnings.append("Output truncated by size or limits")

        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=warnings,
            details={
                "item_count": event.result.item_count,
                "relationship_count": event.result.relationship_count,
                "form_count": event.result.form_count,
                "form_connection_count": event.result.form_connection_count,
                "truncated": event.result.truncated,
                "errors": event.result.errors,
                "items": event.result.items,
                "relationships": event.result.relationships,
                "form_summaries": event.result.form_summaries,
                "form_connections": event.result.form_connections,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Analyzing text forms"
