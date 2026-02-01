from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import re
from typing import TYPE_CHECKING, ClassVar, Any

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


DEFAULT_PROFILE_PATH = Path.home() / ".vibe" / "memory" / "meta_context_profile.json"

CODE_EXTS = {
    ".py",
    ".pyi",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".cs",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".go",
    ".rs",
    ".php",
    ".rb",
    ".swift",
    ".kt",
    ".m",
    ".mm",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
}

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
class _ContextSignals:
    tone: str
    dominant: str
    primary_intent: str
    negation_ratio: float


@dataclass
class _ClusterState:
    token_counts: dict[str, int]
    context_indices: list[int]


class ContextMetaContextConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_items: int = Field(default=200, description="Maximum contexts to process.")
    max_source_bytes: int = Field(
        default=5_000_000, description="Maximum size per context (bytes)."
    )
    max_total_bytes: int = Field(
        default=20_000_000, description="Maximum total bytes across contexts."
    )
    preview_chars: int = Field(default=400, description="Preview length per context.")
    max_tokens_per_context: int = Field(
        default=500, description="Maximum tokens stored per context."
    )
    max_keywords: int = Field(default=20, description="Maximum keywords per context.")
    min_token_length: int = Field(default=3, description="Minimum token length.")
    max_shared_tokens: int = Field(
        default=8, description="Maximum shared tokens returned per link."
    )
    min_overlap_tokens: int = Field(
        default=2, description="Minimum shared tokens to compare contexts."
    )
    min_similarity: float = Field(
        default=0.1, description="Minimum similarity for context links."
    )
    similarity_mode: str = Field(
        default="jaccard", description="jaccard or overlap."
    )
    max_neighbors: int = Field(
        default=6, description="Maximum neighbors per context."
    )
    max_relationships: int = Field(
        default=500, description="Maximum relationships returned."
    )
    cluster_similarity: float = Field(
        default=0.2, description="Similarity threshold for clustering contexts."
    )
    max_clusters: int = Field(default=50, description="Maximum clusters to return.")
    max_profile_tokens: int = Field(
        default=200, description="Maximum tokens stored in the meta profile."
    )
    drift_similarity: float = Field(
        default=0.2, description="Similarity threshold for drift flagging."
    )
    profile_path: Path = Field(
        default=DEFAULT_PROFILE_PATH,
        description="Path to the persistent meta-context profile JSON.",
    )


class ContextMetaContextState(BaseToolState):
    pass


class ContextItem(BaseModel):
    id: str | None = Field(default=None, description="Optional context id.")
    name: str | None = Field(default=None, description="Optional context name.")
    content: str | None = Field(default=None, description="Inline content.")
    path: str | None = Field(default=None, description="Path to a context file.")
    kind: str | None = Field(default=None, description="auto, text, or code.")
    layer: str | None = Field(default=None, description="Optional layer label.")
    source: str | None = Field(default=None, description="Source description.")
    audience: str | None = Field(default=None, description="Intended audience.")
    context: str | None = Field(default=None, description="Context/purpose notes.")
    intent_hint: str | None = Field(default=None, description="Optional intent hint.")
    perspective_hint: str | None = Field(
        default=None, description="Optional perspective hint."
    )
    origin: str | None = Field(default=None, description="Origin metadata.")
    tags: list[str] | None = Field(default=None, description="Optional tags.")


class ContextMetaContextArgs(BaseModel):
    action: str | None = Field(
        default="analyze", description="analyze, update, profile, or clear."
    )
    items: list[ContextItem] | None = Field(
        default=None, description="Context items to analyze."
    )
    profile_path: str | None = Field(
        default=None, description="Override profile path."
    )
    max_profile_tokens: int | None = Field(
        default=None, description="Override max profile tokens."
    )
    max_items: int | None = Field(default=None, description="Override max_items.")
    max_source_bytes: int | None = Field(
        default=None, description="Override max_source_bytes."
    )
    max_total_bytes: int | None = Field(
        default=None, description="Override max_total_bytes."
    )
    preview_chars: int | None = Field(default=None, description="Override preview length.")
    max_tokens_per_context: int | None = Field(
        default=None, description="Override max tokens per context."
    )
    max_keywords: int | None = Field(
        default=None, description="Override max keywords."
    )
    min_token_length: int | None = Field(
        default=None, description="Override min token length."
    )
    max_shared_tokens: int | None = Field(
        default=None, description="Override max shared tokens."
    )
    min_overlap_tokens: int | None = Field(
        default=None, description="Override min overlap tokens."
    )
    min_similarity: float | None = Field(
        default=None, description="Override min similarity."
    )
    similarity_mode: str | None = Field(
        default=None, description="Override similarity mode."
    )
    max_neighbors: int | None = Field(
        default=None, description="Override max neighbors."
    )
    max_relationships: int | None = Field(
        default=None, description="Override max relationships."
    )
    cluster_similarity: float | None = Field(
        default=None, description="Override cluster similarity threshold."
    )
    max_clusters: int | None = Field(
        default=None, description="Override max clusters."
    )
    drift_similarity: float | None = Field(
        default=None, description="Override drift similarity threshold."
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


class ContextAnalysis(BaseModel):
    index: int
    id: str | None
    name: str | None
    kind: str
    layer: str | None
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
    inferred_purpose: str
    rationale_sentences: list[str]
    evidence: EvidenceSignals
    cluster_id: int | None


class ContextRelationship(BaseModel):
    source_index: int
    target_index: int
    similarity: float
    overlap: int
    relationship: str
    shared_tokens: list[str]
    perspective_shift: list[str]


class ContextDrift(BaseModel):
    from_index: int
    to_index: int
    similarity: float
    overlap: int
    intent_change: bool
    tone_change: bool
    perspective_change: bool
    flagged: bool
    notes: list[str]


class ContextCluster(BaseModel):
    cluster_id: int
    context_indices: list[int]
    keywords: list[str]
    size: int


class CountValue(BaseModel):
    value: str
    count: int
    ratio: float


class MetaContextProfile(BaseModel):
    context_count: int
    total_words: int
    total_tokens: int
    top_tokens: list[CountValue]
    intents: list[IntentScore]
    tones: list[CountValue]
    dominant_pronouns: list[CountValue]
    kinds: list[CountValue]
    layers: list[CountValue]
    updated_at: str | None


class ContextMetaContextResult(BaseModel):
    contexts: list[ContextAnalysis]
    relationships: list[ContextRelationship]
    drifts: list[ContextDrift]
    clusters: list[ContextCluster]
    meta_profile: MetaContextProfile | None
    context_count: int
    relationship_count: int
    drift_count: int
    cluster_count: int
    updated: bool
    truncated: bool
    errors: list[str]

class ContextMetaContext(
    BaseTool[
        ContextMetaContextArgs,
        ContextMetaContextResult,
        ContextMetaContextConfig,
        ContextMetaContextState,
    ],
    ToolUIData[ContextMetaContextArgs, ContextMetaContextResult],
):
    description: ClassVar[str] = (
        "Meta-contextual learning across contexts with clustering and drift signals."
    )

    async def run(self, args: ContextMetaContextArgs) -> ContextMetaContextResult:
        action = (args.action or "analyze").strip().lower()
        if action not in {"analyze", "update", "profile", "clear"}:
            raise ToolError("action must be analyze, update, profile, or clear.")

        profile_path = self._resolve_profile_path(args.profile_path)
        max_profile_tokens = (
            args.max_profile_tokens
            if args.max_profile_tokens is not None
            else self.config.max_profile_tokens
        )
        if max_profile_tokens < 0:
            raise ToolError("max_profile_tokens must be >= 0.")

        if action == "profile":
            profile = self._load_profile(profile_path, max_profile_tokens)
            return ContextMetaContextResult(
                contexts=[],
                relationships=[],
                drifts=[],
                clusters=[],
                meta_profile=profile,
                context_count=0,
                relationship_count=0,
                drift_count=0,
                cluster_count=0,
                updated=False,
                truncated=False,
                errors=[],
            )

        if action == "clear":
            removed = self._clear_profile(profile_path)
            return ContextMetaContextResult(
                contexts=[],
                relationships=[],
                drifts=[],
                clusters=[],
                meta_profile=None,
                context_count=0,
                relationship_count=0,
                drift_count=0,
                cluster_count=0,
                updated=removed,
                truncated=False,
                errors=[],
            )

        items = args.items or []
        if not items:
            raise ToolError("items is required for analyze/update.")

        max_items = args.max_items if args.max_items is not None else self.config.max_items
        if max_items <= 0:
            raise ToolError("max_items must be a positive integer.")
        if len(items) > max_items:
            raise ToolError(f"items exceeds max_items ({len(items)} > {max_items}).")

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
        max_tokens_per_context = (
            args.max_tokens_per_context
            if args.max_tokens_per_context is not None
            else self.config.max_tokens_per_context
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
        cluster_similarity = (
            args.cluster_similarity
            if args.cluster_similarity is not None
            else self.config.cluster_similarity
        )
        max_clusters = (
            args.max_clusters if args.max_clusters is not None else self.config.max_clusters
        )
        drift_similarity = (
            args.drift_similarity
            if args.drift_similarity is not None
            else self.config.drift_similarity
        )

        if max_source_bytes <= 0:
            raise ToolError("max_source_bytes must be a positive integer.")
        if max_total_bytes <= 0:
            raise ToolError("max_total_bytes must be a positive integer.")
        if preview_chars < 0:
            raise ToolError("preview_chars must be >= 0.")
        if max_tokens_per_context < 0:
            raise ToolError("max_tokens_per_context must be >= 0.")
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
        if cluster_similarity < 0:
            raise ToolError("cluster_similarity must be >= 0.")
        if max_clusters < 0:
            raise ToolError("max_clusters must be >= 0.")
        if drift_similarity < 0:
            raise ToolError("drift_similarity must be >= 0.")

        contexts: list[ContextAnalysis] = []
        token_sets: list[set[str]] = []
        token_counts: list[dict[str, int]] = []
        signals: list[_ContextSignals] = []
        errors: list[str] = []
        total_bytes = 0
        truncated = False

        for idx, item in enumerate(items, start=1):
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
                    max_tokens_per_context,
                    max_keywords,
                    min_token_length,
                )
                analysis.index = len(contexts) + 1
                contexts.append(analysis)
                token_sets.append(tokens)
                token_counts.append(token_count_map)
                signals.append(signal)
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not contexts:
            raise ToolError("No valid items to process.")

        clusters, assignments = self._build_clusters(
            token_counts,
            cluster_similarity,
            max_clusters,
            similarity_mode,
            max_keywords,
        )
        for idx, cluster_id in enumerate(assignments):
            contexts[idx].cluster_id = cluster_id

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

        drifts = self._build_drifts(
            contexts,
            token_sets,
            signals,
            similarity_mode,
            drift_similarity,
        )

        meta_profile = self._build_meta_profile(contexts, token_counts, max_profile_tokens)
        updated = False
        if action == "update":
            meta_profile = self._update_profile(
                profile_path, meta_profile, token_counts, contexts, max_profile_tokens
            )
            updated = True

        return ContextMetaContextResult(
            contexts=contexts,
            relationships=relationships,
            drifts=drifts,
            clusters=clusters,
            meta_profile=meta_profile,
            context_count=len(contexts),
            relationship_count=len(relationships),
            drift_count=len(drifts),
            cluster_count=len(clusters),
            updated=updated,
            truncated=truncated,
            errors=errors,
        )

    def _resolve_profile_path(self, raw: str | None) -> Path:
        path = Path(raw).expanduser() if raw else self.config.profile_path
        if not path.is_absolute():
            path = self.config.effective_workdir / path
        return path.resolve()

    def _clear_profile(self, path: Path) -> bool:
        if path.exists():
            path.unlink()
            return True
        return False

    def _load_profile(self, path: Path, max_profile_tokens: int) -> MetaContextProfile:
        if not path.exists():
            return self._empty_profile()
        try:
            data = json.loads(path.read_text("utf-8"))
        except Exception as exc:
            raise ToolError(f"Failed to load profile: {exc}") from exc

        if not isinstance(data, dict):
            return self._empty_profile()
        return self._profile_from_store(data, max_profile_tokens)

    def _update_profile(
        self,
        path: Path,
        run_profile: MetaContextProfile,
        token_counts: list[dict[str, int]],
        contexts: list[ContextAnalysis],
        max_profile_tokens: int,
    ) -> MetaContextProfile:
        store = {
            "version": 1,
            "context_count": 0,
            "total_words": 0,
            "total_tokens": 0,
            "token_counts": {},
            "intent_counts": {},
            "tone_counts": {},
            "pronoun_counts": {},
            "kind_counts": {},
            "layer_counts": {},
            "updated_at": None,
        }
        if path.exists():
            try:
                loaded = json.loads(path.read_text("utf-8"))
                if isinstance(loaded, dict):
                    store.update(loaded)
            except Exception:
                pass

        store["context_count"] = int(store.get("context_count", 0)) + run_profile.context_count
        store["total_words"] = int(store.get("total_words", 0)) + run_profile.total_words
        store["total_tokens"] = int(store.get("total_tokens", 0)) + run_profile.total_tokens

        token_bucket = store.get("token_counts")
        if not isinstance(token_bucket, dict):
            token_bucket = {}
            store["token_counts"] = token_bucket

        for counts in token_counts:
            for token, count in counts.items():
                token_bucket[token] = int(token_bucket.get(token, 0)) + int(count)

        self._merge_counts(store, "intent_counts", run_profile.intents)
        self._merge_counts(store, "tone_counts", run_profile.tones)
        self._merge_counts(store, "pronoun_counts", run_profile.dominant_pronouns)
        self._merge_counts(store, "kind_counts", run_profile.kinds)
        self._merge_counts(store, "layer_counts", run_profile.layers)

        store["updated_at"] = datetime.utcnow().isoformat()

        token_bucket = self._prune_bucket(token_bucket, max_profile_tokens)
        store["token_counts"] = token_bucket

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(store, indent=2), "utf-8")
        return self._profile_from_store(store, max_profile_tokens)

    def _merge_counts(
        self, store: dict[str, Any], key: str, values: list[Any]
    ) -> None:
        bucket = store.get(key)
        if not isinstance(bucket, dict):
            bucket = {}
            store[key] = bucket
        for item in values:
            value = getattr(item, "value", None)
            if value is None:
                value = getattr(item, "label", None)
            if value is None:
                continue
            count = int(getattr(item, "count", 0))
            bucket[value] = int(bucket.get(value, 0)) + count

    def _prune_bucket(self, bucket: dict[str, int], max_items: int) -> dict[str, int]:
        if max_items <= 0:
            return bucket
        ordered = sorted(bucket.items(), key=lambda item: (-item[1], item[0]))
        pruned = ordered[:max_items]
        return {key: int(value) for key, value in pruned}

    def _profile_from_store(
        self, store: dict[str, Any], max_profile_tokens: int
    ) -> MetaContextProfile:
        context_count = int(store.get("context_count", 0))
        total_words = int(store.get("total_words", 0))
        total_tokens = int(store.get("total_tokens", 0))
        token_bucket = store.get("token_counts")
        if not isinstance(token_bucket, dict):
            token_bucket = {}

        top_tokens = self._bucket_to_values(token_bucket, max_profile_tokens)
        intents = self._bucket_to_intents(store.get("intent_counts"))
        tones = self._bucket_to_values(store.get("tone_counts"))
        pronouns = self._bucket_to_values(store.get("pronoun_counts"))
        kinds = self._bucket_to_values(store.get("kind_counts"))
        layers = self._bucket_to_values(store.get("layer_counts"))
        updated_at = store.get("updated_at")
        updated_at_value = updated_at if isinstance(updated_at, str) else None

        return MetaContextProfile(
            context_count=context_count,
            total_words=total_words,
            total_tokens=total_tokens,
            top_tokens=top_tokens,
            intents=intents,
            tones=tones,
            dominant_pronouns=pronouns,
            kinds=kinds,
            layers=layers,
            updated_at=updated_at_value,
        )

    def _empty_profile(self) -> MetaContextProfile:
        return MetaContextProfile(
            context_count=0,
            total_words=0,
            total_tokens=0,
            top_tokens=[],
            intents=[],
            tones=[],
            dominant_pronouns=[],
            kinds=[],
            layers=[],
            updated_at=None,
        )

    def _bucket_to_values(
        self, bucket: Any, max_items: int | None = None
    ) -> list[CountValue]:
        if not isinstance(bucket, dict):
            return []
        total = sum(int(value) for value in bucket.values())
        ordered = sorted(bucket.items(), key=lambda item: (-int(item[1]), item[0]))
        if max_items is not None and max_items > 0:
            ordered = ordered[:max_items]
        output: list[CountValue] = []
        for value, count in ordered:
            ratio = float(count) / total if total > 0 else 0.0
            output.append(
                CountValue(value=str(value), count=int(count), ratio=round(ratio, 6))
            )
        return output

    def _bucket_to_intents(self, bucket: Any) -> list[IntentScore]:
        if not isinstance(bucket, dict):
            return []
        total = sum(int(value) for value in bucket.values())
        ordered = sorted(bucket.items(), key=lambda item: (-int(item[1]), item[0]))
        output: list[IntentScore] = []
        for label, count in ordered:
            score = float(count) / total if total > 0 else 0.0
            output.append(
                IntentScore(label=str(label), score=round(score, 6), count=int(count))
            )
        return output

    def _load_item_content(
        self, item: ContextItem, max_source_bytes: int
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

    def _resolve_kind(self, item: ContextItem, source_path: str | None) -> str:
        kind = (item.kind or "auto").strip().lower()
        if kind not in {"auto", "text", "code"}:
            return "text"
        if kind != "auto":
            return kind
        if source_path:
            ext = Path(source_path).suffix.lower()
            if ext in CODE_EXTS:
                return "code"
        return "text"

    def _analyze_item(
        self,
        item: ContextItem,
        content: str,
        source_path: str | None,
        preview_chars: int,
        max_tokens_per_context: int,
        max_keywords: int,
        min_token_length: int,
    ) -> tuple[ContextAnalysis, set[str], dict[str, int], _ContextSignals]:
        kind = self._resolve_kind(item, source_path)
        lines = content.splitlines()
        sentences = self._split_sentences(content)
        sentence_count = len(sentences)
        paragraph_count = self._count_paragraphs(lines)
        word_count = len(TOKEN_RE.findall(content))

        token_counts = self._extract_token_counts(content, min_token_length)
        token_count = sum(token_counts.values())
        keywords = self._select_keywords(token_counts, max_keywords)
        token_set = self._select_token_set(token_counts, max_tokens_per_context)

        intent_profile = self._build_intent_profile(content, sentences, lines)
        perspective_profile = self._build_perspective_profile(token_counts)
        evidence = self._build_evidence_signals(content, lines)
        rationale_sentences = self._extract_rationales(sentences, 3)
        inferred_purpose = self._infer_purpose(intent_profile.primary)
        preview = self._preview_text(content, preview_chars)

        tags = [tag for tag in (item.tags or []) if tag]
        source_value = item.source or source_path
        layer_value = item.layer.strip() if item.layer else None

        analysis = ContextAnalysis(
            index=0,
            id=item.id,
            name=item.name,
            kind=kind,
            layer=layer_value,
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
            inferred_purpose=inferred_purpose,
            rationale_sentences=rationale_sentences,
            evidence=evidence,
            cluster_id=None,
        )

        neg_ratio = 0.0
        if sentence_count > 0:
            neg_ratio = perspective_profile.negations / sentence_count

        signals = _ContextSignals(
            tone=perspective_profile.tone,
            dominant=perspective_profile.dominant,
            primary_intent=intent_profile.primary,
            negation_ratio=neg_ratio,
        )

        return analysis, token_set, token_counts, signals

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

    def _build_clusters(
        self,
        token_counts: list[dict[str, int]],
        cluster_similarity: float,
        max_clusters: int,
        similarity_mode: str,
        max_keywords: int,
    ) -> tuple[list[ContextCluster], list[int | None]]:
        clusters: list[_ClusterState] = []
        assignments: list[int | None] = []

        for idx, counts in enumerate(token_counts):
            tokens = set(counts.keys())
            best_cluster = None
            best_score = 0.0
            for cluster_idx, cluster in enumerate(clusters):
                cluster_tokens = set(cluster.token_counts.keys())
                if similarity_mode == "overlap":
                    overlap = len(tokens & cluster_tokens)
                    score = float(overlap)
                else:
                    union = len(tokens | cluster_tokens)
                    score = len(tokens & cluster_tokens) / union if union else 0.0

                if score > best_score:
                    best_score = score
                    best_cluster = cluster_idx

            assigned = None
            if best_cluster is not None and best_score >= cluster_similarity:
                assigned = best_cluster
            elif max_clusters <= 0:
                assigned = None
            elif len(clusters) < max_clusters:
                assigned = len(clusters)
                clusters.append(_ClusterState(token_counts={}, context_indices=[]))
            elif best_cluster is not None:
                assigned = best_cluster

            if assigned is not None:
                cluster = clusters[assigned]
                for token, count in counts.items():
                    cluster.token_counts[token] = cluster.token_counts.get(token, 0) + count
                cluster.context_indices.append(idx + 1)
                assignments.append(assigned + 1)
            else:
                assignments.append(None)

        output_clusters: list[ContextCluster] = []
        for idx, cluster in enumerate(clusters, start=1):
            keywords = self._select_keywords(cluster.token_counts, max_keywords)
            output_clusters.append(
                ContextCluster(
                    cluster_id=idx,
                    context_indices=cluster.context_indices,
                    keywords=keywords,
                    size=len(cluster.context_indices),
                )
            )

        return output_clusters, assignments

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
    ) -> tuple[list[ContextRelationship], bool]:
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

        relationships: list[ContextRelationship] = []
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
                ContextRelationship(
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
        self, left: _ContextSignals, right: _ContextSignals, meets_similarity: bool
    ) -> tuple[str, list[str]]:
        shifts: list[str] = []
        if left.tone != right.tone:
            if left.tone != "neutral" and right.tone != "neutral":
                shifts.append(f"tone:{left.tone}->{right.tone}")
        if left.dominant != right.dominant:
            if left.dominant != "unknown" and right.dominant != "unknown":
                shifts.append(f"stance:{left.dominant}->{right.dominant}")
        if left.primary_intent != right.primary_intent:
            shifts.append(f"intent:{left.primary_intent}->{right.primary_intent}")
        if abs(left.negation_ratio - right.negation_ratio) >= 0.3:
            shifts.append("negation-shift")

        if meets_similarity and shifts:
            return "contrast", shifts
        if meets_similarity:
            return "reinforcing", shifts
        return "adjacent", shifts

    def _build_drifts(
        self,
        contexts: list[ContextAnalysis],
        token_sets: list[set[str]],
        signals: list[_ContextSignals],
        similarity_mode: str,
        drift_similarity: float,
    ) -> list[ContextDrift]:
        drifts: list[ContextDrift] = []
        if len(contexts) < 2:
            return drifts

        for idx in range(1, len(contexts)):
            left_tokens = token_sets[idx - 1]
            right_tokens = token_sets[idx]
            overlap = len(left_tokens & right_tokens)
            if similarity_mode == "overlap":
                similarity = float(overlap)
            else:
                union = len(left_tokens | right_tokens)
                similarity = overlap / union if union else 0.0

            left_signal = signals[idx - 1]
            right_signal = signals[idx]
            intent_change = left_signal.primary_intent != right_signal.primary_intent
            tone_change = left_signal.tone != right_signal.tone
            perspective_change = left_signal.dominant != right_signal.dominant

            notes: list[str] = []
            if intent_change:
                notes.append(
                    f"intent:{left_signal.primary_intent}->{right_signal.primary_intent}"
                )
            if tone_change:
                notes.append(f"tone:{left_signal.tone}->{right_signal.tone}")
            if perspective_change:
                notes.append(
                    f"stance:{left_signal.dominant}->{right_signal.dominant}"
                )

            flagged = similarity < drift_similarity or intent_change or tone_change
            drifts.append(
                ContextDrift(
                    from_index=contexts[idx - 1].index,
                    to_index=contexts[idx].index,
                    similarity=round(similarity, 6),
                    overlap=overlap,
                    intent_change=intent_change,
                    tone_change=tone_change,
                    perspective_change=perspective_change,
                    flagged=flagged,
                    notes=notes,
                )
            )

        return drifts

    def _build_meta_profile(
        self,
        contexts: list[ContextAnalysis],
        token_counts: list[dict[str, int]],
        max_profile_tokens: int,
    ) -> MetaContextProfile:
        bucket_tokens: dict[str, int] = {}
        intent_counts: dict[str, int] = {}
        tone_counts: dict[str, int] = {}
        pronoun_counts: dict[str, int] = {}
        kind_counts: dict[str, int] = {}
        layer_counts: dict[str, int] = {}

        total_words = 0
        total_tokens = 0
        for idx, context in enumerate(contexts):
            total_words += context.word_count
            total_tokens += context.token_count
            intent_counts[context.intent_profile.primary] = (
                intent_counts.get(context.intent_profile.primary, 0) + 1
            )
            tone_counts[context.perspective_profile.tone] = (
                tone_counts.get(context.perspective_profile.tone, 0) + 1
            )
            pronoun_counts[context.perspective_profile.dominant] = (
                pronoun_counts.get(context.perspective_profile.dominant, 0) + 1
            )
            kind_counts[context.kind] = kind_counts.get(context.kind, 0) + 1
            layer_key = context.layer if context.layer else "unassigned"
            layer_counts[layer_key] = layer_counts.get(layer_key, 0) + 1

            for token, count in token_counts[idx].items():
                bucket_tokens[token] = bucket_tokens.get(token, 0) + count

        bucket_tokens = self._prune_bucket(bucket_tokens, max_profile_tokens)
        return MetaContextProfile(
            context_count=len(contexts),
            total_words=total_words,
            total_tokens=total_tokens,
            top_tokens=self._bucket_to_values(bucket_tokens, max_profile_tokens),
            intents=self._bucket_to_intents(intent_counts),
            tones=self._bucket_to_values(tone_counts),
            dominant_pronouns=self._bucket_to_values(pronoun_counts),
            kinds=self._bucket_to_values(kind_counts),
            layers=self._bucket_to_values(layer_counts),
            updated_at=None,
        )

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextMetaContextArgs):
            return ToolCallDisplay(summary="context_meta_context")

        summary = "context_meta_context"
        return ToolCallDisplay(
            summary=summary,
            details={
                "action": event.args.action,
                "item_count": len(event.args.items or []),
                "min_similarity": event.args.min_similarity,
                "cluster_similarity": event.args.cluster_similarity,
                "max_relationships": event.args.max_relationships,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextMetaContextResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = (
            f"Processed {event.result.context_count} context(s) with "
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
                "context_count": event.result.context_count,
                "relationship_count": event.result.relationship_count,
                "drift_count": event.result.drift_count,
                "cluster_count": event.result.cluster_count,
                "updated": event.result.updated,
                "truncated": event.result.truncated,
                "errors": event.result.errors,
                "contexts": event.result.contexts,
                "relationships": event.result.relationships,
                "drifts": event.result.drifts,
                "clusters": event.result.clusters,
                "meta_profile": event.result.meta_profile,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Analyzing meta-context"
