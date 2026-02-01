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


DEFAULT_PROFILE_PATH = Path.home() / ".vibe" / "memory" / "meta_hierarchy_profile.json"

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
class _HierarchyCandidate:
    parent_index: int
    score: float
    overlap: int
    shared_tokens: list[str]
    reason: str


class ContextMetaHierarchyConfig(BaseToolConfig):
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
        default=8, description="Maximum shared tokens returned per edge."
    )
    similarity_mode: str = Field(
        default="jaccard", description="jaccard or overlap."
    )
    min_parent_similarity: float = Field(
        default=0.2, description="Minimum similarity or overlap to attach a parent."
    )
    use_parent_ids: bool = Field(
        default=True, description="Prefer explicit parent_id links."
    )
    use_layer_path: bool = Field(
        default=True, description="Prefer layer_path prefix links."
    )
    use_depth_hint: bool = Field(
        default=True, description="Prefer depth-based parent hints."
    )
    layer_path_delimiters: list[str] = Field(
        default=["/", ">", "::"], description="Delimiters for layer_path."
    )
    max_profile_tokens: int = Field(
        default=200, description="Maximum tokens stored in the meta profile."
    )
    profile_path: Path = Field(
        default=DEFAULT_PROFILE_PATH,
        description="Path to the persistent meta-hierarchy profile JSON.",
    )


class ContextMetaHierarchyState(BaseToolState):
    pass


class HierarchyItem(BaseModel):
    id: str | None = Field(default=None, description="Optional context id.")
    name: str | None = Field(default=None, description="Optional context name.")
    content: str | None = Field(default=None, description="Inline content.")
    path: str | None = Field(default=None, description="Path to a context file.")
    kind: str | None = Field(default=None, description="auto, text, or code.")
    layer: str | None = Field(default=None, description="Optional layer label.")
    layer_path: str | None = Field(
        default=None, description="Hierarchical layer path (e.g. a/b/c)."
    )
    parent_id: str | None = Field(default=None, description="Explicit parent id.")
    depth: int | None = Field(default=None, description="Depth hint for hierarchy.")
    source: str | None = Field(default=None, description="Source description.")
    audience: str | None = Field(default=None, description="Intended audience.")
    context: str | None = Field(default=None, description="Context/purpose notes.")
    intent_hint: str | None = Field(default=None, description="Optional intent hint.")
    perspective_hint: str | None = Field(
        default=None, description="Optional perspective hint."
    )
    origin: str | None = Field(default=None, description="Origin metadata.")
    tags: list[str] | None = Field(default=None, description="Optional tags.")


class ContextMetaHierarchyArgs(BaseModel):
    action: str | None = Field(
        default="analyze", description="analyze, update, profile, or clear."
    )
    items: list[HierarchyItem] | None = Field(
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
    similarity_mode: str | None = Field(
        default=None, description="Override similarity mode."
    )
    min_parent_similarity: float | None = Field(
        default=None, description="Override min parent similarity."
    )
    use_parent_ids: bool | None = Field(
        default=None, description="Override use_parent_ids."
    )
    use_layer_path: bool | None = Field(
        default=None, description="Override use_layer_path."
    )
    use_depth_hint: bool | None = Field(
        default=None, description="Override use_depth_hint."
    )
    layer_path_delimiters: list[str] | None = Field(
        default=None, description="Override layer path delimiters."
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


class CountValue(BaseModel):
    value: str
    count: int
    ratio: float


class HierarchyNode(BaseModel):
    index: int
    id: str | None
    name: str | None
    kind: str
    layer: str | None
    layer_path: str | None
    parent_index: int | None
    depth: int
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


class HierarchyEdge(BaseModel):
    parent_index: int
    child_index: int
    score: float
    overlap: int
    relationship: str
    reason: str
    shared_tokens: list[str]
    perspective_shift: list[str]


class HierarchyLevelSummary(BaseModel):
    depth: int
    node_count: int
    keywords: list[str]
    intents: list[IntentScore]
    tones: list[CountValue]


class HierarchyStats(BaseModel):
    node_count: int
    root_count: int
    leaf_count: int
    max_depth: int
    avg_depth: float
    avg_branching: float


class MetaHierarchyProfile(BaseModel):
    node_count: int
    total_words: int
    total_tokens: int
    top_tokens: list[CountValue]
    intents: list[IntentScore]
    tones: list[CountValue]
    dominant_pronouns: list[CountValue]
    kinds: list[CountValue]
    layers: list[CountValue]
    depths: list[CountValue]
    stats: HierarchyStats
    updated_at: str | None


class ContextMetaHierarchyResult(BaseModel):
    nodes: list[HierarchyNode]
    edges: list[HierarchyEdge]
    roots: list[int]
    levels: list[HierarchyLevelSummary]
    stats: HierarchyStats
    meta_profile: MetaHierarchyProfile | None
    node_count: int
    edge_count: int
    level_count: int
    updated: bool
    truncated: bool
    errors: list[str]

class ContextMetaHierarchy(
    BaseTool[
        ContextMetaHierarchyArgs,
        ContextMetaHierarchyResult,
        ContextMetaHierarchyConfig,
        ContextMetaHierarchyState,
    ],
    ToolUIData[ContextMetaHierarchyArgs, ContextMetaHierarchyResult],
):
    description: ClassVar[str] = (
        "Meta-hierarchical learning across contexts with parent-child structure."
    )

    async def run(self, args: ContextMetaHierarchyArgs) -> ContextMetaHierarchyResult:
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
            stats = profile.stats if profile else self._empty_stats()
            return ContextMetaHierarchyResult(
                nodes=[],
                edges=[],
                roots=[],
                levels=[],
                stats=stats,
                meta_profile=profile,
                node_count=0,
                edge_count=0,
                level_count=0,
                updated=False,
                truncated=False,
                errors=[],
            )

        if action == "clear":
            removed = self._clear_profile(profile_path)
            return ContextMetaHierarchyResult(
                nodes=[],
                edges=[],
                roots=[],
                levels=[],
                stats=self._empty_stats(),
                meta_profile=None,
                node_count=0,
                edge_count=0,
                level_count=0,
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
        similarity_mode = (
            args.similarity_mode
            if args.similarity_mode is not None
            else self.config.similarity_mode
        ).strip().lower()
        min_parent_similarity = (
            args.min_parent_similarity
            if args.min_parent_similarity is not None
            else self.config.min_parent_similarity
        )
        use_parent_ids = (
            args.use_parent_ids
            if args.use_parent_ids is not None
            else self.config.use_parent_ids
        )
        use_layer_path = (
            args.use_layer_path
            if args.use_layer_path is not None
            else self.config.use_layer_path
        )
        use_depth_hint = (
            args.use_depth_hint
            if args.use_depth_hint is not None
            else self.config.use_depth_hint
        )
        layer_delimiters = (
            args.layer_path_delimiters
            if args.layer_path_delimiters is not None
            else self.config.layer_path_delimiters
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
        if similarity_mode not in {"jaccard", "overlap"}:
            raise ToolError("similarity_mode must be jaccard or overlap.")
        if min_parent_similarity < 0:
            raise ToolError("min_parent_similarity must be >= 0.")

        nodes: list[HierarchyNode] = []
        token_sets: list[set[str]] = []
        token_counts: list[dict[str, int]] = []
        signals: list[_ContextSignals] = []
        layer_paths: list[list[str]] = []
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
                analysis.index = len(nodes) + 1
                nodes.append(analysis)
                token_sets.append(tokens)
                token_counts.append(token_count_map)
                signals.append(signal)
                layer_paths.append(self._split_layer_path(item, layer_delimiters))
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not nodes:
            raise ToolError("No valid items to process.")

        edges, parent_map = self._build_edges(
            items,
            nodes,
            token_sets,
            signals,
            layer_paths,
            use_parent_ids=use_parent_ids,
            use_layer_path=use_layer_path,
            use_depth_hint=use_depth_hint,
            min_parent_similarity=min_parent_similarity,
            similarity_mode=similarity_mode,
            max_shared_tokens=max_shared_tokens,
        )

        roots = self._assign_depths(nodes, parent_map)
        stats = self._build_stats(nodes, edges, roots)
        levels = self._build_level_summaries(
            nodes, token_counts, max_keywords
        )

        meta_profile = self._build_meta_profile(
            nodes, token_counts, stats, max_profile_tokens
        )
        updated = False
        if action == "update":
            meta_profile = self._update_profile(
                profile_path,
                meta_profile,
                token_counts,
                nodes,
                stats,
                max_profile_tokens,
            )
            updated = True

        return ContextMetaHierarchyResult(
            nodes=nodes,
            edges=edges,
            roots=roots,
            levels=levels,
            stats=stats,
            meta_profile=meta_profile,
            node_count=len(nodes),
            edge_count=len(edges),
            level_count=len(levels),
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

    def _load_profile(self, path: Path, max_profile_tokens: int) -> MetaHierarchyProfile:
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
        run_profile: MetaHierarchyProfile,
        token_counts: list[dict[str, int]],
        nodes: list[HierarchyNode],
        stats: HierarchyStats,
        max_profile_tokens: int,
    ) -> MetaHierarchyProfile:
        store = {
            "version": 1,
            "node_count": 0,
            "total_words": 0,
            "total_tokens": 0,
            "token_counts": {},
            "intent_counts": {},
            "tone_counts": {},
            "pronoun_counts": {},
            "kind_counts": {},
            "layer_counts": {},
            "depth_counts": {},
            "root_count": 0,
            "leaf_count": 0,
            "branch_total": 0,
            "branch_parent_count": 0,
            "max_depth": 0,
            "updated_at": None,
        }
        if path.exists():
            try:
                loaded = json.loads(path.read_text("utf-8"))
                if isinstance(loaded, dict):
                    store.update(loaded)
            except Exception:
                pass

        store["node_count"] = int(store.get("node_count", 0)) + run_profile.node_count
        store["total_words"] = int(store.get("total_words", 0)) + run_profile.total_words
        store["total_tokens"] = int(store.get("total_tokens", 0)) + run_profile.total_tokens
        store["root_count"] = int(store.get("root_count", 0)) + stats.root_count
        store["leaf_count"] = int(store.get("leaf_count", 0)) + stats.leaf_count
        child_total = max(stats.node_count - stats.root_count, 0)
        parent_total = max(stats.node_count - stats.leaf_count, 0)
        store["branch_total"] = int(store.get("branch_total", 0)) + child_total
        store["branch_parent_count"] = int(store.get("branch_parent_count", 0)) + parent_total
        store["max_depth"] = max(int(store.get("max_depth", 0)), stats.max_depth)

        token_bucket = store.get("token_counts")
        if not isinstance(token_bucket, dict):
            token_bucket = {}
            store["token_counts"] = token_bucket

        for counts in token_counts:
            for token, count in counts.items():
                token_bucket[token] = int(token_bucket.get(token, 0)) + int(count)

        self._merge_intent_counts(store, "intent_counts", nodes)
        self._merge_value_counts(store, "tone_counts", [
            node.perspective_profile.tone for node in nodes
        ])
        self._merge_value_counts(store, "pronoun_counts", [
            node.perspective_profile.dominant for node in nodes
        ])
        self._merge_value_counts(store, "kind_counts", [node.kind for node in nodes])
        self._merge_value_counts(store, "layer_counts", [
            node.layer if node.layer else "unassigned" for node in nodes
        ])
        self._merge_depth_counts(store, nodes)

        store["updated_at"] = datetime.utcnow().isoformat()
        token_bucket = self._prune_bucket(token_bucket, max_profile_tokens)
        store["token_counts"] = token_bucket

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(store, indent=2), "utf-8")
        return self._profile_from_store(store, max_profile_tokens)

    def _merge_intent_counts(
        self, store: dict[str, Any], key: str, nodes: list[HierarchyNode]
    ) -> None:
        bucket = store.get(key)
        if not isinstance(bucket, dict):
            bucket = {}
            store[key] = bucket
        for node in nodes:
            label = node.intent_profile.primary
            bucket[label] = int(bucket.get(label, 0)) + 1

    def _merge_value_counts(
        self, store: dict[str, Any], key: str, values: list[str]
    ) -> None:
        bucket = store.get(key)
        if not isinstance(bucket, dict):
            bucket = {}
            store[key] = bucket
        for value in values:
            bucket[value] = int(bucket.get(value, 0)) + 1

    def _merge_depth_counts(self, store: dict[str, Any], nodes: list[HierarchyNode]) -> None:
        bucket = store.get("depth_counts")
        if not isinstance(bucket, dict):
            bucket = {}
            store["depth_counts"] = bucket
        for node in nodes:
            key = str(node.depth)
            bucket[key] = int(bucket.get(key, 0)) + 1

    def _prune_bucket(self, bucket: dict[str, int], max_items: int) -> dict[str, int]:
        if max_items <= 0:
            return bucket
        ordered = sorted(bucket.items(), key=lambda item: (-item[1], item[0]))
        pruned = ordered[:max_items]
        return {key: int(value) for key, value in pruned}

    def _profile_from_store(
        self, store: dict[str, Any], max_profile_tokens: int
    ) -> MetaHierarchyProfile:
        node_count = int(store.get("node_count", 0))
        total_words = int(store.get("total_words", 0))
        total_tokens = int(store.get("total_tokens", 0))
        token_bucket = store.get("token_counts")
        if not isinstance(token_bucket, dict):
            token_bucket = {}

        depth_bucket = store.get("depth_counts")
        if not isinstance(depth_bucket, dict):
            depth_bucket = {}

        top_tokens = self._bucket_to_values(token_bucket, max_profile_tokens)
        intents = self._bucket_to_intents(store.get("intent_counts"))
        tones = self._bucket_to_values(store.get("tone_counts"))
        pronouns = self._bucket_to_values(store.get("pronoun_counts"))
        kinds = self._bucket_to_values(store.get("kind_counts"))
        layers = self._bucket_to_values(store.get("layer_counts"))
        depths = self._bucket_to_values(depth_bucket)
        updated_at = store.get("updated_at")
        updated_at_value = updated_at if isinstance(updated_at, str) else None

        avg_depth = 0.0
        if node_count > 0:
            total_depth = sum(int(depth) * int(count) for depth, count in depth_bucket.items())
            avg_depth = total_depth / node_count

        branch_total = int(store.get("branch_total", 0))
        branch_parent_count = int(store.get("branch_parent_count", 0))
        avg_branching = branch_total / branch_parent_count if branch_parent_count > 0 else 0.0

        stats = HierarchyStats(
            node_count=node_count,
            root_count=int(store.get("root_count", 0)),
            leaf_count=int(store.get("leaf_count", 0)),
            max_depth=int(store.get("max_depth", 0)),
            avg_depth=round(avg_depth, 6),
            avg_branching=round(avg_branching, 6),
        )

        return MetaHierarchyProfile(
            node_count=node_count,
            total_words=total_words,
            total_tokens=total_tokens,
            top_tokens=top_tokens,
            intents=intents,
            tones=tones,
            dominant_pronouns=pronouns,
            kinds=kinds,
            layers=layers,
            depths=depths,
            stats=stats,
            updated_at=updated_at_value,
        )

    def _empty_profile(self) -> MetaHierarchyProfile:
        return MetaHierarchyProfile(
            node_count=0,
            total_words=0,
            total_tokens=0,
            top_tokens=[],
            intents=[],
            tones=[],
            dominant_pronouns=[],
            kinds=[],
            layers=[],
            depths=[],
            stats=self._empty_stats(),
            updated_at=None,
        )

    def _empty_stats(self) -> HierarchyStats:
        return HierarchyStats(
            node_count=0,
            root_count=0,
            leaf_count=0,
            max_depth=0,
            avg_depth=0.0,
            avg_branching=0.0,
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
        self, item: HierarchyItem, max_source_bytes: int
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

    def _resolve_kind(self, item: HierarchyItem, source_path: str | None) -> str:
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
        item: HierarchyItem,
        content: str,
        source_path: str | None,
        preview_chars: int,
        max_tokens_per_context: int,
        max_keywords: int,
        min_token_length: int,
    ) -> tuple[HierarchyNode, set[str], dict[str, int], _ContextSignals]:
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
        layer_path_value = item.layer_path.strip() if item.layer_path else None

        analysis = HierarchyNode(
            index=0,
            id=item.id,
            name=item.name,
            kind=kind,
            layer=layer_value,
            layer_path=layer_path_value,
            parent_index=None,
            depth=item.depth or 1,
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

    def _split_layer_path(
        self, item: HierarchyItem, delimiters: list[str]
    ) -> list[str]:
        raw = item.layer_path or item.layer or ""
        value = raw.strip()
        if not value:
            return []
        for delimiter in delimiters:
            if delimiter and delimiter in value:
                parts = [part.strip() for part in value.split(delimiter) if part.strip()]
                if parts:
                    return parts
        return [value]

    def _build_edges(
        self,
        items: list[HierarchyItem],
        nodes: list[HierarchyNode],
        token_sets: list[set[str]],
        signals: list[_ContextSignals],
        layer_paths: list[list[str]],
        *,
        use_parent_ids: bool,
        use_layer_path: bool,
        use_depth_hint: bool,
        min_parent_similarity: float,
        similarity_mode: str,
        max_shared_tokens: int,
    ) -> tuple[list[HierarchyEdge], dict[int, int]]:
        id_map: dict[str, int] = {}
        for node in nodes:
            if node.id:
                id_map[node.id] = node.index

        edges: list[HierarchyEdge] = []
        parent_map: dict[int, int] = {}

        for idx, item in enumerate(items):
            child_index = idx + 1
            candidates: list[_HierarchyCandidate] = []

            if use_parent_ids and item.parent_id:
                parent_index = id_map.get(item.parent_id)
                if parent_index and parent_index != child_index:
                    candidate = self._candidate_for_pair(
                        parent_index,
                        child_index,
                        token_sets,
                        signals,
                        similarity_mode,
                        max_shared_tokens,
                        reason="parent_id",
                    )
                    candidates.append(candidate)

            if use_layer_path and not candidates:
                candidate = self._layer_path_candidate(
                    idx,
                    layer_paths,
                    token_sets,
                    signals,
                    similarity_mode,
                    max_shared_tokens,
                )
                if candidate:
                    candidates.append(candidate)

            if use_depth_hint and not candidates:
                candidate = self._depth_candidate(
                    idx,
                    items,
                    token_sets,
                    signals,
                    similarity_mode,
                    max_shared_tokens,
                )
                if candidate:
                    candidates.append(candidate)

            if not candidates:
                candidate = self._similarity_candidate(
                    idx,
                    token_sets,
                    signals,
                    similarity_mode,
                    max_shared_tokens,
                    min_parent_similarity,
                )
                if candidate:
                    candidates.append(candidate)

            if not candidates:
                continue

            best = max(candidates, key=lambda item: (item.score, item.overlap))
            if similarity_mode == "overlap":
                if best.score < min_parent_similarity and best.reason == "similarity":
                    continue
            else:
                if best.score < min_parent_similarity and best.reason == "similarity":
                    continue

            parent_map[child_index] = best.parent_index
            relationship, shifts = self._classify_relationship(
                signals[best.parent_index - 1],
                signals[child_index - 1],
                True,
            )
            edges.append(
                HierarchyEdge(
                    parent_index=best.parent_index,
                    child_index=child_index,
                    score=round(best.score, 6),
                    overlap=best.overlap,
                    relationship=relationship,
                    reason=best.reason,
                    shared_tokens=best.shared_tokens,
                    perspective_shift=shifts,
                )
            )

        return edges, parent_map

    def _candidate_for_pair(
        self,
        parent_index: int,
        child_index: int,
        token_sets: list[set[str]],
        signals: list[_ContextSignals],
        similarity_mode: str,
        max_shared_tokens: int,
        *,
        reason: str,
    ) -> _HierarchyCandidate:
        score, overlap, shared = self._similarity(
            token_sets[parent_index - 1],
            token_sets[child_index - 1],
            similarity_mode,
            max_shared_tokens,
        )
        return _HierarchyCandidate(
            parent_index=parent_index,
            score=score,
            overlap=overlap,
            shared_tokens=shared,
            reason=reason,
        )

    def _layer_path_candidate(
        self,
        idx: int,
        layer_paths: list[list[str]],
        token_sets: list[set[str]],
        signals: list[_ContextSignals],
        similarity_mode: str,
        max_shared_tokens: int,
    ) -> _HierarchyCandidate | None:
        target_path = layer_paths[idx]
        if not target_path:
            return None

        best: _HierarchyCandidate | None = None
        best_prefix = -1
        for parent_idx in range(idx):
            candidate_path = layer_paths[parent_idx]
            if not candidate_path:
                continue
            if len(candidate_path) >= len(target_path):
                continue
            if target_path[: len(candidate_path)] != candidate_path:
                continue
            prefix_len = len(candidate_path)
            candidate = self._candidate_for_pair(
                parent_idx + 1,
                idx + 1,
                token_sets,
                signals,
                similarity_mode,
                max_shared_tokens,
                reason="layer_path",
            )
            if prefix_len > best_prefix or (
                prefix_len == best_prefix and candidate.score > (best.score if best else 0.0)
            ):
                best_prefix = prefix_len
                best = candidate

        return best

    def _depth_candidate(
        self,
        idx: int,
        items: list[HierarchyItem],
        token_sets: list[set[str]],
        signals: list[_ContextSignals],
        similarity_mode: str,
        max_shared_tokens: int,
    ) -> _HierarchyCandidate | None:
        depth_hint = items[idx].depth
        if depth_hint is None or depth_hint <= 1:
            return None

        best: _HierarchyCandidate | None = None
        for parent_idx in range(idx):
            parent_depth = items[parent_idx].depth
            if parent_depth is None:
                continue
            if parent_depth != depth_hint - 1:
                continue
            candidate = self._candidate_for_pair(
                parent_idx + 1,
                idx + 1,
                token_sets,
                signals,
                similarity_mode,
                max_shared_tokens,
                reason="depth_hint",
            )
            if best is None or candidate.score > best.score:
                best = candidate
        return best

    def _similarity_candidate(
        self,
        idx: int,
        token_sets: list[set[str]],
        signals: list[_ContextSignals],
        similarity_mode: str,
        max_shared_tokens: int,
        min_parent_similarity: float,
    ) -> _HierarchyCandidate | None:
        best: _HierarchyCandidate | None = None
        for parent_idx in range(idx):
            candidate = self._candidate_for_pair(
                parent_idx + 1,
                idx + 1,
                token_sets,
                signals,
                similarity_mode,
                max_shared_tokens,
                reason="similarity",
            )
            if best is None or candidate.score > best.score:
                best = candidate

        if best is None:
            return None
        if best.score < min_parent_similarity:
            return None
        return best

    def _similarity(
        self,
        left_tokens: set[str],
        right_tokens: set[str],
        similarity_mode: str,
        max_shared_tokens: int,
    ) -> tuple[float, int, list[str]]:
        shared = left_tokens & right_tokens
        overlap = len(shared)
        if similarity_mode == "overlap":
            score = float(overlap)
        else:
            union = len(left_tokens | right_tokens)
            score = overlap / union if union else 0.0
        shared_list = sorted(shared)
        if max_shared_tokens > 0 and len(shared_list) > max_shared_tokens:
            shared_list = shared_list[:max_shared_tokens]
        return score, overlap, shared_list

    def _assign_depths(
        self, nodes: list[HierarchyNode], parent_map: dict[int, int]
    ) -> list[int]:
        memo: dict[int, int] = {}
        visiting: set[int] = set()

        def depth_for(index: int) -> int:
            if index in memo:
                return memo[index]
            if index in visiting:
                parent_map.pop(index, None)
                memo[index] = 1
                return 1
            visiting.add(index)
            parent = parent_map.get(index)
            if parent is None:
                value = nodes[index - 1].depth if nodes[index - 1].depth > 0 else 1
            else:
                value = depth_for(parent) + 1
            visiting.remove(index)
            memo[index] = value
            return value

        roots: list[int] = []
        for node in nodes:
            node.parent_index = parent_map.get(node.index)
            node.depth = depth_for(node.index)

        for node in nodes:
            if node.parent_index is None:
                roots.append(node.index)

        return roots

    def _build_stats(
        self, nodes: list[HierarchyNode], edges: list[HierarchyEdge], roots: list[int]
    ) -> HierarchyStats:
        depth_values = [node.depth for node in nodes] or [0]
        max_depth = max(depth_values)
        avg_depth = sum(depth_values) / len(depth_values) if nodes else 0.0

        child_counts: dict[int, int] = {}
        for edge in edges:
            child_counts[edge.parent_index] = child_counts.get(edge.parent_index, 0) + 1

        parent_count = len(child_counts)
        total_children = sum(child_counts.values())
        avg_branching = total_children / parent_count if parent_count > 0 else 0.0

        leaf_count = 0
        for node in nodes:
            if node.index not in child_counts:
                leaf_count += 1

        return HierarchyStats(
            node_count=len(nodes),
            root_count=len(roots),
            leaf_count=leaf_count,
            max_depth=max_depth,
            avg_depth=round(avg_depth, 6),
            avg_branching=round(avg_branching, 6),
        )

    def _build_level_summaries(
        self,
        nodes: list[HierarchyNode],
        token_counts: list[dict[str, int]],
        max_keywords: int,
    ) -> list[HierarchyLevelSummary]:
        levels: dict[int, dict[str, Any]] = {}
        for idx, node in enumerate(nodes):
            level = node.depth
            bucket = levels.setdefault(
                level,
                {
                    "count": 0,
                    "tokens": {},
                    "intents": {},
                    "tones": {},
                },
            )
            bucket["count"] += 1
            for token, count in token_counts[idx].items():
                bucket["tokens"][token] = bucket["tokens"].get(token, 0) + count
            intent = node.intent_profile.primary
            bucket["intents"][intent] = bucket["intents"].get(intent, 0) + 1
            tone = node.perspective_profile.tone
            bucket["tones"][tone] = bucket["tones"].get(tone, 0) + 1

        summaries: list[HierarchyLevelSummary] = []
        for level, data in sorted(levels.items()):
            keywords = self._select_keywords(data["tokens"], max_keywords)
            intents = self._bucket_to_intents(data["intents"])
            tones = self._bucket_to_values(data["tones"])
            summaries.append(
                HierarchyLevelSummary(
                    depth=level,
                    node_count=data["count"],
                    keywords=keywords,
                    intents=intents,
                    tones=tones,
                )
            )

        return summaries

    def _build_meta_profile(
        self,
        nodes: list[HierarchyNode],
        token_counts: list[dict[str, int]],
        stats: HierarchyStats,
        max_profile_tokens: int,
    ) -> MetaHierarchyProfile:
        bucket_tokens: dict[str, int] = {}
        intent_counts: dict[str, int] = {}
        tone_counts: dict[str, int] = {}
        pronoun_counts: dict[str, int] = {}
        kind_counts: dict[str, int] = {}
        layer_counts: dict[str, int] = {}
        depth_counts: dict[str, int] = {}

        total_words = 0
        total_tokens = 0
        for idx, node in enumerate(nodes):
            total_words += node.word_count
            total_tokens += node.token_count
            intent_counts[node.intent_profile.primary] = (
                intent_counts.get(node.intent_profile.primary, 0) + 1
            )
            tone_counts[node.perspective_profile.tone] = (
                tone_counts.get(node.perspective_profile.tone, 0) + 1
            )
            pronoun_counts[node.perspective_profile.dominant] = (
                pronoun_counts.get(node.perspective_profile.dominant, 0) + 1
            )
            kind_counts[node.kind] = kind_counts.get(node.kind, 0) + 1
            layer_key = node.layer if node.layer else "unassigned"
            layer_counts[layer_key] = layer_counts.get(layer_key, 0) + 1
            depth_counts[str(node.depth)] = depth_counts.get(str(node.depth), 0) + 1

            for token, count in token_counts[idx].items():
                bucket_tokens[token] = bucket_tokens.get(token, 0) + count

        bucket_tokens = self._prune_bucket(bucket_tokens, max_profile_tokens)
        return MetaHierarchyProfile(
            node_count=len(nodes),
            total_words=total_words,
            total_tokens=total_tokens,
            top_tokens=self._bucket_to_values(bucket_tokens, max_profile_tokens),
            intents=self._bucket_to_intents(intent_counts),
            tones=self._bucket_to_values(tone_counts),
            dominant_pronouns=self._bucket_to_values(pronoun_counts),
            kinds=self._bucket_to_values(kind_counts),
            layers=self._bucket_to_values(layer_counts),
            depths=self._bucket_to_values(depth_counts),
            stats=stats,
            updated_at=None,
        )

    def _classify_relationship(
        self, parent: _ContextSignals, child: _ContextSignals, meets_similarity: bool
    ) -> tuple[str, list[str]]:
        shifts: list[str] = []
        if parent.tone != child.tone:
            if parent.tone != "neutral" and child.tone != "neutral":
                shifts.append(f"tone:{parent.tone}->{child.tone}")
        if parent.dominant != child.dominant:
            if parent.dominant != "unknown" and child.dominant != "unknown":
                shifts.append(f"stance:{parent.dominant}->{child.dominant}")
        if parent.primary_intent != child.primary_intent:
            shifts.append(f"intent:{parent.primary_intent}->{child.primary_intent}")
        if abs(parent.negation_ratio - child.negation_ratio) >= 0.3:
            shifts.append("negation-shift")

        if meets_similarity and shifts:
            return "contrast", shifts
        if meets_similarity:
            return "reinforcing", shifts
        return "adjacent", shifts

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextMetaHierarchyArgs):
            return ToolCallDisplay(summary="context_meta_hierarchy")

        summary = "context_meta_hierarchy"
        return ToolCallDisplay(
            summary=summary,
            details={
                "action": event.args.action,
                "item_count": len(event.args.items or []),
                "min_parent_similarity": event.args.min_parent_similarity,
                "similarity_mode": event.args.similarity_mode,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextMetaHierarchyResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = (
            f"Processed {event.result.node_count} node(s) with "
            f"{event.result.edge_count} edge(s)"
        )
        warnings = event.result.errors[:]
        if event.result.truncated:
            warnings.append("Output truncated by size or limits")

        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=warnings,
            details={
                "node_count": event.result.node_count,
                "edge_count": event.result.edge_count,
                "level_count": event.result.level_count,
                "updated": event.result.updated,
                "truncated": event.result.truncated,
                "errors": event.result.errors,
                "stats": event.result.stats,
                "nodes": event.result.nodes,
                "edges": event.result.edges,
                "roots": event.result.roots,
                "levels": event.result.levels,
                "meta_profile": event.result.meta_profile,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Analyzing meta-hierarchy"
