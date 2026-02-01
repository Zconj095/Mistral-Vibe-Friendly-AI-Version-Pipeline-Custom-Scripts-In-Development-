
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
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
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

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

CLAIM_RE = re.compile(
    r"\b("
    r"shows|demonstrates|indicates|suggests|causes|leads to|results in|"
    r"correlates|increases|decreases|improves|reduces|finds|concludes|"
    r"supports|evidence"
    r")\b",
    re.IGNORECASE,
)
NEGATION_RE = re.compile(r"\b(no|not|never|cannot|fails to|without)\b", re.IGNORECASE)

TIME_UNIT_NS = {
    "ns": 1,
    "nanosecond": 1,
    "nanoseconds": 1,
    "us": 1_000,
    "microsecond": 1_000,
    "microseconds": 1_000,
    "ms": 1_000_000,
    "millisecond": 1_000_000,
    "milliseconds": 1_000_000,
    "s": 1_000_000_000,
    "sec": 1_000_000_000,
    "secs": 1_000_000_000,
    "second": 1_000_000_000,
    "seconds": 1_000_000_000,
    "m": 60 * 1_000_000_000,
    "min": 60 * 1_000_000_000,
    "mins": 60 * 1_000_000_000,
    "minute": 60 * 1_000_000_000,
    "minutes": 60 * 1_000_000_000,
    "h": 3600 * 1_000_000_000,
    "hr": 3600 * 1_000_000_000,
    "hour": 3600 * 1_000_000_000,
    "hours": 3600 * 1_000_000_000,
    "d": 86400 * 1_000_000_000,
    "day": 86400 * 1_000_000_000,
    "days": 86400 * 1_000_000_000,
    "w": 7 * 86400 * 1_000_000_000,
    "week": 7 * 86400 * 1_000_000_000,
    "weeks": 7 * 86400 * 1_000_000_000,
    "mo": 30 * 86400 * 1_000_000_000,
    "month": 30 * 86400 * 1_000_000_000,
    "months": 30 * 86400 * 1_000_000_000,
    "y": 365 * 86400 * 1_000_000_000,
    "yr": 365 * 86400 * 1_000_000_000,
    "year": 365 * 86400 * 1_000_000_000,
    "years": 365 * 86400 * 1_000_000_000,
    "decade": 10 * 365 * 86400 * 1_000_000_000,
    "decades": 10 * 365 * 86400 * 1_000_000_000,
}


@dataclass
class _ResearchRecord:
    item_id: str
    timestamp_ns: int | None
    source: str | None
    preview: str
    token_counts: dict[str, int]
    tokens: set[str]
    sentences: list[str]


@dataclass
class _ClaimRecord:
    claim: str
    item_id: str
    source: str | None
    tokens: set[str]
    has_negation: bool


class ContextDeepResearchConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    profile_path: Path = Field(
        default=Path.home() / ".vibe" / "memory" / "deep_research_profile.json",
        description="Path to the deep research profile file.",
    )
    max_items: int = Field(default=200, description="Maximum items to process.")
    max_profile_entries: int = Field(
        default=5000, description="Maximum entries stored in the profile."
    )
    max_source_bytes: int = Field(
        default=5_000_000, description="Maximum size per item (bytes)."
    )
    max_total_bytes: int = Field(
        default=20_000_000, description="Maximum total bytes across items."
    )
    preview_chars: int = Field(default=240, description="Preview length per item.")
    min_word_length: int = Field(default=3, description="Minimum token length.")
    max_keywords: int = Field(default=20, description="Maximum keywords per summary.")
    max_claims: int = Field(default=50, description="Maximum claims to return.")
    max_questions: int = Field(default=20, description="Maximum questions to return.")
    max_evidence: int = Field(default=100, description="Maximum evidence entries.")
    max_contradictions: int = Field(
        default=50, description="Maximum contradictions to return."
    )
    max_cross_refs: int = Field(default=200, description="Maximum cross references.")
    min_similarity: float = Field(default=0.1, description="Minimum similarity.")
    min_shared_terms: int = Field(default=2, description="Minimum shared terms.")
    bucket_size: str = Field(default="1d", description="Timeline bucket size.")
    default_time_unit: str = Field(
        default="s", description="Default time unit for numeric timestamps."
    )
    max_items_summary: int = Field(
        default=50, description="Maximum items summarized."
    )


class ContextDeepResearchState(BaseToolState):
    pass


class ResearchItem(BaseModel):
    id: str | None = Field(default=None, description="Optional item id.")
    title: str | None = Field(default=None, description="Optional title.")
    content: str | None = Field(default=None, description="Inline text content.")
    path: str | None = Field(default=None, description="Path to a text file.")
    source: str | None = Field(default=None, description="Source description.")
    timestamp: float | int | str | None = Field(default=None, description="Timestamp.")
    time_unit: str | None = Field(default=None, description="Unit for numeric timestamps.")


class ContextDeepResearchArgs(BaseModel):
    action: str | None = Field(default="analyze", description="analyze or update.")
    question: str | None = Field(default=None, description="Research question.")
    items: list[ResearchItem] | None = Field(
        default=None, description="Research items."
    )
    profile_path: str | None = Field(default=None, description="Override profile path.")
    bucket_size: str | None = Field(default=None, description="Override bucket size.")
    default_time_unit: str | None = Field(
        default=None, description="Override default time unit."
    )
    max_items: int | None = Field(default=None, description="Override max_items.")
    max_profile_entries: int | None = Field(
        default=None, description="Override max_profile_entries."
    )
    max_source_bytes: int | None = Field(
        default=None, description="Override max_source_bytes."
    )
    max_total_bytes: int | None = Field(
        default=None, description="Override max_total_bytes."
    )
    preview_chars: int | None = Field(default=None, description="Override preview_chars.")
    min_word_length: int | None = Field(
        default=None, description="Override min_word_length."
    )
    max_keywords: int | None = Field(
        default=None, description="Override max_keywords."
    )
    max_claims: int | None = Field(default=None, description="Override max_claims.")
    max_questions: int | None = Field(
        default=None, description="Override max_questions."
    )
    max_evidence: int | None = Field(default=None, description="Override max_evidence.")
    max_contradictions: int | None = Field(
        default=None, description="Override max_contradictions."
    )
    max_cross_refs: int | None = Field(
        default=None, description="Override max_cross_refs."
    )
    min_similarity: float | None = Field(
        default=None, description="Override min_similarity."
    )
    min_shared_terms: int | None = Field(
        default=None, description="Override min_shared_terms."
    )
    max_items_summary: int | None = Field(
        default=None, description="Override max_items_summary."
    )

class ResearchSummary(BaseModel):
    item_count: int
    claim_count: int
    question_count: int
    keyword_count: int
    keywords: list[str]
    sources: list[str]


class ResearchQuestion(BaseModel):
    question: str
    source: str


class ResearchClaim(BaseModel):
    claim: str
    source_id: str
    source: str | None
    keywords: list[str]


class EvidenceLink(BaseModel):
    claim: str
    source_id: str
    evidence_source_id: str
    evidence_source: str | None
    shared_terms: list[str]
    similarity: float


class Contradiction(BaseModel):
    claim: str
    claim_source: str
    conflicting_claim: str
    conflicting_source: str
    shared_terms: list[str]


class CrossReference(BaseModel):
    item_id: str
    related_item_id: str
    similarity: float
    shared_terms: list[str]


class TimelineBucket(BaseModel):
    index: int
    start_ns: int
    end_ns: int
    item_count: int
    keywords: list[str]


class ResearchItemSummary(BaseModel):
    item_id: str
    source: str | None
    preview: str
    keywords: list[str]


class ContextDeepResearchResult(BaseModel):
    summary: ResearchSummary
    questions: list[ResearchQuestion]
    claims: list[ResearchClaim]
    evidence: list[EvidenceLink]
    contradictions: list[Contradiction]
    cross_refs: list[CrossReference]
    buckets: list[TimelineBucket]
    items: list[ResearchItemSummary]
    profile_path: str | None
    updated_items: int
    truncated: bool
    errors: list[str]


class ContextDeepResearch(
    BaseTool[
        ContextDeepResearchArgs,
        ContextDeepResearchResult,
        ContextDeepResearchConfig,
        ContextDeepResearchState,
    ],
    ToolUIData[ContextDeepResearchArgs, ContextDeepResearchResult],
):
    description: ClassVar[str] = (
        "Deep research mode for structured multi-source text reasoning."
    )

    async def run(
        self, args: ContextDeepResearchArgs
    ) -> ContextDeepResearchResult:
        action = (args.action or "analyze").strip().lower()
        match action:
            case "analyze" | "update":
                pass
            case _:
                raise ToolError("action must be analyze or update.")

        max_items = args.max_items if args.max_items is not None else self.config.max_items
        max_profile_entries = (
            args.max_profile_entries
            if args.max_profile_entries is not None
            else self.config.max_profile_entries
        )
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
            args.preview_chars if args.preview_chars is not None else self.config.preview_chars
        )
        min_word_length = (
            args.min_word_length
            if args.min_word_length is not None
            else self.config.min_word_length
        )
        max_keywords = (
            args.max_keywords if args.max_keywords is not None else self.config.max_keywords
        )
        max_claims = args.max_claims if args.max_claims is not None else self.config.max_claims
        max_questions = (
            args.max_questions if args.max_questions is not None else self.config.max_questions
        )
        max_evidence = (
            args.max_evidence if args.max_evidence is not None else self.config.max_evidence
        )
        max_contradictions = (
            args.max_contradictions
            if args.max_contradictions is not None
            else self.config.max_contradictions
        )
        max_cross_refs = (
            args.max_cross_refs if args.max_cross_refs is not None else self.config.max_cross_refs
        )
        min_similarity = (
            args.min_similarity if args.min_similarity is not None else self.config.min_similarity
        )
        min_shared_terms = (
            args.min_shared_terms
            if args.min_shared_terms is not None
            else self.config.min_shared_terms
        )
        bucket_size = args.bucket_size or self.config.bucket_size
        default_time_unit = (
            args.default_time_unit
            if args.default_time_unit is not None
            else self.config.default_time_unit
        )
        max_items_summary = (
            args.max_items_summary
            if args.max_items_summary is not None
            else self.config.max_items_summary
        )

        if max_items <= 0:
            raise ToolError("max_items must be a positive integer.")
        if max_profile_entries <= 0:
            raise ToolError("max_profile_entries must be a positive integer.")
        if max_source_bytes <= 0:
            raise ToolError("max_source_bytes must be a positive integer.")
        if max_total_bytes <= 0:
            raise ToolError("max_total_bytes must be a positive integer.")
        if preview_chars < 0:
            raise ToolError("preview_chars must be >= 0.")
        if min_word_length <= 0:
            raise ToolError("min_word_length must be a positive integer.")
        if max_keywords < 0:
            raise ToolError("max_keywords must be >= 0.")
        if max_claims < 0:
            raise ToolError("max_claims must be >= 0.")
        if max_questions < 0:
            raise ToolError("max_questions must be >= 0.")
        if max_evidence < 0:
            raise ToolError("max_evidence must be >= 0.")
        if max_contradictions < 0:
            raise ToolError("max_contradictions must be >= 0.")
        if max_cross_refs < 0:
            raise ToolError("max_cross_refs must be >= 0.")
        if min_similarity < 0:
            raise ToolError("min_similarity must be >= 0.")
        if min_shared_terms < 0:
            raise ToolError("min_shared_terms must be >= 0.")
        if max_items_summary < 0:
            raise ToolError("max_items_summary must be >= 0.")

        profile_path = (
            Path(args.profile_path).expanduser()
            if args.profile_path
            else self.config.profile_path
        )

        if not args.items:
            raise ToolError("items is required.")
        if len(args.items) > max_items:
            raise ToolError(f"items exceeds max_items ({len(args.items)} > {max_items}).")

        errors: list[str] = []
        truncated = False
        total_bytes = 0
        updated_items = 0

        records: list[_ResearchRecord] = []
        for idx, item in enumerate(args.items, start=1):
            try:
                content, size_bytes = self._load_item_content(item, max_source_bytes)
                if content is None:
                    raise ToolError("Item has no content.")
                if size_bytes is not None:
                    if total_bytes + size_bytes > max_total_bytes:
                        truncated = True
                        break
                    total_bytes += size_bytes

                token_counts = self._extract_token_counts(content, min_word_length)
                tokens = set(token_counts.keys())
                sentences = self._split_sentences(content)
                timestamp_ns = self._parse_timestamp(
                    item.timestamp, item.time_unit or default_time_unit
                )
                preview = self._preview_text(content, preview_chars)
                item_id = item.id or item.title or item.path or f"item{idx}"
                records.append(
                    _ResearchRecord(
                        item_id=item_id,
                        timestamp_ns=timestamp_ns,
                        source=item.source,
                        preview=preview,
                        token_counts=token_counts,
                        tokens=tokens,
                        sentences=sentences,
                    )
                )
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not records:
            raise ToolError("No items were processed.")

        claim_records = self._extract_claims(records, max_claims, min_shared_terms)
        questions = self._build_questions(args.question, records, max_questions)
        evidence = self._build_evidence(
            claim_records, records, min_similarity, min_shared_terms, max_evidence
        )
        contradictions = self._build_contradictions(
            claim_records, min_shared_terms, max_contradictions
        )
        cross_refs = self._build_cross_refs(
            records, min_similarity, min_shared_terms, max_cross_refs
        )
        buckets = self._build_buckets(records, bucket_size, default_time_unit, max_keywords)

        keyword_counts: dict[str, int] = {}
        for record in records:
            for token, count in record.token_counts.items():
                keyword_counts[token] = keyword_counts.get(token, 0) + count

        claim_outputs = [
            ResearchClaim(
                claim=claim.claim,
                source_id=claim.item_id,
                source=claim.source,
                keywords=self._select_keywords(
                    self._token_counts_from_tokens(claim.tokens), max_keywords
                ),
            )
            for claim in claim_records[:max_claims] if max_claims > 0
        ]

        item_summaries = [
            ResearchItemSummary(
                item_id=record.item_id,
                source=record.source,
                preview=record.preview,
                keywords=self._select_keywords(record.token_counts, max_keywords),
            )
            for record in records[:max_items_summary] if max_items_summary > 0
        ]

        summary = ResearchSummary(
            item_count=len(records),
            claim_count=len(claim_outputs),
            question_count=len(questions),
            keyword_count=len(keyword_counts),
            keywords=self._select_keywords(keyword_counts, max_keywords),
            sources=sorted({record.source for record in records if record.source}),
        )

        if action == "update":
            profile_entries = self._load_profile(profile_path, errors)
            profile_entries.extend(
                [
                    {
                        "item_id": record.item_id,
                        "timestamp_ns": record.timestamp_ns,
                        "source": record.source,
                        "keywords": self._select_keywords(record.token_counts, max_keywords),
                        "claims": [
                            claim.claim
                            for claim in claim_records
                            if claim.item_id == record.item_id
                        ],
                    }
                    for record in records
                ]
            )
            profile_entries.sort(key=lambda entry: entry.get("timestamp_ns") or 0)
            if len(profile_entries) > max_profile_entries:
                truncated = True
                profile_entries = profile_entries[-max_profile_entries:]
            self._save_profile(profile_path, profile_entries, errors)
            updated_items = len(records)

        return ContextDeepResearchResult(
            summary=summary,
            questions=questions,
            claims=claim_outputs,
            evidence=evidence,
            contradictions=contradictions,
            cross_refs=cross_refs,
            buckets=buckets,
            items=item_summaries,
            profile_path=str(profile_path) if action == "update" else None,
            updated_items=updated_items,
            truncated=truncated,
            errors=errors,
        )

    def _load_item_content(
        self, item: ResearchItem, max_source_bytes: int
    ) -> tuple[str | None, int | None]:
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
            return path.read_text("utf-8", errors="ignore"), size
        if item.content is not None:
            size = len(item.content.encode("utf-8"))
            if size > max_source_bytes:
                raise ToolError(
                    f"content exceeds max_source_bytes ({size} > {max_source_bytes})."
                )
            return item.content, size
        return None, None

    def _split_sentences(self, text: str) -> list[str]:
        cleaned = text.replace("\r", " ").replace("\n", " ")
        return [s.strip() for s in SENTENCE_RE.split(cleaned) if s.strip()]

    def _extract_token_counts(self, text: str, min_word_length: int) -> dict[str, int]:
        counts: dict[str, int] = {}
        for token in TOKEN_RE.findall(text):
            lowered = token.lower()
            if len(lowered) < min_word_length:
                continue
            if lowered in STOPWORDS:
                continue
            counts[lowered] = counts.get(lowered, 0) + 1
        return counts

    def _token_counts_from_tokens(self, tokens: set[str]) -> dict[str, int]:
        return {token: 1 for token in tokens}

    def _select_keywords(self, counts: dict[str, int], limit: int) -> list[str]:
        if limit <= 0:
            return []
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [token for token, _ in ordered[:limit]]

    def _extract_claims(
        self,
        records: list[_ResearchRecord],
        max_claims: int,
        min_shared_terms: int,
    ) -> list[_ClaimRecord]:
        if max_claims <= 0:
            return []
        claims: list[_ClaimRecord] = []
        for record in records:
            for sentence in record.sentences:
                if len(claims) >= max_claims:
                    return claims
                if not CLAIM_RE.search(sentence):
                    continue
                tokens = {
                    token.lower()
                    for token in TOKEN_RE.findall(sentence)
                    if token.lower() not in STOPWORDS
                }
                if len(tokens) < min_shared_terms:
                    continue
                claims.append(
                    _ClaimRecord(
                        claim=sentence.strip(),
                        item_id=record.item_id,
                        source=record.source,
                        tokens=tokens,
                        has_negation=bool(NEGATION_RE.search(sentence)),
                    )
                )
        return claims

    def _build_questions(
        self,
        question: str | None,
        records: list[_ResearchRecord],
        max_questions: int,
    ) -> list[ResearchQuestion]:
        if max_questions <= 0:
            return []
        questions: list[ResearchQuestion] = []
        if question:
            questions.append(ResearchQuestion(question=question, source="user"))
        if len(questions) >= max_questions:
            return questions
        keyword_counts: dict[str, int] = {}
        for record in records:
            for token, count in record.token_counts.items():
                keyword_counts[token] = keyword_counts.get(token, 0) + count
        for token in self._select_keywords(keyword_counts, max_questions):
            if len(questions) >= max_questions:
                break
            questions.append(
                ResearchQuestion(question=f"What is {token}?", source="auto")
            )
        return questions

    def _build_evidence(
        self,
        claims: list[_ClaimRecord],
        records: list[_ResearchRecord],
        min_similarity: float,
        min_shared_terms: int,
        max_evidence: int,
    ) -> list[EvidenceLink]:
        if max_evidence <= 0:
            return []
        evidence: list[EvidenceLink] = []
        for claim in claims:
            for record in records:
                shared = claim.tokens & record.tokens
                if len(shared) < min_shared_terms:
                    continue
                union = claim.tokens | record.tokens
                similarity = len(shared) / len(union) if union else 0.0
                if similarity < min_similarity:
                    continue
                evidence.append(
                    EvidenceLink(
                        claim=claim.claim,
                        source_id=claim.item_id,
                        evidence_source_id=record.item_id,
                        evidence_source=record.source,
                        shared_terms=sorted(shared)[:min_shared_terms] if min_shared_terms > 0 else [],
                        similarity=round(similarity, 6),
                    )
                )
                if len(evidence) >= max_evidence:
                    return evidence
        return evidence

    def _build_contradictions(
        self,
        claims: list[_ClaimRecord],
        min_shared_terms: int,
        max_contradictions: int,
    ) -> list[Contradiction]:
        if max_contradictions <= 0:
            return []
        contradictions: list[Contradiction] = []
        for idx, left in enumerate(claims):
            for right in claims[idx + 1 :]:
                if left.item_id == right.item_id:
                    continue
                shared = left.tokens & right.tokens
                if len(shared) < min_shared_terms:
                    continue
                if left.has_negation == right.has_negation:
                    continue
                contradictions.append(
                    Contradiction(
                        claim=left.claim,
                        claim_source=left.item_id,
                        conflicting_claim=right.claim,
                        conflicting_source=right.item_id,
                        shared_terms=sorted(shared)[:min_shared_terms] if min_shared_terms > 0 else [],
                    )
                )
                if len(contradictions) >= max_contradictions:
                    return contradictions
        return contradictions

    def _build_cross_refs(
        self,
        records: list[_ResearchRecord],
        min_similarity: float,
        min_shared_terms: int,
        max_cross_refs: int,
    ) -> list[CrossReference]:
        if max_cross_refs <= 0:
            return []
        refs: list[CrossReference] = []
        for idx, left in enumerate(records):
            for right in records[idx + 1 :]:
                shared = left.tokens & right.tokens
                if len(shared) < min_shared_terms:
                    continue
                union = left.tokens | right.tokens
                similarity = len(shared) / len(union) if union else 0.0
                if similarity < min_similarity:
                    continue
                refs.append(
                    CrossReference(
                        item_id=left.item_id,
                        related_item_id=right.item_id,
                        similarity=round(similarity, 6),
                        shared_terms=sorted(shared)[:min_shared_terms] if min_shared_terms > 0 else [],
                    )
                )
                if len(refs) >= max_cross_refs:
                    return refs
        return refs

    def _build_buckets(
        self,
        records: list[_ResearchRecord],
        bucket_size: str,
        default_time_unit: str,
        max_keywords: int,
    ) -> list[TimelineBucket]:
        timestamps = [record.timestamp_ns for record in records if record.timestamp_ns is not None]
        if not timestamps:
            return []
        bucket_size_ns = self._parse_duration(bucket_size, default_time_unit)
        if bucket_size_ns <= 0:
            return []
        min_time = min(timestamps)
        max_time = max(timestamps)
        bucket_count = ((max_time - min_time) // bucket_size_ns) + 1
        bucket_tokens: list[dict[str, int]] = [
            {} for _ in range(int(bucket_count))
        ]
        bucket_sizes = [0 for _ in range(int(bucket_count))]
        for record in records:
            if record.timestamp_ns is None:
                continue
            index = (record.timestamp_ns - min_time) // bucket_size_ns
            idx = int(max(index, 0))
            if idx >= len(bucket_tokens):
                continue
            bucket_sizes[idx] += 1
            for token, count in record.token_counts.items():
                bucket_tokens[idx][token] = bucket_tokens[idx].get(token, 0) + count

        buckets: list[TimelineBucket] = []
        for idx in range(int(bucket_count)):
            start_ns = min_time + (idx * bucket_size_ns)
            end_ns = start_ns + bucket_size_ns
            keywords = self._select_keywords(bucket_tokens[idx], max_keywords)
            buckets.append(
                TimelineBucket(
                    index=idx,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    item_count=bucket_sizes[idx],
                    keywords=keywords,
                )
            )
        return buckets

    def _preview_text(self, text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def _parse_duration(self, value: str, default_unit: str) -> int:
        text = value.strip()
        if not text:
            return 0
        if self._is_numeric(text):
            return int(float(text) * self._resolve_time_unit(default_unit))
        match = re.match(r"^\s*([-+]?[0-9]*\.?[0-9]+)\s*([A-Za-z]+)\s*$", text)
        if not match:
            return 0
        amount = float(match.group(1))
        unit = match.group(2).lower()
        return int(amount * self._resolve_time_unit(unit))

    def _parse_timestamp(
        self, value: float | int | str | None, time_unit: str
    ) -> int | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return int(float(value) * self._resolve_time_unit(time_unit))
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if self._is_numeric(text):
                return int(float(text) * self._resolve_time_unit(time_unit))
            try:
                if text.endswith("Z"):
                    text = text[:-1] + "+00:00"
                dt = datetime.fromisoformat(text)
            except ValueError as exc:
                raise ToolError(f"Invalid timestamp string: {value}") from exc
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1_000_000_000)
        raise ToolError("timestamp must be a number or string.")

    def _resolve_time_unit(self, unit: str | None) -> int:
        key = (unit or "").strip().lower()
        if not key:
            raise ToolError("time_unit cannot be empty.")
        if key not in TIME_UNIT_NS:
            raise ToolError(f"Unsupported time unit: {unit}")
        return TIME_UNIT_NS[key]

    def _is_numeric(self, value: str) -> bool:
        return bool(re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value))

    def _load_profile(self, path: Path, errors: list[str]) -> list[dict]:
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"Failed to read profile: {exc}")
            return []
        entries = payload.get("entries", []) if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            return []
        return [entry for entry in entries if isinstance(entry, dict)]

    def _save_profile(self, path: Path, entries: list[dict], errors: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "entries": entries,
        }
        try:
            path.write_text(json.dumps(payload, indent=2), "utf-8")
        except OSError as exc:
            errors.append(f"Failed to write profile: {exc}")

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextDeepResearchArgs):
            return ToolCallDisplay(summary="context_deep_research")

        item_count = len(event.args.items or [])
        summary = f"context_deep_research: {item_count} item(s)"
        return ToolCallDisplay(
            summary=summary,
            details={
                "action": event.args.action,
                "item_count": item_count,
                "question": event.args.question,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextDeepResearchResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = (
            f"Processed {event.result.summary.item_count} item(s) "
            f"with {event.result.summary.claim_count} claim(s)"
        )
        warnings = event.result.errors[:]
        if event.result.truncated:
            warnings.append("Output truncated by size or limits")

        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=warnings,
            details={
                "summary": event.result.summary,
                "questions": event.result.questions,
                "claims": event.result.claims,
                "evidence": event.result.evidence,
                "contradictions": event.result.contradictions,
                "cross_refs": event.result.cross_refs,
                "buckets": event.result.buckets,
                "items": event.result.items,
                "profile_path": event.result.profile_path,
                "updated_items": event.result.updated_items,
                "truncated": event.result.truncated,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Running deep research analysis"
