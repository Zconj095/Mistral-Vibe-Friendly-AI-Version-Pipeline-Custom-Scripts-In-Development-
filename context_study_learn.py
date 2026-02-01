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
DEF_RE = re.compile(r"\b(is|are|means|refers to|defined as)\b", re.IGNORECASE)

QUESTION_WORDS = {
    "what",
    "why",
    "how",
    "who",
    "when",
    "where",
    "which",
}

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


@dataclass
class _StudyEntry:
    item_id: str
    timestamp_ns: int
    source: str | None
    token_counts: dict[str, int]
    definitions: list[str]
    questions: list[str]
    takeaways: list[str]


class ContextStudyLearnConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    profile_path: Path = Field(
        default=Path.home() / ".vibe" / "memory" / "study_profile.json",
        description="Path to the study profile file.",
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
    preview_chars: int = Field(default=200, description="Preview length per item.")
    min_word_length: int = Field(default=3, description="Minimum token length.")
    max_keywords: int = Field(default=20, description="Maximum keywords per summary.")
    max_definitions: int = Field(default=10, description="Maximum definitions to return.")
    max_questions: int = Field(default=10, description="Maximum questions to return.")
    max_takeaways: int = Field(default=10, description="Maximum takeaways to return.")
    max_items_summary: int = Field(
        default=50, description="Maximum items summarized."
    )


class ContextStudyLearnState(BaseToolState):
    pass


class StudyItem(BaseModel):
    id: str | None = Field(default=None, description="Optional item id.")
    title: str | None = Field(default=None, description="Optional title.")
    content: str | None = Field(default=None, description="Inline text content.")
    path: str | None = Field(default=None, description="Path to a text file.")
    source: str | None = Field(default=None, description="Source description.")


class ContextStudyLearnArgs(BaseModel):
    action: str | None = Field(default="study", description="study, learn, or review.")
    items: list[StudyItem] | None = Field(default=None, description="Study items.")
    profile_path: str | None = Field(default=None, description="Override profile path.")
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
    max_definitions: int | None = Field(
        default=None, description="Override max_definitions."
    )
    max_questions: int | None = Field(
        default=None, description="Override max_questions."
    )
    max_takeaways: int | None = Field(
        default=None, description="Override max_takeaways."
    )
    max_items_summary: int | None = Field(
        default=None, description="Override max_items_summary."
    )


class StudyPlan(BaseModel):
    keywords: list[str]
    definitions: list[str]
    questions: list[str]
    takeaways: list[str]
    focus_terms: list[str]


class StudySummary(BaseModel):
    item_count: int
    concept_count: int
    source_count: int
    keywords: list[str]


class StudyItemSummary(BaseModel):
    item_id: str
    source: str | None
    preview: str
    keywords: list[str]
    definitions: list[str]
    questions: list[str]
    takeaways: list[str]


class ContextStudyLearnResult(BaseModel):
    summary: StudySummary
    plan: StudyPlan
    items: list[StudyItemSummary]
    profile_path: str | None
    updated_items: int
    truncated: bool
    errors: list[str]


class ContextStudyLearn(
    BaseTool[
        ContextStudyLearnArgs,
        ContextStudyLearnResult,
        ContextStudyLearnConfig,
        ContextStudyLearnState,
    ],
    ToolUIData[ContextStudyLearnArgs, ContextStudyLearnResult],
):
    description: ClassVar[str] = (
        "Generate study plans and learn from text with a persistent profile."
    )

    async def run(self, args: ContextStudyLearnArgs) -> ContextStudyLearnResult:
        action = (args.action or "study").strip().lower()
        match action:
            case "study" | "learn" | "review":
                pass
            case _:
                raise ToolError("action must be study, learn, or review.")

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
        max_definitions = (
            args.max_definitions
            if args.max_definitions is not None
            else self.config.max_definitions
        )
        max_questions = (
            args.max_questions
            if args.max_questions is not None
            else self.config.max_questions
        )
        max_takeaways = (
            args.max_takeaways
            if args.max_takeaways is not None
            else self.config.max_takeaways
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
        if max_definitions < 0:
            raise ToolError("max_definitions must be >= 0.")
        if max_questions < 0:
            raise ToolError("max_questions must be >= 0.")
        if max_takeaways < 0:
            raise ToolError("max_takeaways must be >= 0.")
        if max_items_summary < 0:
            raise ToolError("max_items_summary must be >= 0.")

        profile_path = (
            Path(args.profile_path).expanduser()
            if args.profile_path
            else self.config.profile_path
        )

        errors: list[str] = []
        truncated = False
        updated_items = 0

        if action == "review":
            profile_entries = self._load_profile(profile_path, errors)
            if not profile_entries:
                raise ToolError("Study profile is empty.")
            return self._build_review_result(
                profile_entries,
                profile_path,
                max_keywords,
                max_definitions,
                max_questions,
                max_takeaways,
                max_items_summary,
                errors,
            )

        if not args.items:
            raise ToolError("items is required.")
        if len(args.items) > max_items:
            raise ToolError(f"items exceeds max_items ({len(args.items)} > {max_items}).")

        total_bytes = 0
        item_entries: list[_StudyEntry] = []
        summaries: list[StudyItemSummary] = []
        keyword_counts: dict[str, int] = {}
        definition_pool: list[str] = []
        question_pool: list[str] = []
        takeaway_pool: list[str] = []

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
                for token, count in token_counts.items():
                    keyword_counts[token] = keyword_counts.get(token, 0) + count
                keywords = self._select_keywords(token_counts, max_keywords)
                sentences = self._split_sentences(content)
                definitions = self._extract_definitions(sentences, max_definitions)
                questions = self._extract_questions(sentences, max_questions)
                takeaways = self._extract_takeaways(
                    sentences, keywords, max_takeaways
                )
                definition_pool.extend(definitions)
                question_pool.extend(questions)
                takeaway_pool.extend(takeaways)
                preview = self._preview_text(content, preview_chars)
                item_id = item.id or item.title or item.path or f"item{idx}"

                summaries.append(
                    StudyItemSummary(
                        item_id=item_id,
                        source=item.source,
                        preview=preview,
                        keywords=keywords,
                        definitions=definitions,
                        questions=questions,
                        takeaways=takeaways,
                    )
                )
                item_entries.append(
                    _StudyEntry(
                        item_id=item_id,
                        timestamp_ns=self._now_ns(),
                        source=item.source,
                        token_counts=token_counts,
                        definitions=definitions,
                        questions=questions,
                        takeaways=takeaways,
                    )
                )
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not summaries:
            raise ToolError("No items were processed.")

        plan = self._build_plan(
            keyword_counts,
            definition_pool,
            question_pool,
            takeaway_pool,
            max_keywords,
            max_definitions,
            max_questions,
            max_takeaways,
            focus_terms=[],
        )

        if action == "learn":
            profile_entries = self._load_profile(profile_path, errors)
            profile_entries.extend(item_entries)
            profile_entries.sort(key=lambda entry: entry.timestamp_ns)
            if len(profile_entries) > max_profile_entries:
                truncated = True
                profile_entries = profile_entries[-max_profile_entries:]
            updated_items = len(item_entries)
            self._save_profile(profile_path, profile_entries, errors)

        summary = StudySummary(
            item_count=len(summaries),
            concept_count=len(keyword_counts),
            source_count=len({summary.source for summary in summaries if summary.source}),
            keywords=self._select_keywords(keyword_counts, max_keywords),
        )

        return ContextStudyLearnResult(
            summary=summary,
            plan=plan,
            items=summaries[:max_items_summary] if max_items_summary > 0 else [],
            profile_path=str(profile_path) if action == "learn" else None,
            updated_items=updated_items,
            truncated=truncated,
            errors=errors,
        )

    def _build_review_result(
        self,
        entries: list[_StudyEntry],
        profile_path: Path,
        max_keywords: int,
        max_definitions: int,
        max_questions: int,
        max_takeaways: int,
        max_items_summary: int,
        errors: list[str],
    ) -> ContextStudyLearnResult:
        concept_counts: dict[str, int] = {}
        concept_last_seen: dict[str, int] = {}
        sources: set[str] = set()
        definition_pool: list[str] = []
        question_pool: list[str] = []
        takeaway_pool: list[str] = []
        summaries: list[StudyItemSummary] = []

        for entry in entries:
            for token, count in entry.token_counts.items():
                concept_counts[token] = concept_counts.get(token, 0) + count
                concept_last_seen[token] = max(
                    concept_last_seen.get(token, 0), entry.timestamp_ns
                )
            if entry.source:
                sources.add(entry.source)
            definition_pool.extend(entry.definitions)
            question_pool.extend(entry.questions)
            takeaway_pool.extend(entry.takeaways)
            if max_items_summary > 0 and len(summaries) < max_items_summary:
                summaries.append(
                    StudyItemSummary(
                        item_id=entry.item_id,
                        source=entry.source,
                        preview=self._preview_text(
                            " ".join(entry.takeaways), self.config.preview_chars
                        ),
                        keywords=self._select_keywords(entry.token_counts, max_keywords),
                        definitions=entry.definitions,
                        questions=entry.questions,
                        takeaways=entry.takeaways,
                    )
                )

        focus_terms = self._focus_terms(concept_counts, concept_last_seen, max_keywords)
        plan = self._build_plan(
            concept_counts,
            definition_pool,
            question_pool,
            takeaway_pool,
            max_keywords,
            max_definitions,
            max_questions,
            max_takeaways,
            focus_terms=focus_terms,
        )
        summary = StudySummary(
            item_count=len(entries),
            concept_count=len(concept_counts),
            source_count=len(sources),
            keywords=self._select_keywords(concept_counts, max_keywords),
        )
        return ContextStudyLearnResult(
            summary=summary,
            plan=plan,
            items=summaries,
            profile_path=str(profile_path),
            updated_items=0,
            truncated=False,
            errors=errors,
        )

    def _focus_terms(
        self,
        counts: dict[str, int],
        last_seen: dict[str, int],
        limit: int,
    ) -> list[str]:
        if limit <= 0:
            return []
        ranked = sorted(
            counts.items(),
            key=lambda item: (item[1], last_seen.get(item[0], 0), item[0]),
        )
        return [token for token, _ in ranked[:limit]]

    def _build_plan(
        self,
        keyword_counts: dict[str, int],
        definitions: list[str],
        questions: list[str],
        takeaways: list[str],
        max_keywords: int,
        max_definitions: int,
        max_questions: int,
        max_takeaways: int,
        focus_terms: list[str],
    ) -> StudyPlan:
        return StudyPlan(
            keywords=self._select_keywords(keyword_counts, max_keywords),
            definitions=self._unique_trim(definitions, max_definitions),
            questions=self._unique_trim(questions, max_questions),
            takeaways=self._unique_trim(takeaways, max_takeaways),
            focus_terms=focus_terms[:max_keywords] if max_keywords > 0 else [],
        )

    def _load_item_content(
        self, item: StudyItem, max_source_bytes: int
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
        sentences = [s.strip() for s in SENTENCE_RE.split(cleaned) if s.strip()]
        return sentences

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

    def _select_keywords(self, counts: dict[str, int], limit: int) -> list[str]:
        if limit <= 0:
            return []
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [token for token, _ in ordered[:limit]]

    def _extract_definitions(self, sentences: list[str], limit: int) -> list[str]:
        if limit <= 0:
            return []
        matches = [
            sentence
            for sentence in sentences
            if DEF_RE.search(sentence) and len(sentence.split()) >= 5
        ]
        return self._unique_trim(matches, limit)

    def _extract_questions(self, sentences: list[str], limit: int) -> list[str]:
        if limit <= 0:
            return []
        questions: list[str] = []
        for sentence in sentences:
            lower = sentence.strip().lower()
            if sentence.strip().endswith("?"):
                questions.append(sentence.strip())
                continue
            first = lower.split()
            if first and first[0] in QUESTION_WORDS:
                questions.append(sentence.strip())
        return self._unique_trim(questions, limit)

    def _extract_takeaways(
        self, sentences: list[str], keywords: list[str], limit: int
    ) -> list[str]:
        if limit <= 0 or not sentences:
            return []
        keyword_set = set(keywords)
        scored: list[tuple[int, str]] = []
        for sentence in sentences:
            tokens = {
                token.lower()
                for token in TOKEN_RE.findall(sentence)
                if token.lower() not in STOPWORDS
            }
            overlap = len(tokens & keyword_set)
            if overlap == 0:
                continue
            scored.append((overlap, sentence))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return self._unique_trim([sentence for _, sentence in scored], limit)

    def _unique_trim(self, items: list[str], limit: int) -> list[str]:
        if limit <= 0:
            return []
        output: list[str] = []
        seen: set[str] = set()
        for item in items:
            trimmed = item.strip()
            if not trimmed or trimmed in seen:
                continue
            seen.add(trimmed)
            output.append(trimmed)
            if len(output) >= limit:
                break
        return output

    def _preview_text(self, text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def _now_ns(self) -> int:
        return int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)

    def _load_profile(self, path: Path, errors: list[str]) -> list[_StudyEntry]:
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"Failed to read profile: {exc}")
            return []
        entries_raw = payload.get("entries", []) if isinstance(payload, dict) else payload
        entries: list[_StudyEntry] = []
        if not isinstance(entries_raw, list):
            return entries
        for entry in entries_raw:
            try:
                entries.append(
                    _StudyEntry(
                        item_id=str(entry.get("item_id")),
                        timestamp_ns=int(entry.get("timestamp_ns", 0)),
                        source=entry.get("source"),
                        token_counts={
                            str(key): int(value)
                            for key, value in (entry.get("token_counts") or {}).items()
                        },
                        definitions=[str(value) for value in entry.get("definitions", [])],
                        questions=[str(value) for value in entry.get("questions", [])],
                        takeaways=[str(value) for value in entry.get("takeaways", [])],
                    )
                )
            except Exception as exc:
                errors.append(f"Invalid profile entry: {exc}")
        return entries

    def _save_profile(
        self, path: Path, entries: list[_StudyEntry], errors: list[str]
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "entries": [
                {
                    "item_id": entry.item_id,
                    "timestamp_ns": entry.timestamp_ns,
                    "source": entry.source,
                    "token_counts": entry.token_counts,
                    "definitions": entry.definitions,
                    "questions": entry.questions,
                    "takeaways": entry.takeaways,
                }
                for entry in entries
            ],
        }
        try:
            path.write_text(json.dumps(payload, indent=2), "utf-8")
        except OSError as exc:
            errors.append(f"Failed to write profile: {exc}")

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextStudyLearnArgs):
            return ToolCallDisplay(summary="context_study_learn")

        item_count = len(event.args.items or [])
        summary = f"context_study_learn: {item_count} item(s)"
        return ToolCallDisplay(
            summary=summary,
            details={
                "action": event.args.action,
                "item_count": item_count,
                "profile_path": event.args.profile_path,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextStudyLearnResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = (
            f"Processed {event.result.summary.item_count} item(s)"
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
                "plan": event.result.plan,
                "items": event.result.items,
                "profile_path": event.result.profile_path,
                "updated_items": event.result.updated_items,
                "truncated": event.result.truncated,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Preparing study and learning context"
