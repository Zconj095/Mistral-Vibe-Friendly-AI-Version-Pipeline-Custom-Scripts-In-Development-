from __future__ import annotations

import csv
import json
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

SUPPORTED_CATALOG_FORMATS = {"auto", "lines", "json", "jsonl", "csv", "tsv"}


class ContextSpeechMultiDocumentReasoningConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_source_bytes: int = Field(
        default=5_000_000, description="Maximum bytes per document."
    )
    max_total_bytes: int = Field(
        default=20_000_000, description="Maximum total bytes across documents."
    )
    max_documents: int = Field(default=2000, description="Maximum documents to load.")
    preview_chars: int = Field(default=240, description="Preview snippet length.")
    min_token_length: int = Field(default=3, description="Minimum token length.")
    max_keywords: int = Field(default=12, description="Max keywords per document.")
    max_key_sentences: int = Field(default=2, description="Max key sentences per doc.")
    max_sentence_chars: int = Field(default=300, description="Max sentence chars.")
    min_shared_docs: int = Field(
        default=2, description="Minimum documents sharing a topic."
    )
    max_shared_topics: int = Field(default=20, description="Max shared topics.")
    max_unique_keywords: int = Field(default=6, description="Max unique keywords per doc.")
    min_similarity: float = Field(default=0.1, description="Min doc similarity.")
    max_comparisons: int = Field(default=20, description="Max doc comparisons.")
    max_shared_keywords: int = Field(
        default=8, description="Max shared keywords per comparison."
    )
    max_doc_segments: int = Field(
        default=10, description="Max document segments for speech."
    )
    max_topic_segments: int = Field(
        default=10, description="Max topic segments for speech."
    )
    max_comparison_segments: int = Field(
        default=6, description="Max comparison segments for speech."
    )
    max_question_keywords: int = Field(
        default=12, description="Max keywords extracted from the question."
    )
    max_answer_segments: int = Field(
        default=6, description="Max answer segments for question responses."
    )
    max_speech_segments: int = Field(
        default=40, description="Maximum total speech segments."
    )


class ContextSpeechMultiDocumentReasoningState(BaseToolState):
    pass


class DocumentInput(BaseModel):
    content: str | None = Field(default=None, description="Document content.")
    path: str | None = Field(default=None, description="Path to a document.")
    label: str | None = Field(default=None, description="Document label.")


class ContextSpeechMultiDocumentReasoningArgs(BaseModel):
    documents: list[DocumentInput] | None = Field(
        default=None, description="Documents with content or path."
    )
    paths: list[str] | None = Field(
        default=None, description="Document paths."
    )
    catalog_path: str | None = Field(
        default=None, description="Catalog file listing document paths."
    )
    catalog_format: str | None = Field(
        default="auto", description="auto, lines, json, jsonl, csv, tsv."
    )
    question: str | None = Field(
        default=None, description="Question to answer across documents."
    )
    max_source_bytes: int | None = Field(
        default=None, description="Override max_source_bytes."
    )
    max_total_bytes: int | None = Field(
        default=None, description="Override max_total_bytes."
    )
    max_documents: int | None = Field(
        default=None, description="Override max_documents."
    )
    min_token_length: int | None = Field(
        default=None, description="Override min_token_length."
    )
    max_keywords: int | None = Field(
        default=None, description="Override max_keywords."
    )
    min_shared_docs: int | None = Field(
        default=None, description="Override min_shared_docs."
    )
    max_shared_topics: int | None = Field(
        default=None, description="Override max_shared_topics."
    )
    max_unique_keywords: int | None = Field(
        default=None, description="Override max_unique_keywords."
    )
    min_similarity: float | None = Field(
        default=None, description="Override min_similarity."
    )
    max_comparisons: int | None = Field(
        default=None, description="Override max_comparisons."
    )
    max_shared_keywords: int | None = Field(
        default=None, description="Override max_shared_keywords."
    )
    max_doc_segments: int | None = Field(
        default=None, description="Override max_doc_segments."
    )
    max_topic_segments: int | None = Field(
        default=None, description="Override max_topic_segments."
    )
    max_comparison_segments: int | None = Field(
        default=None, description="Override max_comparison_segments."
    )
    max_question_keywords: int | None = Field(
        default=None, description="Override max_question_keywords."
    )
    max_answer_segments: int | None = Field(
        default=None, description="Override max_answer_segments."
    )
    max_speech_segments: int | None = Field(
        default=None, description="Override max_speech_segments."
    )
    include_comparisons: bool = Field(
        default=True, description="Include document comparisons."
    )
    include_answer_segments: bool = Field(
        default=True, description="Include answer segments when question is provided."
    )
    include_opening: bool = Field(
        default=True, description="Include speech opening."
    )
    include_closing: bool = Field(
        default=True, description="Include speech closing."
    )


class DocumentSummary(BaseModel):
    index: int
    label: str
    source_path: str | None
    char_count: int
    word_count: int
    preview: str
    keywords: list[str]
    key_sentences: list[str]
    unique_keywords: list[str]


class SharedTopic(BaseModel):
    term: str
    doc_indices: list[int]
    doc_count: int
    keyword_score: int


class DocumentComparison(BaseModel):
    source_index: int
    target_index: int
    similarity: float
    shared_keywords: list[str]


class QuestionDocumentMatch(BaseModel):
    doc_index: int
    doc_label: str
    overlap_keywords: list[str]
    score: float


class SpeechSegment(BaseModel):
    index: int
    kind: str
    doc_indices: list[int]
    cue: str


class ContextSpeechMultiDocumentReasoningResult(BaseModel):
    documents: list[DocumentSummary]
    document_count: int
    question: str | None
    question_keywords: list[str]
    question_matches: list[QuestionDocumentMatch]
    question_shared_topics: list[SharedTopic]
    shared_topics: list[SharedTopic]
    shared_topic_count: int
    comparisons: list[DocumentComparison]
    comparison_count: int
    speech_opening: str
    speech_segments: list[SpeechSegment]
    speech_closing: str
    input_bytes: int
    truncated: bool
    warnings: list[str]


class ContextSpeechMultiDocumentReasoning(
    BaseTool[
        ContextSpeechMultiDocumentReasoningArgs,
        ContextSpeechMultiDocumentReasoningResult,
        ContextSpeechMultiDocumentReasoningConfig,
        ContextSpeechMultiDocumentReasoningState,
    ],
    ToolUIData[
        ContextSpeechMultiDocumentReasoningArgs,
        ContextSpeechMultiDocumentReasoningResult,
    ],
):
    description: ClassVar[str] = (
        "Reason across multiple documents and prepare speech cues."
    )

    async def run(
        self, args: ContextSpeechMultiDocumentReasoningArgs
    ) -> ContextSpeechMultiDocumentReasoningResult:
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
        max_documents = (
            args.max_documents
            if args.max_documents is not None
            else self.config.max_documents
        )
        min_token_length = (
            args.min_token_length
            if args.min_token_length is not None
            else self.config.min_token_length
        )
        max_keywords = (
            args.max_keywords
            if args.max_keywords is not None
            else self.config.max_keywords
        )
        min_shared_docs = (
            args.min_shared_docs
            if args.min_shared_docs is not None
            else self.config.min_shared_docs
        )
        max_shared_topics = (
            args.max_shared_topics
            if args.max_shared_topics is not None
            else self.config.max_shared_topics
        )
        max_unique_keywords = (
            args.max_unique_keywords
            if args.max_unique_keywords is not None
            else self.config.max_unique_keywords
        )
        min_similarity = (
            args.min_similarity
            if args.min_similarity is not None
            else self.config.min_similarity
        )
        max_comparisons = (
            args.max_comparisons
            if args.max_comparisons is not None
            else self.config.max_comparisons
        )
        max_shared_keywords = (
            args.max_shared_keywords
            if args.max_shared_keywords is not None
            else self.config.max_shared_keywords
        )
        max_doc_segments = (
            args.max_doc_segments
            if args.max_doc_segments is not None
            else self.config.max_doc_segments
        )
        max_topic_segments = (
            args.max_topic_segments
            if args.max_topic_segments is not None
            else self.config.max_topic_segments
        )
        max_comparison_segments = (
            args.max_comparison_segments
            if args.max_comparison_segments is not None
            else self.config.max_comparison_segments
        )
        max_question_keywords = (
            args.max_question_keywords
            if args.max_question_keywords is not None
            else self.config.max_question_keywords
        )
        max_answer_segments = (
            args.max_answer_segments
            if args.max_answer_segments is not None
            else self.config.max_answer_segments
        )
        max_speech_segments = (
            args.max_speech_segments
            if args.max_speech_segments is not None
            else self.config.max_speech_segments
        )

        if max_source_bytes <= 0:
            raise ToolError("max_source_bytes must be positive.")
        if max_total_bytes <= 0:
            raise ToolError("max_total_bytes must be positive.")
        if max_documents <= 0:
            raise ToolError("max_documents must be positive.")
        if min_token_length <= 0:
            raise ToolError("min_token_length must be positive.")
        if max_keywords < 0:
            raise ToolError("max_keywords must be non-negative.")
        if min_shared_docs <= 0:
            raise ToolError("min_shared_docs must be positive.")
        if max_shared_topics < 0:
            raise ToolError("max_shared_topics must be non-negative.")
        if max_unique_keywords < 0:
            raise ToolError("max_unique_keywords must be non-negative.")
        if min_similarity < 0:
            raise ToolError("min_similarity must be non-negative.")
        if max_comparisons < 0:
            raise ToolError("max_comparisons must be non-negative.")
        if max_shared_keywords < 0:
            raise ToolError("max_shared_keywords must be non-negative.")
        if max_doc_segments < 0:
            raise ToolError("max_doc_segments must be non-negative.")
        if max_topic_segments < 0:
            raise ToolError("max_topic_segments must be non-negative.")
        if max_comparison_segments < 0:
            raise ToolError("max_comparison_segments must be non-negative.")
        if max_question_keywords < 0:
            raise ToolError("max_question_keywords must be non-negative.")
        if max_answer_segments < 0:
            raise ToolError("max_answer_segments must be non-negative.")
        if max_speech_segments < 0:
            raise ToolError("max_speech_segments must be non-negative.")

        inputs, warnings, truncated_inputs = self._collect_inputs(args)
        if not inputs:
            return ContextSpeechMultiDocumentReasoningResult(
                documents=[],
                document_count=0,
                question=args.question.strip() if args.question else None,
                question_keywords=[],
                question_matches=[],
                question_shared_topics=[],
                shared_topics=[],
                shared_topic_count=0,
                comparisons=[],
                comparison_count=0,
                speech_opening="",
                speech_segments=[],
                speech_closing="",
                input_bytes=0,
                truncated=truncated_inputs,
                warnings=warnings or ["No documents provided."],
            )

        if len(inputs) > max_documents:
            inputs = inputs[:max_documents]
            truncated_inputs = True
            warnings.append("Document list truncated by max_documents.")

        summaries: list[DocumentSummary] = []
        token_counts: list[Counter[str]] = []
        keyword_sets: list[set[str]] = []
        total_bytes = 0
        truncated = truncated_inputs

        for idx, doc in enumerate(inputs, start=1):
            content, source_path, size = self._load_document(doc, max_source_bytes)
            if size is not None and total_bytes + size > max_total_bytes:
                truncated = True
                warnings.append("Total document bytes exceed max_total_bytes.")
                break
            if size is not None:
                total_bytes += size
            label = self._label_for(doc, source_path, idx)
            tokens = self._tokenize(content, min_token_length)
            counts = Counter(tokens)
            keywords = self._top_keywords(counts, max_keywords)
            preview = self._preview(content)
            key_sentences = self._key_sentences(content, keywords)
            summaries.append(
                DocumentSummary(
                    index=idx,
                    label=label,
                    source_path=source_path,
                    char_count=len(content),
                    word_count=len(tokens),
                    preview=preview,
                    keywords=keywords,
                    key_sentences=key_sentences,
                    unique_keywords=[],
                )
            )
            token_counts.append(counts)
            keyword_sets.append(set(keywords))

        if not summaries:
            return ContextSpeechMultiDocumentReasoningResult(
                documents=[],
                document_count=0,
                question=args.question.strip() if args.question else None,
                question_keywords=[],
                question_matches=[],
                question_shared_topics=[],
                shared_topics=[],
                shared_topic_count=0,
                comparisons=[],
                comparison_count=0,
                speech_opening="",
                speech_segments=[],
                speech_closing="",
                input_bytes=total_bytes,
                truncated=truncated,
                warnings=warnings or ["No documents loaded."],
            )

        doc_freq = Counter()
        keyword_scores = Counter()
        for counts, keywords in zip(token_counts, keyword_sets):
            for kw in keywords:
                doc_freq[kw] += 1
                keyword_scores[kw] += counts.get(kw, 0)

        shared_topics = self._shared_topics(
            doc_freq,
            keyword_scores,
            keyword_sets,
            min_shared_docs,
            max_shared_topics,
        )
        for summary in summaries:
            unique = [kw for kw in summary.keywords if doc_freq.get(kw, 0) == 1]
            summary.unique_keywords = unique[:max_unique_keywords]

        comparisons = []
        if args.include_comparisons:
            comparisons = self._compare_documents(
                keyword_sets, min_similarity, max_comparisons, max_shared_keywords
            )

        question = (args.question or "").strip()
        question_keywords: list[str] = []
        question_matches: list[QuestionDocumentMatch] = []
        question_shared_topics: list[SharedTopic] = []
        question_terms: set[str] = set()
        if question:
            question_tokens = self._tokenize(question, min_token_length)
            if not question_tokens:
                warnings.append("Question has no indexable keywords.")
            question_counts = Counter(question_tokens)
            question_keywords = self._top_keywords(
                question_counts, max_question_keywords
            )
            question_terms = set(question_keywords) if question_keywords else set(question_tokens)
            if question_terms:
                for idx, counts in enumerate(token_counts):
                    doc_terms = set(counts.keys())
                    overlap = sorted(question_terms & doc_terms)
                    if not overlap:
                        continue
                    score = len(overlap) / len(question_terms)
                    question_matches.append(
                        QuestionDocumentMatch(
                            doc_index=idx + 1,
                            doc_label=summaries[idx].label,
                            overlap_keywords=overlap[:max_shared_keywords],
                            score=score,
                        )
                    )
                question_matches.sort(
                    key=lambda match: (-match.score, match.doc_index)
                )
                question_shared_topics = [
                    topic for topic in shared_topics if topic.term in question_terms
                ]

        speech_opening = self._speech_opening(
            args,
            len(summaries),
            [topic.term for topic in shared_topics[:3]],
            question,
        )

        speech_segments, segments_truncated = self._speech_segments(
            summaries,
            shared_topics,
            comparisons,
            max_doc_segments,
            max_topic_segments,
            max_comparison_segments,
            max_speech_segments,
            question,
            question_keywords,
            question_matches,
            question_shared_topics,
            args.include_answer_segments,
            max_answer_segments,
        )
        if segments_truncated:
            warnings.append("Speech segments truncated by limits.")

        speech_closing = self._speech_closing(
            args,
            len(shared_topics),
            len(comparisons),
            question,
        )

        return ContextSpeechMultiDocumentReasoningResult(
            documents=summaries,
            document_count=len(summaries),
            question=question or None,
            question_keywords=question_keywords,
            question_matches=question_matches,
            question_shared_topics=question_shared_topics,
            shared_topics=shared_topics,
            shared_topic_count=len(shared_topics),
            comparisons=comparisons,
            comparison_count=len(comparisons),
            speech_opening=speech_opening,
            speech_segments=speech_segments,
            speech_closing=speech_closing,
            input_bytes=total_bytes,
            truncated=truncated,
            warnings=warnings,
        )

    def _collect_inputs(
        self, args: ContextSpeechMultiDocumentReasoningArgs
    ) -> tuple[list[DocumentInput], list[str], bool]:
        inputs: list[DocumentInput] = []
        warnings: list[str] = []
        truncated = False

        if args.documents:
            inputs.extend(args.documents)

        if args.paths:
            for path in args.paths:
                inputs.append(DocumentInput(path=path))

        if args.catalog_path:
            catalog_inputs, catalog_warnings = self._load_catalog(
                args.catalog_path, args.catalog_format
            )
            inputs.extend(catalog_inputs)
            warnings.extend(catalog_warnings)

        if not inputs:
            return [], warnings, truncated

        return inputs, warnings, truncated

    def _load_document(
        self, doc: DocumentInput, max_source_bytes: int
    ) -> tuple[str, str | None, int | None]:
        if doc.content and doc.path:
            raise ToolError("Provide content or path, not both.")
        if doc.content is not None:
            data = doc.content.encode("utf-8")
            if len(data) > max_source_bytes:
                raise ToolError(
                    f"content exceeds max_source_bytes ({len(data)} > {max_source_bytes})."
                )
            return doc.content, None, len(data)
        if not doc.path:
            raise ToolError("Document missing content or path.")
        path = self._resolve_path(doc.path)
        if path.is_dir():
            raise ToolError(f"Path is a directory: {path}")
        size = path.stat().st_size
        if size > max_source_bytes:
            raise ToolError(
                f"{path} exceeds max_source_bytes ({size} > {max_source_bytes})."
            )
        return path.read_text("utf-8", errors="ignore"), str(path), size

    def _load_catalog(
        self, raw_path: str, fmt: str | None
    ) -> tuple[list[DocumentInput], list[str]]:
        errors: list[str] = []
        path = self._resolve_path(raw_path)
        if not path.exists():
            raise ToolError(f"Catalog not found: {path}")
        if path.is_dir():
            raise ToolError(f"Catalog path is a directory: {path}")

        format_value = (fmt or "auto").strip().lower()
        if format_value not in SUPPORTED_CATALOG_FORMATS:
            raise ToolError("catalog_format must be auto, lines, json, jsonl, csv, or tsv.")
        if format_value == "auto":
            format_value = self._detect_format(path)

        inputs: list[DocumentInput] = []
        if format_value == "lines":
            for line in path.read_text("utf-8", errors="ignore").splitlines():
                value = line.strip()
                if not value or value.startswith("#"):
                    continue
                inputs.append(DocumentInput(path=value))
            return inputs, errors

        if format_value == "json":
            try:
                payload = json.loads(path.read_text("utf-8", errors="ignore"))
            except json.JSONDecodeError as exc:
                raise ToolError(f"Invalid catalog JSON: {exc}") from exc
            entries = payload
            if isinstance(payload, dict):
                entries = (
                    payload.get("paths")
                    or payload.get("documents")
                    or payload.get("files")
                    or []
                )
            if isinstance(entries, list):
                for entry in entries:
                    self._append_catalog_entry(inputs, entry)
            else:
                errors.append("Catalog JSON must contain a list.")
            return inputs, errors

        if format_value == "jsonl":
            for line in path.read_text("utf-8", errors="ignore").splitlines():
                value = line.strip()
                if not value:
                    continue
                try:
                    entry = json.loads(value)
                except json.JSONDecodeError:
                    errors.append("Invalid JSONL entry ignored.")
                    continue
                self._append_catalog_entry(inputs, entry)
            return inputs, errors

        delimiter = "," if format_value == "csv" else "\t"
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if reader.fieldnames:
                for row in reader:
                    entry = row.get("path") or row.get("file") or row.get("name")
                    if entry:
                        inputs.append(DocumentInput(path=entry))
            else:
                handle.seek(0)
                reader = csv.reader(handle, delimiter=delimiter)
                for row in reader:
                    if not row:
                        continue
                    entry = row[0].strip()
                    if entry:
                        inputs.append(DocumentInput(path=entry))
        return inputs, errors

    def _append_catalog_entry(self, inputs: list[DocumentInput], entry: object) -> None:
        if isinstance(entry, str):
            inputs.append(DocumentInput(path=entry))
        elif isinstance(entry, dict):
            path = entry.get("path") or entry.get("file")
            content = entry.get("content")
            label = entry.get("label")
            inputs.append(DocumentInput(path=path, content=content, label=label))

    def _detect_format(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".json":
            return "json"
        if suffix == ".jsonl":
            return "jsonl"
        if suffix == ".csv":
            return "csv"
        if suffix == ".tsv":
            return "tsv"
        return "lines"

    def _resolve_path(self, raw_path: str) -> Path:
        if not raw_path.strip():
            raise ToolError("path cannot be empty.")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.config.effective_workdir / path
        return path.resolve()

    def _label_for(self, doc: DocumentInput, source_path: str | None, index: int) -> str:
        if doc.label:
            return doc.label
        if source_path:
            return Path(source_path).name
        return f"Document {index}"

    def _tokenize(self, content: str, min_token_length: int) -> list[str]:
        tokens = [
            token.lower()
            for token in WORD_RE.findall(content)
            if len(token) >= min_token_length and token.lower() not in STOPWORDS
        ]
        return tokens

    def _top_keywords(self, counts: Counter[str], max_keywords: int) -> list[str]:
        if max_keywords <= 0:
            return []
        return [token for token, _ in counts.most_common(max_keywords)]

    def _preview(self, content: str) -> str:
        trimmed = " ".join(content.strip().split())
        if len(trimmed) <= self.config.preview_chars:
            return trimmed
        return trimmed[: self.config.preview_chars]

    def _key_sentences(self, content: str, keywords: list[str]) -> list[str]:
        if not keywords:
            return []
        sentences = [sentence.strip() for sentence in SENTENCE_RE.findall(content)]
        if not sentences:
            return []
        scores: list[tuple[int, str]] = []
        keyword_set = set(keywords)
        for sentence in sentences:
            tokens = {token.lower() for token in WORD_RE.findall(sentence)}
            score = sum(1 for token in tokens if token in keyword_set)
            if score:
                scores.append((score, sentence))
        scores.sort(key=lambda item: (-item[0], item[1]))
        key_sentences = [sentence for _, sentence in scores[: self.config.max_key_sentences]]
        clipped = [
            sentence[: self.config.max_sentence_chars] for sentence in key_sentences
        ]
        return clipped

    def _shared_topics(
        self,
        doc_freq: Counter[str],
        keyword_scores: Counter[str],
        keyword_sets: list[set[str]],
        min_shared_docs: int,
        max_shared_topics: int,
    ) -> list[SharedTopic]:
        topics: list[SharedTopic] = []
        for term, count in doc_freq.items():
            if count < min_shared_docs:
                continue
            doc_indices = [
                idx + 1
                for idx, keywords in enumerate(keyword_sets)
                if term in keywords
            ]
            topics.append(
                SharedTopic(
                    term=term,
                    doc_indices=doc_indices,
                    doc_count=count,
                    keyword_score=keyword_scores.get(term, 0),
                )
            )
        topics.sort(key=lambda topic: (-topic.doc_count, -topic.keyword_score, topic.term))
        if max_shared_topics and len(topics) > max_shared_topics:
            topics = topics[:max_shared_topics]
        return topics

    def _compare_documents(
        self,
        keyword_sets: list[set[str]],
        min_similarity: float,
        max_comparisons: int,
        max_shared_keywords: int,
    ) -> list[DocumentComparison]:
        comparisons: list[DocumentComparison] = []
        for i in range(len(keyword_sets)):
            for j in range(i + 1, len(keyword_sets)):
                shared = sorted(keyword_sets[i] & keyword_sets[j])
                union = keyword_sets[i] | keyword_sets[j]
                if not union:
                    continue
                similarity = len(shared) / len(union)
                if similarity < min_similarity:
                    continue
                comparisons.append(
                    DocumentComparison(
                        source_index=i + 1,
                        target_index=j + 1,
                        similarity=similarity,
                        shared_keywords=shared[:max_shared_keywords],
                    )
                )
        comparisons.sort(
            key=lambda comp: (-comp.similarity, comp.source_index, comp.target_index)
        )
        if max_comparisons and len(comparisons) > max_comparisons:
            comparisons = comparisons[:max_comparisons]
        return comparisons

    def _speech_opening(
        self,
        args: ContextSpeechMultiDocumentReasoningArgs,
        doc_count: int,
        shared_terms: list[str],
        question: str,
    ) -> str:
        if not args.include_opening:
            return ""
        parts = [f"Begin by comparing {doc_count} document(s)."]
        if shared_terms:
            parts.append(f"Shared topics include: {', '.join(shared_terms)}.")
        if question:
            parts.append(f"Question focus: {self._preview(question)}.")
        return " ".join(parts)

    def _speech_segments(
        self,
        documents: list[DocumentSummary],
        shared_topics: list[SharedTopic],
        comparisons: list[DocumentComparison],
        max_doc_segments: int,
        max_topic_segments: int,
        max_comparison_segments: int,
        max_speech_segments: int,
        question: str,
        question_keywords: list[str],
        question_matches: list[QuestionDocumentMatch],
        question_shared_topics: list[SharedTopic],
        include_answer_segments: bool,
        max_answer_segments: int,
    ) -> tuple[list[SpeechSegment], bool]:
        segments: list[SpeechSegment] = []

        if question and include_answer_segments:
            if question_matches:
                for match in question_matches[:max_answer_segments]:
                    overlap = ", ".join(match.overlap_keywords[:6])
                    cue_parts = [
                        f"Answer the question using {match.doc_label}."
                    ]
                    if overlap:
                        cue_parts.append(f"Relevant terms: {overlap}.")
                    segments.append(
                        SpeechSegment(
                            index=len(segments) + 1,
                            kind="question",
                            doc_indices=[match.doc_index],
                            cue=" ".join(cue_parts).strip(),
                        )
                    )
            else:
                cue = f"Answer the question '{self._preview(question)}' using the documents."
                segments.append(
                    SpeechSegment(
                        index=len(segments) + 1,
                        kind="question",
                        doc_indices=[],
                        cue=cue,
                    )
                )
            if question_shared_topics:
                shared_terms = ", ".join(
                    topic.term for topic in question_shared_topics[:4]
                )
                doc_indices = sorted(
                    {doc_idx for topic in question_shared_topics for doc_idx in topic.doc_indices}
                )
                cue = (
                    f"Connect the question to shared topics: {shared_terms} "
                    f"across documents {doc_indices}."
                )
                segments.append(
                    SpeechSegment(
                        index=len(segments) + 1,
                        kind="question_shared",
                        doc_indices=doc_indices,
                        cue=cue,
                    )
                )
            if question_keywords:
                cue = (
                    "Anchor the answer with keywords: "
                    f"{', '.join(question_keywords[:6])}."
                )
                segments.append(
                    SpeechSegment(
                        index=len(segments) + 1,
                        kind="question_focus",
                        doc_indices=[],
                        cue=cue,
                    )
                )

        for doc in documents[:max_doc_segments]:
            keywords = ", ".join(doc.keywords[:6])
            cue_parts = [f"Summarize {doc.label}."]
            if keywords:
                cue_parts.append(f"Key terms: {keywords}.")
            if doc.unique_keywords:
                cue_parts.append(
                    f"Unique terms: {', '.join(doc.unique_keywords[:4])}."
                )
            if doc.preview:
                cue_parts.append(f"Context: {doc.preview}")
            segments.append(
                SpeechSegment(
                    index=len(segments) + 1,
                    kind="document",
                    doc_indices=[doc.index],
                    cue=" ".join(cue_parts).strip(),
                )
            )

        for topic in shared_topics[:max_topic_segments]:
            cue = (
                f"Cross-reference documents {topic.doc_indices} on "
                f"'{topic.term}'."
            )
            segments.append(
                SpeechSegment(
                    index=len(segments) + 1,
                    kind="shared_topic",
                    doc_indices=topic.doc_indices,
                    cue=cue,
                )
            )

        for comp in comparisons[:max_comparison_segments]:
            cue = (
                f"Compare documents {comp.source_index} and {comp.target_index}; "
                f"similarity {comp.similarity:.2f}."
            )
            if comp.shared_keywords:
                cue = f"{cue} Shared terms: {', '.join(comp.shared_keywords)}."
            segments.append(
                SpeechSegment(
                    index=len(segments) + 1,
                    kind="comparison",
                    doc_indices=[comp.source_index, comp.target_index],
                    cue=cue,
                )
            )

        truncated = False
        if max_speech_segments and len(segments) > max_speech_segments:
            segments = segments[:max_speech_segments]
            truncated = True

        for idx, segment in enumerate(segments, start=1):
            segment.index = idx
        return segments, truncated

    def _speech_closing(
        self,
        args: ContextSpeechMultiDocumentReasoningArgs,
        shared_topic_count: int,
        comparison_count: int,
        question: str,
    ) -> str:
        if not args.include_closing:
            return ""
        if shared_topic_count or comparison_count:
            return "Close by summarizing shared themes and key differences."
        if question:
            return "Close by confirming the answer addresses the question."
        return "Close by summarizing each document briefly."

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextSpeechMultiDocumentReasoningArgs):
            return ToolCallDisplay(summary="context_speech_multi_document_reasoning")
        doc_count = len(event.args.documents or []) + len(event.args.paths or [])
        return ToolCallDisplay(
            summary="context_speech_multi_document_reasoning",
            details={
                "document_count": doc_count,
                "catalog_path": event.args.catalog_path,
                "max_documents": event.args.max_documents,
                "question": event.args.question,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpeechMultiDocumentReasoningResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = (
            f"Reasoned across {event.result.document_count} document(s) with "
            f"{event.result.shared_topic_count} shared topic(s)"
        )
        warnings = event.result.warnings[:]
        if event.result.truncated:
            warnings.append("Output truncated by size or limits")
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=warnings,
            details={
                "document_count": event.result.document_count,
                "shared_topic_count": event.result.shared_topic_count,
                "comparison_count": event.result.comparison_count,
                "question": event.result.question,
                "question_match_count": len(event.result.question_matches),
                "truncated": event.result.truncated,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Reasoning across multiple documents for speech"
