from __future__ import annotations

import functools
import importlib.util
import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

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


class ContextVectorSpeechCrossrefConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    ollama_url: str = Field(
        default="http://127.0.0.1:11434",
        description="Base URL for the Ollama/GPT-OSS server.",
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        description="Default embedding model to use with Ollama/GPT-OSS.",
    )
    max_stores: int = Field(default=5000, description="Maximum stores to scan.")
    max_results_per_store: int = Field(
        default=500, description="Maximum results to return per store."
    )
    max_total_results: int = Field(
        default=2000, description="Maximum total results to return."
    )
    max_result_chars: int = Field(
        default=2000, description="Maximum characters to return per result."
    )
    score_mode: str = Field(default="rank", description="rank, raw, or combined.")
    preview_chars: int = Field(default=240, description="Preview snippet length.")
    min_token_length: int = Field(default=3, description="Minimum token length.")
    max_keywords: int = Field(default=8, description="Max keywords per result.")
    min_shared_keywords: int = Field(
        default=2, description="Minimum shared keywords for crossrefs."
    )
    min_crossref_score: float = Field(
        default=0.15, description="Minimum crossref score."
    )
    max_crossrefs: int = Field(default=200, description="Maximum crossrefs to return.")
    max_crossrefs_per_result: int = Field(
        default=4, description="Maximum crossrefs per result."
    )
    max_speech_segments: int = Field(
        default=25, description="Maximum speech segments."
    )
    max_speech_crossrefs: int = Field(
        default=20, description="Maximum speech crossref cues."
    )
    default_aggregate_mode: str = Field(
        default="chunk", description="chunk or document."
    )
    max_documents: int = Field(
        default=2000, description="Maximum documents for document aggregation."
    )
    max_doc_chunks: int = Field(
        default=4, description="Maximum chunks per document used for keywords."
    )


class ContextVectorSpeechCrossrefState(BaseToolState):
    pass


class EmbeddingStore(BaseModel):
    path: str = Field(description="Path to a sqlite DB or manifest file.")
    store_type: str | None = Field(
        default="db", description="db, manifest, or sharded."
    )
    embedding_model: str | None = Field(
        default=None, description="Override embedding model for this store."
    )
    label: str | None = Field(default=None, description="Optional store label.")
    min_score: float | None = Field(default=None, description="Minimum similarity score.")
    max_results: int | None = Field(
        default=None, description="Override max results for this store."
    )
    max_result_chars: int | None = Field(
        default=None, description="Override max chars per result for this store."
    )


class ContextVectorSpeechCrossrefArgs(BaseModel):
    query: str = Field(description="Search query.")
    stores: list[EmbeddingStore] | None = Field(
        default=None, description="Embedding stores to search."
    )
    catalog_path: str | None = Field(
        default=None, description="Optional catalog file listing stores."
    )
    catalog_format: str | None = Field(
        default="auto", description="auto, json, jsonl, csv, tsv."
    )
    max_stores: int | None = Field(
        default=None, description="Override max_stores."
    )
    max_results_per_store: int | None = Field(
        default=None, description="Override max_results_per_store."
    )
    max_total_results: int | None = Field(
        default=None, description="Override max_total_results."
    )
    max_result_chars: int | None = Field(
        default=None, description="Override max_result_chars."
    )
    min_score: float | None = Field(default=None, description="Minimum similarity score.")
    score_mode: str | None = Field(default=None, description="rank, raw, or combined.")
    aggregate_mode: str | None = Field(
        default=None, description="chunk or document."
    )
    min_token_length: int | None = Field(
        default=None, description="Override minimum token length."
    )
    max_keywords: int | None = Field(
        default=None, description="Override max keywords per result."
    )
    min_shared_keywords: int | None = Field(
        default=None, description="Override minimum shared keywords."
    )
    min_crossref_score: float | None = Field(
        default=None, description="Override minimum crossref score."
    )
    max_crossrefs: int | None = Field(
        default=None, description="Override max crossrefs."
    )
    max_crossrefs_per_result: int | None = Field(
        default=None, description="Override max crossrefs per result."
    )
    max_speech_segments: int | None = Field(
        default=None, description="Override max speech segments."
    )
    max_speech_crossrefs: int | None = Field(
        default=None, description="Override max speech crossref cues."
    )
    max_documents: int | None = Field(
        default=None, description="Override max documents for aggregation."
    )
    max_doc_chunks: int | None = Field(
        default=None, description="Override max chunks per document for aggregation."
    )
    include_opening: bool = Field(
        default=True, description="Include speech opening."
    )
    include_closing: bool = Field(
        default=True, description="Include speech closing."
    )


class VectorSpeechResultItem(BaseModel):
    index: int
    score: float
    rank_score: float
    store_rank: int
    store_label: str
    store_path: str
    store_type: str
    embedding_model: str
    source_path: str
    chunk_index: int
    unit: str
    start_index: int | None
    end_index: int | None
    content: str
    shard_path: str | None
    preview: str
    keywords: list[str]


class DocumentSpeechResultItem(BaseModel):
    index: int
    document_id: str
    store_label: str
    store_path: str
    store_type: str
    embedding_model: str
    source_path: str
    chunk_count: int
    max_score: float
    avg_score: float
    top_chunk_indices: list[int]
    preview: str
    keywords: list[str]


class StoreSummary(BaseModel):
    label: str
    store_path: str
    store_type: str
    embedding_model: str
    result_count: int
    truncated: bool
    errors: list[str]


class VectorCrossref(BaseModel):
    source_index: int
    source_label: str
    target_index: int
    target_label: str
    source_path: str | None = None
    target_path: str | None = None
    score: float
    shared_keywords: list[str]


class SpeechSegment(BaseModel):
    index: int
    result_index: int
    store_label: str
    source_path: str
    preview: str
    keywords: list[str]
    speech_prompt: str


class SpeechCrossrefCue(BaseModel):
    index: int
    source_index: int
    target_index: int
    shared_keywords: list[str]
    cue: str


class ContextVectorSpeechCrossrefResult(BaseModel):
    query: str
    aggregate_mode: str
    results: list[VectorSpeechResultItem]
    count: int
    documents: list[DocumentSpeechResultItem]
    document_count: int
    document_truncated: bool
    store_count: int
    truncated: bool
    store_summaries: list[StoreSummary]
    crossrefs: list[VectorCrossref]
    speech_opening: str
    speech_segments: list[SpeechSegment]
    speech_crossrefs: list[SpeechCrossrefCue]
    speech_closing: str
    warnings: list[str]
    errors: list[str]


class _CrossrefItem(BaseModel):
    index: int
    label: str
    source_path: str | None
    keywords: list[str]


class ContextVectorSpeechCrossref(
    BaseTool[
        ContextVectorSpeechCrossrefArgs,
        ContextVectorSpeechCrossrefResult,
        ContextVectorSpeechCrossrefConfig,
        ContextVectorSpeechCrossrefState,
    ],
    ToolUIData[ContextVectorSpeechCrossrefArgs, ContextVectorSpeechCrossrefResult],
):
    description: ClassVar[str] = (
        "Search vector stores, cross-reference results, and prepare speech cues."
    )

    async def run(
        self, args: ContextVectorSpeechCrossrefArgs
    ) -> ContextVectorSpeechCrossrefResult:
        query = args.query.strip()
        if not query:
            raise ToolError("query cannot be empty.")

        aggregate_mode = (
            args.aggregate_mode or self.config.default_aggregate_mode
        ).strip().lower()
        if aggregate_mode not in {"chunk", "document"}:
            raise ToolError("aggregate_mode must be chunk or document.")

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
        min_shared_keywords = (
            args.min_shared_keywords
            if args.min_shared_keywords is not None
            else self.config.min_shared_keywords
        )
        min_crossref_score = (
            args.min_crossref_score
            if args.min_crossref_score is not None
            else self.config.min_crossref_score
        )
        max_crossrefs = (
            args.max_crossrefs
            if args.max_crossrefs is not None
            else self.config.max_crossrefs
        )
        max_crossrefs_per_result = (
            args.max_crossrefs_per_result
            if args.max_crossrefs_per_result is not None
            else self.config.max_crossrefs_per_result
        )
        max_speech_segments = (
            args.max_speech_segments
            if args.max_speech_segments is not None
            else self.config.max_speech_segments
        )
        max_speech_crossrefs = (
            args.max_speech_crossrefs
            if args.max_speech_crossrefs is not None
            else self.config.max_speech_crossrefs
        )
        max_documents = (
            args.max_documents
            if args.max_documents is not None
            else self.config.max_documents
        )
        max_doc_chunks = (
            args.max_doc_chunks
            if args.max_doc_chunks is not None
            else self.config.max_doc_chunks
        )

        if min_token_length <= 0:
            raise ToolError("min_token_length must be positive.")
        if max_keywords < 0:
            raise ToolError("max_keywords must be non-negative.")
        if min_shared_keywords <= 0:
            raise ToolError("min_shared_keywords must be positive.")
        if min_crossref_score < 0:
            raise ToolError("min_crossref_score must be non-negative.")
        if max_crossrefs < 0:
            raise ToolError("max_crossrefs must be non-negative.")
        if max_crossrefs_per_result < 0:
            raise ToolError("max_crossrefs_per_result must be non-negative.")
        if max_speech_segments < 0:
            raise ToolError("max_speech_segments must be non-negative.")
        if max_speech_crossrefs < 0:
            raise ToolError("max_speech_crossrefs must be non-negative.")
        if max_documents < 0:
            raise ToolError("max_documents must be non-negative.")
        if max_doc_chunks < 0:
            raise ToolError("max_doc_chunks must be non-negative.")
        if max_documents == 0:
            max_documents = None

        search_module = self._load_search_module()
        search_config = search_module.VectorSearchFederatedConfig(
            ollama_url=self.config.ollama_url,
            embedding_model=self.config.embedding_model,
            max_stores=self.config.max_stores,
            max_results_per_store=self.config.max_results_per_store,
            max_total_results=self.config.max_total_results,
            max_result_chars=self.config.max_result_chars,
            score_mode=self.config.score_mode,
            workdir=self.config.effective_workdir,
        )
        search_tool = search_module.VectorSearchFederated.from_config(search_config)

        search_stores = None
        if args.stores:
            search_stores = [
                search_module.EmbeddingStore(**store.model_dump())
                for store in args.stores
            ]

        search_args = search_module.VectorSearchFederatedArgs(
            query=query,
            stores=search_stores,
            catalog_path=args.catalog_path,
            catalog_format=args.catalog_format,
            max_stores=args.max_stores,
            max_results_per_store=args.max_results_per_store,
            max_total_results=args.max_total_results,
            max_result_chars=args.max_result_chars,
            min_score=args.min_score,
            score_mode=args.score_mode,
        )

        search_result = await search_tool.run(search_args)

        warnings: list[str] = []
        errors = list(search_result.errors)
        if search_result.truncated:
            warnings.append("Search results truncated by limits.")

        results: list[VectorSpeechResultItem] = []
        for idx, item in enumerate(search_result.results, start=1):
            keywords = self._keywords(
                item.content, min_token_length=min_token_length, max_keywords=max_keywords
            )
            preview = self._preview(item.content, self.config.preview_chars)
            results.append(
                VectorSpeechResultItem(
                    index=idx,
                    score=item.score,
                    rank_score=item.rank_score,
                    store_rank=item.store_rank,
                    store_label=item.store_label,
                    store_path=item.store_path,
                    store_type=item.store_type,
                    embedding_model=item.embedding_model,
                    source_path=item.source_path,
                    chunk_index=item.chunk_index,
                    unit=item.unit,
                    start_index=item.start_index,
                    end_index=item.end_index,
                    content=item.content,
                    shard_path=item.shard_path,
                    preview=preview,
                    keywords=keywords,
                )
            )

        if not results:
            warnings.append("No results found.")

        store_summaries = [
            StoreSummary(
                label=summary.label,
                store_path=summary.store_path,
                store_type=summary.store_type,
                embedding_model=summary.embedding_model,
                result_count=summary.result_count,
                truncated=summary.truncated,
                errors=summary.errors,
            )
            for summary in search_result.store_summaries
        ]

        documents: list[DocumentSpeechResultItem] = []
        document_truncated = False
        crossref_items: list[_CrossrefItem] = []
        if aggregate_mode == "document":
            documents, document_truncated = self._aggregate_documents(
                results=results,
                max_documents=max_documents,
                max_doc_chunks=max_doc_chunks,
                min_token_length=min_token_length,
                max_keywords=max_keywords,
            )
            crossref_items = [
                _CrossrefItem(
                    index=item.index,
                    label=item.store_label,
                    source_path=item.source_path,
                    keywords=item.keywords,
                )
                for item in documents
            ]
            speech_segments, segments_truncated = self._build_speech_segments(
                documents, max_speech_segments
            )
        else:
            crossref_items = [
                _CrossrefItem(
                    index=item.index,
                    label=item.store_label,
                    source_path=item.source_path,
                    keywords=item.keywords,
                )
                for item in results
            ]
            speech_segments, segments_truncated = self._build_speech_segments(
                results, max_speech_segments
            )

        if segments_truncated:
            warnings.append("Speech segments truncated by limits.")

        crossrefs, crossref_truncated = self._build_crossrefs(
            items=crossref_items,
            min_shared_keywords=min_shared_keywords,
            min_crossref_score=min_crossref_score,
            max_crossrefs=max_crossrefs,
            max_crossrefs_per_result=max_crossrefs_per_result,
        )
        if crossref_truncated:
            warnings.append("Cross-reference list truncated by limits.")

        speech_crossrefs, speech_crossref_truncated = self._build_speech_crossrefs(
            crossrefs, max_speech_crossrefs
        )
        if speech_crossref_truncated:
            warnings.append("Speech cross-reference cues truncated by limits.")

        speech_opening = self._speech_opening(
            args,
            query,
            search_result.store_count,
            len(results),
            aggregate_mode,
            len(documents),
        )
        speech_closing = self._speech_closing(
            args, aggregate_mode, len(results), len(documents), len(crossrefs)
        )

        return ContextVectorSpeechCrossrefResult(
            query=query,
            aggregate_mode=aggregate_mode,
            results=results,
            count=len(results),
            documents=documents,
            document_count=len(documents),
            document_truncated=document_truncated,
            store_count=search_result.store_count,
            truncated=search_result.truncated,
            store_summaries=store_summaries,
            crossrefs=crossrefs,
            speech_opening=speech_opening,
            speech_segments=speech_segments,
            speech_crossrefs=speech_crossrefs,
            speech_closing=speech_closing,
            warnings=warnings,
            errors=errors,
        )

    @staticmethod
    @functools.cache
    def _load_search_module() -> Any:
        module_path = Path(__file__).with_name("vector_search_federated.py")
        if not module_path.exists():
            raise ToolError("vector_search_federated.py not found.")
        spec = importlib.util.spec_from_file_location(
            "vibe.tools.vector_search_federated", module_path
        )
        if spec is None or spec.loader is None:
            raise ToolError("Failed to load vector_search_federated module.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _build_crossrefs(
        self,
        items: list[_CrossrefItem],
        min_shared_keywords: int,
        min_crossref_score: float,
        max_crossrefs: int,
        max_crossrefs_per_result: int,
    ) -> tuple[list[VectorCrossref], bool]:
        if max_crossrefs == 0 or max_crossrefs_per_result == 0:
            return [], False

        pairs: list[tuple[float, int, int, list[str]]] = []
        keywords_sets = [set(item.keywords) for item in items]
        for i in range(len(items)):
            if not keywords_sets[i]:
                continue
            for j in range(i + 1, len(items)):
                if not keywords_sets[j]:
                    continue
                shared = sorted(keywords_sets[i] & keywords_sets[j])
                if len(shared) < min_shared_keywords:
                    continue
                union_size = len(keywords_sets[i] | keywords_sets[j])
                score = len(shared) / union_size if union_size else 0.0
                if score < min_crossref_score:
                    continue
                pairs.append((score, i, j, shared))

        pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
        crossrefs: list[VectorCrossref] = []
        counts = Counter()
        for score, i, j, shared in pairs:
            if counts[i] >= max_crossrefs_per_result:
                continue
            if counts[j] >= max_crossrefs_per_result:
                continue
            crossrefs.append(
                VectorCrossref(
                    source_index=items[i].index,
                    source_label=items[i].label,
                    target_index=items[j].index,
                    target_label=items[j].label,
                    source_path=items[i].source_path,
                    target_path=items[j].source_path,
                    score=score,
                    shared_keywords=shared,
                )
            )
            counts[i] += 1
            counts[j] += 1
            if max_crossrefs and len(crossrefs) >= max_crossrefs:
                break

        truncated = len(pairs) > len(crossrefs)
        return crossrefs, truncated

    def _aggregate_documents(
        self,
        results: list[VectorSpeechResultItem],
        max_documents: int | None,
        max_doc_chunks: int,
        min_token_length: int,
        max_keywords: int,
    ) -> tuple[list[DocumentSpeechResultItem], bool]:
        groups: dict[
            tuple[str, str, str, str, str], list[VectorSpeechResultItem]
        ] = {}
        for item in results:
            key = (
                item.store_label,
                item.store_path,
                item.store_type,
                item.embedding_model,
                item.source_path,
            )
            groups.setdefault(key, []).append(item)

        doc_entries: list[dict[str, Any]] = []
        for (
            store_label,
            store_path,
            store_type,
            embedding_model,
            source_path,
        ), items in groups.items():
            items.sort(key=lambda item: (-item.score, -item.rank_score))
            chunk_count = len(items)
            if max_doc_chunks == 0:
                selected_chunks = items
            else:
                selected_chunks = items[:max_doc_chunks]
            combined_text = " ".join(chunk.content for chunk in selected_chunks)
            keywords = self._keywords(
                combined_text,
                min_token_length=min_token_length,
                max_keywords=max_keywords,
            )
            preview = ""
            if selected_chunks:
                preview = selected_chunks[0].preview or self._preview(
                    selected_chunks[0].content, self.config.preview_chars
                )
            scores = [chunk.score for chunk in items]
            max_score = max(scores) if scores else 0.0
            avg_score = sum(scores) / len(scores) if scores else 0.0
            top_chunk_indices = [chunk.chunk_index for chunk in selected_chunks]
            document_id = f"{store_label}:{source_path}"
            doc_entries.append(
                {
                    "document_id": document_id,
                    "store_label": store_label,
                    "store_path": store_path,
                    "store_type": store_type,
                    "embedding_model": embedding_model,
                    "source_path": source_path,
                    "chunk_count": chunk_count,
                    "max_score": max_score,
                    "avg_score": avg_score,
                    "top_chunk_indices": top_chunk_indices,
                    "preview": preview,
                    "keywords": keywords,
                }
            )

        doc_entries.sort(
            key=lambda entry: (
                -entry["max_score"],
                -entry["avg_score"],
                entry["store_label"],
                entry["source_path"],
            )
        )
        truncated = False
        if max_documents is not None and len(doc_entries) > max_documents:
            truncated = True
            doc_entries = doc_entries[:max_documents]

        documents: list[DocumentSpeechResultItem] = []
        for idx, entry in enumerate(doc_entries, start=1):
            documents.append(DocumentSpeechResultItem(index=idx, **entry))
        return documents, truncated

    def _build_speech_segments(
        self,
        items: list[VectorSpeechResultItem | DocumentSpeechResultItem],
        max_segments: int,
    ) -> tuple[list[SpeechSegment], bool]:
        if max_segments == 0:
            return [], False
        truncated = max_segments and len(items) > max_segments
        segments: list[SpeechSegment] = []
        for idx, item in enumerate(items[:max_segments], start=1):
            keywords = ", ".join(item.keywords[:6])
            prompt_parts = [f"Speak about {item.source_path} ({item.store_label})."]
            if keywords:
                prompt_parts.append(f"Emphasize: {keywords}.")
            if item.preview:
                prompt_parts.append(f"Context: {item.preview}")
            segments.append(
                SpeechSegment(
                    index=idx,
                    result_index=item.index,
                    store_label=item.store_label,
                    source_path=item.source_path,
                    preview=item.preview,
                    keywords=item.keywords,
                    speech_prompt=" ".join(prompt_parts).strip(),
                )
            )
        return segments, truncated

    def _build_speech_crossrefs(
        self, crossrefs: list[VectorCrossref], max_crossrefs: int
    ) -> tuple[list[SpeechCrossrefCue], bool]:
        if max_crossrefs == 0:
            return [], False
        truncated = max_crossrefs and len(crossrefs) > max_crossrefs
        cues: list[SpeechCrossrefCue] = []
        for idx, crossref in enumerate(crossrefs[:max_crossrefs], start=1):
            keywords = ", ".join(crossref.shared_keywords[:6])
            cue = (
                f"Connect result {crossref.source_index} with {crossref.target_index}"
            )
            if keywords:
                cue = f"{cue} using shared keywords: {keywords}."
            cues.append(
                SpeechCrossrefCue(
                    index=idx,
                    source_index=crossref.source_index,
                    target_index=crossref.target_index,
                    shared_keywords=crossref.shared_keywords,
                    cue=cue,
                )
            )
        return cues, truncated

    def _speech_opening(
        self,
        args: ContextVectorSpeechCrossrefArgs,
        query: str,
        store_count: int,
        count: int,
        aggregate_mode: str,
        document_count: int,
    ) -> str:
        if not args.include_opening:
            return ""
        query_text = self._clip_text(query, 120)
        parts = [f"Begin speaking about results for '{query_text}'."]
        if store_count:
            parts.append(f"Found {count} result(s) across {store_count} store(s).")
        if aggregate_mode == "document" and document_count:
            parts.append(f"Summarize {document_count} document(s) with the strongest matches.")
        return " ".join(parts)

    def _speech_closing(
        self,
        args: ContextVectorSpeechCrossrefArgs,
        aggregate_mode: str,
        count: int,
        document_count: int,
        crossref_count: int,
    ) -> str:
        if not args.include_closing:
            return ""
        if count == 0:
            return "Conclude by asking if more sources should be searched."
        if crossref_count:
            return "Close by summarizing the strongest shared themes."
        if aggregate_mode == "document" and document_count:
            return "Close by reinforcing the document-level takeaways."
        return "Close by summarizing the most relevant matches."

    def _keywords(self, text: str, min_token_length: int, max_keywords: int) -> list[str]:
        if max_keywords <= 0:
            return []
        tokens = [
            token.lower()
            for token in WORD_RE.findall(text)
            if len(token) >= min_token_length and token.lower() not in STOPWORDS
        ]
        if not tokens:
            return []
        counts = Counter(tokens)
        return [token for token, _ in counts.most_common(max_keywords)]

    def _preview(self, text: str, max_chars: int) -> str:
        return self._clip_text(text, max_chars)

    def _clip_text(self, text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        trimmed = " ".join(text.strip().split())
        if len(trimmed) <= max_chars:
            return trimmed
        return trimmed[:max_chars]

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextVectorSpeechCrossrefArgs):
            return ToolCallDisplay(summary="context_vector_speech_crossref")
        store_count = len(event.args.stores or [])
        summary = "context_vector_speech_crossref"
        if event.args.query:
            summary = f"context_vector_speech_crossref: {event.args.query}"
        return ToolCallDisplay(
            summary=summary,
            details={
                "store_count": store_count,
                "catalog_path": event.args.catalog_path,
                "max_total_results": event.args.max_total_results,
                "min_score": event.args.min_score,
                "aggregate_mode": event.args.aggregate_mode,
                "max_documents": event.args.max_documents,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextVectorSpeechCrossrefResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = (
            f"Prepared {event.result.count} result(s) with "
            f"{len(event.result.crossrefs)} crossref(s)"
        )
        warnings = event.result.warnings[:]
        if event.result.errors:
            warnings.extend(event.result.errors)
        if event.result.truncated:
            warnings.append("Search results truncated by limits")
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=warnings,
            details={
                "count": event.result.count,
                "document_count": event.result.document_count,
                "crossref_count": len(event.result.crossrefs),
                "store_count": event.result.store_count,
                "truncated": event.result.truncated,
                "document_truncated": event.result.document_truncated,
                "aggregate_mode": event.result.aggregate_mode,
                "store_summaries": event.result.store_summaries,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Cross referencing vector search results for speech"
