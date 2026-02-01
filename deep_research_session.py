
from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import time
from array import array
from dataclasses import dataclass
from datetime import datetime, timezone
from heapq import heappush, heappushpop
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
import urllib.error
import urllib.request

from pydantic import BaseModel, Field, field_validator

try:
    from actions_lib import validate_args
except ModuleNotFoundError:  # Fallback when tools directory is not on sys.path.
    _actions_path = Path(__file__).with_name("actions_lib.py")
    _spec = importlib.util.spec_from_file_location("actions_lib", _actions_path)
    if not _spec or not _spec.loader:
        raise
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    validate_args = _module.validate_args

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


PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "subquestions": {"type": "array", "items": {"type": "string"}},
        "queries": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
    "required": ["queries"],
    "additionalProperties": False,
}

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "integer"},
                    "source": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["source_id", "source"],
                "additionalProperties": False,
            },
        },
        "gaps": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string"},
    },
    "required": ["answer", "citations"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class _SearchHit:
    score: float
    source_path: str
    chunk_index: int
    unit: str
    start_index: int | None
    end_index: int | None
    content: str
    query: str


class DeepResearchItem(BaseModel):
    id: str | None = Field(default=None, description="Optional item id.")
    path: str | None = Field(default=None, description="Path to a text file.")
    content: str | None = Field(default=None, description="Inline text content.")
    source: str | None = Field(default=None, description="Source label.")


class EvidenceItem(BaseModel):
    id: int
    source_path: str
    chunk_index: int | None
    unit: str | None
    start_index: int | None
    end_index: int | None
    score: float | None
    query: str | None
    content: str


class DeepResearchSessionArgs(BaseModel):
    question: str = Field(description="Research question.")
    iterations: int = Field(default=1, description="Number of research iterations.")
    queries: list[str] | None = Field(default=None, description="Seed search queries.")
    plan: dict | None = Field(default=None, description="Optional prebuilt plan.")
    items: list[DeepResearchItem] | None = Field(
        default=None, description="Optional inline research items."
    )
    llm_api_base: str | None = Field(
        default=None, description="OpenAI-compatible API base URL."
    )
    llm_model: str | None = Field(default=None, description="LLM model name.")
    llm_temperature: float = Field(default=0.2, description="LLM temperature.")
    llm_max_tokens: int = Field(default=900, description="LLM max tokens.")
    llm_stream: bool = Field(default=False, description="Stream LLM output.")
    db_path: str | None = Field(default=None, description="Vector DB path.")
    embedding_model: str | None = Field(
        default=None, description="Embedding model name."
    )
    top_k: int | None = Field(default=None, description="Results per query.")
    min_score: float | None = Field(default=None, description="Minimum score.")
    max_result_chars: int | None = Field(
        default=None, description="Max chars per result."
    )
    max_evidence: int | None = Field(
        default=None, description="Max evidence items."
    )
    max_sources_for_answer: int | None = Field(
        default=None, description="Max sources passed to synthesis."
    )
    max_item_chars: int | None = Field(
        default=None, description="Max chars per inline item."
    )
    output_path: str | None = Field(
        default=None, description="Output session JSON path."
    )
    resume_path: str | None = Field(
        default=None, description="Resume from a prior session JSON."
    )
    include_plan: bool = Field(default=True, description="Include plan in output.")
    require_citations: bool = Field(
        default=True, description="Require citations in output."
    )
    max_retries: int | None = Field(
        default=None, description="Max retries for JSON output."
    )


class DeepResearchSessionResult(BaseModel):
    question: str
    plan: dict | None
    queries: list[str]
    evidence: list[EvidenceItem]
    answer: dict | None
    session_path: str | None
    warnings: list[str]
    errors: list[str]


class DeepResearchSessionConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    output_dir: Path = Field(
        default=Path.home() / ".vibe" / "memory" / "deep_research_sessions",
        description="Directory for research session outputs.",
    )
    db_path: Path = Field(
        default=Path.home() / ".vibe" / "vectorstores" / "pdf_vectors.sqlite",
        description="Default vector DB path.",
    )
    ollama_url: str = Field(
        default="http://127.0.0.1:11434",
        description="Ollama base URL for embeddings.",
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        description="Embedding model name.",
    )
    llm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL.",
    )
    llm_model: str = Field(
        default="gpt-oss:latest",
        description="Default LLM model name.",
    )
    max_results: int = Field(default=6, description="Default top_k per query.")
    max_result_chars: int = Field(default=1200, description="Max chars per result.")
    max_evidence: int = Field(default=30, description="Max evidence items.")
    max_sources_for_answer: int = Field(
        default=12, description="Max sources passed to synthesis."
    )
    max_item_chars: int = Field(default=2000, description="Max chars per item.")
    max_retries: int = Field(default=2, description="Max JSON retries.")
    min_score: float | None = Field(default=None, description="Min score filter.")

    @field_validator("output_dir", mode="before")
    @classmethod
    def set_default_output_dir(cls, v: Path | str) -> Path:
        if isinstance(v, Path):
            return v
        if not v or not str(v).strip():
            return Path.home() / ".vibe" / "memory" / "deep_research_sessions"
        return Path(v)

    @field_validator("output_dir", mode="after")
    @classmethod
    def expand_output_dir(cls, v: Path) -> Path:
        return v.expanduser().resolve()


class DeepResearchSessionState(BaseToolState):
    pass


class DeepResearchSession(
    BaseTool[
        DeepResearchSessionArgs,
        DeepResearchSessionResult,
        DeepResearchSessionConfig,
        DeepResearchSessionState,
    ],
    ToolUIData[DeepResearchSessionArgs, DeepResearchSessionResult],
):
    description: ClassVar[str] = (
        "Run a local deep research loop with offline retrieval and citations."
    )
    async def run(self, args: DeepResearchSessionArgs) -> DeepResearchSessionResult:
        question = args.question.strip()
        if not question:
            raise ToolError("question cannot be empty.")

        warnings: list[str] = []
        errors: list[str] = []

        plan: dict | None = args.plan
        evidence: list[EvidenceItem] = []
        seen: set[tuple[str, int | None, str]] = set()
        queries: list[str] = []
        created_at = datetime.now(timezone.utc).isoformat()
        updated_at = created_at

        session_data = None
        if args.resume_path:
            session_data = self._load_session(args.resume_path, warnings)
            if session_data:
                stored_question = session_data.get("question")
                if stored_question and stored_question != question:
                    warnings.append("resume_path question does not match new question.")
                if plan is None:
                    plan = session_data.get("plan")
                queries = self._normalize_queries(session_data.get("queries", []))
                evidence = self._load_evidence(session_data.get("evidence", []), warnings)
                seen = {
                    (item.source_path, item.chunk_index, item.content)
                    for item in evidence
                }
                created_at = session_data.get("created_at") or created_at

        if args.include_plan and plan is None:
            plan, plan_errors = self._generate_plan(question, args)
            warnings.extend(plan_errors)

        seed_queries = args.queries or (plan.get("queries") if isinstance(plan, dict) else None) or [question]
        queries = self._normalize_queries(queries + list(seed_queries))

        if args.items:
            evidence = self._add_inline_items(
                evidence,
                seen,
                args.items,
                args.max_item_chars or self.config.max_item_chars,
                warnings,
            )

        max_evidence = args.max_evidence or self.config.max_evidence
        top_k = args.top_k if args.top_k is not None else self.config.max_results
        max_chars = (
            args.max_result_chars
            if args.max_result_chars is not None
            else self.config.max_result_chars
        )
        min_score = args.min_score if args.min_score is not None else self.config.min_score
        db_path = args.db_path or str(self.config.db_path)
        embedding_model = args.embedding_model or self.config.embedding_model

        pending_queries = list(seed_queries)
        iterations = max(1, int(args.iterations))

        for iteration in range(iterations):
            current_queries = self._normalize_queries(pending_queries)
            pending_queries = []
            if not current_queries:
                break
            queries = self._normalize_queries(queries + current_queries)
            for query in current_queries:
                if len(evidence) >= max_evidence:
                    break
                try:
                    hits = self._search_vector_db(
                        query,
                        db_path,
                        embedding_model,
                        top_k,
                        min_score,
                        max_chars,
                    )
                except ToolError as exc:
                    warnings.append(str(exc))
                    continue
                evidence = self._add_hits(evidence, seen, hits, max_evidence)
                if len(evidence) >= max_evidence:
                    break
            if iteration < iterations - 1:
                followup, followup_errors = self._generate_followup_queries(
                    question,
                    evidence,
                    args,
                )
                warnings.extend(followup_errors)
                pending_queries = followup

        answer = None
        if evidence:
            answer, answer_errors = self._synthesize_answer(question, evidence, args)
            errors.extend(answer_errors)
        elif args.require_citations:
            errors.append("No evidence retrieved; cannot synthesize with citations.")

        updated_at = datetime.now(timezone.utc).isoformat()
        session_path = self._resolve_output_path(args.output_path, args.resume_path)
        self._save_session(
            session_path,
            {
                "question": question,
                "created_at": created_at,
                "updated_at": updated_at,
                "plan": plan,
                "queries": queries,
                "evidence": [item.model_dump() for item in evidence],
                "answer": answer,
                "warnings": warnings,
                "errors": errors,
            },
            warnings,
        )

        return DeepResearchSessionResult(
            question=question,
            plan=plan if args.include_plan else None,
            queries=queries,
            evidence=evidence,
            answer=answer,
            session_path=str(session_path) if session_path else None,
            warnings=warnings,
            errors=errors,
        )

    def _generate_plan(
        self, question: str, args: DeepResearchSessionArgs
    ) -> tuple[dict, list[str]]:
        errors: list[str] = []
        system_prompt = (
            "You are a research planner working with a local document library. "
            "Create a short plan and a small set of search queries."
        )
        user_prompt = (
            "Question:\n"
            f"{question}\n\n"
            "Return 3-6 concise search queries that can be used to look up local documents. "
            "Include subquestions if helpful."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        plan, call_errors = self._call_llm_json(PLAN_SCHEMA, messages, args)
        errors.extend(call_errors)
        if not isinstance(plan, dict):
            fallback = {"queries": [question]}
            errors.append("Planner failed; using fallback query.")
            return fallback, errors
        if "queries" not in plan:
            plan["queries"] = [question]
        return plan, errors

    def _generate_followup_queries(
        self,
        question: str,
        evidence: list[EvidenceItem],
        args: DeepResearchSessionArgs,
    ) -> tuple[list[str], list[str]]:
        if not evidence:
            return [], []
        errors: list[str] = []
        max_sources = (
            args.max_sources_for_answer
            if args.max_sources_for_answer is not None
            else self.config.max_sources_for_answer
        )
        selected = self._select_evidence(evidence, max_sources)
        evidence_prompt = self._build_evidence_prompt(selected, max_chars=240)
        system_prompt = (
            "You are a research assistant. Suggest follow-up search queries to fill gaps or verify claims. "
            "If no follow-up is needed, return an empty list of queries."
        )
        user_prompt = (
            "Question:\n"
            f"{question}\n\n"
            "Evidence snippets:\n"
            f"{evidence_prompt}\n\n"
            "Return 0-5 follow-up search queries."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        plan, call_errors = self._call_llm_json(PLAN_SCHEMA, messages, args)
        errors.extend(call_errors)
        if not isinstance(plan, dict):
            return [], errors
        queries = self._normalize_queries(plan.get("queries", []))
        return queries, errors

    def _synthesize_answer(
        self,
        question: str,
        evidence: list[EvidenceItem],
        args: DeepResearchSessionArgs,
    ) -> tuple[dict | None, list[str]]:
        errors: list[str] = []
        max_sources = (
            args.max_sources_for_answer
            if args.max_sources_for_answer is not None
            else self.config.max_sources_for_answer
        )
        selected = self._select_evidence(evidence, max_sources)
        evidence_prompt = self._build_evidence_prompt(selected, max_chars=300)
        system_prompt = (
            "You are a research assistant. Answer the question using ONLY the evidence snippets provided. "
            "Cite sources by source_id from the evidence list. Do not invent sources."
        )
        user_prompt = (
            "Question:\n"
            f"{question}\n\n"
            "Evidence list (use source_id for citations):\n"
            f"{evidence_prompt}\n\n"
            "Provide a concise answer with citations, plus any gaps and a confidence rating."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response, call_errors = self._call_llm_json(ANSWER_SCHEMA, messages, args)
        errors.extend(call_errors)
        if not isinstance(response, dict):
            return None, errors

        citations = response.get("citations", []) if isinstance(response, dict) else []
        if args.require_citations and not citations:
            errors.append("Synthesis returned no citations.")

        valid_ids = {item.id for item in evidence}
        for cite in citations or []:
            if not isinstance(cite, dict):
                continue
            source_id = cite.get("source_id")
            if source_id not in valid_ids:
                errors.append(f"Citation source_id not found in evidence: {source_id}")
        return response, errors

    def _search_vector_db(
        self,
        query: str,
        db_path: str,
        embedding_model: str,
        top_k: int,
        min_score: float | None,
        max_chars: int,
    ) -> list[_SearchHit]:
        if top_k <= 0:
            raise ToolError("top_k must be a positive integer.")
        if max_chars <= 0:
            raise ToolError("max_result_chars must be a positive integer.")
        if not query.strip():
            return []

        resolved = Path(db_path).expanduser()
        if not resolved.is_absolute():
            resolved = self.config.effective_workdir / resolved
        resolved = resolved.resolve()
        if not resolved.exists():
            raise ToolError(f"Database not found at: {resolved}")

        query_vec = self._embed_text(embedding_model, query)

        results: list[tuple[float, _SearchHit]] = []
        try:
            with sqlite3.connect(str(resolved)) as conn:
                cursor = conn.execute(
                    """
                    SELECT source_path, chunk_index, unit, start_index, end_index, content, embedding
                    FROM pdf_chunks
                    """
                )
                for row in cursor:
                    embedding = self._unpack_embedding(row[6])
                    if not embedding:
                        continue
                    score = self._dot(query_vec, embedding)
                    if min_score is not None and score < min_score:
                        continue
                    content = row[5] or ""
                    snippet = content[:max_chars]
                    hit = _SearchHit(
                        score=score,
                        source_path=row[0],
                        chunk_index=row[1],
                        unit=row[2],
                        start_index=row[3],
                        end_index=row[4],
                        content=snippet,
                        query=query,
                    )
                    if len(results) < top_k:
                        heappush(results, (score, hit))
                    else:
                        heappushpop(results, (score, hit))
        except sqlite3.Error as exc:
            raise ToolError(f"Database error: {exc}") from exc

        return [hit for _, hit in sorted(results, key=lambda r: r[0], reverse=True)]

    def _add_hits(
        self,
        evidence: list[EvidenceItem],
        seen: set[tuple[str, int | None, str]],
        hits: list[_SearchHit],
        max_evidence: int,
    ) -> list[EvidenceItem]:
        for hit in hits:
            if len(evidence) >= max_evidence:
                break
            key = (hit.source_path, hit.chunk_index, hit.content)
            if key in seen:
                continue
            seen.add(key)
            evidence.append(
                EvidenceItem(
                    id=len(evidence) + 1,
                    source_path=hit.source_path,
                    chunk_index=hit.chunk_index,
                    unit=hit.unit,
                    start_index=hit.start_index,
                    end_index=hit.end_index,
                    score=hit.score,
                    query=hit.query,
                    content=hit.content,
                )
            )
        return evidence

    def _add_inline_items(
        self,
        evidence: list[EvidenceItem],
        seen: set[tuple[str, int | None, str]],
        items: list[DeepResearchItem],
        max_item_chars: int,
        warnings: list[str],
    ) -> list[EvidenceItem]:
        for idx, item in enumerate(items, start=1):
            if item.content and item.path:
                warnings.append(f"items[{idx}]: Provide content or path, not both.")
                continue
            content = None
            source_path = None
            if item.path:
                path = Path(item.path).expanduser()
                if not path.is_absolute():
                    path = self.config.effective_workdir / path
                path = path.resolve()
                if not path.exists() or path.is_dir():
                    warnings.append(f"items[{idx}]: Path not found: {path}")
                    continue
                try:
                    content = path.read_text("utf-8", errors="ignore")
                    source_path = str(path)
                except OSError as exc:
                    warnings.append(f"items[{idx}]: Failed to read {path}: {exc}")
                    continue
            else:
                content = item.content
                source_path = item.source or item.id or f"inline_{idx}"

            if content is None:
                warnings.append(f"items[{idx}]: Missing content.")
                continue

            snippet = content
            if max_item_chars > 0 and len(snippet) > max_item_chars:
                snippet = snippet[:max_item_chars]
                warnings.append(
                    f"items[{idx}]: Content truncated to {max_item_chars} characters."
                )

            key = (source_path, None, snippet)
            if key in seen:
                continue
            seen.add(key)
            evidence.append(
                EvidenceItem(
                    id=len(evidence) + 1,
                    source_path=source_path,
                    chunk_index=None,
                    unit="inline",
                    start_index=None,
                    end_index=None,
                    score=None,
                    query=None,
                    content=snippet,
                )
            )
        return evidence

    def _normalize_queries(self, queries: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for query in queries or []:
            if query is None:
                continue
            text = str(query).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(text)
        return normalized

    def _load_session(self, path: str, warnings: list[str]) -> dict | None:
        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            resolved = self.config.effective_workdir / resolved
        resolved = resolved.resolve()
        if not resolved.exists():
            warnings.append(f"resume_path not found: {resolved}")
            return None
        try:
            payload = json.loads(resolved.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"Failed to read session: {exc}")
            return None
        if not isinstance(payload, dict):
            warnings.append("Session file must contain a JSON object.")
            return None
        return payload

    def _load_evidence(self, items: list[dict], warnings: list[str]) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        if not isinstance(items, list):
            return evidence
        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                warnings.append(f"evidence[{idx}]: Invalid evidence entry.")
                continue
            try:
                evidence.append(EvidenceItem(**item))
            except Exception as exc:
                warnings.append(f"evidence[{idx}]: {exc}")
        return evidence

    def _save_session(
        self, path: Path | None, payload: dict, warnings: list[str]
    ) -> None:
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), "utf-8")
        except OSError as exc:
            warnings.append(f"Failed to write session: {exc}")

    def _resolve_output_path(
        self, output_path: str | None, resume_path: str | None
    ) -> Path | None:
        if output_path is not None:
            text = output_path.strip()
            if not text:
                return None
            resolved = Path(text).expanduser()
            if not resolved.is_absolute():
                resolved = self.config.effective_workdir / resolved
            return resolved.resolve()
        if resume_path:
            resolved = Path(resume_path).expanduser()
            if not resolved.is_absolute():
                resolved = self.config.effective_workdir / resolved
            return resolved.resolve()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"deep_research_{timestamp}_{int(time.time())}.json"
        return self.config.output_dir / filename

    def _call_llm(
        self, messages: list[dict[str, str]], args: DeepResearchSessionArgs
    ) -> str:
        api_base = (args.llm_api_base or self.config.llm_api_base).rstrip("/")
        url = api_base + "/chat/completions"
        payload = {
            "model": args.llm_model or self.config.llm_model,
            "messages": messages,
            "temperature": args.llm_temperature,
            "max_tokens": args.llm_max_tokens,
            "stream": bool(args.llm_stream),
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=600)
        except urllib.error.URLError as exc:
            raise ToolError(f"LLM request failed: {exc}") from exc

        if not args.llm_stream:
            body = resp.read().decode("utf-8")
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError as exc:
                raise ToolError(f"LLM response parse failed: {exc}") from exc
            return parsed["choices"][0]["message"].get("content", "").strip()

        return self._read_streaming_response(resp)

    def _read_streaming_response(self, resp) -> str:
        parts: list[str] = []
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            if line.startswith("data:"):
                line = line[len("data:") :].strip()
            if line == "[DONE]":
                break
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            choice = chunk.get("choices", [{}])[0]
            delta = choice.get("delta") or choice.get("message") or {}
            content = delta.get("content")
            if content:
                parts.append(content)
                sys.stdout.write(content)
                sys.stdout.flush()
        if parts:
            sys.stdout.write("\n")
            sys.stdout.flush()
        return "".join(parts).strip()

    def _call_llm_json(
        self,
        schema: dict,
        messages: list[dict[str, str]],
        args: DeepResearchSessionArgs,
    ) -> tuple[dict | None, list[str]]:
        errors: list[str] = []
        max_retries = (
            args.max_retries if args.max_retries is not None else self.config.max_retries
        )
        schema_text = json.dumps(schema, ensure_ascii=True)
        system_prompt = (
            "Reply ONLY with valid JSON that conforms to the JSON schema below. "
            "Do not include extra keys, comments, or markdown.\n"
            f"JSON schema:\n{schema_text}"
        )
        full_messages = [{"role": "system", "content": system_prompt}] + list(messages)

        for attempt in range(max_retries + 1):
            raw = self._call_llm(full_messages, args)
            parsed, parse_error = self._parse_json(raw)
            if parse_error:
                errors.append(parse_error)
                if attempt < max_retries:
                    full_messages = self._append_retry(full_messages, raw, parse_error)
                    continue
                return None, errors

            validation_errors = self._validate_json(schema, parsed)
            if validation_errors:
                errors.extend(validation_errors)
                if attempt < max_retries:
                    full_messages = self._append_retry(
                        full_messages, raw, "; ".join(validation_errors)
                    )
                    continue
                return None, errors

            return parsed, errors
        return None, errors

    def _append_retry(
        self, messages: list[dict[str, str]], raw: str, error: str
    ) -> list[dict[str, str]]:
        retry_prompt = (
            "Your previous output was invalid.\n"
            f"Error: {error}\n"
            "Reply again with ONLY valid JSON that matches the schema."
        )
        return messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": retry_prompt},
        ]

    def _parse_json(self, raw: str) -> tuple[Any | None, str | None]:
        text = raw.strip()
        if not text:
            return None, "Empty output"
        try:
            return json.loads(text), None
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            start = min(
                [i for i in (text.find("{"), text.find("[")) if i != -1],
                default=-1,
            )
            if start == -1:
                return None, "No JSON object or array found"
            try:
                obj, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError as exc:
                return None, f"JSON parse error: {exc}"
            return obj, None

    def _validate_json(self, schema: dict, value: Any) -> list[str]:
        return validate_args(schema, value)

    def _embed_text(self, model: str, text: str) -> list[float]:
        payload = json.dumps({"model": model, "prompt": text}).encode("utf-8")
        url = self.config.ollama_url.rstrip("/") + "/api/embeddings"
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise ToolError(f"Ollama/GPT-OSS embeddings failed: {exc}") from exc

        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            raise ToolError("Invalid embeddings response from Ollama/GPT-OSS.")

        return self._normalize_embedding([float(x) for x in embedding])

    def _normalize_embedding(self, embedding: list[float]) -> list[float]:
        norm = sum(x * x for x in embedding) ** 0.5
        if norm == 0:
            return embedding
        return [x / norm for x in embedding]

    def _unpack_embedding(self, blob: bytes) -> list[float]:
        arr = array("f")
        arr.frombytes(blob)
        return list(arr)

    def _dot(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            return 0.0
        return sum(l * r for l, r in zip(left, right))

    def _select_evidence(
        self, evidence: list[EvidenceItem], limit: int
    ) -> list[EvidenceItem]:
        if limit <= 0:
            return []
        return sorted(
            evidence,
            key=lambda item: item.score if item.score is not None else float("-inf"),
            reverse=True,
        )[:limit]

    def _build_evidence_prompt(
        self, evidence: list[EvidenceItem], max_chars: int
    ) -> str:
        lines: list[str] = []
        for item in evidence:
            snippet = item.content.replace("\n", " ").strip()
            if max_chars > 0 and len(snippet) > max_chars:
                snippet = snippet[:max_chars] + "..."
            score = f"{item.score:.4f}" if item.score is not None else "n/a"
            lines.append(
                f"[{item.id}] {item.source_path} (score: {score}): {snippet}"
            )
        return "\n".join(lines)

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, DeepResearchSessionArgs):
            return ToolCallDisplay(summary="deep_research_session")
        query_count = len(event.args.queries or [])
        item_count = len(event.args.items or [])
        return ToolCallDisplay(
            summary="deep_research_session",
            details={
                "question": event.args.question,
                "iterations": event.args.iterations,
                "queries": query_count,
                "items": item_count,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, DeepResearchSessionResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        evidence_count = len(event.result.evidence)
        has_answer = event.result.answer is not None
        message = f"Deep research complete ({evidence_count} evidence)"
        if not has_answer:
            message = f"Deep research complete ({evidence_count} evidence, no answer)"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "question": event.result.question,
                "plan": event.result.plan,
                "queries": event.result.queries,
                "evidence": event.result.evidence,
                "answer": event.result.answer,
                "session_path": event.result.session_path,
                "warnings": event.result.warnings,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Running deep research session"
