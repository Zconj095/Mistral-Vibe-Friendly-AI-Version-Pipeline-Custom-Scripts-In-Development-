from __future__ import annotations

import json
import sqlite3
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
import urllib.error
import urllib.request

from pydantic import BaseModel, Field

try:
    from actions_lib import validate_args
except ModuleNotFoundError:  # Fallback when tools directory is not on sys.path.
    import importlib.util

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


DEFAULT_DB_PATH = Path.home() / ".vibe" / "vectorstores" / "pdf_vectors.sqlite"

VALID_VALIDATION = {"none", "llm"}
VALID_DECISIONS = {"answer", "ask_clarify", "refuse"}

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": sorted(VALID_DECISIONS)},
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "integer"},
                    "quote": {"type": "string"},
                },
                "required": ["source_id"],
                "additionalProperties": False,
            },
        },
        "uncertainties": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["decision", "answer", "confidence"],
    "additionalProperties": False,
}

VALIDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "valid": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "score": {"type": "number"},
    },
    "required": ["valid"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class _QueryResult:
    score: float
    source_path: str
    chunk_index: int
    unit: str
    start_index: int | None
    end_index: int | None
    content: str
    query: str


@dataclass(frozen=True)
class _ValidationResult:
    valid: bool
    issues: list[str]
    score: float | None
    raw: str | None


class EvidenceInput(BaseModel):
    source: str = Field(description="Evidence source label.")
    content: str = Field(description="Evidence text snippet.")
    id: int | None = Field(default=None, description="Optional source id.")
    score: float | None = Field(default=None, description="Optional similarity score.")
    query: str | None = Field(default=None, description="Query that retrieved it.")


class EvidenceItem(BaseModel):
    id: int
    source: str
    content: str
    score: float | None = None
    query: str | None = None


class HallucinationGuardrailsArgs(BaseModel):
    question: str = Field(description="User question to answer.")
    evidence: list[EvidenceInput] | None = Field(
        default=None, description="Inline evidence items."
    )
    include_resonance_quotes: bool = Field(
        default=True,
        description="Include resonance quote bank in system prompt.",
    )
    resonance_quotes_path: str | None = Field(
        default=None, description="Optional path to resonance quotes."
    )
    resonance_quotes_max_chars: int | None = Field(
        default=None, description="Max resonance quote chars to include."
    )
    include_retrieval: bool = Field(
        default=True, description="Retrieve evidence from vector store."
    )
    retrieval_queries: list[str] | None = Field(
        default=None, description="Override retrieval queries."
    )
    db_path: str | None = Field(default=None, description="Vector DB path.")
    embedding_model: str | None = Field(
        default=None, description="Embedding model name."
    )
    top_k: int | None = Field(default=None, description="Max retrieval results.")
    min_score: float | None = Field(
        default=None, description="Minimum similarity score."
    )
    max_result_chars: int | None = Field(
        default=None, description="Max chars per evidence snippet."
    )
    max_evidence: int | None = Field(
        default=None, description="Max evidence items to include."
    )
    require_citations: bool = Field(
        default=True, description="Require citations when answering."
    )
    allow_no_evidence: bool = Field(
        default=False, description="Allow answering without evidence."
    )
    no_evidence_decision: str | None = Field(
        default="ask_clarify",
        description="Decision to return when no evidence exists.",
    )
    output_schema: dict | str | None = Field(
        default=None, description="Override response schema."
    )
    strict_json: bool = Field(
        default=True, description="Require JSON-only output."
    )
    max_retries: int = Field(
        default=2, description="Max retries for schema/validation failures."
    )
    validation_mode: str | None = Field(
        default="none", description="none or llm."
    )
    llm_api_base: str | None = Field(
        default=None, description="OpenAI-compatible API base URL."
    )
    llm_model: str | None = Field(default=None, description="LLM model name.")
    llm_temperature: float = Field(default=0.2, description="LLM temperature.")
    llm_max_tokens: int = Field(default=700, description="LLM max tokens.")
    llm_stream: bool = Field(default=False, description="Stream LLM output.")
    validator_prompt: str | None = Field(
        default=None, description="Custom validator prompt."
    )
    validator_system_prompt: str | None = Field(
        default=None, description="Custom validator system prompt."
    )
    validator_temperature: float = Field(
        default=0.0, description="Validator temperature."
    )
    validator_max_tokens: int = Field(
        default=400, description="Validator max tokens."
    )


class HallucinationGuardrailsResult(BaseModel):
    decision: str | None
    answer: str | None
    citations: list[dict[str, Any]] | None
    confidence: str | None
    uncertainties: list[str] | None
    response: dict | None
    raw: str | None
    evidence: list[EvidenceItem]
    validation: dict | None
    attempts: int
    warnings: list[str]
    errors: list[str]


class HallucinationGuardrailsConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    db_path: Path = Field(
        default=DEFAULT_DB_PATH,
        description="Vector DB path.",
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
    max_results: int = Field(
        default=6, description="Max retrieval results."
    )
    max_result_chars: int = Field(
        default=800, description="Max evidence snippet chars."
    )
    max_evidence: int = Field(
        default=12, description="Max evidence items in prompt."
    )
    resonance_quotes_path: Path = Field(
        default=Path.home()
        / ".vibe"
        / "libraries"
        / "mistral_intelligence"
        / "resonance_quotes.txt",
        description="Default resonance quotes path.",
    )
    resonance_quotes_max_chars: int = Field(
        default=12000, description="Max resonance quote chars to include."
    )


class HallucinationGuardrailsState(BaseToolState):
    pass


class HallucinationGuardrails(
    BaseTool[
        HallucinationGuardrailsArgs,
        HallucinationGuardrailsResult,
        HallucinationGuardrailsConfig,
        HallucinationGuardrailsState,
    ],
    ToolUIData[HallucinationGuardrailsArgs, HallucinationGuardrailsResult],
):
    description: ClassVar[str] = (
        "Answer with evidence grounding, citations, and validation to reduce hallucinations."
    )

    async def run(
        self, args: HallucinationGuardrailsArgs
    ) -> HallucinationGuardrailsResult:
        question = args.question.strip()
        if not question:
            raise ToolError("question cannot be empty.")

        warnings: list[str] = []
        errors: list[str] = []

        validation_mode = self._normalize_validation_mode(args)
        schema = self._load_schema(args.output_schema)
        if schema is None:
            schema = RESPONSE_SCHEMA

        evidence = self._collect_evidence(args, warnings)
        evidence_ids = {item.id for item in evidence}

        if not evidence and not args.allow_no_evidence:
            response = self._no_evidence_response(args)
            return HallucinationGuardrailsResult(
                decision=response.get("decision"),
                answer=response.get("answer"),
                citations=response.get("citations"),
                confidence=response.get("confidence"),
                uncertainties=response.get("uncertainties"),
                response=response,
                raw=None,
                evidence=[],
                validation=None,
                attempts=0,
                warnings=warnings,
                errors=errors,
            )

        resonance_quotes = self._load_resonance_quotes(args, warnings)
        system_prompt = self._build_system_prompt(schema, resonance_quotes)
        messages = self._build_messages(system_prompt, question, evidence)

        attempts = 0
        raw: str | None = None
        parsed: dict | None = None
        validation_payload: dict | None = None
        last_error: str | None = None
        citation_errors: list[str] = []

        max_retries = max(0, int(args.max_retries))
        for attempt in range(max_retries + 1):
            attempts = attempt + 1
            if last_error and raw:
                messages = self._append_retry(messages, raw, last_error)

            raw = self._call_llm(messages, args)
            parsed, parse_error = self._parse_json(raw, args.strict_json)
            if parse_error:
                last_error = parse_error
                continue

            validation_errors = validate_args(schema, parsed)
            if validation_errors:
                last_error = "; ".join(validation_errors)
                continue

            citation_errors = self._check_citations(
                parsed,
                evidence_ids,
                args.require_citations,
                bool(evidence),
                args.allow_no_evidence,
            )
            if citation_errors:
                last_error = "; ".join(citation_errors)
                continue

            if validation_mode == "llm":
                llm_validation = self._validate_with_llm(
                    question, evidence, parsed, args
                )
                validation_payload = self._format_validation(
                    validation_mode, llm_validation
                )
                if not llm_validation.valid:
                    last_error = (
                        "; ".join(llm_validation.issues) or "Validation failed."
                    )
                    continue

            last_error = None
            break

        if last_error:
            errors.append("Validation failed after retries.")
            if citation_errors:
                errors.extend(citation_errors)

        decision = parsed.get("decision") if isinstance(parsed, dict) else None
        answer = parsed.get("answer") if isinstance(parsed, dict) else None
        citations = parsed.get("citations") if isinstance(parsed, dict) else None
        confidence = parsed.get("confidence") if isinstance(parsed, dict) else None
        uncertainties = parsed.get("uncertainties") if isinstance(parsed, dict) else None

        return HallucinationGuardrailsResult(
            decision=decision,
            answer=answer,
            citations=citations,
            confidence=confidence,
            uncertainties=uncertainties,
            response=parsed if isinstance(parsed, dict) else None,
            raw=raw,
            evidence=evidence,
            validation=validation_payload,
            attempts=attempts,
            warnings=warnings,
            errors=errors,
        )

    def _normalize_validation_mode(self, args: HallucinationGuardrailsArgs) -> str:
        mode = (args.validation_mode or "none").strip().lower()
        if mode not in VALID_VALIDATION:
            raise ToolError("validation_mode must be none or llm.")
        return mode

    def _load_resonance_quotes(
        self, args: HallucinationGuardrailsArgs, warnings: list[str]
    ) -> str | None:
        if not args.include_resonance_quotes:
            return None
        path_value = args.resonance_quotes_path or self.config.resonance_quotes_path
        if not path_value:
            return None
        path = Path(path_value).expanduser() if isinstance(path_value, (str, Path)) else None
        if path is None:
            return None
        if not path.is_absolute():
            path = self.config.effective_workdir / path
        path = path.resolve()
        if not path.exists():
            warnings.append(f"Resonance quotes file not found: {path}")
            return None
        if path.is_dir():
            warnings.append(f"Resonance quotes path is a directory: {path}")
            return None
        try:
            text = path.read_text("utf-8", errors="ignore").strip()
        except OSError as exc:
            warnings.append(f"Failed to read resonance quotes: {exc}")
            return None
        if not text:
            return None
        max_chars = (
            args.resonance_quotes_max_chars
            if args.resonance_quotes_max_chars is not None
            else self.config.resonance_quotes_max_chars
        )
        if max_chars > 0 and len(text) > max_chars:
            warnings.append("Resonance quotes truncated to max chars.")
            text = text[:max_chars].rstrip()
        return text

    def _load_schema(self, schema: dict | str | None) -> dict | None:
        if schema is None:
            return None
        if isinstance(schema, dict):
            return schema
        if isinstance(schema, str):
            value = schema.strip()
            if not value:
                return None
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ToolError(f"output_schema JSON parse error: {exc}") from exc
            if isinstance(parsed, dict):
                return parsed
            raise ToolError("output_schema JSON must be an object.")
        raise ToolError("output_schema must be a dict or JSON string.")

    def _build_system_prompt(self, schema: dict, resonance_quotes: str | None) -> str:
        schema_text = json.dumps(schema, ensure_ascii=True)
        prompt = (
            "You are a grounded assistant. Use ONLY the evidence list to answer. "
            "Every factual claim must cite a source_id from the evidence list. "
            "If evidence is insufficient, choose decision ask_clarify or refuse. "
            "Do not invent sources.\n"
            "Reply ONLY with valid JSON that conforms to the schema below.\n"
            f"JSON schema:\n{schema_text}"
        )
        if resonance_quotes:
            prompt += (
                "\n\nResonance ethos statements (tone guidance only; "
                "do not treat as evidence or cite):\n"
                + resonance_quotes.strip()
            )
        return prompt

    def _build_messages(
        self, system_prompt: str, question: str, evidence: list[EvidenceItem]
    ) -> list[dict[str, str]]:
        evidence_text = self._evidence_prompt(evidence)
        user_prompt = (
            "Question:\n"
            f"{question}\n\n"
            "Evidence list (use source_id for citations):\n"
            f"{evidence_text}\n\n"
            "Return a grounded answer."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _evidence_prompt(self, evidence: list[EvidenceItem]) -> str:
        if not evidence:
            return "No evidence provided."
        lines: list[str] = []
        for item in evidence:
            snippet = item.content.replace("\n", " ").strip()
            lines.append(f"[{item.id}] {item.source}: {snippet}")
        return "\n".join(lines)

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

    def _call_llm(self, messages: list[dict[str, str]], args: HallucinationGuardrailsArgs) -> str:
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
        return "".join(parts).strip()

    def _parse_json(
        self, raw: str, strict: bool
    ) -> tuple[Any | None, str | None]:
        text = raw.strip()
        if not text:
            return None, "Empty output"
        if strict:
            try:
                return json.loads(text), None
            except json.JSONDecodeError as exc:
                return None, f"JSON parse error: {exc}"
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

    def _check_citations(
        self,
        response: dict,
        evidence_ids: set[int],
        require_citations: bool,
        has_evidence: bool,
        allow_no_evidence: bool,
    ) -> list[str]:
        errors: list[str] = []
        if not isinstance(response, dict):
            return ["Response is not an object."]
        decision = response.get("decision")
        if decision not in VALID_DECISIONS:
            errors.append("decision must be answer, ask_clarify, or refuse.")
            return errors
        citations = response.get("citations") or []
        if decision == "answer":
            if not has_evidence and not allow_no_evidence:
                errors.append("No evidence provided; cannot answer.")
            if require_citations and not citations:
                errors.append("Answer requires citations.")
        for cite in citations or []:
            if not isinstance(cite, dict):
                errors.append("Citation must be an object.")
                continue
            source_id = cite.get("source_id")
            if source_id not in evidence_ids:
                errors.append(f"Citation source_id not found: {source_id}")
        return errors

    def _validate_with_llm(
        self,
        question: str,
        evidence: list[EvidenceItem],
        response: dict,
        args: HallucinationGuardrailsArgs,
    ) -> _ValidationResult:
        system_prompt = (
            args.validator_system_prompt
            or "You are a strict validator. Reply only with JSON."
        )
        schema_text = json.dumps(VALIDATION_SCHEMA, ensure_ascii=True)
        system_prompt = (
            f"{system_prompt}\nJSON schema for your reply:\n{schema_text}"
        )
        evidence_text = self._evidence_prompt(evidence)
        if args.validator_prompt and args.validator_prompt.strip():
            user_prompt = args.validator_prompt.strip()
        else:
            user_prompt = (
                "Validate the assistant response for grounding against evidence. "
                "Flag unsupported claims or missing citations.\n\n"
                f"Question:\n{question}\n\n"
                f"Evidence:\n{evidence_text}\n\n"
                f"Response JSON:\n{json.dumps(response, ensure_ascii=True)}"
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        raw = self._call_llm(
            messages,
            HallucinationGuardrailsArgs(
                question=question,
                llm_api_base=args.llm_api_base,
                llm_model=args.llm_model,
                llm_temperature=args.validator_temperature,
                llm_max_tokens=args.validator_max_tokens,
                llm_stream=False,
            ),
        )
        parsed, parse_error = self._parse_json(raw, strict=True)
        if parse_error:
            return _ValidationResult(
                valid=False,
                issues=[f"Validator JSON parse error: {parse_error}"],
                score=None,
                raw=raw,
            )
        if not isinstance(parsed, dict):
            return _ValidationResult(
                valid=False,
                issues=["Validator output was not an object."],
                score=None,
                raw=raw,
            )
        validation_errors = validate_args(VALIDATION_SCHEMA, parsed)
        if validation_errors:
            return _ValidationResult(
                valid=False,
                issues=validation_errors,
                score=None,
                raw=raw,
            )

        valid = bool(parsed.get("valid"))
        issues = [
            str(item)
            for item in (parsed.get("issues") or [])
            if isinstance(item, str) and item.strip()
        ]
        score_value = parsed.get("score")
        score = float(score_value) if isinstance(score_value, (int, float)) else None
        return _ValidationResult(
            valid=valid,
            issues=issues,
            score=score,
            raw=raw,
        )

    def _format_validation(
        self, mode: str, result: _ValidationResult | None
    ) -> dict | None:
        if result is None:
            return None
        payload = {
            "mode": mode,
            "valid": result.valid,
            "issues": result.issues,
        }
        if result.score is not None:
            payload["score"] = result.score
        if result.raw is not None:
            payload["raw"] = result.raw
        return payload

    def _collect_evidence(
        self, args: HallucinationGuardrailsArgs, warnings: list[str]
    ) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        if args.evidence:
            for idx, item in enumerate(args.evidence, start=1):
                if not item.content.strip():
                    warnings.append(f"evidence[{idx}] was empty.")
                    continue
                item_id = item.id or len(evidence) + 1
                evidence.append(
                    EvidenceItem(
                        id=item_id,
                        source=item.source,
                        content=item.content.strip(),
                        score=item.score,
                        query=item.query,
                    )
                )

        if args.include_retrieval:
            evidence.extend(self._retrieve_evidence(args, warnings, evidence))

        max_evidence = (
            args.max_evidence
            if args.max_evidence is not None
            else self.config.max_evidence
        )
        if max_evidence > 0 and len(evidence) > max_evidence:
            evidence = evidence[:max_evidence]
        return evidence

    def _retrieve_evidence(
        self,
        args: HallucinationGuardrailsArgs,
        warnings: list[str],
        existing: list[EvidenceItem],
    ) -> list[EvidenceItem]:
        queries = self._resolve_queries(args, warnings)
        if not queries:
            return []

        db_path = Path(args.db_path).expanduser() if args.db_path else self.config.db_path
        if not db_path.exists():
            warnings.append(f"Retrieval database not found: {db_path}")
            return []

        embedding_model = args.embedding_model or self.config.embedding_model
        top_k = args.top_k if args.top_k is not None else self.config.max_results
        if top_k <= 0:
            raise ToolError("top_k must be positive.")
        max_chars = (
            args.max_result_chars
            if args.max_result_chars is not None
            else self.config.max_result_chars
        )
        if max_chars <= 0:
            raise ToolError("max_result_chars must be positive.")

        min_score = args.min_score
        results: list[_QueryResult] = []
        for query in queries:
            results.extend(
                self._search_db(
                    db_path,
                    embedding_model,
                    query,
                    top_k,
                    min_score,
                    max_chars,
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        if len(results) > top_k:
            results = results[:top_k]

        existing_ids = {item.id for item in existing}
        evidence: list[EvidenceItem] = []
        for item in results:
            new_id = len(existing) + len(evidence) + 1
            while new_id in existing_ids:
                new_id += 1
            evidence.append(
                EvidenceItem(
                    id=new_id,
                    source=item.source_path,
                    content=item.content,
                    score=item.score,
                    query=item.query,
                )
            )
        return evidence

    def _resolve_queries(
        self, args: HallucinationGuardrailsArgs, warnings: list[str]
    ) -> list[str]:
        queries: list[str] = []
        if args.retrieval_queries:
            queries.extend(
                [q.strip() for q in args.retrieval_queries if q and q.strip()]
            )

        if not queries:
            if args.question.strip():
                queries.append(args.question.strip())
            else:
                warnings.append("No retrieval query available.")

        return queries

    def _search_db(
        self,
        db_path: Path,
        embedding_model: str,
        query: str,
        top_k: int,
        min_score: float | None,
        max_chars: int,
    ) -> list[_QueryResult]:
        query_vec = self._embed_text(embedding_model, query)

        results: list[tuple[float, _QueryResult]] = []
        with sqlite3.connect(str(db_path)) as conn:
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
                item = _QueryResult(
                    score=score,
                    source_path=row[0],
                    chunk_index=row[1],
                    unit=row[2],
                    start_index=row[3],
                    end_index=row[4],
                    content=str(row[5])[:max_chars],
                    query=query,
                )
                if len(results) < top_k:
                    results.append((score, item))
                    results.sort(key=lambda r: r[0])
                else:
                    if score > results[0][0]:
                        results[0] = (score, item)
                        results.sort(key=lambda r: r[0])

        return [item for _, item in sorted(results, key=lambda r: r[0], reverse=True)]

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
            raise ToolError(f"Ollama embeddings failed: {exc}") from exc

        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            raise ToolError("Invalid embeddings response from Ollama.")

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

    def _no_evidence_response(self, args: HallucinationGuardrailsArgs) -> dict[str, Any]:
        decision = (args.no_evidence_decision or "ask_clarify").strip().lower()
        if decision not in {"ask_clarify", "refuse"}:
            decision = "ask_clarify"
        answer = (
            "I don't have enough evidence to answer reliably. "
            "Please provide more context or sources."
        )
        return {
            "decision": decision,
            "answer": answer,
            "citations": [],
            "uncertainties": ["No evidence available."],
            "confidence": "low",
        }

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, HallucinationGuardrailsArgs):
            return ToolCallDisplay(summary="hallucination_guardrails")
        return ToolCallDisplay(
            summary="hallucination_guardrails",
            details={
                "question": event.args.question,
                "include_retrieval": event.args.include_retrieval,
                "evidence": len(event.args.evidence or []),
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, HallucinationGuardrailsResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = "Guardrails answer ready"
        if event.result.errors:
            message = "Guardrails completed with errors"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "decision": event.result.decision,
                "answer": event.result.answer,
                "citations": event.result.citations,
                "confidence": event.result.confidence,
                "uncertainties": event.result.uncertainties,
                "evidence": event.result.evidence,
                "validation": event.result.validation,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Applying hallucination guardrails"
