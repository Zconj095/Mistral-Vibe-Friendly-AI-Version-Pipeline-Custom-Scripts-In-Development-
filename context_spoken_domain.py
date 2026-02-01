from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import ClassVar, TYPE_CHECKING

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

DEFAULT_DOMAINS: list[dict[str, object]] = [
    {
        "term": "General",
        "description": "Plain language for broad audiences and mixed contexts.",
        "markers": [],
    },
    {
        "term": "Technical",
        "description": "Technical engineering and software language.",
        "markers": [
            "system",
            "api",
            "server",
            "runtime",
            "deploy",
            "compile",
            "model",
            "algorithm",
            "performance",
            "database",
        ],
    },
    {
        "term": "Medical",
        "description": "Clinical and healthcare vocabulary.",
        "markers": [
            "patient",
            "diagnosis",
            "symptom",
            "treatment",
            "therapy",
            "clinical",
            "dosage",
            "procedure",
            "pathology",
        ],
    },
    {
        "term": "Legal",
        "description": "Legal and regulatory language.",
        "markers": [
            "contract",
            "statute",
            "compliance",
            "liability",
            "jurisdiction",
            "agreement",
            "clause",
            "plaintiff",
            "defendant",
        ],
    },
    {
        "term": "Finance",
        "description": "Financial, accounting, and business terminology.",
        "markers": [
            "revenue",
            "cost",
            "margin",
            "budget",
            "forecast",
            "balance",
            "asset",
            "liability",
            "cashflow",
        ],
    },
    {
        "term": "Gaming",
        "description": "Game design, gameplay, and player experience language.",
        "markers": [
            "player",
            "level",
            "quest",
            "npc",
            "loot",
            "combat",
            "mechanic",
            "balance",
            "progression",
        ],
    },
    {
        "term": "Education",
        "description": "Instructional, teaching, and learning-focused language.",
        "markers": [
            "lesson",
            "exercise",
            "objective",
            "curriculum",
            "student",
            "study",
            "practice",
            "explain",
        ],
    },
    {
        "term": "Customer Support",
        "description": "Support and troubleshooting language with clear steps.",
        "markers": [
            "issue",
            "ticket",
            "resolve",
            "troubleshoot",
            "steps",
            "reproduce",
            "support",
            "escalate",
        ],
    },
]


class ContextSpokenDomainConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_source_bytes: int = Field(
        default=5_000_000, description="Maximum bytes per content."
    )
    max_segments: int = Field(default=200, description="Maximum segments to return.")
    preview_chars: int = Field(default=200, description="Preview snippet length.")
    default_segment_by: str = Field(
        default="sentences", description="sentences, lines, or paragraphs."
    )
    min_segment_chars: int = Field(default=12, description="Minimum segment length.")
    min_token_length: int = Field(default=2, description="Minimum token length.")
    max_keywords: int = Field(default=6, description="Max keywords per segment.")
    max_domains: int = Field(default=12, description="Maximum domains to score.")
    max_blend_domains: int = Field(
        default=5, description="Maximum domains in blend summary."
    )
    max_domain_reasoning: int = Field(
        default=6, description="Maximum cross-domain reasoning entries."
    )
    default_assign_mode: str = Field(
        default="single", description="single, per_segment, or sequence."
    )


class ContextSpokenDomainState(BaseToolState):
    pass


class ContextSpokenDomainArgs(BaseModel):
    content: str | None = Field(default=None, description="Content to analyze.")
    path: str | None = Field(default=None, description="Path to content.")
    domain: str | None = Field(default=None, description="Explicit speaking domain.")
    domains: list[str] | None = Field(
        default=None, description="Domain options to choose from."
    )
    domain_sequence: list[str] | None = Field(
        default=None, description="Explicit domain sequence to cycle."
    )
    assign_mode: str | None = Field(
        default=None, description="single, per_segment, or sequence."
    )
    max_blend_domains: int | None = Field(
        default=None, description="Override max blend domains."
    )
    max_domain_reasoning: int | None = Field(
        default=None, description="Override max domain reasoning entries."
    )
    segment_by: str | None = Field(
        default=None, description="sentences, lines, or paragraphs."
    )
    max_segments: int | None = Field(
        default=None, description="Override max segments."
    )
    include_opening: bool = Field(
        default=True, description="Include opening cue."
    )
    include_closing: bool = Field(
        default=True, description="Include closing cue."
    )


class DomainDefinition(BaseModel):
    term: str
    normalized_term: str
    description: str
    markers: list[str]


class DomainScore(BaseModel):
    term: str
    score: int
    matched_markers: list[str]
    description: str


class DomainInterpretation(BaseModel):
    term: str
    description: str
    matched_markers: list[str]
    score: int
    rationale: list[str]


class DomainBlend(BaseModel):
    term: str
    description: str
    score: int
    weight: float
    matched_markers: list[str]


class DomainReasoningItem(BaseModel):
    primary_domain: str
    secondary_domain: str
    primary_weight: float
    secondary_weight: float
    shared_markers: list[str]
    primary_markers: list[str]
    secondary_markers: list[str]
    rationale: list[str]


class SpokenDomainSegment(BaseModel):
    index: int
    text: str
    domain: str
    word_count: int
    keywords: list[str]
    cue: str


class ContextSpokenDomainResult(BaseModel):
    domain: str
    domain_description: str
    domain_prompt: str
    assign_mode: str
    domain_sequence: list[str]
    domain_interpretation: DomainInterpretation
    domain_interpretations: list[DomainInterpretation]
    domain_blend: list[DomainBlend]
    blend_summary: str
    domain_reasoning: list[DomainReasoningItem]
    reasoning_summary: str
    segments: list[SpokenDomainSegment]
    segment_count: int
    domain_scores: list[DomainScore]
    speech_opening: str
    speech_closing: str
    warnings: list[str]


class ContextSpokenDomain(
    BaseTool[
        ContextSpokenDomainArgs,
        ContextSpokenDomainResult,
        ContextSpokenDomainConfig,
        ContextSpokenDomainState,
    ],
    ToolUIData[ContextSpokenDomainArgs, ContextSpokenDomainResult],
):
    description: ClassVar[str] = "Select a speaking domain and build domain cues."

    async def run(self, args: ContextSpokenDomainArgs) -> ContextSpokenDomainResult:
        definitions = self._load_definitions(args)
        if not definitions:
            raise ToolError("No speaking domains available.")

        assign_mode = (
            args.assign_mode or self.config.default_assign_mode
        ).strip().lower()
        if args.domain_sequence:
            assign_mode = "sequence"
        if assign_mode not in {"single", "per_segment", "sequence"}:
            raise ToolError("assign_mode must be single, per_segment, or sequence.")

        content = None
        if args.content or args.path:
            content = self._load_content(args)

        selected, scores, warnings = self._select_domain(
            args, definitions, content
        )
        domain_sequence = self._resolve_domain_sequence(args, definitions, selected, warnings)
        segments = self._build_segments(
            args,
            selected,
            definitions,
            domain_sequence,
            assign_mode,
            content,
            warnings,
        )

        if assign_mode == "single":
            domain_prompt = self._build_domain_prompt(selected)
        else:
            domain_prompt = self._build_domain_prompt_multi(domain_sequence)

        interpretations = self._build_domain_interpretations(
            selected,
            domain_sequence,
            scores,
            content,
            assign_mode,
        )
        primary_interpretation = next(
            (item for item in interpretations if item.term == selected.term),
            DomainInterpretation(
                term=selected.term,
                description=selected.description,
                matched_markers=[],
                score=0,
                rationale=["Defaulted to selected domain."],
            ),
        )
        domain_blend, blend_summary = self._build_domain_blend(
            scores,
            selected,
            args.max_blend_domains,
            content,
        )
        domain_reasoning, reasoning_summary = self._build_domain_reasoning(
            domain_blend,
            scores,
            args.max_domain_reasoning,
            content,
        )

        speech_opening = self._speech_opening(args, selected, domain_sequence, assign_mode)
        speech_closing = self._speech_closing(args)

        return ContextSpokenDomainResult(
            domain=selected.term,
            domain_description=selected.description,
            domain_prompt=domain_prompt,
            assign_mode=assign_mode,
            domain_sequence=[definition.term for definition in domain_sequence],
            domain_interpretation=primary_interpretation,
            domain_interpretations=interpretations,
            domain_blend=domain_blend,
            blend_summary=blend_summary,
            domain_reasoning=domain_reasoning,
            reasoning_summary=reasoning_summary,
            segments=segments,
            segment_count=len(segments),
            domain_scores=scores,
            speech_opening=speech_opening,
            speech_closing=speech_closing,
            warnings=warnings,
        )

    def _load_definitions(self, args: ContextSpokenDomainArgs) -> list[DomainDefinition]:
        definitions: list[DomainDefinition] = []
        for entry in DEFAULT_DOMAINS:
            term = self._normalize_term(str(entry.get("term", "")))
            description = self._normalize_description(str(entry.get("description", "")))
            markers = [
                self._normalize_term(str(marker)).lower()
                for marker in entry.get("markers", [])
                if marker
            ]
            if not term:
                continue
            definitions.append(
                DomainDefinition(
                    term=term,
                    normalized_term=term.lower(),
                    description=description or term,
                    markers=markers,
                )
            )

        if args.domains:
            definitions = self._filter_domains(args.domains, definitions)

        max_domains = self.config.max_domains
        if max_domains > 0 and len(definitions) > max_domains:
            definitions = definitions[:max_domains]
        return definitions

    def _filter_domains(
        self, names: list[str], definitions: list[DomainDefinition]
    ) -> list[DomainDefinition]:
        lookup = {definition.normalized_term: definition for definition in definitions}
        filtered: list[DomainDefinition] = []
        for name in names:
            normalized = self._normalize_term(name).lower()
            if normalized in lookup:
                filtered.append(lookup[normalized])
                continue
            filtered.append(
                DomainDefinition(
                    term=self._normalize_term(name),
                    normalized_term=normalized,
                    description="Custom speaking domain.",
                    markers=[],
                )
            )
        return filtered

    def _select_domain(
        self,
        args: ContextSpokenDomainArgs,
        definitions: list[DomainDefinition],
        content: str | None,
    ) -> tuple[DomainDefinition, list[DomainScore], list[str]]:
        warnings: list[str] = []
        if args.domain:
            normalized = self._normalize_term(args.domain).lower()
            for definition in definitions:
                if definition.normalized_term == normalized:
                    return definition, [], warnings
            return (
                DomainDefinition(
                    term=self._normalize_term(args.domain),
                    normalized_term=normalized,
                    description="Custom speaking domain.",
                    markers=[],
                ),
                [],
                warnings,
            )

        scores: list[DomainScore] = []
        if content:
            scores = self._score_domains(content, definitions)
            if scores:
                best = max(scores, key=lambda item: item.score)
                if best.score > 0:
                    selected = next(
                        definition
                        for definition in definitions
                        if definition.term == best.term
                    )
                    return selected, scores, warnings
                warnings.append("No domain markers matched; using default domain.")

        selected = definitions[0]
        return selected, scores, warnings

    def _resolve_domain_sequence(
        self,
        args: ContextSpokenDomainArgs,
        definitions: list[DomainDefinition],
        fallback: DomainDefinition,
        warnings: list[str],
    ) -> list[DomainDefinition]:
        if args.domain_sequence:
            return self._sequence_from_names(args.domain_sequence, definitions, warnings)
        if args.domains:
            return self._sequence_from_names(args.domains, definitions, warnings)
        if definitions:
            return definitions
        warnings.append("No domain sequence available; using fallback.")
        return [fallback]

    def _sequence_from_names(
        self,
        names: list[str],
        definitions: list[DomainDefinition],
        warnings: list[str],
    ) -> list[DomainDefinition]:
        lookup = {definition.normalized_term: definition for definition in definitions}
        sequence: list[DomainDefinition] = []
        for name in names:
            normalized = self._normalize_term(name).lower()
            definition = lookup.get(normalized)
            if definition is None:
                warnings.append(f"Unknown domain in sequence: {name}")
                definition = DomainDefinition(
                    term=self._normalize_term(name),
                    normalized_term=normalized,
                    description="Custom speaking domain.",
                    markers=[],
                )
            sequence.append(definition)
        return sequence

    def _score_domains(
        self, content: str, definitions: list[DomainDefinition]
    ) -> list[DomainScore]:
        tokens = [
            token.lower()
            for token in WORD_RE.findall(content)
            if len(token) >= self.config.min_token_length
        ]
        counts = Counter(tokens)
        scores: list[DomainScore] = []
        for definition in definitions:
            matched: list[str] = []
            score = 0
            for marker in definition.markers:
                hits = counts.get(marker, 0)
                if hits:
                    matched.append(marker)
                    score += hits
            scores.append(
                DomainScore(
                    term=definition.term,
                    score=score,
                    matched_markers=matched,
                    description=definition.description,
                )
            )
        scores.sort(key=lambda item: (-item.score, item.term))
        return scores

    def _build_segments(
        self,
        args: ContextSpokenDomainArgs,
        domain: DomainDefinition,
        definitions: list[DomainDefinition],
        domain_sequence: list[DomainDefinition],
        assign_mode: str,
        content: str | None,
        warnings: list[str],
    ) -> list[SpokenDomainSegment]:
        if not content:
            if assign_mode != "single":
                warnings.append("No content provided; no segments generated.")
            return []
        segment_by = (args.segment_by or self.config.default_segment_by).strip().lower()
        if segment_by not in {"sentences", "lines", "paragraphs"}:
            raise ToolError("segment_by must be sentences, lines, or paragraphs.")
        max_segments = (
            args.max_segments if args.max_segments is not None else self.config.max_segments
        )
        if max_segments <= 0:
            raise ToolError("max_segments must be positive.")

        segments_raw = self._split_segments(content, segment_by)
        segments: list[SpokenDomainSegment] = []
        if assign_mode == "sequence" and not domain_sequence:
            warnings.append("Domain sequence missing; falling back to single domain.")
            assign_mode = "single"

        for idx, text in enumerate(segments_raw, start=1):
            if len(segments) >= max_segments:
                break
            if len(text) < self.config.min_segment_chars:
                continue
            if assign_mode == "sequence":
                assigned = domain_sequence[(idx - 1) % len(domain_sequence)]
            elif assign_mode == "per_segment":
                assigned = self._select_segment_domain(text, definitions, domain)
            else:
                assigned = domain
            keywords = self._extract_keywords(text, self.config.max_keywords)
            cue = self._build_segment_cue(assigned, keywords)
            segments.append(
                SpokenDomainSegment(
                    index=len(segments) + 1,
                    text=text.strip(),
                    domain=assigned.term,
                    word_count=len(WORD_RE.findall(text)),
                    keywords=keywords,
                    cue=cue,
                )
            )
        return segments

    def _select_segment_domain(
        self,
        text: str,
        definitions: list[DomainDefinition],
        fallback: DomainDefinition,
    ) -> DomainDefinition:
        scores = self._score_domains(text, definitions)
        if not scores:
            return fallback
        best = max(scores, key=lambda item: item.score)
        if best.score <= 0:
            return fallback
        for definition in definitions:
            if definition.term == best.term:
                return definition
        return fallback

    def _extract_keywords(self, text: str, max_items: int) -> list[str]:
        tokens = []
        for token in WORD_RE.findall(text):
            lower = token.lower()
            if len(lower) < self.config.min_token_length:
                continue
            if lower in STOPWORDS:
                continue
            tokens.append(lower)
        return [word for word, _ in Counter(tokens).most_common(max_items)]

    def _split_segments(self, text: str, mode: str) -> list[str]:
        if mode == "lines":
            return [line for line in text.splitlines() if line.strip()]
        if mode == "paragraphs":
            return [chunk for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
        return [seg.strip() for seg in SENTENCE_RE.findall(text) if seg.strip()]

    def _build_domain_prompt(self, domain: DomainDefinition) -> str:
        lines = [
            f"Speaking domain: {domain.term}.",
            f"Domain guidance: {domain.description}",
            "Use domain-appropriate vocabulary and keep terminology consistent.",
        ]
        return " ".join(lines)

    def _build_domain_prompt_multi(self, domains: list[DomainDefinition]) -> str:
        if not domains:
            return "Speaking domains: General. Use consistent vocabulary per domain."
        lines = ["Speaking domains:"]
        for domain in domains:
            lines.append(f"- {domain.term}: {domain.description}")
        lines.append("Keep vocabulary consistent within each domain.")
        return " ".join(lines)

    def _build_domain_interpretations(
        self,
        selected: DomainDefinition,
        domain_sequence: list[DomainDefinition],
        scores: list[DomainScore],
        content: str | None,
        assign_mode: str,
    ) -> list[DomainInterpretation]:
        score_map = {score.term: score for score in scores}
        interpretations: list[DomainInterpretation] = []
        targets = domain_sequence or [selected]
        for definition in targets:
            score_info = score_map.get(definition.term)
            matched = score_info.matched_markers if score_info else []
            score_value = score_info.score if score_info else 0
            rationale: list[str] = []
            if content is None:
                rationale.append("No content provided; interpreted by definition.")
            elif score_value > 0:
                rationale.append(
                    "Matched markers: " + ", ".join(matched) if matched else "Matched markers present."
                )
            else:
                rationale.append("No markers matched; defaulted to domain.")

            if assign_mode == "sequence":
                rationale.append("Sequence mode applies domains by order.")
            elif assign_mode == "per_segment":
                rationale.append("Segment mode selects domains per segment.")

            interpretations.append(
                DomainInterpretation(
                    term=definition.term,
                    description=definition.description,
                    matched_markers=matched,
                    score=score_value,
                    rationale=rationale,
                )
            )
        return interpretations

    def _build_domain_blend(
        self,
        scores: list[DomainScore],
        selected: DomainDefinition,
        max_blend_domains: int | None,
        content: str | None,
    ) -> tuple[list[DomainBlend], str]:
        limit = max_blend_domains if max_blend_domains is not None else self.config.max_blend_domains
        if limit is None or limit <= 0:
            limit = 0
        if not scores:
            blend = [
                DomainBlend(
                    term=selected.term,
                    description=selected.description,
                    score=0,
                    weight=1.0,
                    matched_markers=[],
                )
            ]
            return blend, f"Domain blend: {selected.term} (1.00)."

        ordered = sorted(scores, key=lambda item: (-item.score, item.term))
        total_score = sum(item.score for item in ordered)
        if total_score <= 0:
            blend = [
                DomainBlend(
                    term=selected.term,
                    description=selected.description,
                    score=0,
                    weight=1.0,
                    matched_markers=[],
                )
            ]
            summary = (
                f"Domain blend: {selected.term} (1.00)."
                if content
                else f"Domain blend: {selected.term} (1.00, no content)."
            )
            return blend, summary

        selected_terms = ordered if limit == 0 else ordered[:limit]
        blend: list[DomainBlend] = []
        for item in selected_terms:
            weight = round(item.score / total_score, 4)
            blend.append(
                DomainBlend(
                    term=item.term,
                    description=item.description,
                    score=item.score,
                    weight=weight,
                    matched_markers=item.matched_markers,
                )
            )

        summary_parts = [f"{item.term} ({item.weight:.2f})" for item in blend]
        summary = "Domain blend: " + ", ".join(summary_parts) + "."
        return blend, summary

    def _build_domain_reasoning(
        self,
        domain_blend: list[DomainBlend],
        scores: list[DomainScore],
        max_domain_reasoning: int | None,
        content: str | None,
    ) -> tuple[list[DomainReasoningItem], str]:
        limit = (
            max_domain_reasoning
            if max_domain_reasoning is not None
            else self.config.max_domain_reasoning
        )
        if limit is None or limit <= 0:
            limit = 0

        if len(domain_blend) < 2:
            if domain_blend:
                summary = f"Domain reasoning: {domain_blend[0].term} only."
            else:
                summary = "Domain reasoning: no domains available."
            return [], summary

        score_map = {score.term: score for score in scores}
        reasoning: list[DomainReasoningItem] = []

        for i in range(len(domain_blend)):
            for j in range(i + 1, len(domain_blend)):
                if limit and len(reasoning) >= limit:
                    break
                primary = domain_blend[i]
                secondary = domain_blend[j]
                primary_score = score_map.get(primary.term)
                secondary_score = score_map.get(secondary.term)
                primary_markers = primary_score.matched_markers if primary_score else []
                secondary_markers = secondary_score.matched_markers if secondary_score else []
                shared = sorted(set(primary_markers) & set(secondary_markers))
                primary_only = sorted(set(primary_markers) - set(secondary_markers))
                secondary_only = sorted(set(secondary_markers) - set(primary_markers))

                rationale: list[str] = []
                if content is None:
                    rationale.append("No content; reasoning uses domain definitions.")
                elif shared:
                    rationale.append("Shared markers suggest overlapping subtopics.")
                if primary.score > secondary.score:
                    rationale.append(f"{primary.term} is primary; {secondary.term} supports it.")
                elif secondary.score > primary.score:
                    rationale.append(f"{secondary.term} is primary; {primary.term} supports it.")
                else:
                    rationale.append("Domains are balanced in evidence.")
                if content is not None and not shared and (not primary_markers or not secondary_markers):
                    rationale.append("Limited marker evidence for one domain.")

                reasoning.append(
                    DomainReasoningItem(
                        primary_domain=primary.term,
                        secondary_domain=secondary.term,
                        primary_weight=primary.weight,
                        secondary_weight=secondary.weight,
                        shared_markers=shared,
                        primary_markers=primary_only,
                        secondary_markers=secondary_only,
                        rationale=rationale,
                    )
                )
            if limit and len(reasoning) >= limit:
                break

        if not reasoning:
            summary = "Domain reasoning: no cross-domain pairs available."
        else:
            pairs = [
                f"{item.primary_domain}+{item.secondary_domain}"
                for item in reasoning
            ]
            summary = "Domain reasoning pairs: " + ", ".join(pairs) + "."
        return reasoning, summary

    def _build_segment_cue(
        self, domain: DomainDefinition, keywords: list[str]
    ) -> str:
        if keywords:
            return (
                f"{domain.term} domain; emphasize {', '.join(keywords[:5])}."
            )
        return f"{domain.term} domain; keep vocabulary consistent."

    def _speech_opening(
        self,
        args: ContextSpokenDomainArgs,
        domain: DomainDefinition,
        domain_sequence: list[DomainDefinition],
        assign_mode: str,
    ) -> str:
        if not args.include_opening:
            return ""
        if assign_mode == "single":
            return f"Begin speaking in the {domain.term} domain."
        names = ", ".join(item.term for item in domain_sequence if item.term)
        if not names:
            names = domain.term
        return f"Begin speaking across these domains: {names}."

    def _speech_closing(self, args: ContextSpokenDomainArgs) -> str:
        if not args.include_closing:
            return ""
        return "Close by keeping the speaking domain consistent."

    def _load_content(self, args: ContextSpokenDomainArgs) -> str:
        if args.content and args.path:
            raise ToolError("Provide content or path, not both.")
        if args.content is not None:
            data = args.content.encode("utf-8")
            if len(data) > self.config.max_source_bytes:
                raise ToolError(
                    f"content exceeds max_source_bytes ({len(data)} > {self.config.max_source_bytes})."
                )
            return args.content
        if not args.path:
            raise ToolError("Provide content or path.")
        path = self._resolve_path(args.path)
        size = path.stat().st_size
        if size > self.config.max_source_bytes:
            raise ToolError(
                f"{path} exceeds max_source_bytes ({size} > {self.config.max_source_bytes})."
            )
        return path.read_text("utf-8", errors="ignore")

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.config.effective_workdir / path
        path = path.resolve()
        if not path.exists():
            raise ToolError(f"Path not found: {path}")
        if path.is_dir():
            raise ToolError(f"Path is a directory: {path}")
        return path

    def _normalize_term(self, term: str) -> str:
        return " ".join(term.strip().split())

    def _normalize_description(self, description: str) -> str:
        return " ".join(description.strip().split())

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextSpokenDomainArgs):
            return ToolCallDisplay(summary="context_spoken_domain")
        return ToolCallDisplay(
            summary="context_spoken_domain",
            details={
                "path": event.args.path,
                "domain": event.args.domain,
                "domains": event.args.domains,
                "assign_mode": event.args.assign_mode,
                "domain_sequence": event.args.domain_sequence,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpokenDomainResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=True,
            message=(
                f"Selected {event.result.domain} domain with "
                f"{event.result.segment_count} segment(s)"
            ),
            warnings=event.result.warnings,
            details={
                "domain": event.result.domain,
                "segment_count": event.result.segment_count,
                "assign_mode": event.result.assign_mode,
                "domain_sequence": event.result.domain_sequence,
                "reasoning_summary": event.result.reasoning_summary,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Selecting speaking domain"
