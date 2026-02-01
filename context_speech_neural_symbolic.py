
from __future__ import annotations

import re
from collections import Counter, defaultdict
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
SPEAKER_RE = re.compile(r"^\s*([A-Za-z0-9 _-]{1,24}):\s+(.*)$")

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

DIRECTIVE_MARKERS = {
    "please",
    "need",
    "needs",
    "must",
    "should",
    "ask",
    "tell",
    "do",
    "make",
    "use",
    "try",
}

CONDITIONAL_MARKERS = {"if", "unless", "when", "whenever", "provided"}
CAUSAL_MARKERS = {"because", "therefore", "thus", "so", "since"}
CONTRAST_MARKERS = {"but", "however", "although", "though", "yet"}
AGREEMENT_MARKERS = {"yes", "agree", "correct", "right", "sure", "ok"}
DISAGREEMENT_MARKERS = {"no", "disagree", "wrong", "not", "never"}

POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "positive",
    "success",
    "benefit",
    "helpful",
    "strong",
}

NEGATIVE_WORDS = {
    "bad",
    "poor",
    "negative",
    "risk",
    "issue",
    "problem",
    "fail",
}


class ContextSpeechNeuralSymbolicConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_source_bytes: int = Field(
        default=5_000_000, description="Maximum bytes per content."
    )
    max_total_bytes: int = Field(
        default=20_000_000, description="Maximum total bytes across utterances."
    )
    max_utterances: int = Field(default=500, description="Maximum utterances to process.")
    preview_chars: int = Field(default=240, description="Preview snippet length.")
    min_token_length: int = Field(default=3, description="Minimum token length.")
    max_symbols: int = Field(default=120, description="Maximum symbols to keep.")
    max_rules: int = Field(default=80, description="Maximum symbolic rules to keep.")
    max_relations: int = Field(default=200, description="Maximum relations to keep.")
    max_relation_evidence: int = Field(
        default=5, description="Maximum evidence per relation."
    )
    max_rule_evidence: int = Field(default=4, description="Maximum evidence per rule.")
    max_symbol_segments: int = Field(
        default=10, description="Maximum symbol speech segments."
    )
    max_rule_segments: int = Field(default=10, description="Maximum rule speech segments.")
    max_relation_segments: int = Field(
        default=10, description="Maximum relation speech segments."
    )
    max_utterance_segments: int = Field(
        default=10, description="Maximum utterance speech segments."
    )
    max_speech_segments: int = Field(
        default=30, description="Maximum total speech segments."
    )
    default_split_mode: str = Field(
        default="lines", description="lines or sentences."
    )


class ContextSpeechNeuralSymbolicState(BaseToolState):
    pass


class SpokenUtterance(BaseModel):
    text: str = Field(description="Utterance text.")
    speaker: str | None = Field(default=None, description="Speaker label.")
    timestamp: str | None = Field(default=None, description="Optional timestamp.")


class ContextSpeechNeuralSymbolicArgs(BaseModel):
    content: str | None = Field(default=None, description="Conversation content.")
    path: str | None = Field(default=None, description="Path to transcript file.")
    utterances: list[SpokenUtterance] | None = Field(
        default=None, description="Explicit utterance list."
    )
    language: str | None = Field(
        default="en", description="Language code (default en)."
    )
    split_mode: str | None = Field(default=None, description="lines or sentences.")
    max_source_bytes: int | None = Field(
        default=None, description="Override max_source_bytes."
    )
    max_total_bytes: int | None = Field(
        default=None, description="Override max_total_bytes."
    )
    max_utterances: int | None = Field(
        default=None, description="Override max_utterances."
    )
    min_token_length: int | None = Field(
        default=None, description="Override min_token_length."
    )
    max_symbols: int | None = Field(default=None, description="Override max_symbols.")
    max_rules: int | None = Field(default=None, description="Override max_rules.")
    max_relations: int | None = Field(default=None, description="Override max_relations.")
    max_relation_evidence: int | None = Field(
        default=None, description="Override max_relation_evidence."
    )
    max_rule_evidence: int | None = Field(
        default=None, description="Override max_rule_evidence."
    )
    max_symbol_segments: int | None = Field(
        default=None, description="Override max_symbol_segments."
    )
    max_rule_segments: int | None = Field(
        default=None, description="Override max_rule_segments."
    )
    max_relation_segments: int | None = Field(
        default=None, description="Override max_relation_segments."
    )
    max_utterance_segments: int | None = Field(
        default=None, description="Override max_utterance_segments."
    )
    max_speech_segments: int | None = Field(
        default=None, description="Override max_speech_segments."
    )
    include_opening: bool = Field(
        default=True, description="Include speech opening."
    )
    include_closing: bool = Field(
        default=True, description="Include speech closing."
    )


class UtteranceSummary(BaseModel):
    index: int
    speaker: str | None
    text: str
    intent: str
    sentiment: str
    symbols: list[str]
    salience: float


class NeuralSymbol(BaseModel):
    symbol: str
    frequency: int
    recency: float
    activation: float
    speakers: list[str]
    utterance_indices: list[int]


class SymbolicRule(BaseModel):
    rule_type: str
    description: str
    symbol: str | None
    related_symbols: list[str]
    evidence_indices: list[int]


class SymbolRelation(BaseModel):
    source: str
    target: str
    weight: float
    evidence_indices: list[int]


class SpeechSegment(BaseModel):
    index: int
    kind: str
    symbols: list[str]
    rule_types: list[str]
    utterance_indices: list[int]
    cue: str


class ContextSpeechNeuralSymbolicResult(BaseModel):
    utterances: list[UtteranceSummary]
    symbols: list[NeuralSymbol]
    rules: list[SymbolicRule]
    relations: list[SymbolRelation]
    speech_opening: str
    speech_segments: list[SpeechSegment]
    speech_closing: str
    truncated: bool
    warnings: list[str]


class ContextSpeechNeuralSymbolic(
    BaseTool[
        ContextSpeechNeuralSymbolicArgs,
        ContextSpeechNeuralSymbolicResult,
        ContextSpeechNeuralSymbolicConfig,
        ContextSpeechNeuralSymbolicState,
    ],
    ToolUIData[ContextSpeechNeuralSymbolicArgs, ContextSpeechNeuralSymbolicResult],
):
    description: ClassVar[str] = (
        "Model neural-symbolic reasoning across spoken communication."
    )

    async def run(
        self, args: ContextSpeechNeuralSymbolicArgs
    ) -> ContextSpeechNeuralSymbolicResult:
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
        max_utterances = (
            args.max_utterances
            if args.max_utterances is not None
            else self.config.max_utterances
        )
        min_token_length = (
            args.min_token_length
            if args.min_token_length is not None
            else self.config.min_token_length
        )
        max_symbols = args.max_symbols if args.max_symbols is not None else self.config.max_symbols
        max_rules = args.max_rules if args.max_rules is not None else self.config.max_rules
        max_relations = (
            args.max_relations if args.max_relations is not None else self.config.max_relations
        )
        max_relation_evidence = (
            args.max_relation_evidence
            if args.max_relation_evidence is not None
            else self.config.max_relation_evidence
        )
        max_rule_evidence = (
            args.max_rule_evidence
            if args.max_rule_evidence is not None
            else self.config.max_rule_evidence
        )
        max_symbol_segments = (
            args.max_symbol_segments
            if args.max_symbol_segments is not None
            else self.config.max_symbol_segments
        )
        max_rule_segments = (
            args.max_rule_segments
            if args.max_rule_segments is not None
            else self.config.max_rule_segments
        )
        max_relation_segments = (
            args.max_relation_segments
            if args.max_relation_segments is not None
            else self.config.max_relation_segments
        )
        max_utterance_segments = (
            args.max_utterance_segments
            if args.max_utterance_segments is not None
            else self.config.max_utterance_segments
        )
        max_speech_segments = (
            args.max_speech_segments
            if args.max_speech_segments is not None
            else self.config.max_speech_segments
        )
        split_mode = (
            args.split_mode or self.config.default_split_mode
        ).strip().lower()

        if max_source_bytes <= 0:
            raise ToolError("max_source_bytes must be positive.")
        if max_total_bytes <= 0:
            raise ToolError("max_total_bytes must be positive.")
        if max_utterances <= 0:
            raise ToolError("max_utterances must be positive.")
        if min_token_length <= 0:
            raise ToolError("min_token_length must be positive.")
        if max_symbols < 0:
            raise ToolError("max_symbols must be non-negative.")
        if max_rules < 0:
            raise ToolError("max_rules must be non-negative.")
        if max_relations < 0:
            raise ToolError("max_relations must be non-negative.")
        if max_relation_evidence < 0:
            raise ToolError("max_relation_evidence must be non-negative.")
        if max_rule_evidence < 0:
            raise ToolError("max_rule_evidence must be non-negative.")
        if max_symbol_segments < 0:
            raise ToolError("max_symbol_segments must be non-negative.")
        if max_rule_segments < 0:
            raise ToolError("max_rule_segments must be non-negative.")
        if max_relation_segments < 0:
            raise ToolError("max_relation_segments must be non-negative.")
        if max_utterance_segments < 0:
            raise ToolError("max_utterance_segments must be non-negative.")
        if max_speech_segments < 0:
            raise ToolError("max_speech_segments must be non-negative.")
        if split_mode not in {"lines", "sentences"}:
            raise ToolError("split_mode must be lines or sentences.")

        warnings: list[str] = []
        language = (args.language or "en").strip().lower()
        if language not in {"en", "eng", "english"}:
            warnings.append("Non-English language detected; results may be degraded.")

        utterances = self._load_utterances(args, max_source_bytes, split_mode)
        truncated = False
        total_bytes = sum(len(utt.text.encode("utf-8")) for utt in utterances)
        if total_bytes > max_total_bytes:
            truncated = True
            warnings.append("Total utterance bytes exceed max_total_bytes.")
        if len(utterances) > max_utterances:
            truncated = True
            utterances = utterances[:max_utterances]
            warnings.append("Utterances truncated by max_utterances.")

        if not utterances:
            return ContextSpeechNeuralSymbolicResult(
                utterances=[],
                symbols=[],
                rules=[],
                relations=[],
                speech_opening="",
                speech_segments=[],
                speech_closing="",
                truncated=truncated,
                warnings=warnings or ["No utterances found."],
            )

        tokenized: list[list[str]] = []
        speakers_by_symbol: dict[str, set[str]] = defaultdict(set)
        occurrences: dict[str, list[int]] = defaultdict(list)
        all_counts = Counter()
        utterance_symbols: list[list[str]] = []
        utterance_intent: list[str] = []
        utterance_sentiment: list[str] = []

        for idx, utt in enumerate(utterances, start=1):
            tokens = self._tokenize(utt.text, min_token_length)
            tokenized.append(tokens)
            filtered = [t for t in tokens if t not in STOPWORDS]
            counts = Counter(filtered)
            all_counts.update(counts)
            utterance_intent.append(self._intent_for(utt.text, filtered))
            utterance_sentiment.append(self._sentiment_for(filtered))
            for token in counts:
                occurrences[token].append(idx)
                if utt.speaker:
                    speakers_by_symbol[token].add(utt.speaker)

        symbols = [symbol for symbol, _ in all_counts.most_common(max_symbols)]
        if not symbols:
            return ContextSpeechNeuralSymbolicResult(
                utterances=[],
                symbols=[],
                rules=[],
                relations=[],
                speech_opening="",
                speech_segments=[],
                speech_closing="",
                truncated=truncated,
                warnings=warnings or ["No symbols extracted."],
            )

        max_count = max(all_counts.values()) if all_counts else 1
        total_utterances = len(utterances)
        symbol_stats: list[NeuralSymbol] = []
        activation_map: dict[str, float] = {}

        for symbol in symbols:
            freq = all_counts.get(symbol, 0)
            occ = occurrences.get(symbol, [])
            recency = 0.0
            if occ:
                recency = sum(index / total_utterances for index in occ) / len(occ)
            activation = (freq / max_count) * 0.6 + recency * 0.4
            activation = round(activation, 6)
            activation_map[symbol] = activation
            symbol_stats.append(
                NeuralSymbol(
                    symbol=symbol,
                    frequency=freq,
                    recency=round(recency, 6),
                    activation=activation,
                    speakers=sorted(speakers_by_symbol.get(symbol, set())),
                    utterance_indices=occ[:],
                )
            )

        symbol_stats.sort(key=lambda item: (-item.activation, item.symbol))
        symbol_set = set(symbols)

        utterance_summaries: list[UtteranceSummary] = []
        for idx, utt in enumerate(utterances, start=1):
            symbols_in_utt = sorted({tok for tok in tokenized[idx - 1] if tok in symbol_set})
            utterance_symbols.append(symbols_in_utt)
            salience = round(sum(activation_map.get(sym, 0.0) for sym in symbols_in_utt), 6)
            utterance_summaries.append(
                UtteranceSummary(
                    index=idx,
                    speaker=utt.speaker,
                    text=utt.text,
                    intent=utterance_intent[idx - 1],
                    sentiment=utterance_sentiment[idx - 1],
                    symbols=symbols_in_utt,
                    salience=salience,
                )
            )

        relations = self._build_relations(
            utterance_symbols, max_relations, max_relation_evidence
        )
        rules = self._build_rules(
            utterances,
            utterance_symbols,
            max_rules,
            max_rule_evidence,
        )

        speech_opening = self._speech_opening(args, symbol_stats, rules)
        speech_segments, segments_truncated = self._speech_segments(
            symbol_stats,
            rules,
            relations,
            utterance_summaries,
            max_symbol_segments,
            max_rule_segments,
            max_relation_segments,
            max_utterance_segments,
            max_speech_segments,
        )
        if segments_truncated:
            warnings.append("Speech segments truncated by limits.")
        speech_closing = self._speech_closing(args, relations)

        return ContextSpeechNeuralSymbolicResult(
            utterances=utterance_summaries,
            symbols=symbol_stats,
            rules=rules,
            relations=relations,
            speech_opening=speech_opening,
            speech_segments=speech_segments,
            speech_closing=speech_closing,
            truncated=truncated,
            warnings=warnings,
        )

    def _load_utterances(
        self,
        args: ContextSpeechNeuralSymbolicArgs,
        max_source_bytes: int,
        split_mode: str,
    ) -> list[SpokenUtterance]:
        if args.utterances:
            return args.utterances
        if args.content and args.path:
            raise ToolError("Provide content or path, not both.")
        if args.content is None and args.path is None:
            return []
        if args.content is not None:
            data = args.content.encode("utf-8")
            if len(data) > max_source_bytes:
                raise ToolError(
                    f"content exceeds max_source_bytes ({len(data)} > {max_source_bytes})."
                )
            content = args.content
        else:
            path = self._resolve_path(args.path or "")
            if path.is_dir():
                raise ToolError(f"Path is a directory: {path}")
            size = path.stat().st_size
            if size > max_source_bytes:
                raise ToolError(
                    f"{path} exceeds max_source_bytes ({size} > {max_source_bytes})."
                )
            content = path.read_text("utf-8", errors="ignore")

        utterances: list[SpokenUtterance] = []
        if split_mode == "sentences":
            for sentence in SENTENCE_RE.findall(content):
                text = sentence.strip()
                if text:
                    utterances.append(SpokenUtterance(text=text))
            return utterances

        for line in content.splitlines():
            text = line.strip()
            if not text:
                continue
            speaker = None
            match = SPEAKER_RE.match(text)
            if match:
                speaker = match.group(1).strip()
                text = match.group(2).strip()
            utterances.append(SpokenUtterance(text=text, speaker=speaker))
        return utterances

    def _resolve_path(self, raw_path: str) -> Path:
        if not raw_path.strip():
            raise ToolError("path cannot be empty.")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.config.effective_workdir / path
        return path.resolve()

    def _tokenize(self, text: str, min_token_length: int) -> list[str]:
        tokens = [
            token.lower()
            for token in WORD_RE.findall(text)
            if len(token) >= min_token_length
        ]
        return tokens

    def _intent_for(self, text: str, tokens: list[str]) -> str:
        lowered = text.lower()
        if "?" in text or any(word in tokens for word in QUESTION_WORDS):
            return "question"
        if any(word in tokens for word in DIRECTIVE_MARKERS) or "please" in lowered:
            return "directive"
        if any(word in tokens for word in CONDITIONAL_MARKERS):
            return "conditional"
        if any(word in tokens for word in CAUSAL_MARKERS):
            return "causal"
        if any(word in tokens for word in CONTRAST_MARKERS):
            return "contrast"
        return "statement"

    def _sentiment_for(self, tokens: list[str]) -> str:
        pos = sum(1 for token in tokens if token in POSITIVE_WORDS)
        neg = sum(1 for token in tokens if token in NEGATIVE_WORDS)
        if pos > neg:
            return "positive"
        if neg > pos:
            return "negative"
        return "neutral"

    def _build_relations(
        self,
        utterance_symbols: list[list[str]],
        max_relations: int,
        max_relation_evidence: int,
    ) -> list[SymbolRelation]:
        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        evidence: dict[tuple[str, str], list[int]] = defaultdict(list)
        for idx, symbols in enumerate(utterance_symbols, start=1):
            unique = sorted(set(symbols))
            for i in range(len(unique)):
                for j in range(i + 1, len(unique)):
                    pair = (unique[i], unique[j])
                    pair_counts[pair] += 1
                    if len(evidence[pair]) < max_relation_evidence:
                        evidence[pair].append(idx)
        if not pair_counts:
            return []
        max_count = max(pair_counts.values()) if pair_counts else 1
        ranked = sorted(pair_counts.items(), key=lambda item: (-item[1], item[0]))
        relations: list[SymbolRelation] = []
        for (source, target), count in ranked[:max_relations]:
            weight = round(count / max_count, 6)
            relations.append(
                SymbolRelation(
                    source=source,
                    target=target,
                    weight=weight,
                    evidence_indices=evidence.get((source, target), [])[:],
                )
            )
        return relations

    def _build_rules(
        self,
        utterances: list[SpokenUtterance],
        utterance_symbols: list[list[str]],
        max_rules: int,
        max_rule_evidence: int,
    ) -> list[SymbolicRule]:
        rule_map: dict[tuple[str, str | None, tuple[str, ...]], SymbolicRule] = {}
        definition_re = re.compile(
            r"\b([A-Za-z0-9_]+)\s+(is|means|equals|defined as)\s+([A-Za-z0-9_]+)",
            re.IGNORECASE,
        )
        for idx, utt in enumerate(utterances, start=1):
            text = utt.text
            tokens = [token.lower() for token in WORD_RE.findall(text)]
            symbols = utterance_symbols[idx - 1]
            primary_symbol = symbols[0] if symbols else None
            types: list[tuple[str, str]] = []

            if "?" in text or any(word in tokens for word in QUESTION_WORDS):
                types.append(("question", "Question intent detected."))
            if any(word in tokens for word in DIRECTIVE_MARKERS):
                types.append(("directive", "Directive intent detected."))
            if any(word in tokens for word in CONDITIONAL_MARKERS):
                types.append(("conditional", "Conditional intent detected."))
            if any(word in tokens for word in CAUSAL_MARKERS):
                types.append(("causal", "Causal intent detected."))
            if any(word in tokens for word in CONTRAST_MARKERS):
                types.append(("contrast", "Contrast detected."))
            if any(word in tokens for word in AGREEMENT_MARKERS):
                types.append(("agreement", "Agreement detected."))
            if any(word in tokens for word in DISAGREEMENT_MARKERS):
                types.append(("disagreement", "Disagreement detected."))

            match = definition_re.search(text)
            if match:
                lhs = match.group(1).lower()
                rhs = match.group(3).lower()
                key = ("definition", lhs, (rhs,))
                rule_map[key] = self._merge_rule(
                    rule_map.get(key),
                    rule_type="definition",
                    description=f"Definition: {lhs} -> {rhs}.",
                    symbol=lhs,
                    related_symbols=[rhs],
                    evidence_index=idx,
                    max_evidence=max_rule_evidence,
                )

            for rule_type, description in types:
                key = (rule_type, primary_symbol, tuple())
                rule_map[key] = self._merge_rule(
                    rule_map.get(key),
                    rule_type=rule_type,
                    description=description,
                    symbol=primary_symbol,
                    related_symbols=[],
                    evidence_index=idx,
                    max_evidence=max_rule_evidence,
                )

        rules = list(rule_map.values())
        rules.sort(key=lambda rule: (-len(rule.evidence_indices), rule.rule_type))
        return rules[:max_rules]

    def _merge_rule(
        self,
        existing: SymbolicRule | None,
        rule_type: str,
        description: str,
        symbol: str | None,
        related_symbols: list[str],
        evidence_index: int,
        max_evidence: int,
    ) -> SymbolicRule:
        if existing is None:
            return SymbolicRule(
                rule_type=rule_type,
                description=description,
                symbol=symbol,
                related_symbols=related_symbols,
                evidence_indices=[evidence_index],
            )
        if evidence_index not in existing.evidence_indices and len(
            existing.evidence_indices
        ) < max_evidence:
            existing.evidence_indices.append(evidence_index)
        return existing

    def _speech_opening(
        self,
        args: ContextSpeechNeuralSymbolicArgs,
        symbols: list[NeuralSymbol],
        rules: list[SymbolicRule],
    ) -> str:
        if not args.include_opening:
            return ""
        top_symbols = ", ".join(sym.symbol for sym in symbols[:4])
        top_rules = ", ".join(rule.rule_type for rule in rules[:3])
        parts = ["Begin neural-symbolic reasoning across spoken turns."]
        if top_symbols:
            parts.append(f"Neural focus symbols: {top_symbols}.")
        if top_rules:
            parts.append(f"Symbolic cues: {top_rules}.")
        return " ".join(parts)

    def _speech_segments(
        self,
        symbols: list[NeuralSymbol],
        rules: list[SymbolicRule],
        relations: list[SymbolRelation],
        utterances: list[UtteranceSummary],
        max_symbol_segments: int,
        max_rule_segments: int,
        max_relation_segments: int,
        max_utterance_segments: int,
        max_speech_segments: int,
    ) -> tuple[list[SpeechSegment], bool]:
        segments: list[SpeechSegment] = []

        for symbol in symbols[:max_symbol_segments]:
            cue = (
                f"Neural activation: emphasize {symbol.symbol} (score {symbol.activation})."
            )
            segments.append(
                SpeechSegment(
                    index=len(segments) + 1,
                    kind="symbol",
                    symbols=[symbol.symbol],
                    rule_types=[],
                    utterance_indices=symbol.utterance_indices[:3],
                    cue=cue,
                )
            )

        for rule in rules[:max_rule_segments]:
            cue = f"Symbolic rule {rule.rule_type}: {rule.description}"
            segments.append(
                SpeechSegment(
                    index=len(segments) + 1,
                    kind="rule",
                    symbols=[rule.symbol] if rule.symbol else [],
                    rule_types=[rule.rule_type],
                    utterance_indices=rule.evidence_indices,
                    cue=cue,
                )
            )

        for relation in relations[:max_relation_segments]:
            cue = (
                f"Bridge {relation.source} and {relation.target} "
                f"(weight {relation.weight})."
            )
            segments.append(
                SpeechSegment(
                    index=len(segments) + 1,
                    kind="relation",
                    symbols=[relation.source, relation.target],
                    rule_types=[],
                    utterance_indices=relation.evidence_indices,
                    cue=cue,
                )
            )

        utterance_ranked = sorted(
            utterances, key=lambda item: (-item.salience, item.index)
        )
        for utt in utterance_ranked[:max_utterance_segments]:
            cue = (
                f"Reference utterance {utt.index} ({utt.intent}) "
                f"with symbols {', '.join(utt.symbols[:4])}."
            )
            segments.append(
                SpeechSegment(
                    index=len(segments) + 1,
                    kind="utterance",
                    symbols=utt.symbols[:4],
                    rule_types=[utt.intent],
                    utterance_indices=[utt.index],
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
        self, args: ContextSpeechNeuralSymbolicArgs, relations: list[SymbolRelation]
    ) -> str:
        if not args.include_closing:
            return ""
        if relations:
            return "Close by reinforcing the strongest symbol links."
        return "Close by summarizing the main symbolic rules."

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextSpeechNeuralSymbolicArgs):
            return ToolCallDisplay(summary="context_speech_neural_symbolic")
        utterance_count = len(event.args.utterances or [])
        summary = "context_speech_neural_symbolic"
        return ToolCallDisplay(
            summary=summary,
            details={
                "utterance_count": utterance_count,
                "path": event.args.path,
                "language": event.args.language,
                "split_mode": event.args.split_mode,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpeechNeuralSymbolicResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = (
            f"Processed {len(event.result.utterances)} utterance(s) with "
            f"{len(event.result.symbols)} symbol(s)"
        )
        warnings = event.result.warnings[:]
        if event.result.truncated:
            warnings.append("Output truncated by limits")
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=warnings,
            details={
                "utterance_count": len(event.result.utterances),
                "symbol_count": len(event.result.symbols),
                "rule_count": len(event.result.rules),
                "relation_count": len(event.result.relations),
                "truncated": event.result.truncated,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Modeling neural-symbolic speech reasoning"
