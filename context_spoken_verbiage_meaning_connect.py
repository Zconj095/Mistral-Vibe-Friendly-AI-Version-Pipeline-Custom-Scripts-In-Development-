from __future__ import annotations

import math
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
SPEAKER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _.'-]{0,40}):\s*(.*)$")


def _tokenize(text: str, min_len: int) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text) if len(token) >= min_len]


class ContextSpokenVerbiageMeaningConnectConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_items: int = Field(default=40, description="Maximum items to evaluate.")
    max_source_bytes: int = Field(default=3_000_000, description="Maximum bytes per item.")
    max_total_bytes: int = Field(default=20_000_000, description="Max bytes across items.")
    preview_chars: int = Field(default=400, description="Preview snippet length.")
    max_segments: int = Field(default=200, description="Maximum segments per item.")
    min_segment_chars: int = Field(default=20, description="Minimum segment length.")
    min_token_length: int = Field(default=3, description="Minimum token length.")
    max_keywords: int = Field(default=12, description="Maximum keywords per segment.")
    min_shared_keywords: int = Field(default=2, description="Minimum shared keywords.")
    min_meaning_score: float = Field(default=0.1, description="Minimum meaning score.")
    require_meaning_overlap: bool = Field(
        default=True, description="Require shared meaning tokens to connect."
    )
    max_connections: int = Field(default=500, description="Maximum connections per item.")
    min_combined_score: float = Field(
        default=0.4, description="Minimum combined score to connect."
    )
    verbiage_weight: float = Field(default=0.6, description="Weight for verbiage flow.")
    meaning_weight: float = Field(default=0.4, description="Weight for meaning overlap.")
    max_chain_segments: int = Field(default=12, description="Maximum segments per chain.")
    max_chain_keywords: int = Field(default=12, description="Keywords per chain.")
    allow_cross_speaker: bool = Field(
        default=True, description="Allow connecting across speaker changes."
    )
    joiner: str = Field(default=" ", description="Joiner inserted between segments.")
    continuation_starters: list[str] = Field(
        default_factory=lambda: [
            "and",
            "but",
            "so",
            "because",
            "then",
            "also",
            "or",
            "if",
            "when",
            "while",
            "although",
            "though",
            "that",
            "which",
            "who",
            "where",
            "after",
            "before",
            "since",
        ],
        description="Tokens that indicate a continuation at the start.",
    )
    conjunction_endings: list[str] = Field(
        default_factory=lambda: [
            "and",
            "but",
            "so",
            "because",
            "or",
            "then",
            "also",
            "if",
            "when",
            "while",
            "though",
            "although",
        ],
        description="Tokens that indicate a continuation at the end.",
    )


class ContextSpokenVerbiageMeaningConnectState(BaseToolState):
    pass


class SpokenVerbiageMeaningItem(BaseModel):
    id: str | None = Field(default=None)
    name: str | None = Field(default=None)
    content: str | None = Field(default=None)
    path: str | None = Field(default=None)
    source: str | None = Field(default=None)



class ContextSpokenVerbiageMeaningConnectArgs(BaseModel):
    items: list[SpokenVerbiageMeaningItem] = Field(description="Items to evaluate.")

class SpokenVerbiageMeaningSegment(BaseModel):
    index: int
    speaker: str | None
    text: str
    start: int
    end: int
    token_count: int
    keywords: list[str]
    end_char: str | None
    start_token: str | None
    end_token: str | None


class VerbiageMeaningConnection(BaseModel):
    from_index: int
    to_index: int
    speaker_from: str | None
    speaker_to: str | None
    verbiage_score: float
    meaning_score: float
    combined_score: float
    shared_keywords: list[str]
    reasons: list[str]
    connection_type: str


class VerbiageMeaningChain(BaseModel):
    index: int
    segment_indices: list[int]
    speakers: list[str]
    text: str
    connection_count: int
    average_verbiage_score: float
    average_meaning_score: float
    average_combined_score: float
    bridge_keywords: list[str]


class SpokenVerbiageMeaningInsight(BaseModel):
    index: int
    id: str | None
    name: str | None
    source_path: str | None
    preview: str
    segments: list[SpokenVerbiageMeaningSegment]
    connections: list[VerbiageMeaningConnection]
    chains: list[VerbiageMeaningChain]
    segment_count: int
    connection_count: int
    chain_count: int


class ContextSpokenVerbiageMeaningConnectResult(BaseModel):
    items: list[SpokenVerbiageMeaningInsight]
    item_count: int
    total_connections: int
    truncated: bool
    warnings: list[str]
    errors: list[str]


class ContextSpokenVerbiageMeaningConnect(
    BaseTool[
        ContextSpokenVerbiageMeaningConnectArgs,
        ContextSpokenVerbiageMeaningConnectResult,
        ContextSpokenVerbiageMeaningConnectConfig,
        ContextSpokenVerbiageMeaningConnectState,
    ],
    ToolUIData[ContextSpokenVerbiageMeaningConnectArgs, ContextSpokenVerbiageMeaningConnectResult],
):
    description: ClassVar[str] = (
        "Connect spoken verbiage with meaning-aligned continuity."
    )

    async def run(self, args: ContextSpokenVerbiageMeaningConnectArgs) -> ContextSpokenVerbiageMeaningConnectResult:
        items = list(args.items)
        if not items:
            raise ToolError("items is required.")

        errors: list[str] = []
        warnings: list[str] = []
        insights: list[SpokenVerbiageMeaningInsight] = []
        total_bytes = 0
        truncated = False
        total_connections = 0

        for idx, item in enumerate(items, start=1):
            try:
                content, source_path, size_bytes = self._load_item(item)
                if size_bytes is None:
                    raise ToolError("Item has no content.")
                if total_bytes + size_bytes > self.config.max_total_bytes:
                    truncated = True
                    warnings.append("Budget exceeded; stopping evaluation.")
                    break
                total_bytes += size_bytes

                segments = self._segment_conversation(content)
                segments = self._attach_keywords(segments)
                connections = self._build_connections(segments)
                chains = self._build_chains(segments, connections)
                total_connections += len(connections)

                insights.append(
                    SpokenVerbiageMeaningInsight(
                        index=len(insights) + 1,
                        id=item.id,
                        name=item.name,
                        source_path=source_path,
                        preview=self._preview(content),
                        segments=segments[: self.config.max_segments],
                        connections=connections[: self.config.max_connections],
                        chains=chains,
                        segment_count=len(segments),
                        connection_count=len(connections),
                        chain_count=len(chains),
                    )
                )
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not insights:
            raise ToolError("No insights generated.")

        return ContextSpokenVerbiageMeaningConnectResult(
            items=insights,
            item_count=len(insights),
            total_connections=total_connections,
            truncated=truncated,
            warnings=warnings,
            errors=errors,
        )

    def _segment_conversation(self, text: str) -> list[SpokenVerbiageMeaningSegment]:
        segments: list[SpokenVerbiageMeaningSegment] = []
        lines = text.splitlines()
        current_lines: list[str] = []
        current_speaker: str | None = None
        seg_start = 0
        cursor = 0

        def flush(end_offset: int) -> None:
            nonlocal current_lines, current_speaker, seg_start
            if not current_lines:
                return
            seg_text = "\n".join(current_lines).strip()
            if not seg_text:
                current_lines = []
                return
            if len(seg_text) < self.config.min_segment_chars and segments:
                segments[-1].text = f"{segments[-1].text}\n{seg_text}".strip()
                segments[-1].end = end_offset
                segments[-1].token_count = self._token_count(segments[-1].text)
                segments[-1].end_char = self._end_char(segments[-1].text)
                segments[-1].start_token = self._first_token(segments[-1].text)
                segments[-1].end_token = self._last_token(segments[-1].text)
            else:
                segments.append(
                    SpokenVerbiageMeaningSegment(
                        index=len(segments) + 1,
                        speaker=current_speaker,
                        text=seg_text,
                        start=seg_start,
                        end=end_offset,
                        token_count=self._token_count(seg_text),
                        keywords=[],
                        end_char=self._end_char(seg_text),
                        start_token=self._first_token(seg_text),
                        end_token=self._last_token(seg_text),
                    )
                )
            current_lines = []
            current_speaker = None

        for line in lines:
            line_len = len(line) + 1
            speaker_match = SPEAKER_RE.match(line)
            if speaker_match:
                flush(cursor)
                current_speaker = speaker_match.group(1).strip()
                current_lines.append(speaker_match.group(2))
                seg_start = cursor
            elif not line.strip():
                flush(cursor)
            else:
                if not current_lines:
                    seg_start = cursor
                current_lines.append(line)
            cursor += line_len

        flush(cursor)
        return segments[: self.config.max_segments]

    def _attach_keywords(
        self, segments: list[SpokenVerbiageMeaningSegment]
    ) -> list[SpokenVerbiageMeaningSegment]:
        df_counts: dict[str, int] = {}
        all_tokens: list[list[str]] = []
        for segment in segments:
            tokens = _tokenize(segment.text, self.config.min_token_length)
            all_tokens.append(tokens)
            for token in set(tokens):
                df_counts[token] = df_counts.get(token, 0) + 1

        total_segments = max(len(segments), 1)
        for segment, tokens in zip(segments, all_tokens):
            tf_counts: dict[str, int] = {}
            for token in tokens:
                tf_counts[token] = tf_counts.get(token, 0) + 1
            scored = []
            for token, tf in tf_counts.items():
                df = df_counts.get(token, 1)
                idf = math.log((total_segments + 1) / df) + 1.0
                scored.append((token, tf * idf))
            scored.sort(key=lambda item: item[1], reverse=True)
            segment.keywords = [token for token, _ in scored[: self.config.max_keywords]]
        return segments

    def _build_connections(
        self, segments: list[SpokenVerbiageMeaningSegment]
    ) -> list[VerbiageMeaningConnection]:
        connections: list[VerbiageMeaningConnection] = []
        starters = {token.lower() for token in self.config.continuation_starters}
        endings = {token.lower() for token in self.config.conjunction_endings}
        total_weight = max(self.config.verbiage_weight + self.config.meaning_weight, 0.0)

        for idx in range(len(segments) - 1):
            prev = segments[idx]
            curr = segments[idx + 1]
            same_speaker = (
                prev.speaker is not None
                and curr.speaker is not None
                and prev.speaker == curr.speaker
            )
            if not self.config.allow_cross_speaker and not same_speaker:
                continue

            verbiage_score, reasons = self._score_verbiage(prev, curr, starters, endings)
            meaning_score, shared = self._score_meaning(prev, curr)
            if self.config.require_meaning_overlap:
                if len(shared) < self.config.min_shared_keywords:
                    continue
                if meaning_score < self.config.min_meaning_score:
                    continue

            if total_weight > 0:
                combined = (
                    verbiage_score * self.config.verbiage_weight
                    + meaning_score * self.config.meaning_weight
                ) / total_weight
            else:
                combined = 0.0
            combined = round(max(0.0, min(1.0, combined)), 3)
            if combined < self.config.min_combined_score:
                continue

            reasons = reasons + [f"shared_keywords:{len(shared)}"]
            connection_type = "continuation" if same_speaker else "handoff"
            connections.append(
                VerbiageMeaningConnection(
                    from_index=prev.index,
                    to_index=curr.index,
                    speaker_from=prev.speaker,
                    speaker_to=curr.speaker,
                    verbiage_score=verbiage_score,
                    meaning_score=meaning_score,
                    combined_score=combined,
                    shared_keywords=shared,
                    reasons=reasons,
                    connection_type=connection_type,
                )
            )
            if len(connections) >= self.config.max_connections:
                break

        return connections

    def _build_chains(
        self,
        segments: list[SpokenVerbiageMeaningSegment],
        connections: list[VerbiageMeaningConnection],
    ) -> list[VerbiageMeaningChain]:
        if not segments:
            return []

        connection_map = {
            (conn.from_index, conn.to_index): conn for conn in connections
        }
        chains: list[VerbiageMeaningChain] = []
        chain_index = 1

        current_indices = [segments[0].index]
        current_text = segments[0].text
        current_speakers = set([segments[0].speaker] if segments[0].speaker else [])
        verbiage_scores: list[float] = []
        meaning_scores: list[float] = []
        combined_scores: list[float] = []
        keyword_counter: Counter[str] = Counter()

        for idx in range(1, len(segments)):
            prev = segments[idx - 1]
            curr = segments[idx]
            conn = connection_map.get((prev.index, curr.index))
            can_append = bool(conn) and len(current_indices) < self.config.max_chain_segments
            if can_append:
                current_text = self._join_text(current_text, curr.text)
                current_indices.append(curr.index)
                verbiage_scores.append(conn.verbiage_score)
                meaning_scores.append(conn.meaning_score)
                combined_scores.append(conn.combined_score)
                keyword_counter.update(conn.shared_keywords)
                if curr.speaker:
                    current_speakers.add(curr.speaker)
                continue

            chains.append(
                self._finalize_chain(
                    chain_index,
                    current_indices,
                    current_speakers,
                    current_text,
                    verbiage_scores,
                    meaning_scores,
                    combined_scores,
                    keyword_counter,
                )
            )
            chain_index += 1
            current_indices = [curr.index]
            current_text = curr.text
            current_speakers = set([curr.speaker] if curr.speaker else [])
            verbiage_scores = []
            meaning_scores = []
            combined_scores = []
            keyword_counter = Counter()

        chains.append(
            self._finalize_chain(
                chain_index,
                current_indices,
                current_speakers,
                current_text,
                verbiage_scores,
                meaning_scores,
                combined_scores,
                keyword_counter,
            )
        )
        return chains

    def _finalize_chain(
        self,
        chain_index: int,
        indices: list[int],
        speakers: set[str],
        text: str,
        verbiage_scores: list[float],
        meaning_scores: list[float],
        combined_scores: list[float],
        keyword_counter: Counter[str],
    ) -> VerbiageMeaningChain:
        avg_verbiage = sum(verbiage_scores) / len(verbiage_scores) if verbiage_scores else 0.0
        avg_meaning = sum(meaning_scores) / len(meaning_scores) if meaning_scores else 0.0
        avg_combined = sum(combined_scores) / len(combined_scores) if combined_scores else 0.0
        top_keywords = [word for word, _ in keyword_counter.most_common(self.config.max_chain_keywords)]
        return VerbiageMeaningChain(
            index=chain_index,
            segment_indices=indices,
            speakers=sorted(speakers),
            text=text.strip(),
            connection_count=len(verbiage_scores),
            average_verbiage_score=round(avg_verbiage, 3),
            average_meaning_score=round(avg_meaning, 3),
            average_combined_score=round(avg_combined, 3),
            bridge_keywords=top_keywords,
        )

    def _score_verbiage(
        self,
        prev: SpokenVerbiageMeaningSegment,
        curr: SpokenVerbiageMeaningSegment,
        starters: set[str],
        endings: set[str],
    ) -> tuple[float, list[str]]:
        score = 0.0
        reasons: list[str] = []
        end_char = prev.end_char or ""

        if end_char in ".?!":
            score -= 0.2
            reasons.append("sentence_end")
        elif end_char in ",;:":
            score += 0.25
            reasons.append("soft_pause")
        else:
            score += 0.35
            reasons.append("open_end")

        prev_end_token = (prev.end_token or "").lower()
        if prev_end_token and prev_end_token in endings:
            score += 0.25
            reasons.append("end_conjunction")

        curr_start_token = (curr.start_token or "").lower()
        if curr_start_token and curr_start_token in starters:
            score += 0.2
            reasons.append("continuation_start")

        if curr.text and curr.text[:1].islower():
            score += 0.1
            reasons.append("lowercase_start")

        if prev.speaker and curr.speaker and prev.speaker == curr.speaker:
            score += 0.2
            reasons.append("same_speaker")
        else:
            score -= 0.1
            reasons.append("speaker_change")

        score = max(0.0, min(1.0, score))
        return round(score, 3), reasons

    def _score_meaning(
        self, prev: SpokenVerbiageMeaningSegment, curr: SpokenVerbiageMeaningSegment
    ) -> tuple[float, list[str]]:
        shared = sorted(set(prev.keywords) & set(curr.keywords))
        if not shared:
            return 0.0, []
        union = set(prev.keywords) | set(curr.keywords)
        if not union:
            return 0.0, shared
        score = len(shared) / len(union)
        return round(score, 3), shared

    def _join_text(self, left: str, right: str) -> str:
        left = left.rstrip()
        right = right.lstrip()
        if not left:
            return right
        if not right:
            return left
        if left.endswith("-"):
            return f"{left}{right}"
        if left.endswith(" "):
            return f"{left}{right}"
        return f"{left}{self.config.joiner}{right}"

    def _first_token(self, text: str) -> str | None:
        tokens = WORD_RE.findall(text)
        return tokens[0] if tokens else None

    def _last_token(self, text: str) -> str | None:
        tokens = WORD_RE.findall(text)
        return tokens[-1] if tokens else None

    def _end_char(self, text: str) -> str | None:
        stripped = text.rstrip()
        return stripped[-1] if stripped else None

    def _token_count(self, text: str) -> int:
        return len(WORD_RE.findall(text))

    def _load_item(
        self, item: SpokenVerbiageMeaningItem
    ) -> tuple[str, str | None, int | None]:
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
                raise ToolError(f"Path is a directory: {path}")
            size = path.stat().st_size
            if size > self.config.max_source_bytes:
                raise ToolError(
                    f"{path} exceeds max_source_bytes ({size} > {self.config.max_source_bytes})."
                )
            return path.read_text("utf-8", errors="ignore"), str(path), size
        if item.content is not None:
            size = len(item.content.encode("utf-8"))
            if size > self.config.max_source_bytes:
                raise ToolError(
                    f"content exceeds max_source_bytes ({size} > {self.config.max_source_bytes})."
                )
            return item.content, None, size
        return "", None, 0

    def _preview(self, text: str) -> str:
        max_chars = self.config.preview_chars
        if max_chars <= 0:
            return ""
        return text if len(text) <= max_chars else text[:max_chars]

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextSpokenVerbiageMeaningConnectArgs):
            return ToolCallDisplay(summary="context_spoken_verbiage_meaning_connect")
        if not event.args.items:
            return ToolCallDisplay(summary="context_spoken_verbiage_meaning_connect")
        first = event.args.items[0]
        return ToolCallDisplay(
            summary="context_spoken_verbiage_meaning_connect",
            details={
                "item_count": len(event.args.items),
                "id": first.id,
                "name": first.name,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpokenVerbiageMeaningConnectResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=(
                f"Analyzed {event.result.item_count} item(s) with "
                f"{event.result.total_connections} meaning connections"
            ),
            warnings=event.result.warnings,
            details={
                "item_count": event.result.item_count,
                "total_connections": event.result.total_connections,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Connecting spoken verbiage meaning"
