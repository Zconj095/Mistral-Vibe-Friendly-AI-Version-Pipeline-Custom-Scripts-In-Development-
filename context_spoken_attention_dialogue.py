from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
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


class ContextSpokenAttentionDialogueConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_items: int = Field(default=40, description="Maximum items to evaluate.")
    max_source_bytes: int = Field(default=3_000_000, description="Maximum bytes per item.")
    max_total_bytes: int = Field(default=20_000_000, description="Max bytes across items.")
    preview_chars: int = Field(default=400, description="Preview snippet length.")
    max_segments: int = Field(default=200, description="Maximum segments per item.")
    min_segment_chars: int = Field(default=20, description="Minimum segment length.")
    min_token_length: int = Field(default=3, description="Minimum token length.")
    max_keywords_per_segment: int = Field(default=12, description="Max keywords per segment.")
    attention_window_segments: int = Field(
        default=6, description="Segments to look back for attention links."
    )
    min_shared_keywords: int = Field(default=2, description="Minimum shared keywords.")
    min_similarity: float = Field(default=0.1, description="Minimum similarity threshold.")
    max_links_per_segment: int = Field(default=3, description="Maximum links per segment.")
    max_links_total: int = Field(default=500, description="Maximum total links.")
    shift_similarity_threshold: float = Field(
        default=0.15, description="Similarity threshold for attention shifts."
    )
    max_speaker_keywords: int = Field(default=10, description="Max keywords per speaker.")


class ContextSpokenAttentionDialogueState(BaseToolState):
    pass


class SpokenAttentionItem(BaseModel):
    id: str | None = Field(default=None)
    name: str | None = Field(default=None)
    content: str | None = Field(default=None)
    path: str | None = Field(default=None)
    source: str | None = Field(default=None)



class ContextSpokenAttentionDialogueArgs(BaseModel):
    items: list[SpokenAttentionItem] = Field(description="Items to evaluate.")

class SpokenAttentionSegment(BaseModel):
    index: int
    speaker: str | None
    text: str
    start: int
    end: int
    token_count: int
    keywords: list[str]
    attention_score: float


class AttentionLink(BaseModel):
    from_index: int
    to_index: int
    speaker_from: str | None
    speaker_to: str | None
    shared_keywords: list[str]
    similarity: float


class AttentionShift(BaseModel):
    from_index: int
    to_index: int
    similarity: float
    shift_type: str


class SpeakerSummary(BaseModel):
    speaker: str
    segment_count: int
    total_tokens: int
    avg_attention: float
    top_keywords: list[str]


class SpokenAttentionDialogueInsight(BaseModel):
    index: int
    id: str | None
    name: str | None
    source_path: str | None
    preview: str
    segments: list[SpokenAttentionSegment]
    links: list[AttentionLink]
    shifts: list[AttentionShift]
    speakers: list[SpeakerSummary]
    segment_count: int
    link_count: int
    shift_count: int


class ContextSpokenAttentionDialogueResult(BaseModel):
    items: list[SpokenAttentionDialogueInsight]
    item_count: int
    total_links: int
    truncated: bool
    warnings: list[str]
    errors: list[str]


class ContextSpokenAttentionDialogue(
    BaseTool[
        ContextSpokenAttentionDialogueArgs,
        ContextSpokenAttentionDialogueResult,
        ContextSpokenAttentionDialogueConfig,
        ContextSpokenAttentionDialogueState,
    ],
    ToolUIData[ContextSpokenAttentionDialogueArgs, ContextSpokenAttentionDialogueResult],
):
    description: ClassVar[str] = (
        "Track spoken attention and dialogue structure across segments."
    )

    async def run(self, args: ContextSpokenAttentionDialogueArgs) -> ContextSpokenAttentionDialogueResult:
        items = list(args.items)
        if not items:
            raise ToolError("items is required.")

        errors: list[str] = []
        warnings: list[str] = []
        insights: list[SpokenAttentionDialogueInsight] = []
        total_bytes = 0
        truncated = False
        total_links = 0

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
                links = self._link_segments(segments)
                shifts = self._build_shifts(segments)
                speakers = self._summarize_speakers(segments)
                total_links += len(links)

                insights.append(
                    SpokenAttentionDialogueInsight(
                        index=len(insights) + 1,
                        id=item.id,
                        name=item.name,
                        source_path=source_path,
                        preview=self._preview(content),
                        segments=segments[: self.config.max_segments],
                        links=links[: self.config.max_links_total],
                        shifts=shifts,
                        speakers=speakers,
                        segment_count=len(segments),
                        link_count=len(links),
                        shift_count=len(shifts),
                    )
                )
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not insights:
            raise ToolError("No insights generated.")

        return ContextSpokenAttentionDialogueResult(
            items=insights,
            item_count=len(insights),
            total_links=total_links,
            truncated=truncated,
            warnings=warnings,
            errors=errors,
        )

    def _segment_conversation(self, text: str) -> list[SpokenAttentionSegment]:
        segments: list[SpokenAttentionSegment] = []
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
            else:
                segments.append(
                    SpokenAttentionSegment(
                        index=len(segments) + 1,
                        speaker=current_speaker,
                        text=seg_text,
                        start=seg_start,
                        end=end_offset,
                        token_count=self._token_count(seg_text),
                        keywords=[],
                        attention_score=0.0,
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
        self, segments: list[SpokenAttentionSegment]
    ) -> list[SpokenAttentionSegment]:
        df_counts: dict[str, int] = {}
        all_tokens: list[list[str]] = []
        for segment in segments:
            tokens = _tokenize(segment.text, self.config.min_token_length)
            all_tokens.append(tokens)
            for token in set(tokens):
                df_counts[token] = df_counts.get(token, 0) + 1

        total_segments = max(len(segments), 1)
        scores: list[float] = []
        for segment, tokens in zip(segments, all_tokens):
            tf_counts: dict[str, int] = {}
            for token in tokens:
                tf_counts[token] = tf_counts.get(token, 0) + 1
            scored = []
            total_score = 0.0
            for token, tf in tf_counts.items():
                df = df_counts.get(token, 1)
                idf = math.log((total_segments + 1) / df) + 1.0
                score = tf * idf
                scored.append((token, score))
                total_score += score
            scored.sort(key=lambda item: item[1], reverse=True)
            segment.keywords = [token for token, _ in scored[: self.config.max_keywords_per_segment]]
            attention_score = total_score / max(1, segment.token_count)
            scores.append(attention_score)
            segment.attention_score = attention_score

        max_score = max(scores) if scores else 0.0
        if max_score > 0:
            for segment in segments:
                segment.attention_score = round(segment.attention_score / max_score, 3)
        else:
            for segment in segments:
                segment.attention_score = 0.0

        return segments

    def _link_segments(
        self, segments: list[SpokenAttentionSegment]
    ) -> list[AttentionLink]:
        links: list[AttentionLink] = []
        keyword_sets = [set(segment.keywords) for segment in segments]
        for idx, segment in enumerate(segments):
            candidates: list[AttentionLink] = []
            start = max(0, idx - self.config.attention_window_segments)
            for prev_idx in range(start, idx):
                shared = keyword_sets[idx] & keyword_sets[prev_idx]
                if len(shared) < self.config.min_shared_keywords:
                    continue
                union = keyword_sets[idx] | keyword_sets[prev_idx]
                if not union:
                    continue
                similarity = len(shared) / len(union)
                if similarity < self.config.min_similarity:
                    continue
                candidates.append(
                    AttentionLink(
                        from_index=segment.index,
                        to_index=segments[prev_idx].index,
                        speaker_from=segment.speaker,
                        speaker_to=segments[prev_idx].speaker,
                        shared_keywords=sorted(shared),
                        similarity=round(similarity, 3),
                    )
                )
            candidates.sort(key=lambda link: link.similarity, reverse=True)
            links.extend(candidates[: self.config.max_links_per_segment])
            if len(links) >= self.config.max_links_total:
                break
        return links[: self.config.max_links_total]

    def _build_shifts(self, segments: list[SpokenAttentionSegment]) -> list[AttentionShift]:
        shifts: list[AttentionShift] = []
        keyword_sets = [set(segment.keywords) for segment in segments]
        for idx in range(1, len(segments)):
            prev = keyword_sets[idx - 1]
            curr = keyword_sets[idx]
            union = prev | curr
            similarity = len(prev & curr) / len(union) if union else 0.0
            shift_type = "steady"
            if similarity < self.config.shift_similarity_threshold:
                shift_type = "shift"
            shifts.append(
                AttentionShift(
                    from_index=segments[idx - 1].index,
                    to_index=segments[idx].index,
                    similarity=round(similarity, 3),
                    shift_type=shift_type,
                )
            )
        return shifts

    def _summarize_speakers(
        self, segments: list[SpokenAttentionSegment]
    ) -> list[SpeakerSummary]:
        speaker_tokens: dict[str, Counter[str]] = defaultdict(Counter)
        speaker_segments: dict[str, int] = defaultdict(int)
        speaker_attention: dict[str, list[float]] = defaultdict(list)

        for segment in segments:
            speaker = segment.speaker or "Unknown"
            speaker_segments[speaker] += 1
            speaker_attention[speaker].append(segment.attention_score)
            speaker_tokens[speaker].update(segment.keywords)

        summaries: list[SpeakerSummary] = []
        for speaker, seg_count in speaker_segments.items():
            tokens = speaker_tokens.get(speaker, Counter())
            total_tokens = sum(tokens.values())
            avg_attention = (
                sum(speaker_attention.get(speaker, [])) / max(1, seg_count)
            )
            summaries.append(
                SpeakerSummary(
                    speaker=speaker,
                    segment_count=seg_count,
                    total_tokens=total_tokens,
                    avg_attention=round(avg_attention, 3),
                    top_keywords=[token for token, _ in tokens.most_common(self.config.max_speaker_keywords)],
                )
            )
        summaries.sort(key=lambda item: item.segment_count, reverse=True)
        return summaries

    def _token_count(self, text: str) -> int:
        return len(WORD_RE.findall(text))

    def _load_item(
        self, item: SpokenAttentionItem
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
        if not isinstance(event.args, ContextSpokenAttentionDialogueArgs):
            return ToolCallDisplay(summary="context_spoken_attention_dialogue")
        if not event.args.items:
            return ToolCallDisplay(summary="context_spoken_attention_dialogue")
        first = event.args.items[0]
        return ToolCallDisplay(
            summary="context_spoken_attention_dialogue",
            details={
                "item_count": len(event.args.items),
                "id": first.id,
                "name": first.name,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpokenAttentionDialogueResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=(
                f"Analyzed {event.result.item_count} item(s) with "
                f"{len(event.result.items[0].segments) if event.result.items else 0} segments"
            ),
            warnings=event.result.warnings,
            details={
                "item_count": event.result.item_count,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Tracking spoken attention"
