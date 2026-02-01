from __future__ import annotations

import math
import re
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


WORD_RE = re.compile(r"[A-Za-z0-9_]+")
SPEAKER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _.'-]{0,40}):\s*(.*)$")


class ContextSpokenCorrelationConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_items: int = Field(default=40, description="Maximum items to evaluate.")
    max_source_bytes: int = Field(default=3_000_000, description="Maximum bytes per item.")
    max_total_bytes: int = Field(default=20_000_000, description="Max bytes across items.")
    preview_chars: int = Field(default=400, description="Preview snippet length.")
    max_segments: int = Field(default=200, description="Maximum segments per item.")
    min_segment_chars: int = Field(default=20, description="Minimum segment length.")
    min_token_length: int = Field(default=3, description="Minimum token length.")
    min_shared_tokens: int = Field(default=2, description="Minimum shared tokens to link.")
    min_similarity: float = Field(default=0.1, description="Minimum similarity threshold.")
    link_window_segments: int = Field(
        default=50,
        description="Segments to look back for correlations (0 = all).",
    )
    max_links_per_segment: int = Field(default=3, description="Maximum links per segment.")
    max_total_links: int = Field(default=1000, description="Maximum total links per item.")
    max_shared_tokens: int = Field(default=8, description="Maximum shared tokens to return.")


class ContextSpokenCorrelationState(BaseToolState):
    pass


class SpokenCorrelationItem(BaseModel):
    id: str | None = Field(default=None)
    name: str | None = Field(default=None)
    content: str | None = Field(default=None)
    path: str | None = Field(default=None)
    source: str | None = Field(default=None)



class ContextSpokenCorrelationArgs(BaseModel):
    items: list[SpokenCorrelationItem] = Field(description="Items to evaluate.")

class SpokenSegment(BaseModel):
    index: int
    speaker: str | None
    text: str
    start: int
    end: int
    token_count: int


class SpokenCorrelation(BaseModel):
    from_index: int
    to_index: int
    speaker_from: str | None
    speaker_to: str | None
    shared_tokens: list[str]
    jaccard: float
    cosine: float


class SpokenCorrelationInsight(BaseModel):
    index: int
    id: str | None
    name: str | None
    source_path: str | None
    preview: str
    segments: list[SpokenSegment]
    correlations: list[SpokenCorrelation]
    segment_count: int
    correlation_count: int


class ContextSpokenCorrelationResult(BaseModel):
    items: list[SpokenCorrelationInsight]
    item_count: int
    total_correlations: int
    truncated: bool
    warnings: list[str]
    errors: list[str]


class ContextSpokenCorrelation(
    BaseTool[
        ContextSpokenCorrelationArgs,
        ContextSpokenCorrelationResult,
        ContextSpokenCorrelationConfig,
        ContextSpokenCorrelationState,
    ],
    ToolUIData[ContextSpokenCorrelationArgs, ContextSpokenCorrelationResult],
):
    description: ClassVar[str] = (
        "Cross-correlate spoken words across a conversation."
    )

    async def run(self, args: ContextSpokenCorrelationArgs) -> ContextSpokenCorrelationResult:
        items = list(args.items)
        if not items:
            raise ToolError("items is required.")

        errors: list[str] = []
        warnings: list[str] = []
        insights: list[SpokenCorrelationInsight] = []
        total_bytes = 0
        truncated = False
        total_correlations = 0

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
                correlations = self._correlate_segments(segments)
                total_correlations += len(correlations)

                insights.append(
                    SpokenCorrelationInsight(
                        index=len(insights) + 1,
                        id=item.id,
                        name=item.name,
                        source_path=source_path,
                        preview=self._preview(content),
                        segments=segments[: self.config.max_segments],
                        correlations=correlations[: self.config.max_total_links],
                        segment_count=len(segments),
                        correlation_count=len(correlations),
                    )
                )
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not insights:
            raise ToolError("No insights generated.")

        return ContextSpokenCorrelationResult(
            items=insights,
            item_count=len(insights),
            total_correlations=total_correlations,
            truncated=truncated,
            warnings=warnings,
            errors=errors,
        )

    def _segment_conversation(self, text: str) -> list[SpokenSegment]:
        segments: list[SpokenSegment] = []
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
                    SpokenSegment(
                        index=len(segments) + 1,
                        speaker=current_speaker,
                        text=seg_text,
                        start=seg_start,
                        end=end_offset,
                        token_count=self._token_count(seg_text),
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

    def _correlate_segments(self, segments: list[SpokenSegment]) -> list[SpokenCorrelation]:
        correlations: list[SpokenCorrelation] = []
        token_sets = [self._token_set(seg.text) for seg in segments]
        token_freqs = [self._token_freq(seg.text) for seg in segments]

        for idx, seg in enumerate(segments):
            candidates: list[SpokenCorrelation] = []
            start = 0
            if self.config.link_window_segments > 0:
                start = max(0, idx - self.config.link_window_segments)
            for prev_idx in range(start, idx):
                shared = token_sets[idx] & token_sets[prev_idx]
                if len(shared) < self.config.min_shared_tokens:
                    continue
                union = token_sets[idx] | token_sets[prev_idx]
                if not union:
                    continue
                jaccard = len(shared) / len(union)
                cosine = self._cosine_similarity(token_freqs[idx], token_freqs[prev_idx])
                score = (jaccard + cosine) / 2.0
                if score < self.config.min_similarity:
                    continue
                candidates.append(
                    SpokenCorrelation(
                        from_index=seg.index,
                        to_index=segments[prev_idx].index,
                        speaker_from=seg.speaker,
                        speaker_to=segments[prev_idx].speaker,
                        shared_tokens=sorted(shared)[: self.config.max_shared_tokens],
                        jaccard=round(jaccard, 3),
                        cosine=round(cosine, 3),
                    )
                )
            candidates.sort(key=lambda c: (c.jaccard + c.cosine), reverse=True)
            correlations.extend(candidates[: self.config.max_links_per_segment])
            if len(correlations) >= self.config.max_total_links:
                break
        return correlations[: self.config.max_total_links]

    def _token_set(self, text: str) -> set[str]:
        return {
            token.lower()
            for token in WORD_RE.findall(text)
            if len(token) >= self.config.min_token_length
        }

    def _token_freq(self, text: str) -> dict[str, int]:
        freq: dict[str, int] = {}
        for token in WORD_RE.findall(text):
            token = token.lower()
            if len(token) < self.config.min_token_length:
                continue
            freq[token] = freq.get(token, 0) + 1
        return freq

    def _cosine_similarity(self, left: dict[str, int], right: dict[str, int]) -> float:
        if not left or not right:
            return 0.0
        shared = set(left) & set(right)
        numerator = sum(left[token] * right[token] for token in shared)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)

    def _token_count(self, text: str) -> int:
        return len(WORD_RE.findall(text))

    def _load_item(self, item: SpokenCorrelationItem) -> tuple[str, str | None, int | None]:
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
        if not isinstance(event.args, ContextSpokenCorrelationArgs):
            return ToolCallDisplay(summary="context_spoken_correlation")
        if not event.args.items:
            return ToolCallDisplay(summary="context_spoken_correlation")
        first = event.args.items[0]
        return ToolCallDisplay(
            summary="context_spoken_correlation",
            details={
                "item_count": len(event.args.items),
                "id": first.id,
                "name": first.name,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpokenCorrelationResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=(
                f"Analyzed {event.result.item_count} item(s) with "
                f"{event.result.total_correlations} correlations"
            ),
            warnings=event.result.warnings,
            details={
                "item_count": event.result.item_count,
                "total_correlations": event.result.total_correlations,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Correlating spoken words"
