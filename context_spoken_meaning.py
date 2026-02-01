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


def _tokenize(text: str, min_len: int) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text) if len(token) >= min_len]


class ContextSpokenMeaningConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_items: int = Field(default=40, description="Maximum items to evaluate.")
    max_source_bytes: int = Field(default=3_000_000, description="Maximum bytes per item.")
    max_total_bytes: int = Field(default=20_000_000, description="Max bytes across items.")
    preview_chars: int = Field(default=400, description="Preview snippet length.")
    max_segments: int = Field(default=200, description="Maximum segments per item.")
    min_segment_chars: int = Field(default=20, description="Minimum segment length.")
    min_token_length: int = Field(default=3, description="Minimum token length.")
    max_keywords: int = Field(default=12, description="Maximum keywords per segment.")
    min_shared_keywords: int = Field(default=2, description="Minimum shared keywords to link.")
    min_similarity: float = Field(default=0.1, description="Minimum similarity threshold.")
    link_window_segments: int = Field(
        default=5, description="Segments to look back for context links."
    )
    max_links_per_segment: int = Field(default=3, description="Maximum links per segment.")
    max_links_total: int = Field(default=500, description="Maximum links per item.")


class ContextSpokenMeaningState(BaseToolState):
    pass


class SpokenMeaningItem(BaseModel):
    id: str | None = Field(default=None)
    name: str | None = Field(default=None)
    content: str | None = Field(default=None)
    path: str | None = Field(default=None)
    source: str | None = Field(default=None)



class ContextSpokenMeaningArgs(BaseModel):
    items: list[SpokenMeaningItem] = Field(description="Items to evaluate.")

class SpokenMeaningSegment(BaseModel):
    index: int
    speaker: str | None
    text: str
    start: int
    end: int
    token_count: int
    keywords: list[str]


class MeaningLink(BaseModel):
    from_index: int
    to_index: int
    speaker_from: str | None
    speaker_to: str | None
    shared_keywords: list[str]
    similarity: float


class SpokenMeaningInsight(BaseModel):
    index: int
    id: str | None
    name: str | None
    source_path: str | None
    preview: str
    segments: list[SpokenMeaningSegment]
    links: list[MeaningLink]
    segment_count: int
    link_count: int


class ContextSpokenMeaningResult(BaseModel):
    items: list[SpokenMeaningInsight]
    item_count: int
    total_links: int
    truncated: bool
    warnings: list[str]
    errors: list[str]


class ContextSpokenMeaning(
    BaseTool[
        ContextSpokenMeaningArgs,
        ContextSpokenMeaningResult,
        ContextSpokenMeaningConfig,
        ContextSpokenMeaningState,
    ],
    ToolUIData[ContextSpokenMeaningArgs, ContextSpokenMeaningResult],
):
    description: ClassVar[str] = (
        "Understand contextual meaning across words in spoken dialogue."
    )

    async def run(self, args: ContextSpokenMeaningArgs) -> ContextSpokenMeaningResult:
        items = list(args.items)
        if not items:
            raise ToolError("items is required.")

        errors: list[str] = []
        warnings: list[str] = []
        insights: list[SpokenMeaningInsight] = []
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
                total_links += len(links)

                insights.append(
                    SpokenMeaningInsight(
                        index=len(insights) + 1,
                        id=item.id,
                        name=item.name,
                        source_path=source_path,
                        preview=self._preview(content),
                        segments=segments[: self.config.max_segments],
                        links=links[: self.config.max_links_total],
                        segment_count=len(segments),
                        link_count=len(links),
                    )
                )
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not insights:
            raise ToolError("No insights generated.")

        return ContextSpokenMeaningResult(
            items=insights,
            item_count=len(insights),
            total_links=total_links,
            truncated=truncated,
            warnings=warnings,
            errors=errors,
        )

    def _segment_conversation(self, text: str) -> list[SpokenMeaningSegment]:
        segments: list[SpokenMeaningSegment] = []
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
                    SpokenMeaningSegment(
                        index=len(segments) + 1,
                        speaker=current_speaker,
                        text=seg_text,
                        start=seg_start,
                        end=end_offset,
                        token_count=self._token_count(seg_text),
                        keywords=[],
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

    def _attach_keywords(self, segments: list[SpokenMeaningSegment]) -> list[SpokenMeaningSegment]:
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

    def _link_segments(self, segments: list[SpokenMeaningSegment]) -> list[MeaningLink]:
        links: list[MeaningLink] = []
        keyword_sets = [set(segment.keywords) for segment in segments]
        for idx, segment in enumerate(segments):
            candidates: list[MeaningLink] = []
            start = max(0, idx - self.config.link_window_segments)
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
                    MeaningLink(
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

    def _token_count(self, text: str) -> int:
        return len(WORD_RE.findall(text))

    def _load_item(self, item: SpokenMeaningItem) -> tuple[str, str | None, int | None]:
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
        if not isinstance(event.args, ContextSpokenMeaningArgs):
            return ToolCallDisplay(summary="context_spoken_meaning")
        if not event.args.items:
            return ToolCallDisplay(summary="context_spoken_meaning")
        first = event.args.items[0]
        return ToolCallDisplay(
            summary="context_spoken_meaning",
            details={
                "item_count": len(event.args.items),
                "id": first.id,
                "name": first.name,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpokenMeaningResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=(
                f"Analyzed {event.result.item_count} item(s) with "
                f"{event.result.total_links} context link(s)"
            ),
            warnings=event.result.warnings,
            details={
                "item_count": event.result.item_count,
                "total_links": event.result.total_links,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Linking spoken meaning"
