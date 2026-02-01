from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
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
SPEAKER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _.'-]{0,40})\s*:\s*(.*)$")


@dataclass
class _SegmentData:
    segment: "VowelSegment"
    sequences: set[str]


class ContextSpokenVowelContextualizationConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_items: int = Field(default=40, description="Maximum dialogues to evaluate.")
    max_source_bytes: int = Field(default=3_000_000, description="Maximum bytes per item.")
    max_total_bytes: int = Field(default=20_000_000, description="Max bytes across items.")
    preview_chars: int = Field(default=400, description="Preview snippet length.")
    max_segments: int = Field(default=200, description="Maximum segments per dialogue.")
    min_segment_chars: int = Field(default=20, description="Minimum segment length.")
    min_token_length: int = Field(default=3, description="Minimum token length.")
    vowel_chars: str = Field(default="aeiouy", description="Vowels to track.")
    min_vowel_sequence_len: int = Field(default=1, description="Minimum vowel sequence length.")
    max_sequences_per_segment: int = Field(
        default=10, description="Maximum vowel sequences to keep per segment."
    )
    max_sequences_total: int = Field(default=500, description="Maximum vowel sequences to return.")
    max_words_per_sequence: int = Field(
        default=12, description="Maximum words listed per vowel sequence."
    )
    max_segments_per_sequence: int = Field(
        default=30, description="Maximum segment indices per vowel sequence."
    )
    min_shared_sequences: int = Field(
        default=2, description="Minimum shared sequences to link segments."
    )
    min_similarity: float = Field(
        default=0.15, description="Minimum similarity for segment links."
    )
    max_links_per_segment: int = Field(
        default=4, description="Maximum links per segment."
    )
    max_links_total: int = Field(default=500, description="Maximum total links.")
    temporal_window_segments: int = Field(
        default=3, description="Segments per temporal window."
    )
    max_temporal_windows: int = Field(
        default=50, description="Maximum temporal windows to return."
    )
    max_temporal_runs: int = Field(
        default=50, description="Maximum temporal sequence runs."
    )
    temporal_delta_threshold: float = Field(
        default=0.03, description="Threshold for vowel ratio trend changes."
    )


class ContextSpokenVowelContextualizationState(BaseToolState):
    pass


class SpokenVowelItem(BaseModel):
    id: str | None = Field(default=None)
    name: str | None = Field(default=None)
    dialogue_id: str | None = Field(default=None)
    content: str | None = Field(default=None)
    path: str | None = Field(default=None)
    source: str | None = Field(default=None)



class ContextSpokenVowelContextualizationArgs(BaseModel):
    items: list[SpokenVowelItem] = Field(description="Dialogues to evaluate.")
    include_temporal: bool = Field(
        default=True, description="Include temporal vowel analysis."
    )
    temporal_window_segments: int | None = Field(
        default=None, description="Override temporal window size."
    )


class VowelSegment(BaseModel):
    index: int
    speaker: str | None
    text: str
    start: int
    end: int
    token_count: int
    letter_count: int
    vowel_count: int
    vowel_ratio: float
    vowel_counts: dict[str, int]
    top_sequences: list[str]


class VowelTemporalStep(BaseModel):
    index: int
    vowel_ratio: float
    dominant_sequence: str | None
    delta_ratio: float
    trend: str


class VowelTemporalWindow(BaseModel):
    index: int
    start_index: int
    end_index: int
    average_ratio: float
    dominant_sequences: list[str]


class VowelSequenceRun(BaseModel):
    sequence: str
    start_index: int
    end_index: int
    length: int


class VowelSequenceContext(BaseModel):
    sequence: str
    total_count: int
    unique_words: int
    sample_words: list[str]
    segment_indices: list[int]
    segment_count: int


class VowelLink(BaseModel):
    from_index: int
    to_index: int
    shared_sequences: list[str]
    similarity: float


class SpeakerVowelSummary(BaseModel):
    speaker: str
    segment_count: int
    total_tokens: int
    letter_count: int
    vowel_count: int
    vowel_ratio: float
    top_sequences: list[str]


class SpokenVowelContextInsight(BaseModel):
    index: int
    id: str | None
    name: str | None
    dialogue_id: str
    source_path: str | None
    preview: str
    segments: list[VowelSegment]
    sequences: list[VowelSequenceContext]
    links: list[VowelLink]
    speakers: list[SpeakerVowelSummary]
    temporal_steps: list[VowelTemporalStep]
    temporal_windows: list[VowelTemporalWindow]
    temporal_runs: list[VowelSequenceRun]
    temporal_summary: str
    temporal_window_segments: int
    segment_count: int
    sequence_count: int
    link_count: int


class ContextSpokenVowelContextualizationResult(BaseModel):
    items: list[SpokenVowelContextInsight]
    item_count: int
    total_links: int
    truncated: bool
    warnings: list[str]
    errors: list[str]


class ContextSpokenVowelContextualization(
    BaseTool[
        ContextSpokenVowelContextualizationArgs,
        ContextSpokenVowelContextualizationResult,
        ContextSpokenVowelContextualizationConfig,
        ContextSpokenVowelContextualizationState,
    ],
    ToolUIData[
        ContextSpokenVowelContextualizationArgs,
        ContextSpokenVowelContextualizationResult,
    ],
):
    description: ClassVar[str] = (
        "Contextualize spoken vowels across multiple dialogues and segments."
    )

    async def run(
        self, args: ContextSpokenVowelContextualizationArgs
    ) -> ContextSpokenVowelContextualizationResult:
        items = list(args.items)
        if not items:
            raise ToolError("items is required.")

        errors: list[str] = []
        warnings: list[str] = []
        total_bytes = 0
        truncated = False

        insights: list[SpokenVowelContextInsight] = []
        total_links = 0

        if len(items) > self.config.max_items:
            warnings.append("Item limit reached; truncating input list.")
            items = items[: self.config.max_items]

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

                segments = self._segment_dialogue(content)
                if not segments:
                    raise ToolError("No segments found.")
                if len(segments) > self.config.max_segments:
                    segments = segments[: self.config.max_segments]
                    warnings.append("Segment limit reached; truncating dialogue.")

                sequences, sequence_lookup = self._build_sequence_context(segments)
                links = self._link_segments(segments)
                speakers = self._summarize_speakers(segments, sequence_lookup)
                temporal_steps: list[VowelTemporalStep] = []
                temporal_windows: list[VowelTemporalWindow] = []
                temporal_runs: list[VowelSequenceRun] = []
                temporal_summary = "Temporal analysis disabled."
                window_size = (
                    args.temporal_window_segments
                    if args.temporal_window_segments is not None
                    else self.config.temporal_window_segments
                )
                if window_size <= 0:
                    raise ToolError("temporal_window_segments must be positive.")
                if args.include_temporal:
                    (
                        temporal_steps,
                        temporal_windows,
                        temporal_runs,
                        temporal_summary,
                    ) = self._build_temporal_profile(
                        segments, sequence_lookup, window_size
                    )
                total_links += len(links)

                insights.append(
                    SpokenVowelContextInsight(
                        index=len(insights) + 1,
                        id=item.id,
                        name=item.name,
                        dialogue_id=self._dialogue_id(item, idx),
                        source_path=source_path,
                        preview=self._preview(content),
                        segments=segments,
                        sequences=sequences,
                        links=links,
                        speakers=speakers,
                        temporal_steps=temporal_steps,
                        temporal_windows=temporal_windows,
                        temporal_runs=temporal_runs,
                        temporal_summary=temporal_summary,
                        temporal_window_segments=window_size,
                        segment_count=len(segments),
                        sequence_count=len(sequences),
                        link_count=len(links),
                    )
                )
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not insights:
            raise ToolError("No insights generated.")

        return ContextSpokenVowelContextualizationResult(
            items=insights,
            item_count=len(insights),
            total_links=total_links,
            truncated=truncated,
            warnings=warnings,
            errors=errors,
        )

    def _dialogue_id(self, item: SpokenVowelItem, idx: int) -> str:
        if item.dialogue_id:
            return item.dialogue_id
        if item.id:
            return item.id
        if item.name:
            return item.name
        return f"dialogue_{idx}"

    def _segment_dialogue(self, text: str) -> list[VowelSegment]:
        segments: list[VowelSegment] = []
        buffer: list[str] = []
        current_speaker: str | None = None
        segment_start: int | None = None
        segment_end: int | None = None
        pos = 0

        def flush() -> None:
            nonlocal buffer, segment_start, segment_end, current_speaker
            if not buffer:
                return
            combined = " ".join(buffer).strip()
            if len(combined) >= self.config.min_segment_chars:
                segments.append(self._build_segment(combined, current_speaker, segment_start, segment_end))
            buffer = []
            segment_start = None
            segment_end = None

        for raw_line in text.splitlines(True):
            line = raw_line.rstrip("\r\n")
            line_start = pos
            pos += len(raw_line)
            line_end = pos
            match = SPEAKER_RE.match(line)
            if match:
                flush()
                current_speaker = match.group(1).strip()
                line_text = match.group(2).strip()
                segment_start = line_start
                segment_end = line_end
                if line_text:
                    buffer.append(line_text)
                continue
            if not line.strip():
                flush()
                continue
            if segment_start is None:
                segment_start = line_start
            segment_end = line_end
            buffer.append(line.strip())

        flush()
        if segments:
            return segments
        return self._fallback_segments(text)

    def _fallback_segments(self, text: str) -> list[VowelSegment]:
        chunks = [chunk for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
        segments: list[VowelSegment] = []
        cursor = 0
        for chunk in chunks:
            start = text.find(chunk, cursor)
            end = start + len(chunk)
            cursor = end
            seg_text = chunk.strip()
            if len(seg_text) < self.config.min_segment_chars:
                continue
            segments.append(self._build_segment(seg_text, None, start, end))
        return segments[: self.config.max_segments]

    def _build_segment(
        self, text: str, speaker: str | None, start: int | None, end: int | None
    ) -> VowelSegment:
        tokens = self._tokenize(text)
        vowel_counts = {vowel: 0 for vowel in self._vowel_set()}
        vowel_sequences = Counter()
        letter_count = 0
        vowel_total = 0

        for token in tokens:
            letter_count += len(token)
            for char in token:
                if char in vowel_counts:
                    vowel_counts[char] += 1
                    vowel_total += 1
            sequence = self._vowel_sequence(token)
            if len(sequence) >= self.config.min_vowel_sequence_len:
                vowel_sequences[sequence] += 1

        top_sequences = [
            seq for seq, _ in vowel_sequences.most_common(self.config.max_sequences_per_segment)
        ]
        vowel_ratio = vowel_total / letter_count if letter_count else 0.0

        return VowelSegment(
            index=0,
            speaker=speaker,
            text=text,
            start=start or 0,
            end=end or 0,
            token_count=len(tokens),
            letter_count=letter_count,
            vowel_count=vowel_total,
            vowel_ratio=round(vowel_ratio, 4),
            vowel_counts=vowel_counts,
            top_sequences=top_sequences,
        )

    def _build_sequence_context(
        self, segments: list[VowelSegment]
    ) -> tuple[list[VowelSequenceContext], dict[int, set[str]]]:
        sequence_totals: Counter[str] = Counter()
        sequence_words: dict[str, Counter[str]] = defaultdict(Counter)
        sequence_segments: dict[str, set[int]] = defaultdict(set)
        segment_sequences: dict[int, set[str]] = {}

        for idx, segment in enumerate(segments, start=1):
            segment.index = idx
            tokens = self._tokenize(segment.text)
            sequences: list[str] = []
            for token in tokens:
                sequence = self._vowel_sequence(token)
                if len(sequence) >= self.config.min_vowel_sequence_len:
                    sequence_totals[sequence] += 1
                    sequence_words[sequence][token] += 1
                    sequence_segments[sequence].add(idx)
                    sequences.append(sequence)
            segment_sequences[idx] = set(sequences)
            segment.top_sequences = [
                seq
                for seq, _ in Counter(sequences).most_common(
                    self.config.max_sequences_per_segment
                )
            ]

        contexts: list[VowelSequenceContext] = []
        for sequence, count in sequence_totals.most_common(self.config.max_sequences_total):
            word_counter = sequence_words.get(sequence, Counter())
            segment_indices = sorted(sequence_segments.get(sequence, set()))
            contexts.append(
                VowelSequenceContext(
                    sequence=sequence,
                    total_count=count,
                    unique_words=len(word_counter),
                    sample_words=[w for w, _ in word_counter.most_common(self.config.max_words_per_sequence)],
                    segment_indices=segment_indices[: self.config.max_segments_per_sequence],
                    segment_count=len(segment_indices),
                )
            )

        return contexts, segment_sequences

    def _link_segments(self, segments: list[VowelSegment]) -> list[VowelLink]:
        segment_sets: list[_SegmentData] = []
        for segment in segments:
            sequences = set(segment.top_sequences)
            segment_sets.append(_SegmentData(segment=segment, sequences=sequences))

        links: list[VowelLink] = []
        for idx, entry in enumerate(segment_sets):
            candidates: list[VowelLink] = []
            for jdx, other in enumerate(segment_sets):
                if idx == jdx:
                    continue
                shared = entry.sequences & other.sequences
                if len(shared) < self.config.min_shared_sequences:
                    continue
                union = entry.sequences | other.sequences
                if not union:
                    continue
                similarity = len(shared) / len(union)
                if similarity < self.config.min_similarity:
                    continue
                candidates.append(
                    VowelLink(
                        from_index=entry.segment.index,
                        to_index=other.segment.index,
                        shared_sequences=sorted(shared)[: self.config.max_sequences_per_segment],
                        similarity=round(similarity, 3),
                    )
                )
            candidates.sort(key=lambda link: link.similarity, reverse=True)
            for link in candidates[: self.config.max_links_per_segment]:
                links.append(link)
                if len(links) >= self.config.max_links_total:
                    return links
        return links

    def _summarize_speakers(
        self, segments: list[VowelSegment], segment_sequences: dict[int, set[str]]
    ) -> list[SpeakerVowelSummary]:
        by_speaker: dict[str, list[VowelSegment]] = defaultdict(list)
        for segment in segments:
            speaker = segment.speaker or "unknown"
            by_speaker[speaker].append(segment)

        summaries: list[SpeakerVowelSummary] = []
        for speaker, group in by_speaker.items():
            total_tokens = sum(segment.token_count for segment in group)
            letter_count = sum(segment.letter_count for segment in group)
            vowel_count = sum(segment.vowel_count for segment in group)
            vowel_ratio = vowel_count / letter_count if letter_count else 0.0
            sequence_counter = Counter()
            for segment in group:
                sequence_counter.update(segment_sequences.get(segment.index, set()))
            summaries.append(
                SpeakerVowelSummary(
                    speaker=speaker,
                    segment_count=len(group),
                    total_tokens=total_tokens,
                    letter_count=letter_count,
                    vowel_count=vowel_count,
                    vowel_ratio=round(vowel_ratio, 4),
                    top_sequences=[
                        seq for seq, _ in sequence_counter.most_common(self.config.max_sequences_per_segment)
                    ],
                )
            )

        summaries.sort(key=lambda summary: summary.total_tokens, reverse=True)
        return summaries

    def _build_temporal_profile(
        self,
        segments: list[VowelSegment],
        segment_sequences: dict[int, set[str]],
        window_size: int,
    ) -> tuple[
        list[VowelTemporalStep],
        list[VowelTemporalWindow],
        list[VowelSequenceRun],
        str,
    ]:
        if not segments:
            return [], [], [], "Temporal analysis not available."

        steps: list[VowelTemporalStep] = []
        previous_ratio: float | None = None
        threshold = self.config.temporal_delta_threshold
        for segment in segments:
            dominant = segment.top_sequences[0] if segment.top_sequences else None
            if previous_ratio is None:
                delta = 0.0
                trend = "start"
            else:
                delta = round(segment.vowel_ratio - previous_ratio, 4)
                if delta > threshold:
                    trend = "rise"
                elif delta < -threshold:
                    trend = "fall"
                else:
                    trend = "steady"
            steps.append(
                VowelTemporalStep(
                    index=segment.index,
                    vowel_ratio=segment.vowel_ratio,
                    dominant_sequence=dominant,
                    delta_ratio=delta,
                    trend=trend,
                )
            )
            previous_ratio = segment.vowel_ratio

        windows: list[VowelTemporalWindow] = []
        for idx in range(0, len(segments), window_size):
            window_segments = segments[idx : idx + window_size]
            if not window_segments:
                continue
            avg_ratio = sum(seg.vowel_ratio for seg in window_segments) / len(window_segments)
            sequence_counter = Counter()
            for seg in window_segments:
                sequence_counter.update(seg.top_sequences)
            dominant_sequences = [
                seq
                for seq, _ in sequence_counter.most_common(
                    self.config.max_sequences_per_segment
                )
            ]
            windows.append(
                VowelTemporalWindow(
                    index=len(windows) + 1,
                    start_index=window_segments[0].index,
                    end_index=window_segments[-1].index,
                    average_ratio=round(avg_ratio, 4),
                    dominant_sequences=dominant_sequences,
                )
            )
            if len(windows) >= self.config.max_temporal_windows:
                break

        sequence_map: dict[str, list[int]] = defaultdict(list)
        for idx, sequences in segment_sequences.items():
            for seq in sequences:
                sequence_map[seq].append(idx)

        runs: list[VowelSequenceRun] = []
        for sequence, indices in sequence_map.items():
            ordered = sorted(indices)
            if not ordered:
                continue
            start = ordered[0]
            prev = ordered[0]
            length = 1
            for current in ordered[1:]:
                if current == prev + 1:
                    length += 1
                else:
                    runs.append(
                        VowelSequenceRun(
                            sequence=sequence,
                            start_index=start,
                            end_index=prev,
                            length=length,
                        )
                    )
                    start = current
                    length = 1
                prev = current
            runs.append(
                VowelSequenceRun(
                    sequence=sequence,
                    start_index=start,
                    end_index=prev,
                    length=length,
                )
            )

        runs.sort(key=lambda run: (-run.length, run.sequence))
        if self.config.max_temporal_runs > 0 and len(runs) > self.config.max_temporal_runs:
            runs = runs[: self.config.max_temporal_runs]

        first_ratio = steps[0].vowel_ratio if steps else 0.0
        last_ratio = steps[-1].vowel_ratio if steps else 0.0
        delta_total = last_ratio - first_ratio
        if delta_total > threshold:
            trend_summary = "rising"
        elif delta_total < -threshold:
            trend_summary = "falling"
        else:
            trend_summary = "steady"

        longest_run = runs[0] if runs else None
        if longest_run:
            run_text = (
                f"longest run '{longest_run.sequence}' "
                f"segments {longest_run.start_index}-{longest_run.end_index}"
            )
        else:
            run_text = "no repeated vowel runs"

        summary = (
            f"Temporal vowel trend {trend_summary} "
            f"from {first_ratio:.3f} to {last_ratio:.3f}; {run_text}."
        )
        return steps, windows, runs, summary

    def _tokenize(self, text: str) -> list[str]:
        return [
            token.lower()
            for token in WORD_RE.findall(text)
            if len(token) >= self.config.min_token_length
        ]

    def _vowel_sequence(self, token: str) -> str:
        vowels = self._vowel_set()
        return "".join(char for char in token.lower() if char in vowels)

    def _vowel_set(self) -> set[str]:
        return {char for char in self.config.vowel_chars.lower() if char.isalpha()}

    def _load_item(self, item: SpokenVowelItem) -> tuple[str, str | None, int | None]:
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
        if not isinstance(event.args, ContextSpokenVowelContextualizationArgs):
            return ToolCallDisplay(summary="context_spoken_vowel_contextualization")
        if not event.args.items:
            return ToolCallDisplay(summary="context_spoken_vowel_contextualization")
        first = event.args.items[0]
        return ToolCallDisplay(
            summary="context_spoken_vowel_contextualization",
            details={
                "item_count": len(event.args.items),
                "id": first.id,
                "name": first.name,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpokenVowelContextualizationResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=(
                f"Analyzed {event.result.item_count} dialogue(s) with "
                f"{event.result.total_links} vowel links"
            ),
            warnings=event.result.warnings,
            details={
                "item_count": event.result.item_count,
                "total_links": event.result.total_links,
                "errors": event.result.errors,
                "temporal_summary": event.result.items[0].temporal_summary
                if event.result.items
                else None,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Contextualizing spoken vowels"
