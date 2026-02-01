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
SPEAKER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _.'-]{0,40})\s*:\s*(.*)$")
SENTENCE_RE = re.compile(r"[.!?]+")
NUM_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")

COMPLEX_PAIR_RE = re.compile(
    r"(?P<real>[+-]?\d+(?:\.\d+)?)\s*(?P<sign>[+-])\s*(?P<imag>\d+(?:\.\d+)?)\s*(?P<unit>[ij])\b"
)
IMAG_ONLY_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?P<sign>[+-]?)\s*(?P<imag>\d+(?:\.\d+)?)?\s*(?P<unit>[ij])\b"
)

PUNCT_KEYS = (".", ",", "?", "!", ":", ";", "-")


class ContextSpokenComplexNumbersConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_items: int = Field(default=40, description="Maximum dialogues to evaluate.")
    max_source_bytes: int = Field(default=3_000_000, description="Maximum bytes per item.")
    max_total_bytes: int = Field(default=20_000_000, description="Max bytes across items.")
    preview_chars: int = Field(default=400, description="Preview snippet length.")
    max_segments: int = Field(default=200, description="Maximum segments per dialogue.")
    min_segment_chars: int = Field(default=20, description="Minimum segment length.")
    min_token_length: int = Field(default=3, description="Minimum token length.")
    max_complex_per_segment: int = Field(default=30, description="Max complex hits per segment.")
    include_unlabeled_as_self: bool = Field(
        default=True, description="Treat unlabeled segments as self."
    )
    self_speaker_labels: list[str] = Field(
        default_factory=lambda: [
            "assistant",
            "ai",
            "model",
            "bot",
            "mistral",
            "vibe",
            "system",
        ],
        description="Speaker labels treated as self.",
    )
    allow_spoken_numbers: bool = Field(
        default=True,
        description="Parse spoken patterns like '3 plus 4 i'.",
    )
    allow_imag_only: bool = Field(
        default=True, description="Allow imaginary-only forms like '4i' or 'i'."
    )


class ContextSpokenComplexNumbersState(BaseToolState):
    pass


class SpokenComplexItem(BaseModel):
    id: str | None = Field(default=None)
    name: str | None = Field(default=None)
    dialogue_id: str | None = Field(default=None)
    content: str | None = Field(default=None)
    path: str | None = Field(default=None)
    source: str | None = Field(default=None)


class ContextSpokenComplexNumbersArgs(BaseModel):
    items: list[SpokenComplexItem] = Field(description="Dialogues to evaluate.")


class ComplexNumberHit(BaseModel):
    raw: str
    real: float
    imag: float
    magnitude: float
    phase_radians: float
    unit: str
    kind: str
    source: str
    start: int
    end: int


class ComplexSegment(BaseModel):
    index: int
    speaker: str | None
    text: str
    start: int
    end: int
    token_count: int
    unique_tokens: int
    avg_token_length: float
    avg_sentence_length: float
    uppercase_ratio: float
    punctuation_per_100: dict[str, float]
    complex_numbers: list[ComplexNumberHit]
    real_count: int
    imaginary_count: int
    complex_count: int


class ComplexSpeechProfile(BaseModel):
    segment_count: int
    total_complex: int
    total_real: int
    total_imaginary: int
    total_complex_parts: int
    top_units: list[str]
    top_real_values: list[float]
    top_imag_values: list[float]


class SpokenComplexInsight(BaseModel):
    index: int
    id: str | None
    name: str | None
    dialogue_id: str
    source_path: str | None
    preview: str
    segments: list[ComplexSegment]
    profile: ComplexSpeechProfile
    segment_count: int


class ContextSpokenComplexNumbersResult(BaseModel):
    items: list[SpokenComplexInsight]
    item_count: int
    truncated: bool
    warnings: list[str]
    errors: list[str]


class ContextSpokenComplexNumbers(
    BaseTool[
        ContextSpokenComplexNumbersArgs,
        ContextSpokenComplexNumbersResult,
        ContextSpokenComplexNumbersConfig,
        ContextSpokenComplexNumbersState,
    ],
    ToolUIData[
        ContextSpokenComplexNumbersArgs,
        ContextSpokenComplexNumbersResult,
    ],
):
    description: ClassVar[str] = (
        "Analyze complex numbers in spoken text and relate them to self speech patterns."
    )

    async def run(
        self, args: ContextSpokenComplexNumbersArgs
    ) -> ContextSpokenComplexNumbersResult:
        items = list(args.items)
        if not items:
            raise ToolError("items is required.")

        errors: list[str] = []
        warnings: list[str] = []
        total_bytes = 0
        truncated = False
        insights: list[SpokenComplexInsight] = []

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

                segments = self._select_self_segments(segments)
                if not segments:
                    raise ToolError("No self segments matched.")

                profile = self._build_profile(segments)
                insights.append(
                    SpokenComplexInsight(
                        index=len(insights) + 1,
                        id=item.id,
                        name=item.name,
                        dialogue_id=self._dialogue_id(item, idx),
                        source_path=source_path,
                        preview=self._preview(content),
                        segments=segments,
                        profile=profile,
                        segment_count=len(segments),
                    )
                )
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not insights:
            raise ToolError("No insights generated.")

        return ContextSpokenComplexNumbersResult(
            items=insights,
            item_count=len(insights),
            truncated=truncated,
            warnings=warnings,
            errors=errors,
        )

    def _dialogue_id(self, item: SpokenComplexItem, idx: int) -> str:
        if item.dialogue_id:
            return item.dialogue_id
        if item.id:
            return item.id
        if item.name:
            return item.name
        return f"dialogue_{idx}"

    def _segment_dialogue(self, text: str) -> list[ComplexSegment]:
        segments: list[ComplexSegment] = []
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
                segments.append(
                    self._build_segment(
                        combined,
                        current_speaker,
                        segment_start or 0,
                        segment_end or 0,
                        len(segments) + 1,
                    )
                )
            buffer = []
            segment_start = None
            segment_end = None

        for raw_line in text.splitlines(True):
            line = raw_line.rstrip("\r\n")
            line_start = pos
            pos += len(raw_line)
            segment_end = pos
            match = SPEAKER_RE.match(line)
            if match:
                flush()
                current_speaker = match.group(1).strip()
                line_text = match.group(2).strip()
                segment_start = line_start
                if line_text:
                    buffer.append(line_text)
                continue
            if not line.strip():
                flush()
                continue
            if segment_start is None:
                segment_start = line_start
            buffer.append(line.strip())

        flush()
        if segments:
            return segments

        chunks = [chunk for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
        for chunk in chunks:
            seg_text = chunk.strip()
            if len(seg_text) < self.config.min_segment_chars:
                continue
            segments.append(
                self._build_segment(seg_text, None, 0, 0, len(segments) + 1)
            )

        return segments[: self.config.max_segments]

    def _build_segment(
        self, text: str, speaker: str | None, start: int, end: int, index: int
    ) -> ComplexSegment:
        tokens = self._tokenize(text)
        token_count = len(tokens)
        unique_tokens = len(set(tokens))
        avg_token_length = self._avg_token_length(tokens)
        avg_sentence_length = self._avg_sentence_length(text)
        uppercase_ratio = self._uppercase_ratio(tokens)
        punctuation = self._punctuation_rates(text, token_count)
        complex_hits = self._extract_complex_numbers(text)
        real_count = sum(1 for hit in complex_hits if hit.kind == "real")
        imaginary_count = sum(1 for hit in complex_hits if hit.kind == "imaginary")
        complex_count = sum(1 for hit in complex_hits if hit.kind == "complex")

        return ComplexSegment(
            index=index,
            speaker=speaker,
            text=text,
            start=start,
            end=end,
            token_count=token_count,
            unique_tokens=unique_tokens,
            avg_token_length=round(avg_token_length, 3),
            avg_sentence_length=round(avg_sentence_length, 3),
            uppercase_ratio=round(uppercase_ratio, 4),
            punctuation_per_100=punctuation,
            complex_numbers=complex_hits[: self.config.max_complex_per_segment],
            real_count=real_count,
            imaginary_count=imaginary_count,
            complex_count=complex_count,
        )

    def _select_self_segments(
        self, segments: list[ComplexSegment]
    ) -> list[ComplexSegment]:
        label_set = {label.lower() for label in self.config.self_speaker_labels}
        has_speakers = any(segment.speaker for segment in segments)
        matched = [
            segment
            for segment in segments
            if segment.speaker and segment.speaker.strip().lower() in label_set
        ]
        if matched:
            return matched
        if not has_speakers and self.config.include_unlabeled_as_self:
            return segments
        unlabeled = [segment for segment in segments if segment.speaker is None]
        if unlabeled and self.config.include_unlabeled_as_self:
            return unlabeled
        return []

    def _build_profile(self, segments: list[ComplexSegment]) -> ComplexSpeechProfile:
        total_complex = 0
        total_real = 0
        total_imaginary = 0
        units = Counter()
        real_vals = Counter()
        imag_vals = Counter()

        for segment in segments:
            for hit in segment.complex_numbers:
                total_complex += 1
                if hit.kind == "real":
                    total_real += 1
                elif hit.kind == "imaginary":
                    total_imaginary += 1
                units[hit.unit] += 1
                real_vals[round(hit.real, 3)] += 1
                imag_vals[round(hit.imag, 3)] += 1

        return ComplexSpeechProfile(
            segment_count=len(segments),
            total_complex=total_complex,
            total_real=total_real,
            total_imaginary=total_imaginary,
            total_complex_parts=total_complex,
            top_units=[unit for unit, _ in units.most_common(5)],
            top_real_values=[value for value, _ in real_vals.most_common(5)],
            top_imag_values=[value for value, _ in imag_vals.most_common(5)],
        )

    def _extract_complex_numbers(self, text: str) -> list[ComplexNumberHit]:
        hits: list[ComplexNumberHit] = []
        used_spans: list[tuple[int, int]] = []

        for match in COMPLEX_PAIR_RE.finditer(text):
            real = float(match.group("real"))
            imag = float(match.group("imag"))
            sign = match.group("sign")
            if sign == "-":
                imag = -imag
            unit = match.group("unit")
            hits.append(
                self._build_hit(
                    raw=match.group(0),
                    real=real,
                    imag=imag,
                    unit=unit,
                    start=match.start(),
                    end=match.end(),
                    source="numeric",
                )
            )
            used_spans.append((match.start(), match.end()))

        if self.config.allow_imag_only:
            for match in IMAG_ONLY_RE.finditer(text):
                if self._overlaps(match.start(), match.end(), used_spans):
                    continue
                imag_text = match.group("imag") or "1"
                imag = float(imag_text)
                sign = match.group("sign")
                if sign == "-":
                    imag = -imag
                unit = match.group("unit")
                hits.append(
                    self._build_hit(
                        raw=match.group(0),
                        real=0.0,
                        imag=imag,
                        unit=unit,
                        start=match.start(),
                        end=match.end(),
                        source="numeric",
                    )
                )
                used_spans.append((match.start(), match.end()))

        if self.config.allow_spoken_numbers:
            hits.extend(self._extract_spoken_complex(text, used_spans))

        return hits

    def _extract_spoken_complex(
        self, text: str, used_spans: list[tuple[int, int]]
    ) -> list[ComplexNumberHit]:
        hits: list[ComplexNumberHit] = []
        tokens = [
            {
                "token": match.group(0),
                "lower": match.group(0).lower(),
                "start": match.start(),
                "end": match.end(),
            }
            for match in WORD_RE.finditer(text)
        ]
        if not tokens:
            return hits

        for idx in range(len(tokens) - 3):
            first = tokens[idx]
            second = tokens[idx + 1]
            third = tokens[idx + 2]
            fourth = tokens[idx + 3]
            if not NUM_RE.match(first["token"]):
                continue
            if second["lower"] not in {"plus", "minus"}:
                continue
            if not NUM_RE.match(third["token"]):
                continue
            if fourth["lower"] not in {"i", "j"}:
                continue

            real = float(first["token"])
            imag = float(third["token"])
            if second["lower"] == "minus":
                imag = -imag
            start = first["start"]
            end = fourth["end"]
            if self._overlaps(start, end, used_spans):
                continue
            raw = text[start:end]
            hits.append(
                self._build_hit(
                    raw=raw,
                    real=real,
                    imag=imag,
                    unit=fourth["lower"],
                    start=start,
                    end=end,
                    source="spoken",
                )
            )
            used_spans.append((start, end))

        for idx in range(len(tokens) - 1):
            first = tokens[idx]
            second = tokens[idx + 1]
            if not NUM_RE.match(first["token"]):
                continue
            if second["lower"] not in {"i", "j"}:
                continue
            start = first["start"]
            end = second["end"]
            if self._overlaps(start, end, used_spans):
                continue
            hits.append(
                self._build_hit(
                    raw=text[start:end],
                    real=0.0,
                    imag=float(first["token"]),
                    unit=second["lower"],
                    start=start,
                    end=end,
                    source="spoken",
                )
            )
            used_spans.append((start, end))

        return hits

    def _build_hit(
        self,
        raw: str,
        real: float,
        imag: float,
        unit: str,
        start: int,
        end: int,
        source: str,
    ) -> ComplexNumberHit:
        magnitude = math.sqrt(real ** 2 + imag ** 2)
        phase = math.atan2(imag, real)
        if imag == 0:
            kind = "real"
        elif real == 0:
            kind = "imaginary"
        else:
            kind = "complex"
        return ComplexNumberHit(
            raw=raw.strip(),
            real=real,
            imag=imag,
            magnitude=round(magnitude, 6),
            phase_radians=round(phase, 6),
            unit=unit,
            kind=kind,
            source=source,
            start=start,
            end=end,
        )

    def _overlaps(self, start: int, end: int, spans: list[tuple[int, int]]) -> bool:
        for span_start, span_end in spans:
            if start < span_end and end > span_start:
                return True
        return False

    def _tokenize(self, text: str) -> list[str]:
        return [
            token.lower()
            for token in WORD_RE.findall(text)
            if len(token) >= self.config.min_token_length
        ]

    def _avg_token_length(self, tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        return sum(len(token) for token in tokens) / len(tokens)

    def _sentence_lengths(self, text: str) -> list[int]:
        lengths: list[int] = []
        for sentence in SENTENCE_RE.split(text):
            tokens = self._tokenize(sentence)
            if tokens:
                lengths.append(len(tokens))
        return lengths

    def _avg_sentence_length(self, text: str) -> float:
        lengths = self._sentence_lengths(text)
        if not lengths:
            return 0.0
        return sum(lengths) / len(lengths)

    def _uppercase_ratio(self, tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        count = sum(1 for token in tokens if token.isupper() and len(token) > 1)
        return count / len(tokens)

    def _punctuation_rates(self, text: str, token_count: int) -> dict[str, float]:
        counts = Counter({key: 0 for key in PUNCT_KEYS})
        for key in PUNCT_KEYS:
            counts[key] = text.count(key)
        scale = 100.0 / max(token_count, 1)
        return {key: round(counts[key] * scale, 3) for key in PUNCT_KEYS}

    def _load_item(self, item: SpokenComplexItem) -> tuple[str, str | None, int | None]:
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
        if not isinstance(event.args, ContextSpokenComplexNumbersArgs):
            return ToolCallDisplay(summary="context_spoken_complex_numbers")
        if not event.args.items:
            return ToolCallDisplay(summary="context_spoken_complex_numbers")
        first = event.args.items[0]
        return ToolCallDisplay(
            summary="context_spoken_complex_numbers",
            details={
                "item_count": len(event.args.items),
                "id": first.id,
                "name": first.name,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpokenComplexNumbersResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=(
                f"Analyzed {event.result.item_count} dialogue(s) for complex numbers"
            ),
            warnings=event.result.warnings,
            details={
                "item_count": event.result.item_count,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Analyzing complex numbers in speech"
