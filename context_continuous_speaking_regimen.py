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
SPEAKER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _.'-]{0,40})\s*:\s*(.*)$")


class ContextContinuousSpeakingRegimenConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_items: int = Field(default=40, description="Maximum dialogues to evaluate.")
    max_source_bytes: int = Field(default=3_000_000, description="Maximum bytes per item.")
    max_total_bytes: int = Field(default=20_000_000, description="Max bytes across items.")
    preview_chars: int = Field(default=400, description="Preview snippet length.")
    max_segments: int = Field(default=300, description="Maximum segments per dialogue.")
    min_segment_chars: int = Field(default=20, description="Minimum segment length.")
    min_token_length: int = Field(default=3, description="Minimum token length.")
    window_size: int = Field(default=4, description="Segments per continuous window.")
    window_step: int = Field(default=2, description="Step size for moving window.")
    max_windows: int = Field(default=200, description="Maximum windows per dialogue.")
    max_tokens_per_window: int = Field(default=800, description="Max tokens per window.")
    max_keywords_per_segment: int = Field(default=10, description="Max keywords per segment.")
    max_keywords_per_window: int = Field(default=15, description="Max keywords per window.")
    min_style_hits: int = Field(default=1, description="Minimum hits to enable a style.")
    min_style_ratio: float = Field(default=0.02, description="Minimum ratio to enable style.")
    min_shared_tokens: int = Field(default=2, description="Minimum shared tokens for links.")
    min_shared_styles: int = Field(default=1, description="Minimum shared styles for links.")
    min_similarity: float = Field(default=0.12, description="Minimum link similarity.")
    token_weight: float = Field(default=0.7, description="Token similarity weight.")
    style_weight: float = Field(default=0.3, description="Style similarity weight.")
    max_links_per_window: int = Field(default=4, description="Maximum links per window.")
    max_links_total: int = Field(default=500, description="Maximum total links.")
    formal_markers: list[str] = Field(
        default_factory=lambda: [
            "therefore",
            "however",
            "moreover",
            "thus",
            "hence",
            "consequently",
            "furthermore",
            "whereas",
        ],
        description="Markers for formal speaking style.",
    )
    casual_markers: list[str] = Field(
        default_factory=lambda: [
            "gonna",
            "wanna",
            "kinda",
            "sorta",
            "yeah",
            "yep",
            "nah",
            "ok",
            "okay",
            "cool",
            "lol",
        ],
        description="Markers for casual speaking style.",
    )
    emphatic_markers: list[str] = Field(
        default_factory=lambda: [
            "very",
            "really",
            "extremely",
            "absolutely",
            "definitely",
            "must",
            "always",
            "never",
            "important",
            "urgent",
            "listen",
            "please",
        ],
        description="Markers for emphatic speaking style.",
    )
    narrative_markers: list[str] = Field(
        default_factory=lambda: [
            "then",
            "after",
            "before",
            "when",
            "while",
            "suddenly",
            "finally",
            "meanwhile",
            "later",
        ],
        description="Markers for narrative speaking style.",
    )
    technical_markers: list[str] = Field(
        default_factory=lambda: [
            "function",
            "variable",
            "system",
            "process",
            "module",
            "interface",
            "parameter",
            "compile",
            "runtime",
            "algorithm",
        ],
        description="Markers for technical speaking style.",
    )
    persuasive_markers: list[str] = Field(
        default_factory=lambda: [
            "should",
            "need",
            "because",
            "therefore",
            "so",
            "consider",
            "recommend",
            "suggest",
        ],
        description="Markers for persuasive speaking style.",
    )
    interrogative_markers: list[str] = Field(
        default_factory=lambda: [
            "why",
            "what",
            "how",
            "when",
            "where",
            "who",
            "which",
        ],
        description="Markers for interrogative speaking style.",
    )


class ContextContinuousSpeakingRegimenState(BaseToolState):
    pass


class ContinuousSpeakingItem(BaseModel):
    id: str | None = Field(default=None)
    name: str | None = Field(default=None)
    dialogue_id: str | None = Field(default=None)
    content: str | None = Field(default=None)
    path: str | None = Field(default=None)
    source: str | None = Field(default=None)


class ContextContinuousSpeakingRegimenArgs(BaseModel):
    items: list[ContinuousSpeakingItem] = Field(description="Dialogues to evaluate.")


class WordFormCounts(BaseModel):
    total_tokens: int
    uppercase_tokens: int
    numeric_tokens: int
    alnum_tokens: int
    contractions: int
    hyphenated: int
    repeated_chars: int
    acronym_tokens: int


class SpeakingSegment(BaseModel):
    index: int
    speaker: str | None
    text: str
    start: int
    end: int
    token_count: int
    unique_tokens: int
    keywords: list[str]
    word_forms: WordFormCounts
    style_scores: dict[str, float]
    active_styles: list[str]


class RegimenWindow(BaseModel):
    index: int
    segment_indices: list[int]
    token_count: int
    unique_tokens: int
    keywords: list[str]
    word_forms: WordFormCounts
    style_scores: dict[str, float]
    active_styles: list[str]


class RegimenLink(BaseModel):
    from_index: int
    to_index: int
    shared_tokens: list[str]
    shared_styles: list[str]
    token_similarity: float
    style_similarity: float
    combined_score: float


class RegimenTransition(BaseModel):
    from_index: int
    to_index: int
    from_styles: list[str]
    to_styles: list[str]
    change_type: str


class ContinuousSpeakingInsight(BaseModel):
    index: int
    id: str | None
    name: str | None
    dialogue_id: str
    source_path: str | None
    preview: str
    segments: list[SpeakingSegment]
    windows: list[RegimenWindow]
    links: list[RegimenLink]
    transitions: list[RegimenTransition]
    segment_count: int
    window_count: int
    link_count: int


class ContextContinuousSpeakingRegimenResult(BaseModel):
    items: list[ContinuousSpeakingInsight]
    item_count: int
    total_links: int
    truncated: bool
    warnings: list[str]
    errors: list[str]


class ContextContinuousSpeakingRegimen(
    BaseTool[
        ContextContinuousSpeakingRegimenArgs,
        ContextContinuousSpeakingRegimenResult,
        ContextContinuousSpeakingRegimenConfig,
        ContextContinuousSpeakingRegimenState,
    ],
    ToolUIData[
        ContextContinuousSpeakingRegimenArgs,
        ContextContinuousSpeakingRegimenResult,
    ],
):
    description: ClassVar[str] = (
        "Correlate continuous speaking regimens across spoken word forms and styles."
    )

    async def run(
        self, args: ContextContinuousSpeakingRegimenArgs
    ) -> ContextContinuousSpeakingRegimenResult:
        items = list(args.items)
        if not items:
            raise ToolError("items is required.")

        errors: list[str] = []
        warnings: list[str] = []
        total_bytes = 0
        truncated = False
        insights: list[ContinuousSpeakingInsight] = []
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

                windows = self._build_windows(segments)
                links = self._link_windows(windows)
                transitions = self._build_transitions(windows)
                total_links += len(links)

                insights.append(
                    ContinuousSpeakingInsight(
                        index=len(insights) + 1,
                        id=item.id,
                        name=item.name,
                        dialogue_id=self._dialogue_id(item, idx),
                        source_path=source_path,
                        preview=self._preview(content),
                        segments=segments,
                        windows=windows,
                        links=links,
                        transitions=transitions,
                        segment_count=len(segments),
                        window_count=len(windows),
                        link_count=len(links),
                    )
                )
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not insights:
            raise ToolError("No insights generated.")

        return ContextContinuousSpeakingRegimenResult(
            items=insights,
            item_count=len(insights),
            total_links=total_links,
            truncated=truncated,
            warnings=warnings,
            errors=errors,
        )

    def _dialogue_id(self, item: ContinuousSpeakingItem, idx: int) -> str:
        if item.dialogue_id:
            return item.dialogue_id
        if item.id:
            return item.id
        if item.name:
            return item.name
        return f"dialogue_{idx}"

    def _segment_dialogue(self, text: str) -> list[SpeakingSegment]:
        segments: list[SpeakingSegment] = []
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
    ) -> SpeakingSegment:
        tokens = self._tokenize(text)
        keyword_list = self._top_keywords(tokens, self.config.max_keywords_per_segment)
        word_forms = self._word_forms(tokens)
        style_scores, active_styles = self._style_scores(tokens, text)
        return SpeakingSegment(
            index=index,
            speaker=speaker,
            text=text,
            start=start,
            end=end,
            token_count=len(tokens),
            unique_tokens=len(set(tokens)),
            keywords=keyword_list,
            word_forms=word_forms,
            style_scores=style_scores,
            active_styles=active_styles,
        )

    def _build_windows(self, segments: list[SpeakingSegment]) -> list[RegimenWindow]:
        window_size = max(1, self.config.window_size)
        step = self.config.window_step if self.config.window_step > 0 else window_size
        windows: list[RegimenWindow] = []

        if len(segments) <= window_size:
            windows.append(self._window_from_segments(segments, 1))
            return windows

        window_index = 1
        for start in range(0, len(segments) - window_size + 1, step):
            slice_segments = segments[start : start + window_size]
            windows.append(self._window_from_segments(slice_segments, window_index))
            window_index += 1
            if len(windows) >= self.config.max_windows:
                break

        return windows

    def _window_from_segments(
        self, segments: list[SpeakingSegment], index: int
    ) -> RegimenWindow:
        token_counts = Counter()
        style_counts = Counter()
        word_form_totals = Counter()
        for segment in segments:
            token_counts.update(self._tokenize(segment.text))
            style_counts.update(segment.active_styles)
            word_form_totals.update(segment.word_forms.model_dump())

        tokens = list(token_counts.elements())
        keywords = self._top_keywords(tokens, self.config.max_keywords_per_window)
        style_scores = self._style_score_from_counts(token_counts, style_counts)
        active_styles = self._active_styles(style_scores)

        word_forms = WordFormCounts(
            total_tokens=word_form_totals.get("total_tokens", 0),
            uppercase_tokens=word_form_totals.get("uppercase_tokens", 0),
            numeric_tokens=word_form_totals.get("numeric_tokens", 0),
            alnum_tokens=word_form_totals.get("alnum_tokens", 0),
            contractions=word_form_totals.get("contractions", 0),
            hyphenated=word_form_totals.get("hyphenated", 0),
            repeated_chars=word_form_totals.get("repeated_chars", 0),
            acronym_tokens=word_form_totals.get("acronym_tokens", 0),
        )

        segment_indices = [segment.index for segment in segments]
        token_count = sum(segment.token_count for segment in segments)
        token_count = min(token_count, self.config.max_tokens_per_window)
        return RegimenWindow(
            index=index,
            segment_indices=segment_indices,
            token_count=token_count,
            unique_tokens=len(set(tokens)),
            keywords=keywords,
            word_forms=word_forms,
            style_scores=style_scores,
            active_styles=active_styles,
        )

    def _link_windows(self, windows: list[RegimenWindow]) -> list[RegimenLink]:
        links: list[RegimenLink] = []
        token_sets = [set(window.keywords) for window in windows]
        style_sets = [set(window.active_styles) for window in windows]
        for idx, window in enumerate(windows):
            candidates: list[RegimenLink] = []
            for jdx, other in enumerate(windows):
                if idx == jdx:
                    continue
                shared_tokens = token_sets[idx] & token_sets[jdx]
                if len(shared_tokens) < self.config.min_shared_tokens:
                    continue
                shared_styles = style_sets[idx] & style_sets[jdx]
                if len(shared_styles) < self.config.min_shared_styles:
                    continue
                token_union = token_sets[idx] | token_sets[jdx]
                token_similarity = (
                    len(shared_tokens) / len(token_union) if token_union else 0.0
                )
                style_union = style_sets[idx] | style_sets[jdx]
                style_similarity = (
                    len(shared_styles) / len(style_union) if style_union else 0.0
                )
                combined = (
                    token_similarity * self.config.token_weight
                    + style_similarity * self.config.style_weight
                )
                if combined < self.config.min_similarity:
                    continue
                candidates.append(
                    RegimenLink(
                        from_index=window.index,
                        to_index=other.index,
                        shared_tokens=sorted(shared_tokens)[: self.config.max_keywords_per_window],
                        shared_styles=sorted(shared_styles),
                        token_similarity=round(token_similarity, 3),
                        style_similarity=round(style_similarity, 3),
                        combined_score=round(combined, 3),
                    )
                )
            candidates.sort(key=lambda link: link.combined_score, reverse=True)
            for link in candidates[: self.config.max_links_per_window]:
                links.append(link)
                if len(links) >= self.config.max_links_total:
                    return links
        return links

    def _build_transitions(self, windows: list[RegimenWindow]) -> list[RegimenTransition]:
        transitions: list[RegimenTransition] = []
        for idx in range(1, len(windows)):
            prev = windows[idx - 1]
            curr = windows[idx]
            prev_styles = set(prev.active_styles)
            curr_styles = set(curr.active_styles)
            if prev_styles == curr_styles:
                continue
            change_type = "shift"
            if prev_styles and not curr_styles:
                change_type = "style_drop"
            elif curr_styles and not prev_styles:
                change_type = "style_rise"
            transitions.append(
                RegimenTransition(
                    from_index=prev.index,
                    to_index=curr.index,
                    from_styles=sorted(prev_styles),
                    to_styles=sorted(curr_styles),
                    change_type=change_type,
                )
            )
        return transitions

    def _style_scores(self, tokens: list[str], text: str) -> tuple[dict[str, float], list[str]]:
        counter = Counter(tokens)
        style_hits = {
            "formal": self._count_markers(counter, self.config.formal_markers),
            "casual": self._count_markers(counter, self.config.casual_markers),
            "emphatic": self._count_markers(counter, self.config.emphatic_markers),
            "narrative": self._count_markers(counter, self.config.narrative_markers),
            "technical": self._count_markers(counter, self.config.technical_markers),
            "persuasive": self._count_markers(counter, self.config.persuasive_markers),
            "interrogative": self._count_markers(
                counter, self.config.interrogative_markers
            ),
        }
        if "?" in text:
            style_hits["interrogative"] += text.count("?")
        if "!" in text:
            style_hits["emphatic"] += text.count("!")

        scores = self._style_score_from_counts(counter, style_hits)
        active_styles = self._active_styles(scores)
        return scores, active_styles

    def _style_score_from_counts(
        self, counter: Counter[str], style_hits: Counter[str] | dict[str, int]
    ) -> dict[str, float]:
        total_tokens = max(sum(counter.values()), 1)
        scores: dict[str, float] = {}
        for style, hits in style_hits.items():
            scores[style] = round(hits / total_tokens, 4)
        return scores

    def _active_styles(self, scores: dict[str, float]) -> list[str]:
        active: list[str] = []
        for style, score in scores.items():
            if score >= self.config.min_style_ratio:
                active.append(style)
        for style, score in scores.items():
            if score > 0 and style not in active:
                active.append(style)
        return sorted(set(active))

    def _word_forms(self, tokens: list[str]) -> WordFormCounts:
        uppercase_tokens = 0
        numeric_tokens = 0
        alnum_tokens = 0
        contractions = 0
        hyphenated = 0
        repeated_chars = 0
        acronym_tokens = 0
        for token in tokens:
            if token.isupper() and len(token) > 1:
                uppercase_tokens += 1
                if len(token) <= 5:
                    acronym_tokens += 1
            if token.isdigit():
                numeric_tokens += 1
            if any(char.isdigit() for char in token) and any(char.isalpha() for char in token):
                alnum_tokens += 1
            if "'" in token:
                contractions += 1
            if "-" in token:
                hyphenated += 1
            if self._has_repeated_chars(token):
                repeated_chars += 1
        return WordFormCounts(
            total_tokens=len(tokens),
            uppercase_tokens=uppercase_tokens,
            numeric_tokens=numeric_tokens,
            alnum_tokens=alnum_tokens,
            contractions=contractions,
            hyphenated=hyphenated,
            repeated_chars=repeated_chars,
            acronym_tokens=acronym_tokens,
        )

    def _has_repeated_chars(self, token: str) -> bool:
        last = ""
        streak = 1
        for char in token.lower():
            if char == last:
                streak += 1
                if streak >= 3:
                    return True
            else:
                last = char
                streak = 1
        return False

    def _count_markers(self, counter: Counter[str], markers: list[str]) -> int:
        return sum(counter.get(marker, 0) for marker in markers)

    def _tokenize(self, text: str) -> list[str]:
        return [
            token.lower()
            for token in WORD_RE.findall(text)
            if len(token) >= self.config.min_token_length
        ]

    def _top_keywords(self, tokens: list[str], max_items: int) -> list[str]:
        return [word for word, _ in Counter(tokens).most_common(max_items)]

    def _load_item(self, item: ContinuousSpeakingItem) -> tuple[str, str | None, int | None]:
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
        if not isinstance(event.args, ContextContinuousSpeakingRegimenArgs):
            return ToolCallDisplay(summary="context_continuous_speaking_regimen")
        if not event.args.items:
            return ToolCallDisplay(summary="context_continuous_speaking_regimen")
        first = event.args.items[0]
        return ToolCallDisplay(
            summary="context_continuous_speaking_regimen",
            details={
                "item_count": len(event.args.items),
                "id": first.id,
                "name": first.name,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextContinuousSpeakingRegimenResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=(
                f"Built {event.result.item_count} speaking regimen(s) with "
                f"{event.result.total_links} links"
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
        return "Building continuous speaking regimens"
