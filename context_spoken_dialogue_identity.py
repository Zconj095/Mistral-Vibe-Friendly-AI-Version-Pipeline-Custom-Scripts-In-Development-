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
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
PUNCT_KEYS = (".", ",", "?", "!", ":", ";", "-")


@dataclass
class _SpeakerSignature:
    dialogue_id: str
    speaker: str
    token_set: set[str]
    token_counts: Counter[str]
    segment_count: int
    total_tokens: int
    avg_sentence_length: float
    avg_token_length: float
    uppercase_ratio: float
    filler_ratio: float
    punct_rates: dict[str, float]


class ContextSpokenDialogueIdentityConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_items: int = Field(default=40, description="Maximum dialogues to evaluate.")
    max_source_bytes: int = Field(default=3_000_000, description="Maximum bytes per item.")
    max_total_bytes: int = Field(default=20_000_000, description="Max bytes across items.")
    preview_chars: int = Field(default=400, description="Preview snippet length.")
    max_segments: int = Field(default=200, description="Maximum segments per dialogue.")
    min_segment_chars: int = Field(default=20, description="Minimum segment length.")
    min_token_length: int = Field(default=3, description="Minimum token length.")
    max_segment_keywords: int = Field(default=8, description="Max keywords per segment.")
    max_top_words: int = Field(default=12, description="Max top words per speaker.")
    min_profile_tokens: int = Field(default=30, description="Minimum tokens for profiling.")
    similarity_threshold: float = Field(default=0.2, description="Min similarity to cluster.")
    token_weight: float = Field(default=0.6, description="Weight for token overlap.")
    style_weight: float = Field(default=0.4, description="Weight for style metrics.")
    max_user_candidates: int = Field(default=5, description="Max user candidates per dialogue.")
    exclude_ignored_from_clusters: bool = Field(
        default=True, description="Skip ignored speakers for clustering."
    )
    ignore_speakers: list[str] = Field(
        default_factory=lambda: [
            "assistant",
            "system",
            "ai",
            "model",
            "bot",
            "mistral",
            "vibe",
        ],
        description="Speaker labels to ignore for user identification.",
    )
    filler_words: list[str] = Field(
        default_factory=lambda: [
            "um",
            "uh",
            "hmm",
            "like",
            "okay",
            "ok",
            "yeah",
            "yep",
            "nope",
            "kinda",
            "sorta",
            "well",
        ],
        description="Words that indicate filler speech.",
    )


class ContextSpokenDialogueIdentityState(BaseToolState):
    pass


class SpokenDialogueItem(BaseModel):
    id: str | None = Field(default=None)
    name: str | None = Field(default=None)
    dialogue_id: str | None = Field(default=None)
    content: str | None = Field(default=None)
    path: str | None = Field(default=None)
    source: str | None = Field(default=None)



class ContextSpokenDialogueIdentityArgs(BaseModel):
    items: list[SpokenDialogueItem] = Field(description="Dialogues to evaluate.")


class DialogueSegment(BaseModel):
    index: int
    speaker: str | None
    text: str
    start: int
    end: int
    token_count: int
    unique_tokens: int
    keywords: list[str]


class PunctuationProfile(BaseModel):
    periods: float
    commas: float
    questions: float
    exclamations: float
    colons: float
    semicolons: float
    dashes: float


class SpeakerPattern(BaseModel):
    speaker: str
    segment_count: int
    total_tokens: int
    unique_tokens: int
    vocab_ratio: float
    avg_sentence_length: float
    avg_token_length: float
    uppercase_ratio: float
    filler_ratio: float
    punctuation_per_100: PunctuationProfile
    top_words: list[str]
    cluster_id: str | None = None


class UserCandidate(BaseModel):
    speaker: str
    score: float
    reasons: list[str]
    cluster_id: str | None = None
    confidence: float | None = None


class ClusterMember(BaseModel):
    dialogue_id: str
    speaker: str
    segment_count: int
    total_tokens: int


class UserCluster(BaseModel):
    cluster_id: str
    members: list[ClusterMember]
    top_words: list[str]
    avg_similarity: float


class SpokenDialogueInsight(BaseModel):
    index: int
    id: str | None
    name: str | None
    dialogue_id: str
    source_path: str | None
    preview: str
    segments: list[DialogueSegment]
    speakers: list[SpeakerPattern]
    user_candidates: list[UserCandidate]
    dialogue_top_words: list[str]
    segment_count: int
    speaker_count: int
    recognized_speaker: str | None
    recognized_cluster: str | None
    recognized_confidence: float | None


class ContextSpokenDialogueIdentityResult(BaseModel):
    items: list[SpokenDialogueInsight]
    user_clusters: list[UserCluster]
    item_count: int
    cluster_count: int
    truncated: bool
    warnings: list[str]
    errors: list[str]


class ContextSpokenDialogueIdentity(
    BaseTool[
        ContextSpokenDialogueIdentityArgs,
        ContextSpokenDialogueIdentityResult,
        ContextSpokenDialogueIdentityConfig,
        ContextSpokenDialogueIdentityState,
    ],
    ToolUIData[ContextSpokenDialogueIdentityArgs, ContextSpokenDialogueIdentityResult],
):
    description: ClassVar[str] = (
        "Process many vocal dialogues and infer speaker identities via pattern recognition."
    )

    async def run(
        self, args: ContextSpokenDialogueIdentityArgs
    ) -> ContextSpokenDialogueIdentityResult:
        items = list(args.items)
        if not items:
            raise ToolError("items is required.")

        errors: list[str] = []
        warnings: list[str] = []
        total_bytes = 0
        truncated = False

        if len(items) > self.config.max_items:
            warnings.append("Item limit reached; truncating input list.")
            items = items[: self.config.max_items]

        work_items: list[dict[str, object]] = []
        signatures: list[_SpeakerSignature] = []
        signature_lookup: dict[tuple[str, str], _SpeakerSignature] = {}
        dialogue_top_words: dict[str, list[str]] = {}

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
                if len(segments) > self.config.max_segments:
                    segments = segments[: self.config.max_segments]
                    warnings.append("Segment limit reached; truncating dialogue.")

                speaker_patterns, profile_signatures, top_words = self._build_speaker_patterns(
                    segments
                )

                dialogue_id = self._dialogue_id(item, idx)
                for signature in profile_signatures:
                    signature.dialogue_id = dialogue_id
                    if self._skip_cluster(signature.speaker):
                        continue
                    signatures.append(signature)
                    signature_lookup[(dialogue_id, signature.speaker)] = signature

                user_candidates = self._rank_user_candidates(speaker_patterns)
                dialogue_top_words[dialogue_id] = top_words

                work_items.append(
                    {
                        "index": len(work_items) + 1,
                        "item": item,
                        "dialogue_id": dialogue_id,
                        "source_path": source_path,
                        "preview": self._preview(content),
                        "segments": segments,
                        "speaker_patterns": speaker_patterns,
                        "user_candidates": user_candidates,
                    }
                )
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not work_items:
            raise ToolError("No insights generated.")

        cluster_map, clusters, cluster_scores = self._cluster_signatures(signatures)

        insights: list[SpokenDialogueInsight] = []
        for entry in work_items:
            dialogue_id = entry["dialogue_id"]
            speaker_patterns = self._assign_cluster_ids(
                entry["speaker_patterns"], cluster_map, dialogue_id
            )
            user_candidates = self._assign_user_candidates(
                entry["user_candidates"], cluster_map, dialogue_id, cluster_scores
            )
            recognized = self._select_recognized_user(user_candidates)
            insights.append(
                SpokenDialogueInsight(
                    index=entry["index"],
                    id=entry["item"].id,
                    name=entry["item"].name,
                    dialogue_id=dialogue_id,
                    source_path=entry["source_path"],
                    preview=entry["preview"],
                    segments=entry["segments"],
                    speakers=speaker_patterns,
                    user_candidates=user_candidates,
                    dialogue_top_words=dialogue_top_words.get(dialogue_id, []),
                    segment_count=len(entry["segments"]),
                    speaker_count=len(speaker_patterns),
                    recognized_speaker=recognized[0],
                    recognized_cluster=recognized[1],
                    recognized_confidence=recognized[2],
                )
            )

        return ContextSpokenDialogueIdentityResult(
            items=insights,
            user_clusters=clusters,
            item_count=len(insights),
            cluster_count=len(clusters),
            truncated=truncated,
            warnings=warnings,
            errors=errors,
        )

    def _dialogue_id(self, item: SpokenDialogueItem, idx: int) -> str:
        if item.dialogue_id:
            return item.dialogue_id
        if item.id:
            return item.id
        if item.name:
            return item.name
        return f"dialogue_{idx}"

    def _segment_dialogue(self, text: str) -> list[DialogueSegment]:
        segments: list[DialogueSegment] = []
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
                token_list = self._tokenize(combined)
                token_count = len(token_list)
                keywords = self._top_words(token_list, self.config.max_segment_keywords)
                segments.append(
                    DialogueSegment(
                        index=len(segments) + 1,
                        speaker=current_speaker,
                        text=combined,
                        start=segment_start or 0,
                        end=segment_end or 0,
                        token_count=token_count,
                        unique_tokens=len(set(token_list)),
                        keywords=keywords,
                    )
                )
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

    def _fallback_segments(self, text: str) -> list[DialogueSegment]:
        chunks = [chunk for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
        segments: list[DialogueSegment] = []
        cursor = 0
        for chunk in chunks:
            start = text.find(chunk, cursor)
            end = start + len(chunk)
            cursor = end
            seg_text = chunk.strip()
            if len(seg_text) < self.config.min_segment_chars:
                continue
            token_list = self._tokenize(seg_text)
            segments.append(
                DialogueSegment(
                    index=len(segments) + 1,
                    speaker=None,
                    text=seg_text,
                    start=start,
                    end=end,
                    token_count=len(token_list),
                    unique_tokens=len(set(token_list)),
                    keywords=self._top_words(
                        token_list, self.config.max_segment_keywords
                    ),
                )
            )
        return segments[: self.config.max_segments]

    def _build_speaker_patterns(
        self, segments: list[DialogueSegment]
    ) -> tuple[list[SpeakerPattern], list[_SpeakerSignature], list[str]]:
        by_speaker: dict[str, list[DialogueSegment]] = defaultdict(list)
        all_tokens: Counter[str] = Counter()
        for segment in segments:
            speaker = segment.speaker or "unknown"
            by_speaker[speaker].append(segment)
            all_tokens.update(self._tokenize(segment.text))

        patterns: list[SpeakerPattern] = []
        signatures: list[_SpeakerSignature] = []
        for speaker, group in by_speaker.items():
            token_counts = Counter()
            sentence_lengths: list[int] = []
            token_lengths: list[int] = []
            uppercase_tokens = 0
            filler_tokens = 0
            punct_counts = Counter({key: 0 for key in PUNCT_KEYS})
            total_tokens = 0

            for segment in group:
                tokens = self._tokenize(segment.text)
                token_counts.update(tokens)
                total_tokens += len(tokens)
                token_lengths.extend(len(token) for token in tokens)
                uppercase_tokens += sum(1 for token in tokens if token.isupper() and len(token) > 1)
                filler_tokens += sum(
                    1 for token in tokens if token in self._filler_set()
                )
                for key in PUNCT_KEYS:
                    punct_counts[key] += segment.text.count(key)
                sentence_lengths.extend(self._sentence_lengths(segment.text))

            if total_tokens == 0:
                continue

            unique_tokens = len(token_counts)
            vocab_ratio = unique_tokens / total_tokens if total_tokens else 0.0
            avg_sentence_length = (
                sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0.0
            )
            avg_token_length = (
                sum(token_lengths) / len(token_lengths) if token_lengths else 0.0
            )
            uppercase_ratio = uppercase_tokens / total_tokens if total_tokens else 0.0
            filler_ratio = filler_tokens / total_tokens if total_tokens else 0.0
            punct_rates = self._punct_rates(punct_counts, total_tokens)
            top_words = self._top_words(token_counts.elements(), self.config.max_top_words)

            patterns.append(
                SpeakerPattern(
                    speaker=speaker,
                    segment_count=len(group),
                    total_tokens=total_tokens,
                    unique_tokens=unique_tokens,
                    vocab_ratio=round(vocab_ratio, 4),
                    avg_sentence_length=round(avg_sentence_length, 3),
                    avg_token_length=round(avg_token_length, 3),
                    uppercase_ratio=round(uppercase_ratio, 4),
                    filler_ratio=round(filler_ratio, 4),
                    punctuation_per_100=PunctuationProfile(
                        periods=punct_rates["."],
                        commas=punct_rates[","],
                        questions=punct_rates["?"],
                        exclamations=punct_rates["!"],
                        colons=punct_rates[":"],
                        semicolons=punct_rates[";"],
                        dashes=punct_rates["-"],
                    ),
                    top_words=top_words,
                )
            )

            signatures.append(
                _SpeakerSignature(
                    dialogue_id="",
                    speaker=speaker,
                    token_set=set(top_words),
                    token_counts=token_counts,
                    segment_count=len(group),
                    total_tokens=total_tokens,
                    avg_sentence_length=avg_sentence_length,
                    avg_token_length=avg_token_length,
                    uppercase_ratio=uppercase_ratio,
                    filler_ratio=filler_ratio,
                    punct_rates=punct_rates,
                )
            )

        overall_top_words = self._top_words(all_tokens.elements(), self.config.max_top_words)
        return patterns, signatures, overall_top_words

    def _rank_user_candidates(self, patterns: list[SpeakerPattern]) -> list[UserCandidate]:
        candidates = [
            pattern
            for pattern in patterns
            if not self._is_ignored_speaker(pattern.speaker)
        ]
        if not candidates:
            return []
        max_tokens = max(pattern.total_tokens for pattern in candidates) or 1
        max_segments = max(pattern.segment_count for pattern in candidates) or 1
        scored: list[UserCandidate] = []
        for pattern in candidates:
            score = 0.7 * (pattern.total_tokens / max_tokens) + 0.3 * (
                pattern.segment_count / max_segments
            )
            reasons = []
            if pattern.total_tokens == max_tokens:
                reasons.append("highest_token_share")
            if pattern.segment_count == max_segments:
                reasons.append("most_segments")
            if pattern.total_tokens < self.config.min_profile_tokens:
                reasons.append("low_token_profile")
            scored.append(
                UserCandidate(
                    speaker=pattern.speaker,
                    score=round(score, 3),
                    reasons=reasons,
                )
            )
        scored.sort(key=lambda entry: entry.score, reverse=True)
        return scored[: self.config.max_user_candidates]

    def _cluster_signatures(
        self, signatures: list[_SpeakerSignature]
    ) -> tuple[dict[tuple[str, str], str], list[UserCluster], dict[str, float]]:
        if not signatures:
            return {}, [], {}
        count = len(signatures)
        parent = list(range(count))
        pair_scores: dict[tuple[int, int], float] = {}

        def find(idx: int) -> int:
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        def union(left: int, right: int) -> None:
            root_left = find(left)
            root_right = find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        for i in range(count):
            for j in range(i + 1, count):
                similarity = self._similarity(signatures[i], signatures[j])
                if similarity >= self.config.similarity_threshold:
                    pair_scores[(i, j)] = similarity
                    union(i, j)

        clusters_by_root: dict[int, list[int]] = defaultdict(list)
        for idx in range(count):
            clusters_by_root[find(idx)].append(idx)

        cluster_map: dict[tuple[str, str], str] = {}
        clusters: list[UserCluster] = []
        cluster_scores: dict[str, float] = {}
        for idx, indices in enumerate(sorted(clusters_by_root.values(), key=len, reverse=True), start=1):
            cluster_id = f"user_{idx}"
            members: list[ClusterMember] = []
            cluster_token_counts: Counter[str] = Counter()
            similarities: list[float] = []
            for i, first in enumerate(indices):
                signature = signatures[first]
                cluster_map[(signature.dialogue_id, signature.speaker)] = cluster_id
                members.append(
                    ClusterMember(
                        dialogue_id=signature.dialogue_id,
                        speaker=signature.speaker,
                        segment_count=signature.segment_count,
                        total_tokens=signature.total_tokens,
                    )
                )
                cluster_token_counts.update(signature.token_counts)
                for second in indices[i + 1 :]:
                    key = (min(first, second), max(first, second))
                    if key in pair_scores:
                        similarities.append(pair_scores[key])

            avg_similarity = sum(similarities) / len(similarities) if similarities else 1.0
            top_words = self._top_words(
                cluster_token_counts.elements(), self.config.max_top_words
            )
            clusters.append(
                UserCluster(
                    cluster_id=cluster_id,
                    members=members,
                    top_words=top_words,
                    avg_similarity=round(avg_similarity, 3),
                )
            )
            cluster_scores[cluster_id] = round(avg_similarity, 3)

        return cluster_map, clusters, cluster_scores

    def _assign_cluster_ids(
        self,
        patterns: list[SpeakerPattern],
        cluster_map: dict[tuple[str, str], str],
        dialogue_id: str,
    ) -> list[SpeakerPattern]:
        updated: list[SpeakerPattern] = []
        for pattern in patterns:
            cluster_id = cluster_map.get((dialogue_id, pattern.speaker))
            updated.append(pattern.model_copy(update={"cluster_id": cluster_id}))
        return updated

    def _assign_user_candidates(
        self,
        candidates: list[UserCandidate],
        cluster_map: dict[tuple[str, str], str],
        dialogue_id: str,
        cluster_scores: dict[str, float],
    ) -> list[UserCandidate]:
        updated: list[UserCandidate] = []
        for candidate in candidates:
            cluster_id = cluster_map.get((dialogue_id, candidate.speaker))
            confidence = None
            if cluster_id:
                confidence = cluster_scores.get(cluster_id, 1.0)
            updated.append(
                candidate.model_copy(
                    update={"cluster_id": cluster_id, "confidence": confidence}
                )
            )
        return updated

    def _select_recognized_user(
        self, candidates: list[UserCandidate]
    ) -> tuple[str | None, str | None, float | None]:
        if not candidates:
            return None, None, None
        top = candidates[0]
        confidence = top.confidence
        if confidence is None:
            confidence = top.score
        return top.speaker, top.cluster_id, round(confidence, 3)

    def _skip_cluster(self, speaker: str) -> bool:
        return self.config.exclude_ignored_from_clusters and self._is_ignored_speaker(speaker)

    def _is_ignored_speaker(self, speaker: str) -> bool:
        return speaker.strip().lower() in self._ignore_set()

    def _ignore_set(self) -> set[str]:
        return {value.strip().lower() for value in self.config.ignore_speakers}

    def _filler_set(self) -> set[str]:
        return {value.strip().lower() for value in self.config.filler_words}

    def _similarity(self, left: _SpeakerSignature, right: _SpeakerSignature) -> float:
        union = left.token_set | right.token_set
        if union:
            token_sim = len(left.token_set & right.token_set) / len(union)
        else:
            token_sim = 0.0

        style_scores = [
            self._ratio_similarity(left.avg_sentence_length, right.avg_sentence_length),
            self._ratio_similarity(left.avg_token_length, right.avg_token_length),
            self._ratio_similarity(left.uppercase_ratio, right.uppercase_ratio),
            self._ratio_similarity(left.filler_ratio, right.filler_ratio),
        ]
        for key in PUNCT_KEYS:
            style_scores.append(
                self._ratio_similarity(left.punct_rates.get(key, 0.0), right.punct_rates.get(key, 0.0))
            )
        style_sim = sum(style_scores) / len(style_scores) if style_scores else 0.0

        weight_total = self.config.token_weight + self.config.style_weight
        token_weight = self.config.token_weight / weight_total if weight_total else 0.6
        style_weight = self.config.style_weight / weight_total if weight_total else 0.4
        return token_sim * token_weight + style_sim * style_weight

    def _ratio_similarity(self, left: float, right: float) -> float:
        if left == right:
            return 1.0
        denom = max(left, right, 1e-6)
        return 1.0 - min(1.0, abs(left - right) / denom)

    def _punct_rates(self, counts: Counter[str], total_tokens: int) -> dict[str, float]:
        scale = 100.0 / max(total_tokens, 1)
        return {key: round(counts[key] * scale, 3) for key in PUNCT_KEYS}

    def _sentence_lengths(self, text: str) -> list[int]:
        lengths: list[int] = []
        for sentence in SENTENCE_SPLIT_RE.split(text):
            tokens = self._tokenize(sentence)
            if tokens:
                lengths.append(len(tokens))
        return lengths

    def _tokenize(self, text: str) -> list[str]:
        return [
            token.lower()
            for token in WORD_RE.findall(text)
            if len(token) >= self.config.min_token_length
        ]

    def _top_words(self, tokens: list[str] | Counter[str] | list[str], max_items: int) -> list[str]:
        if isinstance(tokens, Counter):
            counter = tokens
        else:
            counter = Counter(tokens)
        return [word for word, _ in counter.most_common(max_items)]

    def _load_item(self, item: SpokenDialogueItem) -> tuple[str, str | None, int | None]:
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
        if not isinstance(event.args, ContextSpokenDialogueIdentityArgs):
            return ToolCallDisplay(summary="context_spoken_dialogue_identity")
        if not event.args.items:
            return ToolCallDisplay(summary="context_spoken_dialogue_identity")
        first = event.args.items[0]
        return ToolCallDisplay(
            summary="context_spoken_dialogue_identity",
            details={
                "item_count": len(event.args.items),
                "id": first.id,
                "name": first.name,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpokenDialogueIdentityResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=(
                f"Analyzed {event.result.item_count} dialogue(s) with "
                f"{event.result.cluster_count} user cluster(s)"
            ),
            warnings=event.result.warnings,
            details={
                "item_count": event.result.item_count,
                "cluster_count": event.result.cluster_count,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Profiling dialogue speakers"
