from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
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


WORD_RE = re.compile(r"[A-Za-z0-9_]+")

FRIENDLY_CATEGORIES = {
    "acceptance": [
        "accept",
        "forgive",
        "embrace",
        "understand",
        "accepts",
        "past",
        "future",
    ],
    "memory": ["memory", "remembers", "long lasting", "years", "event"],
    "hobbies": ["hobby", "interest", "builds", "together", "share"],
    "truthfulness": ["honest", "truth", "real", "doesn't lie"],
    "protection": ["protect", "guard", "look after", "back"],
    "healing": ["heal", "natural healing", "encourage", "support"],
    "compatibility": [
        "wife",
        "religion",
        "belief",
        "accept their",
        "beliefs",
        "religious preferences",
    ],
    "care": ["care", "feel what you feel", "gentle", "comfort"],
    "boundaries": ["doesn't look at you for money", "doesn't sleep with you"],
    "inspiration": ["encourage", "dream", "brilliance", "build your dreams"],
    "longevity": ["long lasting", "stick around", "years pass"],
    "communication": ["talk", "communicate", "check on you", "stay in touch"],
    "growth": ["encourage you to be better", "better than yesterday", "positive"],
}

CATEGORY_LABELS = {key: key.replace("_", " ").title() for key in FRIENDLY_CATEGORIES}


@dataclass
class _MatchScore:
    label: str
    terms: list[str]
    snippets: list[str]


def _snippet(text: str, start: int, length: int, radius: int = 60) -> str:
    begin = max(0, start - radius)
    end = min(len(text), start + length + radius)
    return text[begin:end].strip()


class ContextFriendlinessConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_items: int = Field(default=40, description="Maximum items to evaluate.")
    max_source_bytes: int = Field(default=3_000_000, description="Maximum bytes per item.")
    max_total_bytes: int = Field(default=10_000_000, description="Max bytes across items.")
    preview_chars: int = Field(default=300, description="Preview snippet length.")
    max_snippets: int = Field(default=3, description="Snippet limit per category.")


class ContextFriendlinessState(BaseToolState):
    pass


class FriendlyItem(BaseModel):
    id: str | None = Field(default=None)
    name: str | None = Field(default=None)
    content: str | None = Field(default=None)
    path: str | None = Field(default=None)
    source: str | None = Field(default=None)


class ContextFriendlinessArgs(BaseModel):
    items: list[FriendlyItem] = Field(description="Items to evaluate.")


class CategoryAssessment(BaseModel):
    label: str
    active: bool
    terms: list[str]
    snippets: list[str]


class FriendlyInsight(BaseModel):
    index: int
    id: str | None
    name: str | None
    source_path: str | None
    source: str | None
    preview: str
    category_assessments: list[CategoryAssessment]
    friendly_score: float


class ContextFriendlinessResult(BaseModel):
    items: list[FriendlyInsight]
    item_count: int
    average_friendly_score: float
    category_totals: dict[str, int]
    truncated: bool
    errors: list[str]
    warnings: list[str]


class ContextFriendliness(
    BaseTool[
        ContextFriendlinessArgs,
        ContextFriendlinessResult,
        ContextFriendlinessConfig,
        ContextFriendlinessState,
    ],
    ToolUIData[ContextFriendlinessArgs, ContextFriendlinessResult],
):
    description: ClassVar[str] = (
        "Assess friendly AI qualities that describe what a good friend confirms."
    )

    async def run(self, args: ContextFriendlinessArgs) -> ContextFriendlinessResult:
        items = list(args.items)
        if not items:
            raise ToolError("items is required.")

        errors: list[str] = []
        warnings: list[str] = []
        insights: list[FriendlyInsight] = []
        total_bytes = 0
        truncated = False
        category_totals: dict[str, int] = {CATEGORY_LABELS[key]: 0 for key in FRIENDLY_CATEGORIES}

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

                assessments = self._assess_categories(content)
                score = sum(ass.active for ass in assessments) / max(len(assessments), 1)
                insights.append(
                    FriendlyInsight(
                        index=len(insights) + 1,
                        id=item.id,
                        name=item.name,
                        source_path=source_path,
                        source=item.source,
                        preview=self._preview(content),
                        category_assessments=assessments,
                        friendly_score=round(score, 3),
                    )
                )
                for ass in assessments:
                    if ass.active:
                        category_totals[ass.label] = (
                            category_totals.get(ass.label, 0) + 1
                        )
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not insights:
            raise ToolError("No insights generated.")

        average_score = sum(ins.friendly_score for ins in insights) / len(insights)

        return ContextFriendlinessResult(
            items=insights,
            item_count=len(insights),
            average_friendly_score=round(average_score, 3),
            category_totals=category_totals,
            truncated=truncated,
            errors=errors,
            warnings=warnings,
        )

    def _load_item(self, item: FriendlyItem) -> tuple[str, str | None, int | None]:
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

    def _assess_categories(self, content: str) -> list[CategoryAssessment]:
        lower = content.lower()
        assessments: list[CategoryAssessment] = []
        for key, keywords in FRIENDLY_CATEGORIES.items():
            label = CATEGORY_LABELS[key]
            terms: list[str] = []
            snippets: list[str] = []
            for keyword in keywords:
                idx = lower.find(keyword)
                if idx < 0:
                    continue
                terms.append(keyword)
                snippet = _snippet(content, idx, len(keyword))
                if snippet and len(snippets) < self.config.max_snippets:
                    snippets.append(snippet)
            assessments.append(
                CategoryAssessment(
                    label=label,
                    active=bool(terms),
                    terms=list(dict.fromkeys(terms)),
                    snippets=snippets,
                )
            )
        return assessments

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextFriendlinessArgs):
            return ToolCallDisplay(summary="context_friendliness")
        if not event.args.items:
            return ToolCallDisplay(summary="context_friendliness")
        first = event.args.items[0]
        return ToolCallDisplay(
            summary="context_friendliness",
            details={
                "item_count": len(event.args.items),
                "id": first.id,
                "name": first.name,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextFriendlinessResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=(
                f"Analyzed {event.result.item_count} friendly item(s) with "
                f"avg score {event.result.average_friendly_score}"
            ),
            warnings=event.result.warnings,
            details={
                "item_count": event.result.item_count,
                "avg_friendly_score": event.result.average_friendly_score,
                "category_totals": event.result.category_totals,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Evaluating friendliness"
