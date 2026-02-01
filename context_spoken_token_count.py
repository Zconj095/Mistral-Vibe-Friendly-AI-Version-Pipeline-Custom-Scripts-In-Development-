from __future__ import annotations

from pathlib import Path
import re
from typing import ClassVar, Literal, TYPE_CHECKING

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
SENTENCE_RE = re.compile(r"[.!?]+")


class ContextSpokenTokenCountConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_items: int = Field(default=40, description="Maximum items to evaluate.")
    max_source_bytes: int = Field(default=3_000_000, description="Maximum bytes per item.")
    max_total_bytes: int = Field(default=20_000_000, description="Max bytes across items.")
    preview_chars: int = Field(default=400, description="Preview snippet length.")
    token_mode: Literal["words", "whitespace", "chars"] = Field(
        default="words",
        description="Token counting mode.",
    )


class ContextSpokenTokenCountState(BaseToolState):
    pass


class SpokenTokenCountItem(BaseModel):
    id: str | None = Field(default=None)
    name: str | None = Field(default=None)
    content: str | None = Field(default=None)
    path: str | None = Field(default=None)
    source: str | None = Field(default=None)



class ContextSpokenTokenCountArgs(BaseModel):
    items: list[SpokenTokenCountItem] = Field(description="Items to evaluate.")

class SpokenTokenCountInsight(BaseModel):
    index: int
    id: str | None
    name: str | None
    source_path: str | None
    preview: str
    token_mode: str
    token_count: int
    word_count: int
    unique_word_count: int
    sentence_count: int
    line_count: int
    char_count: int


class ContextSpokenTokenCountResult(BaseModel):
    items: list[SpokenTokenCountInsight]
    item_count: int
    total_tokens: int
    truncated: bool
    warnings: list[str]
    errors: list[str]


class ContextSpokenTokenCount(
    BaseTool[
        ContextSpokenTokenCountArgs,
        ContextSpokenTokenCountResult,
        ContextSpokenTokenCountConfig,
        ContextSpokenTokenCountState,
    ],
    ToolUIData[ContextSpokenTokenCountArgs, ContextSpokenTokenCountResult],
):
    description: ClassVar[str] = (
        "Calculate spoken token counts and related stats for text."
    )

    async def run(self, args: ContextSpokenTokenCountArgs) -> ContextSpokenTokenCountResult:
        items = list(args.items)
        if not items:
            raise ToolError("items is required.")

        errors: list[str] = []
        warnings: list[str] = []
        insights: list[SpokenTokenCountInsight] = []
        total_bytes = 0
        truncated = False
        total_tokens = 0

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

                counts = self._count_tokens(content)
                total_tokens += counts["token_count"]

                insights.append(
                    SpokenTokenCountInsight(
                        index=len(insights) + 1,
                        id=item.id,
                        name=item.name,
                        source_path=source_path,
                        preview=self._preview(content),
                        token_mode=self.config.token_mode,
                        token_count=counts["token_count"],
                        word_count=counts["word_count"],
                        unique_word_count=counts["unique_word_count"],
                        sentence_count=counts["sentence_count"],
                        line_count=counts["line_count"],
                        char_count=counts["char_count"],
                    )
                )
            except ToolError as exc:
                errors.append(f"item[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"item[{idx}]: {exc}")

        if not insights:
            raise ToolError("No insights generated.")

        return ContextSpokenTokenCountResult(
            items=insights,
            item_count=len(insights),
            total_tokens=total_tokens,
            truncated=truncated,
            warnings=warnings,
            errors=errors,
        )

    def _count_tokens(self, text: str) -> dict[str, int]:
        word_tokens = WORD_RE.findall(text)
        word_count = len(word_tokens)
        unique_word_count = len({token.lower() for token in word_tokens})
        sentence_count = len(SENTENCE_RE.findall(text))
        line_count = text.count("\n") + 1 if text else 0
        char_count = len(text)

        if self.config.token_mode == "chars":
            token_count = char_count
        elif self.config.token_mode == "whitespace":
            token_count = len([part for part in text.split() if part])
        else:
            token_count = word_count

        return {
            "token_count": token_count,
            "word_count": word_count,
            "unique_word_count": unique_word_count,
            "sentence_count": sentence_count,
            "line_count": line_count,
            "char_count": char_count,
        }

    def _load_item(self, item: SpokenTokenCountItem) -> tuple[str, str | None, int | None]:
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
        if not isinstance(event.args, ContextSpokenTokenCountArgs):
            return ToolCallDisplay(summary="context_spoken_token_count")
        if not event.args.items:
            return ToolCallDisplay(summary="context_spoken_token_count")
        first = event.args.items[0]
        return ToolCallDisplay(
            summary="context_spoken_token_count",
            details={
                "item_count": len(event.args.items),
                "id": first.id,
                "name": first.name,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpokenTokenCountResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=(
                f"Analyzed {event.result.item_count} item(s) with "
                f"{event.result.total_tokens} tokens"
            ),
            warnings=event.result.warnings,
            details={
                "item_count": event.result.item_count,
                "total_tokens": event.result.total_tokens,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Counting spoken tokens"
