from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
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


DEFAULT_PROMPT_TEMPLATE = """### GENRE CLASSIFICATION RECOGNITION MODE (OPT-IN)
Recognize and classify genres with supporting cues.
- Identify genre indicators and classify with confidence notes.
- Provide a concise rationale without revealing chain-of-thought.
- Use the genre taxonomy below for classification.

Main_genres:
- Fantasy: Imaginative fiction with magic, mythical creatures, or fantastical worlds.
- Adventure: Exciting, hazardous journeys or quests with exploration and danger.
- Mystical: Wonder or mystery with supernatural or spiritual experiences.
- Romance: Love and emotional relationships, affection, passion, connection.
- Science: Systematic study of the natural world and technological discovery.
- Action: Fast-paced, high-energy sequences with physical feats or combat.
- Magical: Supernatural powers or forces that defy natural laws.
- Nonfiction: Factual events, people, or information.
- Fiction: Imaginative stories not bound by factual accuracy.

Science_and_fiction:
- Science_fiction: Speculative science/technology with futuristic narratives.
- Fictional_science: Imaginary scientific concepts within fictional narratives.
- Mystical_science: Scientific concepts blended with mystical elements.
- Magical_science: Scientific principles intertwined with magical elements.
- Fringe_science: Unconventional or speculative scientific theories/practices.
- Fantasy_science: Fantasy elements merged with scientific ideas.

Adventure_and_action:
- Fantasy_adventure: Fantasy with adventure quests in fantastical worlds.
- Magical_adventure: Thrilling journeys in magic-filled worlds.
- Mystical_adventure: Mystical wonder blended with adventurous quests.
- Magical_action: Action sequences infused with magic.
- Fantasy_based_action: High-stakes action in fantastical settings.
- Supernatural_adventure: Adventures involving ghosts, spirits, or otherworldly beings.

Romance_and_its_variants:
- Mystical_romance: Romance intertwined with mystical elements.
- Magical_romance: Love influenced by magical forces or beings.
- Scientific_romance: Romance with science fiction or scientific themes.
- Fringe_based_romance: Romance against a backdrop of fringe science.
- Religious_romance: Romance within religious beliefs or settings.
- Spiritual_romance: Romance tied to spiritual journeys.
- Romance_and_action: Romance combined with action and danger.
- Romance_and_adventure: Romance merged with adventure and exploration.
- Romantic_adventure: Love story unfolding during a thrilling adventure.
- Supernatural_romance: Romance influenced by paranormal entities.

Ghibli_and_themed_variants:
- Ghibli_anime: Ghibli-inspired, imaginative worlds and emotional storytelling.
- Ghibli_cartoon: Animated works capturing Ghibli-like magic and wonder.
- Ghibli_mystical: Ghibli-inspired stories with mystical spirituality.
- Ghibli_magical: Ghibli-influenced narratives blending everyday and magic.
- Ghibli_fringe: Ghibli-inspired fringe or unconventional themes.

Spiritual_and_religious_variants:
- Mystical_fringe: Mystical elements with fringe ideas.
- Magical_fringe: Magic intersecting with fringe theories or practices.
- Religious_fantasy: Religious themes blended with fantasy elements.
- Religious_suspense: Religious themes creating suspense or moral dilemmas.
- Religious_adventure: Religious or spiritual journeys with adventure.
- Religious_science: Relationship between religious beliefs and science.
- Religious_fiction: Fictional narratives exploring religious beliefs.
- Spiritual_fantasy: Fantasy merged with spiritual themes.
- Fringe_fantasy: Fantasy blended with fringe or unconventional ideas.
- Supernatural_fantasy: Fantasy combined with supernatural occurrences.
- Nonfiction_fantasy: Real-world elements blended with fantastical narratives.

Complex_genre_hybrids:
- Religious_science_fiction_fantasy
- Spiritual_magical_adventure
- Supernatural_magical_adventure
- Magical_fantasy_romance
- Magical_fantasy_adventure
- Magical_romantic_adventure
- Mystical_romantic_adventure
- Spiritual_fantasy_science
- Magical_fringe_science
- Mystical_fringe_science
- Magical_religious_fantasy
- Religious_spiritual_fantasy
- Supernatural_fantasy_romance
- Magical_fringe_romance
- Magical_scientific_fantasy
- Magical_religious_romance
- Magical_religious_science
- Magical_mystical_religion

Focus: {focus}
Show steps: {show_steps}
Max genres: {max_genres}
"""

TOOL_PROMPT = (
    "Use `genre_classification_recognition` to identify and classify genres. "
    "Provide `prompt` or `messages`, and optionally set `focus`, `show_steps`, "
    "and `max_genres`."
)


class GenreClassificationMessage(BaseModel):
    role: str
    content: str


class GenreClassificationArgs(BaseModel):
    prompt: str | None = Field(
        default=None, description="User prompt to solve."
    )
    messages: list[GenreClassificationMessage] | None = Field(
        default=None, description="Optional chat messages."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prefix."
    )
    focus: str | None = Field(
        default=None, description="Focus label or domain."
    )
    show_steps: bool | None = Field(
        default=None, description="Whether to include a step outline."
    )
    max_genres: int | None = Field(
        default=None, description="Maximum genres in the outline."
    )
    llm_api_base: str | None = Field(
        default=None, description="OpenAI-compatible API base URL."
    )
    llm_model: str | None = Field(
        default=None, description="LLM model name."
    )
    llm_temperature: float = Field(
        default=0.2, description="LLM temperature."
    )
    llm_max_tokens: int = Field(
        default=800, description="LLM max tokens."
    )
    llm_stream: bool = Field(
        default=False, description="Stream LLM tokens."
    )


class GenreClassificationResult(BaseModel):
    answer: str
    system_prompt: str
    messages: list[GenreClassificationMessage]
    used_focus: str
    show_steps: bool
    max_genres: int
    template_source: str
    warnings: list[str]
    errors: list[str]
    llm_model: str


class GenreClassificationConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    llm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL.",
    )
    llm_model: str = Field(
        default="gpt-oss:latest", description="Default LLM model name."
    )
    default_focus: str = Field(
        default="Genre classification.",
        description="Default focus label.",
    )
    default_show_steps: bool = Field(
        default=False, description="Default for show_steps."
    )
    default_max_genres: int = Field(
        default=5, description="Default max genres."
    )
    prompt_path: Path | None = Field(
        default=Path.home()
        / "mistral-vibe"
        / "vibe"
        / "core"
        / "prompts"
        / "genre_classification_recognition.md",
        description="Optional path to a prompt template.",
    )
    prompt_max_chars: int = Field(
        default=8000, description="Maximum template characters to load."
    )


class GenreClassificationState(BaseToolState):
    pass


class GenreClassificationRecognition(
    BaseTool[
        GenreClassificationArgs,
        GenreClassificationResult,
        GenreClassificationConfig,
        GenreClassificationState,
    ],
    ToolUIData[GenreClassificationArgs, GenreClassificationResult],
):
    description: ClassVar[str] = (
        "Recognize and classify genres with a dedicated prompt."
    )

    @classmethod
    def get_tool_prompt(cls) -> str | None:
        return TOOL_PROMPT

    async def run(
        self, args: GenreClassificationArgs
    ) -> GenreClassificationResult:
        warnings: list[str] = []
        errors: list[str] = []

        focus = (args.focus or self.config.default_focus).strip()
        if not focus:
            focus = "genre classification"

        show_steps = (
            args.show_steps
            if args.show_steps is not None
            else self.config.default_show_steps
        )
        max_genres = (
            args.max_genres
            if args.max_genres is not None
            else self.config.default_max_genres
        )
        if max_genres <= 0:
            raise ToolError("max_genres must be positive.")

        self._validate_llm_settings(args)

        template, template_source = self._load_template(warnings)
        system_prompt = self._build_system_prompt(
            template, focus, show_steps, max_genres, args.system_prompt
        )

        messages = self._normalize_messages(args, system_prompt)
        answer = self._call_llm(messages, args)

        return GenreClassificationResult(
            answer=answer,
            system_prompt=system_prompt,
            messages=messages,
            used_focus=focus,
            show_steps=show_steps,
            max_genres=max_genres,
            template_source=template_source,
            warnings=warnings,
            errors=errors,
            llm_model=self._resolve_model(args),
        )

    def _validate_llm_settings(self, args: GenreClassificationArgs) -> None:
        if args.llm_temperature < 0:
            raise ToolError("llm_temperature cannot be negative.")
        if args.llm_max_tokens <= 0:
            raise ToolError("llm_max_tokens must be positive.")

    def _load_template(self, warnings: list[str]) -> tuple[str, str]:
        template = DEFAULT_PROMPT_TEMPLATE
        source = "embedded"

        if not self.config.prompt_path:
            return self._truncate_template(template, warnings), source

        path = self._resolve_prompt_path(self.config.prompt_path)
        if not path.exists():
            warnings.append(f"Prompt template not found: {path}")
            return self._truncate_template(template, warnings), source
        if path.is_dir():
            warnings.append(f"Prompt template is a directory: {path}")
            return self._truncate_template(template, warnings), source

        try:
            text = path.read_text("utf-8", errors="ignore").strip()
        except OSError as exc:
            warnings.append(f"Failed to read prompt template: {exc}")
            return self._truncate_template(template, warnings), source

        if not text:
            warnings.append(f"Prompt template empty: {path}")
            return self._truncate_template(template, warnings), source

        template = text
        source = str(path)
        return self._truncate_template(template, warnings), source

    def _resolve_prompt_path(self, raw_path: Path | str) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.config.effective_workdir / path
        return path.resolve()

    def _truncate_template(self, template: str, warnings: list[str]) -> str:
        max_chars = self.config.prompt_max_chars
        if max_chars > 0 and len(template) > max_chars:
            warnings.append("Prompt template truncated to prompt_max_chars.")
            return template[:max_chars].rstrip()
        return template

    def _build_system_prompt(
        self,
        template: str,
        focus: str,
        show_steps: bool,
        max_genres: int,
        prefix: str | None,
    ) -> str:
        show_steps_text = "yes" if show_steps else "no"
        rendered = self._render_template(
            template, focus, show_steps_text, max_genres
        )
        if prefix and prefix.strip():
            return f"{prefix.strip()}\\n\\n{rendered}".strip()
        return rendered.strip()

    def _render_template(
        self,
        template: str,
        focus: str,
        show_steps_text: str,
        max_genres: int,
    ) -> str:
        had_placeholders = (
            "{focus}" in template
            or "{show_steps}" in template
            or "{max_genres}" in template
        )
        rendered = template
        if "{focus}" in template:
            rendered = rendered.replace("{focus}", focus)
        if "{show_steps}" in template:
            rendered = rendered.replace("{show_steps}", show_steps_text)
        if "{max_genres}" in template:
            rendered = rendered.replace("{max_genres}", str(max_genres))

        if had_placeholders:
            return rendered

        extra = (
            f"Focus: {focus}\n"
            f"Show steps: {show_steps_text}\n"
            f"Max genres: {max_genres}"
        )
        return f"{rendered.rstrip()}\n\n{extra}"

    def _normalize_messages(
        self, args: GenreClassificationArgs, system_prompt: str
    ) -> list[GenreClassificationMessage]:
        messages: list[GenreClassificationMessage] = []
        if args.messages:
            for msg in args.messages:
                role = (msg.role or "").strip()
                content = (msg.content or "").strip()
                if not content:
                    continue
                if not role:
                    role = "user"
                messages.append(
                    GenreClassificationMessage(role=role, content=content)
                )
        elif args.prompt and args.prompt.strip():
            messages.append(
                GenreClassificationMessage(
                    role="user", content=args.prompt.strip()
                )
            )
        else:
            raise ToolError("Provide prompt or messages.")

        if not messages:
            raise ToolError("No usable messages provided.")

        if system_prompt.strip():
            messages.insert(
                0,
                GenreClassificationMessage(
                    role="system", content=system_prompt.strip()
                ),
            )
        return messages

    def _call_llm(
        self,
        messages: list[GenreClassificationMessage],
        args: GenreClassificationArgs,
    ) -> str:
        api_base = (args.llm_api_base or self.config.llm_api_base).rstrip("/")
        url = api_base + "/chat/completions"
        payload = {
            "model": self._resolve_model(args),
            "messages": [msg.model_dump() for msg in messages],
            "temperature": args.llm_temperature,
            "max_tokens": args.llm_max_tokens,
            "stream": bool(args.llm_stream),
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=600)
        except urllib.error.URLError as exc:
            raise ToolError(f"LLM request failed: {exc}") from exc

        if not args.llm_stream:
            body = resp.read().decode("utf-8")
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError as exc:
                raise ToolError(f"LLM response parse failed: {exc}") from exc
            return parsed["choices"][0]["message"].get("content", "").strip()

        return self._read_streaming_response(resp)

    def _read_streaming_response(self, resp) -> str:
        parts: list[str] = []
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            if line.startswith("data:"):
                line = line[len("data:") :].strip()
            if line == "[DONE]":
                break
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            choice = chunk.get("choices", [{}])[0]
            delta = choice.get("delta") or choice.get("message") or {}
            content = delta.get("content")
            if content:
                parts.append(content)
                sys.stdout.write(content)
                sys.stdout.flush()
        if parts:
            sys.stdout.write("\n")
            sys.stdout.flush()
        return "".join(parts).strip()

    def _resolve_model(self, args: GenreClassificationArgs) -> str:
        return args.llm_model or self.config.llm_model

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, GenreClassificationArgs):
            return ToolCallDisplay(summary="genre_classification_recognition")
        return ToolCallDisplay(
            summary="genre_classification_recognition",
            details={
                "focus": event.args.focus,
                "show_steps": event.args.show_steps,
                "max_genres": event.args.max_genres,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, GenreClassificationResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = "Genre classification recognition complete"
        if event.result.errors:
            message = "Genre classification recognition finished with errors"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "answer": event.result.answer,
                "used_focus": event.result.used_focus,
                "show_steps": event.result.show_steps,
                "max_genres": event.result.max_genres,
                "template_source": event.result.template_source,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Genre classification recognition"
