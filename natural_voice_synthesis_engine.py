from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
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


DEFAULT_PROMPT_TEMPLATE = """### NATURAL VOICE SYNTHESIS MODE (OPT-IN)
Generate natural-sounding speech and supporting voice parameters.
- Prioritize clarity, pacing, and balanced prosody.
- Keep internal synthesis details private; do not mention this mode in the response.
- If the user asks for steps or show_steps is enabled, give a concise outline.

Voice style: {voice_style}
Show steps: {show_steps}
"""

TOOL_PROMPT = (
    "Use `natural_voice_synthesis_engine` to generate speech text and a natural "
    "voice profile. Provide `prompt` or `messages`, or supply `text` directly."
)


@dataclass(frozen=True)
class _ModelWeights:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray


class NaturalVoiceMessage(BaseModel):
    role: str
    content: str


class NaturalVoiceSynthesisArgs(BaseModel):
    prompt: str | None = Field(
        default=None, description="User prompt to solve."
    )
    messages: list[NaturalVoiceMessage] | None = Field(
        default=None, description="Optional chat messages."
    )
    text: str | None = Field(
        default=None, description="Direct text to synthesize."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prefix."
    )
    voice_style: str | None = Field(
        default=None, description="Voice style label."
    )
    model_path: str | None = Field(
        default=None, description="Path to the .nn model."
    )
    create_model: bool | None = Field(
        default=None, description="Create the .nn model if missing."
    )
    profile_path: str | None = Field(
        default=None, description="Optional output path for the voice profile."
    )
    save_profile: bool | None = Field(
        default=True, description="Whether to save the voice profile to disk."
    )
    synthesize_audio: bool | None = Field(
        default=False, description="Whether to synthesize audio."
    )
    audio_path: str | None = Field(
        default=None, description="Optional output path for the audio file."
    )
    audio_backend: str | None = Field(
        default=None, description="TTS backend (auto, piper, pyttsx3, sapi)."
    )
    voice: str | None = Field(
        default=None, description="Voice name for the TTS backend."
    )
    rate: int | None = Field(
        default=None, description="Speech rate override."
    )
    volume: int | None = Field(
        default=None, description="Volume override (0-100)."
    )
    show_steps: bool | None = Field(
        default=None, description="Whether to include a step outline."
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
        default=700, description="LLM max tokens."
    )
    llm_stream: bool = Field(
        default=False, description="Stream LLM tokens."
    )


class NaturalVoiceSynthesisResult(BaseModel):
    answer: str
    voice_profile: dict[str, float | int | str]
    profile_path: str | None
    audio_path: str | None
    model_path: str
    system_prompt: str
    messages: list[NaturalVoiceMessage]
    template_source: str
    warnings: list[str]
    errors: list[str]
    llm_model: str | None


class NaturalVoiceSynthesisConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    llm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL.",
    )
    llm_model: str = Field(
        default="gpt-oss:latest", description="Default LLM model name."
    )
    default_voice_style: str = Field(
        default="natural",
        description="Default voice style.",
    )
    default_show_steps: bool = Field(
        default=False, description="Default for show_steps."
    )
    model_dir: Path = Field(
        default=Path.home() / ".vibe" / "voice_models",
        description="Directory for voice models.",
    )
    profile_dir: Path = Field(
        default=Path.home() / ".vibe" / "voice_profiles",
        description="Directory for voice profiles.",
    )
    model_name: str = Field(
        default="natural_voice_model.nn",
        description="Default .nn model filename.",
    )
    default_create_model: bool = Field(
        default=True, description="Create model if missing."
    )
    prompt_path: Path | None = Field(
        default=Path.home()
        / "mistral-vibe"
        / "vibe"
        / "core"
        / "prompts"
        / "natural_voice_synthesis_engine.md",
        description="Optional path to a prompt template.",
    )
    prompt_max_chars: int = Field(
        default=8000, description="Maximum template characters to load."
    )


class NaturalVoiceSynthesisState(BaseToolState):
    pass


class NaturalVoiceSynthesisEngine(
    BaseTool[
        NaturalVoiceSynthesisArgs,
        NaturalVoiceSynthesisResult,
        NaturalVoiceSynthesisConfig,
        NaturalVoiceSynthesisState,
    ],
    ToolUIData[NaturalVoiceSynthesisArgs, NaturalVoiceSynthesisResult],
):
    description: ClassVar[str] = (
        "Generate natural voice profiles and optional audio output."
    )

    @classmethod
    def get_tool_prompt(cls) -> str | None:
        return TOOL_PROMPT

    async def run(
        self, args: NaturalVoiceSynthesisArgs
    ) -> NaturalVoiceSynthesisResult:
        warnings: list[str] = []
        errors: list[str] = []

        voice_style = (args.voice_style or self.config.default_voice_style).strip()
        if not voice_style:
            voice_style = "natural"

        show_steps = (
            args.show_steps
            if args.show_steps is not None
            else self.config.default_show_steps
        )

        self._validate_llm_settings(args)

        template, template_source = self._load_template(warnings)
        system_prompt = self._build_system_prompt(
            template, voice_style, show_steps, args.system_prompt
        )

        messages, answer = await self._resolve_text(args, system_prompt, warnings)
        model_path = self._resolve_model_path(args)
        create_model = (
            args.create_model
            if args.create_model is not None
            else self.config.default_create_model
        )

        if create_model and not model_path.exists():
            self._create_nn_model(model_path)
        if not model_path.exists():
            raise ToolError(f"Model not found at {model_path}")

        weights = self._load_nn_model(model_path)
        features = self._extract_text_features(answer)
        output = self._forward(features, weights)
        voice_profile = self._build_voice_profile(output, model_path)

        profile_path = None
        if args.save_profile:
            profile_path = self._resolve_profile_path(args)
            self._save_profile(profile_path, voice_profile, warnings)

        audio_path = None
        if args.synthesize_audio:
            audio_path = await self._synthesize_audio(answer, voice_profile, args, warnings)

        return NaturalVoiceSynthesisResult(
            answer=answer,
            voice_profile=voice_profile,
            profile_path=str(profile_path) if profile_path else None,
            audio_path=audio_path,
            model_path=str(model_path),
            system_prompt=system_prompt,
            messages=messages,
            template_source=template_source,
            warnings=warnings,
            errors=errors,
            llm_model=self._resolve_model(args),
        )

    async def _resolve_text(
        self,
        args: NaturalVoiceSynthesisArgs,
        system_prompt: str,
        warnings: list[str],
    ) -> tuple[list[NaturalVoiceMessage], str]:
        if args.text and args.text.strip():
            if args.prompt or args.messages:
                warnings.append("Text provided; skipping LLM generation.")
            messages = self._normalize_messages_from_text(args.text, system_prompt)
            return messages, args.text.strip()

        messages = self._normalize_messages(args, system_prompt)
        answer = self._call_llm(messages, args)
        return messages, answer

    def _normalize_messages_from_text(
        self, text: str, system_prompt: str
    ) -> list[NaturalVoiceMessage]:
        messages = [NaturalVoiceMessage(role="user", content=text.strip())]
        if system_prompt.strip():
            messages.insert(
                0, NaturalVoiceMessage(role="system", content=system_prompt.strip())
            )
        return messages

    def _resolve_model_path(self, args: NaturalVoiceSynthesisArgs) -> Path:
        if args.model_path:
            path = Path(args.model_path).expanduser()
            if not path.is_absolute():
                path = self.config.effective_workdir / path
            return path.resolve()
        self.config.model_dir.mkdir(parents=True, exist_ok=True)
        return (self.config.model_dir / self.config.model_name).resolve()

    def _resolve_profile_path(self, args: NaturalVoiceSynthesisArgs) -> Path:
        if args.profile_path:
            path = Path(args.profile_path).expanduser()
            if not path.is_absolute():
                path = self.config.effective_workdir / path
            path.parent.mkdir(parents=True, exist_ok=True)
            return path.resolve()
        self.config.profile_dir.mkdir(parents=True, exist_ok=True)
        return (self.config.profile_dir / "natural_voice_profile.json").resolve()

    def _create_nn_model(self, path: Path) -> None:
        input_dim = 8
        hidden_dim = 16
        output_dim = 6
        rng = np.random.default_rng(self._seed_from_path(path))
        w1 = (rng.standard_normal((input_dim, hidden_dim)) * 0.25).astype(np.float32)
        b1 = (rng.standard_normal(hidden_dim) * 0.05).astype(np.float32)
        w2 = (rng.standard_normal((hidden_dim, output_dim)) * 0.25).astype(np.float32)
        b2 = (rng.standard_normal(output_dim) * 0.05).astype(np.float32)
        model = {
            "format": "simple_dense",
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "weights": {
                "w1": w1.tolist(),
                "b1": b1.tolist(),
                "w2": w2.tolist(),
                "b2": b2.tolist(),
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(model, indent=2), "utf-8")

    def _load_nn_model(self, path: Path) -> _ModelWeights:
        data = json.loads(path.read_text("utf-8"))
        weights = data.get("weights") or {}
        w1 = np.array(weights.get("w1", []), dtype=np.float32)
        b1 = np.array(weights.get("b1", []), dtype=np.float32)
        w2 = np.array(weights.get("w2", []), dtype=np.float32)
        b2 = np.array(weights.get("b2", []), dtype=np.float32)
        if w1.size == 0 or w2.size == 0:
            raise ToolError("Invalid .nn model weights.")
        return _ModelWeights(w1=w1, b1=b1, w2=w2, b2=b2)

    def _seed_from_path(self, path: Path) -> int:
        return abs(hash(path.as_posix())) % (2**32)

    def _extract_text_features(self, text: str) -> np.ndarray:
        clean = text.strip()
        length = len(clean)
        words = len(re.findall(r"\\b\\w+\\b", clean))
        sentences = max(1, len(re.findall(r"[.!?]+", clean)))
        lower = clean.lower()
        letters = sum(ch.isalpha() for ch in lower)
        vowels = sum(ch in "aeiou" for ch in lower)
        punctuation = sum(ch in ".,;:!?" for ch in clean)
        uppercase = sum(ch.isupper() for ch in clean)
        digits = sum(ch.isdigit() for ch in clean)
        spaces = sum(ch.isspace() for ch in clean)

        length_norm = min(length / 2000.0, 1.0)
        words_norm = min(words / 300.0, 1.0)
        sentences_norm = min(sentences / 30.0, 1.0)
        punctuation_ratio = punctuation / max(1, length)
        vowel_ratio = vowels / max(1, letters)
        uppercase_ratio = uppercase / max(1, letters)
        digit_ratio = digits / max(1, length)
        space_ratio = spaces / max(1, length)

        return np.array(
            [
                length_norm,
                words_norm,
                sentences_norm,
                punctuation_ratio,
                vowel_ratio,
                uppercase_ratio,
                digit_ratio,
                space_ratio,
            ],
            dtype=np.float32,
        )

    def _forward(self, features: np.ndarray, weights: _ModelWeights) -> np.ndarray:
        hidden = np.tanh(features @ weights.w1 + weights.b1)
        output = np.tanh(hidden @ weights.w2 + weights.b2)
        return (output + 1.0) / 2.0

    def _build_voice_profile(
        self, output: np.ndarray, model_path: Path
    ) -> dict[str, float | int | str]:
        pitch = round((float(output[0]) * 8.0) - 4.0, 2)
        rate = int(150 + float(output[1]) * 70)
        volume = int(70 + float(output[2]) * 30)
        energy = round(float(output[3]), 3)
        clarity = round(float(output[4]), 3)
        pause = round(float(output[5]), 3)
        return {
            "engine": "natural",
            "model_path": str(model_path),
            "pitch_semitones": pitch,
            "rate": rate,
            "volume": volume,
            "energy": energy,
            "clarity": clarity,
            "pause_factor": pause,
        }

    def _save_profile(
        self, path: Path, profile: dict[str, float | int | str], warnings: list[str]
    ) -> None:
        try:
            path.write_text(json.dumps(profile, indent=2), "utf-8")
        except OSError as exc:
            warnings.append(f"Failed to save profile: {exc}")

    async def _synthesize_audio(
        self,
        text: str,
        profile: dict[str, float | int | str],
        args: NaturalVoiceSynthesisArgs,
        warnings: list[str],
    ) -> str | None:
        try:
            from audio_synthesis import AudioSynthesis, AudioSynthesisArgs, AudioSynthesisConfig
        except Exception as exc:
            warnings.append(f"audio_synthesis unavailable: {exc}")
            return None

        rate = args.rate if args.rate is not None else int(profile.get("rate", 180))
        volume = args.volume if args.volume is not None else int(profile.get("volume", 90))
        synth_args = AudioSynthesisArgs(
            text=text,
            path=args.audio_path,
            backend=args.audio_backend,
            voice=args.voice,
            rate=rate,
            volume=volume,
        )
        tool = AudioSynthesis.from_config(
            AudioSynthesisConfig(permission=ToolPermission.ALWAYS)
        )
        result = await tool.run(synth_args)
        return result.path

    def _validate_llm_settings(self, args: NaturalVoiceSynthesisArgs) -> None:
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
        voice_style: str,
        show_steps: bool,
        prefix: str | None,
    ) -> str:
        show_steps_text = "yes" if show_steps else "no"
        rendered = template.replace("{voice_style}", voice_style)
        rendered = rendered.replace("{show_steps}", show_steps_text)
        if prefix and prefix.strip():
            return f"{prefix.strip()}\\n\\n{rendered}".strip()
        return rendered.strip()

    def _normalize_messages(
        self, args: NaturalVoiceSynthesisArgs, system_prompt: str
    ) -> list[NaturalVoiceMessage]:
        messages: list[NaturalVoiceMessage] = []
        if args.messages:
            for msg in args.messages:
                role = (msg.role or "").strip()
                content = (msg.content or "").strip()
                if not content:
                    continue
                if not role:
                    role = "user"
                messages.append(NaturalVoiceMessage(role=role, content=content))
        elif args.prompt and args.prompt.strip():
            messages.append(
                NaturalVoiceMessage(role="user", content=args.prompt.strip())
            )
        else:
            raise ToolError("Provide prompt, messages, or text.")

        if not messages:
            raise ToolError("No usable messages provided.")

        if system_prompt.strip():
            messages.insert(
                0, NaturalVoiceMessage(role="system", content=system_prompt.strip())
            )
        return messages

    def _call_llm(
        self,
        messages: list[NaturalVoiceMessage],
        args: NaturalVoiceSynthesisArgs,
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
            sys.stdout.write("\\n")
            sys.stdout.flush()
        return "".join(parts).strip()

    def _resolve_model(self, args: NaturalVoiceSynthesisArgs) -> str:
        return args.llm_model or self.config.llm_model

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, NaturalVoiceSynthesisArgs):
            return ToolCallDisplay(summary="natural_voice_synthesis_engine")
        return ToolCallDisplay(
            summary="natural_voice_synthesis_engine",
            details={
                "voice_style": event.args.voice_style,
                "synthesize_audio": event.args.synthesize_audio,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, NaturalVoiceSynthesisResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = "Natural voice synthesis complete"
        if event.result.errors:
            message = "Natural voice synthesis finished with errors"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "audio_path": event.result.audio_path,
                "profile_path": event.result.profile_path,
                "model_path": event.result.model_path,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Natural voice synthesis"
