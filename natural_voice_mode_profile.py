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


DEFAULT_PROMPT_TEMPLATE = """### NATURAL VOICE MODE PROFILE (OPT-IN)
Generate natural-sounding speech and a blended voice mode profile.
- Prioritize clarity, balanced prosody, and natural pacing.
- Keep internal synthesis details private; do not mention this mode in the response.
- If the user asks for steps or show_steps is enabled, give a concise outline.

Voice style: {voice_style}
Show steps: {show_steps}
"""

TOOL_PROMPT = (
    "Use `natural_voice_mode_profile` to blend natural and neural voice profiles. "
    "Provide `prompt` or `messages`, or supply `text` directly."
)


class NaturalVoiceModeMessage(BaseModel):
    role: str
    content: str


class NaturalVoiceModeArgs(BaseModel):
    prompt: str | None = Field(
        default=None, description="User prompt to solve."
    )
    messages: list[NaturalVoiceModeMessage] | None = Field(
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
    natural_model_path: str | None = Field(
        default=None, description="Path to the natural .nn model."
    )
    neural_model_path: str | None = Field(
        default=None, description="Path to the neural .h5 model."
    )
    create_natural_model: bool | None = Field(
        default=None, description="Create the .nn model if missing."
    )
    create_neural_model: bool | None = Field(
        default=None, description="Create the .h5 model if missing."
    )
    blend_weight: float | None = Field(
        default=None,
        description="Blend weight for the natural profile (0-1).",
    )
    profile_path: str | None = Field(
        default=None, description="Optional output path for the blended profile."
    )
    save_profile: bool | None = Field(
        default=True, description="Whether to save the blended profile to disk."
    )
    include_components: bool | None = Field(
        default=False,
        description="Include component profiles in the result.",
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


class NaturalVoiceModeResult(BaseModel):
    answer: str
    blended_profile: dict[str, float | int | str]
    natural_profile: dict[str, float | int | str] | None
    neural_profile: dict[str, float | int | str] | None
    profile_path: str | None
    audio_path: str | None
    system_prompt: str
    messages: list[NaturalVoiceModeMessage]
    template_source: str
    warnings: list[str]
    errors: list[str]
    llm_model: str | None
    blend_weight: float


class NaturalVoiceModeConfig(BaseToolConfig):
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
    default_blend_weight: float = Field(
        default=0.55,
        description="Default blend weight for the natural profile.",
    )
    profile_dir: Path = Field(
        default=Path.home() / ".vibe" / "voice_profiles",
        description="Directory for voice profiles.",
    )
    prompt_path: Path | None = Field(
        default=Path.home()
        / "mistral-vibe"
        / "vibe"
        / "core"
        / "prompts"
        / "natural_voice_mode_profile.md",
        description="Optional path to a prompt template.",
    )
    prompt_max_chars: int = Field(
        default=8000, description="Maximum template characters to load."
    )


class NaturalVoiceModeState(BaseToolState):
    pass


class NaturalVoiceModeProfile(
    BaseTool[
        NaturalVoiceModeArgs,
        NaturalVoiceModeResult,
        NaturalVoiceModeConfig,
        NaturalVoiceModeState,
    ],
    ToolUIData[NaturalVoiceModeArgs, NaturalVoiceModeResult],
):
    description: ClassVar[str] = (
        "Blend natural and neural voice profiles into a single voice mode profile."
    )

    @classmethod
    def get_tool_prompt(cls) -> str | None:
        return TOOL_PROMPT

    async def run(self, args: NaturalVoiceModeArgs) -> NaturalVoiceModeResult:
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

        blend_weight = (
            args.blend_weight
            if args.blend_weight is not None
            else self.config.default_blend_weight
        )
        blend_weight = self._clamp_blend_weight(blend_weight, warnings)

        self._validate_llm_settings(args)

        template, template_source = self._load_template(warnings)
        system_prompt = self._build_system_prompt(
            template, voice_style, show_steps, args.system_prompt
        )

        messages, answer = await self._resolve_text(args, system_prompt, warnings)

        natural_profile = await self._run_natural_engine(
            answer, voice_style, args, warnings, errors
        )
        neural_profile = await self._run_neural_engine(
            answer, voice_style, args, warnings, errors
        )

        blended_profile = self._blend_profiles(
            natural_profile, neural_profile, blend_weight
        )

        profile_path = None
        if args.save_profile:
            profile_path = self._resolve_profile_path(args)
            self._save_profile(profile_path, blended_profile, warnings)

        audio_path = None
        if args.synthesize_audio:
            audio_path = await self._synthesize_audio(
                answer, blended_profile, args, warnings
            )

        natural_out = natural_profile if args.include_components else None
        neural_out = neural_profile if args.include_components else None

        return NaturalVoiceModeResult(
            answer=answer,
            blended_profile=blended_profile,
            natural_profile=natural_out,
            neural_profile=neural_out,
            profile_path=str(profile_path) if profile_path else None,
            audio_path=audio_path,
            system_prompt=system_prompt,
            messages=messages,
            template_source=template_source,
            warnings=warnings,
            errors=errors,
            llm_model=self._resolve_model(args),
            blend_weight=blend_weight,
        )

    async def _resolve_text(
        self,
        args: NaturalVoiceModeArgs,
        system_prompt: str,
        warnings: list[str],
    ) -> tuple[list[NaturalVoiceModeMessage], str]:
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
    ) -> list[NaturalVoiceModeMessage]:
        messages = [NaturalVoiceModeMessage(role="user", content=text.strip())]
        if system_prompt.strip():
            messages.insert(
                0, NaturalVoiceModeMessage(role="system", content=system_prompt.strip())
            )
        return messages

    async def _run_natural_engine(
        self,
        text: str,
        voice_style: str,
        args: NaturalVoiceModeArgs,
        warnings: list[str],
        errors: list[str],
    ) -> dict[str, float | int | str]:
        try:
            from natural_voice_synthesis_engine import (
                NaturalVoiceSynthesisEngine,
                NaturalVoiceSynthesisArgs,
                NaturalVoiceSynthesisConfig,
            )
        except Exception as exc:
            raise ToolError("natural_voice_synthesis_engine unavailable.") from exc

        tool = NaturalVoiceSynthesisEngine.from_config(
            NaturalVoiceSynthesisConfig(permission=ToolPermission.ALWAYS)
        )
        tool_args = NaturalVoiceSynthesisArgs(
            text=text,
            voice_style=voice_style,
            model_path=args.natural_model_path,
            create_model=args.create_natural_model,
            save_profile=False,
            synthesize_audio=False,
            show_steps=False,
        )
        result = await tool.run(tool_args)
        for warning in result.warnings:
            warnings.append(f"natural_voice_synthesis_engine: {warning}")
        for err in result.errors:
            errors.append(f"natural_voice_synthesis_engine: {err}")
        return result.voice_profile

    async def _run_neural_engine(
        self,
        text: str,
        voice_style: str,
        args: NaturalVoiceModeArgs,
        warnings: list[str],
        errors: list[str],
    ) -> dict[str, float | int | str]:
        try:
            from neural_voice_engine import (
                NeuralVoiceEngine,
                NeuralVoiceEngineArgs,
                NeuralVoiceEngineConfig,
            )
        except Exception as exc:
            raise ToolError("neural_voice_engine unavailable.") from exc

        tool = NeuralVoiceEngine.from_config(
            NeuralVoiceEngineConfig(permission=ToolPermission.ALWAYS)
        )
        tool_args = NeuralVoiceEngineArgs(
            text=text,
            voice_style=voice_style,
            model_path=args.neural_model_path,
            create_model=args.create_neural_model,
            save_profile=False,
            synthesize_audio=False,
            show_steps=False,
        )
        result = await tool.run(tool_args)
        for warning in result.warnings:
            warnings.append(f"neural_voice_engine: {warning}")
        for err in result.errors:
            errors.append(f"neural_voice_engine: {err}")
        return result.voice_profile

    def _blend_profiles(
        self,
        natural_profile: dict[str, float | int | str],
        neural_profile: dict[str, float | int | str],
        blend_weight: float,
    ) -> dict[str, float | int | str]:
        natural_weight = blend_weight
        neural_weight = 1.0 - blend_weight

        pitch = self._blend_value(
            natural_profile.get("pitch_semitones"),
            neural_profile.get("pitch_semitones"),
            natural_weight,
        )
        rate = self._blend_value(
            natural_profile.get("rate"),
            neural_profile.get("rate"),
            natural_weight,
        )
        volume = self._blend_value(
            natural_profile.get("volume"),
            neural_profile.get("volume"),
            natural_weight,
        )

        energy = self._blend_value(
            natural_profile.get("energy"),
            neural_profile.get("intensity"),
            natural_weight,
        )
        clarity = self._blend_value(
            natural_profile.get("clarity"),
            neural_profile.get("articulation"),
            natural_weight,
        )
        pause_factor = self._to_float(natural_profile.get("pause_factor"))
        timbre = self._to_float(neural_profile.get("timbre"))
        warmth = self._to_float(neural_profile.get("warmth"))
        articulation = self._to_float(neural_profile.get("articulation"))
        breathiness = self._to_float(neural_profile.get("breathiness"))
        intensity = self._to_float(neural_profile.get("intensity"))

        blended: dict[str, float | int | str] = {
            "engine": "natural_voice_mode",
            "natural_weight": round(natural_weight, 3),
            "neural_weight": round(neural_weight, 3),
            "natural_model_path": str(natural_profile.get("model_path", "")),
            "neural_model_path": str(neural_profile.get("model_path", "")),
        }
        if pitch is not None:
            blended["pitch_semitones"] = round(pitch, 2)
        if rate is not None:
            blended["rate"] = int(round(rate))
        if volume is not None:
            blended["volume"] = int(round(volume))
        if energy is not None:
            blended["energy"] = round(energy, 3)
        if clarity is not None:
            blended["clarity"] = round(clarity, 3)
        if pause_factor is not None:
            blended["pause_factor"] = round(pause_factor, 3)
        if timbre is not None:
            blended["timbre"] = round(timbre, 3)
        if warmth is not None:
            blended["warmth"] = round(warmth, 3)
        if articulation is not None:
            blended["articulation"] = round(articulation, 3)
        if breathiness is not None:
            blended["breathiness"] = round(breathiness, 3)
        if intensity is not None:
            blended["intensity"] = round(intensity, 3)

        return blended

    def _blend_value(
        self,
        natural_value: float | int | str | None,
        neural_value: float | int | str | None,
        weight: float,
    ) -> float | None:
        left = self._to_float(natural_value)
        right = self._to_float(neural_value)
        if left is None and right is None:
            return None
        if left is None:
            return right
        if right is None:
            return left
        return (left * weight) + (right * (1.0 - weight))

    def _to_float(self, value: float | int | str | None) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _resolve_profile_path(self, args: NaturalVoiceModeArgs) -> Path:
        if args.profile_path:
            path = Path(args.profile_path).expanduser()
            if not path.is_absolute():
                path = self.config.effective_workdir / path
            path.parent.mkdir(parents=True, exist_ok=True)
            return path.resolve()
        self.config.profile_dir.mkdir(parents=True, exist_ok=True)
        return (self.config.profile_dir / "natural_voice_mode_profile.json").resolve()

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
        args: NaturalVoiceModeArgs,
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

    def _validate_llm_settings(self, args: NaturalVoiceModeArgs) -> None:
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
            return f"{prefix.strip()}\n\n{rendered}".strip()
        return rendered.strip()

    def _normalize_messages(
        self, args: NaturalVoiceModeArgs, system_prompt: str
    ) -> list[NaturalVoiceModeMessage]:
        messages: list[NaturalVoiceModeMessage] = []
        if args.messages:
            for msg in args.messages:
                role = (msg.role or "").strip()
                content = (msg.content or "").strip()
                if not content:
                    continue
                if not role:
                    role = "user"
                messages.append(NaturalVoiceModeMessage(role=role, content=content))
        elif args.prompt and args.prompt.strip():
            messages.append(
                NaturalVoiceModeMessage(role="user", content=args.prompt.strip())
            )
        else:
            raise ToolError("Provide prompt, messages, or text.")

        if not messages:
            raise ToolError("No usable messages provided.")

        if system_prompt.strip():
            messages.insert(
                0, NaturalVoiceModeMessage(role="system", content=system_prompt.strip())
            )
        return messages

    def _call_llm(
        self,
        messages: list[NaturalVoiceModeMessage],
        args: NaturalVoiceModeArgs,
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

    def _resolve_model(self, args: NaturalVoiceModeArgs) -> str:
        return args.llm_model or self.config.llm_model

    def _clamp_blend_weight(self, value: float, warnings: list[str]) -> float:
        if value < 0.0:
            warnings.append("blend_weight below 0; clamped to 0.")
            return 0.0
        if value > 1.0:
            warnings.append("blend_weight above 1; clamped to 1.")
            return 1.0
        return float(value)

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, NaturalVoiceModeArgs):
            return ToolCallDisplay(summary="natural_voice_mode_profile")
        return ToolCallDisplay(
            summary="natural_voice_mode_profile",
            details={
                "voice_style": event.args.voice_style,
                "blend_weight": event.args.blend_weight,
                "synthesize_audio": event.args.synthesize_audio,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, NaturalVoiceModeResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        message = "Natural voice mode profile complete"
        if event.result.errors:
            message = "Natural voice mode profile finished with errors"
        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=event.result.warnings,
            details={
                "audio_path": event.result.audio_path,
                "profile_path": event.result.profile_path,
                "blend_weight": event.result.blend_weight,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Natural voice mode profile"
