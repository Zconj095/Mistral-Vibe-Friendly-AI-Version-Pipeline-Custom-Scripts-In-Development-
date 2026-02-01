from __future__ import annotations

import re
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


HZ_RE = re.compile(r"(?P<value>\\d+(?:\\.\\d+)?)\\s*hz", re.IGNORECASE)


class ContextSpokenVocalWavelengthConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_source_bytes: int = Field(
        default=2_000_000, description="Maximum bytes per content."
    )
    default_frequency_hz: float = Field(
        default=200.0, description="Default frequency when none is provided."
    )
    min_frequency_hz: float = Field(
        default=50.0, description="Minimum allowed frequency."
    )
    max_frequency_hz: float = Field(
        default=2000.0, description="Maximum allowed frequency."
    )
    speed_of_sound_m_s: float = Field(
        default=343.0, description="Speed of sound in meters per second."
    )
    temperature_c: float = Field(
        default=20.0, description="Default temperature in Celsius."
    )


class ContextSpokenVocalWavelengthState(BaseToolState):
    pass


class ContextSpokenVocalWavelengthArgs(BaseModel):
    frequency_hz: float | None = Field(default=None, description="Single frequency.")
    frequencies_hz: list[float] | None = Field(
        default=None, description="Multiple frequencies."
    )
    content: str | None = Field(
        default=None, description="Content containing frequency values (e.g. 220Hz)."
    )
    path: str | None = Field(default=None, description="Path to content.")
    temperature_c: float | None = Field(
        default=None, description="Override temperature in Celsius."
    )
    speed_of_sound_m_s: float | None = Field(
        default=None, description="Override speed of sound."
    )
    voice_profile: str | None = Field(
        default=None, description="male, female, child, or custom."
    )
    include_reference_ranges: bool = Field(
        default=True, description="Include reference voice ranges."
    )


class VocalWavelength(BaseModel):
    frequency_hz: float
    wavelength_m: float
    wavelength_cm: float
    wavelength_in: float


class VoiceRange(BaseModel):
    label: str
    min_hz: float
    max_hz: float
    midpoint_hz: float
    midpoint_wavelength_m: float


class ContextSpokenVocalWavelengthResult(BaseModel):
    speed_of_sound_m_s: float
    temperature_c: float | None
    wavelengths: list[VocalWavelength]
    voice_ranges: list[VoiceRange]
    speech_prompt: str
    warnings: list[str]


class ContextSpokenVocalWavelength(
    BaseTool[
        ContextSpokenVocalWavelengthArgs,
        ContextSpokenVocalWavelengthResult,
        ContextSpokenVocalWavelengthConfig,
        ContextSpokenVocalWavelengthState,
    ],
    ToolUIData[
        ContextSpokenVocalWavelengthArgs,
        ContextSpokenVocalWavelengthResult,
    ],
):
    description: ClassVar[str] = (
        "Compute vocal wavelength from spoken pitch frequencies."
    )

    async def run(
        self, args: ContextSpokenVocalWavelengthArgs
    ) -> ContextSpokenVocalWavelengthResult:
        warnings: list[str] = []
        speed = self._resolve_speed(args)
        frequencies = self._collect_frequencies(args, warnings)
        if not frequencies:
            raise ToolError("No valid frequencies provided.")

        wavelengths = [self._to_wavelength(freq, speed) for freq in frequencies]
        voice_ranges = (
            self._voice_ranges(speed)
            if args.include_reference_ranges
            else []
        )
        speech_prompt = self._speech_prompt(wavelengths, args.voice_profile)

        return ContextSpokenVocalWavelengthResult(
            speed_of_sound_m_s=speed,
            temperature_c=args.temperature_c
            if args.temperature_c is not None
            else self.config.temperature_c,
            wavelengths=wavelengths,
            voice_ranges=voice_ranges,
            speech_prompt=speech_prompt,
            warnings=warnings,
        )

    def _resolve_speed(self, args: ContextSpokenVocalWavelengthArgs) -> float:
        if args.speed_of_sound_m_s is not None:
            if args.speed_of_sound_m_s <= 0:
                raise ToolError("speed_of_sound_m_s must be positive.")
            return args.speed_of_sound_m_s
        if args.temperature_c is not None:
            return 331.0 + 0.6 * args.temperature_c
        return self.config.speed_of_sound_m_s

    def _collect_frequencies(
        self, args: ContextSpokenVocalWavelengthArgs, warnings: list[str]
    ) -> list[float]:
        frequencies: list[float] = []
        if args.frequency_hz is not None:
            frequencies.append(args.frequency_hz)
        if args.frequencies_hz:
            frequencies.extend(args.frequencies_hz)

        if args.content or args.path:
            text = self._load_content(args)
            for match in HZ_RE.finditer(text):
                frequencies.append(float(match.group("value")))

        if not frequencies and args.voice_profile:
            profile = args.voice_profile.strip().lower()
            profile_ranges = self._profile_range(profile)
            if profile_ranges:
                frequencies.extend(profile_ranges)

        if not frequencies:
            warnings.append("No frequencies provided; using default frequency.")
            frequencies = [self.config.default_frequency_hz]

        valid: list[float] = []
        for freq in frequencies:
            if freq <= 0:
                warnings.append(f"Ignoring non-positive frequency: {freq}")
                continue
            if freq < self.config.min_frequency_hz or freq > self.config.max_frequency_hz:
                warnings.append(f"Frequency out of range: {freq}Hz")
            valid.append(freq)
        return valid

    def _profile_range(self, profile: str) -> list[float]:
        if profile == "male":
            return [85.0, 180.0]
        if profile == "female":
            return [165.0, 255.0]
        if profile == "child":
            return [250.0, 400.0]
        return []

    def _load_content(self, args: ContextSpokenVocalWavelengthArgs) -> str:
        if args.content and args.path:
            raise ToolError("Provide content or path, not both.")
        if args.content is not None:
            data = args.content.encode("utf-8")
            if len(data) > self.config.max_source_bytes:
                raise ToolError(
                    f"content exceeds max_source_bytes ({len(data)} > {self.config.max_source_bytes})."
                )
            return args.content
        if not args.path:
            raise ToolError("Provide content or path.")
        path = self._resolve_path(args.path)
        size = path.stat().st_size
        if size > self.config.max_source_bytes:
            raise ToolError(
                f"{path} exceeds max_source_bytes ({size} > {self.config.max_source_bytes})."
            )
        return path.read_text("utf-8", errors="ignore")

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.config.effective_workdir / path
        path = path.resolve()
        if not path.exists():
            raise ToolError(f"Path not found: {path}")
        if path.is_dir():
            raise ToolError(f"Path is a directory: {path}")
        return path

    def _to_wavelength(self, frequency_hz: float, speed: float) -> VocalWavelength:
        wavelength_m = speed / frequency_hz
        return VocalWavelength(
            frequency_hz=frequency_hz,
            wavelength_m=wavelength_m,
            wavelength_cm=wavelength_m * 100.0,
            wavelength_in=wavelength_m * 39.3701,
        )

    def _voice_ranges(self, speed: float) -> list[VoiceRange]:
        ranges = [
            ("male", 85.0, 180.0),
            ("female", 165.0, 255.0),
            ("child", 250.0, 400.0),
        ]
        result: list[VoiceRange] = []
        for label, min_hz, max_hz in ranges:
            midpoint = (min_hz + max_hz) / 2.0
            result.append(
                VoiceRange(
                    label=label,
                    min_hz=min_hz,
                    max_hz=max_hz,
                    midpoint_hz=midpoint,
                    midpoint_wavelength_m=speed / midpoint,
                )
            )
        return result

    def _speech_prompt(
        self, wavelengths: list[VocalWavelength], voice_profile: str | None
    ) -> str:
        if voice_profile:
            return f"Maintain {voice_profile} vocal pitch in the listed wavelength range."
        if wavelengths:
            primary = wavelengths[0]
            return (
                f"Target pitch {primary.frequency_hz:.1f}Hz "
                f"(wavelength {primary.wavelength_m:.2f}m)."
            )
        return "Maintain a steady vocal pitch."

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextSpokenVocalWavelengthArgs):
            return ToolCallDisplay(summary="context_spoken_vocal_wavelength")
        return ToolCallDisplay(
            summary="context_spoken_vocal_wavelength",
            details={
                "frequency_hz": event.args.frequency_hz,
                "frequencies_hz": event.args.frequencies_hz,
                "voice_profile": event.args.voice_profile,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextSpokenVocalWavelengthResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        count = len(event.result.wavelengths)
        message = f"Computed {count} vocal wavelength(s)"
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=event.result.warnings,
            details={
                "count": count,
                "speed_of_sound_m_s": event.result.speed_of_sound_m_s,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Computing vocal wavelength"
