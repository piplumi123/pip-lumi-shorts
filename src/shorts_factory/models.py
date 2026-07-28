from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Shot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number: int = Field(ge=1, le=6)
    beat: Literal["hook", "escalation", "wrong_turn", "discovery", "payoff", "twist"]
    narration: str = Field(min_length=3, max_length=180)
    caption: str = Field(min_length=2, max_length=48)
    scene_prompt: str = Field(min_length=20, max_length=900)
    motion_prompt: str = Field(min_length=20, max_length=700)
    sound_cue: str = Field(min_length=2, max_length=80)

    @field_validator("narration", "caption", "scene_prompt", "motion_prompt", "sound_cue")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return " ".join(value.split())


class EpisodePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_number: int = Field(ge=1)
    episode_title: str = Field(min_length=3, max_length=70)
    story_question: str = Field(min_length=5, max_length=160)
    synopsis: str = Field(min_length=20, max_length=500)
    continuity_advance: str = Field(min_length=8, max_length=240)
    youtube_title: str = Field(min_length=8, max_length=70)
    youtube_description: str = Field(min_length=30, max_length=900)
    hashtags: list[str] = Field(min_length=3, max_length=6)
    shots: list[Shot] = Field(min_length=6, max_length=6)

    @field_validator("episode_title", "story_question", "synopsis", "continuity_advance")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("hashtags")
    @classmethod
    def normalize_hashtags(cls, values: list[str]) -> list[str]:
        cleaned = []
        for value in values:
            tag = "".join(value.split())
            cleaned.append(tag if tag.startswith("#") else f"#{tag}")
        return cleaned

    @model_validator(mode="after")
    def validate_story_shape(self) -> "EpisodePlan":
        expected_numbers = list(range(1, 7))
        if [shot.number for shot in self.shots] != expected_numbers:
            raise ValueError("shots must be numbered 1 through 6 in order")
        expected_beats = [
            "hook",
            "escalation",
            "wrong_turn",
            "discovery",
            "payoff",
            "twist",
        ]
        if [shot.beat for shot in self.shots] != expected_beats:
            raise ValueError(f"shot beats must be {expected_beats}")
        words = sum(len(shot.narration.split()) for shot in self.shots)
        if not 55 <= words <= 82:
            raise ValueError(f"narration must contain 55–82 words, got {words}")
        return self


class QualityReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool
    hook_score: int = Field(ge=1, le=10)
    story_score: int = Field(ge=1, le=10)
    visual_action_score: int = Field(ge=1, le=10)
    emotional_payoff_score: int = Field(ge=1, le=10)
    replay_score: int = Field(ge=1, le=10)
    revision_notes: list[str] = Field(max_length=6)


class FrameReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool
    character_consistency_score: int = Field(ge=1, le=10)
    composition_score: int = Field(ge=1, le=10)
    story_clarity_score: int = Field(ge=1, le=10)
    retry_shots: list[int] = Field(max_length=2)
    problems: list[str] = Field(max_length=6)

    @field_validator("retry_shots")
    @classmethod
    def valid_shot_numbers(cls, values: list[int]) -> list[int]:
        return sorted({value for value in values if 1 <= value <= 6})[:2]

