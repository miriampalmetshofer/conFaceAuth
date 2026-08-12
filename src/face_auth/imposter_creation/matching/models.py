"""Data models for video matching strategies."""
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


class ScenarioMatchingConfig(BaseModel):
    """Configuration for the scenario matching strategy."""
    type: Literal["scenario"]

    model_config = ConfigDict(frozen=True)


class AllVideosMatchingConfig(BaseModel):
    """Configuration for the all-videos matching strategy."""
    type: Literal["all_videos"]

    model_config = ConfigDict(frozen=True)


class RandomSamplingConfig(BaseModel):
    """Configuration for the random sampling matching strategy."""
    type: Literal["random_sampling"]
    imposters_per_genuine: int = Field(..., gt=0)
    random_seed: int = Field(...)

    model_config = ConfigDict(frozen=True)
