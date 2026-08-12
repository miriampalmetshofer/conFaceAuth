"""Configuration for the imposter video creation feature."""

from typing import Annotated, Optional, Union

from pydantic import BaseModel, Field, ConfigDict

from face_auth.imposter_creation.matching.models import (
    ScenarioMatchingConfig,
    AllVideosMatchingConfig,
    RandomSamplingConfig,
)

MatchingStrategyConfig = Annotated[
    Union[ScenarioMatchingConfig, AllVideosMatchingConfig, RandomSamplingConfig],
    Field(discriminator="type"),
]


class ImposterConfig(BaseModel):
    """Imposter video creation configuration."""

    fps: int = Field(..., gt=0)
    genuine_user_seconds: Optional[float] = Field(None, gt=0)
    black_screen_seconds: float = Field(..., ge=0)
    impostor_seconds: Optional[float] = Field(None, gt=0)
    matching_strategy: MatchingStrategyConfig

    model_config = ConfigDict(frozen=True)
