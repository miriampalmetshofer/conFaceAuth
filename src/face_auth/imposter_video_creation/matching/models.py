"""Data models for video matching strategies."""
from pydantic import BaseModel, Field, ConfigDict


class RandomSamplingConfig(BaseModel):
    """Configuration for the random sampling matching strategy."""
    imposters_per_genuine: int = Field(..., gt=0)
    random_seed: int = Field(...)

    model_config = ConfigDict(frozen=True)
