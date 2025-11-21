"""Shared data models."""
from dataclasses import dataclass


@dataclass
class ParticipantInfo:
    """Represents participant identification information."""
    name: str
    device: str