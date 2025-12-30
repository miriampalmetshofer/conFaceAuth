"""Configuration module for face authentication system."""

from face_auth.config.models import (
    ApplicationConfig,
    PathsConfig,
    AuthenticationConfig,
    EnrollmentConfig,
    EnrollmentVideoPreference,
    ModelConfig,
    ProcessingConfig,
    LoggingConfig,
    Participant,
    ProcessingContext,
    Scenario,
    HeadRotation
)
from face_auth.config.loader import ConfigLoader
from face_auth.config.logging_config import setup_logging, get_logger

__all__ = [
    'ApplicationConfig',
    'PathsConfig',
    'AuthenticationConfig',
    'EnrollmentConfig',
    'EnrollmentVideoPreference',
    'ModelConfig',
    'ProcessingConfig',
    'LoggingConfig',
    'Participant',
    'ProcessingContext',
    'Scenario',
    'HeadRotation',
    'ConfigLoader',
    'setup_logging',
    'get_logger'
]
