"""Configuration dataclasses for evaluation studies."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class StudyConfig:
    """Configuration for an evaluation study."""

    name: str
    base_path: Path
    results_filename: str
    config_filename: str
    device_types: List[str]
    grouping_dimensions: List[str]
    has_annotations: bool = True

    @property
    def results_path(self) -> Path:
        """Path to results CSV file."""
        return self.base_path / "data" / self.name / self.results_filename

    @property
    def config_path(self) -> Path:
        """Path to study configuration JSON."""
        return self.base_path / "configs" / self.config_filename

    @property
    def annotations_path(self) -> Path:
        """Path to annotations directory."""
        return self.base_path / "data" / "annotations"

    @property
    def schema_path(self) -> Path:
        """Path to annotation schema JSON."""
        return self.base_path / "src" / "evaluation" / self.name / "annotation_schema.json"

    @property
    def output_path(self) -> Path:
        """Path to output directory."""
        return self.base_path / "src" / "evaluation" / self.name / "output"


def get_base_path() -> Path:
    """Get project base path, handling both script and interactive execution."""
    try:
        # When running as script
        # From src/evaluation/shared/config.py -> go up 4 levels to project root
        base_path = Path(__file__).parent.parent.parent.parent
    except NameError:
        # When running interactively
        base_path = Path.cwd()
        if base_path.name == "evaluation":
            base_path = base_path.parent.parent
        elif base_path.name == "src":
            base_path = base_path.parent

    return base_path
