"""Configuration for in-the-wild study evaluation."""
from evaluation.shared.config import StudyConfig, get_base_path


# In-the-wild study configuration
WILD_STUDY_CONFIG = StudyConfig(
    name="in_the_wild",
    base_path=get_base_path(),
    results_filename="results.csv",
    config_filename="in_the_wild.json",
    device_types=["mobile"],
    grouping_dimensions=["environment", "light_quality_indoor", "light_quality_outdoor", "angle", "main_movement"],
    has_annotations=True
)
