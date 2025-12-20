"""Configuration for controlled study evaluation."""
from evaluation.shared.config import StudyConfig, get_base_path


# Controlled study configuration
CONTROLLED_STUDY_CONFIG = StudyConfig(
    name="controlled_study",
    base_path=get_base_path(),
    results_filename="results.csv",
    config_filename="controlled_study.json",
    device_types=["mobile", "desktop"],
    grouping_dimensions=["device", "scenario"],
    has_annotations=True
)
