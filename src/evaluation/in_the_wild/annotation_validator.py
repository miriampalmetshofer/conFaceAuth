"""Annotation validation and data distribution analysis."""
import json
from pathlib import Path
from typing import Optional, List
from collections import Counter, defaultdict
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Movement(str, Enum):
    """Valid movement types."""
    SITTING = "sitting"
    STANDING = "standing"
    WALKING = "walking"
    LYING = "lying"


class Occlusion(str, Enum):
    """Valid occlusion types."""
    NONE = "none"
    GLASSES = "glasses"
    HAND = "hand"
    EATING = "eating"
    HAT = "hat"
    OTHER = "other"


class Environment(str, Enum):
    """Valid environment types."""
    INDOOR = "indoor"
    OUTDOOR = "outdoor"


class Illumination(str, Enum):
    """Valid illumination types."""
    WELL_LIT = "well_lit"
    SHADOWS = "shadows"
    POORLY_LIT = "poorly_lit"


class Angle(str, Enum):
    """Valid camera angles."""
    DOWNWARD = "downward"
    STRAIGHT = "straight"
    UPWARD = "upward"


class DeviceMovement(str, Enum):
    """Valid device movement types."""
    STABLE = "stable"
    SLIGHT_MOVEMENT = "slight_movement"
    SIGNIFICANT_MOVEMENT = "significant_movement"


class FaceVisibility(str, Enum):
    """Valid face visibility types."""
    VISIBLE = "visible"
    TURNED_AWAY = "turned_away"


class OtherPeople(str, Enum):
    """Valid other people types."""
    NONE_OTHERS = "none_others"
    OTHERS_VISIBLE = "others_visible"


class AnnotationData(BaseModel):
    """Annotation data structure."""
    movement: List[Movement] = Field(..., min_length=1)
    occlusions: List[Occlusion] = Field(..., min_length=1)
    environment: Environment
    illumination: Illumination
    angle: Angle
    device_movement: DeviceMovement
    face_visibility: FaceVisibility
    other_people: OtherPeople
    light_change: Optional[bool] = None

    @field_validator('occlusions')
    @classmethod
    def validate_occlusions_none_conflict(cls, v: List[Occlusion]) -> List[Occlusion]:
        """Validate that 'none' cannot coexist with other occlusion values."""
        occlusion_values = [occ.value for occ in v]
        if 'none' in occlusion_values and len(occlusion_values) > 1:
            raise ValueError(
                f"'none' cannot be combined with other occlusions. Got: {occlusion_values}"
            )
        return v


class AnnotationFile(BaseModel):
    """Complete annotation file structure."""
    user: str
    video_filename: str
    timestamp: str
    annotations: AnnotationData


class ValidationResult:
    """Results from validating annotation files."""

    def __init__(self):
        self.valid_files: List[Path] = []
        self.invalid_files: List[tuple[Path, str]] = []
        self.annotations: List[tuple[Path, AnnotationFile]] = []


class DistributionReport:
    """Data distribution statistics."""

    def __init__(self):
        self.total_videos = 0
        self.by_participant = Counter()
        self.illumination = Counter()
        self.light_change = Counter()
        self.movement_individual = Counter()
        self.movement_complexity = Counter()
        self.occlusions_individual = Counter()
        self.occlusions_combinations = Counter()
        self.face_visibility = Counter()
        self.angle = Counter()
        self.device_movement = Counter()
        self.environment = Counter()
        self.other_people = Counter()


def validate_annotations(base_path: Path) -> ValidationResult:
    """Validate all annotation files."""
    result = ValidationResult()

    annotation_files = sorted(base_path.glob("**/annotations/*.json"))

    for file_path in annotation_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            annotation = AnnotationFile.model_validate(data)
            result.valid_files.append(file_path)
            result.annotations.append((file_path, annotation))

        except Exception as e:
            result.invalid_files.append((file_path, str(e)))

    return result


def analyze_distribution(validation_result: ValidationResult) -> DistributionReport:
    """Analyze data distribution across annotations."""
    report = DistributionReport()
    report.total_videos = len(validation_result.annotations)

    for file_path, annotation in validation_result.annotations:
        ann = annotation.annotations

        # Participant
        report.by_participant[annotation.user.lower()] += 1

        # Illumination
        report.illumination[ann.illumination.value] += 1

        # Light change
        if ann.light_change is not None:
            report.light_change[ann.light_change] += 1

        # Movement - individual and complexity
        movements = [m.value for m in ann.movement]
        for movement in movements:
            report.movement_individual[movement] += 1

        if len(movements) == 1:
            if movements[0] == "walking":
                report.movement_complexity["dynamic"] += 1
            else:
                report.movement_complexity["static"] += 1
        else:
            report.movement_complexity["mixed"] += 1

        # Occlusions - individual and combinations
        occlusions = sorted([o.value for o in ann.occlusions])
        for occlusion in occlusions:
            report.occlusions_individual[occlusion] += 1

        occlusion_key = ", ".join(occlusions)
        report.occlusions_combinations[occlusion_key] += 1

        # Face visibility
        report.face_visibility[ann.face_visibility.value] += 1

        # Angle
        report.angle[ann.angle.value] += 1

        # Device movement
        report.device_movement[ann.device_movement.value] += 1

        # Environment
        report.environment[ann.environment.value] += 1

        # Other people
        report.other_people[ann.other_people.value] += 1

    return report


def print_validation_results(result: ValidationResult):
    """Print validation results."""
    print("=" * 60)
    print("ANNOTATION VALIDATION RESULTS")
    print("=" * 60)
    print(f"\nTotal files found: {len(result.valid_files) + len(result.invalid_files)}")
    print(f"Valid files: {len(result.valid_files)}")
    print(f"Invalid files: {len(result.invalid_files)}")

    if result.invalid_files:
        print("\nINVALID FILES:")
        for file_path, error in result.invalid_files:
            print(f"\n  {file_path.name}")
            print(f"    Error: {error}")


def print_distribution_report(report: DistributionReport):
    """Print data distribution report."""
    print("\n" + "=" * 60)
    print("DATA DISTRIBUTION REPORT")
    print("=" * 60)
    print(f"\nTotal annotated videos: {report.total_videos}")

    print("\n--- By Participant ---")
    for participant, count in report.by_participant.most_common():
        print(f"  {participant}: {count}")

    print("\n--- Illumination ---")
    for value, count in report.illumination.most_common():
        marker = "✓" if count >= 5 else "✗"
        print(f"  {marker} {value}: {count}")

    print("\n--- Light Change (optional field) ---")
    for value, count in report.light_change.most_common():
        marker = "✓" if count >= 5 else "✗"
        print(f"  {marker} {value}: {count}")
    print(f"  Missing: {report.total_videos - sum(report.light_change.values())}")

    print("\n--- Movement (Individual) ---")
    for value, count in report.movement_individual.most_common():
        marker = "✓" if count >= 5 else "✗"
        print(f"  {marker} {value}: {count}")

    print("\n--- Movement Complexity ---")
    for value, count in report.movement_complexity.most_common():
        marker = "✓" if count >= 5 else "✗"
        print(f"  {marker} {value}: {count}")

    print("\n--- Occlusions (Individual) ---")
    for value, count in report.occlusions_individual.most_common():
        marker = "✓" if count >= 5 else "✗"
        print(f"  {marker} {value}: {count}")

    print("\n--- Occlusions (Combinations, top 10) ---")
    for value, count in report.occlusions_combinations.most_common(10):
        marker = "✓" if count >= 5 else "✗"
        print(f"  {marker} {value}: {count}")

    print("\n--- Face Visibility ---")
    for value, count in report.face_visibility.most_common():
        marker = "✓" if count >= 5 else "✗"
        print(f"  {marker} {value}: {count}")

    print("\n--- Camera Angle ---")
    for value, count in report.angle.most_common():
        marker = "✓" if count >= 5 else "✗"
        print(f"  {marker} {value}: {count}")

    print("\n--- Device Movement ---")
    for value, count in report.device_movement.most_common():
        marker = "✓" if count >= 5 else "✗"
        print(f"  {marker} {value}: {count}")

    print("\n--- Environment ---")
    for value, count in report.environment.most_common():
        marker = "✓" if count >= 5 else "✗"
        print(f"  {marker} {value}: {count}")

    print("\n--- Other People ---")
    for value, count in report.other_people.most_common():
        marker = "✓" if count >= 5 else "✗"
        print(f"  {marker} {value}: {count}")

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("\n✓ = Sufficient data (≥5 videos) for meaningful comparison")
    print("✗ = Insufficient data (<5 videos)")

    # Summarize viable comparisons
    viable_comparisons = []

    if all(count >= 5 for count in report.illumination.values()):
        viable_comparisons.append("Illumination (all categories)")
    elif sum(1 for count in report.illumination.values() if count >= 5) >= 2:
        viable = [k for k, v in report.illumination.items() if v >= 5]
        viable_comparisons.append(f"Illumination (subset: {', '.join(viable)})")

    if sum(1 for count in report.movement_complexity.values() if count >= 5) >= 2:
        viable = [k for k, v in report.movement_complexity.items() if v >= 5]
        viable_comparisons.append(f"Movement complexity: {', '.join(viable)}")

    if sum(1 for count in report.face_visibility.values() if count >= 5) >= 2:
        viable_comparisons.append("Face visibility")

    if sum(1 for count in report.occlusions_individual.values() if count >= 5) >= 2:
        viable = [k for k, v in report.occlusions_individual.items() if v >= 5]
        viable_comparisons.append(f"Occlusions: {', '.join(viable)}")

    print("\nViable Comparisons:")
    for comparison in viable_comparisons:
        print(f"  • {comparison}")


def main():
    """Main entry point."""
    base_path = Path(__file__).parent.parent.parent.parent / "data/in_the_wild"

    print(f"Validating annotations in: {base_path}\n")

    validation_result = validate_annotations(base_path)
    print_validation_results(validation_result)

    if validation_result.valid_files:
        report = analyze_distribution(validation_result)
        print_distribution_report(report)


if __name__ == "__main__":
    main()
