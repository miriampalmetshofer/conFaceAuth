"""Analyze in-the-wild annotation groups against authentication metrics.

This script intentionally uses only the Python standard library so it can be run
without the project environment. It joins session-level annotation JSON files to
the in-the-wild evaluation CSV and summarizes trends for genuine-user segments.
"""
import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS = PROJECT_ROOT / "data/in_the_wild/_results_archive/V10/results.csv"
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "data/in_the_wild/mobile"

THESIS_GROUP_KEYS = ["lighting_simple", "visibility_simple", "movement_simple"]
RAW_GROUP_KEYS = [
    "illumination",
    "face_visibility",
    "occlusion_simple",
    "angle_simple",
    "environment",
]

RAW_CHALLENGING_ILLUMINATIONS = {"shadows", "poorly_lit"}
RAW_VISIBILITY_REDUCING_OCCLUSIONS = {"hand", "eating", "hat", "other"}
RAW_SIGNIFICANT_DEVICE_MOVEMENT = "significant_movement"
RAW_WALKING_BODY_MOVEMENT = "walking"
RAW_FACE_OUT_OF_VIEW = "turned_away"

HARD_CONDITION_ORDER = ["lighting", "visibility", "movement"]

HARD_CONDITION_DEFINITIONS = {
    "visibility": "Face visibility is out of view for more than 2 seconds, or occlusion is hand, eating or drinking, headwear, or other",
    "movement": "Device movement is significant movement, body movement is walking, or multiple body movement values are selected",
    "lighting": "Illumination is uneven or shadowed, illumination is poorly lit, or light change is Yes",
}

THESIS_CONDITION_LABELS = {
    "visibility": "Reduced visibility",
    "movement": "Significant movement",
    "lighting": "Challenging lighting",
}

THESIS_PROFILE_ROWS = [
    (0, [()]),
    (1, [("visibility",), ("lighting",), ("movement",)]),
    (2, [("visibility", "lighting"), ("visibility", "movement"), ("lighting", "movement")]),
    (3, [("visibility", "lighting", "movement")]),
]

THESIS_FACTOR_TABLES = [
    (
        "Visibility",
        "visibility_simple",
        [("Clear", "easy"), ("Reduced", "hard")],
    ),
    (
        "Lighting",
        "lighting_simple",
        [("No challenging lighting", "easy"), ("Challenging lighting", "hard")],
    ),
    (
        "Movement",
        "movement_simple",
        [("No significant movement", "easy"), ("Significant movement", "hard")],
    ),
]


def order_hard_conditions(conditions: list[str]) -> list[str]:
    order = {condition: index for index, condition in enumerate(HARD_CONDITION_ORDER)}
    return sorted(conditions, key=lambda condition: order[condition])


def has_challenging_lighting(annotation_values: dict) -> bool:
    return (
        annotation_values.get("illumination") in RAW_CHALLENGING_ILLUMINATIONS
        or annotation_values.get("light_change") is True
    )


def has_reduced_visibility(annotation_values: dict) -> bool:
    occlusions = set(annotation_values.get("occlusions", []))
    return (
        annotation_values.get("face_visibility") == RAW_FACE_OUT_OF_VIEW
        or bool(occlusions & RAW_VISIBILITY_REDUCING_OCCLUSIONS)
    )


def has_significant_movement(annotation_values: dict) -> bool:
    movement = annotation_values.get("movement", [])
    return (
        annotation_values.get("device_movement") == RAW_SIGNIFICANT_DEVICE_MOVEMENT
        or RAW_WALKING_BODY_MOVEMENT in movement
        or len(movement) > 1
    )


def parse_composed_path(video_path: str) -> tuple[str | None, str | None]:
    """Return genuine and unauthorized video ids from a composed video path."""
    name = Path(video_path).name
    if name.endswith(".composed"):
        name = name[:-len(".composed")]
    if "_vs_" not in name:
        return None, None
    return tuple(name.split("_vs_", 1))


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_annotations(base_path: Path) -> dict[str, dict]:
    annotations = {}
    for path in sorted(base_path.glob("*/annotations/*.json")):
        data = json.loads(path.read_text())
        video_id = data["video_filename"].removesuffix(".mp4")
        annotations[video_id] = data
    return annotations


def load_genuine_rows(results_path: Path) -> dict[str, list[dict]]:
    """Load rows belonging to the genuine segment of each composed session."""
    rows_by_video = defaultdict(list)
    with results_path.open(newline="") as file:
        for row in csv.DictReader(file):
            genuine_id, _ = parse_composed_path(row["video_path"])
            if genuine_id and row["source_type"] == genuine_id:
                rows_by_video[genuine_id].append(row)
    return rows_by_video


def deduplicate_frames(rows: list[dict]) -> list[dict]:
    """Remove repeated genuine frames caused by multiple imposter pairings."""
    by_frame = {}
    for row in rows:
        by_frame.setdefault(row["frame"], row)
    return list(by_frame.values())


def calculate_metrics(rows_by_video: dict[str, list[dict]]) -> dict[str, dict]:
    metrics = {}
    for video_id, rows in rows_by_video.items():
        rows = deduplicate_frames(rows)
        trust = [to_float(row["trust_score"]) for row in rows]
        similarity = [to_float(row["similarity"]) for row in rows if row["similarity"]]
        face_detected = [row["face_detected"] == "True" for row in rows]
        locked = [row for row in rows if row["predicted_state"] == "Locked"]

        metrics[video_id] = {
            "mean_trust": statistics.mean(trust),
            "median_trust": statistics.median(trust),
            "min_trust": min(trust),
            "mean_similarity": statistics.mean(similarity) if similarity else math.nan,
            "face_detection_rate": sum(face_detected) / len(face_detected),
            "false_rejection": bool(locked),
            "first_lock_time": min((to_float(row["frame"]) / 30.0 for row in locked), default=None),
            "below_04_rate": sum(value < 0.4 for value in trust) / len(trust),
            "below_05_rate": sum(value < 0.5 for value in trust) / len(trust),
        }
    return metrics


def group_values(annotation: dict) -> dict[str, str]:
    ann = annotation["annotations"]
    occlusions = set(ann.get("occlusions", []))

    return {
        "illumination": ann.get("illumination"),
        "lighting_simple": "hard" if has_challenging_lighting(ann) else "easy",
        "face_visibility": ann.get("face_visibility"),
        "occlusion_simple": "none"
        if occlusions in ({"none"}, {"glasses"})
        else "visibility_reducing_occlusion",
        "visibility_simple": "hard" if has_reduced_visibility(ann) else "easy",
        "angle_simple": "straight" if ann.get("angle") == "straight" else "non_straight",
        "movement_simple": "hard" if has_significant_movement(ann) else "easy",
        "environment": ann.get("environment"),
    }


def hard_conditions(annotation: dict) -> list[str]:
    ann = annotation["annotations"]
    conditions = []

    if has_challenging_lighting(ann):
        conditions.append("lighting")
    if has_reduced_visibility(ann):
        conditions.append("visibility")
    if has_significant_movement(ann):
        conditions.append("movement")

    return order_hard_conditions(conditions)


def fmt(value) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_section(title: str):
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)


def summarize_metric_values(values: list[dict]) -> dict[str, str]:
    false_rejections = sum(value["false_rejection"] for value in values)
    lock_times = [value["first_lock_time"] for value in values if value["first_lock_time"] is not None]
    return {
        "n": str(len(values)),
        "frr": f"{false_rejections}/{len(values)} ({false_rejections / len(values) * 100:.1f}%)",
        "mean_trust": fmt(statistics.mean(value["mean_trust"] for value in values)),
        "min_trust_median": fmt(statistics.median(value["min_trust"] for value in values)),
        "mean_similarity": fmt(statistics.mean(value["mean_similarity"] for value in values)),
        "face_detection_rate": fmt(statistics.mean(value["face_detection_rate"] for value in values)),
        "below_04_rate": fmt(statistics.mean(value["below_04_rate"] for value in values)),
        "below_05_rate": fmt(statistics.mean(value["below_05_rate"] for value in values)),
        "first_lock_time_mean": fmt(statistics.mean(lock_times) if lock_times else None),
    }


def print_metric_table_header(first_column: str, width: int):
    print(
        f"{first_column:<{width}}"
        f"{'N':>4}  "
        f"{'FRR':>13}  "
        f"{'GT mean':>8}  "
        f"{'Min GT med':>10}  "
        f"{'Sim mean':>8}  "
        f"{'Face det.':>9}  "
        f"{'<0.4':>7}  "
        f"{'<0.5':>7}  "
        f"{'GKT mean':>8}"
    )
    print("-" * 92)


def print_metric_table_row(label: str, values: list[dict], width: int):
    summary = summarize_metric_values(values)
    print(
        f"{label:<{width}}"
        f"{summary['n']:>4}  "
        f"{summary['frr']:>13}  "
        f"{summary['mean_trust']:>8}  "
        f"{summary['min_trust_median']:>10}  "
        f"{summary['mean_similarity']:>8}  "
        f"{summary['face_detection_rate']:>9}  "
        f"{summary['below_04_rate']:>7}  "
        f"{summary['below_05_rate']:>7}  "
        f"{summary['first_lock_time_mean']:>8}"
    )


def print_distribution(metrics: dict[str, dict], annotations: dict[str, dict], group_keys: list[str]):
    print_section("DISTRIBUTIONS FOR GENUINE VIDEOS USED IN RESULTS")
    for key in group_keys:
        counter = Counter(group_values(annotations[video_id])[key] for video_id in metrics if video_id in annotations)
        values = ", ".join(f"{value}={count}" for value, count in counter.most_common())
        print(f"{key:<24} {values}")


def print_group_summary(metrics: dict[str, dict], annotations: dict[str, dict], group_key: str):
    groups = defaultdict(list)
    for video_id, metric in metrics.items():
        if video_id in annotations:
            groups[group_values(annotations[video_id])[group_key]].append(metric)

    print(f"\n{group_key.upper()}")
    print("-" * 92)
    print_metric_table_header("Group", 34)
    for group, values in sorted(groups.items(), key=lambda item: str(item[0])):
        print_metric_table_row(str(group), values, 34)


def condition_profile_label(conditions: list[str]) -> str:
    return "none" if not conditions else " + ".join(conditions)


def thesis_condition_profile_label(conditions: tuple[str, ...]) -> str:
    if not conditions:
        return "None"
    return " + ".join(THESIS_CONDITION_LABELS[condition] for condition in conditions)


def percentage(numerator: int, denominator: int) -> float:
    return numerator / denominator * 100 if denominator else math.nan


def summarize_thesis_group(values: list[dict]) -> dict[str, str]:
    false_rejections = sum(value["false_rejection"] for value in values)
    total = len(values)
    return {
        "frr": f"{percentage(false_rejections, total):.1f}% ({false_rejections}/{total})",
        "mean_trust": fmt(statistics.mean(value["mean_trust"] for value in values)),
        "face_detection_rate": f"{statistics.mean(value['face_detection_rate'] for value in values) * 100:.1f}%",
    }


def print_thesis_profile_count_table(metrics: dict[str, dict], annotations: dict[str, dict]):
    false_rejected_profiles = Counter()

    for video_id, metric in metrics.items():
        if video_id not in annotations or not metric["false_rejection"]:
            continue
        false_rejected_profiles[frozenset(hard_conditions(annotations[video_id]))] += 1

    print("\nDifficult-condition count table")
    print("FL = false lockouts.")
    print(f"{'Difficult conditions':<22}{'FL':>4}  Condition profiles")
    print("-" * 82)
    for count, profiles in THESIS_PROFILE_ROWS:
        profile_counts = [
            (thesis_condition_profile_label(profile), false_rejected_profiles[frozenset(profile)])
            for profile in profiles
        ]
        total = sum(profile_count for _, profile_count in profile_counts)
        first_label, first_count = profile_counts[0]
        print(f"{count:<22}{total:>4}  {first_label} ({first_count})")
        for label, profile_count in profile_counts[1:]:
            print(f"{'':<22}{'':>4}  {label} ({profile_count})")


def print_thesis_factor_tables(metrics: dict[str, dict], annotations: dict[str, dict]):
    print("\nGrouped metric tables for thesis text")
    print("FRR counts are false lockouts/genuine videos in the group.")
    for title, group_key, rows in THESIS_FACTOR_TABLES:
        groups = defaultdict(list)
        for video_id, metric in metrics.items():
            if video_id not in annotations:
                continue
            groups[group_values(annotations[video_id])[group_key]].append(metric)

        print(f"\n{title}")
        print(f"{'Group':<28}{'FRR':>18}{'GT':>8}{'Face det.':>12}")
        print("-" * 66)
        for label, group_value in rows:
            summary = summarize_thesis_group(groups[group_value])
            print(
                f"{label:<28}"
                f"{summary['frr']:>18}"
                f"{summary['mean_trust']:>8}"
                f"{summary['face_detection_rate']:>12}"
            )


def print_thesis_table_data(metrics: dict[str, dict], annotations: dict[str, dict]):
    print_section("THESIS TABLE DATA")
    print_thesis_profile_count_table(metrics, annotations)
    print_thesis_factor_tables(metrics, annotations)


def print_condition_profile_summary(metrics: dict[str, dict], annotations: dict[str, dict]):
    print_section("THESIS CONDITION PROFILE SUMMARY")
    print("Profiles use the three condition dimensions from the thesis. They show whether a condition occurs alone or together with other conditions.")

    profile_groups = defaultdict(list)
    single_condition_groups = defaultdict(list)
    baseline = []

    for video_id, metric in metrics.items():
        if video_id not in annotations:
            continue

        conditions = hard_conditions(annotations[video_id])
        profile_groups[condition_profile_label(conditions)].append(metric)

        if not conditions:
            baseline.append(metric)
        elif len(conditions) == 1:
            single_condition_groups[conditions[0]].append(metric)

    print("\nCondition profiles")
    print_metric_table_header("Profile", 34)
    profile_order = [
        "none",
        "lighting",
        "visibility",
        "movement",
        "lighting + visibility",
        "lighting + movement",
        "visibility + movement",
        "lighting + visibility + movement",
    ]
    for profile in profile_order:
        if profile_groups.get(profile):
            print_metric_table_row(profile, profile_groups[profile], 34)

    print("\nSingle-condition contrasts")
    print("Each row compares recordings where exactly one challenging condition is present against recordings with none of the three conditions.")
    print_metric_table_header("Condition alone", 34)
    if baseline:
        print_metric_table_row("none", baseline, 34)
    for condition in HARD_CONDITION_ORDER:
        values = single_condition_groups.get(condition, [])
        if values:
            print_metric_table_row(condition, values, 34)


def print_false_rejections(metrics: dict[str, dict], annotations: dict[str, dict]):
    print_section("FALSE-REJECTED VIDEOS")
    rejected = [
        (video_id, metric)
        for video_id, metric in metrics.items()
        if metric["false_rejection"] and video_id in annotations
    ]
    print(
        f"{'Video':<34}"
        f"{'Lock s':>8}  "
        f"{'GT mean':>8}  "
        f"{'Min GT':>8}  "
        f"{'Face det.':>9}  "
        f"{'Lighting':<8}  "
        f"{'Visibility':<10}  "
        f"{'Movement':<8}"
    )
    print("-" * 96)
    for video_id, metric in sorted(rejected, key=lambda item: item[1]["first_lock_time"]):
        groups = group_values(annotations[video_id])
        print(
            f"{video_id:<34}"
            f"{fmt(metric['first_lock_time']):>8}  "
            f"{fmt(metric['mean_trust']):>8}  "
            f"{fmt(metric['min_trust']):>8}  "
            f"{fmt(metric['face_detection_rate']):>9}  "
            f"{groups['lighting_simple']:<8}  "
            f"{groups['visibility_simple']:<10}  "
            f"{groups['movement_simple']:<8}"
        )


def print_hard_condition_summary(metrics: dict[str, dict], annotations: dict[str, dict]):
    print_section("HARD-CONDITION COUNT SUMMARY")
    print("A video may contain several hard conditions. This section counts them per genuine video.")
    print()
    for condition, definition in HARD_CONDITION_DEFINITIONS.items():
        print(f"{condition:<18} {definition}")

    rows = []
    condition_counter = Counter()
    false_rejection_condition_counter = Counter()
    count_groups = defaultdict(list)
    total_genuine_videos = 0
    total_false_rejections = 0

    for video_id, metric in metrics.items():
        if video_id not in annotations:
            continue
        total_genuine_videos += 1
        conditions = hard_conditions(annotations[video_id])
        condition_counter.update(conditions)
        if metric["false_rejection"]:
            total_false_rejections += 1
            false_rejection_condition_counter.update(conditions)

        bucket = "2+" if len(conditions) >= 2 else str(len(conditions))
        count_groups[bucket].append(metric)
        rows.append((video_id, metric, conditions))

    def count_with_percentage(count: int, denominator: int) -> str:
        if denominator == 0:
            return f"{count} (--)"
        return f"{count} ({count / denominator * 100:.1f}%)"

    print("\nHard-condition occurrence across all genuine videos")
    print(f"{'Condition':<18}{'All videos':>18}{'False-rejected':>22}")
    print("-" * 58)
    for condition in HARD_CONDITION_DEFINITIONS:
        print(
            f"{condition:<18}"
            f"{count_with_percentage(condition_counter[condition], total_genuine_videos):>18}"
            f"{count_with_percentage(false_rejection_condition_counter[condition], total_false_rejections):>22}"
        )

    print("\nMetrics by number of hard conditions")
    print_metric_table_header("Hard cond.", 12)
    for bucket in ["0", "1", "2+"]:
        values = count_groups.get(bucket, [])
        if not values:
            continue
        print_metric_table_row(bucket, values, 12)

    print("\nFalse-rejected videos with hard-condition count")
    print(f"{'Video':<34}{'Count':>7}  {'Hard conditions'}")
    print("-" * 90)
    for video_id, metric, conditions in sorted(
        [row for row in rows if row[1]["false_rejection"]],
        key=lambda row: row[1]["first_lock_time"],
    ):
        print(f"{video_id:<34}{len(conditions):>7}  {', '.join(conditions) if conditions else 'none'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Also print raw annotation-field summaries for diagnostic checks.",
    )
    args = parser.parse_args()

    annotations = load_annotations(args.annotations)
    rows_by_video = load_genuine_rows(args.results)
    metrics = calculate_metrics(rows_by_video)
    missing_annotations = sorted(set(metrics) - set(annotations))

    group_keys = THESIS_GROUP_KEYS + (RAW_GROUP_KEYS if args.include_raw else [])

    print_section("JOIN SUMMARY")
    print(f"Results file:              {args.results}")
    print(f"Annotations loaded:       {len(annotations)}")
    print(f"Genuine videos in results:{len(metrics):>8}")
    print(f"Missing annotations:      {len(missing_annotations):>8}")

    if args.include_raw:
        print_distribution(metrics, annotations, group_keys)
    print_thesis_table_data(metrics, annotations)
    print_condition_profile_summary(metrics, annotations)
    if args.include_raw:
        print_section("GROUPED METRIC SUMMARY")
        print("Metrics: FRR = false rejection rate; GT = trust score; GKT = first lockout time for rejected genuine sessions.")
        print("Rates <0.4/<0.5 show the mean fraction of frames with trust below that threshold.")
        for key in group_keys:
            print_group_summary(metrics, annotations, key)
    print_hard_condition_summary(metrics, annotations)
    print_false_rejections(metrics, annotations)


if __name__ == "__main__":
    main()
