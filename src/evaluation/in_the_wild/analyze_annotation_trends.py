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
DEFAULT_RESULTS = PROJECT_ROOT / "data/in_the_wild/_results_archive/V06/results.csv"
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "data/in_the_wild/mobile"

GROUP_DEFINITIONS = {
    "illumination": "Raw annotation: well_lit, shadows, poorly_lit.",
    "lighting_simple": "easy = well_lit without light_change; hard = shadows, poorly_lit, or light_change=true.",
    "face_visibility": "Raw annotation: visible or turned_away.",
    "occlusion_simple": (
        "none = no occlusion or glasses only; "
        "visibility_reducing_occlusion = any hand, eating, hat, other, or mixed occlusion."
    ),
    "visibility_simple": (
        "easy = visible and no occlusion except glasses; "
        "hard = turned_away or occlusion by hand, eating, hat, or other."
    ),
    "angle_simple": "straight = straight; non_straight = downward or upward.",
    "movement_simple": (
        "easy = stable/slightly moving device with one static user posture; "
        "hard = significant device movement, walking, or multiple selected user movements."
    ),
    "environment": "Raw annotation: indoor or outdoor.",
}

DERIVED_CLASS_MAPPING = {
    "lighting_simple": [
        ("easy", "illumination = well_lit AND light_change is not true"),
        ("hard", "illumination = shadows OR illumination = poorly_lit OR light_change = true"),
    ],
    "occlusion_simple": [
        ("none", "occlusions = [none] OR occlusions = [glasses]"),
        (
            "visibility_reducing_occlusion",
            "occlusions contains hand, eating, hat, or other; mixed occlusion lists also go here",
        ),
    ],
    "visibility_simple": [
        (
            "easy",
            "face_visibility = visible AND occlusions are none or glasses only",
        ),
        (
            "hard",
            "face_visibility = turned_away OR occlusions contain hand, eating, hat, or other",
        ),
    ],
    "angle_simple": [
        ("straight", "angle = straight"),
        ("non_straight", "angle = downward OR angle = upward"),
    ],
    "movement_simple": [
        (
            "easy",
            "device_movement = stable/slight_movement AND movement is exactly one of sitting, standing, or lying",
        ),
        (
            "hard",
            "device_movement = significant_movement OR movement contains walking OR more than one movement is selected",
        ),
    ],
}

HARD_CONDITION_ORDER = ["visibility", "movement", "lighting"]

HARD_CONDITION_DEFINITIONS = {
    "visibility": "face_visibility is turned_away or occlusion by hand/eating/hat/other",
    "movement": "device_movement is significant_movement, or user movement includes walking/multiple movement values",
    "lighting": "illumination is shadows/poorly_lit or light_change=true",
}


def order_hard_conditions(conditions: list[str]) -> list[str]:
    order = {condition: index for index, condition in enumerate(HARD_CONDITION_ORDER)}
    return sorted(conditions, key=lambda condition: order[condition])


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
    visibility_reducing_occlusions = {"hand", "eating", "hat", "other"}
    movement = ann.get("movement", [])
    hard_movement = (
        ann.get("device_movement") == "significant_movement"
        or "walking" in movement
        or len(movement) > 1
    )

    return {
        "illumination": ann.get("illumination"),
        "lighting_simple": (
            "hard"
            if ann.get("illumination") in {"shadows", "poorly_lit"} or ann.get("light_change") is True
            else "easy"
        ),
        "face_visibility": ann.get("face_visibility"),
        "occlusion_simple": "none"
        if occlusions in ({"none"}, {"glasses"})
        else "visibility_reducing_occlusion",
        "visibility_simple": (
            "hard"
            if ann.get("face_visibility") == "turned_away" or occlusions & visibility_reducing_occlusions
            else "easy"
        ),
        "angle_simple": "straight" if ann.get("angle") == "straight" else "non_straight",
        "movement_simple": "hard" if hard_movement else "easy",
        "environment": ann.get("environment"),
    }


def hard_conditions(annotation: dict) -> list[str]:
    ann = annotation["annotations"]
    occlusions = set(ann.get("occlusions", []))
    movement = ann.get("movement", [])
    conditions = []

    if ann.get("face_visibility") == "turned_away" or occlusions & {"hand", "eating", "hat", "other"}:
        conditions.append("visibility")

    if ann.get("device_movement") == "significant_movement" or "walking" in movement or len(movement) > 1:
        conditions.append("movement")

    if ann.get("illumination") in {"shadows", "poorly_lit"} or ann.get("light_change") is True:
        conditions.append("lighting")

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


def print_grouping_rules(group_keys: list[str]):
    print_section("GROUPING RULES")
    for key in group_keys:
        print(f"{key:<24} {GROUP_DEFINITIONS[key]}")


def print_derived_class_mapping():
    print_section("EXACT DERIVED ANNOTATION CLASS MAPPING")
    print("Raw annotation fields that are not listed here are used directly without grouping.")
    for grouped_field, mappings in DERIVED_CLASS_MAPPING.items():
        print(f"\n{grouped_field}")
        print("-" * len(grouped_field))
        for class_name, rule in mappings:
            print(f"  {class_name:<32} {rule}")

    print("\nHard-condition count")
    print("--------------------")
    for condition, rule in HARD_CONDITION_DEFINITIONS.items():
        print(f"  {condition:<32} {rule}")


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
    print(f"Rule: {GROUP_DEFINITIONS[group_key]}")
    print()
    print(
        f"{'Group':<34}"
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
    for group, values in sorted(groups.items(), key=lambda item: str(item[0])):
        false_rejections = sum(value["false_rejection"] for value in values)
        lock_times = [value["first_lock_time"] for value in values if value["first_lock_time"] is not None]
        frr = f"{false_rejections}/{len(values)} ({false_rejections / len(values) * 100:.1f}%)"
        print(
            f"{str(group):<34}"
            f"{len(values):>4}  "
            f"{frr:>13}  "
            f"{fmt(statistics.mean(value['mean_trust'] for value in values)):>8}  "
            f"{fmt(statistics.median(value['min_trust'] for value in values)):>10}  "
            f"{fmt(statistics.mean(value['mean_similarity'] for value in values)):>8}  "
            f"{fmt(statistics.mean(value['face_detection_rate'] for value in values)):>9}  "
            f"{fmt(statistics.mean(value['below_04_rate'] for value in values)):>7}  "
            f"{fmt(statistics.mean(value['below_05_rate'] for value in values)):>7}  "
            f"{fmt(statistics.mean(lock_times) if lock_times else None):>8}"
        )


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
    print(
        f"{'Hard cond.':<12}"
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
    for bucket in ["0", "1", "2+"]:
        values = count_groups.get(bucket, [])
        if not values:
            continue
        false_rejections = sum(value["false_rejection"] for value in values)
        lock_times = [value["first_lock_time"] for value in values if value["first_lock_time"] is not None]
        frr = f"{false_rejections}/{len(values)} ({false_rejections / len(values) * 100:.1f}%)"
        print(
            f"{bucket:<12}"
            f"{len(values):>4}  "
            f"{frr:>13}  "
            f"{fmt(statistics.mean(value['mean_trust'] for value in values)):>8}  "
            f"{fmt(statistics.median(value['min_trust'] for value in values)):>10}  "
            f"{fmt(statistics.mean(value['mean_similarity'] for value in values)):>8}  "
            f"{fmt(statistics.mean(value['face_detection_rate'] for value in values)):>9}  "
            f"{fmt(statistics.mean(value['below_04_rate'] for value in values)):>7}  "
            f"{fmt(statistics.mean(value['below_05_rate'] for value in values)):>7}  "
            f"{fmt(statistics.mean(lock_times) if lock_times else None):>8}"
        )

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
    args = parser.parse_args()

    annotations = load_annotations(args.annotations)
    rows_by_video = load_genuine_rows(args.results)
    metrics = calculate_metrics(rows_by_video)
    missing_annotations = sorted(set(metrics) - set(annotations))

    group_keys = [
        "illumination",
        "lighting_simple",
        "face_visibility",
        "occlusion_simple",
        "visibility_simple",
        "angle_simple",
        "movement_simple",
        "environment",
    ]

    print_section("JOIN SUMMARY")
    print(f"Results file:              {args.results}")
    print(f"Annotations loaded:       {len(annotations)}")
    print(f"Genuine videos in results:{len(metrics):>8}")
    print(f"Missing annotations:      {len(missing_annotations):>8}")

    print_grouping_rules(group_keys)
    print_derived_class_mapping()
    print_distribution(metrics, annotations, group_keys)
    print_section("GROUPED METRIC SUMMARY")
    print("Metrics: FRR = false rejection rate; GT = trust score; GKT = first lockout time for rejected genuine sessions.")
    print("Rates <0.4/<0.5 show the mean fraction of frames with trust below that threshold.")
    for key in group_keys:
        print_group_summary(metrics, annotations, key)
    print_hard_condition_summary(metrics, annotations)
    print_false_rejections(metrics, annotations)


if __name__ == "__main__":
    main()
