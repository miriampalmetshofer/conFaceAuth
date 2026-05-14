"""Statistical significance testing for controlled study.

Compares system performance across scenarios (easy, angle, lighting)
and devices (desktop, mobile) using non-parametric paired tests.

Metrics:
    - Genuine trust score: mean trust score during genuine segments per participant
    - ILT (Imposter Lockout Time): median lockout time per participant

Tests:
    - Friedman test for scenario comparison (3 conditions, repeated measures)
    - Pairwise Wilcoxon signed-rank with Bonferroni correction (post-hoc)
    - Wilcoxon signed-rank per scenario for device comparison (exploratory)
    - Wilcoxon signed-rank collapsed across scenarios for device comparison (main effect)

See STATISTICS.md for full rationale.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from scipy import stats
from itertools import combinations

from evaluation.shared.data_loader import load_evaluation_data
from evaluation.shared.models import SegmentType
from evaluation.shared.metrics import (
    group_frames_by_video,
    find_first_imposter_frame_index,
    find_lockout_transition,
    is_eligible_for_imposter_rejection,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
RESULTS_PATH = PROJECT_ROOT / "data/controlled_study/_results_archive/V03/results.csv"

SCENARIOS = ['easy', 'angle', 'lighting']
DEVICES = ['desktop', 'mobile']
BONFERRONI_ALPHA = 0.05


def compute_genuine_trust_per_participant(frames, participants, scenarios, devices):
    """Compute mean genuine trust score per (participant, scenario, device).

    Returns:
        dict mapping (participant, scenario, device) -> mean genuine trust score
    """
    result = {}
    for participant in participants:
        for scenario in scenarios:
            for device in devices:
                subset = [
                    f for f in frames
                    if f.participant == participant
                    and f.scenario == scenario
                    and f.device == device
                    and f.segment_type == SegmentType.GENUINE
                ]
                if subset:
                    result[(participant, scenario, device)] = np.mean([f.trust_score for f in subset])
    return result


def compute_ilt_per_participant(frames, participants, scenarios, devices, fps):
    """Compute median imposter lockout time per (participant, scenario, device).

    For each participant, takes the median lockout time across all impostor
    videos in that condition. Videos where the device locked before the
    impostor segment, and videos where the impostor is never locked out, are
    excluded from the median.

    Returns:
        dict mapping (participant, scenario, device) -> median ILT in seconds
    """
    result = {}
    for participant in participants:
        for scenario in scenarios:
            for device in devices:
                subset = [
                    f for f in frames
                    if f.participant == participant
                    and f.scenario == scenario
                    and f.device == device
                ]
                if not subset:
                    continue

                videos = group_frames_by_video(subset)
                lockout_times = []
                for vpath, vframes in videos.items():
                    vframes.sort(key=lambda f: f.frame)
                    first_imp_idx = find_first_imposter_frame_index(vframes)
                    if first_imp_idx is None:
                        continue
                    if not is_eligible_for_imposter_rejection(vframes, first_imp_idx):
                        continue
                    lt = find_lockout_transition(vframes, first_imp_idx, fps)
                    if lt is not None:
                        lockout_times.append(lt)

                if lockout_times:
                    result[(participant, scenario, device)] = np.median(lockout_times)
    return result


def run_friedman_scenario(metric_values, participants, scenarios, device):
    """Run Friedman test comparing metric across scenarios for one device.

    Args:
        metric_values: dict (participant, scenario, device) -> value
        participants: list of participant ids
        scenarios: list of scenario names (3 conditions)
        device: which device to test

    Returns:
        dict with statistic, p_value, and per-scenario values
    """
    groups = []
    valid_participants = []

    for p in participants:
        row = [metric_values.get((p, sc, device)) for sc in scenarios]
        if all(v is not None for v in row):
            groups.append(row)
            valid_participants.append(p)

    if len(groups) < 3:
        return None

    groups_by_scenario = list(zip(*groups))
    stat, p_value = stats.friedmanchisquare(*groups_by_scenario)

    return {
        'statistic': stat,
        'p_value': p_value,
        'n': len(valid_participants),
        'participants': valid_participants,
        'groups': {sc: list(groups_by_scenario[i]) for i, sc in enumerate(scenarios)},
    }


def run_pairwise_wilcoxon_posthoc(friedman_result, scenarios):
    """Run pairwise Wilcoxon signed-rank tests with Bonferroni correction.

    Args:
        friedman_result: output from run_friedman_scenario
        scenarios: list of scenario names

    Returns:
        list of dicts with pair, statistic, p_value, p_corrected, significant
    """
    pairs = list(combinations(scenarios, 2))
    alpha_corrected = BONFERRONI_ALPHA / len(pairs)
    results = []

    for sc_a, sc_b in pairs:
        a = friedman_result['groups'][sc_a]
        b = friedman_result['groups'][sc_b]
        stat, p_value = stats.wilcoxon(a, b)
        results.append({
            'pair': (sc_a, sc_b),
            'statistic': stat,
            'p_value': p_value,
            'p_corrected_alpha': alpha_corrected,
            'significant': p_value < alpha_corrected,
        })

    return results


def run_wilcoxon_device(metric_values, participants, scenarios, devices):
    """Run Wilcoxon signed-rank test comparing two devices per scenario.

    Args:
        metric_values: dict (participant, scenario, device) -> value
        participants: list of participant ids
        scenarios: list of scenario names
        devices: list of exactly two device names

    Returns:
        dict mapping scenario -> result dict
    """
    if len(devices) != 2:
        return None

    device_a, device_b = devices
    results = {}

    for scenario in scenarios:
        pairs_a, pairs_b = [], []
        for p in participants:
            a = metric_values.get((p, scenario, device_a))
            b = metric_values.get((p, scenario, device_b))
            if a is not None and b is not None:
                pairs_a.append(a)
                pairs_b.append(b)

        if len(pairs_a) < 3:
            results[scenario] = None
            continue

        stat, p_value = stats.wilcoxon(pairs_a, pairs_b)
        results[scenario] = {
            'statistic': stat,
            'p_value': p_value,
            'n': len(pairs_a),
            'significant': p_value < BONFERRONI_ALPHA / len(scenarios),
            'means': {device_a: np.mean(pairs_a), device_b: np.mean(pairs_b)},
        }

    return results


def print_section(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_friedman_result(result, metric_name, scenarios, device):
    print(f"\n  Friedman test — {metric_name} ({device})")
    print(f"  n = {result['n']} participants")
    for sc in scenarios:
        vals = result['groups'][sc]
        print(f"    {sc:<10} mean={np.mean(vals):.4f}  stdev={np.std(vals):.4f}  "
              f"values={[round(v, 3) for v in sorted(vals)]}")
    sig = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else "n.s."
    print(f"  χ²({len(scenarios)-1}) = {result['statistic']:.3f},  p = {result['p_value']:.6f}  {sig}")


def print_posthoc_results(posthoc_results):
    n_pairs = len(posthoc_results)
    alpha_corrected = BONFERRONI_ALPHA / n_pairs
    print(f"\n  Post-hoc Wilcoxon signed-rank (Bonferroni α = {BONFERRONI_ALPHA}/{n_pairs} = {alpha_corrected:.4f})")
    for r in posthoc_results:
        sc_a, sc_b = r['pair']
        sig = "significant ✓" if r['significant'] else "n.s."
        print(f"    {sc_a} vs {sc_b}:  W = {r['statistic']:.1f},  p = {r['p_value']:.6f}  →  {sig}")


def run_wilcoxon_device_collapsed(metric_values, participants, scenarios, devices):
    """Run Wilcoxon signed-rank test comparing two devices, collapsed across all scenarios.

    For each participant, computes the mean metric value across all scenarios
    for each device, then tests whether the two devices differ overall.
    This tests the main effect of device independent of scenario.
    No Bonferroni correction is applied since this is a single test.

    Args:
        metric_values: dict (participant, scenario, device) -> value
        participants: list of participant ids
        scenarios: list of scenario names to collapse over
        devices: list of exactly two device names

    Returns:
        dict with statistic, p_value, n, significant, means, values_per_device
    """
    if len(devices) != 2:
        return None

    device_a, device_b = devices
    collapsed_a, collapsed_b = [], []

    for p in participants:
        vals_a = [metric_values.get((p, sc, device_a)) for sc in scenarios]
        vals_b = [metric_values.get((p, sc, device_b)) for sc in scenarios]

        if all(v is not None for v in vals_a) and all(v is not None for v in vals_b):
            collapsed_a.append(np.mean(vals_a))
            collapsed_b.append(np.mean(vals_b))

    if len(collapsed_a) < 3:
        return None

    stat, p_value = stats.wilcoxon(collapsed_a, collapsed_b)
    return {
        'statistic': stat,
        'p_value': p_value,
        'n': len(collapsed_a),
        'significant': p_value < BONFERRONI_ALPHA,
        'means': {device_a: np.mean(collapsed_a), device_b: np.mean(collapsed_b)},
        'values': {device_a: collapsed_a, device_b: collapsed_b},
    }


def print_device_results(device_results, metric_name, scenarios, devices):
    n_pairs = len(scenarios)
    alpha_corrected = BONFERRONI_ALPHA / n_pairs
    print(f"\n  Wilcoxon signed-rank — {metric_name} (desktop vs mobile)")
    print(f"  Bonferroni α = {BONFERRONI_ALPHA}/{n_pairs} = {alpha_corrected:.4f}")
    for sc in scenarios:
        r = device_results.get(sc)
        if r is None:
            print(f"    {sc:<10} insufficient data")
            continue
        sig = "significant ✓" if r['significant'] else "n.s."
        means_str = "  ".join(f"{dev}={r['means'][dev]:.4f}" for dev in devices)
        print(f"    {sc:<10} {means_str}  |  W = {r['statistic']:.1f},  p = {r['p_value']:.6f}  →  {sig}")


def main():
    print(f"Loading data from {RESULTS_PATH}")
    data = load_evaluation_data(RESULTS_PATH, parse_scenario=True)
    print(f"Loaded {len(data.frames)} frames from {len(data.videos)} videos")

    participants = sorted(set(f.participant for f in data.frames))
    available_devices = sorted(set(f.device for f in data.frames))
    print(f"Participants ({len(participants)}): {participants}")
    print(f"Devices: {available_devices}")

    # --- Compute per-participant metrics ---
    genuine_trust = compute_genuine_trust_per_participant(
        data.frames, participants, SCENARIOS, available_devices
    )
    ilt = compute_ilt_per_participant(
        data.frames, participants, SCENARIOS, available_devices, data.fps
    )

    # --- Scenario comparison ---
    print_section("SCENARIO COMPARISON (Friedman + post-hoc Wilcoxon)")

    for device in available_devices:
        for metric_name, metric_values in [('Genuine Trust Score', genuine_trust), ('ILT (s)', ilt)]:
            result = run_friedman_scenario(metric_values, participants, SCENARIOS, device)
            if result is None:
                print(f"\n  Skipping {metric_name} / {device}: insufficient data")
                continue

            print_friedman_result(result, metric_name, SCENARIOS, device)

            if result['p_value'] < BONFERRONI_ALPHA:
                posthoc = run_pairwise_wilcoxon_posthoc(result, SCENARIOS)
                print_posthoc_results(posthoc)
            else:
                print("  Friedman not significant — skipping post-hoc tests")

    # --- Device comparison ---
    if len(available_devices) == 2:
        print_section("DEVICE COMPARISON — Main effect (Wilcoxon, collapsed across scenarios)")
        print("  Each participant's value = mean across easy + angle + lighting.")
        print("  Single test per metric — no Bonferroni correction needed.")
        for metric_name, metric_values in [('Genuine Trust Score', genuine_trust), ('ILT (s)', ilt)]:
            result = run_wilcoxon_device_collapsed(
                metric_values, participants, SCENARIOS, available_devices
            )
            if result is None:
                print(f"\n  {metric_name}: insufficient data")
                continue
            sig = "significant ✓" if result['significant'] else "n.s."
            dev_a, dev_b = available_devices
            print(f"\n  Wilcoxon signed-rank — {metric_name} (desktop vs mobile, collapsed)")
            print(f"  n = {result['n']} participants")
            for dev in available_devices:
                vals = result['values'][dev]
                print(f"    {dev:<10} mean={result['means'][dev]:.4f}  "
                      f"values={[round(v, 3) for v in sorted(vals)]}")
            print(f"  W = {result['statistic']:.1f},  p = {result['p_value']:.6f}  →  {sig}")

        print_section("DEVICE COMPARISON — Per scenario (Wilcoxon, exploratory)")
        print("  Tests whether device effect is consistent within each scenario.")
        print(f"  Bonferroni α = {BONFERRONI_ALPHA}/{len(SCENARIOS)} = {BONFERRONI_ALPHA/len(SCENARIOS):.4f}")
        for metric_name, metric_values in [('Genuine Trust Score', genuine_trust), ('ILT (s)', ilt)]:
            device_results = run_wilcoxon_device(
                metric_values, participants, SCENARIOS, available_devices
            )
            print_device_results(device_results, metric_name, SCENARIOS, available_devices)
    else:
        print_section("DEVICE COMPARISON")
        print(f"\n  Only one device found ({available_devices[0]}) — skipping device comparison.")
        print("  Re-run once mobile data is included in results.csv.")


if __name__ == '__main__':
    main()
