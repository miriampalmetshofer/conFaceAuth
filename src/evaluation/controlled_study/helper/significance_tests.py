"""Statistical significance testing for the controlled study.

Compares approach behavior across scenarios (easy, angle, lighting) and devices
(desktop, mobile) using non-parametric paired tests.

Metrics:
    - Genuine trust score: mean trust during genuine segments per participant
    - ULT (Unauthorized User Lockout Time): mean lockout time per participant

Tests:
    - Friedman test for scenario comparison
    - Pairwise Wilcoxon signed-rank with Bonferroni correction
    - Wilcoxon signed-rank per scenario for device comparison
    - Wilcoxon signed-rank collapsed across scenarios for device comparison

See STATISTICS.md for full rationale.
"""
from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy import stats

from face_auth.domain import Device, Scenario
from face_auth.paths import PROJECT_ROOT
from evaluation.shared.data_loader import load_run
from evaluation.shared.metrics import first_lockout_seconds, genuine_stays_authenticated
from evaluation.shared.models import Run, Session
from evaluation.shared.reporting import print_section

BONFERRONI_ALPHA = 0.05
RUN_DIR = PROJECT_ROOT / "data/controlled_study/results/run_2026-08-18_15-22-05_f927"


def get_configured_participants(run: Run) -> list[str]:
    """Return participants from the run config, filtered to those present in the data."""
    observed = {session.participant for session in run.sessions}
    configured = [
        participant.get("name")
        for participant in run.config.raw.get("participants", [])
        if participant.get("name")
    ]

    if configured:
        return [name for name in configured if name in observed]
    return sorted(observed)


def _sessions_by_cell(
    sessions: list[Session],
) -> dict[tuple[str, Scenario, Device], list[Session]]:
    """Group sessions once by (participant, scenario, device) in a single scan."""
    cells: dict[tuple[str, Scenario, Device], list[Session]] = defaultdict(list)
    for s in sessions:
        cells[(s.participant, s.genuine_identity.scenario, s.device)].append(s)
    return cells


def compute_genuine_trust_per_participant(sessions, participants, scenarios, devices):
    """Return mean genuine trust per (participant, scenario, device)."""
    cells = _sessions_by_cell(sessions)
    result = {}
    for participant in participants:
        for scenario in scenarios:
            for device in devices:
                matching = cells.get((participant, scenario, device), [])
                scores = [
                    sample.trust_score
                    for session in matching
                    for sample in session.genuine.samples
                ]
                if scores:
                    result[(participant, scenario, device)] = np.mean(scores)
    return result


def compute_ult_per_participant(sessions, participants, scenarios, devices):
    """Return mean unauthorized-user lockout time (s) per (participant, scenario, device)."""
    cells = _sessions_by_cell(sessions)
    result = {}
    for participant in participants:
        for scenario in scenarios:
            for device in devices:
                matching = cells.get((participant, scenario, device), [])
                lockout_times = [
                    ult for session in matching
                    if genuine_stays_authenticated(session)
                    and (ult := first_lockout_seconds(session.imposter)) is not None
                ]
                if lockout_times:
                    result[(participant, scenario, device)] = np.mean(lockout_times)
    return result


def run_friedman_scenario(metric_values, participants, scenarios, device):
    """Run a Friedman test comparing a metric across scenarios for one device."""
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
    """Run pairwise Wilcoxon signed-rank tests with Bonferroni correction."""
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
    """Run a Wilcoxon signed-rank test comparing two devices per scenario."""
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


def run_wilcoxon_device_collapsed(metric_values, participants, scenarios, devices):
    """Run a Wilcoxon signed-rank test comparing two devices, collapsed across scenarios.

    Each participant's value is the mean across all scenarios per device. Tests the
    main effect of device independent of scenario, with no Bonferroni correction.
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


def print_friedman_result(result, metric_name, scenarios, device):
    print(f"\n  Friedman test — {metric_name} ({device.value})")
    print(f"  n = {result['n']} participants")
    for sc in scenarios:
        vals = result['groups'][sc]
        print(f"    {sc.value:<10} mean={np.mean(vals):.4f}  stdev={np.std(vals):.4f}  "
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
        print(f"    {sc_a.value} vs {sc_b.value}:  W = {r['statistic']:.1f},  p = {r['p_value']:.6f}  →  {sig}")


def print_device_results(device_results, metric_name, scenarios, devices):
    n_pairs = len(scenarios)
    alpha_corrected = BONFERRONI_ALPHA / n_pairs
    print(f"\n  Wilcoxon signed-rank — {metric_name} (desktop vs mobile)")
    print(f"  Bonferroni α = {BONFERRONI_ALPHA}/{n_pairs} = {alpha_corrected:.4f}")
    for sc in scenarios:
        r = device_results.get(sc)
        if r is None:
            print(f"    {sc.value:<10} insufficient data")
            continue
        sig = "significant ✓" if r['significant'] else "n.s."
        means_str = "  ".join(f"{dev.value}={r['means'][dev]:.4f}" for dev in devices)
        print(f"    {sc.value:<10} {means_str}  |  W = {r['statistic']:.1f},  p = {r['p_value']:.6f}  →  {sig}")


def run_significance_tests(run: Run, scenarios: list[Scenario], devices: list[Device]) -> None:
    """Print controlled-study significance tests for a loaded run."""
    sessions = run.sessions
    participants = get_configured_participants(run)
    available_devices = [device for device in devices if any(s.device == device for s in sessions)]

    print(f"Participants ({len(participants)}): {participants}")
    print(f"Scenarios: {[sc.value for sc in scenarios]}")
    print(f"Devices: {[d.value for d in available_devices]}")

    if not scenarios:
        print("\nNo scenarios found — skipping significance tests.")
        return
    if not available_devices:
        print("\nNo configured devices found in the data — skipping significance tests.")
        return

    genuine_trust = compute_genuine_trust_per_participant(sessions, participants, scenarios, available_devices)
    ult = compute_ult_per_participant(sessions, participants, scenarios, available_devices)

    print_section("SCENARIO COMPARISON (Friedman + post-hoc Wilcoxon)", width=60)

    for device in available_devices:
        for metric_name, metric_values in [('Genuine Trust Score', genuine_trust), ('ULT (s)', ult)]:
            result = run_friedman_scenario(metric_values, participants, scenarios, device)
            if result is None:
                print(f"\n  Skipping {metric_name} / {device.value}: insufficient data")
                continue

            print_friedman_result(result, metric_name, scenarios, device)

            if result['p_value'] < BONFERRONI_ALPHA:
                posthoc = run_pairwise_wilcoxon_posthoc(result, scenarios)
                print_posthoc_results(posthoc)
            else:
                print("  Friedman not significant — skipping post-hoc tests")

    if len(available_devices) == 2:
        print_section("DEVICE COMPARISON — Main effect (Wilcoxon, collapsed across scenarios)", width=60)
        print(f"  Each participant's value = mean across {' + '.join(sc.value for sc in scenarios)}.")
        print("  Single test per metric — no Bonferroni correction needed.")
        for metric_name, metric_values in [('Genuine Trust Score', genuine_trust), ('ULT (s)', ult)]:
            result = run_wilcoxon_device_collapsed(metric_values, participants, scenarios, available_devices)
            if result is None:
                print(f"\n  {metric_name}: insufficient data")
                continue
            sig = "significant ✓" if result['significant'] else "n.s."
            print(f"\n  Wilcoxon signed-rank — {metric_name} (desktop vs mobile, collapsed)")
            print(f"  n = {result['n']} participants")
            for dev in available_devices:
                vals = result['values'][dev]
                print(f"    {dev.value:<10} mean={result['means'][dev]:.4f}  "
                      f"values={[round(v, 3) for v in sorted(vals)]}")
            print(f"  W = {result['statistic']:.1f},  p = {result['p_value']:.6f}  →  {sig}")

        print_section("DEVICE COMPARISON — Per scenario (Wilcoxon, exploratory)", width=60)
        print("  Tests whether device effect is consistent within each scenario.")
        print(f"  Bonferroni α = {BONFERRONI_ALPHA}/{len(scenarios)} = {BONFERRONI_ALPHA/len(scenarios):.4f}")
        for metric_name, metric_values in [('Genuine Trust Score', genuine_trust), ('ULT (s)', ult)]:
            device_results = run_wilcoxon_device(metric_values, participants, scenarios, available_devices)
            print_device_results(device_results, metric_name, scenarios, available_devices)
    else:
        print_section("DEVICE COMPARISON", width=60)
        print(f"\n  Only one device found ({available_devices[0].value}) — skipping device comparison.")


def main():
    run = load_run(RUN_DIR)
    print(f"Loaded {len(run.sessions)} sessions from {RUN_DIR.name}")
    scenarios = sorted(
        {s.genuine_identity.scenario for s in run.sessions if s.genuine_identity.scenario},
        key=lambda sc: sc.value,
    )
    run_significance_tests(run, scenarios, run.config.devices)


if __name__ == '__main__':
    main()
