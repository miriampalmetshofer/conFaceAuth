# Statistical Significance Testing — Controlled Study

## What we are testing

The controlled study compares system performance across two factors:

- **Scenarios** (3 levels): `easy`, `angle`, `lighting`
- **Devices** (2 levels): `desktop`, `mobile`

For each factor we test two metrics independently:

- **Genuine trust score** — mean trust score per participant during genuine segments. Measures usability: how confidently the system keeps the real user authenticated.
- **ILT (Imposter Lockout Time)** — median time in seconds until the system locks out the impostor, per participant. Measures security response: how quickly the system reacts.

These two metrics were chosen because they capture both sides of the authentication tradeoff without depending on the binary threshold. FAR, FRR, TAR, TRR and EER are all threshold-crossing counts — shifting the threshold changes all of them simultaneously. Genuine trust and ILT reflect the raw system behavior.

---

## Data structure

12 participants each appear in all conditions (3 scenarios × 2 devices). Per condition, each participant generates ~11 videos (one per pairing with every other participant as impostor). Genuine trust and ILT are aggregated to one value per participant per condition before testing.

This means the same participants contribute observations to every condition — the data is **repeated measures / within-subjects**.

---

## Why these tests

### Scenario comparison — Friedman test

The Friedman test is the non-parametric equivalent of a one-way repeated-measures ANOVA. It is appropriate here because:

1. **Repeated measures**: the same 12 participants appear in all 3 scenarios. Observations are not independent across conditions, which rules out independent-groups tests (Mann-Whitney U, Kruskal-Wallis).
2. **Non-normal distributions**: ILT in particular is right-skewed and contains zeros (immediate lockout). With n=12, normality cannot be reliably verified. Parametric tests (repeated-measures ANOVA) assume normality, which we cannot justify.
3. **3 conditions**: for a single pairwise comparison between 2 conditions, the Wilcoxon signed-rank test would suffice. With 3 conditions, Friedman tests the overall effect first, avoiding inflation of Type I error from running three separate tests without justification.

The Friedman test works by ranking each participant's values across conditions and testing whether the rank sums differ significantly. No distributional assumption is required beyond ordinal measurement.

**Post-hoc tests**: if the Friedman test is significant, pairwise Wilcoxon signed-rank tests identify which specific scenario pairs differ. Bonferroni correction is applied: α = 0.05 / 3 ≈ 0.017.

### Device comparison — Wilcoxon signed-rank test (two approaches)

The Wilcoxon signed-rank test is the non-parametric equivalent of a paired t-test. It is appropriate here because:

1. **Paired observations**: the same participant produces one value on desktop and one on mobile. Pairing controls for between-participant variance and increases statistical power.
2. **Non-normal distributions**: same reasoning as above.
3. **2 conditions**: with only two conditions (desktop vs mobile), no omnibus test is needed — go directly to the pairwise test.

Two variants are run:

**Collapsed across scenarios (primary — main effect of device)**
For each participant, the metric is averaged across all three scenarios before testing. This produces a single mean per participant per device (e.g., `desktop_mean = mean(easy, angle, lighting)`). One Wilcoxon test is run per metric. No Bonferroni correction is needed since there is only one test.

This is the correct approach for answering *"does device type affect overall system performance?"* Collapsing across scenarios treats scenario as a nuisance variable and averages it out, which: (a) aligns with the actual research question, (b) produces more stable per-participant estimates from more data, and (c) avoids the power penalty of a multiple comparison correction.

**Per scenario (exploratory — device × scenario interaction)**
The test is run separately for each scenario. Bonferroni correction is applied: α = 0.05 / 3 ≈ 0.017. This answers the more specific question of whether the device effect differs depending on scenario — effectively testing an interaction. It is included as an exploratory analysis but is not the primary device comparison.

---

## Why other tests were ruled out

| Test | Why not suitable |
|------|-----------------|
| Repeated-measures ANOVA | Parametric — assumes normality. Not justified with n=12 and skewed ILT distribution. |
| Paired t-test | Same: parametric, assumes normality. |
| Kruskal-Wallis | Non-parametric alternative to one-way ANOVA, but treats groups as **independent**. Ignores the within-subjects structure and would inflate Type I error. |
| Mann-Whitney U | Same problem as Kruskal-Wallis — designed for independent groups. |
| Pearson / Spearman correlation | Tests association between two variables, not differences between conditions. |

---

## Notes on interpretation

- **n=12** is a small sample. The Wilcoxon and Friedman tests have limited statistical power for small effects. Large effects (as observed in the scenario comparison) can still be detected reliably.
- The **angle vs lighting ILT** comparison is marginal (p ≈ 0.023, just above the corrected threshold of 0.017 with the current n=8 partial dataset). This may become significant with the full n=12.
- Genuine trust and ILT show **opposite trends** across scenarios: easy has the highest genuine trust but also the longest ILT (worst security response), while lighting has the lowest genuine trust but shortest ILT (best security response). This reflects the inherent security-usability tradeoff and is a finding in itself, independent of significance testing.
- **Never-locked-out videos** (impostor segments where no lockout transition occurred) are excluded from ILT. In the current dataset all impostor videos resulted in a lockout, so no censoring occurs. If this changes with the full dataset, a survival analysis approach (log-rank test) would be more appropriate for ILT.
