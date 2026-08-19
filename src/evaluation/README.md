# Evaluation

Turns a completed batch run into numbers and figures. It reads a
`run_<timestamp>_<suffix>/` directory produced by the pipeline (see the
[top-level README](../../README.md)), computes authentication metrics, and
renders interactive timelines and summary plots.

## What it measures

Every composed video is one *session*: a genuine segment, a black gap, then an
imposter segment. The metrics are computed over these sessions.

| Metric | Meaning |
|---|---|
| **FRR** — False Reject Rate | Fraction of genuine sessions where the real user was wrongly locked out |
| **FAR** — False Accept Rate | Fraction of imposter sessions where the imposter was never locked out |
| **Mean genuine trust** | Average trust score during genuine segments |
| **ULT** — imposter lockout time | How long (seconds) until an imposter drops below the threshold |
| **GKT** — genuine lockout time | How long a genuine user survives before a (wrong) lockout, when one happens |

Times are reported as mean and P90.

## Layout

```
evaluation/
├── shared/                     # Study-agnostic core
│   ├── data_loader.py          # run dir → typed Run / Session / Segment model
│   ├── models.py               # the evaluation domain types
│   ├── metrics.py              # FRR, FAR, trust, lockout times (+ by device/scenario)
│   ├── grouping.py             # group sessions by an ordered key list
│   ├── reporting.py            # console tables
│   ├── font_config.py          # matplotlib fonts for thesis figures
│   └── plots/                  # timeline.py, summary.py, segments.py, save.py
│
├── controlled_study/
│   ├── evaluate.py             # orchestrates the controlled-study evaluation
│   └── helper/significance_tests.py   # Friedman + Wilcoxon across scenarios/devices
│
├── in_the_wild/
│   ├── evaluate.py             # orchestrates the in-the-wild evaluation
│   └── helper/
│       ├── annotation_validator.py    # validate participant self-labels
│       └── analyze_annotation_trends.py
│
├── compare.py                  # diff two runs side by side
└── thesis_image_scripts/       # standalone scripts for specific thesis figures
```

The two `evaluate.py` scripts are thin orchestrators. All real work lives in
`shared/` and is imported by both, so the studies differ only in how they slice
the sessions (controlled study also breaks down by scenario and runs significance
tests; in-the-wild instead validates annotations).

## Running it

Each script picks its input from a `RUN_DIR` constant near the top — **edit it to
point at the run you want**, then run the script from anywhere:

```bash
python src/evaluation/controlled_study/evaluate.py
python src/evaluation/in_the_wild/evaluate.py
python src/evaluation/compare.py            # edit VARIANT1_RUN / VARIANT2_RUN
```

### Controlled study

Breaks metrics down by device, by scenario, and by scenario × device, then runs
significance tests (see [`controlled_study/STATISTICS.md`](controlled_study/STATISTICS.md)).
Generates:

- `trust_timeline_all_videos.html` — all sessions on one interactive timeline
- `trust_timeline_by_device.html` — mobile vs. desktop
- `trust_timeline_by_scenario.html` — one timeline per scenario
- `trust_timeline_scenarios_aggregated.html` — scenarios overlaid
- `summary.png` — trust distribution, error rates, state breakdown

### In-the-wild study

Mobile only, no scenarios. First validates the per-participant annotation JSON
files under `data/in_the_wild/mobile/{name}/annotations/` (schema in
`in_the_wild/annotation_schema.json`) and warns on invalid files. Generates:

- `trust_timeline_all_videos.html`
- `trust_timeline_by_device.html`
- `summary.png`

## Input expected

A run directory written by the pipeline:

```
data/{study}/results/run_{timestamp}_{suffix}/
├── config.json                 # threshold, hz, segment lengths — read by the evaluator
└── {participant}/{device}/
    └── {genuine}_vs_{imposter}.csv     # one session, one row per sampled frame
```

`data_loader.load_run()` reads `config.json` plus every CSV, splits each CSV into
genuine / black / imposter segments by the `source_type` column, and resolves the
genuine and imposter identities from the video stems. Everything downstream works
on the resulting typed `Run` object.
