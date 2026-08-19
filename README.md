# conFaceAuth — Continuous Face Authentication

Research prototype for **continuous** face authentication. Instead of a single
unlock decision when unlocking the system, the system re-checks the user regularily and computes a
**trust score** that rises while the enrolled user is present and decays when the
face changes or disappears. The user is authenticated while trust stays above a
threshold.

This repository is the evaluation harness for a master thesis. It runs the
authentication algorithm over recorded videos, writes per-frame results, and
computes performance metrics (FRR, FAR, lockout times) for two studies.

## How it works

The core idea is the **temporal-decay authenticator** (`core/authenticator`):

- Each sampled frame is embedded and compared to the enrolled user's embeddings,
  producing a **similarity** in roughly `[0, 1]`.
- The trust score is an exponential moving average over real time. When a face is
  present, trust moves toward the new similarity; when no face is detected, trust
  decays toward zero. Both rates are governed by time constants (`k_weight`,
  `k_decay`) applied to the elapsed milliseconds between samples, so the behaviour
  is independent of frame rate.

Because the score is driven by elapsed time rather than frame counts, the same
authenticator runs unchanged on recorded videos and on a live webcam.

## Two ways to run it

| Entry point | Command | Purpose |
|---|---|---|
| **Batch study** | `face-auth` / `python src/face_auth/apps/main.py` | Runs the algorithm over recorded study videos and writes result CSVs. This is the main path. |
| **Live demo** | `face-auth-live` / `python src/face_auth/apps/live.py` | Real-time webcam demo with an on-screen trust bar. For a quick visual sanity check. |

## The batch pipeline

The batch pipeline does **not** authenticate raw study videos directly. For each
participant it builds *composed videos* on the fly and authenticates those.

A composed video is a virtual, in-memory stream stitched from three segments:

```
[ genuine user video ] → [ short black screen ] → [ imposter video ]
```

The genuine segment should keep the user authenticated; after the black gap the
imposter segment should get **locked out** as trust decays. This single stream is
what lets us measure both false rejects (genuine locked out) and false accepts
(imposter stays in).

Per participant × device, one job:

1. **Enroll** the participant — discover their enrollment video, select frames,
   embed them (`enrollment/`, `core/enrollment/`).
2. **Compose** imposter videos — pair the participant's genuine videos with other
   participants' videos via a matching strategy, then stitch genuine + black +
   imposter (`imposter_creation/`).
3. **Authenticate** each composed video — sample frames at `authentication_hz`
   along the real timeline and run each through the temporal-decay authenticator
   (`authentication/`, `core/authenticator/`).
4. **Write** one CSV per composed video (`io_utils.py`).

Jobs run in parallel across processes (`batch.py`, `run.num_workers`).

## Project layout

```
src/face_auth/
├── apps/                 # Entry points: main.py (batch), live.py (webcam demo)
├── configs/              # Run configs (JSON) + Pydantic schema (models.py)
├── pipeline.py           # Batch orchestrator: builds and runs all jobs
├── batch.py              # Parallel/sequential job executor
├── domain.py             # Shared study vocabulary + value objects
├── io_utils.py           # Run directory + per-video CSV writer
│
├── enrollment/           # Study-level enrollment service
├── imposter_creation/    # Pairs videos + stitches composed videos
├── authentication/       # Per-frame + per-video authentication services
├── video_parser/         # Filename parsers for each pool
│
└── core/                 # Reusable, study-agnostic building blocks
    ├── embedder/         # InsightFace / FaceNet embedders + embedding cache
    ├── detection/        # MediaPipe / MTCNN detectors (FaceNet path only)
    ├── enrollment/       # Frame-selection backends (fixed_order / pose)
    └── authenticator/    # Trust scoring (temporal_decay, trust_based)

src/evaluation/           # Reads result CSVs, computes metrics, makes figures
data/                     # Videos, enrollments, and results (not tracked)
```

**`core/` vs the feature packages:** `core/` holds reusable machinery that knows
nothing about this study (embedders, detectors, the trust algorithm). The
top-level feature packages (`enrollment/`, `imposter_creation/`,
`authentication/`) wrap that machinery with the study's path conventions and
domain types.

Most components (embedder, detector, enrollment backend, authenticator, matching
strategy) are chosen by a `type` field in the config and built through a factory,
so swapping an implementation is a config change, not a code change.

## Setup

Requires **Python ≥ 3.11** and **ffmpeg/ffprobe** (used for video rotation
detection). On macOS: `brew install ffmpeg`.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .            # installs the face-auth / face-auth-live commands
```

Models download automatically on first run. The default config uses InsightFace
`buffalo_sc` (~20 MB), which does its own face detection — no separate detector
model is needed. A separate MediaPipe/MTCNN detector is only required if you
switch the embedder to FaceNet.

## Data layout

Videos and results live under `data/` and are not tracked in git. The pipeline
relies on these path conventions and filename patterns.

```
data/
├── enrollments/                      # Shared across both studies
│   └── {device}/{name}/
│       ├── {name}_enrollment_{scenario}_{rotation}_{date}_{time}.mp4
│       └── {name}_enrollment_.../    # extracted enrollment frames (*.jpg)
│
├── controlled_study/                 # Mobile + desktop
│   └── {device}/{name}/
│       └── {name}_{scenario}_{date}_{time}.mp4      # scenario ∈ easy|angle|lighting
│
└── in_the_wild/                      # Mobile only
    └── mobile/{name}/
        ├── videos/{name}_{date}_{time}.mp4
        └── annotations/*.json                        # participant self-labels
```

## Running the batch pipeline

```bash
face-auth                                    # interactive: pick a study
face-auth src/face_auth/configs/controlled_study.json   # or pass a config
```

Two configs ship in `src/face_auth/configs/`:

- `controlled_study.json` — controlled study, mobile + desktop, scenario matching
- `in_the_wild.json` — in-the-wild study, mobile only, random imposter matching

### Config essentials

The config is validated by the Pydantic schema in `configs/models.py`
(**no defaults — every field is required**). The most relevant knobs:

| Field | Meaning |
|---|---|
| `pool` | `controlled_study` or `in_the_wild` (selects the filename parser) |
| `devices` | Devices to process, e.g. `["mobile", "desktop"]` |
| `models.embedder.type` | `insightface` or `facenet` |
| `run.num_workers` | Parallel workers (`1` = sequential, best for debugging) |
| `authentication.authentication_hz` | Sampling rate along the video timeline, in Hz |
| `authentication.backend` | Trust algorithm + its params (`threshold`, `k_weight`, `k_decay`, `similarity_percentile`) |
| `enrollment.force_reenrollment` | Rebuild enrollment frames even if cached |
| `imposter_creation.*_seconds` | Segment lengths of each composed video |
| `imposter_creation.matching_strategy.type` | How genuine/imposter videos are paired (`scenario`, `random_sampling`, `all`) |

### Output

Each run creates its own directory so runs never collide:

```
data/{study}/results/run_{timestamp}_{suffix}/
├── config.json                       # exact config used for this run
└── {participant}/{device}/
    └── {genuine_stem}_vs_{imposter_stem}.csv
```

Each CSV row is one sampled frame:

| Column | Meaning |
|---|---|
| `timestamp_ms` | Position on the composed video's timeline |
| `is_authenticated` | `trust_score >= threshold` |
| `similarity` | Cosine similarity to enrollment (`None` if no face) |
| `trust_score` | Current trust score |
| `face_detected` | Whether a face was found |
| `source_type` | Which segment the frame came from (genuine / `black_frames` / imposter) |

## Live demo

```bash
face-auth-live      # uses src/face_auth/configs/live_config.json
```

Enrolls by sampling frames evenly across `enrollment_video`, then authenticates
every webcam frame and draws lock status, trust, and a threshold-marked trust
bar. Press `q` to quit. Point `enrollment_video` at your own clip (~15 s, turning
your head) and set `camera_index` for your webcam.

## Evaluation

The evaluation package reads a completed run directory, computes metrics, and
produces interactive Plotly timelines and summary figures. See
[`src/evaluation/README.md`](src/evaluation/README.md) for details.
