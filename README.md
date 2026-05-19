# conFaceAuth - Continuous Face Authentication

This repository contains a prototype for a continuous face authentication pipeline.

## Overview

The system provides two main execution modes:
1. **Live Mode (live.py)** - For quick testing and real-time authentication using your webcam
2. **Batch Processing Mode (main.py)** - For processing multiple pre-recorded participant videos from two study pools

## Data Organization

The project uses two separate video pools with a shared enrollment structure:

```
data/
├── enrollments/                    # Shared across all pools
│   ├── mobile/
│   │   └── [participant_name]/
│   │       ├── [name]_enrollment_*.MP4         # Enrollment video
│   │       └── [enrollment_video_name]/        # Subfolder for processed images
│   │           └── *.jpg
│   └── desktop/
│       └── [participant_name]/
│           ├── [name]_enrollment_*.mp4         # Enrollment video
│           └── [enrollment_video_name]/        # Subfolder for processed images
│               └── *.jpg     
│               
├── in_the_wild/                    # Mobile-only videos
│   ├── videos/mobile/
│   │   └── [name]_[timestamp].mp4
│   └── results.csv
│ 
├── controlled_study/               # Mobile + desktop videos
│   ├── videos/
│   │   ├── mobile/
│   │   │   └── [name]_[mode]_[timestamp].mp4
│   │   └── desktop/
│   │       └── [name]_[mode]_[timestamp].mp4
│   └── results.csv
└── annotations/                    # For evaluation
    |__*.json
```

## Quick Start Guide

### Prerequisites
- Python 3.7+
- **ffprobe** (for automatic video rotation detection)
  - macOS: `brew install ffmpeg` (includes ffprobe)
  - Linux: `sudo apt-get install ffmpeg` (includes ffprobe)
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) (includes ffprobe)

### Setup

1. **Create a virtual environment**:
```bash
python3 -m venv venv
```

2. **Activate the virtual environment**:
```bash
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Model Downloads

The project requires several pre-trained models. Some download automatically, others need manual download.

#### Automatic Downloads (No Action Required)

These models download automatically on first use:
- **InsightFace models** (`buffalo_l`, `buffalo_s`, `buffalo_sc`)
  - Downloads to `~/.insightface/models/`
  - Size: 10-326 MB depending on model variant
  - Used for face detection and embedding generation

**3. FaceNet/MTCNN Models** (handled by keras-facenet package)
- Location: `~/.keras/models/`
- Size: ~92 MB total


#### Manual Downloads (REQUIRED)

**1. MediaPipe FaceLandmarker** (for head pose estimation during enrollment)
```bash
curl -L -o src/face_auth/core/enrollment/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```
- Size: 3.6 MB
- Location: `face_auth/authentication/enrollment/face_landmarker.task`
- Used by: HeadPoseEstimator for enrollment video processing

**2. MediaPipe BlazeFace** (for face detection)
```bash
curl -L -o src/face_auth/core/detection/backend/impl/blaze_face_short_range.tflite \
  https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite
```
- Size: 224 KB
- Location: `src/face_auth/core/detection/backend/impl/blaze_face_short_range.tflite`
- Used by: MediaPipeBackend for face detection

### Quick Testing with live.py

For quick testing with real-time webcam authentication, use `live.py`. This script will:
- Access your webcam
- Perform real-time face authentication
- Display authentication results with visual feedback on screen

The script needs an enrollment video. This should be you recording yourself looking at the camera and turning your head up, down and sideways for about 15 seconds. Add the path to the enrollment video in the configuration file described below.

#### Configuration

Before running, you must configure `live_config.json` in the project root. Here's what each field means:

```json
{
  "enrollment_video": "data/enrollments/mobile/your_name/your_enrollment_video.mp4",
  "camera_index": 0,
  "fps": 30,
  "embedder": {
    "model": "insightface",
    "config": {
      "model_name": "buffalo_sc",
      "det_size": [640, 640]
    }
  },
  "temporal_decay": {
    "threshold": 0.5,
    "similarity_percentile": 0.90,
    "k_weight": 40000,
    "k_decay": 30000,
    "initial_confidence": 1.0
  },
  "enrollment": {
    "frame_sampling_interval": 30
  }
}
```

**Configuration Fields:**
- `enrollment_video`: Path to your enrollment video (must already exist, see batch processing for creating enrollments)
- `camera_index`: Camera device number (0 for default webcam, 1 for external camera, etc.)
- `fps`: Target frames per second for authentication updates
- `embedder.model`: Embedding model to use (`"insightface"` or `"facenet"`)
- `embedder.config.model_name`: Specific model variant (e.g., `"buffalo_sc"` for compact, `"buffalo_l"` for accuracy)
- `temporal_decay.threshold`: Trust score threshold (0.0-1.0) - user is locked when score drops below this
- `temporal_decay.k_weight`: Time constant weighting new observations (higher = longer memory)
- `temporal_decay.k_decay`: Time constant for confidence decay when no face detected
- `temporal_decay.initial_confidence`: Starting confidence score (typically 1.0)
- `enrollment.frame_sampling_interval`: Process every Nth frame from enrollment video

#### Running Live Authentication

```bash
python src/live.py
```

Press 'q' to quit the live authentication window.

### Batch Processing with main.py

The batch processing pipeline computes frame-by-frame authentication results (trust scores and lock/unlock states) for all participant videos in a study pool. Results are saved to CSV for later evaluation.

#### Configuration Files

Two configuration files are provided in `configs/`:

1. **`controlled_study.json`** - For controlled study pool (mobile + desktop videos)
2. **`in_the_wild.json`** - For in-the-wild pool (mobile-only videos)

Example configuration structure:
```json
{
  "pool": "controlled_study",
  "devices": ["desktop", "mobile"],
  "paths": {
    "base_path": "data/controlled_study",
    "enrollment_base_path": "data/enrollments",
    "results_file": "{base_path}/results.csv"
  },
  "models": {
    "detector": "mediapipe",
    "embedder": {
      "model": "insightface",
      "config": {
        "min_detection_confidence": 0.6
      }
    }
  },
  "processing": {
    "skip_frames": 30,
    "matching_strategy": {
      "type": "scenario",
      "config": {}
    },
    "num_workers": 4
  },
  "authentication": {
    "backend": "trust_based",
    "trust_based": {
      "threshold": 0.65,
      "window_size": 10,
      "similarity_percentile": 0.90,
      "alpha": 0.3,
      "no_face_penalty": 0.5
    },
    "fps": 30
  },
  "enrollment": {
    "enrollment_video_preference": {
      "scenario": "easy",
      "rotations": ["cw", "ccw"]
    },
    "backend": "fixed_order",
    "config": {
      "window_seconds": 2.0
    },
    "frames_per_direction": 5
  },
  "imposter_creation": {
    "fps": 30,
    "genuine_user_seconds": 180,
    "black_screen_seconds": 3,
    "impostor_seconds": 180
  },
  "participants": [
    {"name": "participant1"},
    {"name": "participant2"}
  ]
}
```

**Key Configuration Fields:**

- `pool`: Study pool identifier (`"controlled_study"` or `"in_the_wild"`)
- `devices`: List of devices to process (`["mobile"]` or `["mobile", "desktop"]`)
- `paths.base_path`: Directory containing study videos
- `paths.enrollment_base_path`: Directory containing enrollment data
- `paths.results_file`: Output CSV path (supports `{base_path}` placeholder)
- `models.detector`: Face detector (`"mediapipe"`, `"mtcnn"`, or `"insightface"`)
- `models.embedder.model`: Embedding model (`"insightface"` or `"facenet"`)
- `models.embedder.config`: Model-specific configuration (e.g., detection confidence threshold)
- `processing.skip_frames`: Process every Nth frame (30 = 1 frame per second at 30 FPS)
- `processing.matching_strategy.type`: Video pairing strategy (`"scenario"` matches same scenarios)
- `processing.num_workers`: Number of parallel processing workers. Set to 1 for debugging.
- `authentication.backend`: Authentication algorithm (`"trust_based"` or `"temporal_decay"`)
- `authentication.fps`: Video frame rate
- `enrollment.backend`: Enrollment frame selection method (`"fixed_order"` or `"pose"`)
- `enrollment.config`: Backend-specific enrollment settings
- `enrollment.frames_per_direction`: How many frames to extract per head rotation direction
- `imposter_creation.genuine_user_seconds`: Duration of genuine user footage in composed videos
- `imposter_creation.impostor_seconds`: Duration of imposter footage in composed videos
- `participants`: List of participants to process

**Note:** Individual models can be configured by changing the `models.detector` and `models.embedder.model` fields. See the "Model Sizes" section below for available options.

#### Running the Pipeline

1. **Interactive mode** (will prompt you to choose a config):
```bash
python src/main.py
```


#### Output CSV Format

The pipeline outputs a CSV file (`results.csv`) with the following columns:
- `frame` - Frame number within the video
- `predicted_state` - Authentication state: "Unlocked" or "Locked"
- `similarity` - Cosine similarity to enrollment embeddings (0.0-1.0, higher = more similar)
- `trust_score` - Computed trust/confidence score (0.0-1.0)
- `face_detected` - Boolean indicating whether a face was detected in the frame
- `source_type` - Video segment type (e.g., "genuine", "imposter", "black_screen")
- `video_path` - Path to the processed video file
- `participant` - Participant name
- `device` - Device used for recording (mobile/desktop)
- `scenario` - Recording scenario (controlled study only, e.g., "easy", "angle", "lighting")

## Evaluation

For detailed evaluation instructions, see [src/evaluation/README.md](src/evaluation/README.md).

The evaluation module analyzes authentication results from `results.csv` and generates:
- Interactive HTML timelines showing trust scores over time
- Summary visualizations with key performance metrics
- Detailed metrics tables broken down by device and scenario

## Model Sizes
 Face Detection:

  - MTCNN: ~2.2 MB 
  - MediaPipe BlazeFace: ~2-3 MB
  - InsightFace RetinaFace (det_10g): 16 MB

  Face Recognition (Embeddings):

  - FaceNet: ~90 MB (Inception-ResNet)
  - InsightFace buffalo_l: 166 MB (ResNet50) - w600k_r50.onnx
  - InsightFace buffalo_s: ~50 MB (MobileFaceNet)
  - InsightFace buffalo_sc: ~10 MB (compressed MobileFaceNet)

  Total Package Sizes:

  - FaceNet + MTCNN: ~92 MB
  - InsightFace buffalo_l: 326 MB (detection + recognition + extras)
  - InsightFace buffalo_sc: ~20 MB (detection + recognition, mobile-optimized)
