# conFaceAuth - Continuous Face Authentication

This repository contains a prototype for a continuous face authentication pipeline.

## Overview

The system provides two main execution modes:
1. **Live Mode** - For quick testing and real-time authentication using your webcam
2. **Batch Processing Mode** - For processing multiple pre-recorded participant videos from two study pools

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
├── in_the_wild/                    # Mobile-only videos
│   ├── videos/mobile/
│   │   └── [name]_[timestamp].mp4
│   └── results.csv
├── controlled_study/               # Mobile + desktop videos
│   ├── videos/
│   │   ├── mobile/
│   │   │   └── [name]_[mode]_[timestamp].mp4
│   │   └── desktop/
│   │       └── [name]_[mode]_[timestamp].mp4
│   └── results.csv
└── annotations/                    # For evaluation
    ├── in_the_wild_annotations.csv
    └── controlled_study_annotations.csv
```

### Key Structure Details

- **Enrollments are organized by device**: Each device (mobile/desktop) has its own enrollment folder hierarchy
- **Device-specific enrollment videos**: Each participant can have different enrollment videos for mobile and desktop
- **Processed images in subfolders**: Enrollment images are stored in a subfolder named after the enrollment video (without extension)
- **Multiple enrollments supported**: Each enrollment video gets its own subfolder for processed images
- **Shared across pools**: Enrollments can be used by both in_the_wild and controlled_study pools
- **Results per pool**: Each pool generates its own `results.csv` file

## Quick Start Guide

Make sure to execute commands from the `src` folder.

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

### Quick Testing with live.py

For quick testing, use `live.py`. This script will:
- Access your webcam
- Perform real-time face authentication
- Display the authentication results on screen

#### Setting Up Enrollment Data for Quick Testing

Before running live.py, create a folder with images of yourself:

1. Create a folder to store your enrollment images (e.g., `data/my_enrollment`)
2. Take several photos of yourself using your webcam
3. Update the `enrollment_folder` path in `src/live_config.json`:

```json
"enrollment": {
  "enrollment_folder": "../data/my_enrollment"
}
```

### Batch Processing with main.py

The processing pipeline now focuses solely on computing trust scores without comparing to annotations. Evaluation against ground truth happens separately.

#### Configuration

Two configuration files are available:

1. **`controlled_study_config.json`** - For controlled study pool (mobile + desktop)
2. **`in_the_wild_config.json`** - For in-the-wild pool (mobile only)

Example configuration structure:
```json
{
  "pool": "controlled_study",
  "base_path": "../data/controlled_study",
  "enrollment_base_path": "../data/enrollments",
  "participants": [
    {"name": "participant1"},
    {"name": "participant2"}
  ],
  "devices": ["mobile", "desktop"],
  "skip_frames": 30,
  "window_size": 5,
  "threshold": 0.8,
  "embedder": "FaceNet",
  "detector": "MTCNN",
  "similarity_computation": 0.1,
  "enrollment_frames_per_direction": 5,
  "no_face_penalty": 1.0,
  "alpha": 1.0
}
```

**Note:** The pipeline automatically discovers all videos for each participant based on the naming conventions above. You don't need to specify individual video files or sessions.

**File Naming Conventions:**

The pipeline automatically discovers videos based on these naming patterns:

1. **Enrollment videos** (in `data/enrollments/{device}/{participant_name}/`):
   - Pattern: `{name}_enrollment_{mode}_{rotation}_{timestamp}.mp4`
   - Mobile example: `miriam_enrollment_easy_cw_2025-10-31_11-16-15.MP4`
   - Desktop example: `miriam_enrollment_easy_cw_2025-10-31_08-01-23.mp4`

2. **Controlled study videos** (in `data/controlled_study/videos/{device}/`):
   - Pattern: `{name}_{mode}_{timestamp}.mp4`
   - Example: `miriam_easy_2025-10-31_08-01-47.mp4`
   - The `mode` field (e.g., "easy", "hard") is included in the filename

3. **In the wild videos** (in `data/in_the_wild/videos/mobile/`):
   - Pattern: `{name}_{timestamp}.mp4`
   - Example: `miriam_2025-10-29_11-20-10.mp4`
   - No mode field in the filename

4. **Processed enrollment images** (generated automatically):
   - Path: `data/enrollments/{device}/{participant_name}/{enrollment_video_name}/*.jpg`
   - Stored in a subfolder named after the enrollment video (without extension)
   - Example: `data/enrollments/mobile/miriam/miriam_enrollment_easy_cw_2025-10-31_11-16-15/*.jpg`

The pipeline will process all videos matching these patterns for each configured participant.

#### Running the Pipeline

You can run the pipeline in two ways:

1. **Interactive mode** (will prompt you to choose a config):
```bash
cd src
python main.py
```

2. **Command line with config file**:
```bash
cd src
python main.py controlled_study_config.json
# or
python main.py in_the_wild_config.json
```

#### Output CSV Format

The pipeline outputs a CSV file with the following columns:
- `frame` - Frame number
- `predicted_state` - "Unlocked" or "Locked" (determined by risk_score)
- `distance` - Distance metric between embedding and enrollment
- `risk_score` - Computed trust/risk score
- `face_detected` - Boolean indicating whether a face was detected in the frame
- `video_path` - Path to the processed video
- Configuration parameters (for reproducibility)

## Video Orientation Handling

The pipeline automatically handles video rotation using metadata-based detection:

- **Automatic Detection**: Uses `ffprobe` to read rotation metadata from video files
- **All Angles Supported**: Handles 0°, 90°, 180°, and 270° rotations
- **Consistent Processing**: Same rotation is applied to enrollment and authentication videos
- **No Manual Configuration**: No need to specify device type or manually rotate videos

**How it works:**
1. When processing starts, the system reads rotation metadata from the video file
2. Every frame is automatically rotated to the correct orientation
3. Frames are processed and saved already corrected
4. Works for both enrollment videos and authentication videos

**Fallback**: If `ffprobe` is not installed or rotation metadata is unavailable, videos are processed as-is (0° rotation assumed).

## Evaluation

Annotations are stored separately in `data/annotations/` and should be used in subsequent evaluation scripts to compare predicted states against ground truth. This separation allows the processing pipeline to focus on prediction while keeping evaluation as a distinct step.

## Project Structure

- `src/` - Source code directory
  - `face_auth/` - Core authentication components
  - `analysis/` - Jupyter notebooks for analysis
  - `live.py` - Script for live webcam authentication
  - `main.py` - Script for batch processing
  - `live_config.json` - Configuration for live mode
  - `controlled_study_config.json` - Configuration for controlled study pool
  - `in_the_wild_config.json` - Configuration for in-the-wild pool
- `data/` - Directory for storing data
  - `in_the_wild/` - In-the-wild video pool (mobile only)
  - `controlled_study/` - Controlled study pool (mobile + desktop)
  - `annotations/` - Ground truth annotations (for evaluation)
  - `_archive/` - Archived data from previous structure


## Sizes
 Face Detection:

  - MTCNN: ~2.2 MB (very lightweight!)
  - MediaPipe BlazeFace: ~2-3 MB (embedded in package)
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