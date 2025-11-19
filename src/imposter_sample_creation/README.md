# Imposter Sample Creation

Scripts for creating stitched videos combining genuine user, black screen, and impostor segments for the controlled study.

The purpose of these scripts is to create videos that simulate impostor attempts. Imposter attempts normally consist of a genuine user segment followed by an impostor segment.
The black screen segment is added in between as usually there is a delay between the genuine user and impostor in real scenarios (maybe imposter covers camera when taking over).

## Scripts

- **`stitch.py`** - Core video stitching functionality. Can be used standalone to stitch two specific videos.
- **`batch_stitch.py`** - Batch processing script that uses `stitch.py` to create stitched videos for all participant combinations.

## Configuration

All frame alignment is controlled by `stitch_config.json`:

```json
{
  "fps": 30,
  "genuine_user_seconds": 10.0,
  "black_screen_seconds": 3.33,
  "impostor_seconds": 10.0
}
```

**Current frame alignment:**
- Frames 1-300: Genuine user (10.0s × 30fps)
- Frames 301-399: Black screen (3.33s × 30fps ≈ 99 frames)
- Frames 400-699: Impostor (10.0s × 30fps)

## Usage

### Option 1: Stitch Two Specific Videos

Use `stitch.py` to stitch two specific videos together:

```bash
python3 stitch.py <video1> <video2> <output> [config_path]
```

**Arguments:**
- `video1` - Path to genuine user video
- `video2` - Path to impostor video
- `output` - Path for output stitched video
- `config_path` - Optional path to config JSON (default: stitch_config.json)

**Example:**
```bash
python3 stitch.py john_easy.mp4 sarah_easy.mp4 john_vs_sarah.mp4
```

### Option 2: Batch Process Participant Videos

Use `batch_stitch.py` to process all videos for a participant and create stitched versions with all other participants:

```bash
python3 batch_stitch.py <participant_name> [device]
```

**Arguments:**
- `participant_name` - Name of the genuine user participant
- `device` - 'mobile' or 'desktop' (default: desktop)

**Output Structure:**
Videos are automatically saved to: `data/impostor_samples/{device}/{participant}/`

**Example:**
```bash
# Process miriam's desktop videos
python3 batch_stitch.py miriam desktop
# Output: data/impostor_samples/desktop/miriam/

# Process john's mobile videos
python3 batch_stitch.py john mobile
# Output: data/impostor_samples/mobile/john/
```

**What it does:**
1. Finds all videos for the specified participant (easy, angle, lighting categories)
2. Finds all other participants with matching categories
3. Creates stitched videos for each combination:
   - `miriam_easy_vs_john.mp4`
   - `miriam_easy_vs_sarah.mp4`
   - `miriam_angle_vs_john.mp4`
   - etc.

## Output Naming Convention

Stitched videos are named: `{genuine_user}_{category}_vs_{impostor}.mp4`

Examples:
- `miriam_easy_vs_john.mp4` - Miriam (easy) as genuine, John (easy) as impostor
- `miriam_angle_vs_sarah.mp4` - Miriam (angle) as genuine, Sarah (angle) as impostor

## Project Structure

```
imposter_sample_creation/
├── stitch.py              # Core stitching functionality
├── batch_stitch.py        # Batch processing script
├── stitch_config.json     # Configuration file
├── README.md
└── tests/                 # Test suite
    ├── __init__.py
    ├── test_stitch.py         # Tests for stitch.py
    └── test_batch_stitch.py   # Tests for batch_stitch.py
```

## Testing

Run all tests using unittest discovery:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Or run individual test files:

```bash
cd tests
python test_stitch.py -v
python test_batch_stitch.py -v
```