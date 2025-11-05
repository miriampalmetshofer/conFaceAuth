#!/usr/bin/env python3

import os
import sys
import json
import re
import subprocess
from pathlib import Path


def parse_video_filename(filename):
    """Extract participant name and category from video filename."""
    # Pattern: {name}_{category}_{timestamp}.mp4
    match = re.match(r'([^_]+)_(easy|angle|lighting)_', filename, re.IGNORECASE)
    if match:
        return match.group(1), match.group(2).lower()
    return None, None


def discover_videos(base_path, device):
    """Discover all videos in controlled study organized by participant and category."""
    videos_by_participant = {}
    video_dir = Path(base_path) / device

    if not video_dir.exists():
        return videos_by_participant

    for video_file in video_dir.glob('*.[mM][pP]4'):
        participant, category = parse_video_filename(video_file.name)
        if participant and category:
            if participant not in videos_by_participant:
                videos_by_participant[participant] = {}
            videos_by_participant[participant][category] = video_file

    return videos_by_participant


def get_video_info(video_path):
    """Get video dimensions, duration, and frame rate."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate',
        '-show_entries', 'format=duration',
        '-of', 'csv=p=0',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout.strip().split('\n')

    # Parse dimensions and frame rate
    stream_info = output[0].split(',')
    width = int(stream_info[0])
    height = int(stream_info[1])
    fps_parts = stream_info[2].split('/')
    fps = int(fps_parts[0]) / int(fps_parts[1]) if len(fps_parts) == 2 else int(fps_parts[0])

    # Parse duration
    duration = float(output[1])

    return width, height, fps, duration


def load_config():
    """Load stitching configuration from stitch_config.json."""
    config_path = os.path.join(os.path.dirname(__file__), 'stitch_config.json')
    with open(config_path, 'r') as f:
        return json.load(f)


def stitch_videos(video1_path, video2_path, output_path, config):
    """Stitch videos using configuration for exact frame alignment."""

    # Extract config values
    fps = config['fps']
    genuine_seconds = config['genuine_user_seconds']
    black_seconds = config['black_screen_seconds']
    impostor_seconds = config['impostor_seconds']

    # Calculate frame ranges
    genuine_frames = int(genuine_seconds * fps)
    black_frames = int(black_seconds * fps)
    impostor_frames = int(impostor_seconds * fps)

    # Calculate exact frame boundaries
    genuine_start = 1
    genuine_end = genuine_frames
    black_start = genuine_end + 1
    black_end = genuine_end + black_frames
    impostor_start = black_end + 1
    impostor_end = black_end + impostor_frames
    total_frames = impostor_end

    # Get video info
    print("    Analyzing videos...")
    width1, height1, fps1, duration1 = get_video_info(video1_path)
    width2, height2, fps2, duration2 = get_video_info(video2_path)

    print(f"    Video 1: {width1}x{height1} @ {fps1:.2f}fps, duration: {duration1:.2f}s")
    print(f"    Video 2: {width2}x{height2} @ {fps2:.2f}fps, duration: {duration2:.2f}s")

    # Use dimensions from first video
    width, height = width1, height1

    # Calculate start time for last impostor_seconds of video2
    start_time2 = max(0, duration2 - impostor_seconds)

    print(f"\n    {'='*60}")
    print(f"    FRAME ALIGNMENT")
    print(f"    {'='*60}")
    print(f"    Genuine user:  Frames {genuine_start:3d}-{genuine_end:3d}  ({genuine_seconds:.2f}s × {fps} fps)")
    print(f"    Black screen:  Frames {black_start:3d}-{black_end:3d}  ({black_seconds:.2f}s × {fps} fps)")
    print(f"    Impostor:      Frames {impostor_start:3d}-{impostor_end:3d}  ({impostor_seconds:.2f}s × {fps} fps)")
    print(f"    Total:         {total_frames} frames ({total_frames/fps:.2f}s)")
    print(f"    {'='*60}")

    print("\n    Processing...")

    # FFmpeg filter chain for precise video stitching:
    # 1. [0:v] - Select video stream from first input (genuine user)
    #    trim=0:{genuine_seconds} - Extract first N seconds
    #    setpts=PTS-STARTPTS - Reset timestamps to start at 0
    #    scale={width}:{height} - Normalize dimensions
    #    [v1] - Label output as v1
    # 2. color=black - Generate black frames
    #    s={width}x{height} - Match video dimensions
    #    d={black_seconds} - Duration of black screen
    #    r={fps} - Frame rate
    #    [vblack] - Label output as vblack
    # 3. [1:v] - Select video stream from second input (impostor)
    #    trim={start_time2}:{duration2} - Extract last N seconds
    #    setpts=PTS-STARTPTS - Reset timestamps to start at 0
    #    scale={width}:{height} - Normalize dimensions
    #    [v2] - Label output as v2
    # 4. [v1][vblack][v2]concat=n=3:v=1:a=0 - Concatenate 3 segments (video only, no audio)
    #    [outv] - Final output stream
    filter_complex = f"""
    [0:v]trim=0:{genuine_seconds},setpts=PTS-STARTPTS,scale={width}:{height}[v1];
    color=black:s={width}x{height}:d={black_seconds}:r={fps}[vblack];
    [1:v]trim={start_time2}:{duration2},setpts=PTS-STARTPTS,scale={width}:{height}[v2];
    [v1][vblack][v2]concat=n=3:v=1:a=0[outv]
    """

    cmd = [
        'ffmpeg',
        '-i', video1_path,              # Input video 1 (genuine user)
        '-i', video2_path,              # Input video 2 (impostor)
        '-filter_complex', filter_complex.strip(),  # Apply trimming, scaling, and concatenation
        '-map', '[outv]',               # Map the filtered output stream
        '-c:v', 'libx264',              # Use H.264 codec for video encoding
        '-preset', 'medium',            # Encoding speed/quality tradeoff
        '-crf', '23',                   # Constant Rate Factor (quality level, 23 is default)
        '-r', str(fps),                 # Force output frame rate from config
        '-vsync', 'cfr',                # Constant frame rate (ensures exact frame count)
        '-an',                          # No audio stream
        '-y',                           # Overwrite output file if it exists
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"\n    ✓ Success! Segment alignment:")
        print(f"      Frames {genuine_start}-{genuine_end}: Genuine user")
        print(f"      Frames {black_start}-{black_end}: Black screen")
        print(f"      Frames {impostor_start}-{impostor_end}: Impostor")
    else:
        raise RuntimeError(f"FFmpeg error: {result.stderr}")


def batch_stitch_participant(participant_name, base_path, device, output_dir, config):
    """Stitch videos for one participant with all other participants."""

    print(f"\n{'='*70}")
    print(f"Processing participant: {participant_name} ({device})")
    print(f"{'='*70}\n")

    # Discover all videos
    all_videos = discover_videos(base_path, device)

    if participant_name not in all_videos:
        print(f"Error: No videos found for participant '{participant_name}'")
        print(f"Available participants: {', '.join(all_videos.keys())}")
        return

    genuine_videos = all_videos[participant_name]
    print(f"Found {len(genuine_videos)} video(s) for {participant_name}:")
    for category, path in genuine_videos.items():
        print(f"  - {category}: {path.name}")

    # Get other participants
    impostor_participants = {p: videos for p, videos in all_videos.items()
                            if p != participant_name}

    if not impostor_participants:
        print(f"\nWarning: No other participants found. Need at least 2 participants for stitching.")
        return

    print(f"\nFound {len(impostor_participants)} impostor participant(s):")
    for impostor_name in impostor_participants.keys():
        print(f"  - {impostor_name}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Stitch videos for each category
    total_videos = 0
    skipped_videos = []

    for category in genuine_videos.keys():
        genuine_video = genuine_videos[category]

        print(f"\n{'-'*70}")
        print(f"Category: {category}")
        print(f"{'-'*70}")

        for impostor_name, impostor_videos in impostor_participants.items():
            if category not in impostor_videos:
                skip_reason = f"no '{category}' video found"
                print(f"  ⚠ Skipping {impostor_name}: {skip_reason}")
                skipped_videos.append((category, impostor_name, skip_reason))
                continue

            impostor_video = impostor_videos[category]

            # Generate output filename
            output_filename = f"{participant_name}_{category}_vs_{impostor_name}.mp4"
            output_file = output_path / output_filename

            print(f"\n  Creating: {output_filename}")
            print(f"    Genuine: {genuine_video.name}")
            print(f"    Impostor: {impostor_video.name}")

            # Validate video lengths before stitching
            _, _, _, duration1 = get_video_info(str(genuine_video))
            _, _, _, duration2 = get_video_info(str(impostor_video))

            # Allow 0.5 second tolerance for rounding/encoding differences
            duration_diff = abs(duration1 - duration2)
            if duration_diff > 0.5:
                error_msg = (
                    f"\n{'='*70}\n"
                    f"ERROR: Video duration mismatch!\n"
                    f"{'='*70}\n"
                    f"Genuine video: {genuine_video.name}\n"
                    f"  Duration: {duration1:.2f}s\n"
                    f"Impostor video: {impostor_video.name}\n"
                    f"  Duration: {duration2:.2f}s\n"
                    f"Difference: {duration_diff:.2f}s\n\n"
                    f"Videos must have the same length (tolerance: 0.5s).\n"
                    f"{'='*70}"
                )
                raise ValueError(error_msg)

            print(f"    Duration check: ✓ (genuine={duration1:.2f}s, impostor={duration2:.2f}s)")

            stitch_videos(str(genuine_video), str(impostor_video), str(output_file), config)
            total_videos += 1

    print(f"\n{'='*70}")
    print(f"✓ Completed! Created {total_videos} stitched video(s)")
    print(f"  Output directory: {output_path}")
    if skipped_videos:
        print(f"\n⚠ Skipped {len(skipped_videos)} video(s):")
        for cat, imp_name, reason in skipped_videos:
            print(f"    {cat} vs {imp_name}: {reason}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python batch_stitch.py <participant_name> [device]")
        print("\nArguments:")
        print("  participant_name  - Name of the genuine user participant")
        print("  device           - 'mobile' or 'desktop' (default: desktop)")
        print("\nExample:")
        print("  python batch_stitch.py miriam desktop")
        print("\nOutput structure:")
        print("  data/impostor_samples/{device}/{participant}/")
        print("    ├── miriam_easy_vs_john.mp4")
        print("    ├── miriam_easy_vs_sarah.mp4")
        print("    └── ...")
        sys.exit(1)

    participant = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else 'desktop'

    # Validate device
    if device not in ['mobile', 'desktop']:
        print(f"Error: device must be 'mobile' or 'desktop', got '{device}'")
        sys.exit(1)

    # Load configuration
    try:
        config = load_config()
        print(f"Loaded configuration from stitch_config.json")
        print(f"  FPS: {config['fps']}")
        print(f"  Genuine user: {config['genuine_user_seconds']}s")
        print(f"  Black screen: {config['black_screen_seconds']}s")
        print(f"  Impostor: {config['impostor_seconds']}s")
    except FileNotFoundError:
        print("Error: stitch_config.json not found in script directory")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in stitch_config.json: {e}")
        sys.exit(1)

    # Set paths
    script_dir = Path(__file__).parent
    base_path = script_dir.parent / 'data' / 'controlled_study'
    output_dir = script_dir.parent / 'data' / 'impostor_samples' / device / participant

    if not base_path.exists():
        print(f"Error: Controlled study directory not found: {base_path}")
        sys.exit(1)

    batch_stitch_participant(participant, str(base_path), device, str(output_dir), config)