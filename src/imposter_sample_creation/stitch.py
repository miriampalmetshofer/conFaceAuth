#!/usr/bin/env python3
"""Core video stitching functionality for creating impostor test videos."""

import os
import sys
import json
import subprocess


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


def load_config(config_path=None):
    """Load stitching configuration from stitch_config.json."""
    if config_path is None:
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
    print("  Analyzing videos...")
    width1, height1, fps1, duration1 = get_video_info(video1_path)
    width2, height2, fps2, duration2 = get_video_info(video2_path)

    print(f"  Video 1: {width1}x{height1} @ {fps1:.2f}fps, duration: {duration1:.2f}s")
    print(f"  Video 2: {width2}x{height2} @ {fps2:.2f}fps, duration: {duration2:.2f}s")

    # Use dimensions from first video
    width, height = width1, height1

    # Calculate start time for last impostor_seconds of video2
    start_time2 = max(0, duration2 - impostor_seconds)

    print(f"\n  {'='*60}")
    print(f"  FRAME ALIGNMENT")
    print(f"  {'='*60}")
    print(f"  Genuine user:  Frames {genuine_start:3d}-{genuine_end:3d}  ({genuine_seconds:.2f}s × {fps} fps)")
    print(f"  Black screen:  Frames {black_start:3d}-{black_end:3d}  ({black_seconds:.2f}s × {fps} fps)")
    print(f"  Impostor:      Frames {impostor_start:3d}-{impostor_end:3d}  ({impostor_seconds:.2f}s × {fps} fps)")
    print(f"  Total:         {total_frames} frames ({total_frames/fps:.2f}s)")
    print(f"  {'='*60}")

    print("\n  Processing...")

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
        print(f"\n  ✓ Success! Segment alignment:")
        print(f"    Frames {genuine_start}-{genuine_end}: Genuine user")
        print(f"    Frames {black_start}-{black_end}: Black screen")
        print(f"    Frames {impostor_start}-{impostor_end}: Impostor")
    else:
        raise RuntimeError(f"FFmpeg error: {result.stderr}")


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python stitch.py <video1> <video2> <output> [config_path]")
        print("\nArguments:")
        print("  video1       - Path to genuine user video")
        print("  video2       - Path to impostor video")
        print("  output       - Path for output stitched video")
        print("  config_path  - Optional path to config JSON (default: stitch_config.json)")
        print("\nExample:")
        print("  python stitch.py john_easy.mp4 sarah_easy.mp4 john_vs_sarah.mp4")
        print("\nConfiguration (stitch_config.json):")
        print("  - fps: Target frame rate")
        print("  - genuine_user_seconds: Duration of genuine user segment")
        print("  - black_screen_seconds: Duration of black screen transition")
        print("  - impostor_seconds: Duration of impostor segment")
        sys.exit(1)

    video1 = sys.argv[1]
    video2 = sys.argv[2]
    output = sys.argv[3]
    config_path = sys.argv[4] if len(sys.argv) > 4 else None

    # Validate input files exist
    if not os.path.exists(video1):
        print(f"Error: Video 1 not found: {video1}")
        sys.exit(1)
    if not os.path.exists(video2):
        print(f"Error: Video 2 not found: {video2}")
        sys.exit(1)

    # Load configuration
    try:
        config = load_config(config_path)
        print(f"Loaded configuration:")
        print(f"  FPS: {config['fps']}")
        print(f"  Genuine user: {config['genuine_user_seconds']}s")
        print(f"  Black screen: {config['black_screen_seconds']}s")
        print(f"  Impostor: {config['impostor_seconds']}s")
    except FileNotFoundError:
        print("Error: stitch_config.json not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {e}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Stitching videos")
    print(f"{'='*70}")
    print(f"  Genuine user: {video1}")
    print(f"  Impostor: {video2}")
    print(f"  Output: {output}")
    print()

    try:
        stitch_videos(video1, video2, output, config)
        print(f"\n{'='*70}")
        print(f"✓ Complete! Output saved to: {output}")
        print(f"{'='*70}\n")
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"✗ Error: {e}")
        print(f"{'='*70}\n")
        sys.exit(1)
