import cv2
import subprocess
import json
from pathlib import Path


def get_video_rotation_from_metadata(video_path: Path) -> int:
    """
    Get the rotation angle from video metadata using ffprobe.

    Returns:
        int: Rotation angle (0, 90, 180, or 270)
    """
    try:
        # Use ffprobe to get video metadata
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            str(video_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode != 0:
            print(f"Warning: ffprobe failed for {video_path}, assuming no rotation")
            return 0

        metadata = json.loads(result.stdout)

        # Look for rotation in video streams
        for stream in metadata.get('streams', []):
            if stream.get('codec_type') == 'video':
                # Check for rotation tag
                rotation = stream.get('tags', {}).get('rotate', '0')
                try:
                    rotation_angle = int(rotation)
                    # Normalize to 0, 90, 180, 270
                    rotation_angle = rotation_angle % 360
                    print(f"Detected rotation: {rotation_angle}° for {video_path.name}")
                    return rotation_angle
                except (ValueError, TypeError):
                    pass

                # Check side_data_list for display matrix rotation
                for side_data in stream.get('side_data_list', []):
                    if side_data.get('side_data_type') == 'Display Matrix':
                        rotation = side_data.get('rotation', 0)
                        try:
                            rotation_angle = int(float(rotation))
                            rotation_angle = (-rotation_angle) % 360  # Display matrix uses negative angles
                            print(f"Detected rotation from display matrix: {rotation_angle}° for {video_path.name}")
                            return rotation_angle
                        except (ValueError, TypeError):
                            pass

        print(f"No rotation metadata found for {video_path.name}, assuming 0°")
        return 0

    except subprocess.TimeoutExpired:
        print(f"Warning: ffprobe timeout for {video_path}, assuming no rotation")
        return 0
    except FileNotFoundError:
        print("Warning: ffprobe not found. Please install ffmpeg for automatic rotation detection.")
        print("Falling back to heuristic detection.")
        return 0
    except Exception as e:
        print(f"Warning: Error detecting rotation for {video_path}: {e}")
        return 0


def rotate_frame(frame, rotation_angle):
    """Rotate a frame based on the rotation angle."""
    if rotation_angle == 0:
        return frame
    elif rotation_angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation_angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        print(f"Warning: Unsupported rotation angle {rotation_angle}, not rotating")
        return frame


def get_frame_dimensions_after_rotation(width, height, rotation_angle):
    """Get the dimensions of a frame after rotation."""
    if rotation_angle in [90, 270]:
        return height, width
    else:
        return width, height