import cv2
import subprocess
import json
import os


def get_video_rotation(video_path):
    """
    Get the rotation angle from video metadata using ffprobe.

    Args:
        video_path: Path to the video file

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
            video_path
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
                    print(f"Detected rotation: {rotation_angle}° for {os.path.basename(video_path)}")
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
                            print(f"Detected rotation from display matrix: {rotation_angle}° for {os.path.basename(video_path)}")
                            return rotation_angle
                        except (ValueError, TypeError):
                            pass

        print(f"No rotation metadata found for {os.path.basename(video_path)}, assuming 0°")
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
    """
    Rotate a frame based on the rotation angle.

    Args:
        frame: The frame to rotate (numpy array)
        rotation_angle: Angle in degrees (0, 90, 180, 270)

    Returns:
        Rotated frame
    """
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
    """
    Get the dimensions of a frame after rotation.

    Args:
        width: Original width
        height: Original height
        rotation_angle: Rotation angle (0, 90, 180, 270)

    Returns:
        tuple: (new_width, new_height)
    """
    if rotation_angle in [90, 270]:
        return (height, width)
    else:
        return (width, height)


def apply_rotation_to_video(input_path, output_path, rotation_angle):
    """
    Create a new video file with rotation applied at the pixel level.
    This is useful for creating properly oriented enrollment videos.

    Args:
        input_path: Path to input video
        output_path: Path to output video
        rotation_angle: Rotation angle (0, 90, 180, 270)
    """
    if rotation_angle == 0:
        print(f"No rotation needed, copying file...")
        import shutil
        shutil.copy2(input_path, output_path)
        return

    cap = cv2.VideoCapture(input_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate dimensions after rotation
    out_width, out_height = get_frame_dimensions_after_rotation(width, height, rotation_angle)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rotated_frame = rotate_frame(frame, rotation_angle)
        out.write(rotated_frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Rotated {frame_count} frames by {rotation_angle}° and saved to {output_path}")