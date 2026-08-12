import cv2
import subprocess
import json
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from face_auth.core.models import Frame
from face_auth.logging_config import get_logger

logger = get_logger(__name__)


def read_frames(
    video_path: Path,
    *,
    sample_every: int = 1,
    start_second: float = 0,
    duration_seconds: Optional[float] = None,
) -> Iterator[Frame]:
    """Read frames from a video as timestamped RGB Frames, with rotation applied.

    Args:
        video_path: Path to the video file
        sample_every: Yield every Nth frame (1 yields all frames)
        start_second: Time offset to start reading from
        duration_seconds: How many seconds to read, or None to read to the end

    Yields:
        Frame instances carrying the RGB image, its timestamp, and the source name

    Raises:
        IOError: If the video cannot be opened
        ValueError: If the video FPS cannot be determined
    """
    rotation_angle = get_video_rotation_from_metadata(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError(f"Could not determine FPS for video {video_path}")

        start_frame = int(start_second * fps)
        end_frame = None if duration_seconds is None else start_frame + int(duration_seconds * fps)

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_index = start_frame
        while end_frame is None or frame_index < end_frame:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_index += 1

            if (frame_index - start_frame) % sample_every != 0:
                continue

            frame_bgr = rotate_frame(frame_bgr, rotation_angle)
            yield Frame(
                image=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                timestamp_ms=(frame_index / fps) * 1000,
                source_name=video_path.stem,
            )
    finally:
        cap.release()


def load_image_rgb(image_path: Path) -> Optional[np.ndarray]:
    """Load an image file from disk as an RGB array.

    Args:
        image_path: Path to the image file

    Returns:
        The RGB image, or None if the file could not be read
    """
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_image_rgb(image_rgb: np.ndarray, image_path: Path) -> None:
    """Save an RGB image array to disk.

    Args:
        image_rgb: RGB image to save
        image_path: Destination path for the written image
    """
    cv2.imwrite(str(image_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


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
            raise RuntimeError(
                f"ffprobe failed for {video_path}. Cannot safely determine video "
                f"rotation metadata. stderr: {result.stderr.strip()}"
            )

        metadata = json.loads(result.stdout)

        # Look for rotation in video streams
        for stream in metadata.get('streams', []):
            if stream.get('codec_type') == 'video':
                # Check for rotation tag
                rotation = stream.get('tags', {}).get('rotate')
                if rotation is not None:
                    try:
                        rotation_angle = int(rotation)
                        # Normalize to 0, 90, 180, 270
                        rotation_angle = rotation_angle % 360
                        if rotation_angle != 0:
                            logger.debug(
                                f"Detected rotation {rotation_angle}° for {video_path.name}"
                            )
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
                            logger.debug(
                                f"Detected display-matrix rotation {rotation_angle}° "
                                f"for {video_path.name}"
                            )
                            return rotation_angle
                        except (ValueError, TypeError):
                            pass

        return 0

    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"ffprobe timed out for {video_path}. Cannot safely determine video "
            "rotation metadata."
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ffprobe was not found on PATH. Install ffmpeg or make ffprobe "
            "available before running the pipeline, because missing rotation "
            "metadata can make phone videos process sideways."
        )
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"Error detecting rotation for {video_path}. Cannot safely assume "
            f"0° rotation. Original error: {e}"
        ) from e


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
        logger.warning(f"Unsupported rotation angle {rotation_angle}; frame not rotated")
        return frame
