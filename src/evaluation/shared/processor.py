"""Data processing utilities for evaluation.shared."""
from pathlib import Path
from typing import Optional
import pandas as pd

from evaluation.shared.models import SegmentBoundaries, VideoBoundaries


def parse_imposter_video_filename(video_path: str) -> tuple[Optional[str], Optional[str]]:
    """Parse imposter video filename to extract genuine and imposter user names.

    Args:
        video_path: Path like 'data/temp/.../miriam_2025-10-29_11-20-10_vs_gudrun_2025-12-04_12-42-43.mp4'

    Returns:
        Tuple of (genuine_user, imposter_user) or (None, None) if not an imposter video
    """
    filename = Path(video_path).stem

    if '_vs_' not in filename:
        return None, None

    parts = filename.split('_vs_')
    genuine_part = parts[0]
    imposter_part = parts[1]

    # Extract just the user name (first part before date)
    genuine_user = genuine_part.split('_')[0]
    imposter_user = imposter_part.split('_')[0]

    return genuine_user, imposter_user


def parse_scenario_from_filename(video_path: str) -> Optional[str]:
    """Parse scenario from video filename.

    Args:
        video_path: Path like 'data/temp/.../miriam_angle_2025-10-29_11-20-10_vs_gudrun_angle_2025-12-04_12-42-43.mp4'

    Returns:
        Scenario name (e.g., 'angle', 'easy', 'lighting') or None if not found
    """
    filename = Path(video_path).stem

    if '_vs_' not in filename:
        return None

    # Take the first part before '_vs_'
    parts = filename.split('_vs_')[0].split('_')

    # Pattern: {user}_{scenario}_{date}_{time}
    # So scenario should be at index 1
    if len(parts) >= 2:
        return parts[1]

    return None


def calculate_segment_boundaries(fps: int, genuine_seconds: float,
                                 black_seconds: float, imposter_seconds: float) -> VideoBoundaries:
    """Calculate frame boundaries for video segments.

    Args:
        fps: Frames per second
        genuine_seconds: Duration of genuine user segment
        black_seconds: Duration of black screen
        imposter_seconds: Duration of imposter segment

    Returns:
        VideoBoundaries dataclass with frame boundaries for all segments
    """
    genuine_frames = int(fps * genuine_seconds)
    black_frames = int(fps * black_seconds)
    imposter_frames = int(fps * imposter_seconds)

    return VideoBoundaries(
        genuine_segment=SegmentBoundaries(
            start_frame=1,
            end_frame=genuine_frames
        ),
        black_segment=SegmentBoundaries(
            start_frame=genuine_frames + 1,
            end_frame=genuine_frames + black_frames
        ),
        imposter_segment=SegmentBoundaries(
            start_frame=genuine_frames + black_frames + 1,
            end_frame=genuine_frames + black_frames + imposter_frames
        )
    )


def categorize_frames(df: pd.DataFrame, fps: int, genuine_seconds: float,
                     black_seconds: float, imposter_seconds: float) -> pd.DataFrame:
    """Categorize each frame as genuine, black, or imposter segment.

    Args:
        df: Results dataframe
        fps: Frames per second
        genuine_seconds: Duration of genuine user segment
        black_seconds: Duration of black screen
        imposter_seconds: Duration of imposter segment

    Returns:
        DataFrame with additional columns: segment_type, genuine_user, imposter_user
    """
    df = df.copy()

    # Parse video filenames to get user info
    video_info = df['video_path'].apply(parse_imposter_video_filename)
    df['genuine_user'] = video_info.apply(lambda x: x[0])
    df['imposter_user'] = video_info.apply(lambda x: x[1])

    # Calculate boundaries
    boundaries = calculate_segment_boundaries(fps, genuine_seconds, black_seconds, imposter_seconds)

    # Categorize frames
    def categorize_frame(frame):
        if frame <= boundaries.genuine_segment.end_frame:
            return 'genuine'
        elif frame <= boundaries.black_segment.end_frame:
            return 'black'
        else:
            return 'imposter'

    df['segment_type'] = df['frame'].apply(categorize_frame)

    return df


def merge_annotations(results_df: pd.DataFrame, annotations_df: pd.DataFrame) -> pd.DataFrame:
    """Merge annotations with results data.

    Args:
        results_df: Results dataframe
        annotations_df: Annotations dataframe

    Returns:
        Merged dataframe
    """
    if annotations_df is None or len(annotations_df) == 0:
        return results_df

    # Extract video filename from video_path for matching
    results_df = results_df.copy()
    results_df['video_filename'] = results_df['video_path'].apply(lambda x: Path(x).name)

    # Merge on video_filename
    merged_df = results_df.merge(
        annotations_df,
        on='video_filename',
        how='left',
        suffixes=('', '_annot')
    )

    print(f"Merged {len(annotations_df)} annotations with {len(results_df)} result rows")
    return merged_df


def add_grouping_columns(df: pd.DataFrame, grouping_dimensions: list[str],
                        annotations_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Add grouping columns from annotations for analysis.

    Args:
        df: Results dataframe
        grouping_dimensions: List of dimension names to group by (e.g., ['device', 'scenario'])
        annotations_df: Optional annotations dataframe to merge

    Returns:
        DataFrame with grouping columns added
    """
    df = df.copy()

    if annotations_df is not None:
        df = merge_annotations(df, annotations_df)

    # Extract scenario from video path if not already present
    if 'scenario' in grouping_dimensions and 'scenario' not in df.columns:
        df['scenario'] = df['video_path'].apply(parse_scenario_from_filename)
        print(f"Extracted 'scenario' column from video filenames")

    # Verify all grouping dimensions exist
    missing_dims = [dim for dim in grouping_dimensions if dim not in df.columns]
    if missing_dims:
        print(f"Warning: Missing grouping dimensions: {missing_dims}")

    return df


def get_video_display_name(video_path: str, include_extension: bool = False) -> str:
    """Get display name for video from path.

    Args:
        video_path: Full video path
        include_extension: Whether to include file extension

    Returns:
        Cleaned video name for display
    """
    filename = Path(video_path).name
    if not include_extension:
        filename = Path(video_path).stem
    return filename
