"""Data loading and parsing utilities."""
import re
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from evaluation.shared.models import (
    EvaluationData,
    FrameData,
    VideoMetadata,
    SegmentType
)


def load_evaluation_data(csv_path: Path, parse_scenario: bool = False) -> EvaluationData:
    """Load CSV and convert to domain objects."""
    print("Loading data...")
    df = pd.read_csv(csv_path)
    threshold = df['threshold'].iloc[0]
    skip_frames = int(df['skip_frames'].iloc[0])

    config_path = csv_path.parent / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    fps = config['imposter_creation']['fps']

    videos = extract_video_metadata(df, parse_scenario)
    frames = []

    total = len(df)
    for idx, row in df.iterrows():
        if idx % 10000 == 0:
            print(f"  {idx}/{total} rows...")

        video_meta = next((v for v in videos if v.video_path == row['video_path']), None)
        segment_type = determine_segment_type(row['source_type'], video_meta)

        frames.append(FrameData(
            frame=row['frame'],
            predicted_state=row['predicted_state'],
            similarity=row['similarity'] if 'similarity' in df.columns else row.get('distance', 0.0),
            trust_score=row['trust_score'] if 'trust_score' in df.columns else row.get('risk_score', 0.0),
            face_detected=row['face_detected'],
            source_type=row['source_type'],
            participant=row['participant'],
            device=row['device'],
            video_path=row['video_path'],
            segment_type=segment_type,
            scenario=video_meta.scenario if video_meta else None
        ))

    return EvaluationData(frames=frames, threshold=threshold, videos=videos, skip_frames=skip_frames, fps=fps)


def extract_video_metadata(df: pd.DataFrame, parse_scenario: bool) -> list[VideoMetadata]:
    """Extract metadata from video paths."""
    videos = []

    for video_path in df['video_path'].unique():
        if parse_scenario:
            match = re.search(
                r'([^/]+)_([^_]+)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_vs_([^_]+)_',
                video_path
            )
            if match:
                videos.append(VideoMetadata(
                    video_path=video_path,
                    genuine_user=match.group(1),
                    scenario=match.group(2),
                    imposter_user=match.group(3)
                ))
        else:
            match = re.search(
                r'([^/]+)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_vs_([^_]+)_',
                video_path
            )
            if match:
                videos.append(VideoMetadata(
                    video_path=video_path,
                    genuine_user=match.group(1),
                    imposter_user=match.group(2)
                ))

    return videos


def determine_segment_type(source_type: str, video_meta: Optional[VideoMetadata]) -> SegmentType:
    """Determine if frame is genuine, imposter, or black segment."""
    if not video_meta:
        return SegmentType.GENUINE

    if 'black' in source_type.lower():
        return SegmentType.BLACK
    elif video_meta.genuine_user in source_type:
        return SegmentType.GENUINE
    elif video_meta.imposter_user in source_type:
        return SegmentType.IMPOSTER
    else:
        return SegmentType.GENUINE
