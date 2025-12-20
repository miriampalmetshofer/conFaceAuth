"""Data loading utilities for evaluation studies."""
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from evaluation.shared.config import StudyConfig
from evaluation.shared.models import StitchConfig


def load_results_csv(results_path: Path) -> pd.DataFrame:
    """Load results CSV file."""
    df = pd.read_csv(results_path)
    print(f"Loaded {len(df)} result rows from {results_path.name}")
    print(f"Columns: {list(df.columns)}")
    return df


def load_annotation_schema(schema_path: Path) -> dict:
    """Load annotation schema from JSON file."""
    with open(schema_path, 'r') as f:
        schema = json.load(f)

    print(f"Annotation schema loaded from {schema_path.name}")
    print(f"  List fields: {schema['list_fields']}")
    print(f"  Single choice fields: {schema['single_choice_fields']}")

    return schema


def load_stitch_config(config_path: Path) -> StitchConfig:
    """Load stitch configuration from JSON config file."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    stitch_data = config.get('imposter_creation', {})

    stitch_config = StitchConfig(
        fps=stitch_data.get('fps', 25),
        genuine_user_seconds=stitch_data.get('genuine_user_seconds', 20.0),
        black_screen_seconds=stitch_data.get('black_screen_seconds', 2.0),
        impostor_seconds=stitch_data.get('impostor_seconds', 20.0)
    )

    print(f"Loaded stitch config from {config_path.name}")
    print(f"  FPS: {stitch_config.fps}")
    print(f"  Genuine seconds: {stitch_config.genuine_user_seconds}")
    print(f"  Black screen seconds: {stitch_config.black_screen_seconds}")
    print(f"  Imposter seconds: {stitch_config.impostor_seconds}")

    return stitch_config


def load_annotations(annotations_path: Path, schema_path: Path) -> pd.DataFrame:
    """Load all annotation JSON files into a DataFrame.

    Args:
        annotations_path: Directory containing annotation JSON files
        schema_path: Path to annotation schema JSON

    Returns:
        DataFrame with all annotations
    """
    schema = load_annotation_schema(schema_path)
    annotations_list = []

    for json_file in annotations_path.glob("*.json"):
        with open(json_file, 'r') as f:
            annotation = json.load(f)

        row = {}

        # Add metadata fields
        for field in schema['metadata_fields']:
            row[field] = annotation.get(field, '')

        annot_data = annotation.get('annotations', {})

        # Store all annotation fields
        all_annotation_fields = schema['list_fields'] + schema['single_choice_fields']
        for key in all_annotation_fields:
            if key in annot_data:
                row[key] = annot_data[key]

        annotations_list.append(row)

    annotations_df = pd.DataFrame(annotations_list)
    print(f"Loaded {len(annotations_df)} annotations from {annotations_path.name}")

    return annotations_df


def load_study_data(config: StudyConfig) -> tuple[pd.DataFrame, StitchConfig, Optional[pd.DataFrame]]:
    """Load all data for a study.

    Args:
        config: Study configuration

    Returns:
        Tuple of (results_df, stitch_config, annotations_df or None)
    """
    print(f"\n{'='*80}")
    print(f"LOADING DATA FOR {config.name.upper().replace('_', ' ')} STUDY")
    print(f"{'='*80}\n")

    results_df = load_results_csv(config.results_path)
    stitch_config = load_stitch_config(config.config_path)

    annotations_df = None
    if config.has_annotations and config.annotations_path.exists():
        annotations_df = load_annotations(config.annotations_path, config.schema_path)

    return results_df, stitch_config, annotations_df
