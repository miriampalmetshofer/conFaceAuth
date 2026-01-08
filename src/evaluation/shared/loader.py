"""Data loading utilities for evaluation studies."""
import json
from pathlib import Path
import pandas as pd


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
