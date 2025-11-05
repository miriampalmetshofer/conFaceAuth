import os
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def print_section(title: str, width: int = 80):
    """Print a section header with separator lines."""
    print("\n" + "="*width)
    print(title)
    print("="*width)


def load_annotation_schema(schema_path: str | Path) -> dict:
    """Load annotation schema from JSON file."""
    with open(schema_path, 'r') as f:
        schema = json.load(f)

    print(f"Annotation schema loaded from {Path(schema_path).name}")
    print(f"  List fields: {schema['list_fields']}")
    print(f"  Single choice fields: {schema['single_choice_fields']}")

    return schema


def get_base_path():
    """Get project base path, handling both script and interactive execution."""
    try:
        # When running as script
        base_path = Path(__file__).parent.parent.parent
    except NameError:
        # When running interactively (cells)
        base_path = Path.cwd()
        if base_path.name == "evaluation":
            base_path = base_path.parent.parent
        elif base_path.name == "src":
            base_path = base_path.parent

    return base_path


def load_results_csv(results_path: str | Path) -> pd.DataFrame:
    """Load results CSV file."""
    df = pd.read_csv(results_path)
    print(f"Loaded {len(df)} result rows from {Path(results_path).name}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head().to_string())

    return df


def save_plot(fig: plt.Figure, output_folder: str | Path, filename: str, dpi: int = 300):
    """Save matplotlib figure to file."""
    output_folder = Path(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    output_file = output_folder / filename
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    return output_file