# %%
"""Evaluation script for in-the-wild study."""
import json
import pandas as pd
import matplotlib.pyplot as plt

from evaluation.evaluation_utils import (
    get_base_path, load_results_csv, save_plot, print_section, load_annotation_schema
)

# %%
# Setup paths
base_path = get_base_path()
results_path = base_path / "data/in_the_wild/results.csv"
annotations_path = base_path / "data/annotations"
output_path = base_path / "src/evaluation/in_the_wild/output"
schema_path = base_path / "src/evaluation/in_the_wild/annotation_schema.json"

print_section("SETUP")
print(f"Base path: {base_path}")
print(f"Results path: {results_path}")
print(f"Annotations path: {annotations_path}")
print(f"Output path: {output_path}")
print(f"Schema path: {schema_path}")

# %%
print_section("LOADING ANNOTATION SCHEMA")
annotation_schema = load_annotation_schema(schema_path)

# %%
# Load results
print_section("LOADING RESULTS")
results_df = load_results_csv(results_path)

# %%
# Load annotations
print_section("LOADING ANNOTATIONS")
annotations_list = []
for json_file in annotations_path.glob("*.json"):
    with open(json_file, 'r') as f:
        annotation = json.load(f)

        # Create flat dictionary for this annotation
        row = {}

        # Add metadata fields
        for field in annotation_schema['metadata_fields']:
            row[field] = annotation.get(field, '')

        annot_data = annotation.get('annotations', {})

        # Store all annotation fields directly (lists or single values)
        all_annotation_fields = annotation_schema['list_fields'] + annotation_schema['single_choice_fields']
        for key in all_annotation_fields:
            if key in annot_data:
                row[key] = annot_data[key]

        annotations_list.append(row)

# Create DataFrame
annotations_df = pd.DataFrame(annotations_list)
print(f"Loaded {len(annotations_df)} annotations")
print(f"Shape: {annotations_df.shape}")
print(f"Columns: {list(annotations_df.columns)}")
print(f"\nAnnotations DataFrame:")
print(annotations_df.to_string())

# %%
print_section("GENERATING PLOTS")

# Get columns from schema
list_cols = annotation_schema['list_fields']
categorical_cols = annotation_schema['single_choice_fields']

# Combine the list-based and categorical annotation fields,
# but only keep the ones that actually exist in the annotations_df.
cols_to_plot = [col for col in list_cols + categorical_cols if col in annotations_df.columns]

num_cols = 3
num_rows = (len(cols_to_plot) + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 4 * num_rows))
axes = axes.flatten()

for ax, col in zip(axes, cols_to_plot):
    data = (
        annotations_df[col].explode()   # handle list-type
        if col in list_cols
        else annotations_df[col]
    )

    data.value_counts().plot(kind='bar', ax=ax)

    ax.set_title(col.replace("_", " ").title())
    ax.set_ylabel("Count")
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

# Hide any unused subplots
for ax in axes[len(cols_to_plot):]:
    ax.set_visible(False)

plt.suptitle("In-the-Wild Study: Annotation Distribution", fontsize=16, y=0.995)
plt.tight_layout()

save_plot(fig, output_path, 'annotation_distribution.png')
print_section("DONE")

plt.show()

# %%