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
# Overall authentication performance metrics
print_section("OVERALL AUTHENTICATION PERFORMANCE")

# Authentication rate
total_frames = len(results_df)
unlocked_frames = len(results_df[results_df['predicted_state'] == 'Unlocked'])
auth_rate = (unlocked_frames / total_frames) * 100

print(f"Total frames: {total_frames}")
print(f"Unlocked frames: {unlocked_frames}")
print(f"Authentication rate: {auth_rate:.2f}%")

# Risk score statistics
avg_risk_score = results_df['risk_score'].mean()
median_risk_score = results_df['risk_score'].median()
std_risk_score = results_df['risk_score'].std()

print(f"\nRisk Score Statistics:")
print(f"  Mean: {avg_risk_score:.4f}")
print(f"  Median: {median_risk_score:.4f}")
print(f"  Std Dev: {std_risk_score:.4f}")

# Distance statistics
avg_distance = results_df['distance'].mean()
median_distance = results_df['distance'].median()
std_distance = results_df['distance'].std()

print(f"\nDistance Statistics:")
print(f"  Mean: {avg_distance:.4f}")
print(f"  Median: {median_distance:.4f}")
print(f"  Std Dev: {std_distance:.4f}")

# State transitions (calculate how many times state changes)
state_changes = (results_df['predicted_state'] != results_df['predicted_state'].shift()).sum() - 1
print(f"\nState transitions: {state_changes}")
print(f"Average frames between transitions: {total_frames / max(state_changes, 1):.1f}")

# Create performance plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Authentication state distribution (pie chart)
state_counts = results_df['predicted_state'].value_counts()
axes[0, 0].pie(state_counts.values, labels=state_counts.index, autopct='%1.1f%%', startangle=90)
axes[0, 0].set_title('Authentication State Distribution')

# 2. Risk score distribution (histogram)
axes[0, 1].hist(results_df['risk_score'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(results_df['threshold'].iloc[0], color='red', linestyle='--', label=f"Threshold ({results_df['threshold'].iloc[0]})")
axes[0, 1].set_xlabel('Risk Score')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Risk Score Distribution')
axes[0, 1].legend()
axes[0, 1].grid(axis='y', alpha=0.3)

# 3. Distance distribution (histogram)
axes[1, 0].hist(results_df['distance'], bins=50, color='green', alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Distance')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Face Distance Distribution')
axes[1, 0].grid(axis='y', alpha=0.3)

# 4. Risk score over frames (line plot with different color per video)
for video_path in results_df['video_path'].unique():
    video_data = results_df[results_df['video_path'] == video_path]
    axes[1, 1].plot(video_data['frame'], video_data['risk_score'], alpha=0.7, linewidth=1.5, label=video_path.split('/')[-1].replace('.mp4', ''))

axes[1, 1].axhline(results_df['threshold'].iloc[0], color='black', linestyle='--', linewidth=2, label='Threshold')
axes[1, 1].set_xlabel('Frame')
axes[1, 1].set_ylabel('Risk Score')
axes[1, 1].set_title('Risk Score Over Time (by Video)')
axes[1, 1].legend(fontsize=8, loc='best')
axes[1, 1].grid(alpha=0.3)

plt.suptitle('Overall Authentication Performance', fontsize=16, y=0.995)
plt.tight_layout()

save_plot(fig, output_path, 'overall_performance.png')
plt.show()

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