# Evaluation Module

This module provides a clean, unified framework for evaluating face authentication system performance across different study types (controlled and in-the-wild).

## Architecture

The evaluation module uses a **functional composition** approach with shared core modules and study-specific configurations:

```
evaluation/
├── config.py              # StudyConfig dataclass and base path utilities
├── loader.py              # Data loading (CSV, annotations, schemas, configs)
├── processor.py           # Data processing (segment categorization, grouping)
├── metrics.py             # Metrics calculation with flexible grouping
├── visualizer.py          # Interactive Plotly visualizations with grouping
├── reporter.py            # Unified reporting and output generation
├── controlled_study/
│   ├── config.py          # Controlled study configuration
│   ├── evaluate.py        # Evaluation script (pure Python)
│   └── annotation_schema.json
└── in_the_wild/
    ├── config.py          # Wild study configuration
    ├── evaluate.py        # Evaluation script (pure Python)
    └── annotation_schema.json
```

### Study-Specific Differences

#### Controlled Study
- **Device types**: Mobile and Desktop
- **Scenarios**: Easy, Lighting, Angle

#### In-the-Wild Study
- **Device types**: Mobile only
- **Scenarios**: Multiple environmental factors

## Usage

### Running Evaluations

#### Controlled Study
```bash
cd /path/to/conFaceAuth
python src/evaluation/controlled_study/evaluate.py
```

Output location: `src/evaluation/controlled_study/output/`

#### In-the-Wild Study
```bash
cd /path/to/conFaceAuth
python src/evaluation/in_the_wild/evaluate.py
```

Output location: `src/evaluation/in_the_wild/output/`

## Core Modules

### config.py
```python
from evaluation.config import StudyConfig, get_base_path

# Create study configuration
config = StudyConfig(
    name="in_the_wild",
    base_path=get_base_path(),
    results_filename="results.csv",
    config_filename="in_the_wild.json",
    device_types=["mobile"],
    grouping_dimensions=["environment", "lighting"],
    has_annotations=True
)
```

### loader.py
```python
from evaluation.loader import load_study_data

# Load all study data at once
results_df, stitch_config, annotations_df = load_study_data(config)
```

### processor.py
```python
from evaluation.processor import categorize_frames, add_grouping_columns

# Categorize frames by segment type using source_type field
results_df = categorize_frames(df)

# Add grouping columns from annotations
results_df = add_grouping_columns(results_df, ['device', 'scenario'], annotations_df)
```

### metrics.py
```python
from evaluation.metrics import (
    calculate_segment_metrics,
    calculate_grouped_metrics,
    calculate_device_comparison
)

# Calculate overall segment metrics
segment_metrics = calculate_segment_metrics(df)
# Returns: {'genuine': {...}, 'imposter': {...}}

# Calculate metrics by any dimension
scenario_metrics = calculate_grouped_metrics(df, 'scenario')
# Returns: {'easy': {'genuine': {...}, 'imposter': {...}}, 'lighting': {...}}

# Device-specific comparison
device_metrics = calculate_device_comparison(df, 'device')
```

### visualizer.py
```python
from evaluation.visualizer import (
    create_risk_score_timeline,
    create_grouped_comparison_plot,
    create_device_breakdown_plot
)

# Interactive timeline with hover details
fig = create_risk_score_timeline(df, threshold=0.95)

# Generic grouped comparison (works for any dimension)
fig = create_grouped_comparison_plot(scenario_metrics, 'Scenario')

# Device-specific breakdown
fig = create_device_breakdown_plot(device_metrics)
```

### reporter.py
```python
from evaluation.reporter import (
    print_study_header,
    print_segment_analysis,
    print_grouped_analysis
)

# Print formatted reports
print_study_header(config)
print_segment_analysis(genuine_metrics, imposter_metrics)
print_grouped_analysis(device_metrics, 'Device')
```

## Configuration

Evaluation scripts automatically load stitch configuration from:
- `configs/in_the_wild.json`
- `configs/controlled_study.json`

Required config section:
```json
{
  "imposter_creation": {
    "fps": 25,
    "genuine_user_seconds": 20.0,
    "black_screen_seconds": 2.0,
    "impostor_seconds": 20.0
  }
}
```

## Output Files

### Interactive HTML Visualizations
- `risk_score_timeline.html` - Timeline with all videos (hover for details)
- `device_comparison.html` - Device performance comparison (controlled only)
- `device_breakdown.html` - Detailed device breakdown (controlled only)
- `scenario_comparison.html` - Scenario comparison (controlled only)
- `environment_comparison.html` - Environment comparison (wild only)

### Static PNG Reports
- `summary.png` - 2x2 summary overview (controlled)
- `comprehensive_analysis.png` - 3x3 comprehensive analysis (wild)
- `annotation_distribution.png` - Annotation analysis (if annotations exist)

## Metrics Explained

### True Accept Rate (TAR)
Percentage of genuine user frames correctly authenticated (unlocked)

### False Reject Rate (FRR)
Percentage of genuine user frames incorrectly rejected (locked)

### True Reject Rate (TRR)
Percentage of imposter frames correctly rejected (locked)

### False Accept Rate (FAR)
Percentage of imposter frames incorrectly authenticated (unlocked) - **Security Risk!**

### Equal Error Rate (EER)
Approximated as (FAR + FRR) / 2 - Lower is better

### Grouped Metrics
All segment metrics can be calculated per group (device, scenario, environment, etc.):
```python
device_metrics = {
    'mobile': {
        'genuine': {'true_accept_rate': 95.2, 'false_reject_rate': 4.8, ...},
        'imposter': {'true_reject_rate': 92.1, 'false_accept_rate': 7.9, ...}
    },
    'desktop': {...}
}
```

## Creating a New Study

1. Create study config:
```python
# src/evaluation/my_study/config.py
from evaluation.config import StudyConfig, get_base_path

MY_STUDY_CONFIG = StudyConfig(
    name="my_study",
    base_path=get_base_path(),
    results_filename="results.csv",
    config_filename="my_study.json",
    device_types=["mobile", "desktop"],
    grouping_dimensions=["custom_dimension"],
    has_annotations=True
)
```

2. Create evaluation script using shared modules:
```python
# src/evaluation/my_study/evaluate.py
from evaluation.my_study.config import MY_STUDY_CONFIG
from evaluation.loader import load_study_data
from evaluation.processor import categorize_frames, add_grouping_columns
from evaluation.metrics import calculate_segment_metrics, calculate_grouped_metrics
from evaluation.reporter import print_study_header, print_segment_analysis

def main():
    config = MY_STUDY_CONFIG
    print_study_header(config)

    results_df, stitch_config, annotations_df = load_study_data(config)
    results_df = categorize_frames(results_df)
    results_df = add_grouping_columns(results_df, config.grouping_dimensions, annotations_df)

    segment_metrics = calculate_segment_metrics(results_df)
    print_segment_analysis(segment_metrics['genuine'], segment_metrics['imposter'])

    # Add custom analysis...

if __name__ == '__main__':
    main()
```

## Benefits of This Architecture

1. **DRY (Don't Repeat Yourself)**: Shared code eliminates duplication
2. **Flexible**: Easy to add new grouping dimensions or study types
3. **Clean**: Clear separation of concerns across modules
4. **Maintainable**: Changes propagate automatically to all studies
5. **Type-Safe**: Comprehensive type hints catch errors early
6. **Consistent**: Both studies follow the same patterns
7. **Extensible**: Simple to create new study types
