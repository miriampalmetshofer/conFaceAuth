# Evaluation Module

This module provides scripts for evaluating face authentication system performance from the authentication pipeline results.

## Overview

The evaluation scripts analyze the `results.csv` file produced by the batch processing pipeline (main.py) and generate:
- **Performance metrics** - FAR (False Accept Rate), FRR (False Reject Rate), mean genuine trust score, imposter/genuine lockout times
- **Interactive HTML timelines** - Trust score visualizations over time
- **Summary visualizations** - PNG images with key metrics tables
- **LaTeX tables** - Publication-ready metric tables
- **Statistical analysis** - Significance tests (controlled study only)

## Architecture

The evaluation module uses a **functional composition** approach with shared core modules and study-specific configurations:

```
evaluation/
├── shared/
│   ├── data_loader.py      # Loads results.csv and parses frame data
│   ├── metrics.py          # Calculates FAR, FRR, lockout times, etc.
│   ├── visualization.py    # Creates Plotly timelines and summary figures
│   └── reporting.py        # Prints metrics and generates LaTeX tables
├── controlled_study/
│   ├── evaluate.py         # Evaluation orchestration script
│   ├── significance_tests.py  # Statistical significance testing
│   └── output/             # Generated outputs (HTML, PNG, CSV)
└── in_the_wild/
    ├── evaluate.py         # Evaluation orchestration script
    ├── helper/
    │   └── annotation_validator.py  # Validates annotation JSON files
    └── output/             # Generated outputs (HTML, PNG, CSV)
```

### Study-Specific Differences

#### Controlled Study
- **Device types**: Mobile and Desktop
- **Scenarios**: Easy, Lighting, Angle
- **Analysis**: Metrics broken down by device and scenario, with significance tests

#### In-the-Wild Study
- **Device types**: Mobile only
- **Scenarios**: Not scenario-based (natural environmental conditions)
- **Additional**: Annotation validation for participant-provided labels

## Configuration

### Required Paths

Both evaluation scripts require configuring the following paths in the respective `evaluate.py` file:

#### Controlled Study (`src/evaluation/controlled_study/evaluate.py`)
```python
RESULTS_FOLDER = "data/controlled_study/_results_archive/V06"  # Update to your results folder
RESULTS_PATH = PROJECT_ROOT / RESULTS_FOLDER / "results.csv"
CONFIG_PATH = PROJECT_ROOT / RESULTS_FOLDER / "config.json"
OUTPUT_PATH = PROJECT_ROOT / "src/evaluation/controlled_study/output"
```

**Required input files:**
- `data/controlled_study/[your_results_folder]/results.csv` - Frame-by-frame authentication results from pipeline
- `data/controlled_study/[your_results_folder]/config.json` - Authentication configuration used during evaluation

**Note:** Update `RESULTS_FOLDER` in the script to point to your specific results folder.

#### In-the-Wild Study (`src/evaluation/in_the_wild/evaluate.py`)
```python
RESULTS_FOLDER = "data/in_the_wild/_results_archive/V10"  # Update to your results folder
RESULTS_PATH = PROJECT_ROOT / RESULTS_FOLDER / "results.csv"
CONFIG_PATH = PROJECT_ROOT / RESULTS_FOLDER / "config.json"
ANNOTATIONS_PATH = PROJECT_ROOT / "data/in_the_wild"
OUTPUT_PATH = PROJECT_ROOT / "src/evaluation/in_the_wild/output"
```

**Required input files:**
- `data/in_the_wild/[your_results_folder]/results.csv` - Frame-by-frame authentication results
- `data/in_the_wild/[your_results_folder]/config.json` - Authentication configuration
- `data/in_the_wild/*.json` - Optional annotation files for validation

**Note:** Update `RESULTS_FOLDER` in the script to point to your specific results folder.

## Usage

### Running Evaluations

#### Controlled Study
```bash
python src/evaluation/controlled_study/evaluate.py
```

**Generates:**
- `trust_timeline_all_videos.html` - Interactive timeline showing all videos
- `trust_timeline_by_device.html` - Separate timelines for mobile/desktop
- `trust_timeline_by_scenario.html` - Separate timelines for each scenario
- `trust_timeline_scenarios_aggregated.html` - Aggregated scenario comparison
- `summary.png` - Overall metrics summary visualization
- `metrics_tables.png` - Combined metrics tables
- `table_scenarios_aggregated.png` - Scenario metrics table
- Console output with detailed metrics and LaTeX tables

#### In-the-Wild Study
```bash
python src/evaluation/in_the_wild/evaluate.py
```

**Generates:**
- `trust_timeline_all_videos.html` - Interactive timeline showing all videos
- `trust_timeline_by_device.html` - Device-specific timelines
- `summary.png` - Overall metrics summary
- `table_results.png` - Results table
- Console output with annotation validation results, metrics, and LaTeX tables

**Note:** Additional analysis scripts exist in `in_the_wild/` (e.g., `analyze_annotation_trends.py`) that may generate other outputs.

## Output Files

All outputs are saved to the respective `output/` directories within each study folder. The HTML timelines are interactive Plotly visualizations that allow zooming, panning, and hovering to see detailed frame-level information.



