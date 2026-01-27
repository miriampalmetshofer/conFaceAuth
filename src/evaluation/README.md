# Evaluation Module

This module provides a unified framework for evaluating face authentication system performance across different study types (controlled and in-the-wild).

## Architecture

The evaluation module uses a **functional composition** approach with shared core modules and study-specific configurations:

```
evaluation/
├── shared/
│   ├── data_loader.py     
│   ├── metrics.py        
│   ├── visualization.py  
│   └── reporting.py     
├── controlled_study/
│   ├── evaluate.py        # Evaluation script (pure Python)
│   └── output/            # Generated outputs
└── in_the_wild/
    ├── evaluate.py        # Evaluation script (pure Python)
    ├── annotation_validator.py  # Annotation validation
    └── output/            # Generated outputs
```

### Study-Specific Differences

#### Controlled Study
- **Device types**: Mobile and Desktop
- **Scenarios**: Easy, Lighting, Angle

#### In-the-Wild Study
- **Device types**: Mobile only
- **Scenarios**: Multiple environmental factors
- **Additional**: Annotation validation

## Configuration

### Required Paths

Both evaluation scripts require configuring the following paths in the respective `evaluate.py` file:

#### Controlled Study (`src/evaluation/controlled_study/evaluate.py`)
```python
RESULTS_FOLDER = "data/controlled_study/"           # Folder containing results.csv
RESULTS_PATH = PROJECT_ROOT / RESULTS_FOLDER / "results.csv"
CONFIG_PATH = PROJECT_ROOT / RESULTS_FOLDER / "config.json"
OUTPUT_PATH = PROJECT_ROOT / "src/evaluation/controlled_study/output"
```

**Required input files:**
- `data/controlled_study/results.csv` - Frame-by-frame authentication results from pipeline
- `data/controlled_study/config.json` - Authentication configuration used during evaluation

#### In-the-Wild Study (`src/evaluation/in_the_wild/evaluate.py`)
```python
RESULTS_FOLDER = "data/in_the_wild/_results_archive/little_less_loose"  # Folder containing results.csv
RESULTS_PATH = PROJECT_ROOT / RESULTS_FOLDER / "results.csv"
CONFIG_PATH = PROJECT_ROOT / RESULTS_FOLDER / "config.json"
ANNOTATIONS_PATH = PROJECT_ROOT / "data/in_the_wild"
OUTPUT_PATH = PROJECT_ROOT / "src/evaluation/in_the_wild/output"
```

**Required input files:**
- `data/in_the_wild/[your_results_folder]/results.csv` - Frame-by-frame authentication results
- `data/in_the_wild/[your_results_folder]/config.json` - Authentication configuration
- `data/in_the_wild/*.json` - Optional annotation files for validation

## Usage

### Running Evaluations

#### Controlled Study
```bash
python src/evaluation/controlled_study/evaluate.py
```


#### In-the-Wild Study
```bash
python src/evaluation/in_the_wild/evaluate.py
```



