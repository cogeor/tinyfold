# TinyFold

AlphaFold3-style binary protein-protein interaction (PPI) predictor.

## Installation

```bash
uv venv
source .venv/Scripts/activate  # Windows
uv pip install -e ".[dev]"
```

## Data Preparation

```bash
# Full pipeline: download DIPS-Plus + preprocess + split
python scripts/prepare_data.py --output-dir data/processed

# With local structure files
python scripts/prepare_data.py --output-dir data/processed --input-dir path/to/pdbs
```

## Testing

```bash
pytest tests/ -v
```
