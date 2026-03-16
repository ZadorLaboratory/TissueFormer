# TissueFormer

A transformer-based framework for classifying biological samples by aggregating groups of single cells. TissueFormer learns to combine information across cells within a spatial neighborhood or donor, outperforming classical baselines as group size grows.

The core idea: instead of classifying cells one at a time, sample a **group** of cells (by spatial proximity or shared donor), encode each cell with a pretrained BERT, and aggregate the group with a set transformer to produce a single prediction. Sweeping the group size reveals how much context the model needs.

## Applications

### Brain region annotation

Classify mouse cortical areas (290 regions) from BARseq spatial transcriptomics. Groups of cells are randomly sampled from cortical columns on a hexagonal grid, and the model predicts which brain area the column belongs to.

```
applications/brain_annotation/
├── data/            # Tokenization, CCF sorting, class weights
├── config/          # Hydra configs (model, data, training, benchmarks)
├── figures/         # Result plots and attention analysis
├── paper_figures/   # Publication figures
├── benchmarks.py    # RF, LR, CellCnn, scAGG, ScRAT baselines
└── analyze_attention.py  # Attention interpretability
```

See the [brain annotation data README](applications/brain_annotation/data/README.md) for dataset preparation.

### COVID-19 severity classification

Classify COVID-19 severity (control / mild / severe) from peripheral blood scRNA-seq across three cohorts (COMBAT, Ren, Stevenson). Cells are grouped by donor, and donor-stratified 5-fold CV ensures no donor leaks between splits.

```
applications/covid/
├── data/            # Tokenization + raw data standardization
├── config/          # Hydra configs
├── figures/         # Accuracy/AUROC vs group size curves
├── tests/           # 31 pytest tests
├── train.py         # TissueFormer training
├── benchmarks.py    # RF, LR, CellCnn, scAGG, ScRAT baselines
└── run_experiments.sh  # Full sweep (3 datasets x 5 folds x 10 group sizes)
```

See the [COVID README](applications/covid/README.md) for detailed pipeline instructions.

## Core library (`tissueformer/`)

Reusable components shared across applications:

| Module | Purpose |
|--------|---------|
| `model.py` | TissueFormer architecture: BERT encoder + set transformer head |
| `samplers.py` | `HexagonalSpatialGroupSampler` (spatial), `DonorGroupSampler` (donor-based), collators, trainers |
| `tokenizer.py` | `TranscriptomeTokenizer` for converting gene expression to discrete token sequences |
| `train.py` | Hydra-based training pipeline with W&B logging |
| `attention_analysis.py` | Collect and visualize attention weights for interpretability |
| `class_weights.py` | Class weight calculation for imbalanced labels |
| `benchmark_models/` | PyTorch re-implementations of CellCnn, scAGG, and ScRAT baselines |

## Installation

Requires Python >= 3.11 and a CUDA-capable GPU.

**With micromamba:**

```bash
git clone <repository-url>
cd brain-annotation
source create_env.sh <env-name>
```

**With uv:**

```bash
git clone <repository-url>
cd brain-annotation
source create_env_uv.sh
```

**With pip (editable install):**

```bash
pip install -e .
```

## Quick start

Both applications follow the same pattern: **tokenize** raw h5ad data, **train** TissueFormer, **run benchmarks**, and **plot results**.

```bash
# 1. Tokenize cells from an h5ad file
python applications/covid/data/tokenize_cells.py \
    --h5ad_path data/combat_processed.h5ad \
    --output_directory data --output_prefix combat_fold0 \
    --cv-fold 0 --raw-counts

# 2. Train TissueFormer
cd applications/covid
python train.py dataset_name=combat data.group_size=32 \
    data.dataset_path=data/combat_fold0.dataset

# 3. Run classical + deep learning benchmarks
python benchmarks.py dataset_name=combat data.group_size=32 \
    data.dataset_path=data/combat_fold0.dataset

# 4. Plot results
python figures/plot_results.py --results_dir outputs --output_dir figures
```

## Configuration

All configuration is managed with [Hydra](https://hydra.cc/). Each application has its own config directory:

```
config/
├── config.yaml              # Root config (seed, debug, dataset_name)
├── model/default.yaml       # Architecture (num_labels, set_layers, attention heads)
├── data/default.yaml        # Dataset path, group_size, label/donor keys
├── training/default.yaml    # HuggingFace TrainingArguments
├── wandb/default.yaml       # Weights & Biases project settings
├── benchmark_models/        # CellCnn, scAGG, ScRAT hyperparameters
└── local/default.yaml       # Machine-specific overrides (gitignored)
```

Machine-specific paths (e.g., pretrained model location) go in `config/local/default.yaml`, which is gitignored. Copy the example to get started:

```bash
cp config/local/default.yaml.example config/local/default.yaml
```

## Benchmarks

TissueFormer is compared against six baselines:

| Method | Type | Description |
|--------|------|-------------|
| Random Forest | Classical | GridSearchCV over pseudobulk or cell-type histogram features |
| Logistic Regression | Classical | L1/L2-regularized, balanced class weights |
| CellCnn | Deep learning | Conv1D + top-k pooling for rare subset detection |
| scAGG | Deep learning | MLP + mean pooling (NoGraph variant) |
| ScRAT | Deep learning | Transformer + mean pooling with sample mixup |
| Single-cell BERT | Deep learning | Per-cell BERT predictions aggregated by majority vote |

See the [benchmark models README](tissueformer/benchmark_models/README.md) for implementation details and deviations from original papers.

## Tests

```bash
python -m pytest applications/covid/tests/ -v
```

## Project structure

```
.
├── tissueformer/              # Core library (pip-installable)
│   ├── model.py               # TissueFormer model
│   ├── samplers.py            # Spatial and donor-based samplers
│   ├── tokenizer.py           # Gene expression tokenizer
│   ├── train.py               # Training pipeline
│   ├── attention_analysis.py  # Attention visualization
│   ├── class_weights.py       # Class weight utilities
│   └── benchmark_models/      # CellCnn, scAGG, ScRAT
├── applications/
│   ├── brain_annotation/      # 290-region mouse brain classification
│   └── covid/                 # 3-class COVID severity classification
├── pyproject.toml             # Package metadata
├── create_env.sh              # Micromamba environment setup
└── create_env_uv.sh           # uv environment setup
```

## License

This project is licensed under the MIT License.
