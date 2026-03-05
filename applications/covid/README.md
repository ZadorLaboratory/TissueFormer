# COVID Severity Detection with TissueFormer

Classify COVID-19 severity (control / mild / severe) from peripheral blood scRNA-seq data using TissueFormer, with Random Forest and Logistic Regression baselines.

The central question: **does aggregating more cells from a single donor (larger group size) improve classification accuracy?** Results are compared across a sweep of group sizes (1–512) with 5-fold donor-stratified cross-validation.

## Directory Layout

```
applications/covid/
├── config/
│   ├── config.yaml                 # Root Hydra config
│   ├── data/default.yaml           # Data/split settings
│   ├── model/default.yaml          # TissueFormer architecture
│   ├── training/default.yaml       # HF TrainingArguments
│   ├── wandb/default.yaml          # Weights & Biases settings
│   └── local/                      # Machine-specific overrides (gitignored)
│       ├── default.yaml            # Your local path overrides
│       └── default.yaml.example    # Template (tracked)
├── data/
│   ├── tokenize_cells.py           # Step 1: tokenize h5ad → HF DatasetDict
│   └── raw_data_standardization.ipynb  # Step 0: standardize raw h5ad files
├── figures/
│   └── plot_results.py             # Manuscript-quality figures
├── tests/                          # 31 tests (pytest)
├── benchmarks.py                   # Step 3: RF + LR baselines
├── train.py                        # Step 2: TissueFormer training
└── run_experiments.sh              # Full experiment sweep
```

## Datasets

Three PBMC scRNA-seq datasets, standardized via `data/raw_data_standardization.ipynb`:

| Dataset | Cells | Donors (post-QC) | Classes |
|---------|-------|-------------------|---------|
| COMBAT | 836k | 85 | control / mild / severe |
| Ren et al. | 1.46M | 185 | control / mild / severe |
| Stevenson et al. | 647k | 99 | control / mild / severe |

Each h5ad is standardized to have `donor_id`, `label` (mild/severe/control), and `cell_type` columns. Donors with fewer than 1,000 cells are dropped.

## Local Configuration

Machine-specific paths (e.g. `bert_path_or_name`) are set in `config/local/default.yaml`, which is gitignored to avoid merge conflicts across machines. To set up:

```bash
cp config/local/default.yaml.example config/local/default.yaml
# Edit config/local/default.yaml with your machine's paths
```

Hydra loads this file automatically via `optional local: default` in `config.yaml`. If the file is missing, defaults from the tracked configs are used.

## Quick Start

```bash
cd applications/covid

# 0. Standardize raw data (one-time, in Jupyter)
#    -> produces data/{combat,ren,stevenson}_processed.h5ad

# 1. Tokenize one dataset + fold
python data/tokenize_cells.py \
    --h5ad_path data/combat_processed.h5ad \
    --output_directory data \
    --output_prefix combat_fold0 \
    --cv-fold 0 --raw-counts

# 2. Train TissueFormer (group_size=32)
python train.py \
    dataset_name=combat \
    data.group_size=32 \
    data.dataset_path=data/combat_fold0.dataset

# 3. Run benchmarks at the same group size
python benchmarks.py \
    dataset_name=combat \
    data.group_size=32 \
    data.dataset_path=data/combat_fold0.dataset

# 4. Plot results
python figures/plot_results.py --results_dir outputs --output_dir figures

# Or run the full sweep (all datasets × folds × group sizes):
bash run_experiments.sh
```

## Pipeline Details

### Step 1: Tokenization (`data/tokenize_cells.py`)

Tokenizes cells using Geneformer's `TranscriptomeTokenizer` and creates donor-stratified splits with `StratifiedGroupKFold(n_splits=5)`. All cells from a donor stay in the same split.

**Key CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--h5ad_path` | required | Path to processed h5ad |
| `--cv-fold` | `0` | Which fold (0–4) is the test set |
| `--raw-counts` | off | Retain raw counts in tokenized data |
| `--n-splits` | `5` | Number of CV folds |
| `--token-dictionary-file` | auto | Geneformer token dict (auto-discovered if omitted) |

**Outputs:** `{prefix}.dataset` (HF DatasetDict), `{prefix}_donor_splits.json`, `{prefix}_label_map.json`

### Step 2: Training (`train.py`)

Hydra-configured training with two paths:

- **`group_size > 1`**: Uses `GroupedDonorTrainer` with `DonorGroupSampler`. Cells are grouped by donor and fed through TissueFormer's BERT encoder + set transformer. The `SpatialGroupCollator` reshapes input to `(batch, group_size, seq_len)` with majority-vote labels.

- **`group_size = 1`**: Uses standard HF `Trainer` with `BertForSequenceClassification`. Single-cell predictions are aggregated to donor level by majority vote.

**Metrics:** accuracy, F1 (macro/weighted), AUROC (OvR macro for 3-class).

```bash
# Single-cell mode
python train.py data.group_size=1 model.pretrained_type=single-cell \
    training.remove_unused_columns=true

# Grouped mode (default)
python train.py data.group_size=64
```

### Step 3: Benchmarks (`benchmarks.py`)

Runs RF and LR with two feature types and two aggregation modes:

**Feature types:**
- `pseudobulk` — mean expression across cells per donor
- `cell_type_histogram` — normalized cell-type composition

**Aggregation modes:**
- `data.group_size=N` — subsample N cells per donor (matches TissueFormer)
- `data.group_size=all` — use all cells (ceiling benchmark)

Both classifiers use `GridSearchCV(cv=3)` with `class_weight="balanced"`:

| Classifier | Hyperparameter grid |
|------------|-------------------|
| Random Forest | `n_estimators`: [100, 200, 500], `max_depth`: [5, 10, 15, None], `max_features`: [sqrt, log2, 0.33] |
| Logistic Regression | `C`: [0.01, 0.1, 1, 10], `penalty`: [l1, l2], `solver`: saga |

### Step 4: Full Sweep (`run_experiments.sh`)

```
Datasets:          combat  ren  stevenson
Folds:             0 1 2 3 4
TissueFormer GS:   1 2 4 8 16 32 64 128 256 512
Benchmark GS:      2 4 8 16 32 64 128 256 512 all
```

Total: 150 TissueFormer runs + 150 benchmark runs (4 classifiers each).

### Step 5: Plotting (`figures/plot_results.py`)

- **Main figure:** accuracy and AUROC vs group_size (log₂ scale), one column per dataset, with error bars across folds. Whole-donor benchmarks shown as dashed horizontal lines.
- **Supplementary:** confusion matrices per method.

```bash
python figures/plot_results.py --results_dir outputs --output_dir figures
```

## Core Components (in `tissueformer/`)

**`DonorGroupSampler`** — A `torch.utils.data.Sampler` that groups cells by `donor_id`:
- Training: randomly picks donors, samples `group_size` cells per donor (with replacement if needed)
- Eval (`iterate_all_donors=True`): visits each donor exactly once
- `group_size=1`: standard random sampling

**`GroupedDonorTrainer`** — A `transformers.Trainer` subclass that wires up `DonorGroupSampler` and `SpatialGroupCollator(relative_positions=False)`. Falls back to standard `Trainer` for `group_size=1`.

## Tests

```bash
micromamba run -n geneformer2 python -m pytest tests/ -v
```

31 tests across 5 files:

| File | Tests | Coverage |
|------|-------|----------|
| `test_donor_sampler.py` | 10 | Donor purity, group sizes, replacement, eval coverage, reproducibility |
| `test_collator.py` | 7 | Output shapes, majority labels, padding, no positions |
| `test_tokenization.py` | 5 | Donor isolation, label integrity, output columns, metadata files |
| `test_benchmarks.py` | 6 | Feature aggregation correctness, classifier smoke tests |
| `test_end_to_end.py` | 2 | Full pipeline: tokenize → benchmark; tokenize → sampler |

## Config Reference

<details>
<summary>Hydra config keys</summary>

**Root:** `seed`, `debug`, `dataset_name`, `output_dir`, `run_test_set`

**`data`:** `dataset_path`, `group_size`, `validation_split`, `donor_key`, `label_key`, `cell_type_key`, `label_names`, `class_weights.{enabled,path}`

**`model`:** `pretrained_type` (bert_only/none/single-cell), `bert_path_or_name`, `num_labels`, `num_set_layers`, `set_hidden_size`, `num_attention_heads`, `dropout_prob`, `single_cell_vs_group_weight`, `relative_positions.enabled`

**`training`:** All HF `TrainingArguments` fields. Key defaults: `num_train_epochs=10`, `learning_rate=5e-5`, `per_device_train_batch_size=2048`, `fp16=true`, `remove_unused_columns=false`

**`wandb`:** `project` (covid-severity), `entity`, `group`, `name`, `tags`, `notes`

</details>
