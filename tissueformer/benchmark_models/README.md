# Benchmark Models

PyTorch re-implementations of three end-to-end trainable grouped-cell classification methods, used as baselines alongside Random Forest and Logistic Regression benchmarks.

## Methods

### CellCnn
Arvaniti & Claassen, "Sensitive detection of rare disease-associated cell subsets via representation learning", *Nature Communications* 2017.
[Paper](https://doi.org/10.1038/s41467-017-00095-5) | [Original code (Keras/TF)](https://github.com/eiriniar/CellCnn)

Conv1D with kernel_size=1 (per-cell linear transform) followed by custom top-k mean pooling that selects the k cells with highest activation per filter, enabling detection of rare cell subsets. Simple architecture: conv -> ReLU -> top-k pool -> dropout -> linear classifier.

### scAGG
Verlaan et al., "scAGG: single-cell gene expression analysis using graph-based aggregation", *CSBJ* 2025.
[Paper](https://doi.org/10.1016/j.csbj.2024.12.031) | [Original code](https://github.com/timoverlaan/scAGG)

Two-layer MLP cell encoder with ELU activations and mean pooling. The paper showed that this simple baseline (NoGraph variant) performs comparably to GAT/attention variants, so we implement only the base model.

### ScRAT
Mao et al., "ScRAT: a transformer-based method for single-cell multi-omic integration", *Bioinformatics* 2024.
[Paper](https://doi.org/10.1093/bioinformatics/btae231) | [Original code](https://github.com/yuzhenmao/ScRAT)

Transformer encoder with mean pooling over cells. Uses BCE loss (one-vs-rest) and cell-type-aware sample mixup augmentation for small sample sizes. Cosine LR schedule with warmup.

## Deviations from originals

- **CellCnn**: The original trains a 15-model ensemble with random hyperparameters and keeps the top 3. We use a single model with median hyperparameters (6 filters, 5% top-k pooling) for simplicity and reproducibility.
- **scAGG**: We implement only the base MLP + mean pooling variant (NoGraph), not the GAT/attention variants, since the paper showed they did not improve performance. The original uses Adagrad; we use Adam as it's more standard.
- **ScRAT**: For multi-class tasks (e.g., 3-class COVID severity), we use BCE with one-hot targets (as the paper does) rather than switching to cross-entropy.

## Usage

Enable via config flags:

```bash
# COVID
python benchmarks.py run_cellcnn=true run_scagg=true run_scrat=true

# Brain annotation
python benchmarks.py run_cellcnn=true run_scagg=true run_scrat=true
```

Override hyperparameters:

```bash
python benchmarks.py run_cellcnn=true benchmark_models.cellcnn.n_filters=8
```

## File structure

```
tissueformer/benchmark_models/
  __init__.py      # Package exports
  cellcnn.py       # CellCnn model + TopKPool
  scagg.py         # scAGG base model (MLP + mean pool)
  scrat.py         # ScRAT transformer + cell_type_mixup()
  trainer.py       # Shared training loop with early stopping
  data.py          # MILDataset, CroppedMILDataset, preprocessing
  README.md        # This file
```

## Hyperparameter defaults

All defaults match the original papers. Per-method configs are in:
- `applications/covid/config/benchmark_models/{cellcnn,scagg,scrat}.yaml`
- `applications/brain_annotation/config/benchmark_models/{cellcnn,scagg,scrat}.yaml`
