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

## Data format: raw expression vs. tokens

TissueFormer tokenizes each cell's gene expression into a discrete token sequence (via `TranscriptomeTokenizer`) and processes these tokens with a transformer that operates on the vocabulary of gene-expression bins. The benchmark models here skip tokenization entirely and operate on **raw continuous expression values** directly, organized as "bags" of cells.

The data pipeline (`data.py`) provides two dataset classes for this:

- **`MILDataset`**: Each item is a bag of cells `(n_cells, n_genes)` with a single label. For COVID, each bag is one donor's cells; for brain annotation, each bag is a spatial group. Bags can be variable-length (padded at collation) or fixed-size (random crop/upsample to `cells_per_sample`).
- **`CroppedMILDataset`**: Draws multiple fixed-size random crops per sample (used by ScRAT, which takes 20 crops/patient during training and 50 during evaluation, then majority-votes across crops).

Per-method preprocessing is applied to the raw expression matrix before batching:

| Method | Preprocessing | Input shape per bag |
|--------|--------------|-------------------|
| CellCnn | z-score (StandardScaler fit on train) | `(200, n_genes)` fixed |
| scAGG | CP10k + log1p, top-1000 HVGs by variance | `(variable, n_genes)` padded |
| ScRAT | raw counts (no normalization) | `(500, n_genes)` fixed crop |

The collate function `mil_collate_fn` pads variable-length bags to the longest bag in the batch and returns a boolean mask `(batch, max_cells)` where `True` = padded. All three models accept this mask and use it for mask-aware pooling.

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
