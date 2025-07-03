# Spatial Transcriptomic Brain Area Annotation Using Transformers

A deep learning framework for annotating brain regions using spatial transcriptomic data. This project implements a hierarchical transformer architecture that leverages spatial proximity of cells to improve brain area classification accuracy.

If you are interested in just the TissueFormer architecture, please consult the source code in `model.py`.

## Overview

This framework addresses a fundamental challenge in neuroscience: accurately mapping the spatial organization of cell types across brain regions. By combining single-cell RNA sequencing data with spatial information, we developed a transformer-based model that groups spatially proximate cells and predicts their anatomical locations within the brain.

### Key Features

- **Hierarchical Transformer Architecture**: Combines BERT-based gene expression encoding with set transformer layers for spatial group processing
- **Spatial Grouping Strategies**: Implements both hexagonal grid and k-nearest neighbor sampling for creating spatially coherent cell groups
- **Multi-scale Learning**: Learns representations at both single-cell and spatial group levels
- **Comprehensive Benchmarking**: Includes comparisons against Random Forest and Logistic Regression baselines
- **Flexible Configuration**: Uses Hydra for experiment management and hyperparameter tuning

## Installation

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU 
- Micromamba or Conda

### Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd brain-annotation
```

2. Create and activate the environment:
```bash
# Create environment named 'spatial_transformer' (or your preferred name)
source create_env.sh spatial_transformer
```

This will install all required dependencies including:
- PyTorch with CUDA support
- Transformers, Datasets (HuggingFace)
- Hydra for configuration management
- Scientific computing libraries (NumPy, SciPy, scikit-learn)
- Visualization tools (Matplotlib, Seaborn)
- Weights & Biases for experiment tracking

## Data Preparation

### Input Data Format

The pipeline expects single-cell RNA-seq data in `.h5ad` (AnnData) format with:
- **Gene expression matrix**: Raw counts in `adata.X`
- **Spatial coordinates**: 3D coordinates in `adata.obsm['CCF_streamlines']`
- **Cell type annotations**: In `adata.obs['H3_type']` (optional, for analysis)
- **Area annotations**: In `adata.obs['CCFano']` (Allen Brain Atlas annotation IDs)

### Data Processing Pipeline

1. **Convert MATLAB files to H5AD format** (if starting from .mat files):
```bash
python data/mat_to_h5.py \
    --input_dir /path/to/mat/files \
    --output_dir /path/to/h5ad/output \
    --force  # Overwrite existing files
```

2. **Tokenize gene expression data**:
```bash
python data/tokenize_cells.py \
    --h5ad_data_directory /path/to/h5ad/files \
    --output_directory /path/to/tokenized/output \
    --output_prefix train_test_barseq \
    --cv-fold 0 \  # Cross-validation fold (0-3 for train, >=4 for test set)
    --raw-counts  # Include raw counts for benchmarking
```

The tokenization process:
- Normalizes gene expression by total counts per cell
- Ranks genes by expression level
- Converts to token sequences compatible with transformer models
- Adds spatial coordinates and metadata

3. **Calculate class weights** (optional, for imbalanced datasets):
```bash
python data/calculate_class_weights.py \
    data.dataset_path=/path/to/tokenized/dataset \
    data.label_key=area_label \
    weighting.method=balanced
```

## Model Architecture

### Hierarchical Transformer Design

The model consists of three main components:

1. **BERT Encoder**: Processes tokenized gene expression for each cell
2. **Spatial Grouping**: Groups nearby cells using configurable strategies
3. **Set Transformer**: Aggregates information from spatial groups

```
Input: Group of spatially proximate cells
  ↓
BERT encoding (per cell)
  ↓
Position encoding (optional)
  ↓
Set Transformer layers
  ↓
Mean pooling
  ↓
Classification head
  ↓
Output: Brain area prediction
```

### Spatial Grouping Strategies

#### Hexagonal Grid Sampling
- Tessellates the tissue with hexagonal grid
- Ensures uniform spatial coverage
- Configurable hex size based on cell density

#### Random Spatial Sampling
- Uses KD-tree for efficient nearest neighbor search
- Adaptively expands search radius when needed
- Suitable for irregular tissue shapes

## Training

### Basic Training

Train a model with default settings:
```bash
python train.py
```

### Advanced Configuration

Customize training using Hydra's override syntax:

```bash
# Train with hexagonal spatial grouping
python train.py \
    data.sampling.strategy=hex \
    data.group_size=32 \
    training.learning_rate=1e-4 \
    training.num_train_epochs=15
```

### Key Configuration Options

#### Model Configuration (`config/model/default.yaml`)
- `pretrained_type`: Model initialization strategy
  - `"none"`: Train from scratch
  - `"bert_only"`: Use pretrained BERT, train set transformer
  - `"full"`: Load complete pretrained model
  - `"single-cell"`: Single-cell baseline without spatial grouping
- `num_set_layers`: Number of set transformer layers (default: 4)
- `set_hidden_size`: Hidden dimension for set transformer (default: 768)
- `relative_positions.enabled`: Use relative position encoding

#### Data Configuration (`config/data/default.yaml`)
- `dataset_path`: Path to tokenized dataset
- `group_size`: Number of cells per spatial group (default: 32)
- `sampling.strategy`: `"hex"` or `"random"`
- `sampling.hex_scaling`: Scaling factor for hexagon size
- `sampling.max_radius_expansions`: Maximum search radius expansions

#### Training Configuration (`config/training/default.yaml`)
- `num_train_epochs`: Number of training epochs
- `per_device_train_batch_size`: Batch size (divided by group_size)
- `learning_rate`: Learning rate (default: 1e-4)
- `warmup_ratio`: Fraction of steps for learning rate warmup

### Multi-GPU Training

For distributed training across multiple GPUs:
```bash
accelerate launch --multi_gpu --num_processes 4 train.py
```

## Evaluation and Benchmarking

### Model Evaluation

The training script automatically evaluates on validation and test sets. Results include:
- Per-class precision, recall, and F1 scores
- Confusion matrices
- Spatial distribution of predictions

### Benchmark Against Classical Methods

Compare against Random Forest and Logistic Regression baselines:

```bash
python benchmarks.py \
    run_bulk_expression_rf=true \
    run_bulk_expression_lr=true \
    run_h3type_rf=true \
    run_h3type_lr=true
```

Benchmark features:
- **Bulk expression**: Average gene expression per spatial group
- **H3 type composition**: Histogram of cell types per group

### Visualization

Visualize hexagonal grid sampling:
```bash
python train.py \
    data.sampling.strategy=hex \
    visualize_hex_grid=true
```

## Advanced Usage

### Custom Spatial Constraints

Group cells within specific categories (e.g., same animal or cell type):
```bash
python train.py \
    data.sampling.group_within_keys=[animal_name,H3_type]
```

### Position Encoding Strategies

Enable relative position encoding:
```bash
python train.py \
    model.relative_positions.enabled=true \
    model.relative_positions.encoding_type=sinusoidal \
    model.relative_positions.encoding_dim=48
```

### Class Weighting

Handle imbalanced datasets:
```bash
python train.py \
    data.class_weights.enabled=true \
    data.class_weights.path=/path/to/weights.npy
```

### Experiment Tracking

Configure Weights & Biases logging:
```bash
python train.py \
    wandb.project=my_project \
    wandb.entity=my_team \
    wandb.name=experiment_name
```

## Output Files

Training produces several output files:

- `model/`: Trained model checkpoints
- `trainer_state.json`: Training state for resuming
- `all_results.json`: Evaluation metrics
- `test_brain_predictions_cells.npy`: Test set predictions with metadata
- `hex_grid_sampling_gs_32.png`: Visualization of spatial grouping (if enabled)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `per_device_train_batch_size`
   - Decrease `data.group_size`
   - Enable gradient checkpointing

2. **Slow Training**
   - Increase `dataloader_num_workers`
   - Use `data.sampling.strategy=hex` (faster than random)
   - Enable mixed precision: `fp16=true`

3. **Poor Performance**
   - Increase `training.num_train_epochs` (15-20 recommended)
   - Tune `data.group_size` based on cell density
   - Enable class weighting for imbalanced data

### Debug Mode

For quick iteration during development:
```bash
python train.py debug=true
```

This limits dataset size and disables wandb logging.

## Citation

If you use this code in your research, please cite:
```bibtex
[Citation information to be added]
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

This work builds upon:
- Geneformer for gene expression tokenization
- HuggingFace Transformers for model implementations
- Allen Brain Atlas for anatomical reference standards

For questions or issues, please open a GitHub issue or contact the maintainers.