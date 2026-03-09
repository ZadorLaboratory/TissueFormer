"""
COVID severity benchmarks.

Classical methods:
  - Random Forest / Logistic Regression on pseudobulk or cell-type histograms

Deep learning methods (end-to-end trainable on raw cells):
  - CellCnn  (Arvaniti & Claassen, Nat Commun 2017)
  - scAGG    (Verlaan et al., CSBJ 2025)
  - ScRAT    (Mao et al., Bioinformatics 2024)

Uses the same donor-stratified CV splits as TissueFormer.

For any group_size < "all", reports both group-level metrics (how well does the
model classify from gs cells?) and donor-level metrics (aggregated across
ceil(n_cells/gs) non-overlapping groups per donor).
"""

import math
import os
import json
import warnings
from typing import Dict, Tuple, Optional

import hydra
import wandb
import numpy as np
import anndata as ad
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
import scipy.sparse


def setup_wandb(cfg: DictConfig):
    """Initialize W&B logging."""
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=f"benchmark_{cfg.wandb.name}",
        group=cfg.wandb.group,
        tags=cfg.wandb.tags + ["benchmark"],
        notes=cfg.wandb.notes,
        config=OmegaConf.to_container(cfg, resolve=True),
    )


# ---------- Feature aggregation ----------

def _get_label_map(adata):
    """Determine label map from data."""
    unique_labels = set(adata.obs["label"].unique())
    if "control" in unique_labels:
        return {"control": 0, "mild": 1, "severe": 2}
    return {"mild": 0, "severe": 1}


def _compute_feature(donor_adata, feature_type, all_cell_types=None):
    """Compute a single feature vector from a subset of cells."""
    if feature_type == "pseudobulk":
        X = donor_adata.X
        if scipy.sparse.issparse(X):
            X = X.toarray()
        return np.mean(X, axis=0).flatten()
    elif feature_type == "cell_type_histogram":
        cell_types = donor_adata.obs["cell_type"].values
        type_to_idx = {t: i for i, t in enumerate(all_cell_types)}
        hist = np.zeros(len(all_cell_types))
        for ct in cell_types:
            hist[type_to_idx[ct]] += 1
        total = hist.sum()
        if total > 0:
            hist /= total
        return hist
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")


def aggregate_donor_features(
    h5ad_path: str,
    donor_ids: list,
    feature_type: str = "pseudobulk",
    group_size: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate cell-level data to donor-level features (one group per donor).

    Used for training: each donor contributes one feature vector from a single
    group of group_size cells (or all cells if group_size is None).
    """
    adata = ad.read_h5ad(h5ad_path)

    # Filter to requested donors
    # Compute all_cell_types from full dataset before filtering to ensure
    # consistent feature dimensions across train/test splits
    all_cell_types = sorted(adata.obs["cell_type"].unique()) if feature_type == "cell_type_histogram" else None

    mask = adata.obs["donor_id"].isin(donor_ids)
    adata = adata[mask]

    rng = np.random.RandomState(seed)
    unique_donors = sorted(set(donor_ids) & set(adata.obs["donor_id"].unique()))

    features_list = []
    labels_list = []
    donors_list = []

    label_map = _get_label_map(adata)

    for donor in unique_donors:
        donor_mask = adata.obs["donor_id"] == donor
        donor_adata = adata[donor_mask]
        donor_label = label_map[donor_adata.obs["label"].iloc[0]]
        n_cells = len(donor_adata)

        if group_size is not None and group_size < n_cells:
            idx = rng.choice(n_cells, size=group_size, replace=False)
            donor_adata = donor_adata[idx]
        elif group_size is not None and group_size > n_cells:
            idx = rng.choice(n_cells, size=group_size, replace=True)
            donor_adata = donor_adata[idx]

        feat = _compute_feature(donor_adata, feature_type, all_cell_types)
        features_list.append(feat)
        labels_list.append(donor_label)
        donors_list.append(donor)

    return np.array(features_list), np.array(labels_list), np.array(donors_list)


def aggregate_donor_features_multi(
    h5ad_path: str,
    donor_ids: list,
    feature_type: str = "pseudobulk",
    group_size: int = 64,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create ceil(n_cells / group_size) non-overlapping groups per donor.

    Shuffles cells, chunks into non-overlapping groups of group_size.
    Last chunk is padded with replacement sampling if < group_size cells.
    Mirrors DonorGroupSampler._sample_all_groups_from_donor in samplers.py.

    Returns:
        features: (total_groups, n_features) array
        labels: (total_groups,) integer labels
        donors: (total_groups,) donor ID array
    """
    adata = ad.read_h5ad(h5ad_path)

    # Compute all_cell_types from full dataset before filtering to ensure
    # consistent feature dimensions across train/test splits
    all_cell_types = sorted(adata.obs["cell_type"].unique()) if feature_type == "cell_type_histogram" else None

    mask = adata.obs["donor_id"].isin(donor_ids)
    adata = adata[mask]

    rng = np.random.RandomState(seed)
    unique_donors = sorted(set(donor_ids) & set(adata.obs["donor_id"].unique()))

    features_list = []
    labels_list = []
    donors_list = []

    label_map = _get_label_map(adata)

    for donor in unique_donors:
        donor_mask = adata.obs["donor_id"] == donor
        donor_adata = adata[donor_mask]
        donor_label = label_map[donor_adata.obs["label"].iloc[0]]
        n_cells = len(donor_adata)

        # Shuffle and chunk into non-overlapping groups
        shuffled_idx = rng.permutation(n_cells)
        n_groups = math.ceil(n_cells / group_size)

        for g in range(n_groups):
            start = g * group_size
            end = start + group_size
            chunk_idx = shuffled_idx[start:end]

            if len(chunk_idx) < group_size:
                # Pad with replacement sampling from this donor
                pad_size = group_size - len(chunk_idx)
                pad_idx = rng.choice(n_cells, size=pad_size, replace=True)
                chunk_idx = np.concatenate([chunk_idx, pad_idx])

            group_adata = donor_adata[chunk_idx]
            feat = _compute_feature(group_adata, feature_type, all_cell_types)
            features_list.append(feat)
            labels_list.append(donor_label)
            donors_list.append(donor)

    return np.array(features_list), np.array(labels_list), np.array(donors_list)


# ---------- Prediction aggregation ----------

def aggregate_to_donor(predictions, labels, probs, donor_ids, n_classes):
    """Aggregate group predictions to donor level.

    Returns dict with 'majority_vote' and 'mean_probs' sub-dicts,
    each containing 'predictions', 'labels', 'probs', 'donor_ids'.
    """
    unique_donors = np.unique(donor_ids)
    mv_preds = []
    mp_probs = []
    donor_labels = []

    for donor in unique_donors:
        mask = donor_ids == donor
        # Majority vote
        mv_preds.append(
            np.bincount(predictions[mask], minlength=n_classes).argmax()
        )
        # Mean probabilities
        mp_probs.append(probs[mask].mean(axis=0))
        donor_labels.append(labels[mask][0])

    mv_preds = np.array(mv_preds)
    mp_probs = np.array(mp_probs)
    mp_preds = np.argmax(mp_probs, axis=-1)
    donor_labels = np.array(donor_labels)

    return {
        "majority_vote": {
            "predictions": mv_preds,
            "labels": donor_labels,
            "probs": mp_probs,  # use mean probs for AUROC even with majority vote preds
            "donor_ids": unique_donors,
        },
        "mean_probs": {
            "predictions": mp_preds,
            "labels": donor_labels,
            "probs": mp_probs,
            "donor_ids": unique_donors,
        },
    }


# ---------- Classifier pipelines ----------

RF_PARAM_GRID = {
    "n_estimators": [100, 200, 500],
    "max_depth": [5, 10, 15, None],
    "max_features": ["sqrt", "log2", 0.33],
}

LR_PARAM_GRID = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l1", "l2"],
}


def create_classifier(classifier_type: str, seed: int = 42):
    """Create a classifier with GridSearchCV for hyperparameter tuning."""
    if classifier_type == "random_forest":
        base_clf = RandomForestClassifier(
            class_weight="balanced", random_state=seed, n_jobs=-1
        )
        return GridSearchCV(
            base_clf, RF_PARAM_GRID, cv=3, scoring="accuracy",
            n_jobs=-1, refit=True
        )
    elif classifier_type == "logistic_regression":
        base_clf = LogisticRegression(
            solver="saga", class_weight="balanced", max_iter=2000, random_state=seed
        )
        return GridSearchCV(
            base_clf, LR_PARAM_GRID, cv=3, scoring="accuracy",
            n_jobs=-1, refit=True
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}")


def evaluate_predictions(labels, predictions, probs, label_names, prefix):
    """Compute and return evaluation metrics."""
    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)

    metrics = {
        f"{prefix}_accuracy": acc,
        f"{prefix}_f1_macro": f1_macro,
        f"{prefix}_f1_weighted": f1_weighted,
    }

    # AUROC
    try:
        n_classes = probs.shape[1] if probs.ndim > 1 else 2
        if n_classes == 2:
            auroc = roc_auc_score(labels, probs[:, 1] if probs.ndim > 1 else probs)
        else:
            auroc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        metrics[f"{prefix}_auroc"] = auroc
    except ValueError as e:
        warnings.warn(f"Could not compute AUROC for {prefix}: {e}")

    if acc < 0.5:
        warnings.warn(f"{prefix}: accuracy {acc:.3f} is below chance level")

    return metrics


def run_benchmark(
    h5ad_path: str,
    train_donors: list,
    test_donors: list,
    classifier_type: str,
    feature_type: str,
    group_size: Optional[int],
    label_names: Dict,
    output_dir: str,
    seed: int = 42,
) -> Dict:
    """
    Run a single classical benchmark.

    Training uses one group of group_size cells per donor.
    Testing creates ceil(n_cells/gs) non-overlapping groups per donor,
    reports group-level metrics and donor-aggregated metrics.
    """
    gs_str = "all" if group_size is None else str(group_size)
    method = f"{classifier_type}_{feature_type}"
    prefix = f"{method}_gs{gs_str}"
    print(f"\n--- {prefix} ---")

    # Aggregate features for training (1 group per donor)
    train_features, train_labels, train_donor_arr = aggregate_donor_features(
        h5ad_path, train_donors, feature_type, group_size, seed=seed
    )

    # For testing: multi-group if group_size is set, single-group if "all"
    if group_size is not None:
        test_features, test_labels, test_donor_arr = aggregate_donor_features_multi(
            h5ad_path, test_donors, feature_type, group_size, seed=seed + 1
        )
    else:
        test_features, test_labels, test_donor_arr = aggregate_donor_features(
            h5ad_path, test_donors, feature_type, group_size, seed=seed + 1
        )

    # Verify donor isolation
    assert len(set(train_donor_arr) & set(test_donor_arr)) == 0, \
        "Train and test donors overlap!"

    print(f"  Train: {train_features.shape}, Test: {test_features.shape}")

    # Scale features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    # Check for NaN/Inf
    assert not np.any(np.isnan(train_features)), "Train features contain NaN"
    assert not np.any(np.isinf(train_features)), "Train features contain Inf"

    # Create and train classifier with HP tuning
    clf = create_classifier(classifier_type, seed=seed)
    clf.fit(train_features, train_labels)

    if hasattr(clf, "best_params_"):
        print(f"  Best params: {clf.best_params_}")

    # Predict on test groups
    test_predictions = clf.predict(test_features)
    test_probs = clf.predict_proba(test_features)

    # Determine n_classes from probs
    n_classes = test_probs.shape[1]

    # Group-level metrics
    group_metrics = evaluate_predictions(
        test_labels, test_predictions, test_probs, label_names,
        f"{prefix}_group",
    )

    # Donor-level metrics (aggregate groups per donor)
    if group_size is not None:
        donor_agg = aggregate_to_donor(
            test_predictions, test_labels, test_probs, test_donor_arr, n_classes
        )
        mv = donor_agg["majority_vote"]
        donor_mv_metrics = evaluate_predictions(
            mv["labels"], mv["predictions"], mv["probs"], label_names,
            f"{prefix}_donor_majority",
        )
        mp = donor_agg["mean_probs"]
        donor_mp_metrics = evaluate_predictions(
            mp["labels"], mp["predictions"], mp["probs"], label_names,
            f"{prefix}_donor_meanprobs",
        )
    else:
        # group_size=None: group IS the donor (all cells), no aggregation needed
        donor_mv_metrics = {}
        donor_mp_metrics = {}

    all_metrics = {**group_metrics, **donor_mv_metrics, **donor_mp_metrics}

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    result = {
        "method": method,
        "group_size": gs_str,
        "group_metrics": {k: float(v) for k, v in group_metrics.items()},
        "donor_majority_metrics": {k: float(v) for k, v in donor_mv_metrics.items()},
        "donor_meanprobs_metrics": {k: float(v) for k, v in donor_mp_metrics.items()},
        "label_names": label_names,
        "classifier_type": classifier_type,
        "feature_type": feature_type,
    }
    if hasattr(clf, "best_params_"):
        result["best_params"] = {k: str(v) for k, v in clf.best_params_.items()}

    result_path = os.path.join(output_dir, f"{prefix}_results.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Group metrics: {group_metrics}")
    if donor_mv_metrics:
        print(f"  Donor majority metrics: {donor_mv_metrics}")
        print(f"  Donor meanprobs metrics: {donor_mp_metrics}")
    wandb.log(all_metrics)

    return all_metrics


# ---------- Deep learning benchmarks ----------

def run_dl_benchmark(
    h5ad_path: str,
    train_donors: list,
    test_donors: list,
    model_name: str,
    model_cfg: DictConfig,
    label_map: Dict[str, int],
    label_names: Dict,
    output_dir: str,
    group_size: Optional[int] = None,
    seed: int = 42,
    donor_key: str = "donor_id",
    label_key: str = "label",
    cell_type_key: str = "cell_type",
) -> Dict:
    """Run a deep learning benchmark (CellCnn, scAGG, or ScRAT).

    When group_size is set, overrides the model's default cells-per-bag setting
    and reports both group-level and donor-aggregated metrics.
    """
    from torch.utils.data import DataLoader
    from tissueformer.benchmark_models.data import (
        MILDataset, CroppedMILDataset, mil_collate_fn,
        load_covid_mil_data, preprocess_zscore, preprocess_cp10k_log1p,
        select_hvgs,
    )
    from tissueformer.benchmark_models.cellcnn import CellCnn
    from tissueformer.benchmark_models.scagg import ScAGG
    from tissueformer.benchmark_models.scrat import ScRAT, cell_type_mixup
    from tissueformer.benchmark_models.trainer import BenchmarkTrainer
    import transformers

    gs_str = "all" if group_size is None else str(group_size)
    print(f"\n--- DL benchmark: {model_name} (gs={gs_str}) ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_classes = len(label_map)

    # Override model cell counts with group_size if specified
    if group_size is not None:
        if model_name == "cellcnn":
            model_cfg = OmegaConf.merge(model_cfg, {"cells_per_input": group_size})
        elif model_name == "scagg":
            model_cfg = OmegaConf.merge(model_cfg, {"cells_per_sample": group_size})
        elif model_name == "scrat":
            model_cfg = OmegaConf.merge(model_cfg, {"cells_per_crop": group_size})

    # Load data organized by donor
    X, sample_indices, labels, cell_types, donor_order = load_covid_mil_data(
        h5ad_path, train_donors + test_donors, label_map,
        donor_key=donor_key, label_key=label_key, cell_type_key=cell_type_key,
    )

    # Split into train/test by donor
    train_set = set(train_donors)
    train_idx = [i for i, d in enumerate(donor_order) if d in train_set]
    test_idx = [i for i, d in enumerate(donor_order) if d not in train_set]

    # Create validation split from train
    rng = np.random.RandomState(seed)
    rng.shuffle(train_idx)
    n_val = max(1, len(train_idx) // 5)
    val_idx = train_idx[:n_val]
    train_idx = train_idx[n_val:]

    # Apply preprocessing
    preprocess = model_cfg.get("preprocess", "raw")
    scaler = None
    if preprocess == "zscore":
        # Fit scaler on train cells only
        train_cell_idx = np.concatenate([sample_indices[i] for i in train_idx])
        scaler = StandardScaler()
        scaler.fit(X[train_cell_idx])
        X, _ = preprocess_zscore(X, scaler)
    elif preprocess == "cp10k_log1p":
        X = preprocess_cp10k_log1p(X)
        n_hvgs = model_cfg.get("n_hvgs", None)
        if n_hvgs:
            X, hvg_idx = select_hvgs(X, n_hvgs)

    n_genes = X.shape[1]

    # Build datasets
    def make_indices(idx_list):
        return [sample_indices[i] for i in idx_list]

    def make_labels(idx_list):
        return labels[idx_list]

    # Compute crops per donor for test sets: ceil(n_cells / group_size)
    def compute_crops_per_donor(idx_list):
        """Compute the number of non-overlapping groups per donor."""
        crops = []
        for i in idx_list:
            n_cells = len(sample_indices[i])
            crops.append(math.ceil(n_cells / group_size) if group_size else 1)
        return crops

    # For test evaluation with group_size set, use CroppedMILDataset for all models
    # to get multiple crops per donor for aggregation
    use_cropped_test = group_size is not None

    if model_name == "scrat":
        cells_per_crop = model_cfg.cells_per_crop
        train_ds = CroppedMILDataset(
            X, make_indices(train_idx), make_labels(train_idx),
            cells_per_crop=cells_per_crop,
            crops_per_sample=model_cfg.crops_per_patient_train,
        )
        val_ds = CroppedMILDataset(
            X, make_indices(val_idx), make_labels(val_idx),
            cells_per_crop=cells_per_crop,
            crops_per_sample=model_cfg.crops_per_patient_test,
        )
        test_ds = CroppedMILDataset(
            X, make_indices(test_idx), make_labels(test_idx),
            cells_per_crop=cells_per_crop,
            crops_per_sample=model_cfg.crops_per_patient_test,
        )
        test_crops_per = model_cfg.crops_per_patient_test
    elif model_name == "cellcnn":
        cells_per_input = model_cfg.get("cells_per_input", 200)
        train_ds = MILDataset(
            X, make_indices(train_idx), make_labels(train_idx),
            cells_per_sample=cells_per_input,
        )
        val_ds = MILDataset(
            X, make_indices(val_idx), make_labels(val_idx),
            cells_per_sample=cells_per_input,
        )
        if use_cropped_test:
            # Use max crops across test donors for uniform CroppedMILDataset
            test_crops_list = compute_crops_per_donor(test_idx)
            test_crops_per = max(test_crops_list)
            test_ds = CroppedMILDataset(
                X, make_indices(test_idx), make_labels(test_idx),
                cells_per_crop=cells_per_input,
                crops_per_sample=test_crops_per,
            )
        else:
            test_ds = MILDataset(
                X, make_indices(test_idx), make_labels(test_idx),
                cells_per_sample=cells_per_input,
            )
            test_crops_per = 1
    else:
        # scagg
        cells_per = model_cfg.get("cells_per_sample", None)
        train_ds = MILDataset(
            X, make_indices(train_idx), make_labels(train_idx),
            cells_per_sample=cells_per,
        )
        val_ds = MILDataset(
            X, make_indices(val_idx), make_labels(val_idx),
            cells_per_sample=cells_per,
        )
        if use_cropped_test:
            test_crops_list = compute_crops_per_donor(test_idx)
            test_crops_per = max(test_crops_list)
            test_ds = CroppedMILDataset(
                X, make_indices(test_idx), make_labels(test_idx),
                cells_per_crop=cells_per if cells_per else 5000,
                crops_per_sample=test_crops_per,
            )
        else:
            test_ds = MILDataset(
                X, make_indices(test_idx), make_labels(test_idx),
                cells_per_sample=cells_per,
            )
            test_crops_per = 1

    batch_size = model_cfg.batch_size
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=mil_collate_fn, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=mil_collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=mil_collate_fn,
    )

    # Build model
    if model_name == "cellcnn":
        model = CellCnn(
            n_genes=n_genes,
            n_classes=n_classes,
            n_filters=model_cfg.n_filters,
            maxpool_percentage=model_cfg.maxpool_percentage,
            dropout=model_cfg.dropout,
            cells_per_input=model_cfg.cells_per_input,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=model_cfg.lr,
            weight_decay=model_cfg.l2_reg,
        )
        l1_fn = lambda: model.l1_loss(model_cfg.l1_reg)
        trainer = BenchmarkTrainer(
            model=model, optimizer=optimizer,
            train_loader=train_loader, val_loader=val_loader,
            device=device, n_epochs=model_cfg.n_epochs,
            early_stopping_patience=model_cfg.early_stopping_patience,
            l1_loss_fn=l1_fn, model_name=model_name,
            n_classes=n_classes,
        )
    elif model_name == "scagg":
        model = ScAGG(
            n_genes=n_genes,
            n_classes=n_classes,
            hidden_dim=model_cfg.hidden_dim,
            n_heads=model_cfg.n_heads,
            n_heads2=model_cfg.n_heads2,
            dropout=model_cfg.dropout,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=model_cfg.lr,
            weight_decay=model_cfg.weight_decay,
        )
        loss_fn = torch.nn.CrossEntropyLoss(
            reduction=model_cfg.loss_reduction,
        )
        trainer = BenchmarkTrainer(
            model=model, optimizer=optimizer,
            train_loader=train_loader, val_loader=val_loader,
            device=device, n_epochs=model_cfg.n_epochs,
            loss_fn=loss_fn,
            early_stopping_patience=None,
            model_name=model_name, n_classes=n_classes,
        )
    elif model_name == "scrat":
        model = ScRAT(
            n_genes=n_genes,
            n_classes=n_classes,
            hidden_dim=model_cfg.hidden_dim,
            n_heads=model_cfg.n_heads,
            n_layers=model_cfg.n_layers,
            dropout=model_cfg.dropout,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=model_cfg.lr,
            weight_decay=model_cfg.weight_decay,
        )
        # Cosine schedule with warmup
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=model_cfg.lr_warmup_epochs,
            num_training_steps=model_cfg.n_epochs,
        )
        trainer = BenchmarkTrainer(
            model=model, optimizer=optimizer,
            train_loader=train_loader, val_loader=val_loader,
            device=device, n_epochs=model_cfg.n_epochs,
            early_stopping_patience=model_cfg.early_stopping_patience,
            early_stopping_start_epoch=model_cfg.early_stopping_start_epoch,
            scheduler=scheduler, model_name=model_name,
            use_sigmoid=True, n_classes=n_classes,
        )
    else:
        raise ValueError(f"Unknown DL benchmark model: {model_name}")

    # Train
    trainer.train()

    # Evaluate on test set
    test_loss, test_preds, test_labels, test_probs = trainer.eval_epoch(test_loader)

    prefix = f"{model_name}_gs{gs_str}"

    # Group-level metrics (per-crop accuracy)
    group_metrics = evaluate_predictions(
        test_labels, test_preds, test_probs, label_names, f"{prefix}_group"
    )

    # Donor-level metrics (aggregate crops per donor)
    n_patients = len(test_idx)
    if test_crops_per > 1:
        # Build donor_ids array matching the per-crop predictions
        test_donor_ids = np.array([
            donor_order[test_idx[p]]
            for p in range(n_patients)
            for _ in range(test_crops_per)
        ])
        donor_agg = aggregate_to_donor(
            test_preds, test_labels, test_probs, test_donor_ids, n_classes
        )
        mv = donor_agg["majority_vote"]
        donor_mv_metrics = evaluate_predictions(
            mv["labels"], mv["predictions"], mv["probs"], label_names,
            f"{prefix}_donor_majority",
        )
        mp = donor_agg["mean_probs"]
        donor_mp_metrics = evaluate_predictions(
            mp["labels"], mp["predictions"], mp["probs"], label_names,
            f"{prefix}_donor_meanprobs",
        )
    else:
        # 1 prediction per donor, group == donor
        donor_mv_metrics = {}
        donor_mp_metrics = {}

    all_metrics = {**group_metrics, **donor_mv_metrics, **donor_mp_metrics}

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    result = {
        "method": model_name,
        "group_size": gs_str,
        "group_metrics": {k: float(v) for k, v in group_metrics.items()},
        "donor_majority_metrics": {k: float(v) for k, v in donor_mv_metrics.items()},
        "donor_meanprobs_metrics": {k: float(v) for k, v in donor_mp_metrics.items()},
        "model_name": model_name,
    }
    result_path = os.path.join(output_dir, f"{prefix}_results.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Group metrics: {group_metrics}")
    if donor_mv_metrics:
        print(f"  Donor majority metrics: {donor_mv_metrics}")
        print(f"  Donor meanprobs metrics: {donor_mp_metrics}")
    wandb.log(all_metrics)

    return all_metrics


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Running COVID benchmarks...")
    print(OmegaConf.to_yaml(cfg))

    setup_wandb(cfg)

    # Load donor splits
    with open(cfg.data.splits_path) as f:
        split_info = json.load(f)

    fold = cfg.data.cv_fold
    train_donors = split_info["folds"][str(fold)]["train_donors"]
    test_donors = split_info["folds"][str(fold)]["test_donors"]

    h5ad_path = cfg.data.h5ad_path

    label_names = OmegaConf.to_container(cfg.data.label_names)

    group_size = cfg.data.group_size
    if isinstance(group_size, str) and group_size == "all":
        group_size = None

    all_metrics = {}
    for clf_type in ["random_forest", "logistic_regression"]:
        for feat_type in ["pseudobulk", "cell_type_histogram"]:
            metrics = run_benchmark(
                h5ad_path=h5ad_path,
                train_donors=train_donors,
                test_donors=test_donors,
                classifier_type=clf_type,
                feature_type=feat_type,
                group_size=group_size,
                label_names=label_names,
                output_dir=cfg.training.output_dir,
                seed=cfg.seed,
            )
            all_metrics.update(metrics)

    # Deep learning benchmarks
    label_map = _get_label_map(ad.read_h5ad(h5ad_path, backed="r"))

    dl_models = {
        "cellcnn": cfg.get("run_cellcnn", False),
        "scagg": cfg.get("run_scagg", False),
        "scrat": cfg.get("run_scrat", False),
    }

    for model_name, should_run in dl_models.items():
        if not should_run:
            continue
        model_cfg = OmegaConf.load(
            os.path.join(
                os.path.dirname(__file__),
                "config", "benchmark_models", f"{model_name}.yaml",
            )
        )
        # Allow overrides from main config
        if hasattr(cfg, "benchmark_models") and hasattr(cfg.benchmark_models, model_name):
            model_cfg = OmegaConf.merge(model_cfg, cfg.benchmark_models[model_name])

        metrics = run_dl_benchmark(
            h5ad_path=h5ad_path,
            train_donors=train_donors,
            test_donors=test_donors,
            model_name=model_name,
            model_cfg=model_cfg,
            label_map=label_map,
            label_names=label_names,
            output_dir=cfg.training.output_dir,
            group_size=group_size,
            seed=cfg.seed,
            donor_key=cfg.data.get("donor_key", "donor_id"),
            label_key=cfg.data.get("label_key", "label"),
            cell_type_key=cfg.data.get("cell_type_key", "cell_type"),
        )
        all_metrics.update(metrics)

    # Save combined metrics
    combined_path = os.path.join(cfg.training.output_dir, "benchmark_metrics.json")
    with open(combined_path, "w") as f:
        json.dump({k: float(v) for k, v in all_metrics.items()}, f, indent=2)

    wandb.finish()


if __name__ == "__main__":
    main()
