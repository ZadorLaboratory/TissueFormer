"""
COVID severity benchmarks.

Classical methods:
  - Random Forest / Logistic Regression on pseudobulk or cell-type histograms

Deep learning methods (end-to-end trainable on raw cells):
  - CellCnn  (Arvaniti & Claassen, Nat Commun 2017)
  - scAGG    (Verlaan et al., CSBJ 2025)
  - ScRAT    (Mao et al., Bioinformatics 2024)

Uses the same donor-stratified CV splits as TissueFormer.
"""

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

def aggregate_donor_features(
    h5ad_path: str,
    donor_ids: list,
    feature_type: str = "pseudobulk",
    group_size: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate cell-level data to donor-level features.

    Args:
        h5ad_path: Path to processed h5ad file.
        donor_ids: List of donor IDs to include.
        feature_type: "pseudobulk" (mean expression) or "cell_type_histogram".
        group_size: If set, sample this many cells per donor per group.
                    If None, use all cells (whole-donor mode).
        seed: Random seed for subsampling.

    Returns:
        features: (n_donors, n_features) array
        labels: (n_donors,) integer labels
        donors: (n_donors,) donor ID array
    """
    adata = ad.read_h5ad(h5ad_path)

    # Filter to requested donors
    mask = adata.obs["donor_id"].isin(donor_ids)
    adata = adata[mask]

    rng = np.random.RandomState(seed)
    unique_donors = sorted(set(donor_ids) & set(adata.obs["donor_id"].unique()))

    features_list = []
    labels_list = []
    donors_list = []

    # Pre-compute label map
    label_map = _get_label_map(adata)

    for donor in unique_donors:
        donor_mask = adata.obs["donor_id"] == donor
        donor_adata = adata[donor_mask]
        donor_label = label_map[donor_adata.obs["label"].iloc[0]]
        n_cells = len(donor_adata)

        if group_size is not None and group_size < n_cells:
            # Sample group_size cells
            idx = rng.choice(n_cells, size=group_size, replace=False)
            donor_adata = donor_adata[idx]
        elif group_size is not None and group_size > n_cells:
            # Sample with replacement
            idx = rng.choice(n_cells, size=group_size, replace=True)
            donor_adata = donor_adata[idx]

        if feature_type == "pseudobulk":
            X = donor_adata.X
            if scipy.sparse.issparse(X):
                X = X.toarray()
            feat = np.mean(X, axis=0).flatten()
        elif feature_type == "cell_type_histogram":
            cell_types = donor_adata.obs["cell_type"].values
            unique_types = sorted(adata.obs["cell_type"].unique())
            type_to_idx = {t: i for i, t in enumerate(unique_types)}
            hist = np.zeros(len(unique_types))
            for ct in cell_types:
                hist[type_to_idx[ct]] += 1
            total = hist.sum()
            if total > 0:
                hist /= total
            feat = hist
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

        features_list.append(feat)
        labels_list.append(donor_label)
        donors_list.append(donor)

    return np.array(features_list), np.array(labels_list), np.array(donors_list)


def _get_label_map(adata):
    """Determine label map from data."""
    unique_labels = set(adata.obs["label"].unique())
    if "control" in unique_labels:
        return {"control": 0, "mild": 1, "severe": 2}
    return {"mild": 0, "severe": 1}


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
    Run a single benchmark: train classifier on train donors, evaluate on test donors.
    """
    gs_str = "all" if group_size is None else str(group_size)
    prefix = f"{classifier_type}_{feature_type}_gs{gs_str}"
    print(f"\n--- {prefix} ---")

    # Aggregate features
    train_features, train_labels, train_donor_arr = aggregate_donor_features(
        h5ad_path, train_donors, feature_type, group_size, seed=seed
    )
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

    # Evaluate on test
    test_predictions = clf.predict(test_features)
    test_probs = clf.predict_proba(test_features)

    metrics = evaluate_predictions(
        test_labels, test_predictions, test_probs, label_names, prefix
    )

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    result = {
        "metrics": {k: float(v) for k, v in metrics.items()},
        "predictions": test_predictions.tolist(),
        "labels": test_labels.tolist(),
        "donor_ids": test_donor_arr.tolist(),
        "label_names": label_names,
        "group_size": gs_str,
        "classifier_type": classifier_type,
        "feature_type": feature_type,
    }
    if hasattr(clf, "best_params_"):
        result["best_params"] = {k: str(v) for k, v in clf.best_params_.items()}

    result_path = os.path.join(output_dir, f"{prefix}_results.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Metrics: {metrics}")
    wandb.log(metrics)

    return metrics


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
    seed: int = 42,
    donor_key: str = "donor_id",
    label_key: str = "label",
    cell_type_key: str = "cell_type",
) -> Dict:
    """Run a deep learning benchmark (CellCnn, scAGG, or ScRAT)."""
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

    print(f"\n--- DL benchmark: {model_name} ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_classes = len(label_map)

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

    if model_name == "scrat":
        train_ds = CroppedMILDataset(
            X, make_indices(train_idx), make_labels(train_idx),
            cells_per_crop=model_cfg.cells_per_crop,
            crops_per_sample=model_cfg.crops_per_patient_train,
        )
        val_ds = CroppedMILDataset(
            X, make_indices(val_idx), make_labels(val_idx),
            cells_per_crop=model_cfg.cells_per_crop,
            crops_per_sample=model_cfg.crops_per_patient_test,
        )
        test_ds = CroppedMILDataset(
            X, make_indices(test_idx), make_labels(test_idx),
            cells_per_crop=model_cfg.cells_per_crop,
            crops_per_sample=model_cfg.crops_per_patient_test,
        )
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
        test_ds = MILDataset(
            X, make_indices(test_idx), make_labels(test_idx),
            cells_per_sample=cells_per_input,
        )
    else:
        # scagg: variable bag sizes
        train_ds = MILDataset(
            X, make_indices(train_idx), make_labels(train_idx),
        )
        val_ds = MILDataset(
            X, make_indices(val_idx), make_labels(val_idx),
        )
        test_ds = MILDataset(
            X, make_indices(test_idx), make_labels(test_idx),
        )

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

    # For ScRAT with multiple crops per patient, do majority voting
    if model_name == "scrat":
        crops_per = model_cfg.crops_per_patient_test
        n_patients = len(test_idx)
        voted_preds = []
        voted_labels = []
        voted_probs = []
        for p in range(n_patients):
            start = p * crops_per
            end = start + crops_per
            patient_preds = test_preds[start:end]
            patient_probs = test_probs[start:end]
            # Majority vote
            voted_preds.append(np.bincount(patient_preds, minlength=n_classes).argmax())
            voted_labels.append(test_labels[start])
            voted_probs.append(patient_probs.mean(axis=0))
        test_preds = np.array(voted_preds)
        test_labels = np.array(voted_labels)
        test_probs = np.array(voted_probs)

    prefix = f"{model_name}_test"
    metrics = evaluate_predictions(
        test_labels, test_preds, test_probs, label_names, prefix
    )

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    result = {
        "metrics": {k: float(v) for k, v in metrics.items()},
        "predictions": test_preds.tolist(),
        "labels": test_labels.tolist(),
        "model_name": model_name,
    }
    result_path = os.path.join(output_dir, f"{model_name}_results.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Metrics: {metrics}")
    wandb.log(metrics)

    return metrics


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
