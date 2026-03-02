import os
from typing import Dict, Optional, Union

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from datasets import load_from_disk, disable_caching
import wandb

from tissueformer.class_weights import calculate_class_weights


def setup_wandb(cfg: DictConfig) -> None:
    """Initialize W&B logging if enabled"""
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"{cfg.wandb.name}_class_weights",
            group=cfg.wandb.group,
            tags=cfg.wandb.tags + ["class_weights"],
            notes=cfg.wandb.notes,
            config=OmegaConf.to_container(cfg, resolve=True),
        )


def infer_num_classes(labels) -> int:
    """
    Infer the total number of classes from the dataset.
    Checks all splits to ensure we capture all possible classes.
    
    Args:
        dataset: HuggingFace dataset with multiple splits
        label_key: Name of the label column
        
    Returns:
        int: Total number of unique classes across all splits
    """
    
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    
    print(f"\nLabel Analysis:")
    print(f"Found {num_classes} unique classes")
    print(f"Label range: [{unique_labels.min()}, {unique_labels.max()}]")
    print(f"Unique labels: {unique_labels}")
    
    return num_classes


def verify_balanced_weights(labels: np.ndarray, weights: np.ndarray) -> None:
    """
    Verify that the balanced weights have the expected properties:
    1. Equal to 1 when class frequencies are identical
    2. Sum of weighted samples per class is constant
    """
    counts = np.bincount(labels, minlength=len(weights))
    present_classes = counts > 0
    
    print("\nWeight Verification:")
    
    # Test with synthetic balanced data
    n_classes = np.sum(present_classes)
    balanced_labels = np.repeat(np.arange(n_classes), len(labels) // n_classes)
    balanced_weights = calculate_class_weights(balanced_labels, method="balanced")
    balanced_weight_mean = np.mean(balanced_weights[balanced_weights > 0])
    print(f"Weight for balanced classes: {balanced_weight_mean:.3f} "
          f"(should be close to 1.0)")
    
    # Check weighted samples per class
    weighted_samples = counts * weights
    weighted_samples_mean = np.mean(weighted_samples[present_classes])
    weighted_samples_std = np.std(weighted_samples[present_classes])
    print(f"Weighted samples per class: mean={weighted_samples_mean:.3f}, "
          f"std={weighted_samples_std:.3f} (std should be close to 0)")


def log_weight_statistics(weights: np.ndarray, label_names: Optional[Dict[int, str]] = None) -> None:
    """Log weight statistics to console and W&B if enabled"""
    print("\nClass Weights Statistics:")
    non_zero_weights = weights[weights > 0]
    print(f"Mean (non-zero): {np.mean(non_zero_weights):.3f}")
    print(f"Std (non-zero): {np.std(non_zero_weights):.3f}")
    print(f"Min (non-zero): {np.min(non_zero_weights):.3f}")
    print(f"Max: {np.max(weights):.3f}")
    
    print("\nPer-class weights:")
    for i, weight in enumerate(weights):
        if weight > 0 or i in (label_names or {}):
            label = label_names.get(i, str(i)) if label_names else str(i)
            print(f"Class {label}: {weight:.3f}")
        
    if wandb.run is not None:
        wandb.run.summary.update({
            "class_weights/mean": np.mean(non_zero_weights),
            "class_weights/std": np.std(non_zero_weights),
            "class_weights/min": np.min(non_zero_weights),
            "class_weights/max": np.max(weights),
        })
        
        wandb.log({
            "class_weights/distribution": wandb.Histogram(non_zero_weights),
            "class_weights/per_class": {f"class_{i}": w for i, w in enumerate(weights) if w > 0}
        })


@hydra.main(version_base=None, config_path="config", config_name="class_weights_config")
def main(cfg: DictConfig) -> None:
    """Calculate and save class weights from dataset labels"""
    print(OmegaConf.to_yaml(cfg))
    
    setup_wandb(cfg)
    disable_caching()
    
    dataset = load_from_disk(cfg.data.dataset_path)
    
    if cfg.data.split not in dataset:
        raise ValueError(f"Split {cfg.data.split} not found in dataset")
    
    if cfg.data.label_key not in dataset[cfg.data.split].features:
        raise ValueError(f"Label key {cfg.data.label_key} not found in dataset")
        
    # Calculate weights using all splits
    train_labels = np.concatenate([np.array(dataset["train"][cfg.data.label_key]), np.array(dataset["test"][cfg.data.label_key])])

    num_classes = infer_num_classes(train_labels)

    weights = calculate_class_weights(
        train_labels,
        method=cfg.weighting.method,
        beta=cfg.weighting.get("beta", 0.999),
    )
    
    # Verify balanced weights properties
    if cfg.weighting.method == "balanced":
        verify_balanced_weights(train_labels, weights)
    
    log_weight_statistics(weights, cfg.data.get("label_names"))
    
    output_path = os.path.join(cfg.data.dataset_path, "class_weights.npy")
    # os.makedirs(cfg.output_dir, exist_ok=True)
    np.save(output_path, weights)
    print(f"\nSaved class weights to {output_path}")
    
    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()