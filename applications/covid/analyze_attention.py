"""Attention interpretability analysis for COVID severity classification.

Loads a trained TissueFormer checkpoint, collects attention weights, and
generates figures showing which cell types receive the most attention.

Usage:
    python analyze_attention.py model_checkpoint_path=/path/to/model
"""

import os
import json

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from datasets import load_from_disk, DatasetDict, disable_caching
from transformers import set_seed

from tissueformer.model import TissueFormer, TissueFormerConfig
from tissueformer.samplers import SpatialGroupCollator, DonorGroupSampler
from tissueformer.attention_analysis import (
    AttentionCollector,
    cell_type_attention_summary,
    cell_type_total_attention_summary,
    plot_single_group_heatmap,
    plot_attention_per_label,
    plot_overall_attention_ranking,
    plot_total_attention_ranking,
    plot_abundance_vs_attention,
)

from train import prepare_datasets


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    assert cfg.model_checkpoint_path is not None, (
        "model_checkpoint_path must be set (path to a trained TissueFormer checkpoint)"
    )

    # Defaults for attention analysis config
    attn_cfg = OmegaConf.to_container(cfg.get("attention_analysis", {}), resolve=True)
    layer_idx = attn_cfg.get("layer", -1)
    top_k = attn_cfg.get("top_k", 15)
    split = attn_cfg.get("split", "test")
    max_groups = attn_cfg.get("max_groups", None)

    set_seed(cfg.seed)
    disable_caching()

    # Load dataset and splits (same as train.py)
    dataset = load_from_disk(cfg.data.dataset_path)
    if isinstance(dataset, DatasetDict):
        raise TypeError(
            f"Expected a single Dataset at {cfg.data.dataset_path}, got a DatasetDict."
        )

    with open(cfg.data.splits_path) as f:
        donor_splits = json.load(f)

    datasets = prepare_datasets(dataset, donor_splits, cfg.data.cv_fold, cfg)
    eval_dataset = datasets[split]
    print(f"Using {split} split: {len(eval_dataset)} cells")

    # Load trained model — override num_labels from Hydra config since
    # older checkpoints may not have saved it correctly
    checkpoint_path = os.path.expanduser(cfg.model_checkpoint_path)
    model_config = TissueFormerConfig.from_pretrained(checkpoint_path)
    model_config.num_labels = cfg.model.num_labels
    model = TissueFormer.from_pretrained(checkpoint_path, config=model_config)
    model.class_weights = None  # not needed for inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Loaded model from {checkpoint_path}")

    # Create collator with metadata_keys to preserve cell types
    cell_type_key = cfg.data.cell_type_key
    collator = SpatialGroupCollator(
        group_size=cfg.data.group_size,
        label_key="labels",
        feature_keys=["input_ids"],
        pad_token_id=0,
        add_single_cell_labels=False,
        index_key=None,
        relative_positions=False,
        metadata_keys=[cell_type_key],
    )

    # Create sampler in eval mode (iterate all donors)
    sampler = DonorGroupSampler(
        dataset=eval_dataset,
        batch_size=cfg.data.group_size,
        group_size=cfg.data.group_size,
        donor_key=cfg.data.donor_key,
        seed=42,
        iterate_all_donors=True,
    )

    label_names = {int(k): v for k, v in cfg.data.label_names.items()}

    # Collect attention weights
    collector = AttentionCollector(
        model=model,
        dataset=eval_dataset,
        collator=collator,
        sampler=sampler,
        cell_type_key=cell_type_key,
        label_names=label_names,
        batch_size=cfg.data.group_size,
        max_groups=max_groups,
    )
    print("Collecting attention weights...")
    results = collector.collect()
    print(f"Collected {len(results.attentions)} groups")

    # Generate figures
    output_dir = cfg.training.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 1. Heatmap for first group
    fig = plot_single_group_heatmap(results, group_idx=0, layer_idx=layer_idx)
    path = os.path.join(output_dir, "attention_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")

    # 2. Per-label attention bar charts
    summary = cell_type_attention_summary(results, layer_idx=layer_idx)
    fig = plot_attention_per_label(summary, top_k=top_k)
    path = os.path.join(output_dir, "attention_per_label.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")

    # 3. Overall ranking (per-cell mean — upweights rare types)
    fig = plot_overall_attention_ranking(summary, top_k=top_k + 5)
    path = os.path.join(output_dir, "attention_ranking.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")

    # 4. Total attention ranking (abundance-weighted — common types rank higher)
    total_summary = cell_type_total_attention_summary(results, layer_idx=layer_idx)
    fig = plot_total_attention_ranking(total_summary, top_k=top_k + 5)
    path = os.path.join(output_dir, "attention_total_ranking.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")

    # 5. Abundance vs. attention scatter
    fig = plot_abundance_vs_attention(results, layer_idx=layer_idx)
    path = os.path.join(output_dir, "abundance_vs_attention.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")

    # Save summary CSV
    csv_path = os.path.join(output_dir, "attention_summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
