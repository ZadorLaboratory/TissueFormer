"""Attention interpretability analysis for brain region annotation.

Loads a trained TissueFormer checkpoint, collects attention weights, and
generates figures showing which cell types receive the most attention.

Usage:
    python -m applications.brain_annotation.analyze_attention \
        model_checkpoint_path=/path/to/model
"""

import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from datasets import DatasetDict, load_from_disk, disable_caching
from transformers import set_seed

from tissueformer.model import TissueFormer, TissueFormerConfig
from tissueformer.samplers import SpatialGroupCollator, SpatialGroupSampler
from tissueformer.attention_analysis import (
    AttentionCollector,
    cell_type_attention_summary,
    plot_single_group_heatmap,
    plot_attention_per_label,
    plot_overall_attention_ranking,
)
from tissueformer.train import prepare_datasets


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
    # With 290 labels, only show a subset in per-label plots
    max_labels_in_plot = attn_cfg.get("max_labels_in_plot", 12)

    set_seed(cfg.seed)
    disable_caching()

    # Load dataset (brain annotation uses DatasetDict with train/test)
    dataset_dict = load_from_disk(cfg.data.dataset_path)
    datasets = prepare_datasets(dataset_dict, cfg)
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
    cell_type_key = "H2_type"
    collator = SpatialGroupCollator(
        group_size=cfg.data.group_size,
        label_key="labels",
        feature_keys=["input_ids"],
        pad_token_id=0,
        add_single_cell_labels=False,
        index_key=None,
        relative_positions=cfg.model.relative_positions.enabled,
        coordinate_key=cfg.data.coordinate_key,
        absolute_Z=cfg.model.relative_positions.absolute_Z,
        metadata_keys=[cell_type_key],
    )

    # Create spatial sampler for brain data
    sampler = SpatialGroupSampler(
        dataset=eval_dataset,
        batch_size=cfg.data.group_size,
        group_size=cfg.data.group_size,
        coordinate_key=cfg.data.coordinate_key,
        seed=42,
        reflect_points=cfg.data.sampling.reflect_points,
        group_within_keys=cfg.data.sampling.group_within_keys,
        max_radius_expansions=cfg.data.sampling.max_radius_expansions,
        iterate_all_points=True,
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

    # 1. Heatmap for first correctly-predicted group
    correct_mask = results.predictions == results.labels
    correct_idx = np.where(correct_mask)[0]
    heatmap_idx = int(correct_idx[0]) if len(correct_idx) > 0 else 0
    fig = plot_single_group_heatmap(results, group_idx=heatmap_idx, layer_idx=layer_idx)
    path = os.path.join(output_dir, "attention_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")

    # 2. Per-label attention (subset of labels for readability)
    summary = cell_type_attention_summary(results, layer_idx=layer_idx)

    # Filter to labels with the most groups for a readable plot
    if len(summary["label"].unique()) > max_labels_in_plot:
        label_counts = summary.groupby("label")["count"].sum().nlargest(max_labels_in_plot)
        summary_subset = summary[summary["label"].isin(label_counts.index)]
    else:
        summary_subset = summary

    fig = plot_attention_per_label(summary_subset, top_k=top_k)
    path = os.path.join(output_dir, "attention_per_label.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")

    # 3. Overall ranking
    fig = plot_overall_attention_ranking(summary, top_k=top_k + 5)
    path = os.path.join(output_dir, "attention_ranking.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")

    # Save summary CSV
    csv_path = os.path.join(output_dir, "attention_summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
