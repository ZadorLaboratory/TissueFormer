"""Re-plot attention and LOO figures from saved CSVs (no model needed).

Usage:
    python applications/brain_annotation/replot_attention.py [figures_dir]

Defaults to applications/brain_annotation/figures/.
"""

import glob
import os
import sys

import anndata as ad
import matplotlib as mpl
import pandas as pd

mpl.rcParams.update({
    "font.size": 22,
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})

from tissueformer.attention_analysis import (
    plot_attention_per_label,
    plot_overall_attention_ranking,
    plot_loo_importance_ranking,
    plot_loo_importance_per_label,
)

TOP_K = 5
MAX_LABELS_IN_PLOT = 18


def load_color_map():
    cell_type_key = "H2_type"
    h5ad_dir = os.path.join(os.path.dirname(__file__), "data", "anndatas")
    h5ad_files = sorted(glob.glob(os.path.join(h5ad_dir, "*.h5ad")))
    if h5ad_files:
        try:
            from colormycells import get_colormap as get_cell_colormap
            ref_adata = ad.read_h5ad(h5ad_files[0], backed="r")
            return get_cell_colormap(ref_adata, key=cell_type_key, plot_colorspace=False)
        except Exception as e:
            print(f"Could not generate colormap: {e}")
    return None


def save_fig(fig, name, output_dir):
    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    import matplotlib.pyplot as plt
    plt.close(fig)


def main():
    figures_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "figures"
    )

    color_map = load_color_map()

    # --- Attention plots ---
    attn_csv = os.path.join(figures_dir, "attention_summary.csv")
    if os.path.exists(attn_csv):
        summary = pd.read_csv(attn_csv)
        print(f"Loaded {len(summary)} rows from {attn_csv}")

        if len(summary["label"].unique()) > MAX_LABELS_IN_PLOT:
            label_counts = summary.groupby("label")["count"].sum().nlargest(MAX_LABELS_IN_PLOT)
            summary_subset = summary[summary["label"].isin(label_counts.index)]
        else:
            summary_subset = summary

        fig = plot_attention_per_label(summary_subset, top_k=TOP_K, color_map=color_map)
        save_fig(fig, "attention_per_label", figures_dir)

        fig = plot_overall_attention_ranking(summary, top_k=TOP_K + 5, color_map=color_map)
        save_fig(fig, "attention_ranking", figures_dir)
    else:
        print(f"No {attn_csv} found, skipping attention plots")

    # --- LOO plots ---
    loo_csv = os.path.join(figures_dir, "loo_importance_summary.csv")
    if os.path.exists(loo_csv):
        loo_summary = pd.read_csv(loo_csv)
        print(f"Loaded {len(loo_summary)} rows from {loo_csv}")

        if len(loo_summary["label"].unique()) > MAX_LABELS_IN_PLOT:
            loo_label_counts = loo_summary.groupby("label")["n_groups"].sum().nlargest(MAX_LABELS_IN_PLOT)
            loo_summary_subset = loo_summary[loo_summary["label"].isin(loo_label_counts.index)]
        else:
            loo_summary_subset = loo_summary

        fig = plot_loo_importance_ranking(
            loo_summary, top_k=TOP_K + 5, figsize=(10, 4),
            vertical=True, bar_color="black",
        )
        save_fig(fig, "loo_importance_ranking", figures_dir)

        fig = plot_loo_importance_per_label(loo_summary_subset, top_k=TOP_K, color_map=color_map)
        save_fig(fig, "loo_importance_per_label", figures_dir)
    else:
        print(f"No {loo_csv} found, skipping LOO plots")


if __name__ == "__main__":
    main()
