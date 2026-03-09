"""
Diagnostic figures for COVID severity classification.

Pulls results from wandb and produces a single figure:
  Row 1: TissueFormer eval vs test accuracy by group_size
  Row 2: Benchmark test accuracy by group_size (one line per method)
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import wandb

from plot_results import (
    fetch_runs, classify_runs,
    DATASETS, DATASET_LABELS, GROUP_SIZES, METHODS,
    _build_benchmark_col_name,
)


plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})

TF_METRIC_KEYS = [
    ("Eval", "eval/accuracy"),
    ("Test", "test/accuracy"),
]

TF_COLORS = {"Eval": "#FF9800", "Test": "#2196F3"}

BENCH_TEST_SUFFIX = "group_accuracy"
BENCH_VAL_SUFFIX = "val_group_accuracy"


def _plot_errorbar(ax, means, stds, **kwargs):
    """Plot errorbar with asymmetric bars clamped to [0, 1]."""
    yerr_lower = np.minimum(stds.values, means.values)
    yerr_upper = np.minimum(stds.values, 1.0 - means.values)
    ax.errorbar(
        means.index, means.values, yerr=[yerr_lower, yerr_upper],
        capsize=3, linewidth=1.5, markersize=5, **kwargs,
    )


def plot_diagnostics(tf_df, bench_df, output_dir):
    """Two-row diagnostic figure: TissueFormer (row 1), benchmarks (row 2)."""
    n_cols = len(DATASETS)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 7),
                             sharex=True)

    for col, dataset in enumerate(DATASETS):
        # --- Row 0: TissueFormer eval vs test ---
        ax = axes[0, col]
        df = tf_df[tf_df["dataset_name"] == dataset]

        for label, key in TF_METRIC_KEYS:
            if key not in df.columns:
                continue
            subset = df[["data.group_size", key]].dropna()
            if subset.empty:
                continue
            grouped = subset.groupby("data.group_size")[key]
            _plot_errorbar(ax, grouped.mean(), grouped.std().fillna(0),
                           color=TF_COLORS[label], marker="o", label=label)

        ax.set_title(DATASET_LABELS.get(dataset, dataset))
        ax.grid(True, alpha=0.3)

        # --- Row 1: Benchmarks val + test accuracy ---
        ax = axes[1, col]
        bench_ds = bench_df[bench_df["dataset_name"] == dataset]

        for method_key, style in METHODS.items():
            if method_key == "tissueformer":
                continue

            for suffix, linestyle, label_suffix in [
                (BENCH_VAL_SUFFIX, "--", " (val)"),
                (BENCH_TEST_SUFFIX, "-", " (test)"),
            ]:
                x_vals, mean_vals, std_vals = [], [], []
                for gs in GROUP_SIZES:
                    col_name = _build_benchmark_col_name(method_key, gs, suffix)
                    if col_name not in bench_ds.columns:
                        continue
                    values = pd.to_numeric(bench_ds[col_name], errors="coerce").dropna()
                    if len(values) > 0:
                        x_vals.append(gs)
                        mean_vals.append(values.mean())
                        std_vals.append(values.std() if len(values) > 1 else 0)

                if x_vals:
                    means = pd.Series(mean_vals, index=x_vals)
                    stds = pd.Series(std_vals, index=x_vals)
                    _plot_errorbar(ax, means, stds,
                                   color=style["color"], marker=style["marker"],
                                   linestyle=linestyle,
                                   label=style["label"] + label_suffix)

        ax.grid(True, alpha=0.3)

    # Shared formatting
    for row in range(2):
        axes[row, 0].set_ylabel("Accuracy")
        for col in range(n_cols):
            axes[row, col].set_xscale("log", base=2)
            axes[row, col].xaxis.set_major_formatter(ScalarFormatter())
    for col in range(n_cols):
        axes[1, col].set_xlabel("Group size")

    # Row legends
    h0, l0 = axes[0, 0].get_legend_handles_labels()
    if h0:
        axes[0, -1].legend(h0, l0, loc="lower right")

    # Deduplicate benchmark legend across all columns
    h1, l1 = [], []
    for col in range(n_cols):
        h, l = axes[1, col].get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in l1:
                h1.append(hi)
                l1.append(li)
    if h1:
        axes[1, -1].legend(h1, l1, loc="lower right", fontsize=8)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "diagnostics.png")
    fig.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot COVID diagnostic figures from wandb")
    parser.add_argument("--entity", type=str, default="zadorlab")
    parser.add_argument("--project", type=str, default="covid-severity")
    parser.add_argument("--output_dir", type=str, default="figures")
    args = parser.parse_args()

    print(f"Fetching runs from {args.entity}/{args.project}...")
    df = fetch_runs(args.entity, args.project)
    print(f"Fetched {len(df)} finished runs")

    if df.empty:
        print("No finished runs found.")
        return

    tf_df, bench_df = classify_runs(df)
    print(f"TissueFormer runs: {len(tf_df)}, Benchmark runs: {len(bench_df)}")

    plot_diagnostics(tf_df, bench_df, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
