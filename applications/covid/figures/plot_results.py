"""
Manuscript-quality figures for COVID severity classification results.

Pulls results from wandb (project: covid-severity) and produces:
  1. Main figure: Group accuracy, donor accuracy, and donor AUROC vs group_size
     for TissueFormer + benchmarks
  2. Supplementary: Confusion matrices (TODO — requires logged artifacts)
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


# Style matching brain_annotation paper figures
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

DATASETS = ["combat", "ren", "stevenson"]
DATASET_LABELS = {"combat": "COMBAT", "ren": "Ren et al.", "stevenson": "Stevenson et al."}
GROUP_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# Method display config: maps method key -> plotting style
METHODS = {
    "tissueformer": {"color": "#2196F3", "marker": "o", "label": "TissueFormer"},
    "random_forest_pseudobulk": {"color": "#4CAF50", "marker": "s", "label": "RF (pseudobulk)"},
    "logistic_regression_pseudobulk": {"color": "#FF9800", "marker": "^", "label": "LR (pseudobulk)"},
    "random_forest_cell_type_histogram": {"color": "#9C27B0", "marker": "D", "label": "RF (cell type)"},
    "logistic_regression_cell_type_histogram": {"color": "#F44336", "marker": "v", "label": "LR (cell type)"},
}

# Metric rows: (display_label, tissueformer_key, benchmark_key_suffix)
# For TissueFormer, metrics are logged as test/{key}
# For benchmarks, metrics are logged as {clf}_{feat}_gs{N}_{suffix}
METRIC_ROWS = [
    ("Group Accuracy", "test/group_accuracy", "accuracy"),
    ("Donor Accuracy\n(mean logits)", "test/donor_meanlogits_accuracy", None),
    ("Donor AUROC\n(mean logits)", "test/donor_meanlogits_auroc", None),
]


def fetch_runs(entity: str, project: str, filters: dict = None) -> pd.DataFrame:
    """Fetch runs from wandb and return a DataFrame with config + summary."""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters=filters or {})

    records = []
    for run in runs:
        if run.state != "finished":
            continue
        summary = run.summary._json_dict.copy()
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        record = {
            "run_id": run.id,
            "run_name": run.name,
            "tags": run.tags,
            **summary,
        }
        # Flatten config with dot notation
        config_flat = pd.json_normalize(config, sep=".").to_dict(orient="records")[0]
        record.update(config_flat)
        records.append(record)

    return pd.DataFrame(records)


def classify_runs(df: pd.DataFrame):
    """Split DataFrame into TissueFormer and benchmark runs based on tags."""
    is_benchmark = df["tags"].apply(lambda t: "benchmark" in t if isinstance(t, list) else False)
    return df[~is_benchmark].copy(), df[is_benchmark].copy()


def plot_accuracy_auroc_vs_groupsize(tf_df, bench_df, output_dir):
    """
    Main figure: group accuracy, donor accuracy, and donor AUROC vs group_size.
    One column per dataset, three rows.
    """
    n_rows = len(METRIC_ROWS)
    fig, axes = plt.subplots(n_rows, len(DATASETS),
                             figsize=(4 * len(DATASETS), 3 * n_rows),
                             sharex=True, squeeze=False)

    for col, dataset in enumerate(DATASETS):
        # --- TissueFormer ---
        tf_ds = tf_df[tf_df["dataset_name"] == dataset]
        style = METHODS["tissueformer"]
        for row, (row_label, tf_key, _) in enumerate(METRIC_ROWS):
            if tf_key not in tf_ds.columns:
                continue
            subset = tf_ds[["data.group_size", tf_key]].dropna()
            if subset.empty:
                continue
            grouped = subset.groupby("data.group_size")[tf_key]
            means = grouped.mean()
            stds = grouped.std().fillna(0)
            axes[row, col].errorbar(
                means.index, means.values, yerr=stds.values,
                color=style["color"], marker=style["marker"],
                label=style["label"], capsize=3, linewidth=1.5, markersize=5,
            )

        # --- Benchmarks ---
        bench_ds = bench_df[bench_df["dataset_name"] == dataset]
        for method_key, style in METHODS.items():
            if method_key == "tissueformer":
                continue
            # method_key e.g. "random_forest_pseudobulk" -> clf="random_forest", feat="pseudobulk"
            tokens = method_key.split("_")
            if len(tokens) < 3:
                continue
            clf_type = "_".join(tokens[:2])
            feat_type = "_".join(tokens[2:])

            for row, (row_label, _, bench_suffix) in enumerate(METRIC_ROWS):
                if bench_suffix is None:
                    continue  # Benchmarks don't have donor-level aggregation

                # Find all columns matching this benchmark pattern
                # Metric keys: {clf}_{feat}_gs{N}_{suffix}
                x_vals, means, stds = [], [], []
                for gs in GROUP_SIZES:
                    col_name = f"{clf_type}_{feat_type}_gs{gs}_{bench_suffix}"
                    if col_name not in bench_ds.columns:
                        continue
                    values = bench_ds[col_name].dropna()
                    if len(values) > 0:
                        x_vals.append(gs)
                        means.append(values.mean())
                        stds.append(values.std() if len(values) > 1 else 0)

                if x_vals:
                    axes[row, col].errorbar(
                        x_vals, means, yerr=stds,
                        color=style["color"], marker=style["marker"],
                        label=style["label"], capsize=3, linewidth=1.5, markersize=5,
                    )

        # Formatting
        axes[0, col].set_title(DATASET_LABELS.get(dataset, dataset))
        axes[-1, col].set_xlabel("Group size")
        for row in range(n_rows):
            axes[row, col].set_xscale("log", base=2)
            axes[row, col].xaxis.set_major_formatter(ScalarFormatter())
            axes[row, col].grid(True, alpha=0.3)

    for row, (row_label, _, _) in enumerate(METRIC_ROWS):
        axes[row, 0].set_ylabel(row_label)

    # Single legend at top — deduplicate labels
    handles, labels = [], []
    for ax_row in axes:
        for ax in ax_row:
            h, l = ax.get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li not in labels:
                    handles.append(hi)
                    labels.append(li)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(handles), 3),
                   bbox_to_anchor=(0.5, 1.08))

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "accuracy_auroc_vs_groupsize.pdf")
    fig.savefig(save_path)
    fig.savefig(save_path.replace(".pdf", ".png"))
    print(f"Saved main figure to {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot COVID severity results from wandb")
    parser.add_argument("--entity", type=str, default="zadorlab",
                        help="wandb entity")
    parser.add_argument("--project", type=str, default="covid-severity",
                        help="wandb project name")
    parser.add_argument("--output_dir", type=str, default="figures",
                        help="Output directory for figures")
    args = parser.parse_args()

    print(f"Fetching runs from {args.entity}/{args.project}...")
    df = fetch_runs(args.entity, args.project)
    print(f"Fetched {len(df)} finished runs")

    if df.empty:
        print("No finished runs found.")
        return

    tf_df, bench_df = classify_runs(df)
    print(f"  TissueFormer runs: {len(tf_df)}, Benchmark runs: {len(bench_df)}")

    plot_accuracy_auroc_vs_groupsize(tf_df, bench_df, args.output_dir)
    print("Plotting complete.")


if __name__ == "__main__":
    main()
