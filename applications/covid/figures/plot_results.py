"""
Manuscript-quality figures for COVID severity classification results.

Pulls results from wandb (project: covid-severity) and produces:
  1. Main figure: Group accuracy, donor accuracy, and donor AUROC vs group_size
     for TissueFormer + benchmarks
  2. Supplementary: Confusion matrices (TODO — requires logged artifacts)
"""

import os
import argparse

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

DATASETS = ["combat", "ren", "stevenson", "combined"]
DATASET_LABELS = {"combat": "COMBAT", "ren": "Ren et al.", "stevenson": "Stevenson et al.", "combined": "Combined"}
GROUP_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
ALL_X_POS = 2048  # x-position for the disconnected "all" point (2 log2-steps past 512)

# Method display config: maps method key -> plotting style
# Classical methods use {clf}_{feat} keys; DL methods use model name directly
CLASSICAL_METHODS = {
    "random_forest_pseudobulk": {"color": "#4CAF50", "marker": "s", "label": "RF (pseudobulk)"},
    "logistic_regression_pseudobulk": {"color": "#FF9800", "marker": "^", "label": "LR (pseudobulk)"},
    "random_forest_cell_type_histogram": {"color": "#9C27B0", "marker": "D", "label": "RF (cell type)"},
    "logistic_regression_cell_type_histogram": {"color": "#F44336", "marker": "v", "label": "LR (cell type)"},
}

DL_METHODS = {
    "cellcnn": {"color": "#00BCD4", "marker": "P", "label": "CellCnn"},
    "scagg": {"color": "#795548", "marker": "X", "label": "scAGG"},
    "scrat": {"color": "#607D8B", "marker": "h", "label": "ScRAT"},
}

TISSUEFORMER = {"tissueformer": {"color": "#2196F3", "marker": "o", "label": "TissueFormer"}}

# Combined dict for backward compatibility (used by plot_diagnostics)
METHODS = {**TISSUEFORMER, **CLASSICAL_METHODS, **DL_METHODS}

# Metric rows: (display_label, tissueformer_key, benchmark_suffix)
# Benchmark metrics are logged as {method}_gs{N}_{suffix}
METRIC_ROWS = [
    ("Group Accuracy", "test/group_accuracy", "group_accuracy"),
    ("Donor Accuracy\n(majority vote)", "test/donor_majority_accuracy", "donor_majority_accuracy"),
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


def _build_benchmark_col_name(method_key, gs, suffix):
    """Build the wandb metric column name for a benchmark method.

    Classical methods: {clf}_{feat}_gs{N}_{suffix}
      e.g. random_forest_pseudobulk_gs64_group_accuracy
    DL methods: {model}_gs{N}_{suffix}
      e.g. cellcnn_gs64_group_accuracy
    """
    return f"{method_key}_gs{gs}_{suffix}"


def _get_benchmark_methods(benchmark_type):
    """Return the benchmark methods dict for the given type."""
    if benchmark_type == "classical":
        return CLASSICAL_METHODS
    elif benchmark_type == "dl":
        return DL_METHODS
    else:
        raise ValueError(f"Unknown benchmark_type: {benchmark_type!r}. Use 'classical' or 'dl'.")


def plot_accuracy_auroc_vs_groupsize(tf_df, bench_df, output_dir, benchmark_type="classical"):
    """
    Main figure: group accuracy, donor accuracy, and donor AUROC vs group_size.
    One column per dataset, one row per metric.
    benchmark_type: 'classical' or 'dl' — selects which benchmarks to plot.
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
            # Numeric group sizes (main line)
            subset = tf_ds[["data.group_size", tf_key]].dropna()
            numeric_mask = pd.to_numeric(subset["data.group_size"], errors="coerce").notna()
            numeric_subset = subset[numeric_mask].copy()
            numeric_subset["data.group_size"] = pd.to_numeric(numeric_subset["data.group_size"])
            if not numeric_subset.empty:
                grouped = numeric_subset.groupby("data.group_size")[tf_key]
                means = grouped.mean()
                stds = grouped.std().fillna(0)
                axes[row, col].errorbar(
                    means.index, means.values, yerr=stds.values,
                    color=style["color"], marker=style["marker"],
                    label=style["label"], capsize=3, linewidth=1.5, markersize=5,
                )
            # "all" point (disconnected)
            all_subset = subset[subset["data.group_size"] == "all"]
            if not all_subset.empty:
                all_mean = all_subset[tf_key].mean()
                all_std = all_subset[tf_key].std() if len(all_subset) > 1 else 0
                axes[row, col].errorbar(
                    [ALL_X_POS], [all_mean], yerr=[all_std],
                    color=style["color"], marker=style["marker"],
                    capsize=3, linewidth=0, markersize=5,
                )

        # --- Benchmarks ---
        bench_ds = bench_df[bench_df["dataset_name"] == dataset]
        bench_methods = _get_benchmark_methods(benchmark_type)
        for method_key, style in bench_methods.items():

            for row, (row_label, _, bench_suffix) in enumerate(METRIC_ROWS):
                if bench_suffix is None:
                    continue

                x_vals, means, stds = [], [], []
                for gs in GROUP_SIZES:
                    col_name = _build_benchmark_col_name(method_key, gs, bench_suffix)
                    if col_name not in bench_ds.columns:
                        continue
                    values = pd.to_numeric(bench_ds[col_name], errors="coerce").dropna()
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

                # "all" point (disconnected)
                all_col = _build_benchmark_col_name(method_key, "all", bench_suffix)
                if all_col in bench_ds.columns:
                    values = pd.to_numeric(bench_ds[all_col], errors="coerce").dropna()
                    if len(values) > 0:
                        axes[row, col].errorbar(
                            [ALL_X_POS], [values.mean()],
                            yerr=[values.std() if len(values) > 1 else 0],
                            color=style["color"], marker=style["marker"],
                            capsize=3, linewidth=0, markersize=5,
                        )

        # Formatting
        axes[0, col].set_title(DATASET_LABELS.get(dataset, dataset))
        axes[-1, col].set_xlabel("Group size")
        for row in range(n_rows):
            axes[row, col].set_xscale("log", base=2)
            axes[row, col].xaxis.set_major_formatter(ScalarFormatter())
            axes[row, col].grid(True, alpha=0.3)
            # Custom x-ticks: numeric group sizes + "all"
            tick_positions = GROUP_SIZES + [ALL_X_POS]
            tick_labels = [str(gs) for gs in GROUP_SIZES] + ["all"]
            axes[row, col].set_xticks(tick_positions)
            axes[row, col].set_xticklabels(tick_labels)

    # Draw axis-break marks between 512 and "all"
    for col in range(len(DATASETS)):
        for row in range(n_rows):
            ax = axes[row, col]
            # Convert data coords to axes fraction for the break position
            # Place break marks at the geometric mean of 512 and ALL_X_POS
            break_x = (512 * ALL_X_POS) ** 0.5
            trans = ax.get_xaxis_transform()
            kwargs = dict(transform=trans, color="k", clip_on=False,
                          linewidth=0.8)
            d = 0.015  # size of break marks
            ax.plot((break_x * 0.92, break_x * 1.08), (-d, d), **kwargs)
            ax.plot((break_x * 0.92, break_x * 1.08), (-d - 0.01, d - 0.01), **kwargs)

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
        fig.legend(handles, labels, loc="upper center", ncol=min(len(handles), 4),
                   bbox_to_anchor=(0.5, 1.08))

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"accuracy_auroc_vs_groupsize_{benchmark_type}.pdf")
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
    parser.add_argument("--benchmark_type", type=str, default="classical",
                        choices=["classical", "dl"],
                        help="Which benchmarks to plot: 'classical' (RF/LR) or 'dl' (CellCnn/scAGG/ScRAT)")
    args = parser.parse_args()

    print(f"Fetching runs from {args.entity}/{args.project}...")
    df = fetch_runs(args.entity, args.project)
    print(f"Fetched {len(df)} finished runs")

    if df.empty:
        print("No finished runs found.")
        return

    tf_df, bench_df = classify_runs(df)
    tf_df = tf_df[tf_df["tags"].apply(lambda t: "with_val" in t if isinstance(t, list) else False)]
    print(f"  TissueFormer runs: {len(tf_df)} (filtered by 'with_val' tag), Benchmark runs: {len(bench_df)}")

    plot_accuracy_auroc_vs_groupsize(tf_df, bench_df, args.output_dir, args.benchmark_type)
    print("Plotting complete.")


if __name__ == "__main__":
    main()
