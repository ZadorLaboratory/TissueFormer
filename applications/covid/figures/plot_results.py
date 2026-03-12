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
N_CLASSES = 3  # control, mild, severe

# Majority-class fraction per dataset (donor-level), used as chance for accuracy metrics.
# For balanced accuracy, chance is always 1/N_CLASSES regardless of imbalance.
MAJORITY_CLASS_CHANCE = {
    "combat": 46 / 85,      # severe: 46, mild: 29, control: 10
    "ren": 83 / 185,         # severe: 83, mild: 77, control: 25
    "stevenson": 55 / 106,   # mild: 55, severe: 28, control: 23
    "combined": 161 / 376,   # mild: 161, severe: 157, control: 58
}
BALANCED_CHANCE = 1.0 / N_CLASSES

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

# Metric rows: (display_label, tissueformer_key, benchmark_suffix, is_balanced)
# Benchmark metrics are logged as {method}_gs{N}_{suffix}
# is_balanced: True → chance = 1/N_CLASSES; False → chance = majority class fraction
METRIC_ROWS = [
    ("Group Accuracy", "test/group_accuracy", "group_accuracy", False),
    ("Group Balanced\nAccuracy", "test/balanced_accuracy", "group_balanced_accuracy", True),
    ("Donor Accuracy\n(majority vote)", "test/donor_majority_accuracy", "donor_majority_accuracy", False),
    ("Donor Balanced Acc.\n(majority vote)", "test/donor_majority_balanced_accuracy", "donor_majority_balanced_accuracy", True),
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


def plot_accuracy_auroc_vs_groupsize(tf_df, bench_df, output_dir, benchmark_type="classical", sharex=True, sharey=True):
    """
    One figure per metric: datasets as columns, saved as separate files.
    benchmark_type: 'classical' or 'dl' — selects which benchmarks to plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    for row_label, tf_key, bench_suffix, is_balanced in METRIC_ROWS:
        fig, axes = plt.subplots(1, len(DATASETS),
                                 figsize=(4 * len(DATASETS), 3),
                                 sharex=sharex, sharey=sharey, squeeze=False)

        for col, dataset in enumerate(DATASETS):
            ax = axes[0, col]

            # Chance line (1/N for balanced accuracy, majority-class fraction for accuracy)
            chance = BALANCED_CHANCE if is_balanced else MAJORITY_CLASS_CHANCE[dataset]
            ax.axhline(chance, color="grey", linestyle=":", linewidth=0.8, label="Chance" if col == 0 else None)

            # --- TissueFormer ---
            tf_ds = tf_df[tf_df["dataset_name"] == dataset]
            style = METHODS["tissueformer"]
            if tf_key in tf_ds.columns:
                subset = tf_ds[["data.group_size", tf_key]].dropna()
                numeric_mask = pd.to_numeric(subset["data.group_size"], errors="coerce").notna()
                numeric_subset = subset[numeric_mask].copy()
                numeric_subset["data.group_size"] = pd.to_numeric(numeric_subset["data.group_size"])
                if not numeric_subset.empty:
                    grouped = numeric_subset.groupby("data.group_size")[tf_key]
                    means = grouped.mean()
                    stds = grouped.std().fillna(0)
                    ax.errorbar(
                        means.index, means.values, yerr=stds.values,
                        color=style["color"], marker=style["marker"],
                        label=style["label"], capsize=3, linewidth=1.5, markersize=5,
                    )
                all_subset = subset[subset["data.group_size"] == "all"]
                if not all_subset.empty:
                    all_mean = all_subset[tf_key].mean()
                    all_std = all_subset[tf_key].std() if len(all_subset) > 1 else 0
                    ax.errorbar(
                        [ALL_X_POS], [all_mean], yerr=[all_std],
                        color=style["color"], marker=style["marker"],
                        capsize=3, linestyle="none", markersize=5,
                    )

            # --- Benchmarks ---
            if bench_suffix is not None:
                bench_ds = bench_df[bench_df["dataset_name"] == dataset]
                bench_methods = _get_benchmark_methods(benchmark_type)
                for method_key, mstyle in bench_methods.items():
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
                        ax.errorbar(
                            x_vals, means, yerr=stds,
                            color=mstyle["color"], marker=mstyle["marker"],
                            label=mstyle["label"], capsize=3, linewidth=1.5, markersize=5,
                        )

                    all_col = _build_benchmark_col_name(method_key, "all", bench_suffix)
                    if all_col in bench_ds.columns:
                        values = pd.to_numeric(bench_ds[all_col], errors="coerce").dropna()
                        if len(values) > 0:
                            ax.errorbar(
                                [ALL_X_POS], [values.mean()],
                                yerr=[values.std() if len(values) > 1 else 0],
                                color=mstyle["color"], marker=mstyle["marker"],
                                capsize=3, linestyle="none", markersize=5,
                            )

            # Formatting
            ax.set_title(DATASET_LABELS.get(dataset, dataset))
            ax.set_xlabel("Group size")
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.grid(True, alpha=0.3)
            tick_positions = GROUP_SIZES + [ALL_X_POS]
            tick_labels = [str(gs) for gs in GROUP_SIZES] + ["all"]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)

            # Axis-break marks between 512 and "all"
            break_x = (512 * ALL_X_POS) ** 0.5
            trans = ax.get_xaxis_transform()
            bkwargs = dict(transform=trans, color="k", clip_on=False, linewidth=0.8)
            d = 0.015
            ax.plot((break_x * 0.92, break_x * 1.08), (-d, d), **bkwargs)
            ax.plot((break_x * 0.92, break_x * 1.08), (-d - 0.01, d - 0.01), **bkwargs)

        axes[0, 0].set_ylabel(row_label)

        # Legend — deduplicate
        handles, labels = [], []
        for ax in axes[0]:
            h, l = ax.get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li not in labels:
                    handles.append(hi)
                    labels.append(li)
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(len(handles), 6),
                       bbox_to_anchor=(0.5, 1.12))

        plt.tight_layout()
        # Filename from metric label (strip newlines, lowercase, underscores)
        metric_slug = row_label.replace("\n", " ").replace("(", "").replace(")", "").strip()
        metric_slug = "_".join(metric_slug.lower().split())
        save_path = os.path.join(output_dir, f"{metric_slug}_{benchmark_type}.pdf")
        fig.savefig(save_path)
        fig.savefig(save_path.replace(".pdf", ".png"))
        print(f"Saved {save_path}")
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
    parser.add_argument("--no-sharex", action="store_true",
                        help="Disable shared x-axis across subplot rows")
    parser.add_argument("--no-sharey", action="store_true",
                        help="Disable shared y-axis across subplot columns within each row")
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

    plot_accuracy_auroc_vs_groupsize(tf_df, bench_df, args.output_dir, args.benchmark_type,
                                     sharex=not args.no_sharex,
                                     sharey=not args.no_sharey)
    print("Plotting complete.")


if __name__ == "__main__":
    main()
