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
import matplotlib.gridspec as gridspec
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
    ("Accuracy", "test/group_accuracy", "group_accuracy", False),
    ("Balanced Accuracy", "test/balanced_accuracy", "group_balanced_accuracy", True),
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
        is_donor_majority = "donor_majority" in tf_key
        has_all_panel = not is_donor_majority

        # 2×2 layout; each cell holds a main panel (+ narrow "all" panel if applicable).
        n_rows, n_cols = 2, 2
        fig = plt.figure(figsize=(13.5, 6.5))
        outer = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.35, wspace=0.3)

        main_axes = []
        all_axes = []

        for idx, dataset in enumerate(DATASETS):
            row, col = divmod(idx, n_cols)
            if has_all_panel:
                inner = gridspec.GridSpecFromSubplotSpec(
                    1, 2, subplot_spec=outer[row, col],
                    width_ratios=[4, 1], wspace=0.05)
                ax = fig.add_subplot(inner[0, 0],
                                     sharey=main_axes[0] if (sharey and main_axes) else None)
                ax_all = fig.add_subplot(inner[0, 1], sharey=ax)
                all_axes.append(ax_all)
            else:
                ax = fig.add_subplot(outer[row, col],
                                     sharey=main_axes[0] if (sharey and main_axes) else None)
            main_axes.append(ax)

            # Chance line
            chance = BALANCED_CHANCE if is_balanced else MAJORITY_CLASS_CHANCE[dataset]
            ax.axhline(chance, color="red", linestyle=":", linewidth=0.8, label="Chance" if idx == 0 else None)
            if has_all_panel:
                ax_all.axhline(chance, color="red", linestyle=":", linewidth=0.8)

            # --- TissueFormer ---
            tf_ds = tf_df[tf_df["dataset_name"] == dataset]
            style = METHODS["tissueformer"]
            all_points = []  # (mean, std, style_dict) for the "all" panel
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
                if is_donor_majority:
                    pass
                elif tf_key == "test/balanced_accuracy":
                    # Use the max donor_majority_balanced_accuracy as the "all" point
                    donor_key = "test/donor_majority_balanced_accuracy"
                    if donor_key in tf_ds.columns:
                        donor_subset = tf_ds[["data.group_size", donor_key]].dropna()
                        donor_numeric = donor_subset[pd.to_numeric(donor_subset["data.group_size"], errors="coerce").notna()].copy()
                        donor_numeric["data.group_size"] = pd.to_numeric(donor_numeric["data.group_size"])
                        if not donor_numeric.empty:
                            donor_grouped = donor_numeric.groupby("data.group_size")[donor_key]
                            donor_means = donor_grouped.mean()
                            donor_stds = donor_grouped.std().fillna(0)
                            best_gs = donor_means.idxmax()
                            best_val = donor_means[best_gs]
                            best_std = donor_stds[best_gs]
                            all_points.append((best_val, best_std, style))
                            print(f"  [{dataset}] Transplanted TissueFormer donor_majority_balanced_accuracy "
                                  f"@ gs={int(best_gs)} ({best_val:.3f}) as 'all' point for balanced_accuracy")
                else:
                    all_subset = subset[subset["data.group_size"] == "all"]
                    if not all_subset.empty:
                        all_mean = all_subset[tf_key].mean()
                        all_std = all_subset[tf_key].std() if len(all_subset) > 1 else 0
                        all_points.append((all_mean, all_std, style))

            # --- Benchmarks ---
            if bench_suffix is not None:
                bench_ds = bench_df[bench_df["dataset_name"] == dataset]
                bench_methods = _get_benchmark_methods(benchmark_type)
                for method_key, mstyle in bench_methods.items():
                    x_vals, means, stds = [], [], []
                    for gs_val in GROUP_SIZES:
                        col_name = _build_benchmark_col_name(method_key, gs_val, bench_suffix)
                        if col_name not in bench_ds.columns:
                            continue
                        values = pd.to_numeric(bench_ds[col_name], errors="coerce").dropna()
                        if len(values) > 0:
                            x_vals.append(gs_val)
                            means.append(values.mean())
                            stds.append(values.std() if len(values) > 1 else 0)

                    if x_vals:
                        ax.errorbar(
                            x_vals, means, yerr=stds,
                            color=mstyle["color"], marker=mstyle["marker"],
                            label=mstyle["label"], capsize=3, linewidth=1.5, markersize=5,
                        )

                    if has_all_panel:
                        all_col = _build_benchmark_col_name(method_key, "all", bench_suffix)
                        if all_col in bench_ds.columns:
                            values = pd.to_numeric(bench_ds[all_col], errors="coerce").dropna()
                            if len(values) > 0:
                                all_points.append((values.mean(),
                                                   values.std() if len(values) > 1 else 0,
                                                   mstyle))

            # --- "All donor cells" panel ---
            if has_all_panel and all_points:
                all_points.sort(key=lambda t: t[0])
                for i, (mean, std, sty) in enumerate(all_points):
                    ax_all.errorbar(
                        [0], [mean], yerr=[std],
                        color=sty["color"], marker=sty["marker"],
                        capsize=3, linestyle="none", markersize=5,
                    )

            # --- Main axis formatting ---
            ax.set_title(DATASET_LABELS.get(dataset, dataset))
            ax.set_xlabel("# sampled cells")
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.grid(True, alpha=0.3)
            ax.set_xticks(GROUP_SIZES)
            ax.set_xticklabels([str(gs_val) for gs_val in GROUP_SIZES])

            # --- "All" panel formatting ---
            if has_all_panel:
                ax_all.set_xlim(-0.5, 0.5)
                ax_all.set_xticks([0])
                ax_all.set_xticklabels(["all\ndonor\ncells"], fontsize=8)
                ax_all.grid(True, alpha=0.3, axis="y")
                ax_all.tick_params(axis="y", labelleft=False)

        # Y-label on left-column panels (indices 0 and 2 in the 2×2 grid)
        for i, ax in enumerate(main_axes):
            if i % n_cols == 0:
                ax.set_ylabel(row_label)

        # Legend — deduplicate across all axes
        handles, labels = [], []
        for a in main_axes + all_axes:
            h, l = a.get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li not in labels:
                    handles.append(hi)
                    labels.append(li)
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(len(handles), 6),
                       bbox_to_anchor=(0.5, 1.02))
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

    bench_df = bench_df[bench_df["tags"].apply(lambda t: "balanced" in t if isinstance(t, list) else False)]

    print(f"  TissueFormer runs: {len(tf_df)} (filtered by 'with_val' tag), Benchmark runs: {len(bench_df)}")

    plot_accuracy_auroc_vs_groupsize(tf_df, bench_df, args.output_dir, args.benchmark_type,
                                     sharex=not args.no_sharex,
                                     sharey=not args.no_sharey)
    print("Plotting complete.")


if __name__ == "__main__":
    main()
