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
DATASET_LABELS = {"combat": "COMBAT", "ren": "Ren et al.", "stevenson": "Stevenson\n et al.", "combined": "Combined"}
DATASET_STATS = {
    "combat": {"cells": 637266, "donors": 85},
    "ren": {"cells": 1456806, "donors": 185},
    "stevenson": {"cells": 585153, "donors": 106},
    "combined": {"cells": 2679225, "donors": 376},
}
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


def _plot_cell(ax, ax_all, tf_df, bench_df, dataset, benchmark_type,
               tf_key, bench_suffix, is_balanced, show_chance_label=False):
    """Plot one dataset cell: main panel + optional 'all donor cells' panel.

    Returns (all_points list) so the caller can inspect what was plotted.
    """
    is_donor_majority = "donor_majority" in tf_key
    has_all_panel = ax_all is not None

    # Chance line
    chance = BALANCED_CHANCE if is_balanced else MAJORITY_CLASS_CHANCE[dataset]
    ax.axhline(chance, color="red", linestyle=":", linewidth=0.8,
               label="Chance" if show_chance_label else None)
    if has_all_panel:
        ax_all.axhline(chance, color="red", linestyle=":", linewidth=0.8)

    # --- TissueFormer ---
    tf_ds = tf_df[tf_df["dataset_name"] == dataset]
    tf_style = METHODS["tissueformer"]
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
                color=tf_style["color"], marker=tf_style["marker"],
                label=tf_style["label"], capsize=3, linewidth=1.5, markersize=5,
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
                    all_points.append((best_val, best_std, tf_style))
                    print(f"  [{dataset}] Transplanted TissueFormer donor_majority_balanced_accuracy "
                          f"@ gs={int(best_gs)} ({best_val:.3f}) as 'all' point for balanced_accuracy")
        else:
            all_subset = subset[subset["data.group_size"] == "all"]
            if not all_subset.empty:
                all_mean = all_subset[tf_key].mean()
                all_std = all_subset[tf_key].std() if len(all_subset) > 1 else 0
                all_points.append((all_mean, all_std, tf_style))

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
                if tf_key == "test/balanced_accuracy":
                    # Transplant: use best donor_majority_balanced_accuracy across group sizes
                    donor_suffix = "donor_majority_balanced_accuracy"
                    donor_means_list, donor_stds_list, donor_gs_list = [], [], []
                    for gs_val in GROUP_SIZES:
                        dcol = _build_benchmark_col_name(method_key, gs_val, donor_suffix)
                        if dcol not in bench_ds.columns:
                            continue
                        vals = pd.to_numeric(bench_ds[dcol], errors="coerce").dropna()
                        if len(vals) > 0:
                            donor_means_list.append(vals.mean())
                            donor_stds_list.append(vals.std() if len(vals) > 1 else 0)
                            donor_gs_list.append(gs_val)
                    if donor_means_list:
                        best_idx = max(range(len(donor_means_list)), key=lambda j: donor_means_list[j])
                        all_points.append((donor_means_list[best_idx],
                                           donor_stds_list[best_idx],
                                           mstyle))
                        print(f"  [{dataset}] Transplanted {method_key} donor_majority_balanced_accuracy "
                              f"@ gs={donor_gs_list[best_idx]} ({donor_means_list[best_idx]:.3f}) "
                              f"as 'all' point for balanced_accuracy")
                else:
                    all_col = _build_benchmark_col_name(method_key, "all", bench_suffix)
                    if all_col in bench_ds.columns:
                        values = pd.to_numeric(bench_ds[all_col], errors="coerce").dropna()
                        if len(values) > 0:
                            all_points.append((values.mean(),
                                               values.std() if len(values) > 1 else 0,
                                               mstyle))

    # --- "All donor cells" panel ---
    if has_all_panel and all_points:
        # TissueFormer always rightmost; others sorted by value
        tf_pts = [p for p in all_points if p[2] is tf_style]
        other_pts = sorted([p for p in all_points if p[2] is not tf_style],
                           key=lambda t: t[0])
        ordered = other_pts + tf_pts
        n_pts = len(ordered)
        for i, (mean, std, sty) in enumerate(ordered):
            x = 0.12 * (i - (n_pts - 1) / 2)
            ax_all.errorbar(
                [x], [mean], yerr=[std],
                color=sty["color"], marker=sty["marker"],
                capsize=3, linestyle="none", markersize=5,
            )

    # --- Main axis formatting ---
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


def plot_accuracy_auroc_vs_groupsize(tf_df, bench_df, output_dir, benchmark_type="classical", sharex=True, sharey=True):
    """
    One figure per metric: datasets in a 2×2 grid, saved as separate files.
    benchmark_type: 'classical' or 'dl' — selects which benchmarks to plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    for row_label, tf_key, bench_suffix, is_balanced in METRIC_ROWS:
        is_donor_majority = "donor_majority" in tf_key
        has_all_panel = not is_donor_majority

        n_rows, n_cols = 2, 2
        fig = plt.figure(figsize=(13.5, 6.5))
        outer = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.45, wspace=0.2)

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
                ax_all = None
            main_axes.append(ax)

            _plot_cell(ax, ax_all, tf_df, bench_df, dataset, benchmark_type,
                       tf_key, bench_suffix, is_balanced, show_chance_label=(idx == 0))
            stats = DATASET_STATS.get(dataset, {})
            title = DATASET_LABELS.get(dataset, dataset)
            if stats:
                title += f"\n({stats['cells']:,} cells, {stats['donors']} donors)"
            ax.set_title(title)

        # Y-label on left-column panels
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


def plot_combined(tf_df, bench_df, output_dir, sharex=True, sharey=True):
    """
    Combined figure per metric: 4 rows (datasets) × 2 columns (Classical, DL).
    Each cell has a main panel + narrow 'all donor cells' panel.
    """
    os.makedirs(output_dir, exist_ok=True)
    bench_types = [("classical", "Classical Methods"), ("dl", "Deep Learning Methods")]

    for row_label, tf_key, bench_suffix, is_balanced in METRIC_ROWS:
        is_donor_majority = "donor_majority" in tf_key
        has_all_panel = not is_donor_majority

        n_ds = len(DATASETS)
        n_bt = len(bench_types)
        fig = plt.figure(figsize=(13.5, 3 * n_ds))
        outer = gridspec.GridSpec(n_ds, n_bt, figure=fig, hspace=0.35, wspace=0.2,
                                  left=0.12)

        main_axes = []
        all_axes_list = []

        for ds_idx, dataset in enumerate(DATASETS):
            for bt_idx, (bt_key, bt_label) in enumerate(bench_types):
                if has_all_panel:
                    inner = gridspec.GridSpecFromSubplotSpec(
                        1, 2, subplot_spec=outer[ds_idx, bt_idx],
                        width_ratios=[4, 1], wspace=0.05)
                    ax = fig.add_subplot(inner[0, 0])
                    ax_all = fig.add_subplot(inner[0, 1], sharey=ax)
                    all_axes_list.append(ax_all)
                else:
                    ax = fig.add_subplot(outer[ds_idx, bt_idx])
                    ax_all = None
                main_axes.append(ax)

                _plot_cell(ax, ax_all, tf_df, bench_df, dataset, bt_key,
                           tf_key, bench_suffix, is_balanced,
                           show_chance_label=(ds_idx == 0 and bt_idx == 0))

                # Column titles on top row only
                if ds_idx == 0:
                    ax.set_title(bt_label, fontsize=12, pad=12)

                ax.set_ylabel(row_label if bt_idx == 0 else "")

                # Only show x-axis label on bottom row
                if ds_idx < n_ds - 1:
                    ax.set_xlabel("")

        # Horizontal dataset labels to the left of each row
        for ds_idx, dataset in enumerate(DATASETS):
            # Get the vertical center of this row from the GridSpec
            row_top = outer[ds_idx, 0].get_position(fig).y1
            row_bot = outer[ds_idx, 0].get_position(fig).y0
            y_center = (row_top + row_bot) / 2
            stats = DATASET_STATS.get(dataset, {})
            label = DATASET_LABELS.get(dataset, dataset)
            if stats:
                label += f"\n({stats['cells']:,} cells,\n{stats['donors']} donors)"
            fig.text(-0.01, y_center, label,
                     ha="left", va="center", fontsize=12)

        # Legend — deduplicate across all axes
        handles, labels = [], []
        for a in main_axes + all_axes_list:
            h, l = a.get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li not in labels:
                    handles.append(hi)
                    labels.append(li)
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(len(handles), 8),
                       bbox_to_anchor=(0.5, 1.0), fontsize=11)

        metric_slug = row_label.replace("\n", " ").replace("(", "").replace(")", "").strip()
        metric_slug = "_".join(metric_slug.lower().split())
        save_path = os.path.join(output_dir, f"{metric_slug}_combined.pdf")
        fig.savefig(save_path)
        fig.savefig(save_path.replace(".pdf", ".png"))
        print(f"Saved {save_path}")
        plt.close(fig)


def save_results_csv(tf_df, bench_df, output_dir):
    """Save all plotted data to a CSV for programmatic consumption."""
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    for dataset in DATASETS:
        # --- TissueFormer ---
        tf_ds = tf_df[tf_df["dataset_name"] == dataset]
        for _, tf_key, bench_suffix, is_balanced in METRIC_ROWS:
            metric_name = bench_suffix or tf_key.split("/")[-1]
            if tf_key in tf_ds.columns:
                subset = tf_ds[["data.group_size", tf_key]].dropna()
                for gs_val, grp in subset.groupby("data.group_size"):
                    vals = pd.to_numeric(grp[tf_key], errors="coerce").dropna()
                    if len(vals) > 0:
                        rows.append({
                            "dataset": dataset,
                            "method": "tissueformer",
                            "group_size": gs_val,
                            "metric": metric_name,
                            "mean": vals.mean(),
                            "std": vals.std() if len(vals) > 1 else 0,
                            "n_runs": len(vals),
                        })

        # --- Benchmarks ---
        bench_ds = bench_df[bench_df["dataset_name"] == dataset]
        all_bench = {**CLASSICAL_METHODS, **DL_METHODS}
        for method_key in all_bench:
            for _, tf_key, bench_suffix, is_balanced in METRIC_ROWS:
                if bench_suffix is None:
                    continue
                for gs_val in GROUP_SIZES + ["all"]:
                    col_name = _build_benchmark_col_name(method_key, gs_val, bench_suffix)
                    if col_name not in bench_ds.columns:
                        continue
                    vals = pd.to_numeric(bench_ds[col_name], errors="coerce").dropna()
                    if len(vals) > 0:
                        rows.append({
                            "dataset": dataset,
                            "method": method_key,
                            "group_size": gs_val,
                            "metric": bench_suffix,
                            "mean": vals.mean(),
                            "std": vals.std() if len(vals) > 1 else 0,
                            "n_runs": len(vals),
                        })

    csv_df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "results.csv")
    csv_df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path} ({len(csv_df)} rows)")


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
    parser.add_argument("--combined", action="store_true",
                        help="Also generate combined 4×2 plots (datasets × benchmark type)")
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

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_results_csv(tf_df, bench_df, script_dir)

    plot_accuracy_auroc_vs_groupsize(tf_df, bench_df, args.output_dir, args.benchmark_type,
                                     sharex=not args.no_sharex,
                                     sharey=not args.no_sharey)
    if args.combined:
        plot_combined(tf_df, bench_df, args.output_dir,
                      sharex=not args.no_sharex,
                      sharey=not args.no_sharey)
    print("Plotting complete.")


if __name__ == "__main__":
    main()
