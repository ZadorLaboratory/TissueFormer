"""
Manuscript-quality figures for COVID severity classification results.

Produces:
  1. Main figure: Group accuracy, donor accuracy, and donor AUROC vs group_size
     for TissueFormer + benchmarks
  2. Supplementary: Confusion matrices
"""

import os
import json
import glob
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from matplotlib.ticker import ScalarFormatter


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
METHODS = {
    "tissueformer": {"color": "#2196F3", "marker": "o", "label": "TissueFormer"},
    "random_forest_pseudobulk": {"color": "#4CAF50", "marker": "s", "label": "RF (pseudobulk)"},
    "logistic_regression_pseudobulk": {"color": "#FF9800", "marker": "^", "label": "LR (pseudobulk)"},
    "random_forest_cell_type_histogram": {"color": "#9C27B0", "marker": "D", "label": "RF (cell type)"},
    "logistic_regression_cell_type_histogram": {"color": "#F44336", "marker": "v", "label": "LR (cell type)"},
    "cellcnn_dl": {"color": "#00BCD4", "marker": "P", "label": "CellCnn"},
    "scagg_dl": {"color": "#795548", "marker": "X", "label": "scAGG"},
    "scrat_dl": {"color": "#607D8B", "marker": "h", "label": "ScRAT"},
}

# Metric rows to plot: (row_label, metric_key_extractor)
# Each extractor takes a metrics dict and returns the value or None.
METRIC_ROWS = [
    ("Group Accuracy", "group_accuracy"),
    ("Donor Accuracy\n(mean logits)", "donor_meanlogits_accuracy"),
    ("Donor AUROC\n(mean logits)", "donor_meanlogits_auroc"),
]


def load_results(results_dir: str) -> dict:
    """
    Load all result JSON files from the results directory.

    Handles two formats:
    - Benchmark results: have classifier_type + feature_type fields,
      metrics stored in 'metrics' dict
    - TissueFormer results: have method='tissueformer',
      metrics stored in 'group_metrics', 'donor_majority_metrics',
      'donor_meanlogits_metrics' dicts

    Returns nested dict: results[dataset][method][group_size][fold] = metrics
    where metrics is a flat dict with keys like 'group_accuracy',
    'donor_meanlogits_accuracy', 'donor_meanlogits_auroc', etc.
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for path in glob.glob(os.path.join(results_dir, "**/*_results.json"), recursive=True):
        with open(path) as f:
            data = json.load(f)

        gs = data.get("group_size", "unknown")

        # Try to infer dataset and fold from directory structure
        parts = path.split(os.sep)
        dataset = None
        fold = 0
        for p in parts:
            for ds in DATASETS:
                if ds in p.lower():
                    dataset = ds
            if "fold" in p:
                try:
                    fold = int(p.split("fold")[-1].split("_")[0].split(".")[0])
                except ValueError:
                    pass

        if not dataset:
            continue

        if data.get("method") == "tissueformer":
            # TissueFormer result with multi-level metrics
            metrics = {}
            for sub_key in ("group_metrics", "donor_majority_metrics", "donor_meanlogits_metrics"):
                sub = data.get(sub_key, {})
                metrics.update(sub)
            results[dataset]["tissueformer"][gs][fold] = metrics
        else:
            # Benchmark result — store with donor_ prefix for consistency
            raw_metrics = data.get("metrics", {})
            metrics = {}
            for k, v in raw_metrics.items():
                # Benchmark metrics have prefix like "random_forest_pseudobulk_gsN_accuracy"
                # Extract the metric name (last part after the gs prefix)
                suffix = k.rsplit("_", 1)[-1] if "_" in k else k
                metrics[f"donor_meanlogits_{suffix}"] = v

            clf = data.get("classifier_type", "unknown")
            feat = data.get("feature_type", "unknown")
            method = f"{clf}_{feat}"
            results[dataset][method][gs][fold] = metrics

    return dict(results)


def _extract_metric_values(fold_metrics, metric_key):
    """Extract values for a metric key across folds."""
    values = []
    for fold, metrics in fold_metrics.items():
        if metric_key in metrics:
            values.append(metrics[metric_key])
    return values


def plot_accuracy_auroc_vs_groupsize(results, output_dir):
    """
    Main figure: group accuracy, donor accuracy, and donor AUROC vs group_size.
    One column per dataset, three rows.
    """
    n_rows = len(METRIC_ROWS)
    fig, axes = plt.subplots(n_rows, len(DATASETS),
                             figsize=(4 * len(DATASETS), 3 * n_rows),
                             sharex=True, squeeze=False)

    for col, dataset in enumerate(DATASETS):
        if dataset not in results:
            continue

        ds_results = results[dataset]

        for method_key, style in METHODS.items():
            if method_key not in ds_results:
                continue

            gs_data = ds_results[method_key]

            for row, (row_label, metric_key) in enumerate(METRIC_ROWS):
                x_vals, means, stds = [], [], []

                for gs in GROUP_SIZES:
                    gs_str = str(gs)
                    if gs_str not in gs_data:
                        continue

                    values = _extract_metric_values(gs_data[gs_str], metric_key)
                    if values:
                        x_vals.append(gs)
                        means.append(np.mean(values))
                        stds.append(np.std(values))

                if x_vals:
                    axes[row, col].errorbar(
                        x_vals, means, yerr=stds,
                        color=style["color"], marker=style["marker"],
                        label=style["label"], capsize=3, linewidth=1.5, markersize=5
                    )

                # Plot whole-donor benchmark as horizontal line
                if "all" in gs_data:
                    values = _extract_metric_values(gs_data["all"], metric_key)
                    if values:
                        for v in values:
                            axes[row, col].axhline(
                                y=v, color=style["color"], linestyle="--",
                                alpha=0.5, linewidth=1
                            )

        axes[0, col].set_title(DATASET_LABELS.get(dataset, dataset))
        axes[-1, col].set_xlabel("Group size")

        for row in range(n_rows):
            axes[row, col].set_xscale("log", base=2)
            axes[row, col].xaxis.set_major_formatter(ScalarFormatter())
            axes[row, col].grid(True, alpha=0.3)

    for row, (row_label, _) in enumerate(METRIC_ROWS):
        axes[row, 0].set_ylabel(row_label)

    # Single legend at top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(handles), 4),
                   bbox_to_anchor=(0.5, 1.12))

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "accuracy_auroc_vs_groupsize.pdf")
    fig.savefig(save_path)
    fig.savefig(save_path.replace(".pdf", ".png"))
    print(f"Saved main figure to {save_path}")
    plt.close(fig)


def plot_confusion_matrices(results_dir, output_dir):
    """Supplementary: confusion matrices per dataset per method (best group_size)."""
    label_names = ["control", "mild", "severe"]

    for path in glob.glob(os.path.join(results_dir, "**/*_results.json"), recursive=True):
        with open(path) as f:
            data = json.load(f)

        if "predictions" not in data or "labels" not in data:
            continue

        preds = np.array(data["predictions"])
        labels = np.array(data["labels"])
        clf = data.get("classifier_type", "")
        feat = data.get("feature_type", "")
        gs = data.get("group_size", "")

        # Get label names from data if available
        data_label_names = data.get("label_names", {})
        if data_label_names:
            n_labels = max(int(k) for k in data_label_names.keys()) + 1
            display_names = [data_label_names.get(str(i), str(i)) for i in range(n_labels)]
        else:
            display_names = label_names

        cm = confusion_matrix(labels, preds)
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(cm, display_labels=display_names)
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"{clf} ({feat}) gs={gs}")
        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        fname = f"confusion_{clf}_{feat}_gs{gs}.png"
        fig.savefig(os.path.join(output_dir, fname))
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot COVID severity results")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing result JSON files")
    parser.add_argument("--output_dir", type=str, default="figures",
                        help="Output directory for figures")
    args = parser.parse_args()

    results = load_results(args.results_dir)

    if results:
        plot_accuracy_auroc_vs_groupsize(results, args.output_dir)
        plot_confusion_matrices(args.results_dir, args.output_dir)
        print("Plotting complete.")
    else:
        print(f"No results found in {args.results_dir}")


if __name__ == "__main__":
    main()
