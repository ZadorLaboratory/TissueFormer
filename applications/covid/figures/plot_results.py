"""
Manuscript-quality figures for COVID severity classification results.

Produces:
  1. Main figure: Accuracy and AUROC vs group_size for TissueFormer + benchmarks
  2. Supplementary: Confusion matrices, ROC curves, cell-type composition
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
}


def load_results(results_dir: str) -> dict:
    """
    Load all result JSON files from the results directory.
    Returns nested dict: results[dataset][method][group_size][fold] = metrics
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for path in glob.glob(os.path.join(results_dir, "**/*_results.json"), recursive=True):
        with open(path) as f:
            data = json.load(f)
        # Parse from path or data
        metrics = data.get("metrics", {})
        gs = data.get("group_size", "unknown")
        clf = data.get("classifier_type", "unknown")
        feat = data.get("feature_type", "unknown")
        method = f"{clf}_{feat}"

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

        if dataset:
            results[dataset][method][gs][fold] = metrics

    return dict(results)


def _aggregate_across_folds(fold_dict):
    """Aggregate metrics across folds: return mean and std."""
    if not fold_dict:
        return None, None
    values = list(fold_dict.values())
    return np.mean(values), np.std(values)


def plot_accuracy_auroc_vs_groupsize(results, output_dir):
    """
    Main figure: accuracy and AUROC vs group_size (log scale).
    One column per dataset, two rows (accuracy, auroc).
    """
    fig, axes = plt.subplots(2, len(DATASETS), figsize=(4 * len(DATASETS), 6),
                             sharex=True, squeeze=False)

    for col, dataset in enumerate(DATASETS):
        if dataset not in results:
            continue

        ds_results = results[dataset]

        for method_key, style in METHODS.items():
            if method_key not in ds_results:
                continue

            gs_data = ds_results[method_key]
            x_vals, acc_means, acc_stds = [], [], []
            auroc_means, auroc_stds = [], []

            for gs in GROUP_SIZES:
                gs_str = str(gs)
                if gs_str not in gs_data:
                    continue

                fold_metrics = gs_data[gs_str]
                # Extract accuracy
                acc_values = []
                auroc_values = []
                for fold, metrics in fold_metrics.items():
                    # Find accuracy key
                    for k, v in metrics.items():
                        if "accuracy" in k:
                            acc_values.append(v)
                        if "auroc" in k:
                            auroc_values.append(v)

                if acc_values:
                    x_vals.append(gs)
                    acc_means.append(np.mean(acc_values))
                    acc_stds.append(np.std(acc_values))
                if auroc_values:
                    auroc_means.append(np.mean(auroc_values))
                    auroc_stds.append(np.std(auroc_values))

            if x_vals:
                axes[0, col].errorbar(
                    x_vals, acc_means, yerr=acc_stds,
                    color=style["color"], marker=style["marker"],
                    label=style["label"], capsize=3, linewidth=1.5, markersize=5
                )
            if auroc_means and len(auroc_means) == len(x_vals):
                axes[1, col].errorbar(
                    x_vals, auroc_means, yerr=auroc_stds,
                    color=style["color"], marker=style["marker"],
                    label=style["label"], capsize=3, linewidth=1.5, markersize=5
                )

            # Plot whole-donor benchmark as horizontal line
            if "all" in gs_data:
                for fold, metrics in gs_data["all"].items():
                    for k, v in metrics.items():
                        if "accuracy" in k:
                            axes[0, col].axhline(
                                y=v, color=style["color"], linestyle="--",
                                alpha=0.5, linewidth=1
                            )
                        if "auroc" in k:
                            axes[1, col].axhline(
                                y=v, color=style["color"], linestyle="--",
                                alpha=0.5, linewidth=1
                            )

        axes[0, col].set_title(DATASET_LABELS.get(dataset, dataset))
        axes[1, col].set_xlabel("Group size")
        axes[0, col].set_xscale("log", base=2)
        axes[1, col].set_xscale("log", base=2)

        for row in range(2):
            axes[row, col].xaxis.set_major_formatter(ScalarFormatter())
            axes[row, col].grid(True, alpha=0.3)

    axes[0, 0].set_ylabel("Accuracy")
    axes[1, 0].set_ylabel("AUROC")

    # Single legend at top
    handles, labels = axes[0, 0].get_legend_handles_labels()
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
