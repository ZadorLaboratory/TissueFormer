"""
Per-brain predictions and SVC boundary plots for enucleated brains.

Loads existing TissueFormer predictions (trained on 4 control brains) and splits
them by animal to create per-brain scatter plots and SVC decision boundary plots
for each of the 3 enucleated brains (D077_2L, D078_2L, D079_4L).
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import reflect_points_to_left

from svc_plotting import (
    create_decision_boundary_plot_with_density_mask,
    load_ccf_boundaries,
    plot_scatter_style,
    SuppressOutput,
)

ROOT_DATA_PATH = os.environ["ROOT_DATA_PATH"]
GROUP_SIZE = 32
PRED_DIR = f"foldtest_enucleated_animal_name_class_weights2_{GROUP_SIZE}"
PRED_PATH = os.path.join(
    ROOT_DATA_PATH, "barseq", "annotation", PRED_DIR, "test_brain_predictions_cells.npy"
)
DATASET_PATH = os.path.join(
    ROOT_DATA_PATH, "barseq", "Chen2023", "train_test_barseq_all_exhausted_test_enucleated.dataset"
)
OUTPUT_DIR = os.path.dirname(__file__)

ENUCLEATED_ANIMALS = ["D077", "D078", "D079"]

# SVC hyperparameters
SVC_PARAMS = dict(kernel="rbf", gamma=1e-5, C=1)
PLOT_PARAMS = dict(
    grid_resolution=750,
    density_bandwidth=12,
    batch_size=4096,
    density_mask_alpha=0.95,
    density_threshold=0.04,
    subsample=2,
)


def build_colormap():
    """Build a colormap from area class IDs to colors, using the area mapping files."""
    files_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "files")
    with open(os.path.join(files_dir, "area_ancestor_id_map.json")) as f:
        area_ancestor_id_map = json.load(f)
    with open(os.path.join(files_dir, "area_name_map.json")) as f:
        area_name_map = json.load(f)

    area_name_map["0"] = "outside_brain"
    annotation2area_int = {0.0: 0}
    for a in area_ancestor_id_map:
        higher = area_ancestor_id_map[str(int(a))]
        annotation2area_int[float(a)] = higher[1] if len(higher) > 1 else int(a)

    unique_areas = np.unique(list(annotation2area_int.values()))
    area_classes = np.arange(len(unique_areas))
    id2id = {float(k): v for k, v in zip(unique_areas, area_classes)}
    id2id_rev = {v: k for k, v in id2id.items()}
    area_class2area_name = {k: area_name_map[str(int(v))] for k, v in id2id_rev.items()}

    # Create deterministic colormap from area names
    all_names = sorted(set(area_class2area_name.values()))
    colormaps = ["tab20", "tab20b", "tab20c"]
    colors = np.vstack([plt.cm.get_cmap(cmap)(np.linspace(0, 1, 20)) for cmap in colormaps])
    color_indices = np.arange(len(all_names)) % len(colors)
    name2color = dict(zip(all_names, colors[color_indices]))

    # Map from area class ID -> color
    name2class = {v: k for k, v in area_class2area_name.items()}
    color_map = {name2class[name]: name2color[name] for name in all_names if name in name2class}

    return color_map, area_class2area_name


def main():
    from datasets import load_from_disk
    from cuml.svm import SVC

    print("Loading predictions and dataset...")
    pred_dict = np.load(PRED_PATH, allow_pickle=True).item()
    dataset = load_from_disk(DATASET_PATH)
    test_ds = dataset["test"]

    predictions = pred_dict["predictions"]
    labels = pred_dict["labels"]
    indices = np.array(pred_dict["indices"])

    # Look up animal_name for each prediction using dataset indices
    animal_names = np.array(test_ds["animal_name"])[indices]
    ccf_streamlines = np.array(test_ds["CCF_streamlines"])[indices]

    print(f"Total test predictions: {len(predictions)}")

    color_map, area_class2area_name = build_colormap()
    bf_left_boundaries_flat = load_ccf_boundaries()

    for animal in ENUCLEATED_ANIMALS:
        print(f"\n{'='*60}")
        print(f"Processing {animal}...")
        mask = animal_names == animal
        n_cells = mask.sum()
        print(f"  Cells: {n_cells}")

        preds = predictions[mask]
        lbls = labels[mask]
        xyz = ccf_streamlines[mask]

        # Reflect to left hemisphere
        xyz_reflected = reflect_points_to_left(xyz[:, :2])
        x, y = xyz_reflected[:, 0], xyz_reflected[:, 1]

        # Accuracy
        acc = (preds == lbls).mean()
        print(f"  Accuracy: {acc:.4f}")

        # --- Scatter plot ---
        f, ax = plot_scatter_style(x, y, lbls, preds, color_map, bf_left_boundaries_flat, alpha=0.3)
        f.suptitle(f"{animal} enucleated (acc={acc:.3f})", fontsize=14)
        scatter_path = os.path.join(OUTPUT_DIR, f"{animal}_scatter_enucleated.png")
        f.savefig(scatter_path, dpi=200, bbox_inches="tight")
        plt.close(f)
        print(f"  Saved {scatter_path}")

        # --- SVC decision boundary plot ---
        print(f"  Fitting SVC (gamma={SVC_PARAMS['gamma']}, C={SVC_PARAMS['C']})...")
        xy = np.column_stack([x, y]).astype(np.float32)

        svc = SVC(**SVC_PARAMS)
        with SuppressOutput():
            svc.fit(xy, preds.astype(np.float32))

        fig, ax = plt.subplots(figsize=(10, 8))
        print(f"  Creating decision boundary plot...")
        create_decision_boundary_plot_with_density_mask(
            model=svc,
            X=xy,
            color_map=color_map,
            ax=ax,
            **PLOT_PARAMS,
        )
        # Overlay CCF boundaries
        for k, boundary_coords in bf_left_boundaries_flat.items():
            ax.plot(*boundary_coords.T, c="k", lw=0.5)
        ax.set_title(f"{animal} enucleated SVC boundaries (acc={acc:.3f})")
        svc_path = os.path.join(OUTPUT_DIR, f"{animal}_svc_enucleated.png")
        fig.savefig(svc_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {svc_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
