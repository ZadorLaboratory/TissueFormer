"""
Compute VISp (primary visual cortex, label 110) area across all 7 brains
using SVC decision boundary predictions on a uniform shared grid.

Control brains use pre-trained SVC models from each fold.
Enucleated brains get per-brain SVC models fit from TissueFormer predictions.
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load as joblib_load

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import reflect_points_to_left
from svc_plotting import SuppressOutput

ROOT_DATA_PATH = os.environ["ROOT_DATA_PATH"]
GROUP_SIZE = 32
GRID_RESOLUTION = 750
VISP_LABEL = 110
SVC_GAMMA = 1e-5
SVC_C = 1
OUTPUT_DIR = os.path.dirname(__file__)

# Control brains: fold -> (animal_name_in_dataset, display_name)
CONTROL_BRAINS = {
    0: ("D076", "D076_1L"),
    1: ("D077", "D077_1L"),
    2: ("D078", "D078_1L"),
    3: ("D079", "D079_3L"),
}

ENUCLEATED_ANIMALS = ["D077", "D078", "D079"]
ENUCLEATED_DISPLAY = {"D077": "D077_2L", "D078": "D078_2L", "D079": "D079_4L"}


def load_control_coords_and_models():
    """Load reflected coordinates and saved SVC models for each control brain."""
    from datasets import load_from_disk

    brains = {}
    for fold, (animal_ds_name, display_name) in CONTROL_BRAINS.items():
        fold_dir = f"fold{fold}_animal_name_class_weights2_{GROUP_SIZE}"
        base = os.path.join(ROOT_DATA_PATH, "barseq", "annotation", fold_dir)

        # Load SVC model
        svc_path = os.path.join(base, "svc_boundaries", "svm_gamma_0.00001000.joblib")
        model = joblib_load(svc_path)

        # Load coordinates from dataset
        dataset_path = os.path.join(
            ROOT_DATA_PATH, "barseq", "Chen2023",
            f"train_test_barseq_all_exhausted_fold{fold}.dataset",
        )
        dataset = load_from_disk(dataset_path)
        test_ds = dataset["test"]
        ccf = np.array(test_ds["CCF_streamlines"])
        xy = reflect_points_to_left(ccf[:, :2])

        brains[display_name] = {
            "condition": "control",
            "model": model,
            "xy": xy,
        }
        print(f"  Loaded control {display_name} (fold {fold}): {len(xy)} cells")

    return brains


def load_enucleated_coords_and_fit_models():
    """Load enucleated predictions, filter per brain, fit per-brain SVCs."""
    from datasets import load_from_disk
    from cuml.svm import SVC

    pred_dir = f"foldtest_enucleated_animal_name_class_weights2_{GROUP_SIZE}"
    pred_path = os.path.join(
        ROOT_DATA_PATH, "barseq", "annotation", pred_dir,
        "test_brain_predictions_cells.npy",
    )
    dataset_path = os.path.join(
        ROOT_DATA_PATH, "barseq", "Chen2023",
        "train_test_barseq_all_exhausted_test_enucleated.dataset",
    )

    pred_dict = np.load(pred_path, allow_pickle=True).item()
    dataset = load_from_disk(dataset_path)
    test_ds = dataset["test"]

    predictions = pred_dict["predictions"]
    indices = np.array(pred_dict["indices"])
    animal_names = np.array(test_ds["animal_name"])[indices]
    ccf_streamlines = np.array(test_ds["CCF_streamlines"])[indices]

    brains = {}
    for animal in ENUCLEATED_ANIMALS:
        display_name = ENUCLEATED_DISPLAY[animal]
        mask = animal_names == animal
        preds = predictions[mask]
        xyz = ccf_streamlines[mask]
        xy = reflect_points_to_left(xyz[:, :2])

        print(f"  Fitting SVC for {display_name}: {mask.sum()} cells...")
        xy_f32 = xy.astype(np.float32)
        svc = SVC(kernel="rbf", gamma=SVC_GAMMA, C=SVC_C)
        with SuppressOutput():
            svc.fit(xy_f32, preds.astype(np.float32))

        brains[display_name] = {
            "condition": "enucleated",
            "model": svc,
            "xy": xy,
        }

    return brains


def compute_shared_grid(all_brains):
    """Compute a shared grid from the union of all brains' coordinate ranges."""
    all_xy = np.vstack([b["xy"] for b in all_brains.values()])
    x_min, y_min = all_xy.min(axis=0)
    x_max, y_max = all_xy.max(axis=0)
    # Small padding
    eps = 0.01
    x_min -= eps
    y_min -= eps
    x_max += eps
    y_max += eps

    xx = np.linspace(x_min, x_max, GRID_RESOLUTION)
    yy = np.linspace(y_min, y_max, GRID_RESOLUTION)
    xx0, xx1 = np.meshgrid(xx, yy)
    X_grid = np.column_stack([xx0.ravel(), xx1.ravel()]).astype(np.float32)

    dx = (x_max - x_min) / GRID_RESOLUTION
    dy = (y_max - y_min) / GRID_RESOLUTION
    pixel_area = dx * dy

    return X_grid, pixel_area, (x_min, x_max, y_min, y_max)


def compute_visp_areas(all_brains, X_grid, pixel_area):
    """Predict on shared grid and count VISp pixels for each brain."""
    results = []
    for name, info in all_brains.items():
        model = info["model"]
        print(f"  Predicting on grid for {name}...")
        with SuppressOutput():
            preds = model.predict(X_grid)
        if hasattr(preds, "get"):
            preds = preds.get()
        preds = np.asarray(preds)

        visp_count = (preds == VISP_LABEL).sum()
        total_count = len(preds)
        visp_area = visp_count * pixel_area

        results.append({
            "brain": name,
            "condition": info["condition"],
            "visp_pixels": int(visp_count),
            "total_pixels": int(total_count),
            "visp_area": visp_area,
            "visp_fraction": visp_count / total_count,
        })
        print(f"    {name}: VISp pixels={visp_count}, area={visp_area:.2f}")

    return pd.DataFrame(results)


def plot_visp_comparison(df):
    """Two-column strip plot with individual points, means, and t-test significance bracket."""
    from scipy import stats

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.family": "sans-serif",
    })

    fig, ax = plt.subplots(figsize=(3.5, 4.5))

    colors = {"control": "#4878CF", "enucleated": "#E24A33"}
    x_positions = {"control": 0, "enucleated": 1}

    ctrl_vals = df[df["condition"] == "control"]["visp_area"].values
    enuc_vals = df[df["condition"] == "enucleated"]["visp_area"].values

    # Individual points with jitter
    rng = np.random.default_rng(42)
    for condition, vals in [("control", ctrl_vals), ("enucleated", enuc_vals)]:
        x_base = x_positions[condition]
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(
            np.full(len(vals), x_base) + jitter, vals,
            color=colors[condition], s=50, zorder=3, edgecolors="black", linewidths=0.5,
        )
        # Mean bar
        ax.hlines(vals.mean(), x_base - 0.15, x_base + 0.15,
                  color="black", linewidth=2, zorder=4)

    # T-test
    t_stat, p_val = stats.ttest_ind(ctrl_vals, enuc_vals)
    print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f}")

    # Significance bracket
    y_max = df["visp_area"].max()
    bracket_y = y_max * 1.08
    bar_y = y_max * 1.12
    ax.plot([0, 0, 1, 1], [bracket_y, bar_y, bar_y, bracket_y],
            color="black", linewidth=1, clip_on=False)
    if p_val < 0.05:
        sig_text = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
        ax.text(0.5, bar_y, sig_text, ha="center", va="bottom", fontsize=14, fontweight="bold")
    else:
        ax.text(0.5, bar_y, "N.S.", ha="center", va="bottom", fontsize=9)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Control", "Enucleated"])
    ax.set_ylabel("VISp area (grid units\u00b2)")
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, y_max * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        out_path = os.path.join(OUTPUT_DIR, f"visp_area_comparison.{ext}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {os.path.join(OUTPUT_DIR, 'visp_area_comparison.png')}")


def main():
    warnings.filterwarnings("ignore", module="cuml.*")

    print("Loading control brains...")
    control_brains = load_control_coords_and_models()

    print("Loading enucleated brains...")
    enucleated_brains = load_enucleated_coords_and_fit_models()

    all_brains = {**control_brains, **enucleated_brains}

    print("Computing shared grid...")
    X_grid, pixel_area, bounds = compute_shared_grid(all_brains)
    print(f"  Grid: {GRID_RESOLUTION}x{GRID_RESOLUTION}, pixel area={pixel_area:.4f}")
    print(f"  Bounds: x=[{bounds[0]:.1f}, {bounds[1]:.1f}], y=[{bounds[2]:.1f}, {bounds[3]:.1f}]")

    print("Computing VISp areas...")
    df = compute_visp_areas(all_brains, X_grid, pixel_area)

    # Print results table
    print("\n" + "=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)

    # Summary stats
    ctrl = df[df["condition"] == "control"]["visp_area"]
    enuc = df[df["condition"] == "enucleated"]["visp_area"]
    print(f"\nControl mean VISp area:     {ctrl.mean():.2f} ± {ctrl.std():.2f}")
    print(f"Enucleated mean VISp area:  {enuc.mean():.2f} ± {enuc.std():.2f}")
    if ctrl.mean() > 0:
        print(f"Ratio (enucleated/control): {enuc.mean() / ctrl.mean():.3f}")

    # Save outputs
    csv_path = os.path.join(OUTPUT_DIR, "visp_area_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    plot_visp_comparison(df)
    print("Done!")


if __name__ == "__main__":
    main()
