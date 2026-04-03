"""
Compute physical area of all brain regions across all 8 brains using SVC
decision boundary predictions on a uniform shared grid.

Control brains use pre-trained SVC models from each fold.
Enucleated brains get per-brain SVC models fit from TissueFormer predictions.

Outputs:
  - all_area_results.csv: area (mm²) per region per brain
  - visp_area_comparison.{png,pdf}: paired strip plot of VISp area
  - visp_ratio_comparison.{png,pdf}: VISp / total higher visual areas ratio
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
from utils import reflect_points_to_left, compute_flatmap_pixel_area_map, interpolate_pixel_area_to_grid
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

# D076_4L was predicted separately via predict_single_brain.py
D076_4L_PREDICTION_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "prediction_output"
)


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


def load_d076_4l():
    """Load D076_4L predictions from the single-brain pipeline and fit SVC."""
    from datasets import load_from_disk
    from cuml.svm import SVC

    pred_path = os.path.join(D076_4L_PREDICTION_DIR, "predictions.npy")
    dataset_path = os.path.join(D076_4L_PREDICTION_DIR, "tokenized.dataset")

    pred_dict = np.load(pred_path, allow_pickle=True).item()
    dataset = load_from_disk(dataset_path)

    predictions = pred_dict["predictions"]
    indices = np.array(pred_dict["indices"])  # (n_groups, group_size)
    all_ccf = np.array(dataset["CCF_streamlines"])

    # Average coordinates across each group
    group_ccf = all_ccf[indices]  # (n_groups, group_size, 3)
    ccf_mean = np.nanmean(group_ccf, axis=1)  # (n_groups, 3)

    xy = reflect_points_to_left(ccf_mean[:, :2])

    print(f"  Fitting SVC for D076_4L: {len(predictions)} groups...")
    xy_f32 = xy.astype(np.float32)
    svc = SVC(kernel="rbf", gamma=SVC_GAMMA, C=SVC_C)
    with SuppressOutput():
        svc.fit(xy_f32, predictions.astype(np.float32))

    return {"D076_4L": {
        "condition": "enucleated",
        "model": svc,
        "xy": xy,
    }}


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

    return X_grid, xx0, xx1, (x_min, x_max, y_min, y_max)


def load_label_names():
    """Load label class ID -> area name mapping from config."""
    import yaml
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "config", "data", "default.yaml"
    )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return {int(k): v for k, v in cfg["label_names"].items()}


def compute_all_areas(all_brains, X_grid, pixel_area_per_point, label_names):
    """Predict on shared grid and compute physical area for every region per brain.

    Returns a DataFrame with columns: brain, condition, label, area_name, area_mm2, pixel_count.
    """
    results = []
    for name, info in all_brains.items():
        model = info["model"]
        print(f"  Predicting on grid for {name}...")
        with SuppressOutput():
            preds = model.predict(X_grid)
        if hasattr(preds, "get"):
            preds = preds.get()
        preds = np.asarray(preds).astype(int)

        unique_labels = np.unique(preds)
        for label in unique_labels:
            mask = preds == label
            area_mm2 = pixel_area_per_point[mask].sum()
            results.append({
                "brain": name,
                "condition": info["condition"],
                "label": int(label),
                "area_name": label_names.get(label, f"unknown_{label}"),
                "area_mm2": area_mm2,
                "pixel_count": int(mask.sum()),
            })

        n_labels = len(unique_labels)
        visp_area = pixel_area_per_point[preds == VISP_LABEL].sum()
        print(f"    {name}: {n_labels} regions, VISp={visp_area:.4f} mm²")

    return pd.DataFrame(results)


def _paired_strip_plot(ctrl_vals, enuc_vals, litters, ylabel, filename):
    """Generic paired strip plot with littermate lines, means, and paired t-test."""
    from scipy import stats

    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "font.family": "sans-serif",
    })

    fig, ax = plt.subplots(figsize=(3.5, 4.5))
    colors = {"control": "#4878CF", "enucleated": "#E24A33"}
    x_ctrl, x_enuc = 0, 1

    # Connecting lines
    for c_val, e_val in zip(ctrl_vals, enuc_vals):
        ax.plot([x_ctrl, x_enuc], [c_val, e_val],
                color="gray", linewidth=0.8, zorder=2, alpha=0.6)

    # Points
    ax.scatter(np.full(len(ctrl_vals), x_ctrl), ctrl_vals,
               color=colors["control"], s=50, zorder=3, edgecolors="black", linewidths=0.5)
    ax.scatter(np.full(len(enuc_vals), x_enuc), enuc_vals,
               color=colors["enucleated"], s=50, zorder=3, edgecolors="black", linewidths=0.5)

    # Mean bars
    ax.hlines(ctrl_vals.mean(), x_ctrl - 0.15, x_ctrl + 0.15,
              color="black", linewidth=2, zorder=4)
    ax.hlines(enuc_vals.mean(), x_enuc - 0.15, x_enuc + 0.15,
              color="black", linewidth=2, zorder=4)

    # Paired t-test
    t_stat, p_val = stats.ttest_rel(ctrl_vals, enuc_vals)
    print(f"  {filename} paired t-test: t={t_stat:.3f}, p={p_val:.4f}")

    # Significance bracket
    y_max = max(ctrl_vals.max(), enuc_vals.max())
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
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, y_max * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        out_path = os.path.join(OUTPUT_DIR, f"{filename}.{ext}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.join(OUTPUT_DIR, f'{filename}.png')}")


def _get_paired_values(per_brain_df):
    """Given a df with brain/condition/value columns, return paired ctrl/enuc arrays and litters."""
    df = per_brain_df.copy()
    df["litter"] = df["brain"].str[:4]
    ctrl = df[df["condition"] == "control"].set_index("litter")
    enuc = df[df["condition"] == "enucleated"].set_index("litter")
    litters = sorted(ctrl.index.intersection(enuc.index))
    return ctrl.loc[litters], enuc.loc[litters], litters


# Labels for "higher visual areas" — those with "visual area" in the name, excluding VISp
HIGHER_VISUAL_AREA_LABELS = [
    113, 117, 120, 121, 123, 135, 136, 137, 138, 139, 159,
]


def plot_visp_comparison(df_all):
    """Plot VISp area comparison from the full per-area DataFrame."""
    visp = df_all[df_all["label"] == VISP_LABEL][["brain", "condition", "area_mm2"]].copy()
    ctrl, enuc, litters = _get_paired_values(visp)
    _paired_strip_plot(ctrl["area_mm2"].values, enuc["area_mm2"].values, litters,
                       "VISp area (mm²)", "visp_area_comparison")


def plot_visp_ratio(df_all):
    """Plot VISp / (VISp + higher visual areas) ratio."""
    visual_labels = [VISP_LABEL] + HIGHER_VISUAL_AREA_LABELS
    vis_df = df_all[df_all["label"].isin(visual_labels)].copy()

    # Sum area per brain for VISp vs all visual
    visp_per_brain = df_all[df_all["label"] == VISP_LABEL].groupby(
        ["brain", "condition"])["area_mm2"].sum().reset_index().rename(columns={"area_mm2": "visp"})
    total_vis_per_brain = vis_df.groupby(
        ["brain", "condition"])["area_mm2"].sum().reset_index().rename(columns={"area_mm2": "total_vis"})

    merged = visp_per_brain.merge(total_vis_per_brain, on=["brain", "condition"])
    merged["visp_ratio"] = merged["visp"] / merged["total_vis"]

    print("\n  VISp ratio (VISp / all visual areas):")
    for _, row in merged.iterrows():
        print(f"    {row['brain']}: {row['visp_ratio']:.3f}  "
              f"(VISp={row['visp']:.2f}, total={row['total_vis']:.2f} mm²)")

    ctrl, enuc, litters = _get_paired_values(merged)
    _paired_strip_plot(ctrl["visp_ratio"].values, enuc["visp_ratio"].values, litters,
                       "VISp / visual areas", "visp_ratio_comparison")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip computation, just re-plot from existing all_area_results.csv")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", module="cuml.*")

    csv_path = os.path.join(OUTPUT_DIR, "all_area_results.csv")

    if args.plot_only:
        print(f"Loading existing results from {csv_path}...")
        df = pd.read_csv(csv_path)
    else:
        print("Loading control brains...")
        control_brains = load_control_coords_and_models()

        print("Loading enucleated brains...")
        enucleated_brains = load_enucleated_coords_and_fit_models()

        print("Loading D076_4L...")
        d076_4l = load_d076_4l()

        all_brains = {**control_brains, **enucleated_brains, **d076_4l}

        print("Computing shared grid...")
        X_grid, xx0, xx1, bounds = compute_shared_grid(all_brains)
        x_min, x_max, y_min, y_max = bounds
        svc_dx = (x_max - x_min) / GRID_RESOLUTION
        svc_dy = (y_max - y_min) / GRID_RESOLUTION
        print(f"  Grid: {GRID_RESOLUTION}x{GRID_RESOLUTION}, SVC pixel={svc_dx:.4f}x{svc_dy:.4f} flatmap px")
        print(f"  Bounds: x=[{x_min:.1f}, {x_max:.1f}], y=[{y_min:.1f}, {y_max:.1f}]")

        print("Computing flatmap pixel area map...")
        flatmap_h5 = os.path.join(ROOT_DATA_PATH, "CCF_files", "flatmap_butterfly.h5")
        pixel_area_map = compute_flatmap_pixel_area_map(flatmap_h5)
        area_density = interpolate_pixel_area_to_grid(pixel_area_map, xx0, xx1)
        pixel_area_per_point = (area_density * svc_dx * svc_dy) / 1e6
        print(f"  Mean pixel area: {np.nanmean(pixel_area_per_point[pixel_area_per_point > 0]):.6f} mm²")

        label_names = load_label_names()

        print("Computing all region areas...")
        df = compute_all_areas(all_brains, X_grid, pixel_area_per_point.ravel(), label_names)

        df.to_csv(csv_path, index=False)
        print(f"\nSaved CSV: {csv_path} ({len(df)} rows)")

    # VISp summary
    visp = df[df["label"] == VISP_LABEL]
    ctrl = visp[visp["condition"] == "control"]["area_mm2"]
    enuc = visp[visp["condition"] == "enucleated"]["area_mm2"]
    print(f"\nControl mean VISp area:     {ctrl.mean():.4f} ± {ctrl.std():.4f} mm²")
    print(f"Enucleated mean VISp area:  {enuc.mean():.4f} ± {enuc.std():.4f} mm²")
    if ctrl.mean() > 0:
        print(f"Ratio (enucleated/control): {enuc.mean() / ctrl.mean():.3f}")

    print("\nPlotting VISp comparison...")
    plot_visp_comparison(df)

    print("\nPlotting VISp ratio...")
    plot_visp_ratio(df)

    print("Done!")


if __name__ == "__main__":
    main()
