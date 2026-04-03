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


def load_all_brain_data():
    """Load coordinates and predictions for all 8 brains. Returns list of dicts."""
    from datasets import load_from_disk

    brains = []

    # Control brains
    for fold, (animal_ds_name, display_name) in CONTROL_BRAINS.items():
        fold_dir = f"fold{fold}_animal_name_class_weights2_{GROUP_SIZE}"
        base = os.path.join(ROOT_DATA_PATH, "barseq", "annotation", fold_dir)

        pred_path = os.path.join(base, "test_brain_predictions_cells.npy")
        pred_dict = np.load(pred_path, allow_pickle=True).item()
        predictions = pred_dict["predictions"]
        indices = np.array(pred_dict["indices"])

        dataset_path = os.path.join(
            ROOT_DATA_PATH, "barseq", "Chen2023",
            f"train_test_barseq_all_exhausted_fold{fold}.dataset",
        )
        dataset = load_from_disk(dataset_path)
        ccf = np.array(dataset["test"]["CCF_streamlines"])[indices]
        xy = reflect_points_to_left(ccf[:, :2])

        brains.append({
            "name": display_name, "condition": "control",
            "xy": xy, "predictions": predictions,
        })
        print(f"  Loaded {display_name}: {len(predictions)} cells")

    # Enucleated brains (D077, D078, D079)
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

    for animal in ENUCLEATED_ANIMALS:
        display_name = ENUCLEATED_DISPLAY[animal]
        mask = animal_names == animal
        preds = predictions[mask]
        xy = reflect_points_to_left(ccf_streamlines[mask, :2])
        brains.append({
            "name": display_name, "condition": "enucleated",
            "xy": xy, "predictions": preds,
        })
        print(f"  Loaded {display_name}: {mask.sum()} cells")

    # D076_4L
    d076_pred = np.load(os.path.join(D076_4L_PREDICTION_DIR, "predictions.npy"),
                        allow_pickle=True).item()
    d076_ds = load_from_disk(os.path.join(D076_4L_PREDICTION_DIR, "tokenized.dataset"))
    d076_indices = np.array(d076_pred["indices"])
    d076_ccf = np.array(d076_ds["CCF_streamlines"])
    group_ccf = d076_ccf[d076_indices]
    ccf_mean = np.nanmean(group_ccf, axis=1)
    xy = reflect_points_to_left(ccf_mean[:, :2])
    brains.append({
        "name": "D076_4L", "condition": "enucleated",
        "xy": xy, "predictions": d076_pred["predictions"],
    })
    print(f"  Loaded D076_4L: {len(d076_pred['predictions'])} groups")

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


def fit_predict_and_release(xy, predictions, X_grid):
    """Fit a cuML SVC, predict on the grid, and release GPU memory."""
    import gc
    from cuml.svm import SVC
    from svc_plotting import SuppressOutput

    svc = SVC(kernel="rbf", gamma=SVC_GAMMA, C=SVC_C)
    with SuppressOutput():
        svc.fit(xy.astype(np.float32), predictions.astype(np.float32))
        grid_preds = svc.predict(X_grid)
    if hasattr(grid_preds, "get"):
        grid_preds = grid_preds.get()
    grid_preds = np.asarray(grid_preds).astype(int)

    del svc
    gc.collect()
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except ImportError:
        pass

    return grid_preds


def compute_all_areas_from_grid_preds(all_grid_preds, pixel_area_per_point, label_names):
    """Compute physical area for every region per brain from pre-computed grid predictions.

    Returns a DataFrame with columns: brain, condition, label, area_name, area_mm2, pixel_count.
    """
    results = []
    for name, info in all_grid_preds.items():
        preds = info["grid_preds"]
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


BRAIN_ORDER = ["D076_1L", "D077_1L", "D078_1L", "D079_3L",
               "D076_4L", "D077_2L", "D078_2L", "D079_4L"]

HIGHER_VISUAL_LABELS = [113, 117, 120, 121, 123, 135, 136, 137, 138, 139, 159]


def plot_diagnostic_svc_maps(all_grid_preds, grid_shape):
    """Plot SVC prediction maps for all 8 brains on the shared grid."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for ax, name in zip(axes.ravel(), BRAIN_ORDER):
        preds = all_grid_preds[name]["grid_preds"].reshape(grid_shape)
        cond = all_grid_preds[name]["condition"]
        ax.imshow(preds, origin="lower", aspect="auto", cmap="tab20")
        ax.set_title(f"{name} ({cond})\n{len(np.unique(preds))} labels")
    plt.suptitle("SVC prediction maps on shared grid (all cuML)", fontsize=14)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "diagnostic_svc_maps.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_diagnostic_visp_hva(all_grid_preds, grid_shape):
    """Plot VISp (blue) vs Higher Visual Areas (red) vs other (gray) for all brains."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for ax, name in zip(axes.ravel(), BRAIN_ORDER):
        preds = all_grid_preds[name]["grid_preds"].reshape(grid_shape)
        rgb = np.ones((*preds.shape, 3)) * 0.9
        rgb[preds == VISP_LABEL] = [0.3, 0.5, 0.9]
        for lbl in HIGHER_VISUAL_LABELS:
            rgb[preds == lbl] = [0.9, 0.3, 0.2]
        ax.imshow(rgb, origin="lower", aspect="auto")
        visp_n = (preds == VISP_LABEL).sum()
        hva_n = sum((preds == lbl).sum() for lbl in HIGHER_VISUAL_LABELS)
        ax.set_title(f"{name}\nVISp={visp_n}px  HVA={hva_n}px")
    plt.suptitle("VISp (blue) vs Higher Visual Areas (red) vs Other (gray)", fontsize=14)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "diagnostic_visp_vs_hva.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip computation, just re-plot from existing all_area_results.csv")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Save diagnostic SVC maps (requires full computation)")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", module="cuml.*")

    csv_path = os.path.join(OUTPUT_DIR, "all_area_results.csv")

    if args.plot_only:
        print(f"Loading existing results from {csv_path}...")
        df = pd.read_csv(csv_path)
    else:
        print("Loading all brain data...")
        brain_data = load_all_brain_data()

        print("Computing shared grid...")
        # Build a temporary dict for compute_shared_grid
        tmp = {b["name"]: {"xy": b["xy"]} for b in brain_data}
        X_grid, xx0, xx1, bounds = compute_shared_grid(tmp)
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

        # Fit SVC, predict on grid, release GPU memory — one brain at a time
        print("Fitting SVCs and predicting on grid (one at a time for GPU memory)...")
        all_grid_preds = {}
        for brain in brain_data:
            name = brain["name"]
            print(f"  {name}...")
            grid_preds = fit_predict_and_release(
                brain["xy"], brain["predictions"], X_grid
            )
            all_grid_preds[name] = {
                "condition": brain["condition"],
                "grid_preds": grid_preds,
            }

        if args.diagnostics:
            grid_shape = (GRID_RESOLUTION, GRID_RESOLUTION)
            print("\nSaving diagnostic plots...")
            plot_diagnostic_svc_maps(all_grid_preds, grid_shape)
            plot_diagnostic_visp_hva(all_grid_preds, grid_shape)

        print("Computing all region areas...")
        df = compute_all_areas_from_grid_preds(
            all_grid_preds, pixel_area_per_point.ravel(), label_names
        )

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
