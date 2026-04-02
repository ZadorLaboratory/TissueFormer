#!/usr/bin/env python
"""
Tokenize a single h5ad file, predict brain area labels with a pretrained
TissueFormer model, and optionally compute SVC decision boundaries.

Stages are checkpointed to disk so the script can resume after failures:
  Stage 1: Tokenize  ->  <output_dir>/tokenized.dataset/
  Stage 2: Predict   ->  <output_dir>/predictions.npy
  Stage 3: SVC plot  ->  <output_dir>/svc_boundaries.png
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILES_DIR = os.path.join(SCRIPT_DIR, "data", "files")
ANNDATAS_DIR = os.path.join(SCRIPT_DIR, "data", "anndatas")

DEFAULT_H5AD = os.path.join(
    SCRIPT_DIR, "data", "anndatas_revised", "filt_neurons_D076_4L_CCFv2_newtypes.h5ad"
)
DEFAULT_CHECKPOINT = (
    "/home/benjami/mnt/zador_data_norepl/Ari/transcriptomics/"
    "checkpoints/outputs/foldtest_enucleated_animal_name_class_weights2_32/model"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict brain areas for a single h5ad file")
    parser.add_argument("--h5ad-path", default=DEFAULT_H5AD, help="Path to h5ad file")
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT, help="Model checkpoint")
    parser.add_argument("--output-dir", default="./prediction_output", help="Output directory")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--nproc", type=int, default=24)
    parser.add_argument("--skip-svc", action="store_true", help="Skip SVC boundary stage")
    parser.add_argument("--start-stage", type=int, default=1, choices=[1, 2, 3],
                        help="Force re-run from this stage onward")
    return parser.parse_args()


def build_annotation_mapping():
    """Build CCFano -> area_class and area_class -> area_name mappings."""
    with open(os.path.join(FILES_DIR, "area_ancestor_id_map.json")) as f:
        area_ancestor_id_map = json.load(f)
    with open(os.path.join(FILES_DIR, "area_name_map.json")) as f:
        area_name_map = json.load(f)

    area_name_map["0"] = "outside_brain"
    annotation2area_int = {0.0: 0}
    for a in area_ancestor_id_map.keys():
        higher_area_id = (
            area_ancestor_id_map[str(int(a))][1]
            if len(area_ancestor_id_map[str(int(a))]) > 1
            else a
        )
        annotation2area_int[float(a)] = higher_area_id

    unique_areas = np.unique(list(annotation2area_int.values()))
    area_classes = np.arange(len(unique_areas))
    id2id = {float(k): v for k, v in zip(unique_areas, area_classes)}
    annoation2area_class = {k: id2id[int(v)] for k, v in annotation2area_int.items()}
    id2id_rev = {v: k for k, v in id2id.items()}
    area_class2area_name = {k: area_name_map[str(int(v))] for k, v in id2id_rev.items()}

    return annoation2area_class, area_class2area_name


# ── Stage 1: Tokenize ─────────────────────────────────────────────────────────

def stage_tokenize(h5ad_path, output_dir, annoation2area_class, nproc):
    """Tokenize a single h5ad file and save to disk."""
    from tissueformer.tokenizer import TranscriptomeTokenizer

    dataset_path = os.path.join(output_dir, "tokenized.dataset")
    original_h5ad_path = h5ad_path  # preserve for animal name extraction

    with open(os.path.join(FILES_DIR, "barseq_gene_panel.pkl"), "rb") as f:
        gene_panel = pickle.load(f)

    labels = {"CCF", "CCF_streamlines", "H1_type", "CCFname", "CCFparentname",
              "id", "H2_type", "CCFano", "H3_type"}
    label_dict = {label: label for label in labels}

    tk = TranscriptomeTokenizer(
        label_dict,
        gene_median_file=None,
        token_dictionary_file=os.path.join(FILES_DIR, "barseq_token_dict_cls.pkl"),
        gene_panel=gene_panel,
        nproc=nproc,
    )

    # Check if var_names need fixing (some revised files have integer indices
    # instead of Ensembl IDs). If so, create a corrected temp file.
    import anndata as ad
    adata = ad.read_h5ad(h5ad_path)
    if not adata.var_names[0].startswith("ENSMUSG"):
        print("  Fixing var_names: mapping integer indices to Ensembl IDs from gene panel")
        assert len(adata.var_names) == len(gene_panel), (
            f"var_names length {len(adata.var_names)} != gene_panel length {len(gene_panel)}"
        )
        adata.var_names = gene_panel
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False)
        adata.write_h5ad(tmp.name)
        h5ad_path = tmp.name
        print(f"  Wrote corrected file to {h5ad_path}")
    del adata

    print(f"Tokenizing {h5ad_path} ...")
    dataset, _ = tk.tokenize_data(h5ad_path, "/tmp", "tmp", save=False)
    print(f"  Cells before filtering: {len(dataset)}")

    dataset = dataset.filter(lambda x: not np.isnan(np.sum(x["CCF_streamlines"])))
    dataset = dataset.map(lambda x: {"area_label": annoation2area_class[x["CCFano"]]})

    animal_name = os.path.basename(original_h5ad_path).split("_")[2]
    dataset = dataset.map(lambda x: {"animal_name": animal_name})

    # Prepare for trainer
    dataset = dataset.rename_column("area_label", "labels")
    dataset = dataset.add_column("uuid", list(range(len(dataset))))

    print(f"  Cells after filtering: {len(dataset)}")
    dataset.save_to_disk(dataset_path)
    print(f"  Saved to {dataset_path}")
    return dataset


# ── Stage 2: Predict ───────────────────────────────────────────────────────────

def stage_predict(output_dir, checkpoint_path, group_size):
    """Load model, run prediction, save results."""
    from datasets import load_from_disk
    from transformers import TrainingArguments

    from tissueformer.model import TissueFormer
    from tissueformer.samplers import GroupedSpatialTrainer

    dataset_path = os.path.join(output_dir, "tokenized.dataset")
    pred_path = os.path.join(output_dir, "predictions.npy")

    print(f"Loading dataset from {dataset_path} ...")
    dataset = load_from_disk(dataset_path)

    print(f"Loading model from {checkpoint_path} ...")
    model = TissueFormer.from_pretrained(checkpoint_path, num_labels=290)
    model.class_weights = None  # not needed for inference; avoids list vs tensor issue

    training_args = TrainingArguments(
        output_dir=output_dir,
        remove_unused_columns=False,
        per_device_eval_batch_size=4096,
        per_device_train_batch_size=group_size,
        report_to="none",
    )

    trainer = GroupedSpatialTrainer(
        model=model,
        args=training_args,
        spatial_group_size=group_size,
        spatial_label_key="labels",
        coordinate_key="CCF_streamlines",
        relative_positions=False,
        absolute_Z=False,
        hex_scaling=1.2,
        reflect_points=True,
        sampling_strategy="random",
        use_train_hex_grid_on_eval=True,
        max_radius_expansions=5,
        group_within_keys=None,
    )

    trainer.accelerator.gradient_state._reset_state()

    print("Running prediction ...")
    output, indices = trainer.predict(dataset)

    if isinstance(output.predictions, tuple):
        predictions = np.argmax(output.predictions[0], axis=-1)
    else:
        predictions = np.argmax(output.predictions, axis=-1)
    labels = output.label_ids[0] if isinstance(output.label_ids, tuple) else output.label_ids

    acc = (predictions == labels).mean()
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Predictions: {len(predictions)}, unique labels: {len(np.unique(predictions))}")

    output_dict = {
        "predictions": predictions,
        "labels": labels,
        "indices": indices,
    }
    np.save(pred_path, output_dict)
    print(f"  Saved to {pred_path}")


# ── Stage 3: SVC boundaries ───────────────────────────────────────────────────

def stage_svc(output_dir, area_class2area_name):
    """Fit SVC on predictions and plot decision boundaries."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cuml.svm import SVC
    from datasets import load_from_disk

    sys.path.insert(0, os.path.join(SCRIPT_DIR, "paper_figures", "fig3"))
    sys.path.insert(0, os.path.join(SCRIPT_DIR, "paper_figures"))
    from utils import reflect_points_to_left
    from svc_plotting import (
        create_decision_boundary_plot_with_density_mask,
        create_master_colormap,
        load_ccf_boundaries,
        SuppressOutput,
    )

    dataset_path = os.path.join(output_dir, "tokenized.dataset")
    pred_path = os.path.join(output_dir, "predictions.npy")
    svc_path = os.path.join(output_dir, "svc_boundaries.png")

    print("Loading predictions and dataset for SVC ...")
    pred_dict = np.load(pred_path, allow_pickle=True).item()
    dataset = load_from_disk(dataset_path)

    predictions = pred_dict["predictions"]
    indices = np.array(pred_dict["indices"])  # (n_groups, group_size)
    all_ccf = np.array(dataset["CCF_streamlines"])  # (n_cells, 3)

    # Average coordinates across each group to get one point per prediction
    group_ccf = all_ccf[indices]  # (n_groups, group_size, 3)
    ccf_streamlines = np.nanmean(group_ccf, axis=1)  # (n_groups, 3)

    xy = reflect_points_to_left(ccf_streamlines[:, :2]).astype(np.float32)

    # Build colormap from control animals
    import anndata as ad
    control_files = [
        "filt_neurons_D076_1L_CCFv2_newtypes.h5ad",
        "filt_neurons_D077_1L_CCFv2_newtypes.h5ad",
        "filt_neurons_D078_1L_CCFv2_newtypes.h5ad",
        "filt_neurons_D079_3L_CCFv2_newtypes.h5ad",
    ]
    annoation2area_class, _ = build_annotation_mapping()
    adata_list = []
    for fname in control_files:
        fpath = os.path.join(ANNDATAS_DIR, fname)
        adata = ad.read_h5ad(fpath)
        adata.obs["CCFano"] = adata.obs["CCFano"].astype("category")
        adata.obs["area_label"] = adata.obs["CCFano"].map(annoation2area_class).astype("category")
        adata.obs["area_name"] = adata.obs["area_label"].map(area_class2area_name).astype("category")
        adata = adata[adata.obs["area_name"] != "outside_brain"]
        subcortical_mask = np.isnan(adata.obsm["CCF_streamlines"]).any(axis=1)
        adata = adata[~subcortical_mask]
        adata_list.append(adata)

    color_map, label_names = create_master_colormap(adata_list, area_class2area_name)

    # Ensure all predicted class IDs have a color entry
    all_ids = set(predictions)
    colormaps = ["tab20", "tab20b", "tab20c"]
    fallback_colors = np.vstack(
        [plt.colormaps.get_cmap(cmap)(np.linspace(0, 1, 20)) for cmap in colormaps]
    )
    for cls_id in all_ids:
        if cls_id not in color_map:
            color_map[cls_id] = fallback_colors[int(cls_id) % len(fallback_colors)]

    # Fit SVC
    print("Fitting SVC ...")
    svc = SVC(kernel="rbf", gamma=1e-5, C=1)
    with SuppressOutput():
        svc.fit(xy, predictions.astype(np.float32))

    # Plot
    print("Creating boundary plot ...")
    fig, ax = plt.subplots(figsize=(10, 8))
    create_decision_boundary_plot_with_density_mask(
        model=svc,
        X=xy,
        color_map=color_map,
        ax=ax,
        grid_resolution=750,
        density_bandwidth=12,
        batch_size=4096,
        density_mask_alpha=0.95,
        density_threshold=0.04,
        subsample=2,
    )

    # Overlay CCF boundaries
    bf_left_boundaries_flat = load_ccf_boundaries()
    for k, boundary_coords in bf_left_boundaries_flat.items():
        ax.plot(*boundary_coords.T, c="w", lw=0.25)

    animal_name = os.path.basename(dataset_path).split("_")[0]  # fallback
    # Try to get from dataset
    if "animal_name" in dataset.column_names:
        animal_name = dataset["animal_name"][0]

    acc = (predictions == pred_dict["labels"]).mean()
    ax.set_title(f"{animal_name} SVC boundaries (acc={acc:.3f})")
    fig.savefig(svc_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved to {svc_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    annoation2area_class, area_class2area_name = build_annotation_mapping()

    def should_run(stage_num, output_path):
        """Run if forced by --start-stage, or if the output doesn't exist yet."""
        if stage_num < args.start_stage:
            return False  # earlier stage, don't redo
        if stage_num >= args.start_stage and args.start_stage > 1:
            return True  # forced re-run from start_stage onward
        return not os.path.exists(output_path)

    # Stage 1: Tokenize
    tok_path = os.path.join(args.output_dir, "tokenized.dataset")
    if should_run(1, tok_path):
        stage_tokenize(args.h5ad_path, args.output_dir, annoation2area_class, args.nproc)
    else:
        print("Stage 1 (tokenize): skipping, output exists")

    # Stage 2: Predict
    pred_path = os.path.join(args.output_dir, "predictions.npy")
    if should_run(2, pred_path):
        stage_predict(args.output_dir, args.checkpoint_path, args.group_size)
    else:
        print("Stage 2 (predict): skipping, output exists")

    # Stage 3: SVC
    svc_path = os.path.join(args.output_dir, "svc_boundaries.png")
    if args.skip_svc:
        print("Stage 3 (SVC): skipped by --skip-svc")
    elif should_run(3, svc_path):
        stage_svc(args.output_dir, area_class2area_name)
    else:
        print("Stage 3 (SVC): skipping, output exists")

    print("\nDone!")


if __name__ == "__main__":
    main()
