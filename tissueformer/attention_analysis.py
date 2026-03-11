"""Attention interpretability analysis for TissueFormer.

Collects attention weights from the Set Transformer layers and aggregates
them by cell type to understand which cell types receive the most attention
for each predicted label.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


@dataclass
class AttentionResults:
    """Container for collected attention weights and metadata."""
    attentions: List[np.ndarray] = field(default_factory=list)  # per-group (n_layers, group_size, group_size)
    cell_types: List[List[str]] = field(default_factory=list)   # per-group list of cell type strings
    predictions: np.ndarray = None  # (n_groups,) predicted label indices
    labels: np.ndarray = None       # (n_groups,) true label indices
    label_names: Dict[int, str] = field(default_factory=dict)


class AttentionCollector:
    """Collects attention weights from a trained TissueFormer model."""

    def __init__(
        self,
        model,
        dataset,
        collator,
        sampler,
        cell_type_key: str = "cell_type",
        label_names: Optional[Dict] = None,
        batch_size: int = 64,
        max_groups: Optional[int] = None,
        num_workers: int = 0,
    ):
        self.model = model
        self.dataset = dataset
        self.collator = collator
        self.sampler = sampler
        self.cell_type_key = cell_type_key
        self.label_names = label_names or {}
        self.batch_size = batch_size
        self.max_groups = max_groups
        self.num_workers = num_workers

    def collect(self) -> AttentionResults:
        """Run inference and collect attention weights for all groups."""
        device = next(self.model.parameters()).device
        self.model.eval()

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            pin_memory=False,
        )

        all_attentions = []
        all_cell_types = []
        all_preds = []
        all_labels = []
        total_groups = 0

        with torch.no_grad():
            for batch in dataloader:
                # Extract metadata before sending to model
                batch_cell_types = batch.pop(self.cell_type_key, None)

                # Extract labels before sending to model (avoid loss computation)
                batch_labels = batch.pop("labels", None)
                batch.pop("single_cell_labels", None)

                # Move tensors to device, skip non-tensor items
                model_inputs = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        model_inputs[k] = v.to(device)

                outputs = self.model(
                    **model_inputs,
                    output_attentions=True,
                    return_dict=True,
                )

                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                labels = batch_labels.numpy() if batch_labels is not None else np.full(len(preds), -1)

                # attentions: tuple of (batch_size, group_size, group_size) per layer
                attn_tuple = outputs.attentions
                n_groups_in_batch = preds.shape[0]

                for i in range(n_groups_in_batch):
                    if self.max_groups is not None and total_groups >= self.max_groups:
                        break

                    # Stack all layers: (n_layers, group_size, group_size)
                    group_attn = np.stack(
                        [layer[i].cpu().to(torch.float16).numpy() for layer in attn_tuple],
                        axis=0,
                    )
                    all_attentions.append(group_attn)

                    if batch_cell_types is not None:
                        all_cell_types.append(batch_cell_types[i])
                    else:
                        all_cell_types.append([])

                    all_preds.append(preds[i])
                    all_labels.append(labels[i])
                    total_groups += 1

                if self.max_groups is not None and total_groups >= self.max_groups:
                    break

        return AttentionResults(
            attentions=all_attentions,
            cell_types=all_cell_types,
            predictions=np.array(all_preds),
            labels=np.array(all_labels),
            label_names={int(k): v for k, v in self.label_names.items()},
        )


def aggregate_attention_by_cell_type(
    results: AttentionResults,
    layer_idx: int = -1,
) -> pd.DataFrame:
    """Compute mean attention received per cell, mapped to cell type.

    For each group, takes the column-mean of the attention matrix at
    ``layer_idx`` (how much attention flows TO each cell), then records
    each cell's attention alongside its cell type and the group's label.

    Returns a DataFrame with columns:
        [cell_type, mean_attention_received, label, group_idx]
    """
    rows = []
    for group_idx, (attn, ctypes) in enumerate(
        zip(results.attentions, results.cell_types)
    ):
        if not ctypes:
            continue
        # attn shape: (n_layers, group_size, group_size)
        attn_matrix = attn[layer_idx].astype(np.float32)  # (gs, gs)
        col_mean = attn_matrix.mean(axis=0)  # attention received per cell

        label_idx = int(results.predictions[group_idx])
        label_name = results.label_names.get(label_idx, str(label_idx))

        for cell_idx, (ct, attn_val) in enumerate(zip(ctypes, col_mean)):
            rows.append({
                "cell_type": ct,
                "mean_attention_received": float(attn_val),
                "label": label_name,
                "group_idx": group_idx,
            })

    return pd.DataFrame(rows)


def cell_type_attention_summary(results: AttentionResults, layer_idx: int = -1) -> pd.DataFrame:
    """Summarise mean attention per (label, cell_type) pair.

    Returns a DataFrame with columns:
        [label, cell_type, mean_attention, sem, count]
    where ``count`` is the number of cells contributing to the mean
    and ``sem`` is the standard error of the mean.
    """
    df = aggregate_attention_by_cell_type(results, layer_idx=layer_idx)
    if df.empty:
        return pd.DataFrame(columns=["label", "cell_type", "mean_attention", "sem", "count"])

    summary = (
        df.groupby(["label", "cell_type"])["mean_attention_received"]
        .agg(["mean", "sem", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_attention"})
    )
    summary["sem"] = summary["sem"].fillna(0)
    return summary.sort_values("mean_attention", ascending=False)


def cell_type_total_attention_summary(
    results: AttentionResults, layer_idx: int = -1
) -> pd.DataFrame:
    """Total attention captured per cell type, averaged across groups.

    Unlike :func:`cell_type_attention_summary` which averages per-cell
    attention (upweighting rare types), this sums attention within each
    group first, then averages across groups.  The result reflects how
    much of the group's total attention each cell type captures,
    accounting for its abundance.

    Returns a DataFrame with columns:
        [label, cell_type, mean_total_attention, sem, n_groups]
    """
    df = aggregate_attention_by_cell_type(results, layer_idx=layer_idx)
    if df.empty:
        return pd.DataFrame(
            columns=["label", "cell_type", "mean_total_attention", "sem", "n_groups"]
        )

    # Sum attention per (group, label, cell_type) — total attention captured
    group_totals = (
        df.groupby(["group_idx", "label", "cell_type"])["mean_attention_received"]
        .sum()
        .reset_index()
        .rename(columns={"mean_attention_received": "total_attention"})
    )

    # Average across groups
    summary = (
        group_totals.groupby(["label", "cell_type"])["total_attention"]
        .agg(["mean", "sem", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_total_attention", "count": "n_groups"})
    )
    summary["sem"] = summary["sem"].fillna(0)
    return summary.sort_values("mean_total_attention", ascending=False)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_single_group_heatmap(
    results: AttentionResults,
    group_idx: int = 0,
    layer_idx: int = -1,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """Attention heatmap for a single group, sorted by cell type."""
    attn = results.attentions[group_idx][layer_idx].astype(np.float32)
    ctypes = results.cell_types[group_idx]
    gs = attn.shape[0]

    # Sort by cell type
    sorted_indices = sorted(range(gs), key=lambda i: ctypes[i] if i < len(ctypes) else "")
    sorted_ctypes = [ctypes[i] if i < len(ctypes) else "pad" for i in sorted_indices]
    attn_sorted = attn[np.ix_(sorted_indices, sorted_indices)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(attn_sorted, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="Attention weight")

    # Annotate cell type boundaries
    unique_types = []
    tick_positions = []
    prev_type = None
    for i, ct in enumerate(sorted_ctypes):
        if ct != prev_type:
            unique_types.append(ct)
            tick_positions.append(i)
            if prev_type is not None:
                ax.axhline(y=i - 0.5, color="white", linewidth=0.5, alpha=0.7)
                ax.axvline(x=i - 0.5, color="white", linewidth=0.5, alpha=0.7)
            prev_type = ct

    # Use midpoints for tick labels
    mid_positions = []
    for j in range(len(tick_positions)):
        start = tick_positions[j]
        end = tick_positions[j + 1] if j + 1 < len(tick_positions) else gs
        mid_positions.append((start + end) / 2)

    ax.set_xticks(mid_positions)
    ax.set_xticklabels(unique_types, rotation=90, fontsize=7)
    ax.set_yticks(mid_positions)
    ax.set_yticklabels(unique_types, fontsize=7)

    label_idx = int(results.predictions[group_idx])
    label_name = results.label_names.get(label_idx, str(label_idx))
    true_idx = int(results.labels[group_idx])
    true_name = results.label_names.get(true_idx, str(true_idx))
    ax.set_title(f"Group {group_idx} — pred: {label_name}, true: {true_name}")

    fig.tight_layout()
    return fig


def _get_bar_colors(cell_types, color_map):
    """Map cell type names to colors using the provided color_map dict."""
    if color_map is None:
        return None
    return [color_map.get(ct, (0.7, 0.7, 0.7)) for ct in cell_types]


def plot_attention_per_label(
    summary_df: pd.DataFrame,
    top_k: int = 5,
    figsize_per_subplot: tuple = (5, 2.2),
    color_map: Optional[Dict] = None,
) -> plt.Figure:
    """Horizontal bar chart of top-K cell types per label class.

    Parameters
    ----------
    color_map : dict, optional
        Mapping of cell type name to color (e.g. from ``colormycells.get_colormap``).
    """
    labels = sorted(summary_df["label"].unique())
    n_labels = len(labels)
    if n_labels == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    ncols = min(3, n_labels)
    nrows = (n_labels + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_subplot[0] * ncols, figsize_per_subplot[1] * nrows),
        sharex=True,
        squeeze=False,
    )

    for idx, label in enumerate(labels):
        ax = axes[idx // ncols][idx % ncols]
        subset = summary_df[summary_df["label"] == label].nlargest(top_k, "mean_attention")
        cell_types = subset["cell_type"].values[::-1]
        means = subset["mean_attention"].values[::-1]
        sem_vals = subset["sem"].values[::-1] if "sem" in subset.columns else None
        colors = _get_bar_colors(cell_types, color_map)
        ax.barh(
            cell_types,
            means,
            xerr=sem_vals,
            capsize=2,
            color=colors,
        )
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", labelsize=9)

    # Only bottom row gets x-axis labels
    for col in range(ncols):
        axes[-1][col].set_xlabel("Mean attention received", fontsize=10)

    # Hide unused axes
    for idx in range(n_labels, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.supylabel(f"Top {top_k} cell types by attention", fontsize=11)
    fig.suptitle("Attention received per label", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_overall_attention_ranking(
    summary_df: pd.DataFrame,
    top_k: int = 20,
    figsize: tuple = (8, 6),
    color_map: Optional[Dict] = None,
) -> plt.Figure:
    """Bar chart of cell types ranked by mean attention across all labels.

    Parameters
    ----------
    color_map : dict, optional
        Mapping of cell type name to color (e.g. from ``colormycells.get_colormap``).
    """
    grouped = summary_df.groupby("cell_type")["mean_attention"]
    overall_mean = grouped.mean()
    overall_sem = grouped.sem().fillna(0)
    top_types = overall_mean.nlargest(top_k).index
    overall_mean = overall_mean[top_types].sort_values()
    overall_sem = overall_sem[top_types].reindex(overall_mean.index)
    colors = _get_bar_colors(overall_mean.index, color_map)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(overall_mean.index, overall_mean.values, xerr=overall_sem.values, capsize=2, color=colors)
    ax.set_xlabel("Mean attention received (across all labels)")
    ax.set_title(f"Top {top_k} cell types by attention received")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return fig


def plot_total_attention_ranking(
    total_summary_df: pd.DataFrame,
    top_k: int = 20,
    figsize: tuple = (8, 6),
    color_map: Optional[Dict] = None,
) -> plt.Figure:
    """Bar chart of cell types ranked by total attention captured per group.

    This complements :func:`plot_overall_attention_ranking` by showing
    abundance-weighted attention — common cell types naturally rank higher
    because they capture more of the group's total attention.

    Parameters
    ----------
    total_summary_df : DataFrame
        Output of :func:`cell_type_total_attention_summary`.
    color_map : dict, optional
        Mapping of cell type name to color.
    """
    grouped = total_summary_df.groupby("cell_type")["mean_total_attention"]
    overall_mean = grouped.mean()
    overall_sem = grouped.sem().fillna(0)
    top_types = overall_mean.nlargest(top_k).index
    overall_mean = overall_mean[top_types].sort_values()
    overall_sem = overall_sem[top_types].reindex(overall_mean.index)
    colors = _get_bar_colors(overall_mean.index, color_map)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(overall_mean.index, overall_mean.values, xerr=overall_sem.values, capsize=2, color=colors)
    ax.set_xlabel("Total attention captured per group (across all labels)")
    ax.set_title(f"Top {top_k} cell types by total attention (abundance-weighted)")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return fig


def plot_abundance_vs_attention(
    results: AttentionResults,
    layer_idx: int = -1,
    figsize: tuple = (8, 7),
    color_map: Optional[Dict] = None,
) -> plt.Figure:
    """Scatter plot of cell type abundance vs. mean attention received.

    Each point is a cell type. X-axis is mean fractional abundance within
    groups (fraction of cells in a group that are this type, including
    groups where the type is absent as zeros). Y-axis is mean per-cell
    attention received. A linear regression line and significance test
    are shown.

    Parameters
    ----------
    color_map : dict, optional
        Mapping of cell type name to color.
    """
    from scipy import stats

    df = aggregate_attention_by_cell_type(results, layer_idx=layer_idx)
    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    # For each group, compute fraction of cells that are each type.
    # Include zeros for types absent from a group.
    all_cell_types = df["cell_type"].unique()
    all_group_ids = df["group_idx"].unique()
    n_groups = len(all_group_ids)

    group_type_counts = df.groupby(["group_idx", "cell_type"]).size().reset_index(name="n_cells")
    group_totals = df.groupby("group_idx").size().reset_index(name="group_size")
    group_type_counts = group_type_counts.merge(group_totals, on="group_idx")
    group_type_counts["fraction"] = group_type_counts["n_cells"] / group_type_counts["group_size"]

    # Sum fractions across groups that contain this type, divide by total groups
    abundance = group_type_counts.groupby("cell_type")["fraction"].sum() / n_groups
    attention = df.groupby("cell_type")["mean_attention_received"].mean()

    ct_df = pd.DataFrame({"abundance": abundance, "attention": attention}).dropna()

    colors = None
    if color_map is not None:
        colors = [color_map.get(ct, (0.7, 0.7, 0.7)) for ct in ct_df.index]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(ct_df["abundance"], ct_df["attention"], c=colors, s=60, edgecolors="k", linewidths=0.5, alpha=0.85)

    # Label each point
    for ct in ct_df.index:
        ax.annotate(
            ct,
            (ct_df.loc[ct, "abundance"], ct_df.loc[ct, "attention"]),
            fontsize=6,
            alpha=0.8,
            xytext=(4, 4),
            textcoords="offset points",
        )

    # Linear regression with significance test
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        ct_df["abundance"], ct_df["attention"]
    )
    x_fit = np.array([ct_df["abundance"].min(), ct_df["abundance"].max()])
    ax.plot(x_fit, slope * x_fit + intercept, "k-", alpha=0.5)

    sig_str = f"p = {p_value:.2e}" if p_value < 0.001 else f"p = {p_value:.3f}"
    sig_label = "significant" if p_value < 0.05 else "not significant"
    box_text = f"slope = {slope:.3f}\nR\u00b2 = {r_value**2:.3f}\n{sig_str} ({sig_label})"
    ax.text(
        0.97, 0.03, box_text,
        transform=ax.transAxes, fontsize=8,
        verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8),
    )

    ax.set_xlabel("Mean fractional abundance in group")
    ax.set_ylabel("Mean attention received per cell")
    ax.set_title("Cell type abundance vs. attention received")
    fig.tight_layout()
    return fig
