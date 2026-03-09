#!/usr/bin/env python
"""Create PowerPoint slides summarizing COVID scRNA-seq datasets."""

import anndata
import numpy as np
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
import os
import io

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "dataset_summary.pptx")

DATASETS = {
    "combat": {
        "title": "COMBAT Consortium",
        "citation": "COMBAT Consortium, Cell (2022)",
        "tissue": "Whole blood (PBMCs)",
    },
    "ren": {
        "title": "Ren et al.",
        "citation": "Ren et al., Cell (2021)",
        "tissue": "PBMCs + airway samples",
    },
    "stevenson": {
        "title": "Stephenson et al.",
        "citation": "Stephenson et al., Nature Medicine (2021)",
        "tissue": "Whole blood (PBMCs)",
    },
}


def make_histogram(adata, title):
    """Create a cells-per-donor histogram with log x-scale, truncated at 50k."""
    counts = adata.obs["donor_id"].value_counts().values
    # Clip at 50000 for display
    counts_clipped = np.clip(counts, None, 50000)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bins = np.logspace(np.log10(counts_clipped.min()), np.log10(50000), 21)
    ax.hist(counts_clipped, bins=bins, edgecolor="white", color="#4C72B0")
    ax.set_xscale("log")
    ax.set_xlabel("Cells per donor", fontsize=12)
    ax.set_ylabel("Number of donors", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def add_summary_slide(prs, name, info, adata):
    """Add a slide summarizing one dataset."""
    slide = prs.slides.add_slide(prs.slide_layouts[5])  # Blank layout

    # Title
    txBox = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.8))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = info["title"]
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0x2C, 0x3E, 0x50)

    # Citation
    p2 = tf.add_paragraph()
    p2.text = info["citation"]
    p2.font.size = Pt(14)
    p2.font.italic = True
    p2.font.color.rgb = RGBColor(0x7F, 0x8C, 0x8D)

    # Stats table
    assays = ", ".join(adata.obs["assay"].unique().tolist())
    n_cells = f"{adata.n_obs:,}"
    n_donors = adata.obs["donor_id"].nunique()
    donors_per_label = adata.obs.groupby("label", observed=True)["donor_id"].nunique()

    lines = [
        ("Tissue", info["tissue"]),
        ("scRNA-seq technology", assays),
        ("Total cells (post-QC)", n_cells),
        ("Total donors", str(n_donors)),
        ("  Control", str(donors_per_label.get("control", 0))),
        ("  Mild", str(donors_per_label.get("mild", 0))),
        ("  Severe", str(donors_per_label.get("severe", 0))),
    ]

    table_top = Inches(1.6)
    table = slide.shapes.add_table(
        len(lines), 2, Inches(0.5), table_top, Inches(4.5), Inches(0.35 * len(lines))
    ).table
    table.columns[0].width = Inches(2.3)
    table.columns[1].width = Inches(2.2)

    for i, (label, value) in enumerate(lines):
        for j, text in enumerate([label, value]):
            cell = table.cell(i, j)
            cell.text = text
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(13)
                if label.startswith("  "):
                    paragraph.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    # Histogram
    hist_buf = make_histogram(adata, f"Cells per donor — {info['title']}")
    slide.shapes.add_picture(hist_buf, Inches(5.2), Inches(1.6), Inches(4.5))


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    for name, info in DATASETS.items():
        path = os.path.join(DATA_DIR, f"{name}_processed.h5ad")
        print(f"Loading {path}...")
        adata = anndata.read_h5ad(path)
        add_summary_slide(prs, name, info, adata)

    prs.save(OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
