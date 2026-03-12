from typing import Literal
from scipy.sparse import csr_matrix
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import uniform_filter
import numpy as np
import anndata as ad
import h5py

from sklearn import manifold
from scipy.stats import special_ortho_group
import colour
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from collections import defaultdict
import warnings

def reflect_points_to_left(coords: np.ndarray) -> np.ndarray:
    """
    Reflects coordinates that are right of x=1176.5 to the left side.
    Points already on the left remain unchanged.

    This is a one-time thing needed for the data from Chen et al. (2020); all data should be left hemisphere.
    
    Args:
        coords: Array of shape (N, 2) containing x,y coordinates
        
    Returns:
        reflected: Array of same shape with right-side points reflected
    """
    x_line = 1176.5
    reflected = coords.copy()
    
    # Only reflect points where x > x_line
    right_side_mask = coords[:, 0] > x_line
    reflected[right_side_mask, 0] = 2 * x_line - coords[right_side_mask, 0]
    
    return reflected


def compute_flatmap_pixel_area_map(flatmap_h5_path: str, smooth_kernel: int = 31) -> np.ndarray:
    """Compute physical area (µm²) per flatmap pixel using the 3D↔2D mapping.

    Each flatmap pixel maps to a 3D CCF coordinate. We estimate the physical
    area of each pixel by computing the cross product of finite-difference
    tangent vectors in the row and column directions, which gives the area of
    the parallelogram spanned by one pixel step in each direction.

    The raw Jacobian is noisy because 3D coordinates are integer voxel indices,
    so we smooth with a uniform (box) filter. This preserves total area.

    Parameters
    ----------
    flatmap_h5_path : str
        Path to ``flatmap_butterfly.h5``.
    smooth_kernel : int
        Side length of the uniform smoothing kernel. Set to 1 for no smoothing.

    Returns
    -------
    np.ndarray
        (H, W) array of µm² per pixel, NaN where unmapped.
    """
    # Constants from ccf_streamlines projection (not stored in the HDF5 file)
    view_size = (1360, 1360)  # butterfly flatmap dimensions
    vol_size = (1320, 800, 1140)  # CCF volume at 10µm resolution
    spacing = np.array([10.0, 10.0, 10.0])  # µm per voxel

    with h5py.File(flatmap_h5_path, "r") as f:
        view_lookup = f["view lookup"][:]  # (N, 2): col0=flat_2d, col1=flat_3d

    flat_2d = view_lookup[:, 0]
    flat_3d = view_lookup[:, 1]

    rows, cols = np.unravel_index(flat_2d, view_size)
    d0, d1, d2 = np.unravel_index(flat_3d, vol_size)

    # Build 3D coordinate map on the flatmap grid
    coord_3d = np.full((*view_size, 3), np.nan, dtype=np.float64)
    coord_3d[rows, cols, 0] = d0
    coord_3d[rows, cols, 1] = d1
    coord_3d[rows, cols, 2] = d2

    # Finite differences along row (du) and column (dv) directions
    du = coord_3d[1:, :-1, :] - coord_3d[:-1, :-1, :]  # (H-1, W-1, 3)
    dv = coord_3d[:-1, 1:, :] - coord_3d[:-1, :-1, :]  # (H-1, W-1, 3)

    # Clamp discontinuities: where |du| or |dv| > 50 voxels, set to NaN
    du_norm = np.linalg.norm(du, axis=-1)
    dv_norm = np.linalg.norm(dv, axis=-1)
    discontinuous = (du_norm > 50) | (dv_norm > 50)

    # Cross product gives area of parallelogram in voxel² units
    cross = np.cross(du, dv)  # (H-1, W-1, 3)
    area_voxels = np.linalg.norm(cross, axis=-1)  # (H-1, W-1)
    area_voxels[discontinuous] = np.nan

    # Convert voxel² to µm²: each voxel is spacing[i] µm along axis i.
    # The cross product of vectors in voxel units needs scaling by the
    # product of spacings for each pair of axes:
    #   |cross(s*du, s*dv)| where s = diag(spacing)
    # For uniform spacing (10,10,10): factor = 10*10 = 100
    # General: cross components are (s1*s2, s0*s2, s0*s1) scaled
    s = spacing.astype(np.float64)
    # Scale each component of du/dv by spacing, then recompute cross product
    du_phys = du * s[np.newaxis, np.newaxis, :]
    dv_phys = dv * s[np.newaxis, np.newaxis, :]
    cross_phys = np.cross(du_phys, dv_phys)
    area_um2_inner = np.linalg.norm(cross_phys, axis=-1)
    area_um2_inner[discontinuous] = np.nan

    # Pad to full (H, W) with NaN on right and bottom edges
    area_um2 = np.full(view_size, np.nan, dtype=np.float64)
    area_um2[:-1, :-1] = area_um2_inner

    # Smooth the noisy Jacobian with a NaN-aware uniform filter
    if smooth_kernel > 1:
        valid = ~np.isnan(area_um2)
        filled = np.nan_to_num(area_um2, nan=0.0)
        num = uniform_filter(filled, size=smooth_kernel)
        den = uniform_filter(valid.astype(np.float64), size=smooth_kernel)
        area_um2 = np.where(den > 0.1, num / den, np.nan)

    return area_um2


def interpolate_pixel_area_to_grid(
    pixel_area_map: np.ndarray,
    grid_rows: np.ndarray,
    grid_cols: np.ndarray,
) -> np.ndarray:
    """Interpolate the native-resolution pixel area map onto an SVC grid.

    Parameters
    ----------
    pixel_area_map : np.ndarray
        (H, W) array of µm² per native flatmap pixel (from
        ``compute_flatmap_pixel_area_map``).
    grid_rows, grid_cols : np.ndarray
        Meshgrid arrays (from ``np.meshgrid``) giving the flatmap row and
        column coordinates of each SVC grid point.

    Returns
    -------
    np.ndarray
        Same shape as ``grid_rows``, giving µm² *density* (per native pixel)
        at each grid point. Caller must multiply by the SVC pixel area in
        native-pixel units (``svc_dx * svc_dy``) to get total µm² per SVC pixel.
    """
    H, W = pixel_area_map.shape
    # Replace NaN with 0 so discontinuities contribute zero area
    area_filled = np.nan_to_num(pixel_area_map, nan=0.0)

    row_coords = np.arange(H, dtype=np.float64)
    col_coords = np.arange(W, dtype=np.float64)

    interp = RegularGridInterpolator(
        (row_coords, col_coords),
        area_filled,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    points = np.column_stack([grid_rows.ravel(), grid_cols.ravel()])
    result = interp(points).reshape(grid_rows.shape)
    return result


def compute_hierarchical_averages(df, h3_vectors):
    """
    Compute average vectors for hierarchical types based on their contained H3 types.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with columns H1_type, H2_type, and H3_type
    h3_vectors (dict): Dictionary mapping H3_type to 3D vectors (as numpy arrays or lists)
    
    Returns:
    tuple: (h1_vectors, h2_vectors)
        - h1_vectors: Dictionary mapping H1_type to average vectors
        - h2_vectors: Dictionary mapping H2_type to average vectors
    """
    # Convert all vectors to numpy arrays if they aren't already
    h3_vectors = {k: np.array(v) for k, v in h3_vectors.items()}
    
    # Create mappings for the hierarchy
    h2_to_h1 = df[['H2_type', 'H1_type']].drop_duplicates().set_index('H2_type')['H1_type'].to_dict()
    h3_to_h2 = df[['H3_type', 'H2_type']].drop_duplicates().set_index('H3_type')['H2_type'].to_dict()
    
    # Check for H3 types in vectors that don't appear in DataFrame
    unknown_h3 = set(h3_vectors.keys()) - set(df['H3_type'])
    if unknown_h3:
        warnings.warn(f"Some H3 types in vectors not found in DataFrame and will be ignored: {unknown_h3}")
        # Remove unknown H3 types from vectors
        h3_vectors = {k: v for k, v in h3_vectors.items() if k not in unknown_h3}
    
    # Calculate H2 averages
    h2_vectors = defaultdict(list)
    for h3_type, vector in h3_vectors.items():
        h2_type = h3_to_h2[h3_type]
        h2_vectors[h2_type].append(vector)
    
    h2_vectors = {h2: np.mean(vectors, axis=0) 
                 for h2, vectors in h2_vectors.items()}
    
    # Calculate H1 averages
    h1_vectors = defaultdict(list)
    for h2_type, vector in h2_vectors.items():
        h1_type = h2_to_h1[h2_type]
        h1_vectors[h1_type].append(vector)
    
    h1_vectors = {h1: np.mean(vectors, axis=0) 
                 for h1, vectors in h1_vectors.items()}
    
    return h1_vectors, h2_vectors

def split_hierarchy_inplace(adata,level):
    """Split cell types into hierarchy, e.g. 'IT_5_2' -> 'IT_5' or 'IT_5_2' -> 'IT', depending on level.
    Returns adata with column 'cell_type'"""
    if level==0:
        def filt(x):
            if x == 'non_Exc':
                return x
            return x.split('_')[0]
    elif level==1:
        def filt(x):
            if x == 'non_Exc':
                return x
            return "_".join(x.split('_')[:2])
    else:
        filt = lambda x: x
    
    adata.obs['cell_type'] = list(map(filt, adata.obs['cell_type'].values))
    return adata

def get_colormap(adata, key="cell_type", plot_colorspace=False, include_unknown=False, unknown_color='w',
                deficiency: Literal[None, "Deuteranomaly", "Protanomaly", "Tritanomaly"] = None,
                severity=0):
    """ Returns a dictionary of colors in which the perceptual distance is equal to the type/type dissimilarity.
    The colormap changes each time this is run. 

    Optionally, you can specify a color deficiency (e.g. "Deuteranomaly") and severity (0-100) to create colors
    that are approximately perceptually uniform for a certain form of colorblindness.
    This uses the CVD simulator from Machado et al. (2009).
    
    Similarity is specifically 3d MDS embedding of the "psuedobulk" expression of cells in each cell type.
            What is the average gene expression across types? What is the similarity between those?
            we'll use these to define the cell-cell similarity and then select a colormap in which
            perceptual distance is equal to the type/type dissimilarity.
            
    Uses the LUV color space. Check the code to make this more brighter/vivid
    """
    labels = adata.obs[key].unique()
    if not include_unknown:
        labels = labels[labels!="Unknown"]
    bulks = []
    for label in labels:
        pseudobulk = adata[adata.obs[key]==label].X.mean(0)
        bulks.append(pseudobulk)
    bulks = np.array(np.stack(bulks))
    similarities = np.corrcoef(bulks- bulks.mean(axis=0,keepdims=True))
    if plot_colorspace:
        sns.heatmap(pd.DataFrame(similarities, index = labels, columns=labels))
        plt.show()

    embed3 = manifold.MDS(n_components=3, dissimilarity="precomputed", )
    colors3 = embed3.fit_transform(1-similarities)
    random_3d_rotation = special_ortho_group.rvs(3)
    colors3 = np.matmul(colors3,random_3d_rotation)

    luv=colors3.copy()
    luv[:,0]=luv[:,0]*0.5 + .5 # squish the lightness and make it lighter
    luv[:,1:]*=2 # more vivid
    xyz = colour.Luv_to_XYZ(luv*100)
    colors_rgb = np.maximum(np.minimum(colour.XYZ_to_sRGB(xyz, ),1),0)
    if deficiency is not None:
        matrix = colour.blindness.matrix_cvd_Machado2009(deficiency, severity)
        # this maps normal rgb -> simulated rgb. how can we choose colors in this space?
        raise NotImplementedError
    
    if plot_colorspace:
        embed = manifold.MDS(n_components=2, dissimilarity="precomputed")
        colors = embed.fit_transform(1-similarities)
        plt.scatter(colors[:,0],colors[:,1], 
                    c=colors_rgb, s=100)
        plt.gca().set_facecolor('gray')

    d= {cat: c for cat, c in zip(labels, colors_rgb)}
    if not include_unknown:
        d['Unknown'] = unknown_color
    return d
