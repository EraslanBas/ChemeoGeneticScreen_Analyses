#!/usr/bin/env python3
from __future__ import annotations

from libraries import *
from pathlib import Path
from typing import List, Optional

import numpy as np
import anndata as ad
import scanpy as sc

import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_distances


def collect_and_sample_controls(
    root: str | Path,
    out: str | Path,
    pattern: str = "*_controls.h5ad",
    max_cells: int = 20000,
    seed: int = 0,
    join: str = "inner",           # "inner" or "outer"
    backed: bool = True,
    compression: Optional[str] = "None",
) -> Path:
    """
    Recursively find control h5ad files, sample up to max_cells from each,
    and concatenate into a single AnnData file.

    Returns
    -------
    Path
        Path to the written combined .h5ad file.
    """
    root = Path(root).expanduser().resolve()
    out = Path(out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(root.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found under {root} matching {pattern!r}")

    rng = np.random.default_rng(seed)
    sampled: List[ad.AnnData] = []

    mode = "r" if backed else None
    total_cells = 0

    for f in files:
        print(f"[read] {f}")
        adata = ad.read_h5ad(f, backed=mode)

        n = adata.n_obs
        if n > max_cells:
            idx = rng.choice(n, size=max_cells, replace=False)
            sub = adata[idx].to_memory()
        else:
            sub = adata.to_memory() if adata.isbacked else adata.copy()

        # provenance
        sub.obs = sub.obs.copy()
        sub.obs["source_file"] = str(f)
        sub.obs["source_basename"] = f.name

        sampled.append(sub)
        total_cells += sub.n_obs
        print(f"  -> kept {sub.n_obs} cells (total so far: {total_cells})")

        if getattr(adata, "isbacked", False):
            adata.file.close()

    print(f"[concat] {len(sampled)} files, {total_cells} cells total (join={join})")

    combined = ad.concat(
        sampled,
        axis=0,
        join=join,
        merge="same",
        label="batch",
        keys=[p.stem for p in files],
        index_unique="-",
    )

    comp = None if compression in (None, "None") else compression
    print(f"[write] {out}")
    combined.write_h5ad(out, compression=comp)

    return out


# ---------------- CLI wrapper ----------------


def scale_and_compute_pca(
    adataPath,
    n_components: int = 100,
    max_value: float = 10.0,
    out_file: str | Path | None = None,
    *,
    verbose: bool = True,
):
    """
    Scale data and compute PCA on an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object (modified in place).
    n_components : int
        Number of PCA components to compute.
    max_value : float
        Max value for clipping during scaling.
    out_file : str or Path, optional
        If provided, write the AnnData to this file.
    verbose : bool
        Whether to print variance explained.
    """
    adata = ad.read_h5ad(adataPath)

    print(adata)
    
    # Scale
    sc.pp.scale(adata, max_value=max_value)

    # PCA
    sc.tl.pca(adata, n_comps=n_components)

    # Variance explained
    var_ratio = adata.uns["pca"]["variance_ratio"][:n_components]
    total_var = var_ratio.sum()

    if verbose:
        print("PCA variance ratio per component:")
        print(var_ratio)
        print(f"Total variance explained by {n_components} PCs: {total_var:.4f}")

    # Optional write
    if out_file is not None:
        out_file = Path(out_file)
        adata.write(out_file)

    return adata



def compute_sample_cosine_distances(
    adata_or_path,
    *,
    sample_key: str = "sample",
    use_rep: str = "X_pca",                 # key in .obsm (e.g., "X_pca")
    rep_source: Literal["obsm", "X"] = "obsm",
    out_csv: str | Path | None = None,
    return_centroids: bool = False,
    dtype: str = "float32",
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute cosine distance between samples by taking per-sample centroids in a representation
    (default: adata.obsm["X_pca"]) and computing cosine distances between centroids.

    Parameters
    ----------
    adata_or_path
        AnnData object or path to .h5ad
    sample_key
        Column in adata.obs containing sample labels.
    use_rep
        Representation key. If rep_source="obsm", this is a key in adata.obsm.
        If rep_source="X", this is ignored and adata.X is used.
    rep_source
        Where to read the feature matrix from: "obsm" or "X".
    out_csv
        If provided, write the cosine distance matrix to this CSV.
    return_centroids
        If True, also return a DataFrame of centroids (rows = samples, cols = features).
    dtype
        dtype for centroid computation.

    Returns
    -------
    cos_df : pd.DataFrame
        Cosine distance matrix (n_samples x n_samples).
    centroids_df : pd.DataFrame (optional)
        Per-sample centroid matrix.
    """
    # Load
    if isinstance(adata_or_path, (str, Path)):
        adata = sc.read_h5ad(str(adata_or_path))
    else:
        adata = adata_or_path

    if sample_key not in adata.obs:
        raise KeyError(f"'{sample_key}' not found in adata.obs")

    # Pick matrix
    if rep_source == "obsm":
        if use_rep not in adata.obsm:
            raise KeyError(f"'{use_rep}' not found in adata.obsm")
        X = adata.obsm[use_rep]
        feature_names = [f"{use_rep}_{i}" for i in range(X.shape[1])]
    elif rep_source == "X":
        X = adata.X
        feature_names = list(adata.var_names) if hasattr(adata, "var_names") else [f"gene_{i}" for i in range(X.shape[1])]
    else:
        raise ValueError("rep_source must be 'obsm' or 'X'")

    # Samples
    samples = adata.obs[sample_key].astype("string")
    unique_samples = np.array(sorted(samples.unique()))
    n_samples = len(unique_samples)
    if n_samples == 0:
        raise ValueError(f"No samples found in obs['{sample_key}'].")

    # Compute centroids
    centroids = []
    for s in unique_samples:
        mask = (samples == s).to_numpy()
        if not mask.any():
            raise ValueError(f"No cells found for sample {s!r}.")

        X_s = X[mask]

        if sp.issparse(X_s):
            mean_vec = np.asarray(X_s.mean(axis=0)).ravel()
        else:
            mean_vec = np.asarray(X_s).mean(axis=0)

        centroids.append(mean_vec.astype(dtype, copy=False))

    centroids = np.vstack(centroids)  # (n_samples, n_features)

    # Cosine distance between centroids
    cos_mat = cosine_distances(centroids, centroids)
    cos_df = pd.DataFrame(cos_mat, index=unique_samples, columns=unique_samples)

    # Optional write
    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        cos_df.to_csv(out_csv)
        print(f"Wrote cosine distance matrix to: {out_csv}")

    if return_centroids:
        centroids_df = pd.DataFrame(centroids, index=unique_samples, columns=feature_names)
        return cos_df, centroids_df

    return cos_df




# collect_and_sample_controls(
#         root="/processed_datasets/VCI/ChemoGenetic_H1_Basak/",
#         out="/processed_datasets/VCI/ChemoGenetic_H1_Basak/AllControls.h5ad",
# )


# scale_and_compute_pca(
#     adataPath="/processed_datasets/VCI/ChemoGenetic_H1_Basak/AllControls.h5ad",
#     n_components=200,
#     out_file="/processed_datasets/VCI/ChemoGenetic_H1_Basak/AllControls_scaled.h5ad")

compute_sample_cosine_distances(
    adata_or_path="/processed_datasets/VCI/ChemoGenetic_H1_Basak/AllControls_scaled.h5ad",
    out_csv = "DrugControlCosineDistances.csv")