#!/usr/bin/env python3
import os
from pathlib import Path
import math
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

from pathlib import Path
import math
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad


def split_by_perturbation(
    input_h5ad: str,
    out_dir: str,
    target_col: str = "target_gene",
    control_labels=("non_targeting", "non-targeting"),
    max_per_file: int = 100,
    compression: str | None = None,  # "gzip" or None
    read_backed: bool = True,
    dense_dtype: str = "float32",     # dtype for dense X
    control_out_name: str | None = None,  # optional custom filename
):
    """
    Split a big h5ad into multiple files by perturbation (excluding controls),
    and additionally save ALL control cells as a separate .h5ad file.

    Notes
    -----
    - Split files contain ONLY the batch perturbation cells (no controls).
    - Controls are written once to: <out_dir>/<control_out_name or "<stem>_controls.h5ad">.
    - Forces sub.X to be dense before writing.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = "r" if read_backed else None
    adata = ad.read_h5ad(input_h5ad, backed=mode)

    try:
        if target_col not in adata.obs:
            raise KeyError(f"Column '{target_col}' not found in adata.obs")

        obs_targets = adata.obs[target_col].astype("string")
        is_control = obs_targets.isin(control_labels)
        is_valid = obs_targets.notna()

        # -------- Save controls once --------
        n_controls = int(is_control.sum())
        print(f"Control labels: {control_labels}. Control cells: {n_controls}")

        if n_controls > 0:
            stem = Path(input_h5ad).stem
            control_fname = control_out_name or f"{stem}_controls.h5ad"
            control_path = out_dir / control_fname

            print(f"Writing controls to: {control_path}")
            controls = adata[is_control].to_memory()

            # Ensure dense
            if sp.issparse(controls.X):
                controls.X = controls.X.toarray()
            else:
                controls.X = np.asarray(controls.X)

            if dense_dtype is not None:
                controls.X = controls.X.astype(dense_dtype, copy=False)

            if target_col in controls.obs:
                controls.obs[target_col] = controls.obs[target_col].astype("category")

            controls.write_h5ad(control_path, compression=compression)
        else:
            print("No control cells found; skipping control .h5ad write.")

        # -------- Split non-control perturbations --------
        perts = pd.Index(obs_targets[is_valid & (~is_control)].unique()).sort_values()
        n_perts = len(perts)
        if n_perts == 0:
            raise ValueError("No non-control perturbations found to split on.")

        print(f"Found {n_perts} perturbations (excluding controls).")

        n_chunks = math.ceil(n_perts / max_per_file)

        for i in range(n_chunks):
            batch_perts = perts[i * max_per_file : (i + 1) * max_per_file]

            # Select ONLY batch perturbations (controls excluded)
            sel = obs_targets.isin(batch_perts)
            n_cells = int(sel.sum())
            n_batch_perts = len(batch_perts)

            if n_cells == 0:
                print(f"[{i+1}/{n_chunks}] Empty selection, skipping.")
                continue

            print(f"[{i+1}/{n_chunks}] Writing {n_cells} cells | {n_batch_perts} perts ...")

            sub = adata[sel].to_memory()

            # Ensure dense and desired dtype
            if sp.issparse(sub.X):
                sub.X = sub.X.toarray()
            else:
                sub.X = np.asarray(sub.X)

            if dense_dtype is not None:
                sub.X = sub.X.astype(dense_dtype, copy=False)

            if target_col in sub.obs:
                sub.obs[target_col] = sub.obs[target_col].astype("category")

            out_path = out_dir / f"ad_{i+1:03d}.h5ad"
            sub.write_h5ad(out_path, compression=compression)

        print("Done.")
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Split h5ad by perturbation with shared controls (dense X).")
    ap.add_argument("input_h5ad", type=str)
    ap.add_argument("out_dir", type=str)
    ap.add_argument("--target-col", type=str, default="target_gene")
    ap.add_argument("--control-labels", type=str, nargs="+",
                    default=["non_targeting", "non-targeting"])
    ap.add_argument("--max-per-file", type=int, default=100)
    ap.add_argument("--compression", type=str, default="none", choices=["gzip", "none"])
    ap.add_argument("--no-backed", action="store_true",
                    help="Disable backed reading (loads entire matrix).")
    ap.add_argument("--dense-dtype", type=str, default="float32",
                    help='dtype to cast dense X to (e.g., "float32", "float64").')
    args = ap.parse_args()

    split_by_perturbation(
        input_h5ad=args.input_h5ad,
        out_dir=args.out_dir,
        target_col=args.target_col,
        control_labels=tuple(args.control_labels),
        max_per_file=args.max_per_file,
        compression=None if args.compression == "none" else args.compression,
        read_backed=not args.no_backed,
        dense_dtype=args.dense_dtype,
    )
