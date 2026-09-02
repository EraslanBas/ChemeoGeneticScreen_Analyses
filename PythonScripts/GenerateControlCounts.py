from __future__ import annotations

from typing import Sequence, Literal
import numpy as np
import anndata as ad


from pathlib import Path
from typing import Sequence, Literal, Optional, List
import numpy as np
import pandas as pd
import anndata as ad


def _intersect_var_names_backed(paths: Sequence[str]) -> List[str]:
    """Find common var_names without loading .X into memory."""
    common = None
    for p in paths:
        a = ad.read_h5ad(p, backed="r")
        v = pd.Index(a.var_names.astype(str))
        common = v if common is None else common.intersection(v)
        try:
            a.file.close()
        except Exception:
            pass
    return common.tolist() if common is not None else []


def subsample_ntc_save_each_then_concat(
    paths: Sequence[str],
    sample_names: Sequence[str],
    *,
    out_dir: str | Path,
    target_col: str = "target_gene",
    nt_label: str = "non-targeting",
    n_cells: int = 20_000,
    sample_col: str = "sample",
    seed: int = 0,
    on_missing: Literal["raise", "skip"] = "raise",
    join: Literal["inner", "outer"] = "inner",
    concat_outfile: Optional[str | Path] = None,
    compression: Literal["gzip", "lzf"] = "gzip",
) -> Optional[ad.AnnData]:
    """
    Backed-safe version: never makes a view-of-a-view.
    Subsamples N non-targeting cells per file, saves each subset to disk,
    and optionally concatenates saved subsets.
    """
    if len(paths) != len(sample_names):
        raise ValueError("paths and sample_names must have same length")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # 1) Common genes if inner join
    common_genes = None
    if join == "inner":
        common_genes = _intersect_var_names_backed(paths)
        if len(common_genes) == 0:
            raise ValueError("No common genes found across datasets for join='inner'")
        print(f"[join=inner] common genes across all files: {len(common_genes):,}")

    saved_paths: List[Path] = []

    for path, sname in zip(paths, sample_names):
        path = str(path)
        print(f"\nProcessing: {sname}")

        a = ad.read_h5ad(path, backed="r")  # does NOT load X

        try:
            if target_col not in a.obs.columns:
                msg = f"{sname}: obs['{target_col}'] not found"
                if on_missing == "skip":
                    print("WARNING:", msg)
                    continue
                raise KeyError(msg)

            mask = (a.obs[target_col].astype(str) == nt_label).to_numpy()
            idx_all = np.where(mask)[0]

            if idx_all.size < n_cells:
                msg = f"{sname}: only {idx_all.size} '{nt_label}' cells (< {n_cells})"
                if on_missing == "skip":
                    print("WARNING:", msg)
                    continue
                raise ValueError(msg)

            obs_idx = rng.choice(idx_all, size=n_cells, replace=False)
            obs_idx.sort()

            # IMPORTANT: do row+col indexing ONCE for backed AnnData
            if join == "inner":
                # map common_genes -> integer var indices for this file
                var_index = pd.Index(a.var_names.astype(str))
                var_idx = var_index.get_indexer(common_genes)
                if np.any(var_idx < 0):
                    # should not happen given intersection, but keep safe
                    missing = [common_genes[i] for i, j in enumerate(var_idx) if j < 0][:10]
                    raise ValueError(f"{sname}: unexpected missing genes after intersection, e.g. {missing}")
                a_sub_view = a[obs_idx, var_idx]
            else:
                a_sub_view = a[obs_idx, :]

            # materialize only the subset
            a_sub = a_sub_view.to_memory()

            # annotate sample
            a_sub.obs[sample_col] = sname

            out_path = out_dir / f"{sname}.ntc{n_cells}.h5ad"
            a_sub.write_h5ad(out_path, compression=compression)
            saved_paths.append(out_path)

            print(f"  Saved: {out_path} (cells={a_sub.n_obs:,}, genes={a_sub.n_vars:,})")

        finally:
            # close file handle even if something errors
            try:
                a.file.close()
            except Exception:
                pass
            del a

    if not saved_paths:
        raise ValueError("No per-sample subsets were saved (everything skipped or missing).")

    if concat_outfile is None:
        print("\nDone. Per-sample NTC subsets written; not concatenating.")
        return None

    print("\nConcatenating saved subsets...")
    adatas = [ad.read_h5ad(p) for p in saved_paths]
    adata_concat = ad.concat(
        adatas,
        axis=0,
        join="inner" if join == "inner" else "outer",
        merge="same",
        label="batch",
        keys=[p.stem for p in saved_paths],
    )

    concat_outfile = Path(concat_outfile)
    concat_outfile.parent.mkdir(parents=True, exist_ok=True)
    adata_concat.write_h5ad(concat_outfile, compression=compression)

    print(f"Saved concatenated AnnData: {concat_outfile}")
    print(f"  Final: cells={adata_concat.n_obs:,}, genes={adata_concat.n_vars:,}")

    return adata_concat

paths=["/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set1/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/AR-A014418/scanpy/ad_gene_guide_complete.gene.h5ad",
               "/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set1/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/AZD4573/scanpy/ad_gene_guide_complete.gene.h5ad",
               "/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set1/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/CHIR-98014/scanpy/ad_gene_guide_complete.gene.h5ad",
               "/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set1/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/DMSO/scanpy/ad_gene_guide_complete.gene.h5ad",
               
"/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set1/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/Lexibulin/scanpy/ad_gene_guide_complete.gene.h5ad",
               
"/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set1/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/PP121/scanpy/ad_gene_guide_complete.gene.h5ad",
               
"/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set1/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/Romidepsin/scanpy/ad_gene_guide_complete.gene.h5ad",
               
"/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set1/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/Stattic/scanpy/ad_gene_guide_complete.gene.h5ad",
               "/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set2/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/Bisindolylmaleimide-I/scanpy/ad_gene_guide_complete.gene.h5ad",
"/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set2/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/DG-172/scanpy/ad_gene_guide_complete.gene.h5ad",
"/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set2/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/DMSO/scanpy/ad_gene_guide_complete.gene.h5ad",
"/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set2/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/JTE-607/scanpy/ad_gene_guide_complete.gene.h5ad",
"/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set2/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/LDN-193189/scanpy/ad_gene_guide_complete.gene.h5ad",
"/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set2/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/LY2090314/scanpy/ad_gene_guide_complete.gene.h5ad",
"/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set2/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/NSC95397/scanpy/ad_gene_guide_complete.gene.h5ad",
"/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set2/Full/PertQC_CytoGMM/DualGuideOnly_ntgUPM5/VX-11e/scanpy/ad_gene_guide_complete.gene.h5ad"]

sample_names=["AR-A014418", "AZD4573", "CHIR-98014", "DMSO_round2", "Lexibulin", "PP121", "Romidepsin", "Stattic",
"Bisindolylmaleimide-I", "DG-172",  "DMSO_round2_batch2", "JTE-607",  "LDN-193189", "LY2090314", "NSC95397", "VX-11e"]

adata_all = subsample_ntc_save_each_then_concat(
    paths=paths,
    sample_names=sample_names,
    out_dir="/processed_datasets/VCI/ChemoGenetic_H1_Basak/NTC_subsamples/",
    target_col="target_gene",
    nt_label="non-targeting",  # NOTE: you used non-targeting (with hyphen)
    n_cells=20000,
    seed=42,
    join="inner",
    concat_outfile="/processed_datasets/VCI/ChemoGenetic_H1_Basak/AllControls_counts.h5ad",
    compression="gzip",
)

