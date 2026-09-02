#!/usr/bin/env python3

from libraries import *
from parameters import *
from util import *
import sys
import scanpy as sc
from pathlib import Path
from pdex import parallel_differential_expression


def build_null_de_background(
    ad_ctrl,
    reference: 'non-targeting',
    n_fake_cells: int = 1000,
    n_iters: int = 1000,
    random_state: int = 42,
    tmp_key: str = "tmp_fake_group",
    progress_every: int = 50,
):
    """
    Repeatedly (n_iters) select n_fake_cells from the reference group,
    assign them label 'fake_perturbation' in a temporary grouping column,
    and run `parallel_differential_expression(adata, groupby_key=tmp_key, reference=reference)`.
    Concatenate all returned DataFrames and save to CSV.

    Parameters
    ----------
    adata : anndata.AnnData
    reference : str
        The label of the control/reference group within `groupby_key`.
    n_fake_cells : int
        Number of control cells to mark as 'fake_perturbation' in each iteration.
    n_iters : int
        Number of null iterations.
    random_state : int
        RNG seed for reproducibility.
    tmp_key : str
        Temporary column name to store fake grouping used by the DE function.
    progress_every : int
        Print a progress line every this many iterations.

    Returns
    -------
    pd.DataFrame
        The concatenated null DE results (also written to `out_csv`).
    """
    rng = np.random.default_rng(random_state)

    # Pre-create the temporary grouping column in the view (all set to reference)
    ad_ctrl.obs[tmp_key] = reference

    all_results = []

    for it in range(n_iters):
        # Sample fake-perturbation cells from the control pool
        fake_idx_local = rng.choice(ad_ctrl.n_obs, size=n_fake_cells, replace=False)

        # Reset the temporary group column to reference for all, then set fake ones
        ad_ctrl.obs[tmp_key] = reference
        ad_ctrl.obs[tmp_key].iloc[fake_idx_local] = "fake_perturbation"

        # Call your existing DE function
        de_df = parallel_differential_expression(
            ad_ctrl,
            groupby_key=tmp_key,
            reference=reference
        )

        # Annotate with iteration metadata
        de_df = de_df.copy()
        de_df["iter_id"] = it
        de_df["n_fake"] = int(n_fake_cells)
        de_df["n_ref"]  = int(ad_ctrl.n_obs - n_fake_cells)

        all_results.append(de_df)

        if progress_every and ((it + 1) % progress_every == 0):
            print(f"[null DE] completed {it + 1}/{n_iters} iterations")

    # Concatenate and save
    out_df = pd.concat(all_results, ignore_index=True)

    return out_df




def main():
    if len(sys.argv) < 2:
        print("Usage: python run_null_de_background.py <condition>")
        sys.exit(1)

    cond = sys.argv[1]
    base_dir = "/processed_datasets/VCI/ChemoGenetic_H1_Basak"
    out_dir = f"./{cond}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    in_path = f"{base_dir}/{cond}/SPLIT/{cond}_controls.h5ad"
    a = sc.read_h5ad(in_path)

    null_df = build_null_de_background(
        a,
        reference="non-targeting",
        n_fake_cells=2000,
        n_iters=70,
        random_state=42
    )
    null_df["condition"] = cond

    out_path = f"{out_dir}/{cond}_null_de_background.csv"
    null_df.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()
