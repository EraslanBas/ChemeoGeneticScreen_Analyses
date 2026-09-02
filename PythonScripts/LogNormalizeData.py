from __future__ import annotations

from libraries import *
from parameters import *
from util import *



from pathlib import Path
import os
import anndata as ad
import scanpy as sc

from pdex import parallel_differential_expression
import concurrent.futures
from typing import Iterable, Optional, Union

PathLike = Union[str, Path]

def run_differential_expression(
    input_file: str | Path,
    control_file: str | Path,
    output_dir: str | Path,
    *,
    groupby_key: str = "target_gene",
    reference: str = "non-targeting",
    is_log1p: bool = True,
    num_workers: int = 128,
) -> Path:
    """
    Run differential expression by concatenating perturbed and control AnnData objects.

    Parameters
    ----------
    input_file
        Path to perturbation .h5ad file.
    control_file
        Path to control .h5ad file.
    output_dir
        Directory where results will be written.
    groupby_key
        obs column used for grouping.
    reference
        Reference group label.
    is_log1p
        Whether the data are log1p-transformed.
    num_workers
        Number of parallel workers.

    Returns
    -------
    Path
        Path to the written results CSV.
    """
    input_file = Path(input_file)
    control_file = Path(control_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running DE on: {input_file}")
    print(f"Using controls from: {control_file}")
    print(f"Saving results to: {output_dir}")

    # Read data
    adata = ad.read_h5ad(input_file)
    adata_control = ad.read_h5ad(control_file)

    # Concatenate
    adata = sc.concat([adata, adata_control])

    # Output file
    base = input_file.stem
    output_file = output_dir / f"{base}_results.csv"

    # Run DE
    degs = parallel_differential_expression(
        adata,
        groupby_key=groupby_key,
        reference=reference,
        is_log1p=is_log1p,
        num_workers=num_workers,
    )

    # Save results
    degs.to_csv(output_file)

    return output_file


def run_de_jobs(
    *,
    input_dir: PathLike,
    out_dir: PathLike,
    control_file: PathLike,
    pattern: str = "ad_*.h5ad",
    max_jobs: int = 3,
    # forwarded to run_differential_expression
    groupby_key: str = "target_gene",
    reference: str = "non-targeting",
    is_log1p: bool = True,
    num_workers: int = 128,
) -> list[Path]:
    """
    Run differential expression for all matching .h5ad files in input_dir, in parallel,
    by calling `run_differential_expression(...)` directly (no subprocess).

    Parameters
    ----------
    input_dir
        Directory containing split .h5ad files.
    out_dir
        Output directory to write results into (created if needed).
    control_file
        Control .h5ad path.
    pattern
        Glob pattern for input files (default "ad_*.h5ad").
    max_jobs
        Max concurrent jobs (default 1). Increase to parallelize across files.
    groupby_key, reference, is_log1p, num_workers
        Passed through to `run_differential_expression`.

    Returns
    -------
    list[Path]
        Paths to the written result files (one per input file).
    """
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    control_file = Path(control_file)

    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern!r} under {input_dir}")

    if not control_file.exists():
        raise FileNotFoundError(f"Control file not found: {control_file}")

    if max_jobs < 1:
        raise ValueError("max_jobs must be >= 1")

    def _one(f: Path) -> Path:
        print(f"Starting: {f.name}")
        return run_differential_expression(
            input_file=f,
            control_file=control_file,
            output_dir=out_dir,
            groupby_key=groupby_key,
            reference=reference,
            is_log1p=is_log1p,
            num_workers=num_workers,
        )

    results: list[Path] = []
    # ThreadPool is usually fine because the heavy lifting is in numpy/scanpy
    # (which releases the GIL) + I/O. If your DE function is pure Python heavy,
    # swap to ProcessPoolExecutor.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_jobs) as ex:
        future_to_file = {ex.submit(_one, f): f for f in files}
        for fut in concurrent.futures.as_completed(future_to_file):
            f = future_to_file[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                # Fail fast with file context
                raise RuntimeError(f"DE job failed for {f}") from e

    print("All done.")
    return results



# run_de_jobs(
#     input_dir="/processed_datasets/VCI/ChemoGenetic_H1_Basak/Bisindolylmaleimide-I/SPLIT",
#     out_dir="./Bisindolylmaleimide-I",
#     control_file="/processed_datasets/VCI/ChemoGenetic_H1_Basak/Bisindolylmaleimide-I/SPLIT/Bisindolylmaleimide-I_controls.h5ad",
#     max_jobs=4,
#     num_workers=128,  # forwarded into run_differential_expression
# )

run_de_jobs(
    input_dir="/processed_datasets/VCI/ChemoGenetic_H1_Basak/NSC95397/SPLIT",
    out_dir="./NSC95397",
    control_file="/processed_datasets/VCI/ChemoGenetic_H1_Basak/NSC95397/SPLIT/NSC95397_controls.h5ad",
    max_jobs=2,
    num_workers=128,  # forwarded into run_differential_expression
)
