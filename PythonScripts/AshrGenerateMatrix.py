#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
import pandas as pd
from parameters import *


ASHR_RE = re.compile(r"^AshrResult_chunk_\d{6}\.csv$")


def find_chunk_files(d: Path) -> list[Path]:
    files = [p for p in d.iterdir() if p.is_file() and ASHR_RE.match(p.name)]
    return sorted(files, key=lambda p: p.name)


def pick_value_col(df: pd.DataFrame) -> str:
    # User asked for "PosteriorMean" specifically, but your pipeline might name it differently.
    candidates = ["PosteriorMean", "ash_postmean", "posterior_mean", "PosteriorMeanEstimate"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not find a PosteriorMean column. Available columns include: {list(df.columns)[:30]} ..."
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Read AshrResult_chunk_*.csv files, pivot each into perturbation x gene matrix of PosteriorMean, "
            "row-concat into one giant matrix, and write a single output file."
        )
    )
    ap.add_argument("--dir", type=Path, required=True, help="Directory containing AshrResult_chunk_*.csv")
    ap.add_argument("--out", type=Path, required=True, help="Output path for the giant matrix (csv or parquet)")
    ap.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default="csv",
        help="Output format (default: csv). Parquet is strongly recommended for big matrices.",
    )
    ap.add_argument(
        "--delete-inputs",
        action="store_true",
        help="Delete the AshrResult_chunk_*.csv files after successfully writing the giant matrix.",
    )
    args = ap.parse_args()

    d = args.dir
    if not d.is_dir():
        raise FileNotFoundError(f"Not a directory: {d}")

    files = find_chunk_files(d)
    if not files:
        raise RuntimeError(f"No files matching AshrResult_chunk_*.csv found in {d}")

    mats: list[pd.DataFrame] = []

    for f in files:
        df = pd.read_csv(f)

        required = {"perturbation", "gene"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"{f.name} missing required columns: {sorted(missing)}")

        value_col = pick_value_col(df)

        # Build wide matrix for this file: rows=perturbation, cols=gene, values=PosteriorMean
        # Using pivot_table with aggfunc='first' assumes unique (perturbation,gene) per file.
        wide = df.pivot_table(
            index="perturbation",
            columns="gene",
            values=value_col,
            aggfunc="first",
        )

        # Make sure index is plain column name-friendly
        wide.index.name = "perturbation"
        mats.append(wide)

        print(f"[ok] {f.name}: {wide.shape[0]} perts x {wide.shape[1]} genes")

    # Row-concat all per-file matrices
    giant = pd.concat(mats, axis=0, join="outer", sort=False)

    # Optional: if the same perturbation appears in multiple chunks, you may want to deduplicate.
    # Here we keep the first occurrence. Comment this out if you want duplicates preserved.
    if giant.index.duplicated().any():
        ndup = int(giant.index.duplicated().sum())
        print(f"[warn] {ndup} duplicated perturbation rows; keeping first occurrence.")
        giant = giant[~giant.index.duplicated(keep="first")]

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "parquet":
        # Parquet preserves types & is far smaller/faster than CSV for wide matrices.
        giant.to_parquet(args.out)
    else:
        giant.to_csv(args.out)

    print(f"[done] wrote giant matrix: {args.out}  shape={giant.shape}")


if __name__ == "__main__":
    main()
