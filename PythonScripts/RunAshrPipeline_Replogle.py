#!/usr/bin/env python3
"""
End-to-end ashr shrunken-logFC pipeline for the 4 Replogle_Basak cell lines.

Per cell line:
  1. ComputeSE_Replogle.py  -> chunk_XXXXXX.csv  (log-normalized Welch mean_diff + se)
  2. run_ashr_on_chunk.R     -> AshrResult_chunk_XXXXXX.csv  (ash() per chunk, parallel)
  3. MergeAshrRes.py         -> merge metadata back + AshrResult_all_chunks.csv
  4. AshrGenerateMatrix.py   -> PosteriorMean_matrix.csv  (perturbation x gene)

Each step is skipped if its outputs already exist, so the job is resumable.
Run from the PythonScripts directory (so parameters.py etc. are importable):
  cd .../SRC/PythonScripts && python RunAshrPipeline_Replogle.py
"""
from __future__ import annotations

import concurrent.futures as cf
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable                                   # py312 python running this
RSCRIPT = "/home/beraslan/miniconda/envs/py312/bin/Rscript"   # the R with ashr
ASHR_R = HERE / "run_ashr_on_chunk.R"                 # in ../RScripts, resolved below

# run_ashr_on_chunk.R actually lives in SRC/RScripts
ASHR_R = HERE.parent / "RScripts" / "run_ashr_on_chunk.R"

BASE = Path("/processed_datasets/VCI/Replogle_Basak")
OUT_BASE = BASE / "Ashr"

CELL_LINES = {
    "hepg2":  BASE / "hepg2.h5ad",
    "jurkat": BASE / "jurkat.h5ad",
    "K562":   BASE / "K562.h5ad",
    "rpe1":   BASE / "rpe1.h5ad",
}

GROUP_KEY = "gene"
CONTROL = "non-targeting"
TARGET_SUM = 200000
MIN_CELLS = 20
CHUNK_PERTS = 100
ASHR_JOBS = 24            # parallel ash() processes per cell line


def run(cmd: list[str]) -> None:
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=str(HERE))


def step1_chunks(h5ad: Path, out_dir: Path) -> None:
    if sorted(out_dir.glob("chunk_*.csv")):
        print(f"[skip] chunks exist in {out_dir}")
        return
    run([PY, HERE / "ComputeSE_Replogle.py",
         "--adata", h5ad, "--out-dir", out_dir,
         "--group-key", GROUP_KEY, "--control-label", CONTROL,
         "--target-sum", TARGET_SUM, "--min-cells", MIN_CELLS,
         "--chunk-perts", CHUNK_PERTS])


def step2_ashr(out_dir: Path) -> None:
    chunks = sorted(out_dir.glob("chunk_*.csv"))
    todo = [c for c in chunks
            if not (c.parent / f"AshrResult_{c.name}").exists()]
    if not todo:
        print(f"[skip] ashr already done for {out_dir}")
        return
    print(f"[ashr] {len(todo)}/{len(chunks)} chunks, {ASHR_JOBS} parallel")

    def one(chunk: Path):
        subprocess.run([RSCRIPT, str(ASHR_R), str(chunk)],
                       check=True, cwd=str(HERE))
        return chunk.name

    with cf.ThreadPoolExecutor(max_workers=ASHR_JOBS) as ex:
        for name in ex.map(one, todo):
            print(f"  [ashr ok] {name}", flush=True)


def step3_merge(out_dir: Path) -> None:
    run([PY, HERE / "MergeAshrRes.py", "--dir", out_dir])


def step4_matrix(out_dir: Path) -> None:
    run([PY, HERE / "AshrGenerateMatrix.py",
         "--dir", out_dir,
         "--out", out_dir / "PosteriorMean_matrix.csv",
         "--format", "csv"])


def main() -> None:
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    for name, h5ad in CELL_LINES.items():
        out_dir = OUT_BASE / name
        print("\n" + "=" * 70)
        print(f"CELL LINE: {name}  ({h5ad})")
        print("=" * 70, flush=True)
        if not h5ad.exists():
            print(f"[skip] missing {h5ad}")
            continue
        step1_chunks(h5ad, out_dir)
        step2_ashr(out_dir)
        step3_merge(out_dir)
        step4_matrix(out_dir)
        print(f"[DONE] {name}: {out_dir/'PosteriorMean_matrix.csv'}", flush=True)

    print("\n[ALL DONE] PosteriorMean matrices under", OUT_BASE)


if __name__ == "__main__":
    main()
