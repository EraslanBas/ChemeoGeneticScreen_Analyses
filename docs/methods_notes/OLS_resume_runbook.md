# OLS resume runbook (PertPathwayDrugInteraction)

Pipeline: `SRC/PythonScripts/FitPathwayDrugPertOLS.py` fits one OLS per
(pathway, batch, group) — see `PertPathwayOLS/scoring_and_ols.md` for the
full model. Outputs land at
`PertPathwayDrugInteraction/group_NN/{pathway}__{batch1,batch2}.csv`.

The script has a skip-fast short-circuit: when both batch CSVs already
exist for a pathway it returns in microseconds. So launching the same
group again is safe — only missing fits run.

## What the universe is

- 22 groups (`group_00` … `group_21`)
- 2 batches per group (`batch1`, `batch2`)
- 415 expected pathways per (group, batch) after exclusions:
    - all pathways scored at
      `/processed_datasets/VCI/ChemoGenetic_H1_Basak_Pathways/pathway_scores_concat_group_NN/score_*.csv`
    - minus names listed in
      `PathwayORA/KEGG/pathways_exclude.txt`
    - minus regex `^KEGG_MEDICUS_PATHOGEN_|^KEGG_PATHOGENIC_`

So the full target = `22 × 415 × 2 = 18,260` CSVs.

## Status check (ad hoc)

Build the expected list and count missing per group:

```bash
mkdir -p /tmp/ols
ls /processed_datasets/VCI/ChemoGenetic_H1_Basak_Pathways/pathway_scores_concat_group_00/score_*.csv \
  | sed -E 's|.*/score_(.*)\.csv|\1|' | sort > /tmp/ols/all_pathways.txt
grep -v -E '^\s*#|^\s*$' \
  /home/beraslan/Projects/ChemoGeneticScreens/PathwayORA/KEGG/pathways_exclude.txt \
  | sort -u > /tmp/ols/excl.txt
comm -23 /tmp/ols/all_pathways.txt /tmp/ols/excl.txt \
  | grep -vE '^(KEGG_MEDICUS_PATHOGEN_|KEGG_PATHOGENIC_)' \
  > /tmp/ols/expected.txt
echo "Expected pathways: $(wc -l < /tmp/ols/expected.txt)"

for g in $(seq -w 0 21); do
  ls /home/beraslan/Projects/ChemoGeneticScreens/PertPathwayDrugInteraction/group_${g}/*__batch1.csv 2>/dev/null \
    | sed -E "s|.*/||; s|__batch1.csv$||" | sort > /tmp/ols/d_b1.txt
  ls /home/beraslan/Projects/ChemoGeneticScreens/PertPathwayDrugInteraction/group_${g}/*__batch2.csv 2>/dev/null \
    | sed -E "s|.*/||; s|__batch2.csv$||" | sort > /tmp/ols/d_b2.txt
  m1=$(comm -23 /tmp/ols/expected.txt /tmp/ols/d_b1.txt | wc -l)
  m2=$(comm -23 /tmp/ols/expected.txt /tmp/ols/d_b2.txt | wc -l)
  printf "group_%s  miss_b1=%4d  miss_b2=%4d  total=%4d\n" "$g" $m1 $m2 $((m1+m2))
done
```

## Launch pattern

For each group with missing fits:
1. compute the union of missing pathways across batch1 and batch2,
2. split into 4 chunks of roughly equal size,
3. launch four `python -u FitPathwayDrugPertOLS.py --group group_NN
   --pathways …` processes in parallel via `nohup`,
4. cap BLAS threads per process to avoid the oversubscription that
   choked earlier 22-proc runs.

Concrete recipe (pure shell, run from any cwd; do **not** rely on
`bash` array literals — sub-bash spawned via shebang inherits
`SHELLOPTS=onecmd` from the Claude tool's parent shell on this box and
breaks `(05 06 …)` parsing):

```bash
PYTHON=/home/beraslan/miniconda/envs/py312/bin/python
SCRIPT=/home/beraslan/Projects/ChemoGeneticScreens/SRC/PythonScripts/FitPathwayDrugPertOLS.py
LOG_DIR=/home/beraslan/Projects/ChemoGeneticScreens/SRC/PythonScripts/logs
mkdir -p $LOG_DIR
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
TS=$(date +%Y%m%d_%H%M%S)

for g in 06 07 09 11 14 15 16 17; do   # ← edit list
  ls /home/beraslan/Projects/ChemoGeneticScreens/PertPathwayDrugInteraction/group_${g}/*__batch1.csv 2>/dev/null \
    | sed -E "s|.*/||; s|__batch1.csv$||" | sort > /tmp/ols/d_b1.txt
  ls /home/beraslan/Projects/ChemoGeneticScreens/PertPathwayDrugInteraction/group_${g}/*__batch2.csv 2>/dev/null \
    | sed -E "s|.*/||; s|__batch2.csv$||" | sort > /tmp/ols/d_b2.txt
  cat <(comm -23 /tmp/ols/expected.txt /tmp/ols/d_b1.txt) \
      <(comm -23 /tmp/ols/expected.txt /tmp/ols/d_b2.txt) \
    | sort -u > /tmp/ols/g${g}_miss.txt
  total=$(wc -l < /tmp/ols/g${g}_miss.txt)
  [ $total -eq 0 ] && { echo "group_${g} already complete"; continue; }
  chunk=$(( (total + 3) / 4 ))
  rm -f /tmp/ols/g${g}_chunk_*
  split -d -l $chunk /tmp/ols/g${g}_miss.txt /tmp/ols/g${g}_chunk_

  for c in 00 01 02 03; do
    [ -f /tmp/ols/g${g}_chunk_${c} ] || continue
    PATHS=$(tr '\n' ' ' < /tmp/ols/g${g}_chunk_${c})
    LOG=$LOG_DIR/${TS}_g${g}x4_chunk${c}.log
    nohup $PYTHON -u $SCRIPT --group group_${g} --pathways $PATHS > $LOG 2>&1 &
    echo "group_${g} chunk_${c}  PID=$!  n=$(wc -l < /tmp/ols/g${g}_chunk_${c})"
  done
done
```

## Resource budget (114-core / 2 TB RAM box)

- `OMP_NUM_THREADS=6` per process.
- Per process ≈ 3 cores actually used and ~12 GB peak RAM after the
  score CSV is read.
- Comfortable: **24 procs** (= 6 groups × 4 chunks) at any time, ~70
  cores in use, ~290 GB RAM.
- Tight but works: **32 procs** (= 8 groups × 4 chunks); per-fit ~25%
  slower from BLAS contention.
- Avoid 60+ procs — load > 600 ⇒ each fit ~5× slower.

## Per-fit timing

- Design build: 20–30 s (≈ 700k–1.9M cells × 800 columns).
- OLS solve: 70–170 s.
- Pathway = 2 batches ⇒ ~250–350 s end to end.

A group with N missing pathway-batches across 4 chunks ⇒ wallclock ≈
`(N / 8) × ~270 s` (the /8 comes from 4 parallel chunks each fitting
both batches sequentially per pathway). Example: 80 missing pathways ⇒
~45 min.

## SSH detachment

Processes are launched with `nohup …  &` and reparented to PID 1 once
the launching shell exits, so they survive SSH disconnect. To verify:

```bash
pgrep -af FitPathwayDrugPertOLS\.py
ps -o pid,ppid,etime --no-headers -p $(pgrep -f FitPathwayDrugPertOLS\.py | head -1)
```

PPID should be `1` (systemd) within seconds of launch.

## Stopping

```bash
pgrep -f FitPathwayDrugPertOLS\.py | xargs -r kill
```

The script saves atomically per (pathway, batch) via a `.csv.tmp` →
rename, so a kill mid-fit only loses the in-progress fit.

## Watching

```bash
# count saves across all running logs of a tag
grep -c " → " /home/beraslan/Projects/ChemoGeneticScreens/SRC/PythonScripts/logs/<TS>_*chunk*.log

# tail one chunk log
tail -f /home/beraslan/Projects/ChemoGeneticScreens/SRC/PythonScripts/logs/<TS>_g11x4_chunk00.log
```

## Known gotchas

1. **Sub-bash inherits `SHELLOPTS=onecmd`** when launched via `#!/bin/bash`
   from the Claude Bash tool on this box. Symptom: array literals like
   `(05 06 07)` collapse to a single bogus element such as `10009`,
   the for-loop runs once, then exits silently. Workaround: use
   space-separated strings (`for g in 06 07 09 …`) or invoke the
   pipeline directly from the parent shell rather than through a
   shebang script. Putting `set +t; unset SHELLOPTS` at the top does
   **not** help (`SHELLOPTS` is read-only).

2. **BLAS oversubscription.** With more than ~32 concurrent procs at
   `OMP_NUM_THREADS=6`, load average climbs into the hundreds and each
   fit becomes 3–5× slower — net throughput drops.

3. **Disk hiccups.** Earlier May 5–6 runs occasionally hit
   `OSError: Cannot save file into a non-existent directory` mid-run
   even though the dir was created at startup; ~30–45 fits per group
   were lost that way. Hasn't recurred since the patched script (which
   skips the read on the skip-fast path); if it does, add
   `out_fp.parent.mkdir(parents=True, exist_ok=True)` immediately
   before the `to_csv` call in `fit_one_pathway`.
