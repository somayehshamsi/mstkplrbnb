# Frozen final benchmark (MSTKP / LR-BnB)

## Files

| file | status | role |
|---|---|---|
| `lagrangianrelaxation.py` | modified | instrumentation only (separation / indicator time, pool histogram, usage counters, cut log for V) |
| `mstkpbranchandbound.py` | modified | **probe-override fix**, cutoff mode, probe / forced-decision counters |
| `mstkpinstance.py` | modified | correlation is an explicit argument (instances bit-identical); lazy matplotlib |
| `benchmark_mstkp_.py` | rewritten | single-run engine `run_lrbnb()`; old driver removed |
| `gurobi_baselines.py` | new | SCF, directed MCF, directed cut-set (fractional separation), lazy cut-set |
| `final_suite.py` | new | design, instances, parallel runner, collection, calibration |
| `validate_cuts.py` | new | experiment V (brute force) |
| `smoke_check.py` | new | asserts every configuration ran what it claims |
| `analyze_final.py` | new | pre-declared analysis |
| `run_final.sh` | new | stages in order |

`branchandbound.py` is unchanged.

## Run

Commit the files first: the freeze records a hash of the solver files and
refuses to run if they change afterwards.

```bash
export MSTKP_FINAL_ROOT=$HOME/mstkp_final     # results (outside the repo)
bash run_final.sh check                       # read the recommended --jobs
bash run_final.sh validate                    # must end with "0 failures"
setsid nohup bash run_final.sh smoke > smoke.out 2>&1 &    # ~1 h, must pass
setsid nohup bash run_final.sh final > final.out 2>&1 &    # the suite
bash run_final.sh status
bash run_final.sh report                      # collect + checks + analysis
```

If the Jupyter server is stopped or culled, run the same command again:
finished jobs are skipped, interrupted ones rerun, nothing is overwritten.
`final_suite.py run --retry-failed` reruns `error`/`crashed` jobs only.

## Quick one-off runs (replaces running benchmark_mstkp.py directly)

```bash
python3 final_suite.py try --num-nodes 400 --density 0.2 --seed 7 --configs R5-rel-dw
python3 final_suite.py try --num-nodes 400 --density 0.2 --configs ladder,controls,GRB-DCUT
```

Generates one instance, runs the named configurations one after another
and prints a table; writes nothing into any benchmark root.  `--beta`
(default 0.15), `--knob`, `--time-limit` (default 1800), `--verbose`,
`--out results.json`.  Sets: `ladder`, `controls`, `branching`, `gurobi`.

## Laptop rehearsal (before the server)

```bash
pip install numpy scipy networkx pandas psutil gurobipy   # + your Gurobi licence
bash run_final.sh laptop                                  # Linux, or macOS:
nohup caffeinate -i bash run_final.sh laptop > laptop.out 2>&1 &
```

Runs machine check, V, the whole smoke pipeline, and a real-size pilot
(families A, E, AGRB at n = 300, 600 s limit) under `~/mstkp_laptop`, and
ends with `LAPTOP REHEARSAL PASSED` plus per-configuration pilot times and
memory.  The pilot uses its own instances (different seeds), so it never
touches the final instances.  Laptop results are a rehearsal only: never
copy them into the server root or use them in the paper.  Keep the laptop
plugged in and awake.  Windows: run inside WSL2 (Ubuntu).  Without a full
Gurobi licence set `SMOKE_ALLOW_STATUS=error` (Gurobi jobs then fail, the
LR-BnB checks still run).

## Rules the runner enforces

* One result file per (cell, configuration, instance), written atomically
  and never overwritten; instances are hash-checked and never regenerated.
* One fresh, single-threaded worker process per run; BLAS/OpenMP threads
  forced to 1; CPU/wall ratio and thread count recorded per run.
* Claims prevent two runners from executing the same job.
* Limits per run, identical for every solver: 1800 s wall clock and 16 GB
  (Gurobi's directed MCF gets more where the model alone needs it).
* Jobs are shuffled (fixed seed) so all configurations see the same load.
* Changing the design or the solver code under an existing root is refused.

## Experiments

* **V** brute force on n = 7, 8: every generated cut (probes included)
  valid, every optimum and bound correct, Gurobi formulations agree.
* **A** headline ladder R0-R5 (n = 300, d = 0.05, beta = 0.15) plus the
  iteration-matched controls R0-rit40 (R1's budget) and R0-it20 (R2-R5's).
* **G** the ladder under most-fractional: no probes.
* **ACUT** the ladder with UB = z* for pruning/RC fixing: no primal channel.
* **E** exact cut dual off (R2, R5) and RC fixing off (R0, R5).
* **AGRB / D / C** Gurobi comparison; correlation sweep; scaling at
  average degree 14.95.
* **B** density x beta grid.
* **CAL -> select-beta -> F** branching rules x indicator source.
* **H** dense end: complete graphs n = 200-500 (up to 124 750 edges), beta
  0.5 / 0.15; R0, R2 (literature), R5 (yours), matched R0, Gurobi SCF / DCUT.
* **L** large sparse end: n = 1000, 2000, 4000, 8000 at average degree 14.95
  (up to ~60 000 edges); same configurations as H.  Large and dense graphs
  have tiny Lagrangian gaps, so H and L show that the method scales, while
  the cut and branching contributions are measured in A, B, D and F.
* **W** size x density x budget grid: n = 500, 1000, 2000; average degree
  15, 50, 150 (3 750 - 150 000 edges); beta 0.15, 0.30, 0.50; 10 instances
  per cell; configurations as H.
* **BL** loose budgets beta 0.50, 0.70 on exactly B's graphs (n = 300,
  d = 0.05 / 0.10 / 0.20): with B one budget sweep from 0.10 to 0.70.
* optional **O1, O2, GLAZY**.
* **X** confirmation, added after the core results, on FRESH instances (seed
  groups never used elsewhere): R5 as designed, R0 with 20 / 40 / 80 dual
  iterations per node, and R2 / R5 with max_iter 20 (80 iterations per node
  including the cut phase, so R0-it80 is their equal-effort control).  Cells:
  headline; loose budgets (d 0.10 / 0.20, beta 0.70); complete graphs n 200 /
  300 at beta 0.5; degree-150 graphs n 500 / 1000 at beta 0.5; n 2000 sparse.
  `bash run_final.sh confirm`.
* **XB** budget curve on exactly X's instances: R0 with 5 (default) and 10
  iterations per node, completing 5 / 10 / 20 / 40 / 80.  `bash run_final.sh curve`.

Paper tables: `bash run_final.sh paper` writes ROOT/paper/ (Markdown, LaTeX,
CSV): pooled confirmation tests over all fresh instances (stratified
bootstrap, Holm), the budget curve, the headline ladder, your cuts vs the
literature in every cell, LR-BnB vs Gurobi, branching, components, outcomes.

Coverage: n from 200 to 8000; average degree from 15 to complete (up to
150 000 edges); beta from 0.08 to 0.70; correlation from +0.5 to -0.9.
Memory limit: one per instance for every solver (16 GB, more only for
graphs with more than ~44 000 edges).  A run that reaches it stops cleanly
with its bounds (status `memory`): Gurobi through SoftMemLimit, LR-BnB
through the same stop its time limit uses.  If the whole machine runs short
of memory, the runner stops its largest job and reruns it later, so no
result is ever recorded under memory pressure.  `python3
diagnose_failures.py ~/mstkp_final` summarises memory stops and crashes.

Freeze rule: families and configurations in `frozen/design.json` can never
change under the same root; a family added later is accepted and recorded
in `frozen/added_<family>.json` with its date.
