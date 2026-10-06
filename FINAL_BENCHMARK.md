# Frozen final benchmark (MSTKP / LR-BnB)

## Cut phase by column generation (added after the classical baseline)

New variant `cg` (file `cg_cut_phase.py`).  At a node with cover cuts, the
cut rungs normally run a subgradient cut phase on (lambda, mu) after the
exact plain dual, and then the exact cut-dual step.  With `-cg` that phase is
replaced by column generation on the node's cut-augmented Dantzig-Wolfe
master (convexity, budget and one row per active cover; MST pricing under
w + lambda l + sum mu_k 1[S_k]; HiGHS through highspy, warm-started after
every added column).  Every pricing step gives a valid Lagrangian bound and
the node keeps the best one.  After the master converges, the budget-violating
tree with the largest weight in its solution is separated (the same Section 6
separation), violated covers enter as rows (pool cap 5; covers with mu = 0
make room first), and the master is solved again -- one separation round per
node.  Column generation stops early once the bound prunes the node.  The
master's solution, summed per edge, is the node's LP solution under its
covers and is the branching indicator.  Strong-branching probes are
unchanged (they never run this phase), so `R5-...-cg` differs from `R5-...`
only in the cut phase of the node solves.

Every other configuration is unchanged: `cut_dual` is passed to the solver
only by the variant, and `--compare-root` checks R0 / R5 against the old root
run by run (objective and node count).

Configurations (families CG and CGX):

* `R0-rel-dw`, `R5-rel-dw`, `R5-rel-dw-cg`   reliability branching
* `R0-pc-dw`,  `R5-pc-dw`,  `R5-pc-dw-cg`    pseudo-cost branching
* `R2-pc-dw-cg`                             literature covers with the CG cut phase (CG only)

Cells: **CG** = A's headline cell (100), D's correlations rho = +0.5, -0.5
(25 each) and -0.9 (40, as F), B's grid, BL's loose budgets, C's scaling
cells and H's complete graphs (same instances as those families);
**CGX** = X's fresh confirmation cells.  5530 + 690 runs.

New metrics: `cg_nodes`, `cg_lp_solves`, `cg_lp_time`, `cg_msts` (pricing
MSTs, included in `mst_evaluations`), `root_cg_msts`, `cg_gain_nodes`,
`cg_gain`, `cg_early_stops`, `cg_sep_rounds`.  Checks: V (brute force)
includes the three `-cg` configurations; the smoke check asserts that the
node solver runs the configured cut phase, that probes never run it, and that
only `-cg` configurations ever enter it.

```bash
pip install highspy                                    # needed by variant cg
export MSTKP_FINAL_ROOT=$HOME/mstkp_cutdual            # NEW root
bash run_final.sh validate                             # must end with "0 failures"
bash run_final.sh cutdual-smoke                        # ~10 min, must PASS
COMPARE_ROOT=$HOME/mstkp_final_v2 setsid nohup bash run_final.sh cutdual > cutdual.out 2>&1 &
```

Files: new `cg_cut_phase.py` (the variant) and `cutdual_report.py` (tables);
changed `lagrangianrelaxation.py` (three hooks: the call after the exact
plain dual, the reset of the LP solution, the indicator shortcut; counters),
`benchmark_mstkp_.py` (override, metrics), `final_suite.py` (variant, families,
columns), `smoke_check.py`, `run_final.sh`, this file.  Nothing else changed.
Checked before release: R0-rel-dw and R5-rel-dw reproduce the old code's
objective, node count, root bound and cut count on headline instances;
`validate_cuts.py` 0 failures (1692 LR-BnB runs, the three `-cg` ones
included); `cutdual-smoke` passes.

Tables: `ROOT/paper/CG_pooled`, `CG_cells`, `CG_headline` (.md / .tex / .csv),
also printed at the end of `cutdual.out`; rerun them alone with
`python3 cutdual_report.py --root $MSTKP_FINAL_ROOT --compare-root <old root>`.


## Revision (exact plain dual) -- read first

The revision changes one thing in the solver: at every node and every
strong-branching probe, the Lagrangian dual of the budget constraint
(no cuts yet) is solved **exactly** by a breakpoint (Newton / Dinkelbach)
search over spanning trees, instead of a few clipped subgradient steps.
The cut phase (R1-R5) then starts from (lambda*, mu = 0), or, at a node
that inherits cut multipliers, from the parent's (lambda, mu) after both
points have been priced; the node bound is the better of the two.  In
the cut phase lambda keeps the first study's step rule (at most 0.02 per
iteration), as do the `-subgr` reference rows.  The branching study uses
the Dantzig-Wolfe indicator only: R0 runs no subgradient steps, so the
averaged indicator has nothing to average.
`exact_plain_dual = True` is the default in every configuration; the
variant `-subgr` restores the first study's dual (bit-identical results)
as a reference.  New / changed configurations and families:

* `R0-rel-dw-subgr`, `R5-rel-dw-subgr` (family A): the first study's dual.
* `R5-rel-dw-cp2`, `R5-rel-dw-cp1` (family X): cut phase 2x / 1x max_iter
  instead of 3x.
* F: eight branching rules (DW indicator) on R0 **and** on R5.
* X: a NEW fresh instance set (seed groups `confirm2_*`): R0, R2, R5,
  R5-cp2, R5-cp1, Gurobi DCUT.
* Removed (empty): the iteration-matched controls (R0-rit40, R0-it20,
  R0-it80), O1, XB / `curve` and the budget-curve figure F3 -- with the
  exact dual R0 runs no subgradient iterations, so there is no per-node
  budget on lambda left to vary.
* New metrics: `plain_dual_calls`, `plain_dual_msts`, `plain_dual_capped`,
  `root_plain_dual_msts`, `mst_evaluations` (= subgradient iterations +
  exact-dual MSTs; the work measure reported as "MSTs / node").
* Checks: V and the smoke test assert root_lb >= L* whenever the exact
  dual is on.

`DESIGN_VERSION` is unchanged, so every instance of families A-W is
regenerated bit-identically; run the revision under a NEW root.

## Files

| file | status | role |
|---|---|---|
| `lagrangianrelaxation.py` | modified | instrumentation (separation / indicator time, pool histogram, usage counters, cut log for V); **revision: exact plain dual `_exact_plain_dual`** |
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
* **A** headline ladder R0-R5 (n = 300, d = 0.05, beta = 0.15) plus R0 and
  R5 with the first study's subgradient dual (`-subgr`).
* **G** the ladder under most-fractional: no probes.
* **ACUT** the ladder with UB = z* for pruning/RC fixing: no primal channel.
* **E** exact cut dual off (R2, R5) and RC fixing off (R0, R5).
* **AGRB / D / C** Gurobi comparison; correlation sweep; scaling at
  average degree 14.95.
* **B** density x beta grid.
* **CAL -> select-beta -> F** eight branching rules (DW indicator) on R0 and on R5.
* **H** dense end: complete graphs n = 200-500 (up to 124 750 edges), beta
  0.5 / 0.15; R0, R2 (literature), R5 (yours), Gurobi SCF / DCUT.
* **L** large sparse end: n = 1000, 2000, 4000, 8000 at average degree 14.95
  (up to ~60 000 edges); same configurations as H.  Large and dense graphs
  have tiny Lagrangian gaps, so H and L show that the method scales, while
  the cut and branching contributions are measured in A, B, D and F.
* **W** size x density x budget grid: n = 500, 1000, 2000; average degree
  15, 50, 150 (3 750 - 150 000 edges); beta 0.15, 0.30, 0.50; 10 instances
  per cell; configurations as H.
* **BL** loose budgets beta 0.50, 0.70 on exactly B's graphs (n = 300,
  d = 0.05 / 0.10 / 0.20): with B one budget sweep from 0.10 to 0.70.
* optional **O2, GLAZY** (O1 is empty in the revision).
* **X** confirmation on a NEW fresh set (seed groups `confirm2_*`, used
  nowhere else and never by the first study): R0, R2, R5, R5 with a cut
  phase of 2x and 1x max_iter, and Gurobi DCUT.  Cells: headline; loose
  budgets (d 0.10 / 0.20, beta 0.70); complete graphs n 200 / 300 at beta
  0.5; degree-150 graphs n 500 / 1000 at beta 0.5; n 2000 sparse.
  `bash run_final.sh confirm`.
* **XB** empty in the revision (`bash run_final.sh curve` does nothing).

Paper tables: `bash run_final.sh paper` writes ROOT/paper/ (Markdown, LaTeX,
CSV): pooled confirmation tests over all fresh instances (stratified
bootstrap, Holm), the headline ladder, your cuts vs the
literature in every cell, LR-BnB vs Gurobi, branching, components, outcomes,
the dual budget / robustness in every core cell, the probe and primal channels,
the remaining gaps of unsolved runs, calibration and validation facts, and
figures (performance profile, scaling, cells) as PDF and PNG.

Coverage: n from 200 to 8000; average degree from 15 to complete (up to
150 000 edges); beta from 0.08 to 0.70; correlation from +0.5 to -0.9.
Memory limit: one per instance for every solver (16 GB, more only for
graphs with more than 87 500 edges: limit = max(16, 2 (1 + m / 12 500)) GB).  A run that reaches it stops cleanly
with its bounds (status `memory`): Gurobi through SoftMemLimit, LR-BnB
through the same stop its time limit uses.  If the whole machine runs short
of memory, the runner stops its largest job and reruns it later, so no
result is ever recorded under memory pressure.  `python3
diagnose_failures.py ~/mstkp_final` summarises memory stops and crashes.

Freeze rule: families and configurations in `frozen/design.json` can never
change under the same root; a family added later is accepted and recorded
in `frozen/added_<family>.json` with its date.
