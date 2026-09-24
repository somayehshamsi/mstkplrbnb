#!/usr/bin/env python3
"""Frozen final benchmark for the LR-BnB / MSTKP paper.

One command-line tool for the whole suite:

  machine-check   what this machine/container really offers (cgroup CPU
                  quota, physical cores, memory limit, other users' load,
                  Gurobi licence) and the concurrency to use
  describe        families, cells, configurations and job counts
  generate        instance files (+ the frozen design snapshot)
  run             execute (instance, configuration) jobs in parallel
  run-one         execute one job (same safety rules; for reruns/debugging)
  plan            job list, optionally as one shell command per line
  status          done / failed / pending per family
  collect         merge results into CSV tables + integrity report
  select-beta     apply the pre-declared calibration rule for family F

Safety model.  Every job is (cell, configuration, instance index) and owns
exactly one result file; results are written by atomic link(), so nothing is
ever overwritten.  Jobs run in fresh worker processes (the solver keeps
class-level state), single-threaded, claimed through O_EXCL claim files so
two concurrent runners never execute the same job, and a runner that is
interrupted (e.g. JupyterHub culling the server) can simply be started again:
finished jobs are skipped and stale claims of dead workers are reclaimed.
"""

import os

# Thread control must happen before numpy / scipy are imported anywhere in
# this process, and is inherited by every worker.  Forced, not setdefault:
# a user environment with OMP_NUM_THREADS=24 would otherwise leak through.
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
for _v in _THREAD_VARS:
    os.environ[_v] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

DESIGN_VERSION = "mstkp-final-v1"
SOLVER_FILES = ("lagrangianrelaxation.py", "mstkpbranchandbound.py",
                "branchandbound.py", "mstkpinstance.py", "benchmark_mstkp_.py",
                "gurobi_baselines.py")
HOST = socket.gethostname()
IS_WINDOWS = os.name == "nt"


# --- portability (Linux server, macOS / Linux laptop; native Windows is
# best-effort -- WSL2 is the supported way to run on Windows) ---------------
def _loadavg():
    try:
        return round(os.getloadavg()[0], 2)
    except (AttributeError, OSError):
        return None


def _peak_rss_mb():
    """Peak resident memory of this process in MB (ru_maxrss is KB on
    Linux but BYTES on macOS)."""
    try:
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return v / 2 ** 20 if sys.platform == "darwin" else v / 1024.0
    except ImportError:
        try:
            import psutil
            mi = psutil.Process().memory_info()
            return getattr(mi, "peak_wset", mi.rss) / 2 ** 20
        except Exception:
            return None


def _popen_isolation():
    """Own process group/session, so a worker and anything it starts can be
    signalled together and a Ctrl-C in the terminal does not reach it."""
    if IS_WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _highs_single_thread():
    """Force every HiGHS LP solve (scipy.optimize.linprog, method="highs",
    used for the Dantzig-Wolfe master) onto ONE thread.  HiGHS ignores the
    OMP/BLAS variables and by default starts (cores + 1) // 2 threads -- 12 on
    the 24-core server.  SciPy hands HiGHS an options dict through
    _linprog_highs._highs_wrapper in every version; injecting threads=1 there
    changes only the thread count (the solve itself is serial dual simplex)."""
    try:
        import scipy.optimize._linprog_highs as lh
    except Exception:
        return "scipy HiGHS interface not found"
    orig = getattr(lh, "_highs_wrapper", None)
    if orig is None:
        return "scipy HiGHS wrapper not found"
    if getattr(orig, "_mstkp_one_thread", False):
        return "already single-threaded"

    def one_thread(*args, **kwargs):
        if "options" in kwargs:
            kwargs["options"] = dict(kwargs["options"] or {}, threads=1)
        elif args and isinstance(args[-1], dict):
            args = args[:-1] + (dict(args[-1], threads=1),)
        return orig(*args, **kwargs)

    one_thread._mstkp_one_thread = True
    lh._highs_wrapper = one_thread
    return "single-threaded"


def _os_threads():
    try:
        import psutil
        return psutil.Process().num_threads()
    except Exception:
        return None


def _highs_probe(fix):
    """Thread count after one HiGHS LP solve, in this (fresh) process."""
    import numpy as np
    if fix:
        _highs_single_thread()
    from scipy.optimize import linprog
    rng = np.random.default_rng(0)
    A = rng.random((40, 120))
    linprog(rng.random(120), A_ub=-A, b_ub=-A.sum(1) * 0.3, bounds=(0, 1), method="highs")
    print(_os_threads())


def _install_soft_memory_stop(limit_gb, frac=1.0):
    """LR-BnB counterpart of Gurobi's SoftMemLimit.  Once this process's RSS
    passes frac * limit, the branch-and-bound loop ends at its next node
    boundary exactly as at a time limit, so the run reports its incumbent and
    lower bound (status "memory") instead of being killed without either.
    It fires at exactly the per-run limit (the runner's hard kill moved 10%
    above it), so every run that stays below the limit behaves exactly as
    before this stop existed.
    Checked by a 1 s interval timer in the MAIN thread (no helper thread);
    a run that never reaches the threshold is not affected in any way."""
    import signal as _signal
    try:
        import psutil
        proc = psutil.Process()
    except Exception:
        return None
    if os.environ.get("MSTKP_TEST_SOFT_GB"):          # smoke test only
        limit_gb, frac = float(os.environ["MSTKP_TEST_SOFT_GB"]), 1.0
    state = {"fired": False, "rss_gb": None, "threshold_gb": frac * float(limit_gb)}
    real_time = time.time

    class _StopClock:
        # Stands in for the `time` module inside branchandbound only: its
        # single time-limit test (time.time() - start > limit) then trips.
        def __getattr__(self, name):
            return getattr(time, name)

        @staticmethod
        def time():
            return real_time() + 1e12

    def _check(signum, frame):
        if state["fired"]:
            return
        try:
            rss = proc.memory_info().rss
        except Exception:
            return
        if rss / 1e9 > state["threshold_gb"]:
            state["fired"], state["rss_gb"] = True, round(rss / 1e9, 2)
            import branchandbound
            branchandbound.time = _StopClock()

    _signal.signal(_signal.SIGALRM, _check)
    _signal.setitimer(_signal.ITIMER_REAL, 1.0, 1.0)
    return state


def _signal_worker(proc, hard=False):
    try:
        if IS_WINDOWS:
            (proc.kill if hard else proc.terminate)()
        else:
            os.killpg(proc.pid, signal.SIGKILL if hard else signal.SIGTERM)
    except Exception:
        pass


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def solver_code_hash():
    h = hashlib.sha256()
    for name in SOLVER_FILES:
        h.update(name.encode())
        h.update(sha256_file(os.path.join(CODE_DIR, name)).encode())
    return h.hexdigest()[:16]


def git_state():
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CODE_DIR,
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain", "--"] + list(SOLVER_FILES),
                               cwd=CODE_DIR, capture_output=True, text=True, timeout=10)
        if rev.returncode == 0:
            return rev.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        pass
    return None


# =============================================================================
# DESIGN (frozen)
# =============================================================================
TIME_LIMIT = 1800.0
DEG = 14.95            # average degree of the headline cell: 0.05 * (300 - 1)

PROFILES = {
    # final: the paper.  smoke / tiny: same families and configurations on
    # small graphs, for end-to-end testing only (never analysed as results).
    "final": {"time_limit": 1800.0, "sizes": {}, "complete": {}, "count_cap": None},
    "smoke": {"time_limit": 30.0,
              "sizes": {200: 30, 300: 40, 400: 50, 600: 60, 800: 70,
                        500: 55, 1000: 80, 2000: 90, 4000: 100, 8000: 110},
              "complete": {200: 14, 300: 16, 400: 18, 500: 20}, "count_cap": 2},
    # pilot: the real graph sizes, 2-3 instances per cell, 600 s.  Its
    # instance seeds differ from `final` (the profile is part of the seed),
    # so looking at pilot results never touches the frozen final instances.
    "pilot": {"time_limit": 600.0, "sizes": {}, "complete": {}, "count_cap": 2},
    "tiny": {"time_limit": 15.0,
             "sizes": {200: 10, 300: 11, 400: 12, 600: 12, 800: 13,
                       500: 12, 1000: 13, 2000: 13, 4000: 14, 8000: 14},
             "complete": {200: 8, 300: 9, 400: 10, 500: 11}, "count_cap": 2},
}

# Ladder rungs.  Every cut rung prices its covers with the exact cut dual;
# E switches it off separately.
RUNGS = {
    "R0": dict(cover_cuts=False, cut_strengthening=None, cut_root_only=False,
               max_active_cuts=None, rank_lift=True),
    "R1": dict(cover_cuts=True, cut_strengthening="literature", cut_root_only=True,
               max_active_cuts=5, rank_lift=True),
    "R2": dict(cover_cuts=True, cut_strengthening="literature", cut_root_only=False,
               max_active_cuts=5, rank_lift=True),
    "R3": dict(cover_cuts=True, cut_strengthening="lemma1", cut_root_only=False,
               max_active_cuts=5, rank_lift=True),
    "R4": dict(cover_cuts=True, cut_strengthening="full", cut_root_only=False,
               max_active_cuts=5, rank_lift=False),
    "R5": dict(cover_cuts=True, cut_strengthening="full", cut_root_only=False,
               max_active_cuts=5, rank_lift=True),
    # optional O2: strengthening levels with root-only separation
    "R3root": dict(cover_cuts=True, cut_strengthening="lemma1", cut_root_only=True,
                   max_active_cuts=5, rank_lift=True),
    "R4root": dict(cover_cuts=True, cut_strengthening="full", cut_root_only=True,
                   max_active_cuts=5, rank_lift=False),
    "R5root": dict(cover_cuts=True, cut_strengthening="full", cut_root_only=True,
                   max_active_cuts=5, rank_lift=True),
}
RULES = {  # tag -> (code name, uses the LR indicator)
    "rel": ("reliability", True), "mf": ("most_fractional", True),
    "pc": ("pseudocost", True), "sbf": ("sb_fractional", True),
    "hyb": ("hybrid_strong_fractional", True), "rfrac": ("random_fractional", True),
    "rmst": ("random_mst", False), "sbmst": ("strong_branching", False),
}
VARIANTS = {
    "": {}, "noexact": {"exact_cut_dual": False}, "norc": {"use_rc_fixing": False},
    "cutoff": {"cutoff": True}, "it10": {"max_iter": 10}, "it20": {"max_iter": 20},
    # Iteration-matched no-cut controls.  Every cut rung runs its cut phase
    # (cut_phase_frac = 3) on top of the plain dual: 40 root iterations
    # against R0's 10, and ~20 per node against ~7 for R2-R5 (R1 separates
    # only at the root).  rit40 matches R1's budget, it20 matches R2-R5's,
    # so each ladder step can be read at equal dual effort.
    "rit40": {"root_max_iter": 40},
}
GRB_FORMS = ("SCF", "DMCF", "DCUT", "CUTSETLAZY")


def lr_config_id(rung, rule, src="dw", variant=""):
    cid = f"{rung}-{rule}" + (f"-{src}" if RULES[rule][1] else "")
    return cid + (f"-{variant}" if variant else "")


def make_config(cid):
    if cid.startswith("GRB-"):
        form = cid[4:]
        assert form in GRB_FORMS, cid
        return {"solver": "gurobi", "formulation": form, "threads": 1,
                "mip_gap": 0.0, "mip_gap_abs": 0.999, "mip_start": "min_length_tree"}
    parts = cid.split("-")
    rung, rule = parts[0], parts[1]
    src = parts[2] if RULES[rule][1] else "dw"
    if src not in ("dw", "avg"):
        raise ValueError(f"{cid}: indicator source must be dw or avg")
    variant = parts[3] if RULES[rule][1] and len(parts) > 3 else (
        parts[2] if not RULES[rule][1] and len(parts) > 2 else "")
    cfg = {"solver": "lrbnb", "rung": rung, "rule_tag": rule,
           "branching_rule": RULES[rule][0], "frac_source": src,
           "max_iter": 5, "inherit_lambda": True, "inherit_step_size": False,
           "duality_gap_threshold": 0.0, "objective_granularity": 1.0,
           "exact_cut_dual": True, "use_rc_fixing": True, "cutoff": False,
           "root_max_iter": None, "variant": variant}
    cfg.update(RUNGS[rung])
    cfg.update(VARIANTS[variant])
    assert lr_config_id(rung, rule, src, variant) == cid, cid
    return cfg


LADDER6 = ["R0", "R1", "R2", "R3", "R4", "R5"]
LADDER5 = ["R0", "R2", "R3", "R4", "R5"]
BRANCH14 = ([lr_config_id("R5", "rmst"), lr_config_id("R5", "sbmst")]
            + [lr_config_id("R5", t, s) for s in ("dw", "avg")
               for t in ("rfrac", "mf", "pc", "sbf", "rel", "hyb")])
GRB3 = ["GRB-SCF", "GRB-DMCF", "GRB-DCUT"]
KNOBS_D = {+0.5: 0.366, 0.0: 0.0, -0.5: -0.366, -0.9: -0.674}   # rho -> knob
CAL_CANDIDATES = {"F1": {"knob": 0.0, "betas": [0.08, 0.10, 0.12]},
                  "F2": {"knob": -0.674, "betas": [0.10, 0.12, 0.15]}}
CAL_RULE = {"reference_config": lr_config_id("R5", "rmst"),
            "min_median_nodes": 1000, "min_solved_share": 0.80}

FAMILY_INFO = {
    "A": ("core", "Headline ladder R0-R5 + iteration-matched R0 controls, reliability (DW)."),
    "G": ("core", "Ladder under most-fractional (no probes): removes the probe channel."),
    "E": ("core", "Exact cut dual off on R2/R5; reduced-cost fixing off on R0/R5."),
    "ACUT": ("core", "Ladder with UB = z* from the start: removes the primal channel. Needs A."),
    "AGRB": ("core", "Gurobi SCF / DMCF / DCUT on the first 50 headline instances."),
    "B": ("core", "Density x beta grid, ladder R0,R2-R5."),
    "D": ("core", "Correlation sweep rho in {+0.5,0,-0.5,-0.9}, ladder + Gurobi."),
    "CAL": ("core", "Calibration for F (5 calibration seeds, Random(MST)); then select-beta."),
    "F": ("core", "Branching: 14 rule x source configs on F0/F1/F2. Needs select-beta."),
    "C": ("core", "Scaling at constant average degree 14.95, R0/R2/R5 + Gurobi."),
    "H": ("core", "Dense end: complete graphs n = 200-500, beta 0.5 / 0.15."),
    "L": ("core", "Large sparse end: n = 1000-8000 at average degree 14.95."),
    "W": ("core", "Size x density x budget grid: n 500/1000/2000, degree 15/50/150, beta .15/.30/.50."),
    "BL": ("core", "Loose budgets beta 0.50 / 0.70 on B's graphs (n = 300, d .05/.10/.20)."),
    "O1": ("optional", "R0 with 10 / 20 dual iterations (it20 is also in A; adds it10)."),
    "O2": ("optional", "Strengthening levels with root-only separation (ladder-order check)."),
    "GLAZY": ("optional", "Lazy undirected cut-set (continuity with the previous version)."),
}
CORE_FAMILIES = [f for f, (kind, _) in FAMILY_INFO.items() if kind == "core"]


def _sized(profile, n, complete=False):
    P = PROFILES[profile]
    return (P["complete"] if complete else P["sizes"]).get(n, n)


def _count(profile, k):
    cap = PROFILES[profile]["count_cap"]
    return k if cap is None else min(k, cap if k < 100 else cap + 1)


def make_cell(profile, n, density, beta, knob, group, complete=False,
              keep_degree=True):
    """A cell: fixed (n, density, beta, knob) and a seed group."""
    n_eff = _sized(profile, n, complete)
    if complete:
        d = 1.0
    elif n_eff != n and keep_degree:
        d = min(1.0, round(density * (n - 1) / (n_eff - 1), 6))
    else:
        d = density
    cid = f"n{n_eff}_d{d:.6f}_b{beta:.3f}_k{knob:+.4f}_{group}"
    return {"id": cid, "n": n_eff, "density": d, "beta": float(beta),
            "knob": float(knob), "group": group, "complete": bool(complete)}


def scale_density(n):
    return 0.05 if n == 300 else round(DEG / (n - 1), 6)


def load_f_beta(root):
    p = os.path.join(root, "frozen", "F_beta.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def build_families(profile, root):
    """family -> list of (cell, [idx], [config ids]).  Deterministic."""
    core = lambda beta, knob=0.0, group="core": make_cell(profile, 300, 0.05, beta, knob, group)
    rel = lambda rungs: [lr_config_id(r, "rel") for r in rungs]
    fam = {}
    R0M = [lr_config_id("R0", "rel", variant="rit40"), lr_config_id("R0", "rel", variant="it20")]
    R0M20 = [lr_config_id("R0", "rel", variant="it20")]
    fam["A"] = [(core(0.15), list(range(_count(profile, 100))), rel(LADDER6) + R0M)]
    fam["G"] = [(core(0.15), list(range(_count(profile, 100))),
                 [lr_config_id(r, "mf") for r in LADDER6]
                 + [lr_config_id("R0", "mf", variant="it20")])]
    fam["E"] = [(core(0.15), list(range(_count(profile, 100))),
                 [lr_config_id("R2", "rel", variant="noexact"),
                  lr_config_id("R5", "rel", variant="noexact"),
                  lr_config_id("R0", "rel", variant="norc"),
                  lr_config_id("R5", "rel", variant="norc")])]
    fam["ACUT"] = [(core(0.15), list(range(_count(profile, 100))),
                    [lr_config_id(r, "rel", variant="cutoff") for r in LADDER6])]
    fam["AGRB"] = [(core(0.15), list(range(_count(profile, 50))), GRB3)]
    fam["B"] = []
    for d, group in ((0.05, "core"), (0.10, "grid_d0.10"), (0.20, "grid_d0.20")):
        for beta in (0.10, 0.15, 0.20, 0.30):
            fam["B"].append((make_cell(profile, 300, d, beta, 0.0, group),
                             list(range(_count(profile, 20))), rel(LADDER5) + R0M20))
    fam["D"] = [(core(0.15, knob), list(range(_count(profile, 25))), rel(LADDER5) + R0M20 + GRB3)
                for knob in KNOBS_D.values()]
    fam["CAL"] = []
    for spec in CAL_CANDIDATES.values():
        for beta in spec["betas"]:
            fam["CAL"].append((core(beta, spec["knob"], "calib"),
                               list(range(_count(profile, 5))), [CAL_RULE["reference_config"]]))
    fam["F"] = [(core(0.15), list(range(_count(profile, 50))), BRANCH14)]
    fb = load_f_beta(root)
    if fb is not None:
        fam["F"].append((core(fb["F1"]["beta"], 0.0), list(range(_count(profile, 40))), BRANCH14))
        fam["F"].append((core(fb["F2"]["beta"], CAL_CANDIDATES["F2"]["knob"]),
                         list(range(_count(profile, 40))), BRANCH14))
    fam["C"] = [(make_cell(profile, n, scale_density(n), 0.15, 0.0,
                           "core" if n == 300 else f"scale_n{n}", keep_degree=False)
                 if profile == "final" else
                 make_cell(profile, n, 0.05, 0.15, 0.0,
                           "core" if n == 300 else f"scale_n{n}"),
                 list(range(_count(profile, 25))), rel(["R0", "R2", "R5"]) + R0M20 + GRB3)
                for n in (200, 300, 400, 600, 800)]
    # Dense end: complete graphs.  Literature cuts (R2) against yours (R5),
    # no cuts (R0) and its matched control; the directed MCF cannot be built
    # here (it would need ~2 m (n-1) = 1.2e8 flow variables at n = 500).
    fam["H"] = [(make_cell(profile, n, 1.0, beta, 0.0, f"complete_n{n}", complete=True),
                 list(range(_count(profile, 20))),
                 rel(["R0", "R2", "R5"]) + R0M20 + ["GRB-SCF", "GRB-DCUT"])
                for n in (200, 300, 400, 500) for beta in (0.5, 0.15)]
    # Large sparse end: C's constant average degree (14.95) continued to
    # n = 8000 (about 60 000 edges).
    fam["L"] = [(make_cell(profile, n, round(DEG / (n - 1), 6), 0.15, 0.0, f"large_n{n}"),
                 list(range(_count(profile, 10))),
                 rel(["R0", "R2", "R5"]) + R0M20 + ["GRB-SCF", "GRB-DCUT"])
                for n in (1000, 2000, 4000, 8000)]
    # Full size x density x budget grid beyond n = 300: average degree 15
    # (sparse), 50 and 150 (dense, up to 150 000 edges), tight to loose
    # budgets.  Cells differing only in beta share their graphs.
    fam["W"] = [(make_cell(profile, n, round(deg / (n - 1), 6), beta, 0.0, f"wide_n{n}_deg{deg}"),
                 list(range(_count(profile, 10))),
                 rel(["R0", "R2", "R5"]) + R0M20 + ["GRB-SCF", "GRB-DCUT"])
                for n in (500, 1000, 2000) for deg in (15, 50, 150)
                for beta in (0.15, 0.30, 0.50)]
    # Loose budgets on exactly B's graphs (same seed groups), so B + BL is
    # one budget sweep 0.10 ... 0.70 at n = 300 for every density.
    fam["BL"] = []
    for d, group in ((0.05, "core"), (0.10, "grid_d0.10"), (0.20, "grid_d0.20")):
        for beta in (0.50, 0.70):
            fam["BL"].append((make_cell(profile, 300, d, beta, 0.0, group),
                              list(range(_count(profile, 20))), rel(LADDER5) + R0M20))
    fam["O1"] = [(core(0.15), list(range(_count(profile, 50))),
                  [lr_config_id("R0", "rel", variant="it10"),
                   lr_config_id("R0", "rel", variant="it20")])]
    fam["O2"] = [(core(0.15), list(range(_count(profile, 50))),
                  rel(["R3root", "R4root", "R5root"]))]
    fam["GLAZY"] = [(core(0.15), list(range(_count(profile, 50))), ["GRB-CUTSETLAZY"])]
    return fam


def design_snapshot(profile, root):
    fam = build_families(profile, root)
    configs = sorted({c for blocks in fam.values() for _, _, cs in blocks for c in cs})
    return {
        "design_version": DESIGN_VERSION, "profile": profile,
        "time_limit": PROFILES[profile]["time_limit"],
        "seed_rule": "sha256(design_version|profile|group|idx) -> 1e8 + x mod 9e8",
        "families": {k: [{"cell": c, "idx": [i[0], i[-1], len(i)], "configs": cs}
                         for c, i, cs in v] for k, v in fam.items()},
        "configs": {c: make_config(c) for c in configs},
        "calibration": {"candidates": CAL_CANDIDATES, "rule": CAL_RULE},
    }


def design_core(snap):
    """The part of the design that must not change once frozen.  F's F1/F2
    cells appear only after select-beta, so they are excluded here and
    frozen separately in F_beta.json."""
    s = json.loads(json.dumps(snap))
    s["families"]["F"] = [b for b in s["families"]["F"]
                          if b["cell"]["beta"] == 0.15 and b["cell"]["knob"] == 0.0]
    return s


def instance_seed(profile, group, idx):
    h = hashlib.sha256(f"{DESIGN_VERSION}|{profile}|{group}|{idx}".encode()).digest()
    # >= 1e8: disjoint from every pilot seed (random.randint(0, 10**6)).
    return 10 ** 8 + int.from_bytes(h[:8], "big") % (9 * 10 ** 8)


# =============================================================================
# Paths and atomic I/O
# =============================================================================
def p_inst(root, cid, idx):
    return os.path.join(root, "instances", cid, f"{idx:03d}.json.gz")


def p_result(root, cid, cfg, idx):
    return os.path.join(root, "results", cid, cfg, f"{idx:03d}.json")


def p_claim(root, cid, cfg, idx):
    return os.path.join(root, "claims", cid, cfg, f"{idx:03d}.claim")


def p_log(root, cid, cfg, idx):
    return os.path.join(root, "logs", cid, cfg, f"{idx:03d}.log")


def p_monitor(root, cid, cfg, idx):
    return os.path.join(root, "monitor", cid, cfg, f"{idx:03d}.json")


def write_noclobber(path, data_bytes):
    """Write atomically; never replace an existing file.  Returns False if
    the target already existed (the first writer wins)."""
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{os.path.basename(path)}.{os.getpid()}.{time.time_ns()}.tmp")
    with open(tmp, "wb") as f:
        f.write(data_bytes)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)
        return True
    except FileExistsError:
        return False
    except OSError:
        # Filesystem without hard links: exclusive create still never clobbers.
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        with os.fdopen(fd, "wb") as f:
            f.write(data_bytes)
            f.flush()
            os.fsync(f.fileno())
        return True
    finally:
        os.unlink(tmp)


def write_replace(path, data_bytes):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{os.path.basename(path)}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        f.write(data_bytes)
    os.replace(tmp, path)


def jdump(obj):
    return (json.dumps(obj, indent=1, sort_keys=True, default=_jdefault) + "\n").encode()


def _jdefault(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
    except Exception:
        pass
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def read_json(path):
    with open(path) as f:
        return json.load(f)


# =============================================================================
# Instances
# =============================================================================
class StoredInstance:
    __slots__ = ("num_nodes", "budget", "edges", "density", "beta", "hash", "seed", "meta")

    def __init__(self, rec):
        self.num_nodes = int(rec["num_nodes"])
        self.budget = int(rec["budget"])
        self.edges = [tuple(int(v) for v in e) for e in rec["edges"]]
        self.density = rec["cell"]["density"]
        self.beta = rec["cell"]["beta"]
        self.hash = rec["hash"]
        self.seed = rec["instance_seed"]
        self.meta = rec.get("meta", {})


def instance_hash(n, budget, edges):
    blob = json.dumps([int(n), int(budget), edges], separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def graph_hash(n, edges):
    blob = json.dumps([int(n), [[e[0], e[1], e[2]] for e in edges]],
                      separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:20]


def plain_lagrangian(n, edges, B):
    """Exact max_lambda L(lambda) of the budget-dualised MST (Dinkelbach on the
    two supporting trees).  Solver-independent: the denominator of root gap
    closed.  Returns (L*, lambda*, weight of the min-length tree)."""
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.csgraph import minimum_spanning_tree
    U = np.array([e[0] for e in edges]); V = np.array([e[1] for e in edges])
    W = np.array([e[2] for e in edges], float); L = np.array([e[3] for e in edges], float)
    key = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(U.tolist(), V.tolist()))}

    def tree(cost):
        T = minimum_spanning_tree(sp.coo_matrix((cost, (U, V)), shape=(n, n)).tocsr()).tocoo()
        idx = [key[(min(a, b), max(a, b))] for a, b in zip(T.row.tolist(), T.col.tolist())]
        return float(W[idx].sum()), float(L[idx].sum())

    ml_w, ml_l = tree(L + 1e-7 * W)
    wl, ll = tree(W + 1e-7 * L)
    if ll <= B:
        return wl, 0.0, ml_w
    lam_hi = 1.0
    wh, lh = tree(W + lam_hi * L)
    while lh > B and lam_hi < 1e9:
        lam_hi *= 2.0
        wh, lh = tree(W + lam_hi * L)
    for _ in range(1000):
        lam = (wh - wl) / (ll - lh)
        wt, lt = tree(W + lam * L)
        val = wt + lam * (lt - B)
        line = wl + lam * (ll - B)
        if val >= line - 1e-9 * max(1.0, abs(line)):
            return line, lam, ml_w
        if lt > B:
            wl, ll = wt, lt
        else:
            wh, lh = wt, lt
    raise RuntimeError("plain Lagrangian did not converge")


def generate_one(args):
    profile, cell, idx, root, verify = args
    path = p_inst(root, cell["id"], idx)
    if os.path.exists(path) and not verify:
        return path, "exists", None
    from mstkpinstance import MSTKPInstance
    seed = instance_seed(profile, cell["group"], idx)
    random.seed(seed)
    inst = MSTKPInstance(cell["n"], cell["density"], beta=cell["beta"], corr=cell["knob"])
    edges = [[int(u), int(v), int(w), int(l)] for u, v, w, l in inst.edges]
    n, B = int(cell["n"]), int(inst.budget)
    h = instance_hash(n, B, edges)
    if os.path.exists(path):
        with gzip.open(path, "rt") as f:
            old = json.load(f)
        if old["hash"] != h:
            raise RuntimeError(f"{path}: stored instance differs from regeneration "
                               f"(generator or Python random changed?)")
        return path, "verified", old["meta"]["graph_hash"]
    tw_l = sum(d["length"] for *_, d in inst.tw.edges(data=True))
    tl_l = sum(d["length"] for *_, d in inst.tl.edges(data=True))
    Ls, lam, mlw = plain_lagrangian(n, edges, B)
    rec = {
        "format": 1, "design_version": DESIGN_VERSION, "profile": profile,
        "cell": cell, "idx": idx, "instance_seed": seed, "num_nodes": n,
        "budget": B, "edges": edges, "hash": h,
        "meta": {"m": len(edges), "avg_degree": 2.0 * len(edges) / n,
                 "rho_realized": inst.empirical_correlation(),
                 "len_Tw": tw_l, "len_Tl": tl_l,
                 "plain_lr_bound": Ls, "plain_lr_lambda": lam,
                 "min_length_tree_weight": mlw,
                 "graph_hash": graph_hash(n, edges),
                 "python": platform.python_version(), "created": now_iso()},
    }
    data = gzip.compress(json.dumps(rec, separators=(",", ":")).encode(), mtime=0)
    if not write_noclobber(path, data):
        return path, "raced", None
    return path, "created", rec["meta"]["graph_hash"]


def load_instance(path):
    with gzip.open(path, "rt") as f:
        return StoredInstance(json.load(f))


# =============================================================================
# Jobs
# =============================================================================
class Job:
    __slots__ = ("cell", "cfg", "idx", "families")

    def __init__(self, cell, cfg, idx):
        self.cell, self.cfg, self.idx, self.families = cell, cfg, idx, set()

    @property
    def key(self):
        return (self.cell["id"], self.cfg, self.idx)

    def label(self):
        return f"{self.cell['id']}/{self.cfg}/{self.idx:03d}"


_NOTED = {}


def expand_jobs(profile, root, family_names):
    fam = build_families(profile, root)
    jobs = {}
    for name in family_names:
        if name not in fam:
            raise SystemExit(f"unknown family {name!r}; known: {', '.join(fam)}")
        if name == "F" and load_f_beta(root) is None and not _NOTED.get("F"):
            _NOTED["F"] = True
            print("NOTE: F1/F2 cells need `select-beta` (after CAL); only F0 is planned.",
                  file=sys.stderr)
        for cell, idxs, cfgs in fam[name]:
            for cfg in cfgs:
                for i in idxs:
                    j = jobs.setdefault((cell["id"], cfg, i), Job(cell, cfg, i))
                    j.families.add(name)
    return list(jobs.values())


def parse_families(spec):
    """Comma list of families; `core` and `all` expand in place."""
    out = []
    for tok in (t.strip() for t in spec.split(",")):
        names = (CORE_FAMILIES if tok == "core" else
                 list(FAMILY_INFO) if tok == "all" else [tok] if tok else [])
        out.extend(n for n in names if n not in out)
    return out


def expected_m(cell):
    n = cell["n"]
    tot = n * (n - 1) // 2
    return max(n - 1, round(cell["density"] * tot))


# Declared resource limit per run, identical for every solver: 1800 s and
# RUN_MEM_LIMIT_GB.  (Measured LR-BnB growth at n=300, d=0.05: ~0.09 MB per
# processed node, i.e. up to ~10 GB for a run that goes the full 1800 s.)
RUN_MEM_LIMIT_GB = 16.0


def mem_reservation_gb(job):
    """(initial admission reservation, per-run limit, hard-kill threshold) in GB.

    The reservation only has to cover a job's START; admission afterwards
    uses each running job's measured RSS (see Runner._effective_mem)."""
    cfg = make_config(job.cfg)
    n, m = job.cell["n"], expected_m(job.cell)
    # One limit per INSTANCE, the same for every solver: 16 GB, or more on
    # graphs large enough to need it (only m > ~44 000 edges).
    inst_limit = max(RUN_MEM_LIMIT_GB, 2.0 * (1.0 + m / 12500.0))
    if cfg["solver"] == "gurobi":
        from gurobi_baselines import estimate_memory_gb
        r = estimate_memory_gb(cfg["formulation"], n, m)
        limit = max(inst_limit, r)                 # DMCF may need its model size
        kill = 1.25 * limit + 2.0                  # Gurobi SoftMemLimit acts first
    else:
        r = 1.0 + m / 12500.0
        limit = inst_limit
        kill = 1.1 * limit                         # backstop; the worker stops itself at limit
    if os.environ.get("MSTKP_TEST_KILL_GB"):          # smoke test only
        kill = float(os.environ["MSTKP_TEST_KILL_GB"])
    return r, limit, kill


def zstar_for(root, cid, idx):
    """Agreed optimum over every optimal non-cutoff result on this instance."""
    base = os.path.join(root, "results", cid)
    vals = []
    if os.path.isdir(base):
        for cfg in sorted(os.listdir(base)):
            p = os.path.join(base, cfg, f"{idx:03d}.json")
            if not os.path.exists(p) or cfg.endswith("-cutoff"):
                continue
            try:
                r = read_json(p)
            except Exception:
                continue
            mt = r.get("metrics", {})
            if mt.get("status") == "optimal" and mt.get("obj") is not None:
                vals.append(float(mt["obj"]))
    if not vals:
        return None, 0, False
    return min(vals), len(vals), (max(vals) - min(vals) <= 1e-6)


# =============================================================================
# Machine inspection
# =============================================================================
def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return None


def machine_info():
    info = {"host": HOST, "logical_cpus": os.cpu_count()}
    try:
        aff = sorted(os.sched_getaffinity(0))
    except Exception:
        aff = list(range(os.cpu_count() or 1))
    info["affinity_cpus"] = len(aff)
    quota = None
    v2 = _read("/sys/fs/cgroup/cpu.max")
    if v2 and not v2.startswith("max"):
        q, per = v2.split()[:2]
        quota = int(q) / int(per)
    else:
        q = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        per = _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if q and per and int(q) > 0:
            quota = int(q) / int(per)
    info["cgroup_cpu_quota"] = quota
    cores = set()
    for c in aff:
        core = _read(f"/sys/devices/system/cpu/cpu{c}/topology/core_id")
        pkg = _read(f"/sys/devices/system/cpu/cpu{c}/topology/physical_package_id")
        cores.add((pkg, core) if core is not None else ("cpu", c))
    if all(k[0] == "cpu" for k in cores):          # no /sys topology (macOS, Windows)
        try:
            import psutil
            phys = psutil.cpu_count(logical=False) or len(aff)
        except Exception:
            phys = len(aff)
        info["physical_cores_in_affinity"] = min(phys, len(aff))
    else:
        info["physical_cores_in_affinity"] = len(cores)
    info["performance_cores"] = None
    if sys.platform == "darwin":                  # Apple silicon: P-cores only
        try:
            pc = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                                capture_output=True, text=True, timeout=5).stdout.strip()
            info["performance_cores"] = int(pc) if pc.isdigit() else None
        except Exception:
            pass
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["mem_total_gb"] = vm.total / 1e9
        info["mem_available_gb"] = vm.available / 1e9
        me = psutil.Process().username()
        others = 0.0
        procs = [p for p in psutil.process_iter(["username"])]
        for p in procs:
            try:
                p.cpu_percent(None)
            except Exception:
                pass
        time.sleep(1.0)
        for p in procs:
            try:
                if p.info.get("username") != me:
                    others += p.cpu_percent(None) / 100.0
            except Exception:
                pass
        info["other_users_busy_cores"] = round(others, 2)
    except Exception:
        info["mem_total_gb"] = info["mem_available_gb"] = None
        info["other_users_busy_cores"] = None
    lim = _read("/sys/fs/cgroup/memory.max") or _read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    info["cgroup_mem_limit_gb"] = (int(lim) / 1e9 if lim and lim.isdigit()
                                   and int(lim) < 1 << 60 else None)
    info["loadavg"] = _loadavg()
    usable = min(x for x in (info["affinity_cpus"], info["physical_cores_in_affinity"],
                             info["performance_cores"], quota if quota else 10 ** 6) if x)
    busy = info["other_users_busy_cores"] or 0.0
    headroom = max(2, math.ceil(0.15 * usable))
    info["recommended_jobs"] = int(max(1, min(20, math.floor(usable) - headroom - math.ceil(busy))))
    mem = min(x for x in (info["mem_total_gb"], info["cgroup_mem_limit_gb"]) if x) \
        if info["mem_total_gb"] else 16.0
    budget = mem - max(8.0, 0.1 * mem)
    if info.get("mem_available_gb"):              # a laptop is never idle
        budget = min(budget, info["mem_available_gb"] - 2.0)
    info["recommended_mem_budget_gb"] = round(max(3.0, budget), 1)
    return info


def cmd_machine_check(a):
    info = machine_info()
    for k, v in info.items():
        print(f"  {k:28s} {v}")
    for tool in ("tmux", "screen", "setsid", "nohup", "git"):
        print(f"  {'has ' + tool:28s} {bool(shutil.which(tool))}")
    root = a.root
    os.makedirs(root, exist_ok=True)
    du = shutil.disk_usage(root)
    print(f"  {'free disk at root (GB)':28s} {du.free / 1e9:.1f}")
    try:
        import gurobipy as gp
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        mdl = gp.Model(env=env)
        mdl.addVars(2100)
        mdl.update()
        mdl.optimize()
        print(f"  {'gurobi':28s} {gp.gurobi.version()} full licence (2100-var model OK)")
        mdl.dispose(); env.dispose()
    except ImportError:
        print(f"  {'gurobi':28s} gurobipy NOT importable -> Gurobi jobs will error")
    except Exception as exc:
        print(f"  {'gurobi':28s} PROBLEM: {exc}  (size-limited or missing licence?)")
    probes = {}
    for label, flag in (("default", "nofix"), ("benchmark", "fix")):
        try:
            out = subprocess.run([sys.executable, os.path.abspath(__file__), "highs-probe", flag],
                                 capture_output=True, text=True, timeout=120, cwd=CODE_DIR)
            probes[label] = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "?"
        except Exception as exc:
            probes[label] = f"? ({exc})"
    print(f"  {'HiGHS threads per LP solve':28s} default {probes['default']}, "
          f"in the benchmark {probes['benchmark']}  (must be 1)")
    if info["cgroup_cpu_quota"] and info["cgroup_cpu_quota"] < info["affinity_cpus"]:
        print("  WARNING: the container's CPU quota is below the visible CPU count; "
              "jobs beyond the quota would share CPUs and distort timings.")
    if info["physical_cores_in_affinity"] < info["affinity_cpus"]:
        print("  NOTE: hyper-threads present; concurrency is capped by physical cores.")
    print(f"\n  => use --jobs {info['recommended_jobs']} "
          f"--mem-budget {info['recommended_mem_budget_gb']}")


# =============================================================================
# Worker (runs inside a fresh process)
# =============================================================================
def verify_solution(inst, sol, obj):
    if sol is None:
        return None, "no solution recorded"
    n = inst.num_nodes
    attr = {(u, v): (w, l) for u, v, w, l in inst.edges}
    if len(sol) != n - 1:
        return False, f"{len(sol)} edges, expected {n - 1}"
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    tw = tl = 0
    for u, v in sol:
        e = (min(u, v), max(u, v))
        if e not in attr:
            return False, f"edge {e} not in instance"
        ru, rv = find(e[0]), find(e[1])
        if ru == rv:
            return False, "cycle"
        parent[ru] = rv
        tw += attr[e][0]
        tl += attr[e][1]
    if tl > inst.budget:
        return False, f"length {tl} > budget {inst.budget}"
    if obj is None or abs(tw - float(obj)) > 1e-6:
        return False, f"weight {tw} != reported {obj}"
    return True, "ok"


def _test_fault(spec):
    """Fault injection for the smoke test ONLY.  Inactive unless both
    MSTKP_TEST_FAULT and MSTKP_TEST_FAULT_CFG are set in the environment;
    the benchmark commands never set them."""
    fault = os.environ.get("MSTKP_TEST_FAULT")
    if not fault or os.environ.get("MSTKP_TEST_FAULT_CFG") != spec["config_id"]:
        return
    if fault == "raise":
        raise RuntimeError("injected test fault")
    if fault == "abort":
        os.abort()
    if fault == "hang":
        time.sleep(10 ** 6)
    if fault == "mem":
        hog = []
        while True:
            hog.append(bytearray(64 * 2 ** 20))
            time.sleep(0.05)


def worker_main(spec_json):
    spec = json.loads(spec_json)
    cpu = spec.get("cpu")
    if cpu is not None:
        try:
            os.sched_setaffinity(0, {int(cpu)})
        except Exception:
            pass
    import numpy, scipy, networkx
    t_wall0 = time.time()
    res = {
        "key": {"cell_id": spec["cell"]["id"], "config_id": spec["config_id"], "idx": spec["idx"]},
        "cell": spec["cell"], "config": spec["config"],
        "instance": {"path": spec["instance_path"], "hash": spec["instance_hash"]},
        "run": dict(spec["run"], worker_pid=os.getpid(), started=now_iso(),
                    python=platform.python_version(), numpy=numpy.__version__,
                    scipy=scipy.__version__, networkx=networkx.__version__,
                    threads_env={v: os.environ.get(v) for v in _THREAD_VARS},
                    pythonhashseed=os.environ.get("PYTHONHASHSEED")),
        "metrics": {"status": "error", "solved": False},
        "diag": {}, "solution_edges": None,
        "verification": {"solution_ok": None, "reason": None}, "error": None,
    }
    try:
        inst = load_instance(spec["instance_path"])
        if inst.hash != spec["instance_hash"]:
            raise RuntimeError("instance hash mismatch")
        res["instance"].update(seed=inst.seed, n=inst.num_nodes, m=len(inst.edges),
                               budget=inst.budget)
        cfg = spec["config"]
        res["run"]["highs_threads"] = _highs_single_thread()
        _test_fault(spec)
        if cfg["solver"] == "lrbnb":
            from benchmark_mstkp_ import run_lrbnb
            soft = (_install_soft_memory_stop(spec["mem_limit_gb"])
                    if spec.get("mem_limit_gb") else None)
            try:
                m = run_lrbnb(inst, cfg, spec["time_limit"], inst.seed,
                              cutoff=spec.get("cutoff"))
            finally:
                if soft is not None:
                    import signal as _signal
                    _signal.setitimer(_signal.ITIMER_REAL, 0, 0)
            res["diag"] = m.pop("diag", {})
            if soft is not None:
                res["run"]["soft_memory_stop_gb"] = soft["threshold_gb"]
                if (soft["fired"] and m.get("status") == "timeout"
                        and m.get("wall_time", 0) < float(spec["time_limit"])):
                    m["status"], m["solved"] = "memory", False
                    m["memory_stop_rss_gb"] = soft["rss_gb"]
        else:
            from gurobi_baselines import run_gurobi
            m = run_gurobi(inst, cfg["formulation"], spec["time_limit"],
                           mem_limit_gb=spec.get("mem_limit_gb"), seed=0)
            res["run"]["gurobi"] = m.get("grb_version")
        res["solution_edges"] = m.pop("solution_edges", None)
        res["metrics"] = m
        if m.get("cutoff") is None:
            ok, why = verify_solution(inst, res["solution_edges"], m.get("obj"))
            res["verification"] = {"solution_ok": ok, "reason": why}
        else:
            res["verification"] = {"solution_ok": None, "reason": "cutoff run"}
    except MemoryError:
        res["metrics"] = {"status": "memory", "solved": False}
        res["error"] = traceback.format_exc()
    except Exception:
        res["metrics"] = {"status": "error", "solved": False}
        res["error"] = traceback.format_exc()
    finally:
        res["run"]["finished"] = now_iso()
        res["run"]["worker_wall"] = time.time() - t_wall0
        res["run"]["peak_rss_mb"] = _peak_rss_mb()
        res["run"]["os_threads_end"] = _os_threads()
        res["run"]["platform"] = sys.platform
        res["metrics"].setdefault("wall_time", time.time() - t_wall0)
    if not write_noclobber(spec["result_path"], jdump(res)):
        print("result already existed; this run's output was discarded", file=sys.stderr)
    if res["error"]:
        print(res["error"], file=sys.stderr)
    return 0


# =============================================================================
# Claims
# =============================================================================
def pid_alive(pid):
    """Is `pid` a live final_suite.py process?  Uses psutil everywhere:
    os.kill(pid, 0) would TERMINATE the process on Windows."""
    if not pid:
        return False
    try:
        import psutil
    except ImportError:
        if IS_WINDOWS:
            return True                      # cannot tell: never reclaim
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        cmd = _read(f"/proc/{pid}/cmdline")
        return cmd is None or "final_suite.py" in cmd
    try:
        p = psutil.Process(int(pid))
        if p.status() == psutil.STATUS_ZOMBIE:
            return False
        return "final_suite.py" in " ".join(p.cmdline())
    except psutil.NoSuchProcess:
        return False
    except (psutil.AccessDenied, psutil.ZombieProcess):
        return True


def try_claim(path, info, reclaim_stale=True, max_age=4 * 3600.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            with os.fdopen(fd, "w") as f:
                json.dump(info, f)
            return True
        except FileExistsError:
            try:
                old = read_json(path)
            except Exception:
                old = {}
            same_host_dead = (old.get("host") == HOST
                              and not pid_alive(old.get("worker_pid"))
                              and not pid_alive(old.get("runner_pid")))
            too_old = time.time() - float(old.get("claimed_epoch", time.time())) > max_age
            stale = same_host_dead or too_old
            if not (stale and reclaim_stale):
                return False
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
    return False


# =============================================================================
# Scheduler
# =============================================================================
class Runner:
    def __init__(self, a, jobs):
        self.a = a
        self.root = a.root
        self.jobs = jobs
        self.stop = False
        self.running = {}
        self.stats = {"done": 0, "skipped_done": 0, "busy_elsewhere": 0, "blocked": 0}
        self.run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{HOST}-{os.getpid()}"
        os.makedirs(os.path.join(self.root, "runs"), exist_ok=True)
        self.logf = open(os.path.join(self.root, "runs", f"{self.run_id}.log"), "a")
        self.code_hash = solver_code_hash()
        self.git = git_state()
        self.driver_hash = sha256_file(os.path.abspath(__file__))[:16]
        self.cpu_slots = self._cpu_slots() if a.pin else []
        self.free_slots = list(self.cpu_slots)
        self.heavy = set()          # jobs preempted for memory: reserve their limit
        self.pending = []
        self.last_preempt = 0.0
        self.n_preempted = 0
        try:
            import psutil
            self.mem_total_gb = psutil.virtual_memory().total / 1e9
        except Exception:
            self.mem_total_gb = None

    def log(self, msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.logf.write(line + "\n")
        self.logf.flush()

    def _cpu_slots(self):
        if not hasattr(os, "sched_getaffinity"):
            print("NOTE: --pin is not supported on this OS; running unpinned.")
            return []
        seen, slots = set(), []
        for c in sorted(os.sched_getaffinity(0)):
            core = (_read(f"/sys/devices/system/cpu/cpu{c}/topology/physical_package_id"),
                    _read(f"/sys/devices/system/cpu/cpu{c}/topology/core_id"))
            if core not in seen:
                seen.add(core)
                slots.append(c)
        return slots

    def _signal(self, signum, _frame):
        if not self.stop:
            self.log(f"signal {signum}: stopping -- no new jobs; running jobs are terminated "
                     f"and will be rerun next time")
        self.stop = True

    def _spec(self, job, cutoff, mem_res, mem_limit, cpu):
        cfg = make_config(job.cfg)
        inst_path = p_inst(self.root, job.cell["id"], job.idx)
        with gzip.open(inst_path, "rt") as f:
            ih = json.load(f)["hash"]
        return {
            "cell": job.cell, "config_id": job.cfg, "config": cfg, "idx": job.idx,
            "instance_path": inst_path, "instance_hash": ih,
            "result_path": p_result(self.root, job.cell["id"], job.cfg, job.idx),
            "time_limit": self.time_limit, "cutoff": cutoff,
            "mem_reserve_gb": round(mem_res, 2), "mem_limit_gb": round(mem_limit, 2),
            "cpu": cpu,
            "run": {"run_id": self.run_id, "host": HOST, "profile": self.a.profile,
                    "design_version": DESIGN_VERSION, "solver_code_hash": self.code_hash,
                    "driver_hash": self.driver_hash, "git": self.git,
                    "time_limit": self.time_limit, "mem_limit_gb": round(mem_limit, 2),
                    "jobs_setting": self.a.jobs, "running_at_start": len(self.running) + 1,
                    "cpu_slot": cpu, "loadavg_at_start": _loadavg(),
                    "platform": sys.platform},
        }

    def _reservation(self, job):
        r, limit, _ = mem_reservation_gb(job)
        return limit if job.key in self.heavy else r

    def _launch(self, job, cutoff):
        mem_res, mem_limit, kill_gb = mem_reservation_gb(job)
        if job.key in self.heavy:
            mem_res = mem_limit
        cpu = self.free_slots.pop(0) if self.free_slots else None
        spec = self._spec(job, cutoff, mem_res, mem_limit, cpu)
        logp = p_log(self.root, job.cell["id"], job.cfg, job.idx)
        os.makedirs(os.path.dirname(logp), exist_ok=True)
        logh = open(logp, "a")
        logh.write(f"==== {now_iso()} run {self.run_id} ====\n")
        logh.flush()
        env = dict(os.environ)
        env.update({v: "1" for v in _THREAD_VARS})
        env["PYTHONHASHSEED"] = "0"
        env["MPLBACKEND"] = "Agg"
        proc = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "worker", json.dumps(spec)],
            stdout=logh, stderr=subprocess.STDOUT, env=env, cwd=CODE_DIR,
            **_popen_isolation())
        claim = p_claim(self.root, job.cell["id"], job.cfg, job.idx)
        try:
            info = read_json(claim)
            info["worker_pid"] = proc.pid
            write_replace(claim, json.dumps(info).encode())
        except Exception:
            pass
        ps = None
        try:
            import psutil
            ps = psutil.Process(proc.pid)
        except Exception:
            pass
        self.running[proc.pid] = {
            "job": job, "proc": proc, "t0": time.time(), "mem": mem_res,
            "kill_gb": kill_gb, "cpu": cpu, "log": logh, "ps": ps,
            "max_rss": 0.0, "cur_rss": 0.0, "max_threads": 0, "killed": None,
            "claim": claim}

    def _effective_mem(self):
        """Admission load: each running job counts with the larger of its
        start reservation and 1.2 x its measured RSS, so jobs whose search
        tree is growing throttle new launches instead of overcommitting."""
        return sum(max(r["mem"], 1.2 * r["cur_rss"] / 1024.0) for r in self.running.values())

    def _kill(self, r, why):
        r["killed"] = why
        _signal_worker(r["proc"], hard=False)
        r["kill_t"] = time.time()

    def _finish(self, pid, r, interrupted=False):
        job = r["job"]
        preempted = r["killed"] == "preempted"
        interrupted = interrupted or r["killed"] in ("interrupted", "preempted")
        if preempted and not os.path.exists(p_result(self.root, job.cell["id"], job.cfg, job.idx)):
            self.pending.append(job)            # rerun later, alone with its full reservation
        r["log"].close()
        if r["cpu"] is not None:
            self.free_slots.append(r["cpu"])
        rp = p_result(self.root, job.cell["id"], job.cfg, job.idx)
        elapsed = time.time() - r["t0"]
        rc = r["proc"].returncode
        status = None
        if os.path.exists(rp):
            try:
                status = read_json(rp)["metrics"].get("status")
            except Exception:
                status = "unreadable"
        elif not interrupted:
            status = {"timeout": "killed_timeout", "memory": "killed_memory"}.get(
                r["killed"], "crashed")
            tail = ""
            try:
                with open(p_log(self.root, job.cell["id"], job.cfg, job.idx)) as f:
                    tail = f.read()[-4000:]
            except Exception:
                pass
            fail = {
                "key": {"cell_id": job.cell["id"], "config_id": job.cfg, "idx": job.idx},
                "cell": job.cell, "config": make_config(job.cfg),
                "instance": {"path": p_inst(self.root, job.cell["id"], job.idx)},
                "run": {"run_id": self.run_id, "host": HOST, "profile": self.a.profile,
                        "design_version": DESIGN_VERSION, "solver_code_hash": self.code_hash,
                        "time_limit": self.time_limit,
                        "mem_limit_gb": round(mem_reservation_gb(job)[1], 2),
                        "returncode": rc, "finished": now_iso()},
                "metrics": {"status": status, "solved": False, "wall_time": elapsed},
                "diag": {}, "solution_edges": None,
                "verification": {"solution_ok": None, "reason": status},
                "error": f"worker ended without a result (returncode {rc}, "
                         f"killed={r['killed']}, peak RSS {r['max_rss']:.0f} MB)\n{tail}",
            }
            write_noclobber(rp, jdump(fail))
        if not interrupted or os.path.exists(rp):
            write_replace(p_monitor(self.root, job.cell["id"], job.cfg, job.idx), jdump({
                "run_id": self.run_id, "elapsed": elapsed, "returncode": rc,
                "max_rss_mb": r["max_rss"], "max_threads": r["max_threads"],
                "killed": r["killed"], "cpu_slot": r["cpu"], "mem_reserved_gb": r["mem"]}))
        try:
            os.unlink(r["claim"])
        except FileNotFoundError:
            pass
        if status is not None:
            self.stats["done"] += 1
            self.stats[status] = self.stats.get(status, 0) + 1
            self.log(f"{self.stats['done']:>5}/{self.n_todo} {job.label():64s} "
                     f"{status:14s} {elapsed:8.1f}s rss={r['max_rss']:.0f}MB")

    def _memory_guard(self):
        """If the MACHINE (all users) is about to run out of memory, stop the
        running job using most memory and requeue it: it reruns later with
        its full limit reserved.  No result is ever recorded under memory
        pressure, so every stored result ran against its own limit only."""
        try:
            import psutil
            free = psutil.virtual_memory().available / 1e9
        except Exception:
            return
        if os.environ.get("MSTKP_TEST_FREE_GB"):          # smoke test only
            free = float(os.environ["MSTKP_TEST_FREE_GB"])
            if self.n_preempted >= int(os.environ.get("MSTKP_TEST_PREEMPTS", "1")):
                return
        testing = bool(os.environ.get("MSTKP_TEST_FREE_GB"))
        floor = max(1.0, 0.05 * (self.mem_total_gb or 80.0))   # 4.6 GB on the 92 GB server
        live = [r for r in self.running.values() if not r["killed"]]
        if free >= floor or len(live) < 2 or time.time() - self.last_preempt < 15:
            return
        victim = max(live, key=lambda r: r["cur_rss"])
        if victim["cur_rss"] < 1024 and not testing:
            return          # low memory is not ours: stopping a small job would not help
        self.log(f"memory pressure ({free:.1f} GB free < {floor:.1f} GB): stopping "
                 f"{victim['job'].label()} ({victim['cur_rss']:.0f} MB); it is requeued and "
                 f"will rerun with its full memory reserved")
        self.heavy.add(victim["job"].key)
        self._kill(victim, "preempted")
        self.last_preempt = time.time()
        self.n_preempted += 1

    def _monitor(self):
        for pid, r in list(self.running.items()):
            proc = r["proc"]
            if proc.poll() is not None:
                del self.running[pid]
                self._finish(pid, r)
                continue
            if r["ps"] is not None:
                try:
                    rss = r["ps"].memory_info().rss
                    thr = r["ps"].num_threads()
                    for ch in r["ps"].children(recursive=True):
                        rss += ch.memory_info().rss
                    r["cur_rss"] = rss / 2 ** 20
                    r["max_rss"] = max(r["max_rss"], rss / 2 ** 20)
                    r["max_threads"] = max(r["max_threads"], thr)
                    if rss / 1e9 > r["kill_gb"] and not r["killed"]:
                        self.log(f"memory limit {r['kill_gb']:.1f} GB exceeded: {r['job'].label()}")
                        self._kill(r, "memory")
                except Exception:
                    pass
            if not r["killed"] and time.time() - r["t0"] > self.time_limit + self.grace:
                self.log(f"hard time cap exceeded: {r['job'].label()}")
                self._kill(r, "timeout")
            if r["killed"] and time.time() - r.get("kill_t", 0) > 15:
                _signal_worker(proc, hard=True)
        self._memory_guard()

    def run(self):
        a = self.a
        self.time_limit = float(a.time_limit or PROFILES[a.profile]["time_limit"])
        self.grace = (float(a.grace) if a.grace is not None
                      else max(60.0, 0.05 * self.time_limit))
        rng = random.Random(a.order_seed)
        todo = [j for j in self.jobs
                if not os.path.exists(p_result(self.root, j.cell["id"], j.cfg, j.idx))]
        self.stats["skipped_done"] = len(self.jobs) - len(todo)
        missing = [j for j in todo if not os.path.exists(p_inst(self.root, j.cell["id"], j.idx))]
        if missing:
            raise SystemExit(f"{len(missing)} jobs have no instance file (first: "
                             f"{missing[0].label()}); run `generate` for these families first.")
        rng.shuffle(todo)
        self.n_todo = len(todo)
        mem_budget = float(a.mem_budget)
        self.log(f"run {self.run_id}: {len(self.jobs)} jobs, {self.stats['skipped_done']} already "
                 f"done, {len(todo)} to run; jobs={a.jobs} mem_budget={mem_budget} GB "
                 f"time_limit={self.time_limit}s pin={bool(self.cpu_slots)} "
                 f"code={self.code_hash} git={self.git}")
        for name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGBREAK"):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), self._signal)
        self.pending = pending = todo
        head_wait = None
        last_beat = time.time()
        while (pending and not self.stop) or self.running:
            launched = True
            while launched and pending and not self.stop and len(self.running) < a.jobs:
                launched = False
                used = self._effective_mem()
                for pos, job in enumerate(pending):
                    res_gb = self._reservation(job)
                    if used + res_gb > mem_budget and self.running:
                        if pos == 0:
                            head_wait = head_wait or time.time()
                            if time.time() - head_wait > 600:
                                break      # stop bypassing: let memory drain for the head
                        continue
                    if pos == 0:
                        head_wait = None
                    pending.pop(pos)
                    if os.path.exists(p_result(self.root, job.cell["id"], job.cfg, job.idx)):
                        self.stats["skipped_done"] += 1
                        self.n_todo -= 1
                        launched = True
                        break
                    cutoff = None
                    if make_config(job.cfg).get("cutoff"):
                        cutoff, k, agree = zstar_for(self.root, job.cell["id"], job.idx)
                        if cutoff is None or not agree:
                            self.stats["blocked"] += 1
                            self.n_todo -= 1
                            self.log(f"blocked (no agreed z* yet): {job.label()}")
                            launched = True
                            break
                    claim = p_claim(self.root, job.cell["id"], job.cfg, job.idx)
                    if not try_claim(claim, {"host": HOST, "runner_pid": os.getpid(),
                                             "run_id": self.run_id, "claimed": now_iso(),
                                             "claimed_epoch": time.time()},
                                     max_age=self.time_limit + self.grace + 3600.0):
                        self.stats["busy_elsewhere"] += 1
                        self.n_todo -= 1
                        launched = True
                        break
                    self._launch(job, cutoff)
                    launched = True
                    break
            time.sleep(0.5)
            self._monitor()
            if self.stop and self.running:
                for r in self.running.values():
                    if not r["killed"]:
                        self._kill(r, "interrupted")
            if time.time() - last_beat > 120:
                last_beat = time.time()
                self.log(f"heartbeat: running={len(self.running)} pending={len(pending)} "
                         f"mem_effective={self._effective_mem():.1f}GB "
                         f"rss_total={sum(r['cur_rss'] for r in self.running.values()) / 1024:.1f}GB "
                         f"load={_loadavg()}")
        if self.stop:
            for pid, r in list(self.running.items()):
                r["proc"].wait()
                self._finish(pid, r, interrupted=True)
        self.stats["preempted_and_rerun"] = self.n_preempted
        self.log(f"finished run {self.run_id}: {json.dumps(self.stats)}")
        return 130 if self.stop else 0


def check_design(root, profile, families):
    """Families and configurations frozen in frozen/design.json must be
    unchanged; families that did not exist at the freeze may be ADDED -- each
    is recorded once in frozen/added_<family>.json with its definition and
    the date, so the audit trail shows what came later."""
    frozen = os.path.join(root, "frozen", "design.json")
    fz = read_json(frozen)["design"]
    cur = design_core(design_snapshot(profile, root))
    for k in ("design_version", "profile", "time_limit", "seed_rule", "calibration"):
        if json.dumps(cur.get(k), sort_keys=True) != json.dumps(fz.get(k), sort_keys=True):
            raise SystemExit(f"design setting {k!r} differs from frozen/design.json -- "
                             f"refusing. Use a new --root for a new design.")
    for f in families:
        if f in fz["families"]:
            if json.dumps(cur["families"][f], sort_keys=True) != \
                    json.dumps(fz["families"][f], sort_keys=True):
                raise SystemExit(f"family {f} differs from its frozen definition -- refusing. "
                                 f"Run it under a new --root.")
        else:
            write_noclobber(os.path.join(root, "frozen", f"added_{f}.json"),
                            jdump({"family": f, "definition": cur["families"][f],
                                   "added": now_iso(), "solver_code_hash": solver_code_hash()}))
        for block in cur["families"][f]:
            for c in block["configs"]:
                if c in fz["configs"] and json.dumps(fz["configs"][c], sort_keys=True) != \
                        json.dumps(cur["configs"][c], sort_keys=True):
                    raise SystemExit(f"configuration {c} differs from its frozen definition")


def _common_check(a, families):
    frozen = os.path.join(a.root, "frozen", "design.json")
    if not os.path.exists(frozen):
        raise SystemExit("no frozen design: run `generate` first")
    fz = read_json(frozen)
    if fz["profile"] != a.profile:
        raise SystemExit(f"root {a.root} was frozen for profile {fz['profile']!r}")
    check_design(a.root, a.profile, families)
    ch = solver_code_hash()
    if ch != fz["solver_code_hash"] and not a.allow_code_change:
        raise SystemExit(f"solver code hash {ch} != frozen {fz['solver_code_hash']}: the "
                         f"algorithm files changed since the freeze. Refusing to mix code "
                         f"versions (override: --allow-code-change).")


def cmd_run(a):
    _common_check(a, parse_families(a.family))
    if a.jobs == "auto" or a.mem_budget == "auto":
        info = machine_info()
        if a.jobs == "auto":
            a.jobs = info["recommended_jobs"]
        if a.mem_budget == "auto":
            a.mem_budget = info["recommended_mem_budget_gb"]
    a.jobs = int(a.jobs)
    a.mem_budget = float(a.mem_budget)
    jobs = expand_jobs(a.profile, a.root, parse_families(a.family))
    if a.retry_failed:
        n = 0
        for j in jobs:
            rp = p_result(a.root, j.cell["id"], j.cfg, j.idx)
            if os.path.exists(rp):
                st = read_json(rp)["metrics"].get("status")
                if st in ("error", "crashed", "unreadable", "killed_memory"):
                    dst = os.path.join(a.root, "results_failed", j.cell["id"], j.cfg,
                                       f"{j.idx:03d}.{int(time.time())}.json")
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    os.replace(rp, dst)
                    n += 1
        print(f"moved {n} failed results to results_failed/ for rerun")
    return Runner(a, jobs).run()


def cmd_run_one(a):
    fam = build_families(a.profile, a.root)
    cell, owners = None, []
    for name, blocks in fam.items():
        for c, idxs, cfgs in blocks:
            if c["id"] == a.cell and a.config in cfgs and a.idx in idxs:
                cell = c
                owners.append(name)
    if cell is None:
        raise SystemExit("no such (cell, config, idx) in the design")
    _common_check(a, owners)
    a.jobs, a.mem_budget = 1, 1e9
    return Runner(a, [Job(cell, a.config, a.idx)]).run()


# =============================================================================
# generate / describe / plan / status
# =============================================================================
def cmd_generate(a):
    snap = design_snapshot(a.profile, a.root)
    frozen = os.path.join(a.root, "frozen", "design.json")
    core = design_core(snap)
    if os.path.exists(frozen):
        check_design(a.root, a.profile, parse_families(a.family))
    else:
        write_noclobber(frozen, jdump({"design": core, "solver_code_hash": solver_code_hash(),
                                       "git": git_state(), "created": now_iso(),
                                       "profile": a.profile}))
        print(f"froze design -> {frozen}")
    fam = build_families(a.profile, a.root)
    work, seen = [], set()
    for name in parse_families(a.family):
        for cell, idxs, _ in fam[name]:
            for i in idxs:
                if (cell["id"], i) not in seen:
                    seen.add((cell["id"], i))
                    work.append((a.profile, cell, i, a.root, a.verify))
    print(f"{len(work)} instances")
    counts = {}
    ghash = {}
    if a.jobs > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.jobs) as pool:
            outs = pool.map(generate_one, work, chunksize=1)
    else:
        outs = [generate_one(w) for w in work]
    for (prof, cell, i, _, _), (path, st, gh) in zip(work, outs):
        counts[st] = counts.get(st, 0) + 1
        if gh is None and os.path.exists(path):
            with gzip.open(path, "rt") as f:
                gh = json.load(f)["meta"]["graph_hash"]
        ghash.setdefault((cell["group"], cell["n"], cell["density"], i), set()).add(gh)
    bad = {k: v for k, v in ghash.items() if len(v) > 1}
    print(f"instances: {counts}")
    if bad:
        raise SystemExit(f"common-random-numbers check FAILED for {len(bad)} groups: cells "
                         f"sharing a seed group do not share the graph and weights")
    print(f"common-random-numbers check passed: {len(ghash)} (group, n, d, idx) graphs, "
          f"each identical across its beta / correlation cells")


def cmd_describe(a):
    fam = build_families(a.profile, a.root)
    tot = expand_jobs(a.profile, a.root, list(fam))
    for name, (kind, text) in FAMILY_INFO.items():
        js = expand_jobs(a.profile, a.root, [name])
        cells = len({c["id"] for c, _, _ in fam[name]})
        print(f"{name:6s} [{kind:8s}] {len(js):5d} jobs  {cells:2d} cells  {text}")
    print(f"total distinct jobs (all families): {len(tot)}; core: "
          f"{len(expand_jobs(a.profile, a.root, CORE_FAMILIES))}")


def cmd_plan(a):
    jobs = expand_jobs(a.profile, a.root, parse_families(a.family))
    lines = [f"{sys.executable} {os.path.abspath(__file__)} run-one --root {a.root} "
             f"--profile {a.profile} --cell {j.cell['id']} --config {j.cfg} --idx {j.idx}"
             for j in jobs]
    if a.out:
        with open(a.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"{len(lines)} commands -> {a.out}")
    else:
        print(f"{len(lines)} jobs")


def cmd_status(a):
    for name in parse_families(a.family):
        js = expand_jobs(a.profile, a.root, [name])
        st = {}
        for j in js:
            rp = p_result(a.root, j.cell["id"], j.cfg, j.idx)
            if os.path.exists(rp):
                s = read_json(rp)["metrics"].get("status", "?")
            elif os.path.exists(p_claim(a.root, j.cell["id"], j.cfg, j.idx)):
                s = "claimed/running"
            else:
                s = "pending"
            st[s] = st.get(s, 0) + 1
        print(f"{name:6s} {len(js):5d} jobs  " + "  ".join(f"{k}={v}" for k, v in sorted(st.items())))


# =============================================================================
# collect
# =============================================================================
PAPER_METRICS = [
    "status", "solved", "obj", "final_lb", "root_lb", "wall_time", "cpu_time",
    "root_time", "nodes", "lr_iterations", "root_lr_iterations", "probes", "probe_time",
    "forced_decisions", "forced_empty", "sep_calls", "sep_time", "cuts_separated",
    "cuts_infeasible", "cut_forced_exclusions", "exact_dual_nodes", "exact_dual_gain",
    "rc_edges_excluded", "rc_edges_fixed", "indicator_calls", "indicator_none",
    "indicator_time", "pool_median", "pool_singleton_share", "pool_empty_share",
    "cutoff", "cutoff_violated",
    # gurobi
    "build_time", "grb_runtime", "user_cuts", "lazy_cuts",
]
CFG_COLS = ["solver", "rung", "rule_tag", "branching_rule", "frac_source", "variant",
            "cover_cuts", "cut_strengthening", "cut_root_only", "max_active_cuts",
            "rank_lift", "exact_cut_dual", "use_rc_fixing", "cutoff", "max_iter",
            "root_max_iter", "formulation"]
META_COLS = ["m", "avg_degree", "rho_realized", "plain_lr_bound", "plain_lr_lambda",
             "min_length_tree_weight", "len_Tw", "len_Tl"]


def cmd_collect(a):
    names = parse_families(a.family)
    fam = build_families(a.profile, a.root)
    out_dir = os.path.join(a.root, "tables")
    os.makedirs(out_dir, exist_ok=True)
    meta_cache, z_cache = {}, {}
    integrity = {"missing": 0, "status": {}, "zstar_disagreements": [],
                 "solution_failures": [], "cutoff_violations": [], "code_hashes": {},
                 "time_limits": {}, "key_mismatch": [], "threads_env_not_1": 0,
                 "multi_threaded_workers": []}
    all_rows = []
    for name in names:
        jobs = expand_jobs(a.profile, a.root, [name])
        rows, drows = [], []
        for j in jobs:
            cid = j.cell["id"]
            if (cid, j.idx) not in meta_cache:
                ip = p_inst(a.root, cid, j.idx)
                meta_cache[(cid, j.idx)] = (load_instance(ip).meta if os.path.exists(ip) else {})
            if (cid, j.idx) not in z_cache:
                z_cache[(cid, j.idx)] = zstar_for(a.root, cid, j.idx)
            zs, zk, agree = z_cache[(cid, j.idx)]
            if zk and not agree:
                integrity["zstar_disagreements"].append(f"{cid}/{j.idx:03d}")
            row = {"family": name, "cell_id": cid, "config_id": j.cfg, "idx": j.idx,
                   "n": j.cell["n"], "density": j.cell["density"], "beta": j.cell["beta"],
                   "knob": j.cell["knob"], "group": j.cell["group"], "zstar": zs}
            cfg = make_config(j.cfg)
            for c in CFG_COLS:
                row["cfg_" + c] = cfg.get(c)
            for c in META_COLS:
                row[c] = meta_cache[(cid, j.idx)].get(c)
            rp = p_result(a.root, cid, j.cfg, j.idx)
            if not os.path.exists(rp):
                row["status"] = "missing"
                integrity["missing"] += 1
                rows.append(row)
                continue
            r = read_json(rp)
            if (r["key"]["cell_id"], r["key"]["config_id"], r["key"]["idx"]) != j.key:
                integrity["key_mismatch"].append(j.label())
            mt = r.get("metrics", {})
            for c in PAPER_METRICS:
                row[c] = mt.get(c)
            run = r.get("run", {})
            row["time_limit"] = run.get("time_limit")
            integrity["time_limits"][str(row["time_limit"])] = \
                integrity["time_limits"].get(str(row["time_limit"]), 0) + 1
            row["solution_ok"] = r.get("verification", {}).get("solution_ok")
            row["solver_code_hash"] = run.get("solver_code_hash")
            row["run_id"] = run.get("run_id")
            row["peak_rss_mb"] = run.get("peak_rss_mb")
            row["os_threads_end"] = run.get("os_threads_end")
            row["error"] = (r.get("error") or "")[:300].replace("\n", " | ")
            mon = p_monitor(a.root, cid, j.cfg, j.idx)
            if os.path.exists(mon):
                mo = read_json(mon)
                row["max_threads"] = mo.get("max_threads")
                row["monitor_rss_mb"] = mo.get("max_rss_mb")
            st = row["status"]
            integrity["status"][st] = integrity["status"].get(st, 0) + 1
            integrity["code_hashes"][row["solver_code_hash"]] = \
                integrity["code_hashes"].get(row["solver_code_hash"], 0) + 1
            if row["solution_ok"] is False:
                integrity["solution_failures"].append(j.label())
            if row.get("cutoff_violated"):
                integrity["cutoff_violations"].append(j.label())
            te = run.get("threads_env") or {}
            if te and any(v != "1" for v in te.values()):
                integrity["threads_env_not_1"] += 1
            thr = row.get("os_threads_end") or row.get("max_threads")
            if thr and cfg["solver"] == "lrbnb" and thr > 1:
                integrity["multi_threaded_workers"].append(f"{j.label()}:{thr}")
            rows.append(row)
            d = {"family": name, "cell_id": cid, "config_id": j.cfg, "idx": j.idx}
            for k, v in (r.get("diag") or {}).items():
                d[k] = json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
            d["cpu_over_wall"] = (mt.get("cpu_time") / mt["wall_time"]
                                  if mt.get("cpu_time") and mt.get("wall_time") else None)
            drows.append(d)
        _write_csv(os.path.join(out_dir, f"{name}_results.csv"), rows)
        _write_csv(os.path.join(out_dir, f"{name}_diagnostics.csv"), drows)
        all_rows.extend(rows)
        print(f"{name:6s} {len(rows)} rows -> tables/{name}_results.csv")
    tag = a.family.replace(",", "_")
    _write_csv(os.path.join(out_dir, f"{tag}_all_results.csv"), all_rows)
    integrity["multi_threaded_workers"] = integrity["multi_threaded_workers"][:50]
    ok = (not integrity["zstar_disagreements"] and not integrity["solution_failures"]
          and not integrity["cutoff_violations"] and not integrity["key_mismatch"]
          and len(integrity["code_hashes"]) <= 1 and not integrity["threads_env_not_1"]
          and not integrity["multi_threaded_workers"])
    integrity["ok"] = ok
    write_replace(os.path.join(out_dir, f"{tag}_integrity.json"), jdump(integrity))
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in integrity.items()},
                     indent=1))
    if not ok:
        print(f"INTEGRITY PROBLEMS -- see tables/{tag}_integrity.json")
        return 2
    return 0


def _write_csv(path, rows):
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)


# =============================================================================
# select-beta (calibration rule for F, frozen before F runs)
# =============================================================================
def cmd_select_beta(a):
    out = os.path.join(a.root, "frozen", "F_beta.json")
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists (frozen); delete it by hand only if you "
                         f"deliberately redo the calibration")
    fam = build_families(a.profile, a.root)
    decision, table = {}, {}
    for fname, spec in CAL_CANDIDATES.items():
        rows = []
        for cell, idxs, cfgs in fam["CAL"]:
            if cell["knob"] != spec["knob"] or cell["beta"] not in spec["betas"]:
                continue
            nodes, solved, total = [], 0, 0
            for i in idxs:
                rp = p_result(a.root, cell["id"], cfgs[0], i)
                if not os.path.exists(rp):
                    raise SystemExit(f"calibration incomplete: {rp}")
                mt = read_json(rp)["metrics"]
                total += 1
                solved += bool(mt.get("solved"))
                if mt.get("nodes") is not None:
                    nodes.append(mt["nodes"])
            med = sorted(nodes)[len(nodes) // 2] if nodes else 0
            rows.append({"beta": cell["beta"], "median_nodes": med,
                         "solved_share": solved / max(1, total)})
        rows.sort(key=lambda r: -r["beta"])        # easiest first
        ok = [r for r in rows if r["median_nodes"] >= CAL_RULE["min_median_nodes"]
              and r["solved_share"] >= CAL_RULE["min_solved_share"]]
        if ok:
            pick, why = ok[0], "largest beta meeting both thresholds"
        else:
            solv = [r for r in rows if r["solved_share"] >= CAL_RULE["min_solved_share"]]
            if solv:
                pick, why = max(solv, key=lambda r: r["median_nodes"]), \
                    "no beta met the node threshold: hardest beta with enough solved"
            else:
                pick, why = max(rows, key=lambda r: r["solved_share"]), \
                    "no beta met the solved threshold: most-solved beta"
        decision[fname] = {"beta": pick["beta"], "knob": spec["knob"], "reason": why}
        table[fname] = rows
        print(f"{fname}: beta = {pick['beta']}  ({why})")
        for r in rows:
            print(f"     beta {r['beta']:.2f}: median nodes {r['median_nodes']:>8}, "
                  f"solved {100 * r['solved_share']:.0f}%")
    write_noclobber(out, jdump({**decision, "table": table, "rule": CAL_RULE,
                                "created": now_iso()}))
    print(f"frozen -> {out}")


# =============================================================================
# try: ad-hoc runs outside the frozen design (the old benchmark_mstkp.py use)
# =============================================================================
TRY_SETS = {
    "ladder": LADDER6, "branching": BRANCH14, "gurobi": GRB3,
    "controls": ["R0-rel-dw-rit40", "R0-rel-dw-it20"],
}


def _expand_try_configs(spec):
    out = []
    for tok in (t.strip() for t in spec.split(",") if t.strip()):
        if tok == "ladder":
            out += [lr_config_id(r, "rel") for r in LADDER6]
        elif tok in TRY_SETS:
            out += TRY_SETS[tok]
        else:
            out.append(tok)
    return list(dict.fromkeys(out))


def cmd_try(a):
    """One instance from (n, density, beta, knob, seed); the named
    configurations run one after another in this process (each run resets
    the solver's class state -- verified to reproduce fresh-process runs
    exactly).  Nothing is written under any benchmark root; --out saves the
    table as JSON if you want it."""
    import contextlib
    import io
    from mstkpinstance import MSTKPInstance
    _highs_single_thread()
    cfg_ids = _expand_try_configs(a.configs)
    for cid in cfg_ids:
        try:
            make_config(cid)
        except Exception:
            raise SystemExit(f"unknown configuration {cid!r}.  Examples: R0-rel-dw, R5-rel-dw, "
                             f"R2-rel-dw-noexact, R0-rel-dw-it20, R5-rmst, R5-mf-avg, GRB-DCUT; "
                             f"sets: ladder, controls, branching, gurobi")
    random.seed(a.seed)
    gen = MSTKPInstance(a.n, a.density, beta=a.beta, corr=a.knob)
    edges = [[int(u), int(v), int(w), int(l)] for u, v, w, l in gen.edges]
    inst = StoredInstance({"num_nodes": a.n, "budget": int(gen.budget), "edges": edges,
                           "cell": {"density": a.density, "beta": a.beta},
                           "hash": instance_hash(a.n, int(gen.budget), edges),
                           "instance_seed": a.seed})
    Ls, lam, mlw = plain_lagrangian(a.n, edges, inst.budget)
    print(f"instance: n={a.n} m={len(edges)} avg degree={2 * len(edges) / a.n:.2f} "
          f"beta={a.beta} knob={a.knob} (rho={gen.empirical_correlation():+.3f}) seed={a.seed} "
          f"budget={inst.budget}\n          plain Lagrangian bound L*={Ls:.1f}  "
          f"min-length-tree weight={mlw:.0f}  time limit={a.time_limit:.0f}s\n")
    head = (f"{'config':22s} {'status':9s} {'obj':>10s} {'root_lb':>11s} {'final_lb':>11s} "
            f"{'nodes':>8s} {'LR iters':>9s} {'probes':>7s} {'cuts':>6s} {'time s':>8s} {'ok':>3s}")
    print(head)
    print("-" * len(head))
    rows, zstar = [], None

    def save():
        if a.out:
            write_replace(os.path.abspath(a.out), jdump({"instance": {
                "n": a.n, "density": a.density, "beta": a.beta, "knob": a.knob,
                "seed": a.seed, "m": len(edges), "budget": inst.budget,
                "plain_lr_bound": Ls}, "runs": rows}))

    for cid in cfg_ids:
        cfg = make_config(cid)
        cutoff = None
        if cfg.get("cutoff"):
            if zstar is None:
                print(f"{cid:22s} skipped: needs an earlier configuration in the list to solve first")
                continue
            cutoff = zstar
        buf = io.StringIO()
        ctx = contextlib.nullcontext() if a.verbose else contextlib.redirect_stdout(buf)
        t0 = time.time()
        try:
            with ctx:
                if cfg["solver"] == "lrbnb":
                    from benchmark_mstkp_ import run_lrbnb
                    m = run_lrbnb(inst, cfg, a.time_limit, a.seed, cutoff=cutoff)
                    m.pop("diag", None)
                else:
                    from gurobi_baselines import run_gurobi
                    m = run_gurobi(inst, cfg["formulation"], a.time_limit)
        except Exception as exc:
            print(f"{cid:22s} ERROR {type(exc).__name__}: {str(exc)[:90]}")
            rows.append({"config": cid, "status": "error", "error": str(exc)})
            save()
            continue
        ok = None if cutoff is not None else verify_solution(inst, m.get("solution_edges"),
                                                             m.get("obj"))[0]
        if m.get("status") == "optimal" and cutoff is None:
            zstar = m["obj"] if zstar is None else min(zstar, m["obj"])
        f = lambda v, w, d=1: (f"{v:{w}.{d}f}" if isinstance(v, (int, float)) and v is not None
                               else f"{'-':>{w}s}")
        print(f"{cid:22s} {m.get('status', '?'):9s} {f(m.get('obj'), 10, 0)} {f(m.get('root_lb'), 11)} "
              f"{f(m.get('final_lb'), 11)} {f(m.get('nodes'), 8, 0)} {f(m.get('lr_iterations'), 9, 0)} "
              f"{f(m.get('probes'), 7, 0)} {f(m.get('cuts_separated', m.get('user_cuts')), 6, 0)} "
              f"{f(m.get('wall_time', time.time() - t0), 8)} "
              f"{'yes' if ok else ('-' if ok is None else 'NO'):>3s}", flush=True)
        m.pop("solution_edges", None)
        rows.append({"config": cid, **m, "solution_ok": ok})
        save()                    # after EVERY run: an interrupted session keeps its rows
    if a.out:
        print(f"\nsaved -> {os.path.abspath(a.out)}")


# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("MSTKP_FINAL_ROOT",
                                                     os.path.join(CODE_DIR, "final_benchmark")))
    ap.add_argument("--profile", default="final", choices=sorted(PROFILES))
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("machine-check")
    sub.add_parser("describe")
    g = sub.add_parser("generate")
    g.add_argument("--family", default="core")
    g.add_argument("--jobs", type=int, default=1)
    g.add_argument("--verify", action="store_true",
                   help="regenerate existing instances and check their hashes")
    for name in ("run", "run-one"):
        r = sub.add_parser(name)
        r.add_argument("--jobs", default="auto")
        r.add_argument("--mem-budget", default="auto")
        r.add_argument("--time-limit", type=float, default=None,
                       help="testing only; the frozen limit is the profile's")
        r.add_argument("--grace", type=float, default=None,
                       help="testing only; seconds past the limit before a hard kill")
        r.add_argument("--pin", action="store_true",
                       help="pin each worker to its own physical core")
        r.add_argument("--order-seed", type=int, default=20260923)
        r.add_argument("--allow-code-change", action="store_true")
        r.add_argument("--retry-failed", action="store_true",
                       help="move error / crashed / killed_memory results aside and rerun them")
        if name == "run":
            r.add_argument("--family", required=True)
        else:
            r.add_argument("--cell", required=True)
            r.add_argument("--config", required=True)
            r.add_argument("--idx", type=int, required=True)
    for name in ("plan", "status", "collect"):
        r = sub.add_parser(name)
        r.add_argument("--family", default="core")
        if name == "plan":
            r.add_argument("--out", default=None)
    sub.add_parser("select-beta")
    hp = sub.add_parser("highs-probe")          # internal, used by machine-check
    hp.add_argument("mode", choices=["fix", "nofix"])
    t = sub.add_parser("try", help="ad-hoc run(s) on one generated instance; writes nothing")
    t.add_argument("--n", "--num-nodes", dest="n", type=int, required=True)
    t.add_argument("--density", type=float, required=True)
    t.add_argument("--beta", type=float, default=0.15)
    t.add_argument("--knob", type=float, default=0.0,
                   help="correlation knob; rho +0.5/-0.5/-0.9 = 0.366/-0.366/-0.674")
    t.add_argument("--seed", type=int, default=7)
    t.add_argument("--configs", default="R5-rel-dw",
                   help="comma list of configuration ids and/or sets: ladder, controls, "
                        "branching, gurobi")
    t.add_argument("--time-limit", type=float, default=1800.0)
    t.add_argument("--verbose", action="store_true", help="show the solver's own output")
    t.add_argument("--out", default=None, help="optional JSON file for the table")
    w = sub.add_parser("worker")
    w.add_argument("spec")

    a = ap.parse_args(argv)
    a.root = os.path.abspath(a.root)
    if a.cmd == "worker":
        return worker_main(a.spec)
    if a.cmd == "highs-probe":
        return _highs_probe(a.mode == "fix") or 0
    return {"machine-check": cmd_machine_check, "describe": cmd_describe,
            "generate": cmd_generate, "run": cmd_run, "run-one": cmd_run_one,
            "plan": cmd_plan, "status": cmd_status, "collect": cmd_collect,
            "select-beta": cmd_select_beta, "try": cmd_try}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
