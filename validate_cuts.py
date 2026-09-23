#!/usr/bin/env python3
"""Experiment V: correctness of the frozen benchmark code on small instances.

For every instance (n = 7, 8; dense and sparse; three budgets; three
correlations) all spanning trees are enumerated, and then:

  1. every cover cut generated anywhere (node solves AND strong-branching
     probes) under every cut configuration is checked against every
     budget-feasible spanning tree consistent with the fixings of the node
     that generated it;
  2. every LR-BnB configuration of the design returns the brute-force
     optimum (cutoff configurations are handed that optimum);
  3. every root and final lower bound is <= z*;
  4. the exact plain Lagrangian bound L* (the root-gap denominator) is <= z*;
  5. every Gurobi formulation returns z*, and DMCF / DCUT root bounds are
     >= L* (their LP is the spanning-tree polytope), as claimed in the paper.

    python validate_cuts.py [--out validation_report.json] [--no-gurobi]

Exit status 0 only if every check passes.  Runs in one process, so it also
exercises the per-run reset of the solver's class-level state.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import hashlib
import io
import itertools
import json
import random
import sys
import time
from contextlib import redirect_stdout

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_suite as FS                         # noqa: E402
from mstkpinstance import MSTKPInstance          # noqa: E402
from lagrangianrelaxation import LagrangianMST   # noqa: E402
from benchmark_mstkp_ import run_lrbnb           # noqa: E402


def prufer_trees(n):
    """All labelled trees on n vertices as tuples of sorted (u, v) pairs."""
    out = []
    for seq in itertools.product(range(n), repeat=n - 2):
        degree = [1] * n
        for x in seq:
            degree[x] += 1
        edges = []
        for x in seq:
            for leaf in range(n):
                if degree[leaf] == 1:
                    edges.append((min(leaf, x), max(leaf, x)))
                    degree[leaf] -= 1
                    degree[x] -= 1
                    break
        u, v = [i for i in range(n) if degree[i] == 1]
        edges.append((u, v))
        out.append(tuple(edges))
    return out


_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount(a):
    a = a.astype(np.uint64)
    return sum(_POP[((a >> np.uint64(8 * k)) & np.uint64(255)).astype(np.int64)].astype(np.int64)
               for k in range(8))


class Brute:
    def __init__(self, inst, trees_by_n):
        self.idx = {(u, v): i for i, (u, v, w, l) in enumerate(inst.edges)}
        W = [e[2] for e in inst.edges]
        L = [e[3] for e in inst.edges]
        masks, ws, ls = [], [], []
        for t in trees_by_n[inst.num_nodes]:
            try:
                ids = [self.idx[e] for e in t]
            except KeyError:
                continue
            masks.append(sum(1 << i for i in ids))
            ws.append(sum(W[i] for i in ids))
            ls.append(sum(L[i] for i in ids))
        self.mask = np.array(masks, dtype=np.uint64)
        self.w = np.array(ws, dtype=np.int64)
        self.l = np.array(ls, dtype=np.int64)
        self.B = inst.budget
        feas = self.l <= self.B
        self.zstar = int(self.w[feas].min())
        self.n_trees = len(masks)

    def emask(self, edges):
        m = 0
        for e in edges:
            m |= 1 << self.idx[(min(e), max(e))]
        return np.uint64(m)

    def cut_violations(self, fixed, excluded, budget, cuts):
        fm, xm = self.emask(fixed), self.emask(excluded)
        ok = ((self.mask & fm) == fm) & ((self.mask & xm) == 0) & (self.l <= budget)
        T = self.mask[ok]
        bad = []
        if T.size == 0:
            return bad, 0
        for support, rhs in cuts:
            cnt = popcount(T & self.emask(support))
            if int(cnt.max()) > rhs:
                bad.append((sorted(support), rhs, int(cnt.max())))
        return bad, int(T.size)


def instances():
    out = []
    k = 0
    for n in (7, 8):
        for d in (1.0, 0.7):
            for beta in (0.08, 0.15, 0.30):
                for knob in (0.0, -0.674, 0.366):
                    seed = FS.instance_seed("validate", f"n{n}", k)
                    random.seed(seed)
                    inst = MSTKPInstance(n, d, beta=beta, corr=knob)
                    out.append((f"v{k:02d}_n{n}_d{d}_b{beta}_k{knob:+.3f}", seed, inst))
                    k += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="validation_report.json")
    ap.add_argument("--no-gurobi", action="store_true")
    ap.add_argument("--only-gurobi", action="store_true",
                    help="skip the LR-BnB runs (quick recheck of the MIP baselines)")
    ap.add_argument("--time-limit", type=float, default=60.0)
    a = ap.parse_args()

    fam = FS.build_families("final", "/nonexistent")
    configs = sorted({c for b in fam.values() for _, _, cs in b for c in cs
                      if not c.startswith("GRB-")})
    trees = {n: prufer_trees(n) for n in (7, 8)}
    report = {"configs": configs, "instances": [], "failures": [], "counts": {}}
    C = report["counts"]
    for key in ("lr_runs", "grb_runs", "cuts_checked", "cut_events", "probe_cut_events",
                "tree_checks"):
        C[key] = 0
    t0 = time.time()
    for name, seed, inst in instances():
        bf = Brute(inst, trees)
        Ls, lam, _ = FS.plain_lagrangian(inst.num_nodes, [list(e) for e in inst.edges], inst.budget)
        rec = {"name": name, "m": len(inst.edges), "trees": bf.n_trees, "zstar": bf.zstar,
               "plain_lr": Ls}
        report["instances"].append(rec)
        if Ls > bf.zstar + 1e-6:
            report["failures"].append(f"{name}: L* {Ls} > z* {bf.zstar}")
        for cid in ([] if a.only_gurobi else configs):
            cfg = FS.make_config(cid)
            LagrangianMST.cut_log = []
            with redirect_stdout(io.StringIO()):
                r = run_lrbnb(inst, cfg, a.time_limit, seed,
                              cutoff=bf.zstar if cfg.get("cutoff") else None)
            log, LagrangianMST.cut_log = LagrangianMST.cut_log, None
            C["lr_runs"] += 1
            tag = f"{name}/{cid}"
            if r["status"] != "optimal":
                report["failures"].append(f"{tag}: status {r['status']}")
                continue
            if abs(r["obj"] - bf.zstar) > 1e-6:
                report["failures"].append(f"{tag}: obj {r['obj']} != z* {bf.zstar}")
            for bound in ("root_lb", "final_lb"):
                if r[bound] is not None and r[bound] > bf.zstar + 1e-6:
                    report["failures"].append(f"{tag}: {bound} {r[bound]} > z* {bf.zstar}")
            if not cfg.get("cutoff"):
                ok, why = FS.verify_solution(FS.StoredInstance({
                    "num_nodes": inst.num_nodes, "budget": inst.budget,
                    "edges": [list(e) for e in inst.edges],
                    "cell": {"density": 0, "beta": 0}, "hash": "", "instance_seed": seed}),
                    r["solution_edges"], r["obj"])
                if not ok:
                    report["failures"].append(f"{tag}: solution {why}")
            for fixed, excluded, budget, cuts, is_probe in log:
                if not cuts:
                    continue
                C["cut_events"] += 1
                C["probe_cut_events"] += int(is_probe)
                C["cuts_checked"] += len(cuts)
                bad, ntrees = bf.cut_violations(fixed, excluded, budget, cuts)
                C["tree_checks"] += ntrees * len(cuts)
                for support, rhs, got in bad[:3]:
                    report["failures"].append(
                        f"{tag}: INVALID CUT {'(probe) ' if is_probe else ''}"
                        f"sum x[{support}] <= {rhs} violated by a feasible tree ({got})")
        if not a.no_gurobi:
            try:
                from gurobi_baselines import run_gurobi
                for form in FS.GRB_FORMS:
                    g = run_gurobi(inst, form, a.time_limit)
                    C["grb_runs"] += 1
                    tag = f"{name}/GRB-{form}"
                    if g["status"] != "optimal" or abs(g["obj"] - bf.zstar) > 1e-6:
                        report["failures"].append(f"{tag}: {g['status']} obj {g['obj']} "
                                                  f"z* {bf.zstar}")
                    if form in ("DMCF", "DCUT") and g["root_lb"] < Ls - 1e-4 * max(1, abs(Ls)):
                        report["failures"].append(f"{tag}: root bound {g['root_lb']} < L* {Ls}")
            except ImportError:
                report["gurobi"] = "gurobipy not available: Gurobi checks skipped"
        print(f"{name:34s} trees={bf.n_trees:6d} z*={bf.zstar:6d} L*={Ls:10.1f} "
              f"failures so far={len(report['failures'])}", flush=True)
    report["seconds"] = time.time() - t0
    report["ok"] = not report["failures"]
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(C), f"\n{len(report['failures'])} failures -> {a.out}")
    for f in report["failures"][:40]:
        print("  FAIL", f)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
