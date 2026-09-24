#!/usr/bin/env python3
"""Assertions over a finished smoke/tiny run (or the final run).

    python smoke_check.py --root ROOT --profile smoke --family core

Checks, per result file:
  * every planned job has exactly one result, keyed by its own path;
  * configuration ids map to distinct configurations;
  * the parameters read back FROM THE SOLVER OBJECTS (node solver and
    strong-branching probe solver) equal the declared configuration;
  * behavioural signatures: the components a configuration switches off
    really did not run (no cuts in R0, no separation below the root in R1,
    no rank lifting in R4 -- probes included --, no exact cut dual with
    `noexact`, no reduced-cost fixing with `norc`, no probes under rules that
    have none, no indicator under MST rules, cutoff runs used z*);
  * all paper metrics are present; solutions verified; z* agrees across
    every configuration and formulation; one solver code version; thread
    environment pinned to 1 and single-threaded LR workers.
Exit status 0 only if everything passes.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_suite as FS  # noqa: E402

LR_REQUIRED = ["status", "solved", "obj", "final_lb", "root_lb", "wall_time", "cpu_time",
               "nodes", "lr_iterations", "root_lr_iterations", "probes", "sep_time", "cuts_separated",
               "exact_dual_nodes", "rc_edges_excluded", "indicator_calls", "indicator_time"]
GRB_REQUIRED = ["status", "solved", "obj", "final_lb", "root_lb", "wall_time", "nodes",
                "build_time", "grb_runtime", "formulation"]
OK_STATUSES = {"optimal", "timeout", "memory"}   # memory = stopped at the memory limit, with bounds


class Report:
    def __init__(self):
        self.fail, self.passed, self.warn = [], 0, []

    def note(self, cond, msg):
        """Behaviour that depends on the instance, not the configuration:
        reported, never a failure."""
        if not cond:
            self.warn.append(msg)

    def check(self, cond, msg):
        if cond:
            self.passed += 1
        else:
            self.fail.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--family", default="core")
    ap.add_argument("--allow-status", default="",
                    help="comma list of extra statuses to accept (e.g. error for "
                         "Gurobi under a size-limited licence)")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    allowed = OK_STATUSES | {s for s in a.allow_status.split(",") if s}
    R = Report()

    # configuration ids are distinct configurations
    fam = FS.build_families(a.profile, root)
    all_cfgs = sorted({c for b in fam.values() for _, _, cs in b for c in cs})
    blobs = {}
    for c in all_cfgs:
        blob = json.dumps(FS.make_config(c), sort_keys=True)
        R.check(blob not in blobs, f"configs {c} and {blobs.get(blob)} are identical")
        blobs[blob] = c

    jobs = FS.expand_jobs(a.profile, root, FS.parse_families(a.family))
    agg = {}          # (cell, cfg) -> accumulated behaviour
    zs = {}
    hashes = set()
    for j in jobs:
        rp = FS.p_result(root, j.cell["id"], j.cfg, j.idx)
        if not os.path.exists(rp):
            R.check(False, f"missing result {j.label()}")
            continue
        r = FS.read_json(rp)
        lab = j.label()
        R.check((r["key"]["cell_id"], r["key"]["config_id"], r["key"]["idx"]) == j.key,
                f"{lab}: result key does not match its path")
        cfg = FS.make_config(j.cfg)
        R.check(r["config"] == cfg, f"{lab}: stored config differs from the design")
        mt, dg, run = r["metrics"], r.get("diag") or {}, r.get("run") or {}
        st = mt.get("status")
        R.check(st in allowed, f"{lab}: status {st}: {(r.get('error') or '')[-300:]}")
        hashes.add(run.get("solver_code_hash"))
        te = run.get("threads_env") or {}
        R.check(te and all(v == "1" for v in te.values()), f"{lab}: thread env {te}")
        if cfg["solver"] == "lrbnb":
            thr = run.get("os_threads_end")
            if thr is None:
                mon = FS.p_monitor(root, j.cell["id"], j.cfg, j.idx)
                thr = FS.read_json(mon).get("max_threads") if os.path.exists(mon) else None
            R.check(not thr or thr <= 1, f"{lab}: LR worker ran {thr} threads")
        if st not in OK_STATUSES:
            continue
        req = LR_REQUIRED if cfg["solver"] == "lrbnb" else GRB_REQUIRED
        for k in req:
            R.check(k in mt and (mt[k] is not None or k in ("final_lb", "obj")),
                    f"{lab}: metric {k} missing")
        if st == "optimal" and not cfg.get("cutoff"):
            R.check(r["verification"]["solution_ok"] is True,
                    f"{lab}: solution not verified: {r['verification']}")
            zs.setdefault((j.cell["id"], j.idx), set()).add(round(float(mt["obj"]), 6))
        if cfg["solver"] != "lrbnb":
            R.check(mt.get("formulation") == cfg["formulation"], f"{lab}: formulation")
            continue

        eff = dg.get("effective_params") or {}
        peff = dg.get("effective_probe_params") or {}
        R.check(eff.get("branching_rule") == cfg["branching_rule"], f"{lab}: branching rule")
        R.check(eff.get("use_cover_cuts") == cfg["cover_cuts"], f"{lab}: use_cover_cuts")
        R.check(eff.get("max_iter") == cfg["max_iter"], f"{lab}: max_iter {eff.get('max_iter')}")
        R.check(eff.get("root_max_iter") == cfg.get("root_max_iter"),
                f"{lab}: root_max_iter {eff.get('root_max_iter')}")
        want_root = (cfg.get("root_max_iter") or 2 * cfg["max_iter"]) * (4 if cfg["cover_cuts"] else 1)
        R.check(mt.get("root_lr_iterations", 0) <= want_root,
                f"{lab}: {mt.get('root_lr_iterations')} root iterations > budget {want_root}")
        for k in ("rank_lift", "exact_cut_dual", "use_rc_fixing", "frac_source"):
            R.check(eff.get(k) == cfg[k], f"{lab}: node solver {k}={eff.get(k)} != {cfg[k]}")
        if cfg["cover_cuts"]:
            R.check(eff.get("cut_strengthening") == cfg["cut_strengthening"], f"{lab}: strength")
            R.check(eff.get("max_active_cuts") == cfg["max_active_cuts"], f"{lab}: pool cap")
            want_depth = 0 if cfg["cut_root_only"] else "inf"
            R.check(eff.get("max_cut_depth") == want_depth, f"{lab}: max_cut_depth")
            if peff:
                for k in ("cut_strengthening", "rank_lift", "exact_cut_dual", "max_active_cuts"):
                    R.check(peff.get(k) == cfg[k],
                            f"{lab}: PROBE solver {k}={peff.get(k)} != {cfg[k]}")
                R.check(peff.get("max_cut_depth") == want_depth, f"{lab}: probe depth cap")
        # per-run behavioural signatures
        if not cfg["cover_cuts"]:
            R.check(mt["cuts_separated"] == 0 and mt["sep_calls"] == 0, f"{lab}: R0 separated cuts")
            R.check(mt["exact_dual_nodes"] == 0, f"{lab}: R0 ran the exact cut dual")
        if cfg["cover_cuts"] and cfg["cut_root_only"]:
            R.check(dg.get("sep_max_depth", -1) <= 0, f"{lab}: root-only separated at depth "
                                                      f"{dg.get('sep_max_depth')}")
        if not cfg["rank_lift"] or not cfg["cover_cuts"] or cfg["cut_strengthening"] != "full":
            R.check(dg.get("rank_lift_calls", 0) == 0, f"{lab}: rank lifting ran")
        if not cfg["exact_cut_dual"]:
            R.check(mt["exact_dual_nodes"] == 0 and dg.get("exact_dual_probe_nodes", 0) == 0,
                    f"{lab}: exact cut dual ran with exact_cut_dual=False")
        if not cfg["use_rc_fixing"]:
            R.check(mt["rc_edges_excluded"] == 0 and mt["rc_edges_fixed"] == 0,
                    f"{lab}: RC fixing ran with use_rc_fixing=False")
        tag = cfg["rule_tag"]
        if tag in ("rmst", "sbmst"):
            R.check(mt["indicator_calls"] == 0, f"{lab}: MST rule built an indicator")
        if tag in ("rmst", "mf", "rfrac", "pc"):
            R.check(mt["probes"] == 0, f"{lab}: rule without probes ran {mt['probes']} probes")
        if cfg.get("cutoff"):
            R.check(mt.get("cutoff") is not None and dg.get("step_reference") is not None,
                    f"{lab}: cutoff run without cutoff/step reference")
            R.check(not mt.get("cutoff_violated"), f"{lab}: CUTOFF VIOLATED (z* wrong?)")
            if st == "optimal":
                R.check(abs(mt["obj"] - mt["cutoff"]) < 1e-6, f"{lab}: cutoff obj")
        else:
            R.check(dg.get("step_reference") is None, f"{lab}: step reference outside cutoff")
        A = agg.setdefault((j.cell["id"], j.cfg), {"cfg": cfg, "nodes": 0, "probes": 0,
                                                    "rank": 0, "exact": 0, "rc": 0,
                                                    "ind": 0, "cuts": 0, "branched": 0,
                                                    "solved_branched": 0})
        A["nodes"] += mt["nodes"]
        A["probes"] += mt["probes"]
        A["rank"] += dg.get("rank_lift_calls", 0)
        A["exact"] += mt["exact_dual_nodes"]
        A["rc"] += mt["rc_edges_excluded"] + mt["rc_edges_fixed"]
        A["ind"] += mt["indicator_calls"]
        A["cuts"] += mt["cuts_separated"]
        A["branched"] += int(mt["nodes"] > 1)
        A["solved_branched"] += int(mt["nodes"] > 1 and st == "optimal")

    # aggregate signatures: the components a configuration switches ON did run
    for (cid, c), A in agg.items():
        cfg = A["cfg"]
        if not A["branched"]:
            continue
        lab = f"{cid}/{c}"
        if cfg["rule_tag"] in ("rel", "sbf", "hyb", "sbmst"):
            R.check(A["probes"] > 0, f"{lab}: probing rule never probed")
        if cfg["rule_tag"] not in ("rmst", "sbmst"):
            R.check(A["ind"] > 0, f"{lab}: indicator rule never built an indicator")
        if cfg["cover_cuts"]:
            R.check(A["cuts"] > 0, f"{lab}: cut rung separated no cuts")
            if cfg["exact_cut_dual"]:
                R.note(A["exact"] > 0, f"{lab}: exact cut dual never raised a bound "
                                       f"(enabled, read back from the solvers; nothing to improve)")
        # RC fixing needs an incumbent near the bound; a group whose runs all
        # timed out on the seeded incumbent legitimately fixes nothing.
        if cfg["use_rc_fixing"] and A["solved_branched"]:
            R.note(A["rc"] > 0, f"{lab}: RC fixing never fixed an edge")

    for key, vals in zs.items():
        R.check(len(vals) == 1, f"{key}: optimal objectives disagree across configs: {vals}")
    R.check(len(hashes - {None}) <= 1, f"results come from several code versions: {hashes}")

    print(f"{R.passed} checks passed, {len(R.fail)} failed "
          f"({len(jobs)} jobs, {len(agg)} LR (cell, config) groups, {len(zs)} instances with z*)")
    for w in R.warn[:40]:
        print("  note", w)
    for f in R.fail[:80]:
        print("  FAIL", f)
    return 1 if R.fail else 0


if __name__ == "__main__":
    sys.exit(main())
