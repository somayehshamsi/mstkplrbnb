#!/usr/bin/env python3
"""Tables for the cut phase by column generation (families CG and CGX).

    python3 final_suite.py --root ROOT collect --family CG,CGX
    python3 cutdual_report.py --root ROOT [--compare-root OLD_ROOT]

Writes ROOT/paper/CG_pooled, CG_cells, CG_headline (.md, .tex, .csv) and
prints them.  Same definitions as analyze_final.py / paper_tables.py: capped
time (unsolved = time limit), SGM shifts 1 s / 10 nodes, node ratios over
instances solved by both configurations, paired ratios b/a with bootstrap 95%
CIs (stratified by cell when pooled over cells), Wilcoxon p-values with Holm
across the comparisons of one scope.

Configurations (per branching rule, reliability `rel` and pseudo-cost `pc`):
  R0-<rule>-dw      no cuts
  R5-<rule>-dw      strengthened covers, subgradient cut phase (as in the paper)
  R5-<rule>-dw-cg   strengthened covers, cut phase by column generation
  R2-pc-dw-cg       literature covers, cut phase by column generation

--compare-root OLD_ROOT: checks that the instances are bit-identical to the
ones stored under the old root, and that every configuration the old root
also ran (R0/R5 with rel or pc) reproduces its objective and node count
instance by instance -- i.e. that the new code leaves them unchanged.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_final as AF   # noqa: E402
import final_suite as FS     # noqa: E402
import paper_tables as PT    # noqa: E402

C = lambda rung, rule, v="": FS.lr_config_id(rung, rule, "dw", v)


def load(root, TL):
    paths = [os.path.join(root, "tables", f"{f}_results.csv") for f in ("CG", "CGX")]
    have = [p for p in paths if os.path.exists(p)]
    if not have:
        raise SystemExit(f"no tables: run `final_suite.py --root {root} collect --family CG,CGX`")
    df = pd.concat([pd.read_csv(p) for p in have], ignore_index=True)
    df = df.drop_duplicates(["cell_id", "config_id", "idx"])
    return AF.prep(df, TL)


def scopes(fam):
    """Named groups of cells, fixed in advance."""
    cg = [c for c, _, _ in fam["CG"]]
    head = [c["id"] for c in cg if c["group"] == "core" and c["beta"] == 0.15
            and c["knob"] == 0.0 and not c["complete"]]
    corr = [c["id"] for c in cg if abs(c["knob"]) > 1e-9]
    n300 = lambda c: c["group"] in ("core", "grid_d0.10", "grid_d0.20") and not c["complete"]
    grid = [c["id"] for c in cg if n300(c) and c["knob"] == 0.0
            and c["beta"] <= 0.30 and c["id"] not in head]
    loose = [c["id"] for c in cg if n300(c) and c["beta"] >= 0.5]
    scale = [c["id"] for c in cg if str(c["group"]).startswith("scale_")]
    comp = [c["id"] for c in cg if c["complete"]]
    fresh = [c["id"] for c, _, _ in fam["CGX"]]
    out = [("all CG cells", [c["id"] for c in cg]), ("headline", head),
           ("correlation (rho = +0.5, -0.5, -0.9)", corr), ("density x budget grid", grid),
           ("loose budgets (beta 0.5, 0.7)", loose), ("scaling (n = 200-800, degree 15)", scale),
           ("complete graphs (n = 200-500)", comp), ("fresh instances (X)", fresh)]
    return [(name, cells) for name, cells in out if cells]


COMPARISONS = [
    ("cuts, subgradient cut phase, vs none", "R0 → R5", lambda r: (C("R0", r), C("R5", r))),
    ("cuts, CG cut phase, vs none", "R0 → R5-cg", lambda r: (C("R0", r), C("R5", r, "cg"))),
    ("CG vs subgradient cut phase", "R5 → R5-cg", lambda r: (C("R5", r), C("R5", r, "cg"))),
]


def compare_roots(root, old, fam):
    bad_inst, bad_run, n_inst, n_run = [], [], 0, 0
    for f in ("CG", "CGX"):
        for cell, idxs, cfgs in fam[f]:
            for i in idxs:
                a, b = FS.p_inst(root, cell["id"], i), FS.p_inst(old, cell["id"], i)
                if os.path.exists(a) and os.path.exists(b):
                    n_inst += 1
                    if FS.load_instance(a).hash != FS.load_instance(b).hash:
                        bad_inst.append(f"{cell['id']}/{i}")
                for cfg in cfgs:
                    ra, rb = FS.p_result(root, cell["id"], cfg, i), FS.p_result(old, cell["id"], cfg, i)
                    if os.path.exists(ra) and os.path.exists(rb):
                        ma, mb = FS.read_json(ra)["metrics"], FS.read_json(rb)["metrics"]
                        if ma.get("status") == mb.get("status") == "optimal":
                            n_run += 1
                            if ma.get("obj") != mb.get("obj") or ma.get("nodes") != mb.get("nodes"):
                                bad_run.append(f"{cell['id']}/{cfg}/{i}: obj {ma.get('obj')} vs "
                                               f"{mb.get('obj')}, nodes {ma.get('nodes')} vs {mb.get('nodes')}")
    print(f"compare-root: {n_inst} instances compared, {len(bad_inst)} differ; "
          f"{n_run} runs of unchanged configurations compared (obj and nodes), {len(bad_run)} differ")
    for x in (bad_inst + bad_run)[:20]:
        print("   ", x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--profile", default="final")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--compare-root", default=None)
    a = ap.parse_args()
    root = os.path.abspath(os.path.expanduser(a.root))
    TL = FS.PROFILES[a.profile]["time_limit"]
    out = os.path.join(root, "paper")
    os.makedirs(out, exist_ok=True)
    fam = FS.build_families(a.profile, root)
    if a.compare_root:
        compare_roots(root, os.path.abspath(os.path.expanduser(a.compare_root)), fam)
    df = load(root, TL)
    rng = np.random.default_rng(PT.SEED)
    info = {c["id"]: c for f in ("CG", "CGX") for c, _, _ in fam[f]}
    present = lambda cid, cfg: bool(((df.cell_id == cid) & (df.config_id == cfg)).any())

    # ---- CG_pooled: the declared comparisons, per scope -----------------------
    rows = []
    for scope, cells in scopes(fam):
        cells = [c for c in cells if (df.cell_id == c).any()]
        if not cells:
            continue
        block = []
        for label, arrow, pair in COMPARISONS:
            for rule in ("rel", "pc"):
                x, y = pair(rule)
                r = PT.pooled(df, x, y, cells, a.boot, rng)
                if r:
                    block.append({"scope": scope, "comparison": f"{label} ({rule})",
                                  "a → b": f"{x} → {y}", **r})
        if all(present(c, C("R2", "pc", "cg")) for c in cells):
            r = PT.pooled(df, C("R2", "pc", "cg"), C("R5", "pc", "cg"), cells, a.boot, rng)
            if r:
                block.append({"scope": scope, "comparison": "strengthened vs literature covers, CG (pc)",
                              "a → b": f"{C('R2', 'pc', 'cg')} → {C('R5', 'pc', 'cg')}", **r})
        r = PT.pooled(df, C("R0", "rel"), C("R5", "pc", "cg"), cells, a.boot, rng)
        if r:
            block.append({"scope": scope, "comparison": "CG cuts with pc vs the paper's R0 (rel)",
                          "a → b": f"{C('R0', 'rel')} → {C('R5', 'pc', 'cg')}", **r})
        if block:
            B = pd.DataFrame(block)
            B["node_p_holm"] = AF.holm(B.node_p.fillna(1.0))
            B["time_p_holm"] = AF.holm(B.time_p.fillna(1.0))
            rows.append(B)
    if rows:
        R = pd.concat(rows, ignore_index=True)
        R.to_csv(os.path.join(out, "CG_pooled_raw.csv"), index=False)
        T = pd.DataFrame({
            "scope": R.scope, "comparison": R.comparison, "a → b": R["a → b"],
            "instances": R.n, "solved a / b": R.solved_a.astype(str) + " / " + R.solved_b.astype(str),
            "node ratio [95% CI]": [PT.ratio_ci(*v) for v in zip(R.node_ratio, R.node_lo, R.node_hi)],
            "p (nodes, Holm)": [PT.fmt_num(v) for v in R.node_p_holm],
            "time ratio [95% CI]": [PT.ratio_ci(*v) for v in zip(R.time_ratio, R.time_lo, R.time_hi)],
            "p (time, Holm)": [PT.fmt_num(v) for v in R.time_p_holm]})
        PT.write_table(out, "CG_pooled", "Cut phase by column generation: pooled comparisons", T,
                       "Ratio b/a of geometric means over paired instances (< 1: b better); nodes on "
                       "instances solved by both; 95% CI from a bootstrap stratified by cell; Wilcoxon "
                       "p-values, Holm-adjusted within each scope.")

    # ---- CG_cells: every cell -----------------------------------------------------
    rows = []
    counts = {}
    for cid, c in info.items():
        sub = df[df.cell_id == cid]
        if sub.empty:
            continue
        row = {"cell": PT.cell_label(c), "instances": int(sub.idx.nunique()),
               "covers (median, R5-pc-cg)": PT.fmt_num(
                   sub[sub.config_id == C("R5", "pc", "cg")].cuts_separated.median())}
        for rule in ("pc", "rel"):
            for name, x, y in (("R5-cg/R0", C("R0", rule), C("R5", rule, "cg")),
                               ("R5-cg/R5", C("R5", rule), C("R5", rule, "cg")),
                               ("R5/R0", C("R0", rule), C("R5", rule))):
                if not (present(cid, x) and present(cid, y)):
                    continue
                r = AF.paired(sub, x, y, min(a.boot, 5000), rng)
                if name != "R5/R0":
                    row[f"nodes {name} ({rule})"] = PT.fmt_num(r["node_ratio"])
                    row[f"time {name} ({rule})"] = PT.fmt_num(r["time_ratio"])
                k = (name, rule)
                tot, below_n, below_t = counts.get(k, (0, 0, 0))
                counts[k] = (tot + 1, below_n + int(r["node_ratio"] < 1), below_t + int(r["time_ratio"] < 1))
        rows.append(row)
    if rows:
        note = "; ".join(f"{name} ({rule}): nodes < 1 in {bn} of {t} cells, time < 1 in {bt}"
                         for (name, rule), (t, bn, bt) in sorted(counts.items()))
        PT.write_table(out, "CG_cells", "Cut phase by column generation: every cell",
                       pd.DataFrame(rows), "Paired ratios (< 1: the second configuration better); "
                       "nodes on instances solved by both. " + note + ".")
        print(note)

    # ---- CG_headline: work per node on the headline cell -----------------------------
    head = next((cid for cid, c in info.items() if c["group"] == "core" and c["beta"] == 0.15
                 and c["knob"] == 0.0 and not c["complete"]), None)
    if head is not None and (df.cell_id == head).any():
        order = [(C("R0", "rel"), "R0, reliability"), (C("R5", "rel"), "R5, reliability"),
                 (C("R5", "rel", "cg"), "R5-cg, reliability"), (C("R0", "pc"), "R0, pseudo-cost"),
                 (C("R5", "pc"), "R5, pseudo-cost"), (C("R5", "pc", "cg"), "R5-cg, pseudo-cost"),
                 (C("R2", "pc", "cg"), "R2-cg, pseudo-cost")]
        order = [(c, l) for c, l in order if present(head, c)]
        H = PT.common_rows(df, head, [c for c, _ in order])
        S = AF.summary(H, [c for c, _ in order]).set_index("config")
        rows = []
        for c, l in order:
            g = H[H.config_id == c]
            nodes = g.nodes.clip(lower=1).astype(float)
            wall = g.wall_time.astype(float).clip(lower=1e-9)
            rows.append({
                "configuration": l, "solved": f"{int(g.solved.sum())}/{len(g)}",
                "SGM time (s)": PT.fmt_num(S.loc[c, "sgm_time"]),
                "SGM nodes": PT.fmt_num(S.loc[c, "sgm_nodes_common"]),
                "MSTs / node": PT.fmt_num(float((g.mst_evaluations.astype(float) / nodes).median())),
                "root gap (%)": PT.fmt_num(S.loc[c, "median_root_gap_pct"]),
                "covers (median)": PT.fmt_num(float(g.cuts_separated.median())),
                "separation share": PT.fmt_num(float((g.sep_time.astype(float) / wall).median())),
                "LP share": PT.fmt_num(float((g.cg_lp_time.fillna(0).astype(float) / wall).median()))
                if "cg_lp_time" in g else "–"})
        PT.write_table(out, "CG_headline", f"Headline cell ({PT.cell_label(info[head])}): work per node",
                       pd.DataFrame(rows), "SGM time over all instances, SGM nodes over the commonly solved "
                       "ones; MSTs per node include the exact plain dual, the cut phase (subgradient "
                       "iterations or pricing steps) and strong-branching probes; shares are medians of "
                       "per-run shares of the wall time.")

    for name in ("CG_pooled", "CG_cells", "CG_headline"):
        p = os.path.join(out, f"{name}.md")
        if os.path.exists(p):
            print(open(p).read())
    print(f"tables -> {out}/CG_pooled.*, CG_cells.*, CG_headline.*")


if __name__ == "__main__":
    sys.exit(main())
