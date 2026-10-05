#!/usr/bin/env python3
"""Tables for the classical baseline (family K) and the DCUT callback timing (GT).

    python3 final_suite.py --root ROOT collect --family K,GT
    python3 classical_report.py --root ROOT [--compare-root OLD_ROOT]

Writes ROOT/paper/K_classical.{md,tex,csv} and ROOT/paper/GT_callback.{md,tex,csv}
and prints them.  Same definitions as analyze_final.py / paper_tables.py:
capped time (unsolved = time limit), SGM shifts 1 s / 10 nodes, node statistics
over instances solved by every configuration compared, paired ratios b/a with
bootstrap 95% CIs (stratified by cell when pooled over cells).

--compare-root OLD_ROOT: checks that every K/GT instance is bit-identical to
the one stored under the old root, and that R0-rel-dw (unchanged by the new
code) reproduces the old root's objective and node count instance by instance.
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

OURS = FS.lr_config_id("R0", "rel")
NAMES = {
    FS.lr_config_id("R0", "rel"): "fractional edges, reliability; fixing at every node (ours)",
    FS.lr_config_id("R0", "rfrac"): "fractional edges, random; fixing at every node",
    FS.lr_config_id("R0", "ftree"): "feasible-tree partition; fixing at every node",
    FS.lr_config_id("R0", "rel", variant="rootfix"): "fractional edges, reliability; fixing at the root only",
    FS.lr_config_id("R0", "ftree", variant="rootfix"): "feasible-tree partition; fixing at the root only (classical)",
}


def load(root, fams, TL):
    paths = [os.path.join(root, "tables", f"{f}_results.csv") for f in fams]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing {missing}: run `final_suite.py --root {root} collect --family K,GT`")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df.drop_duplicates(["cell_id", "config_id", "idx"])
    return AF.prep(df, TL)


def compare_roots(root, old, fam):
    bad_inst, bad_run, n_inst, n_run = [], [], 0, 0
    for f in ("K", "GT"):
        for cell, idxs, cfgs in fam[f]:
            for i in idxs:
                a, b = FS.p_inst(root, cell["id"], i), FS.p_inst(old, cell["id"], i)
                if os.path.exists(a) and os.path.exists(b):
                    n_inst += 1
                    if FS.load_instance(a).hash != FS.load_instance(b).hash:
                        bad_inst.append(f"{cell['id']}/{i}")
                if f == "K":
                    ra, rb = FS.p_result(root, cell["id"], OURS, i), FS.p_result(old, cell["id"], OURS, i)
                    if os.path.exists(ra) and os.path.exists(rb):
                        ma, mb = FS.read_json(ra)["metrics"], FS.read_json(rb)["metrics"]
                        if ma.get("status") == mb.get("status") == "optimal":
                            n_run += 1
                            if ma.get("obj") != mb.get("obj") or ma.get("nodes") != mb.get("nodes"):
                                bad_run.append(f"{cell['id']}/{i}: obj {ma.get('obj')} vs {mb.get('obj')}, "
                                               f"nodes {ma.get('nodes')} vs {mb.get('nodes')}")
    print(f"compare-root: {n_inst} instances compared, {len(bad_inst)} differ; "
          f"{n_run} {OURS} runs compared (obj and nodes), {len(bad_run)} differ")
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
    df = load(root, ["K", "GT"], TL)
    rng = np.random.default_rng(PT.SEED)
    kcells = [c for c, _, _ in fam["K"]]
    label = {c["id"]: PT.cell_label(c) for c in kcells}

    # ---- K: one row per (cell, configuration), plus all cells pooled --------
    rows = []
    for scope in [[c["id"]] for c in kcells] + [[c["id"] for c in kcells]]:
        name = label[scope[0]] if len(scope) == 1 else "all three cells"
        sub = pd.concat([PT.common_rows(df, cid, FS.CLASSICAL) for cid in scope])
        solved_all = (sub.pivot_table(index=["cell_id", "idx"], columns="config_id", values="solved",
                                      aggfunc="first").reindex(columns=FS.CLASSICAL)
                      .fillna(False).astype(bool).all(axis=1))
        common = set(solved_all[solved_all].index)
        for cfg in FS.CLASSICAL:
            g = sub[sub.config_id == cfg]
            gc = g[[(c, i) in common for c, i in zip(g.cell_id, g.idx)]]
            row = {"instances": name, "configuration": NAMES[cfg],
                   "solved": f"{int(g.solved.sum())}/{len(g)}",
                   "SGM time (s)": PT.fmt_num(AF.sgm(g.capped_time, AF.SHIFT_T)),
                   "SGM nodes": PT.fmt_num(AF.sgm(gc.nodes, AF.SHIFT_N))}
            if cfg == OURS:
                row["time ratio to ours [95% CI]"] = "1 (reference)"
            else:
                r = PT.pooled(df, OURS, cfg, scope, a.boot, rng)
                row["time ratio to ours [95% CI]"] = (PT.ratio_ci(r["time_ratio"], r["time_lo"], r["time_hi"])
                                                       if r else "–")
            rows.append(row)
    PT.write_table(out, "K_classical", "Classical baseline: branching and reduced-cost fixing",
                   pd.DataFrame(rows),
                   "All configurations without cuts and with the exact node dual. Time ratio: paired, "
                   "configuration / ours (> 1: slower than ours), bootstrap 95% CI stratified by cell. "
                   "SGM nodes over the instances solved by all five configurations.")

    # ---- K: the 2 x 2 effects, pooled over the three cells -------------------
    allc = [c["id"] for c in kcells]
    C = lambda rule, v="": FS.lr_config_id("R0", rule, variant=v)
    eff = [("fixing at the root only instead of every node | fractional reliability", C("rel"), C("rel", "rootfix")),
           ("fixing at the root only instead of every node | feasible-tree partition", C("ftree"), C("ftree", "rootfix")),
           ("feasible-tree partition instead of fractional reliability | fixing at every node", C("rel"), C("ftree")),
           ("feasible-tree partition instead of fractional random | fixing at every node", C("rfrac"), C("ftree")),
           ("feasible-tree partition instead of fractional reliability | fixing at the root", C("rel", "rootfix"), C("ftree", "rootfix")),
           ("classical scheme instead of ours", C("rel"), C("ftree", "rootfix"))]
    erows = []
    for lab, x, y in eff:
        r = PT.pooled(df, x, y, allc, a.boot, rng)
        if r:
            erows.append({"change (a → b)": lab, "instances": r["n"],
                          "node ratio b/a [95% CI]": PT.ratio_ci(r["node_ratio"], r["node_lo"], r["node_hi"]),
                          "time ratio b/a [95% CI]": PT.ratio_ci(r["time_ratio"], r["time_lo"], r["time_hi"]),
                          "solved a / b": f"{r['solved_a']} / {r['solved_b']}"})
    if erows:
        PT.write_table(out, "K_effects", "Classical baseline: effect of each component (all three cells)",
                       pd.DataFrame(erows), "Paired over the instances of the three cells; nodes on instances "
                       "solved by both; > 1: the change makes the search larger or slower.")

    # ---- GT: DCUT time inside the Python callback ----------------------------
    grows = []
    for c, idxs, _ in fam["GT"]:
        g = df[(df.cell_id == c["id"]) & (df.config_id == "GRB-DCUT")]
        if g.empty:
            continue
        rt = g.grb_runtime.astype(float).clip(lower=1e-9)
        share = g.callback_time.astype(float) / rt
        sep = g.sep_callback_time.astype(float) / rt
        # DCUT with its callback time removed (optimistic: a free callback),
        # against ours on the same instances.
        ours = df[(df.cell_id == c["id"]) & (df.config_id == OURS)].set_index("idx")
        gg = g.set_index("idx")
        ids = sorted(set(ours.index) & set(gg.index))
        free = (gg.loc[ids, "capped_time"].astype(float)
                - gg.loc[ids, "callback_time"].astype(float).fillna(0)).clip(lower=0)
        lt_now = np.log((gg.loc[ids, "capped_time"].astype(float) + AF.SHIFT_T)
                        / (ours.loc[ids, "capped_time"].astype(float) + AF.SHIFT_T))
        lt_free = np.log((free + AF.SHIFT_T) / (ours.loc[ids, "capped_time"].astype(float) + AF.SHIFT_T))
        grows.append({"instances": label.get(c["id"], c["id"]),
                      "DCUT solved": f"{int(g.solved.sum())}/{len(g)}",
                      "DCUT SGM time (s)": PT.fmt_num(AF.sgm(g.capped_time, AF.SHIFT_T)),
                      "callback share, median [IQR]": f"{PT.fmt_num(share.median())} "
                                                      f"[{PT.fmt_num(share.quantile(.25))}, {PT.fmt_num(share.quantile(.75))}]",
                      "separation share, median": PT.fmt_num(sep.median()),
                      "DCUT / ours": PT.fmt_num(float(np.exp(lt_now.mean()))) if ids else "–",
                      "DCUT without callback / ours": PT.fmt_num(float(np.exp(lt_free.mean()))) if ids else "–"})
    if grows:
        PT.write_table(out, "GT_callback", "Gurobi DCUT: time spent in the Python callback",
                       pd.DataFrame(grows),
                       "Shares of Gurobi's runtime spent inside the callback (all of it) and in its cut-set "
                       "separation (lazy and user cuts). Ratios: geometric means of paired time ratios "
                       "against ours (R0-rel-dw) on the same instances; 'without callback' subtracts the whole "
                       "callback time from DCUT's time, which is optimistic for DCUT.")

    for name in ("K_classical", "K_effects", "GT_callback"):
        p = os.path.join(out, f"{name}.md")
        if os.path.exists(p):
            print(open(p).read())
    print(f"tables -> {out}/K_classical.*, K_effects.*, GT_callback.*")


if __name__ == "__main__":
    sys.exit(main())
