#!/usr/bin/env python3
"""Pooled confirmation tests and paper-ready tables.

    python3 paper_tables.py --root ~/mstkp_final [--profile final] [--boot 10000]

Reads tables/*_results.csv only (run `final_suite.py collect` first) and
writes ROOT/paper/: every table as Markdown (.md), LaTeX booktabs (.tex) and
CSV, plus paper_summary.md with all of them.

Pooled tests (declared before looking at X's per-instance data): paired
geometric-mean ratios over ALL fresh instances of families X / XB, with a
bootstrap stratified by cell (instances resampled within their cell), exact
Wilcoxon on the pooled log-ratios, McNemar on solved/unsolved, Holm across the
declared pooled comparisons for each metric.  Reported overall and for the two
regimes fixed in advance: sparse / loose cells, and complete graphs (where no
cover cut is ever separated).
Same definitions as analyze_final.py: capped time (unsolved = time limit),
shift 1 s for time and 10 for nodes, node ratios on instances solved by both.
"""
import argparse
import glob
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_final as AF   # noqa: E402
import final_suite as FS     # noqa: E402

SEED = 20260923
C = lambda rung, rule="rel", variant="": FS.lr_config_id(rung, rule, "dw", variant) \
    if FS.RULES[rule][1] else FS.lr_config_id(rung, rule, variant=variant)
KNOB_RHO = {v: k for k, v in FS.KNOBS_D.items()}


# ----------------------------------------------------------------- helpers
def fmt_num(v, digits=3):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "–"
    if isinstance(v, (int, np.integer)) or (float(v).is_integer() and abs(v) < 1e7):
        return f"{int(v):,}" if abs(v) >= 10000 else str(int(v))
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    return f"{v:.{digits}g}"


def ratio_ci(r, lo, hi):
    return f"{fmt_num(r)} [{fmt_num(lo)}, {fmt_num(hi)}]" if not pd.isna(r) else "–"


def cell_label(cell):
    n, d, b, k = int(cell["n"]), float(cell["density"]), float(cell["beta"]), float(cell["knob"])
    s = f"n={n}, " + ("complete" if d >= 1.0 else f"deg {d * (n - 1):.0f}") + f", β={b:g}"
    if abs(k) > 1e-9:
        s += f", ρ={KNOB_RHO.get(round(k, 4), k):+g}"
    if str(cell.get("group", "")).startswith("confirm"):
        s += " (fresh)"
    return s


def write_table(out, name, title, df, note=""):
    df = df.copy()
    md = f"### {title}\n\n" + (f"{note}\n\n" if note else "")
    try:
        md += df.to_markdown(index=False)
    except ImportError:
        md += "```\n" + df.to_string(index=False) + "\n```"
    md += "\n"
    with open(os.path.join(out, f"{name}.md"), "w") as f:
        f.write(md)
    df.to_csv(os.path.join(out, f"{name}.csv"), index=False)
    esc = lambda x: (str(x).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
                     .replace("β", r"$\beta$").replace("ρ", r"$\rho$").replace("–", "--"))
    tex = [r"\begin{table}[htbp]", r"\centering", r"\small", rf"\caption{{{esc(title)}}}",
           rf"\label{{tab:{name}}}", r"\begin{tabular}{" + "l" * len(df.columns) + "}",
           r"\toprule", " & ".join(esc(c) for c in df.columns) + r" \\", r"\midrule"]
    tex += [" & ".join(esc(v) for v in row) + r" \\" for row in df.itertuples(index=False)]
    tex += [r"\bottomrule", r"\end{tabular}"]
    if note:
        tex.append(rf"\par\smallskip\footnotesize {esc(note)}")
    tex.append(r"\end{table}")
    with open(os.path.join(out, f"{name}.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")
    return md


def load(root, time_limit):
    paths = [p for p in sorted(glob.glob(os.path.join(root, "tables", "*_results.csv")))
             if "_all_results" not in os.path.basename(p)]
    if not paths:
        raise SystemExit("no tables: run `final_suite.py collect` first")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df.drop_duplicates(["cell_id", "config_id", "idx"])
    return AF.prep(df, time_limit)


def aligned(df, a, b, cells):
    """Per-instance pairs (a, b) over the given cells."""
    out = []
    for cid in cells:
        A = df[(df.cell_id == cid) & (df.config_id == a)].set_index("idx")
        B = df[(df.cell_id == cid) & (df.config_id == b)].set_index("idx")
        ids = sorted(set(A.index) & set(B.index))
        if not ids:
            continue
        out.append(pd.DataFrame({
            "cell": cid, "idx": ids,
            "ta": A.loc[ids, "capped_time"].values, "tb": B.loc[ids, "capped_time"].values,
            "na": A.loc[ids, "nodes"].astype(float).values,
            "nb": B.loc[ids, "nodes"].astype(float).values,
            "sa": A.loc[ids, "solved"].values, "sb": B.loc[ids, "solved"].values}))
    return pd.concat(out, ignore_index=True) if out else None


def strat_boot(logs, groups, B, rng):
    """Percentile CI of exp(mean(log)) with resampling inside each group."""
    logs, groups = np.asarray(logs, float), np.asarray(groups)
    if logs.size < 2:
        return float("nan"), float("nan")
    tot = np.zeros(B)
    for g in np.unique(groups):
        x = logs[groups == g]
        tot += x[rng.integers(0, x.size, size=(B, x.size))].sum(axis=1)
    means = tot / logs.size
    return float(np.exp(np.quantile(means, 0.025))), float(np.exp(np.quantile(means, 0.975)))


def pooled(df, a, b, cells, B, rng):
    P = aligned(df, a, b, cells)
    if P is None:
        return None
    lt = np.log((P.tb + AF.SHIFT_T) / (P.ta + AF.SHIFT_T))
    both = P.sa & P.sb
    ln = np.log((P.nb[both] + AF.SHIFT_N) / (P.na[both] + AF.SHIFT_N))
    tlo, thi = strat_boot(lt, P.cell, B, rng)
    nlo, nhi = strat_boot(ln, P.cell[both], B, rng) if both.sum() > 1 else (np.nan, np.nan)
    a_only, b_only = int((P.sa & ~P.sb).sum()), int((P.sb & ~P.sa).sum())
    return {
        "n": len(P), "cells": P.cell.nunique(), "solved_a": int(P.sa.sum()), "solved_b": int(P.sb.sum()),
        "time_ratio": float(np.exp(lt.mean())), "time_lo": tlo, "time_hi": thi,
        "time_p": AF.wilcoxon_p(lt),
        "node_ratio": float(np.exp(ln.mean())) if len(ln) else np.nan, "node_lo": nlo, "node_hi": nhi,
        "node_p": AF.wilcoxon_p(ln) if len(ln) else 1.0, "n_both": int(both.sum()),
        "mcnemar_p": (float(stats.binomtest(min(a_only, b_only), a_only + b_only, 0.5).pvalue)
                      if a_only + b_only else 1.0),
    }


# ----------------------------------------------------------------- tables
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--profile", default="final")
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    root = os.path.abspath(os.path.expanduser(a.root))
    TL = FS.PROFILES[a.profile]["time_limit"]
    out = os.path.join(root, "paper")
    os.makedirs(out, exist_ok=True)
    df = load(root, TL)
    fams = FS.build_families(a.profile, root)
    cellinfo = {c["id"]: c for blocks in fams.values() for c, _, _ in blocks}
    present = lambda cid, cfg: bool(((df.cell_id == cid) & (df.config_id == cfg)).any())
    rng = np.random.default_rng(SEED)
    sections = []

    # ---- P: pooled confirmation on the fresh instances (X / XB) ----------
    xcells = [c["id"] for c, _, _ in fams.get("X", [])]
    xcells = [c for c in xcells if (df.cell_id == c).any()]
    if xcells:
        regimes = {"all fresh cells": xcells,
                   "sparse / loose": [c for c in xcells if cellinfo[c]["density"] < 1.0],
                   "complete graphs": [c for c in xcells if cellinfo[c]["density"] >= 1.0]}
        declared = [
            ("your cuts vs literature cuts, same budget", C("R2", variant="it20"), C("R5", variant="it20")),
            ("cuts vs same effort spent on the dual (80 it./node)", C("R0", variant="it80"), C("R5", variant="it20")),
            ("literature cuts vs same effort on the dual", C("R0", variant="it80"), C("R2", variant="it20")),
            ("cuts on top of the fastest cut-free setting", C("R0", variant="it20"), C("R5", variant="it20")),
            ("larger budget for R5 as designed", C("R5"), C("R5", variant="it20")),
        ]
        rows = []
        for reg, cells in regimes.items():
            for label, x, y in declared:
                r = pooled(df, x, y, cells, a.boot, rng) if cells else None
                if r:
                    rows.append({"regime": reg, "comparison": label, "a": x, "b": y, **r})
        R = pd.DataFrame(rows)
        if len(R):
            for m in ("time_p", "node_p"):
                R[m + "_holm"] = np.nan
                for reg in R.regime.unique():
                    sel = R.regime == reg
                    R.loc[sel, m + "_holm"] = AF.holm(R.loc[sel, m].fillna(1.0))
            R.to_csv(os.path.join(out, "P_pooled_raw.csv"), index=False)
            T = pd.DataFrame({
                "regime": R.regime, "comparison (b vs a)": R.comparison,
                "a → b": R.a + " → " + R.b, "instances": R.n,
                "solved a / b": R.solved_a.astype(str) + " / " + R.solved_b.astype(str),
                "node ratio [95% CI]": [ratio_ci(*v) for v in zip(R.node_ratio, R.node_lo, R.node_hi)],
                "p (nodes, Holm)": [fmt_num(v) for v in R.node_p_holm],
                "time ratio [95% CI]": [ratio_ci(*v) for v in zip(R.time_ratio, R.time_lo, R.time_hi)],
                "p (time, Holm)": [fmt_num(v) for v in R.time_p_holm]})
            sections.append(write_table(
                out, "P_pooled_confirmation", "Pooled confirmation on fresh instances", T,
                "Ratio b/a of geometric means over paired instances (< 1: b better); nodes on "
                "instances solved by both; 95% CI from a bootstrap stratified by cell; Wilcoxon "
                "p-values, Holm-adjusted within each regime."))

        # ---- T6: dual budget curve ----------------------------------------
        curve = [(5, C("R0")), (10, C("R0", variant="it10")), (20, C("R0", variant="it20")),
                 (40, C("R0", variant="it40")), (80, C("R0", variant="it80"))]
        curve = [(k, c) for k, c in curve if (df.config_id == c).any()
                 and df[(df.config_id == c) & df.cell_id.isin(xcells)].shape[0]]
        if len(curve) >= 2:
            rows = []
            for reg, cells in regimes.items():
                if not cells:
                    continue
                sub = df[df.cell_id.isin(cells)]
                for k, c in curve:
                    g = sub[sub.config_id == c]
                    rows.append({"regime": reg, "iterations / node": k, "config": c,
                                 "solved": f"{int(g.solved.sum())}/{len(g)}",
                                 "memory stops": int(g.status.eq("memory").sum()),
                                 "SGM time (s)": fmt_num(AF.sgm(g.capped_time, AF.SHIFT_T)),
                                 "SGM nodes (solved)": fmt_num(AF.sgm(g.loc[g.solved, "nodes"], AF.SHIFT_N))})
                for (k1, c1), (k2, c2) in zip(curve[:-1], curve[1:]):
                    r = pooled(df, c1, c2, cells, a.boot, rng)
                    if r:
                        rows.append({"regime": reg, "iterations / node": f"{k1} → {k2}", "config": "",
                                     "solved": f"{r['solved_a']} → {r['solved_b']}",
                                     "memory stops": "",
                                     "SGM time (s)": "ratio " + ratio_ci(r["time_ratio"], r["time_lo"], r["time_hi"]),
                                     "SGM nodes (solved)": "ratio " + ratio_ci(r["node_ratio"], r["node_lo"], r["node_hi"])})
            sections.append(write_table(
                out, "T6_budget_curve", "Dual iterations per node (R0, no cuts) on the fresh instances",
                pd.DataFrame(rows), "Ratios: next budget / previous budget, paired, stratified bootstrap 95% CI."))

    # ---- T1: headline ladder --------------------------------------------
    head = next((c for c, _, _ in fams.get("A", []) if (df.cell_id == c["id"]).any()), None)
    if head is not None:
        ladder = [C(r) for r in FS.LADDER6] + [C("R0", variant="rit40"), C("R0", variant="it20")]
        ladder = [c for c in ladder if present(head["id"], c)]
        S = AF.summary(df[df.cell_id == head["id"]], ladder)
        T = pd.DataFrame({
            "configuration": S.config, "solved": S.solved.astype(str) + "/" + S.N.astype(str),
            "SGM time (s)": [fmt_num(v) for v in S.sgm_time],
            "SGM nodes": [fmt_num(v) for v in S.sgm_nodes_common],
            "root iterations": [fmt_num(v) for v in S.median_root_iters],
            "root gap (%)": [fmt_num(v) for v in S.median_root_gap_pct],
            "cuts (median)": [fmt_num(v) for v in S.median_cuts],
            "separation share": [fmt_num(v) for v in S.median_sep_share]})
        sections.append(write_table(out, "T1_headline", f"Headline cell: {cell_label(head)}", T,
                                    f"SGM over all instances (time) and commonly solved ones (nodes); "
                                    f"L* gap {fmt_num(float(S.median_lstar_gap_pct.iloc[0]))}%."))
        pairs = [(C("R0"), C("R1")), (C("R0", variant="rit40"), C("R1")), (C("R1"), C("R2")),
                 (C("R2"), C("R3")), (C("R3"), C("R4")), (C("R4"), C("R5")), (C("R2"), C("R5")),
                 (C("R0", variant="it20"), C("R2")), (C("R0", variant="it20"), C("R5"))]
        rows = []
        for x, y in pairs:
            if present(head["id"], x) and present(head["id"], y):
                r = AF.paired(df[df.cell_id == head["id"]], x, y, a.boot, rng)
                rows.append(r)
        if rows:
            P = pd.DataFrame(rows)
            P["node_p_holm"] = AF.holm(P.node_p.fillna(1.0))
            P["time_p_holm"] = AF.holm(P.time_p)
            T = pd.DataFrame({"a → b": P.a + " → " + P.b,
                              "node ratio [95% CI]": [ratio_ci(*v) for v in zip(P.node_ratio, P.node_ci_lo, P.node_ci_hi)],
                              "p (Holm)": [fmt_num(v) for v in P.node_p_holm],
                              "time ratio [95% CI]": [ratio_ci(*v) for v in zip(P.time_ratio, P.time_ci_lo, P.time_ci_hi)],
                              "p (Holm) ": [fmt_num(v) for v in P.time_p_holm],
                              "time wins / ties / losses of b": P.time_wins_b.astype(str) + " / "
                              + P.time_ties.astype(str) + " / " + P.time_losses_b.astype(str)})
            sections.append(write_table(out, "T1b_headline_pairs", "Headline cell: paired comparisons", T,
                                        "Ratio b/a (< 1: b better); Holm across the rows of this table."))

    # ---- T2: your cuts vs literature cuts in every cell ----------------------
    rows = []
    for cid, info in cellinfo.items():
        for lit, own, budget in ((C("R2"), C("R5"), 5), (C("R2", variant="it20"), C("R5", variant="it20"), 20)):
            if present(cid, lit) and present(cid, own):
                r = AF.paired(df[df.cell_id == cid], lit, own, min(a.boot, 5000), rng)
                cuts = df[(df.cell_id == cid) & (df.config_id == own)].cuts_separated.median()
                rows.append({"cell": cell_label(info), "budget": budget, "n": r["n"],
                             "both solved": r["n_both_solved"], "cuts (median)": cuts, **r})
    if rows:
        R = pd.DataFrame(rows).sort_values(["budget", "cell"])
        R["node_p_holm"] = AF.holm(R.node_p.fillna(1.0))
        T = pd.DataFrame({"cell": R.cell, "λ-iterations / node": R.budget, "instances": R.n,
                          "cuts (median, R5)": [fmt_num(v) for v in R["cuts (median)"]],
                          "node ratio R5/R2 [95% CI]": [ratio_ci(*v) for v in zip(R.node_ratio, R.node_ci_lo, R.node_ci_hi)],
                          "p (Holm)": [fmt_num(v) for v in R.node_p_holm],
                          "time ratio R5/R2 [95% CI]": [ratio_ci(*v) for v in zip(R.time_ratio, R.time_ci_lo, R.time_ci_hi)]})
        sections.append(write_table(out, "T2_strengthening", "Your strengthened cuts (R5) vs literature cuts (R2), every cell", T,
                                    "Paired, same dual budget; nodes on instances solved by both; cells with 0 cuts "
                                    "separated show no effect by construction."))

    # ---- T3: LR-BnB vs Gurobi -----------------------------------------------
    cfgs = [C("R5"), C("R0", variant="it20"), "GRB-SCF", "GRB-DMCF", "GRB-DCUT"]
    rows = []
    for cid, info in cellinfo.items():
        if not any(present(cid, g) for g in cfgs[2:]):
            continue
        row = {"cell": cell_label(info)}
        for c in cfgs:
            g = df[(df.cell_id == cid) & (df.config_id == c)]
            row[c] = (f"{int(g.solved.sum())}/{len(g)} ({fmt_num(AF.sgm(g.capped_time, AF.SHIFT_T))} s)"
                      if len(g) else "–")
        rows.append(row)
    if rows:
        sections.append(write_table(out, "T3_gurobi", "LR-BnB vs Gurobi: solved / instances (SGM time)",
                                    pd.DataFrame(rows), f"Time limit {TL:.0f} s; unsolved runs count at the limit."))

    # ---- T4: branching --------------------------------------------------------
    rows = []
    for c, idxs, cf in fams.get("F", []):
        if not (df.cell_id == c["id"]).any():
            continue
        S = AF.summary(df[df.cell_id == c["id"]], [x for x in cf if present(c["id"], x)])
        for _, s in S.iterrows():
            rows.append({"cell": cell_label(c), "rule": s.config, "solved": f"{s.solved}/{s.N}",
                         "SGM time (s)": fmt_num(s.sgm_time), "SGM nodes": fmt_num(s.sgm_nodes_common),
                         "probes / node": fmt_num(s.median_probes_per_node),
                         "empty-pool share": fmt_num(s.median_empty_pool_share)})
    if rows:
        sections.append(write_table(out, "T4_branching", "Branching rules (R5 cuts)", pd.DataFrame(rows),
                                    "Empty-pool share: decisions where the indicator had no strictly fractional "
                                    "free edge (the rule then falls back to the indicator's support)."))

    # ---- T5: components --------------------------------------------------------
    if head is not None:
        comps = [("exact cut dual off (R2)", C("R2"), C("R2", variant="noexact")),
                 ("exact cut dual off (R5)", C("R5"), C("R5", variant="noexact")),
                 ("reduced-cost fixing off (R0)", C("R0"), C("R0", variant="norc")),
                 ("reduced-cost fixing off (R5)", C("R5"), C("R5", variant="norc"))]
        rows = []
        for label, x, y in comps:
            if present(head["id"], x) and present(head["id"], y):
                r = AF.paired(df[df.cell_id == head["id"]], x, y, a.boot, rng)
                rows.append({"component switched off": label, "node ratio [95% CI]":
                             ratio_ci(r["node_ratio"], r["node_ci_lo"], r["node_ci_hi"]),
                             "time ratio [95% CI]": ratio_ci(r["time_ratio"], r["time_ci_lo"], r["time_ci_hi"])})
        if rows:
            sections.append(write_table(out, "T5_components", "Components (headline cell)", pd.DataFrame(rows),
                                        "Ratio off/on (> 1: the component helps)."))

    # ---- T7: outcomes per family ------------------------------------------------
    rows = []
    for f in list(FS.FAMILY_INFO):
        path = os.path.join(root, "tables", f"{f}_results.csv")
        if os.path.exists(path):
            g = pd.read_csv(path)
            vc = g.status.value_counts()
            rows.append({"family": f, "runs": len(g), "optimal": int(vc.get("optimal", 0)),
                         "timeout": int(vc.get("timeout", 0)), "memory": int(vc.get("memory", 0)),
                         "other": int(len(g) - vc.get("optimal", 0) - vc.get("timeout", 0) - vc.get("memory", 0))})
    if rows:
        sections.append(write_table(out, "T7_outcomes", "Run outcomes per family", pd.DataFrame(rows)))

    with open(os.path.join(out, "paper_summary.md"), "w") as f:
        f.write("# Paper tables\n\n" + "\n".join(sections))
    print(f"{len(sections)} tables -> {out}/  (paper_summary.md has all of them; .tex for LaTeX)")


if __name__ == "__main__":
    sys.exit(main())
