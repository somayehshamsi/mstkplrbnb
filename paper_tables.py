#!/usr/bin/env python3
"""Pooled confirmation tests and paper-ready tables.

    python3 paper_tables.py --root ~/mstkp_final [--profile final] [--boot 10000]

Reads tables/*_results.csv only (run `final_suite.py collect` first) and
writes ROOT/paper/: every table as Markdown (.md), LaTeX booktabs (.tex) and
CSV, plus paper_summary.md with all of them.  The .tex tables need the LaTeX packages booktabs and adjustbox (wide
tables are shrunk to the text width).

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
    # REVISION: keep trailing zeros (1.00, 0.950, 3.60) so every ratio and
    # p-value shows three significant figures.
    return f"{v:#.{digits}g}".rstrip(".")


def ratio_ci(r, lo, hi):
    return f"{fmt_num(r)} [{fmt_num(lo)}, {fmt_num(hi)}]" if not pd.isna(r) else "–"


GROUP_TAG = (("confirm", "X"), ("large_", "L"), ("wide_", "W"), ("scale_", "C"),
             ("complete_", "H"), ("grid_", "B"), ("calib", "CAL"), ("core", ""))


def cell_label(cell):
    n, d, b, k = int(cell["n"]), float(cell["density"]), float(cell["beta"]), float(cell["knob"])
    g = str(cell.get("group", ""))
    tag = next((t for pre, t in GROUP_TAG if g.startswith(pre)), "")
    s = (f"[{tag}] " if tag and tag != "X" else "") + f"n={n}, " + \
        ("complete" if d >= 1.0 else f"deg {d * (n - 1):.0f}") + f", β={b:g}"
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
    symbols = {"β": r"$\beta$", "ρ": r"$\rho$", "λ": r"$\lambda$", "τ": r"$\tau$",
               "→": r"$\to$", "≈": r"$\approx$", "×": r"$\times$", "≥": r"$\geq$",
               "≤": r"$\leq$", "±": r"$\pm$", "–": "--", "<": r"$<$", ">": r"$>$"}

    def esc(x):
        x = str(x).replace("\\", r"\textbackslash{}")
        for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
            x = x.replace(a, b)
        # Brackets in braces: a row starting with "[" (e.g. "[L] n=1000") would
        # otherwise be read as the optional argument of the previous "\\".
        x = x.replace("[", "{[}").replace("]", "{]}")
        for a, b in symbols.items():
            x = x.replace(a, b)
        return x
    # adjustbox shrinks a table that is wider than the text, never enlarges it
    tex = [r"\begin{table}[htbp]", r"\centering", r"\small", rf"\caption{{{esc(title)}}}",
           rf"\label{{tab:{name}}}", r"\begin{adjustbox}{max width=\linewidth}",
           r"\begin{tabular}{" + "l" * len(df.columns) + "}",
           r"\toprule", " & ".join(esc(c) for c in df.columns) + r" \\", r"\midrule"]
    tex += [" & ".join(esc(v) for v in row) + r" \\" for row in df.itertuples(index=False)]
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"]
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


# ----------------------------------------------------------------- figures
def make_figures(df, fams, cellinfo, present, out, TL, plt):
    made = []
    # REVISION: exact plain dual; no titles inside the figures (the captions
    # carry them); legend placed clear of the curves.
    style = {C("R0"): ("R0 no cuts", "tab:green", "-"), C("R2"): ("R2 literature cuts", "tab:blue", "-"),
             C("R5"): ("R5 strengthened cuts", "tab:red", "-"),
             C("R0", variant="subgr"): ("R0, subgradient dual", "tab:gray", "-"),
             "GRB-SCF": ("Gurobi SCF", "tab:purple", "--"), "GRB-DMCF": ("Gurobi DMCF", "tab:brown", "--"),
             "GRB-DCUT": ("Gurobi DCUT", "black", "--")}

    # F1: performance profile over all cells with a Gurobi comparison
    cfgs = [C("R0"), C("R2"), C("R5"), "GRB-SCF", "GRB-DCUT"]
    cells = [cid for cid in cellinfo if all(present(cid, c) for c in cfgs)]
    if cells:
        sub = df[df.cell_id.isin(cells) & df.config_id.isin(cfgs)]
        T = sub.pivot_table(index=["cell_id", "idx"], columns="config_id", values="wall_time", aggfunc="first")
        S = sub.pivot_table(index=["cell_id", "idx"], columns="config_id", values="solved", aggfunc="first")
        T = T[cfgs].where(S[cfgs].astype(bool), np.inf).dropna(how="any")
        best = T.min(axis=1)
        T = T[np.isfinite(best)]
        best = best[np.isfinite(best)].clip(lower=1e-3)
        taus = np.logspace(0, np.log10(TL), 300)
        fig, ax = plt.subplots(figsize=(6, 4))
        for c in cfgs:
            r = (T[c].clip(lower=1e-3) / best).values
            lab, col, ls = style[c]
            ax.step(taus, [(r <= t).mean() for t in taus], where="post", label=lab, color=col, ls=ls)
        ax.set_xscale("log"); ax.set_xlabel("time ratio to the best configuration (τ)")
        ax.set_ylabel("share of instances"); ax.set_ylim(0, 1.02)
        print(f"F1 performance profile: {len(T)} instances, {len(cells)} cells; solved: "
              + ", ".join(f"{c} {int(np.isfinite(T[c]).sum())}" for c in cfgs))
        ax.legend(fontsize=8, loc="center right"); ax.grid(alpha=.3)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out, f"F1_performance_profile.{ext}"), bbox_inches="tight", dpi=200)
        plt.close(fig)
        made.append("F1_performance_profile.pdf/png")

    # F2: scaling at constant average degree (families C and L)
    pts = []
    for f in ("C", "L"):
        for c, idxs, _ in fams.get(f, []):
            if (df.cell_id == c["id"]).any():
                pts.append((c, idxs))
    if pts:
        pts = sorted({c["n"]: (c, i) for c, i in pts}.values(), key=lambda ci: ci[0]["n"])
        fig, ax = plt.subplots(figsize=(6, 4))
        plotted = [C("R0"), C("R2"), C("R5"), "GRB-SCF", "GRB-DCUT"]
        # each point: the family's own instances that every plotted configuration has
        subs = {c["id"]: common_rows(df, c["id"], [x for x in plotted if present(c["id"], x)], i)
                for c, i in pts}
        pts = [c for c, _ in pts]
        for cfg in plotted:
            xs, ys = [], []
            for c in pts:
                g = subs[c["id"]][subs[c["id"]].config_id == cfg]
                if len(g):
                    xs.append(c["n"]); ys.append(AF.sgm(g.capped_time, AF.SHIFT_T))
            if xs:
                lab, col, ls = style[cfg]
                ax.plot(xs, ys, marker="o", label=lab, color=col, ls=ls)
        ax.axhline(TL, color="k", lw=.6, ls=":"); ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("number of vertices n (average degree ≈ 15)"); ax.set_ylabel("SGM time (s)")
        from matplotlib.ticker import FixedLocator, NullFormatter
        ns = [c["n"] for c in pts]
        ax.xaxis.set_major_locator(FixedLocator(ns))
        ax.set_xticklabels([str(v) for v in ns])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out, f"F2_scaling.{ext}"), bbox_inches="tight", dpi=200)
        plt.close(fig)
        made.append("F2_scaling.pdf/png")

    # REVISION: F3 (dual budget curve, R0 with 5-80 iterations) is gone --
    # with the exact plain dual R0 has no iteration budget left to vary.
    return made


def common_rows(df, cid, cfgs, idxs=None):
    """Rows of cell `cid` for `cfgs`, restricted to the instances that every
    listed configuration has (and to `idxs` if given), so that all numbers in
    one table row describe the same instances."""
    sub = df[(df.cell_id == cid) & df.config_id.isin(cfgs)]
    if idxs is not None:
        sub = sub[sub.idx.isin(set(idxs))]
    have = [set(sub[sub.config_id == c].idx) for c in cfgs if (sub.config_id == c).any()]
    if not have:
        return sub
    common = set.intersection(*have)
    return sub[sub.idx.isin(common)]


# ----------------------------------------------------------------- main-text tables
def find_cell(cellinfo, group, beta, knob=0.0):
    for c in cellinfo.values():
        if c["group"] == group and abs(c["beta"] - beta) < 1e-9 and abs(c["knob"] - knob) < 1e-6:
            return c
    return None


def solved_sgm(g):
    mem = int(g.status.eq("memory").sum())
    txt = f"{int(g.solved.sum())}/{len(g)} ({fmt_num(AF.sgm(g.capped_time, AF.SHIFT_T))} s)"
    return txt + (f", {mem} mem" if mem else "")


def main_text_tables(df, fams, cellinfo, present, out, a, rng):
    """Compact tables for the main text (the full tables go to the appendix)."""
    sec = []
    present_any = lambda c: c is not None and (df.cell_id == c["id"]).any()

    # M1: LR-BnB vs the MIP formulations on representative cells
    specs = [("headline: $n=300$, deg 15, β=0.15", "core", 0.15, 0.0),
             ("correlated: $n=300$, ρ=-0.9", "core", 0.15, -0.674),
             ("sparse: $n=800$, deg 15", "scale_n800", 0.15, 0.0),
             ("sparse: $n=2000$, deg 15", "large_n2000", 0.15, 0.0),
             ("sparse: $n=4000$, deg 15", "large_n4000", 0.15, 0.0),
             ("dense: $n=2000$, deg 150, β=0.3", "wide_n2000_deg150", 0.30, 0.0),
             ("complete: $n=500$, β=0.15", "complete_n500", 0.15, 0.0),
             ("complete: $n=500$, β=0.5", "complete_n500", 0.50, 0.0)]
    cfgs = [C("R0"), C("R5"), "GRB-SCF", "GRB-DMCF", "GRB-DCUT"]
    names = {C("R0"): "LR-BnB R0 (no cuts)", C("R5"): "LR-BnB R5 (cuts)",
             "GRB-SCF": "Gurobi SCF", "GRB-DMCF": "Gurobi DMCF", "GRB-DCUT": "Gurobi DCUT"}
    rows = []
    for label, grp, beta, knob in specs:
        c = find_cell(cellinfo, grp, beta, knob)
        if not present_any(c):
            continue
        sub = common_rows(df, c["id"], [x for x in cfgs if present(c["id"], x)])
        row = {"instances": label}
        for x in cfgs:
            g = sub[sub.config_id == x]
            row[names[x]] = solved_sgm(g) if len(g) else "–"
        rows.append(row)
    if rows:
        sec.append(write_table(out, "M1_mip", "LR-BnB and the MIP formulations: solved / instances (SGM time)",
                               pd.DataFrame(rows), "Representative cells; every entry of a row uses the same "
                               "instances.  Time limit 1800 s; 'mem' = runs stopped at the memory limit; DMCF was "
                               "not run where its model would exceed $10^7$ variables.  All cells: appendix."))

    # M2a: the dual and the cuts on the headline cell.  REVISION: the
    # iteration-matched controls are gone; the first study's subgradient dual
    # (variant subgr) is shown next to the exact one on the same instances.
    head = find_cell(cellinfo, "core", 0.15, 0.0)
    if present_any(head):
        order = [(C("R0", variant="subgr"), "R0, subgradient dual (first study)"),
                 (C("R0"), "R0: exact dual, no cuts"),
                 (C("R1"), "R1: literature cuts, root only"),
                 (C("R2"), "R2: literature cuts, node-local"),
                 (C("R5"), "R5: strengthened cuts"),
                 (C("R5", variant="subgr"), "R5, subgradient dual (first study)")]
        order = [(c, l) for c, l in order if present(head["id"], c)]
        H = common_rows(df, head["id"], [c for c, _ in order])
        S = AF.summary(H, [c for c, _ in order]).set_index("config")
        rows = []
        for c, l in order:
            g = H[H.config_id == c]
            work = g["mst_evaluations"] if "mst_evaluations" in g and g["mst_evaluations"].notna().any() \
                else g["lr_iterations"]
            ipn = float((work.astype(float) / g.nodes.clip(lower=1).astype(float)).median())
            rows.append({"configuration": l, "MSTs / node": fmt_num(ipn),
                         "root gap (%)": fmt_num(S.loc[c, "median_root_gap_pct"]),
                         "SGM nodes": fmt_num(S.loc[c, "sgm_nodes_common"]),
                         "SGM time (s)": fmt_num(S.loc[c, "sgm_time"])})
        sec.append(write_table(out, "M2a_dual", "Headline cell: node dual and cuts", pd.DataFrame(rows),
                               f"{len(H) // max(1, len(order))} instances per configuration; MSTs per node "
                               "include breakpoint steps, cut-phase iterations and strong-branching probes; "
                               "root gap relative to the optimum."))

    # M2b: the cells where the first study's five-iteration dual failed.
    specs = [("complete, $n=300$, β=0.5", "complete_n300", 0.50, 0.0),
             ("complete, $n=500$, β=0.5", "complete_n500", 0.50, 0.0),
             ("deg 150, $n=1000$, β=0.5", "wide_n1000_deg150", 0.50, 0.0),
             ("deg 150, $n=2000$, β=0.3", "wide_n2000_deg150", 0.30, 0.0),
             ("deg 60, $n=300$, β=0.7", "grid_d0.20", 0.70, 0.0),
             ("deg 15, $n=300$, ρ=-0.5", "core", 0.15, -0.366)]
    cfg8 = [C("R0"), C("R2"), C("R5")]
    rows = []
    for label, grp, beta, knob in specs:
        c = find_cell(cellinfo, grp, beta, knob)
        if not present_any(c) or not all(present(c["id"], x) for x in cfg8):
            continue
        sub = common_rows(df, c["id"], cfg8)
        rows.append({"instances": label, **{x: solved_sgm(sub[sub.config_id == x]) for x in cfg8}})
    if rows:
        sec.append(write_table(out, "M2b_regimes", "The hardest cells of the first study: solved / instances "
                               "(SGM time)", pd.DataFrame(rows), "Exact node dual; cut rungs add a cut phase when "
                               "covers are found.  All cells: appendix."))

    # M3: strengthened cuts -- headline steps and fresh pooled comparisons
    rows = []
    if present_any(head):
        Hd = df[df.cell_id == head["id"]]
        for label, x, y in [("unit lifting (R2 → R3)", C("R2"), C("R3")),
                            ("tree-completion lifting (R3 → R4)", C("R3"), C("R4")),
                            ("rank lifting (R4 → R5)", C("R4"), C("R5")),
                            ("all strengthening (R2 → R5)", C("R2"), C("R5")),
                            ("cuts vs none (R0 → R5)", C("R0"), C("R5"))]:
            if present(head["id"], x) and present(head["id"], y):
                r = AF.paired(Hd, x, y, a.boot, rng)
                rows.append({"setting": "headline", "comparison": label, "n": r["n"],
                             "nr": r["node_ratio"], "nlo": r["node_ci_lo"], "nhi": r["node_ci_hi"], "np": r["node_p"],
                             "tr": r["time_ratio"], "tlo": r["time_ci_lo"], "thi": r["time_ci_hi"], "tp": r["time_p"]})
    xcells = [c["id"] for c, _, _ in fams.get("X", []) if (df.cell_id == c["id"]).any()]
    if xcells:
        for label, x, y in [("strengthened vs literature (R2 → R5)", C("R2"), C("R5")),
                            ("cuts vs none (R0 → R5)", C("R0"), C("R5")),
                            ("cut phase 2× instead of 3× (R5 → R5-cp2)", C("R5"), C("R5", variant="cp2")),
                            ("cut phase 1× instead of 3× (R5 → R5-cp1)", C("R5"), C("R5", variant="cp1"))]:
            r = pooled(df, x, y, xcells, a.boot, rng)
            if r:
                rows.append({"setting": "fresh", "comparison": label, "n": r["n"],
                             "nr": r["node_ratio"], "nlo": r["node_lo"], "nhi": r["node_hi"], "np": r["node_p"],
                             "tr": r["time_ratio"], "tlo": r["time_lo"], "thi": r["time_hi"], "tp": r["time_p"]})
    if rows:
        R = pd.DataFrame(rows)
        R["np_h"], R["tp_h"] = AF.holm(R.np.fillna(1.0)), AF.holm(R.tp.fillna(1.0))
        T = pd.DataFrame({"setting": R.setting, "comparison (a → b)": R.comparison, "instances": R.n,
                          "node ratio [95% CI]": [ratio_ci(*v) for v in zip(R.nr, R.nlo, R.nhi)],
                          "p": [fmt_num(v) for v in R.np_h],
                          "time ratio [95% CI]": [ratio_ci(*v) for v in zip(R.tr, R.tlo, R.thi)],
                          "p ": [fmt_num(v) for v in R.tp_h]})
        sec.append(write_table(out, "M3_cuts", "Strengthened cover cuts: node and time ratios", T,
                               "Ratio b/a of geometric means over paired instances (< 1: b better); fresh rows pooled "
                               "over the confirmation instances (bootstrap stratified by cell); Wilcoxon p, Holm "
                               "across the rows of this table."))

    # M4: branching, pooled over the three cells of family F, on each rung.
    # REVISION: rungs R0 and R5, DW indicator only.
    fcells = [c["id"] for c, _, _ in fams.get("F", []) if (df.cell_id == c["id"]).any()]
    rules = [("Random (MST)", "rmst"), ("Strong branching (MST)", "sbmst"), ("Random (frac.)", "rfrac"),
             ("Most fractional", "mf"), ("Pseudo-cost", "pc"), ("Strong branching (frac.)", "sbf"),
             ("Reliability", "rel"), ("Hybrid", "hyb")]
    for rung in FS.BRANCH_RUNGS:
        ref = FS.lr_config_id(rung, "rmst")
        if not (fcells and any(present(c, ref) for c in fcells)):
            continue
        rows = []
        for name, tag in rules:
            cid_cfg = FS.lr_config_id(rung, tag, "dw") if FS.RULES[tag][1] else FS.lr_config_id(rung, tag)
            row = {"rule": name}
            if cid_cfg == ref:
                row["nodes"] = row["time"] = "1 (reference)"
            else:
                r = pooled(df, ref, cid_cfg, fcells, a.boot, rng)
                row["nodes"] = ratio_ci(r["node_ratio"], r["node_lo"], r["node_hi"]) if r else "–"
                row["time"] = ratio_ci(r["time_ratio"], r["time_lo"], r["time_hi"]) if r else "–"
            g = df[df.cell_id.isin(fcells) & (df.config_id == cid_cfg)]
            row["probes / node"] = fmt_num(float((g.probes.astype(float) / g.nodes.clip(lower=1)).median()))
            if "pool_empty_share" in g:
                row["no fractional edge (share of decisions)"] = fmt_num(float(g.pool_empty_share.median()))
            rows.append(row)
        # REVISION: the cells and the instance count are read from the data
        # (F1 / F2 come from select-beta), not written into the note.
        _fi = [c for c, _, _ in fams.get("F", []) if c["id"] in fcells]
        _rho = {round(v, 3): k for k, v in FS.KNOBS_D.items()}
        _desc = "; ".join("headline" if (abs(c["beta"] - 0.15) < 1e-9 and abs(c["knob"]) < 1e-9)
                          else (f"β={c['beta']:g}" + (f" with ρ={_rho.get(round(c['knob'], 3), c['knob']):g}"
                                                      if abs(c["knob"]) > 1e-9 else ""))
                          for c in _fi)
        _ninst = int(len(df[df.cell_id.isin(fcells) & (df.config_id == ref)]))
        sec.append(write_table(out, f"M4_branching_{rung}",
                               f"Branching rules ({rung}): ratios to Random (MST)",
                               pd.DataFrame(rows), f"Pooled over the {len(_fi)} branching cells ({_desc}; "
                               f"{_ninst} instances), paired ratios to Random (MST) with bootstrap 95% CI "
                               "stratified by cell; < 1: fewer nodes / less time.  Dantzig-Wolfe indicator.  "
                               "Per-cell results: appendix."))
    return sec


def forest_figure(R, out, plt):
    R = R[R.node_ratio.notna()].copy()
    if R.empty:
        return False

    def group(label):
        if "(fresh)" in label:
            return "fresh instances, 20 it./node"
        for tag, name in (("[H]", "complete graphs"), ("[W]", "size×density×budget grid"),
                          ("[L]", "large sparse"), ("[C]", "scaling"), ("[B]", "density×budget grid")):
            if label.startswith(tag):
                return name
        return "$n=300$, degree 15"
    R["group"] = R.cell.map(group)
    R = R.sort_values("node_ratio").reset_index(drop=True)
    colors = dict(zip(sorted(R.group.unique()), plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    fig, ax = plt.subplots(figsize=(6.2, 1.2 + 0.11 * len(R)))
    for grp, g in R.groupby("group"):
        lo = (g.node_ratio - g.node_ci_lo).clip(lower=0).fillna(0)
        hi = (g.node_ci_hi - g.node_ratio).clip(lower=0).fillna(0)
        ax.errorbar(g.node_ratio, g.index, xerr=[lo, hi], fmt="o", ms=3.5, lw=0.8, capsize=0,
                    color=colors[grp], label=grp)
    ax.axvline(1.0, color="k", lw=0.8, ls="--")
    below = int((R.node_ratio < 1).sum())
    ax.set_yticks([]); ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator, FormatStrFormatter
    lo_x = max(0.3, float(np.nanmin(R.node_ci_lo.fillna(R.node_ratio))) * 0.95)
    hi_x = min(3.0, float(np.nanmax(R.node_ci_hi.fillna(R.node_ratio))) * 1.05)
    ticks = [t for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0) if lo_x <= t <= hi_x]
    ax.set_xlim(lo_x, hi_x)
    ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.set_xlabel("node ratio, strengthened (R5) / literature (R2) cuts")
    print(f"F4: {below} of {len(R)} cells with a defined ratio below 1")
    ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=.3, axis="x")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"F4_strengthening_cells.{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    return True


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
        # REVISION: comparisons declared for the new fresh set
        declared = [
            ("strengthened vs literature cuts", C("R2"), C("R5")),
            ("literature cuts vs none", C("R0"), C("R2")),
            ("strengthened cuts vs none", C("R0"), C("R5")),
            ("cut phase 2x instead of 3x", C("R5"), C("R5", variant="cp2")),
            ("cut phase 1x instead of 3x", C("R5"), C("R5", variant="cp1")),
            ("LR-BnB R0 vs Gurobi DCUT", "GRB-DCUT", C("R0")),
            ("LR-BnB R5 vs Gurobi DCUT", "GRB-DCUT", C("R5")),
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

        # ---- T6: cut-phase length (REVISION: replaces the dual budget curve) --
        curve = [("1x", C("R5", variant="cp1")), ("2x", C("R5", variant="cp2")), ("3x", C("R5"))]
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
                    rows.append({"regime": reg, "cut phase (x max_iter)": k, "config": c,
                                 "solved": f"{int(g.solved.sum())}/{len(g)}",
                                 "memory stops": int(g.status.eq("memory").sum()),
                                 "SGM time (s)": fmt_num(AF.sgm(g.capped_time, AF.SHIFT_T)),
                                 "SGM nodes (solved)": fmt_num(AF.sgm(g.loc[g.solved, "nodes"], AF.SHIFT_N))})
                for (k1, c1), (k2, c2) in zip(curve[:-1], curve[1:]):
                    r = pooled(df, c1, c2, cells, a.boot, rng)
                    if r:
                        rows.append({"regime": reg, "cut phase (x max_iter)": f"{k1} → {k2}", "config": "",
                                     "solved": f"{r['solved_a']} → {r['solved_b']}",
                                     "memory stops": "",
                                     "SGM time (s)": "ratio " + ratio_ci(r["time_ratio"], r["time_lo"], r["time_hi"]),
                                     "SGM nodes (solved)": "ratio " + ratio_ci(r["node_ratio"], r["node_lo"], r["node_hi"])})
            sections.append(write_table(
                out, "T6_cut_phase", "Length of the cut phase (R5) on the fresh instances",
                pd.DataFrame(rows), "Cut phase of 1, 2 or 3 times max_iter (5) iterations per node; ratios: "
                "next setting / previous setting, paired, stratified bootstrap 95% CI."))

    # ---- T1: headline ladder --------------------------------------------
    head = next((c for c, _, _ in fams.get("A", []) if (df.cell_id == c["id"]).any()), None)
    if head is not None:
        ladder = [C(r) for r in FS.LADDER6] + [C("R0", variant="subgr"), C("R5", variant="subgr")]
        ladder = [c for c in ladder if present(head["id"], c)]
        S = AF.summary(df[df.cell_id == head["id"]], ladder)
        Hc = df[df.cell_id == head["id"]]
        _work = "mst_evaluations" if "mst_evaluations" in Hc else "lr_iterations"
        ipn = {c: float((Hc[Hc.config_id == c][_work].astype(float)
                         / Hc[Hc.config_id == c].nodes.clip(lower=1).astype(float)).median())
               for c in ladder}
        T = pd.DataFrame({
            "configuration": S.config, "solved": S.solved.astype(str) + "/" + S.N.astype(str),
            "SGM time (s)": [fmt_num(v) for v in S.sgm_time],
            "SGM nodes": [fmt_num(v) for v in S.sgm_nodes_common],
            "MSTs / node": [fmt_num(ipn[c]) for c in S.config],
            "root gap (%)": [fmt_num(v) for v in S.median_root_gap_pct],
            "cuts (median)": [fmt_num(v) for v in S.median_cuts],
            "separation share": [fmt_num(v) for v in S.median_sep_share]})
        sections.append(write_table(out, "T1_headline", f"Headline cell: {cell_label(head)}", T,
                                    f"SGM over all instances (time) and commonly solved ones (nodes); MSTs per "
                                    f"node: median over runs, breakpoint steps, cut-phase iterations and "
                                    f"strong-branching probes included; "
                                    f"L* gap {fmt_num(float(S.median_lstar_gap_pct.iloc[0]))}%."))
        pairs = [(C("R0", variant="subgr"), C("R0")), (C("R5", variant="subgr"), C("R5")),
                 (C("R0"), C("R1")), (C("R1"), C("R2")),
                 (C("R2"), C("R3")), (C("R3"), C("R4")), (C("R4"), C("R5")), (C("R2"), C("R5")),
                 (C("R0"), C("R2")), (C("R0"), C("R5"))]
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
    T2R = None
    rows = []
    for cid, info in cellinfo.items():
        for lit, own, budget in ((C("R2"), C("R5"), 5),):
            if present(cid, lit) and present(cid, own):
                r = AF.paired(df[df.cell_id == cid], lit, own, min(a.boot, 5000), rng)
                cuts = df[(df.cell_id == cid) & (df.config_id == own)].cuts_separated.median()
                rows.append({"cell": cell_label(info), "budget": budget, "n": r["n"],
                             "both solved": r["n_both_solved"], "cuts (median)": cuts, **r})
    if rows:
        R = pd.DataFrame(rows).sort_values(["budget", "cell"])
        R["node_p_holm"] = AF.holm(R.node_p.fillna(1.0))
        T2R = R.copy()
        T = pd.DataFrame({"cell": R.cell, "instances": R.n,
                          "cuts (median, R5)": [fmt_num(v) for v in R["cuts (median)"]],
                          "node ratio R5/R2 [95% CI]": [ratio_ci(*v) for v in zip(R.node_ratio, R.node_ci_lo, R.node_ci_hi)],
                          "p (Holm)": [fmt_num(v) for v in R.node_p_holm],
                          "time ratio R5/R2 [95% CI]": [ratio_ci(*v) for v in zip(R.time_ratio, R.time_ci_lo, R.time_ci_hi)]})
        sections.append(write_table(out, "T2_strengthening", "Strengthened cuts (R5) vs literature cuts (R2), every cell", T,
                                    "Paired; nodes on instances solved by both; cells with 0 cuts "
                                    "separated show no effect by construction."))

    # ---- T3: LR-BnB vs Gurobi -----------------------------------------------
    cfgs = [C("R0"), C("R5"), "GRB-SCF", "GRB-DMCF", "GRB-DCUT"]
    rows = []
    for cid, info in cellinfo.items():
        if not any(present(cid, g) for g in cfgs[2:]):
            continue
        row = {"cell": cell_label(info)}
        sub3 = common_rows(df, cid, [c for c in cfgs if present(cid, c)])
        for c in cfgs:
            g = sub3[sub3.config_id == c]
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
        cf_here = [x for x in cf if present(c["id"], x)]
        S = AF.summary(common_rows(df, c["id"], cf_here, idxs), cf_here)
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
        sections.append(write_table(out, "T7_outcomes", "Run outcomes per family", pd.DataFrame(rows),
                                    "A run shared by several families (same instance and configuration) "
                                    "is counted in each of them, so the rows sum to more than the number of "
                                    "unique runs."))

    # ---- T8: dual budget / robustness in every core cell ---------------------
    order = ["A", "B", "BL", "D", "C", "H", "L", "W"]
    cfg8 = [C("R0"), C("R2"), C("R5")]
    seen, rows = set(), []
    for f in order:
        for c, _, cf in fams.get(f, []):
            cid = c["id"]
            if cid in seen or not all(present(cid, x) for x in cfg8):
                continue
            seen.add(cid)
            row = {"family": f, "cell": cell_label(c)}
            sub8 = common_rows(df, cid, cfg8)
            for x in cfg8:
                g = sub8[sub8.config_id == x]
                mem = int(g.status.eq("memory").sum())
                row[x] = (f"{int(g.solved.sum())}/{len(g)} ({fmt_num(AF.sgm(g.capped_time, AF.SHIFT_T))} s)"
                          + (f", {mem} mem" if mem else ""))
            rows.append(row)
    if rows:
        sections.append(write_table(
            out, "T8_regimes", "Dual budget and robustness: solved / instances (SGM time) in every cell",
            pd.DataFrame(rows), "Exact node dual; cut rungs add a cut phase when covers are found.  "
            "'mem' = runs stopped at the memory limit."))

    # ---- T9: channels and interactions (headline cell) ------------------------
    if head is not None:
        H = df[df.cell_id == head["id"]]
        specs = [("probe channel: R0→R5 under reliability vs most-fractional",
                  (C("R0"), C("R5")), (C("R0", "mf"), C("R5", "mf"))),
                 ("primal channel: R0→R5 as run vs with UB = z* from the start",
                  (C("R0"), C("R5")), (C("R0", variant="cutoff"), C("R5", variant="cutoff"))),
                 ("exact cut dual: its effect at R5 vs at R2",
                  (C("R5", variant="noexact"), C("R5")), (C("R2", variant="noexact"), C("R2"))),
                 ("RC fixing: its effect at R5 vs at R0",
                  (C("R5", variant="norc"), C("R5")), (C("R0", variant="norc"), C("R0")))]
        rows = []
        for label, (a1, b1), (a2, b2) in specs:
            if not all(present(head["id"], x) for x in (a1, b1, a2, b2)):
                continue
            for metric, mname in (("nodes", "nodes"), ("capped_time", "time")):
                r = AF.interaction(H, a1, b1, H, a2, b2, metric, a.boot, rng)
                if r.get("n"):
                    rows.append({"contrast": label, "metric": mname, "instances": r["n"],
                                 "ratio of ratios [95% CI]": ratio_ci(r["ratio_of_ratios"], r["ci_lo"], r["ci_hi"]),
                                 "p": fmt_num(r["p"])})
        if rows:
            sections.append(write_table(out, "T9_channels", "Channels and interactions (headline cell)",
                                        pd.DataFrame(rows), "1 = no interaction; < 1: the first ratio is the "
                                        "stronger improvement."))

    # ---- T10: remaining gaps of unsolved runs --------------------------------------
    rows = []
    for f in list(FS.FAMILY_INFO):
        path = os.path.join(root, "tables", f"{f}_results.csv")
        if not os.path.exists(path):
            continue
        g = pd.read_csv(path)
        un = g[g.status.isin(["timeout", "memory"])].copy()
        if un.empty:
            continue
        un["gap_pct"] = 100.0 * (un.obj - un.final_lb) / un.obj
        # A gap of 100% or more means the lower bound was <= 0 or missing:
        # no usable bound, so such runs are counted rather than averaged.
        un["useful"] = un.final_lb.notna() & (un.final_lb > 0) & (un.gap_pct < 100.0)
        for cfg, u in un.groupby("config_id"):
            ok = u[u.useful]
            rows.append({"family": f, "configuration": cfg, "unsolved": len(u),
                         "of which memory": int(u.status.eq("memory").sum()),
                         "no usable bound": int((~u.useful).sum()),
                         "median final gap (%)": fmt_num(ok.gap_pct.median()) if len(ok) else "no bound",
                         "max final gap (%)": fmt_num(ok.gap_pct.max()) if len(ok) else "no bound"})
    if rows:
        sections.append(write_table(out, "T10_unsolved_gaps", "Unsolved runs: remaining optimality gap",
                                    pd.DataFrame(rows), "Gap = 100 (incumbent - lower bound) / incumbent at "
                                    "the time or memory limit, over runs with a usable (positive) lower bound; "
                                    "runs without one are counted separately."))

    # ---- facts: calibration and validation ----------------------------------------
    facts = []
    fb = os.path.join(root, "frozen", "F_beta.json")
    if os.path.exists(fb):
        import json
        j = json.load(open(fb))
        for k in ("F1", "F2"):
            if k in j:
                facts.append(f"- Calibration {k}: beta = {j[k]['beta']} (knob {j[k]['knob']}); rule: "
                             f"{j[k]['reason']}.")
    vr = os.path.join(root, "validation_report.json")
    if os.path.exists(vr):
        import json
        v = json.load(open(vr))
        cnt = v.get("counts", {})
        facts.append(f"- Validation (brute force, n = 7-8, {len(v.get('instances', []))} instances): "
                     f"{cnt.get('lr_runs', 0)} LR-BnB runs and {cnt.get('grb_runs', 0)} Gurobi runs agree "
                     f"with the optimum; {cnt.get('cuts_checked', 0)} generated cuts "
                     f"({cnt.get('probe_cut_events', 0)} separation events inside probes) checked against "
                     f"{cnt.get('tree_checks', 0)} feasible trees; failures: {len(v.get('failures', []))}.")
    if facts:
        sections.append("### Facts for the text\n\n" + "\n".join(facts) + "\n")

    # ---- main-text tables (M1-M4) and the cell-by-cell strengthening figure -------
    sections += main_text_tables(df, fams, cellinfo, present, out, a, rng)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if T2R is not None and forest_figure(T2R, out, plt):
            sections.append("### Figure F4\n\n- `F4_strengthening_cells.pdf/png`\n")
    except ImportError:
        pass

    # ---- figures -------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figs = make_figures(df, fams, cellinfo, present, out, TL, plt)
        if figs:
            sections.append("### Figures\n\n" + "\n".join(f"- `{x}`" for x in figs) + "\n")
    except ImportError:
        print("matplotlib not available: figures skipped")

    with open(os.path.join(out, "paper_summary.md"), "w") as f:
        f.write("# Paper tables\n\n" + "\n".join(sections))
    print(f"{len(sections)} tables -> {out}/  (paper_summary.md has all of them; .tex for LaTeX)")


if __name__ == "__main__":
    sys.exit(main())
