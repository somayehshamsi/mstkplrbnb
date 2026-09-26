#!/usr/bin/env python3
"""Pre-declared analysis of the frozen benchmark (frozen together with it).

    python final_suite.py --root ROOT collect --family core
    python analyze_final.py --root ROOT --family core

Reads tables/<family>_results.csv and writes analysis/<family>.md plus the
underlying CSVs.  Definitions (declared before any final run):

  capped time   wall time if solved, else the time limit (timeouts, memory
                kills and errors all count as unsolved at the limit)
  SGM           exp(mean(log(x + s))) - s; s = 1 s for time, 10 for nodes
                and LR iterations
  common-solved instances solved by every configuration of the comparison
                group in that cell; node and iteration statistics use only
                these
  paired ratio  geometric mean over instances of (b + s) / (a + s); 95%
                percentile bootstrap CI (10 000 resamples, seed 20260923);
                Wilcoxon signed-rank p, Holm-adjusted within the family and
                metric; exact McNemar p on solved/unsolved; time wins/ties/
                losses with a +-5% tie band; "non-inferior" = upper CI < 1.10
  root gap      100 * (z* - root bound) / z*  (%), z* the agreed optimum;
                reported next to 100 * (z* - L*) / z*, L* the exact plain
                Lagrangian bound, and the share of runs whose root bound lies
                below L* (the LR root is not a converged dual)
  interaction   ratio of paired ratios, e.g. [R5/R0 | reliability] /
                [R5/R0 | most-fractional] on the same instances
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
import final_suite as FS  # noqa: E402

SHIFT_T, SHIFT_N = 1.0, 10.0
RNG_SEED = 20260923


def sgm(x, s):
    x = np.asarray([v for v in x if v is not None and not pd.isna(v)], float)
    return float(np.exp(np.mean(np.log(x + s))) - s) if x.size else float("nan")


def prep(df, time_limit):
    df = df.copy()
    df["solved"] = df["status"].eq("optimal")
    df["capped_time"] = np.where(df["solved"], df["wall_time"].astype(float), time_limit)
    z = df["zstar"].astype(float)
    df["root_gap_pct"] = 100.0 * (z - df["root_lb"].astype(float)) / z
    df["lstar_gap_pct"] = 100.0 * (z - df["plain_lr_bound"].astype(float)) / z
    df["root_below_lstar"] = df["root_lb"].astype(float) < df["plain_lr_bound"].astype(float) - 1e-6
    for c in ("sep_time", "indicator_time", "probe_time"):
        if c in df:
            df[c.replace("_time", "_share")] = df[c].astype(float) / df["wall_time"].astype(float)
    if "probes" in df:
        df["probes_per_node"] = df["probes"].astype(float) / df["nodes"].clip(lower=1).astype(float)
    return df


def summary(df, cfgs):
    sub = df[df["config_id"].isin(cfgs)]
    solved_by_all = (sub.pivot_table(index="idx", columns="config_id", values="solved",
                                     aggfunc="first").reindex(columns=cfgs).fillna(False)
                     .all(axis=1))
    common = set(solved_by_all[solved_by_all].index)
    rows = []
    for c in cfgs:
        g = sub[sub["config_id"] == c]
        cs = g[g["idx"].isin(common)]
        rows.append({
            "config": c, "N": len(g), "solved": int(g["solved"].sum()),
            "timeouts": int(g["status"].eq("timeout").sum()),
            "memory": int(g["status"].eq("memory").sum()),
            "failures": int((~g["status"].isin(["optimal", "timeout", "memory"])).sum()),
            "sgm_time": sgm(g["capped_time"], SHIFT_T),
            "n_common": len(common),
            "sgm_nodes_common": sgm(cs["nodes"], SHIFT_N),
            "sgm_lr_iters_common": sgm(cs.get("lr_iterations", pd.Series(dtype=float)), SHIFT_N),
            "median_root_iters": float(g["root_lr_iterations"].median())
            if "root_lr_iterations" in g else np.nan,
            "share_root_below_Lstar": float(g.loc[g["root_lb"].notna(), "root_below_lstar"].mean())
            if g["root_lb"].notna().any() else np.nan,
            "median_root_gap_pct": float(g["root_gap_pct"].median()),
            "median_lstar_gap_pct": float(g["lstar_gap_pct"].median()),
            "median_cuts": float(cs["cuts_separated"].median()) if "cuts_separated" in cs else np.nan,
            "median_sep_share": float(g["sep_share"].median()) if "sep_share" in g else np.nan,
            "median_probe_share": float(g["probe_share"].median()) if "probe_share" in g else np.nan,
            "median_probes_per_node": float(g["probes_per_node"].median())
            if "probes_per_node" in g else np.nan,
            "median_pool": float(g["pool_median"].median()) if "pool_median" in g else np.nan,
            "median_singleton_share": float(g["pool_singleton_share"].median())
            if "pool_singleton_share" in g else np.nan,
            "median_empty_pool_share": float(g["pool_empty_share"].median())
            if "pool_empty_share" in g else np.nan,
        })
    return pd.DataFrame(rows)


def boot_ci(logs, B, rng):
    logs = np.asarray(logs, float)
    if logs.size < 2:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, logs.size, size=(B, logs.size))
    means = logs[idx].mean(axis=1)
    return float(np.exp(np.quantile(means, 0.025))), float(np.exp(np.quantile(means, 0.975)))


def wilcoxon_p(logs):
    logs = np.asarray(logs, float)
    if logs.size < 2 or np.allclose(logs, 0):
        return 1.0
    try:
        return float(stats.wilcoxon(logs).pvalue)
    except ValueError:
        return 1.0


def paired(df, a, b, B, rng):
    A = df[df["config_id"] == a].set_index("idx")
    Bf = df[df["config_id"] == b].set_index("idx")
    ids = sorted(set(A.index) & set(Bf.index))
    A, Bf = A.loc[ids], Bf.loc[ids]
    lt = np.log((Bf["capped_time"] + SHIFT_T) / (A["capped_time"] + SHIFT_T))
    both = A["solved"] & Bf["solved"]
    ln = np.log((Bf.loc[both, "nodes"].astype(float) + SHIFT_N)
                / (A.loc[both, "nodes"].astype(float) + SHIFT_N))
    ratio_t = (Bf["capped_time"] + SHIFT_T) / (A["capped_time"] + SHIFT_T)
    b_only = int((Bf["solved"] & ~A["solved"]).sum())
    a_only = int((A["solved"] & ~Bf["solved"]).sum())
    mc = (float(stats.binomtest(min(a_only, b_only), a_only + b_only, 0.5).pvalue)
          if a_only + b_only else 1.0)
    ci_t, ci_n = boot_ci(lt, B, rng), boot_ci(ln, B, rng)
    return {
        "a": a, "b": b, "n": len(ids), "n_both_solved": int(both.sum()),
        "time_ratio": float(np.exp(lt.mean())) if len(lt) else np.nan,
        "time_ci_lo": ci_t[0], "time_ci_hi": ci_t[1], "time_p": wilcoxon_p(lt),
        "node_ratio": float(np.exp(ln.mean())) if len(ln) else np.nan,
        "node_ci_lo": ci_n[0], "node_ci_hi": ci_n[1], "node_p": wilcoxon_p(ln),
        "time_wins_b": int((ratio_t < 0.95).sum()), "time_ties": int(((ratio_t >= 0.95)
                                                                   & (ratio_t <= 1.05)).sum()),
        "time_losses_b": int((ratio_t > 1.05).sum()),
        "solved_only_a": a_only, "solved_only_b": b_only, "mcnemar_p": mc,
        "time_noninferior_10pct": bool(ci_t[1] < 1.10),
    }


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def interaction(df1, a1, b1, df2, a2, b2, metric, B, rng):
    """GM[(b1/a1)] / GM[(b2/a2)] on shared instances (log difference, paired)."""
    s = SHIFT_T if metric == "capped_time" else SHIFT_N

    def logs(df, a, b):
        A = df[df["config_id"] == a].set_index("idx")
        Bf = df[df["config_id"] == b].set_index("idx")
        ids = set(A.index) & set(Bf.index)
        if metric != "capped_time":
            ids = {i for i in ids if A.loc[i, "solved"] and Bf.loc[i, "solved"]}
        return {i: math.log((float(Bf.loc[i, metric]) + s) / (float(A.loc[i, metric]) + s))
                for i in ids}
    l1, l2 = logs(df1, a1, b1), logs(df2, a2, b2)
    ids = sorted(set(l1) & set(l2))
    d = np.array([l1[i] - l2[i] for i in ids])
    if not ids:
        return {"n": 0}
    lo, hi = boot_ci(d, B, rng)
    return {"n": len(ids), "ratio_of_ratios": float(np.exp(d.mean())), "ci_lo": lo,
            "ci_hi": hi, "p": wilcoxon_p(d)}


def comparisons(family, cfgs):
    rel = lambda rs, rule="rel": [FS.lr_config_id(r, rule) for r in rs]
    if family in ("A", "G", "ACUT"):
        rule = {"A": "rel", "G": "mf", "ACUT": "rel"}[family]
        var = "cutoff" if family == "ACUT" else ""
        ids = [FS.lr_config_id(r, rule, variant=var) for r in FS.LADDER6]
        pairs = list(zip(ids[:-1], ids[1:])) + [(ids[0], ids[-1])]
        # equal-dual-effort reading of the ladder (controls present in A / G):
        # R1 against R0 with R1's root budget, R2 and R5 against R0 with their
        # node budget
        it20 = FS.lr_config_id("R0", rule, variant="it20")
        rit = FS.lr_config_id("R0", rule, variant="rit40")
        pairs += [(ids[0], rit), (rit, ids[1]), (ids[0], it20), (it20, ids[2]), (it20, ids[5])]
        return pairs
    if family == "E":
        return [(FS.lr_config_id("R2", "rel"), FS.lr_config_id("R2", "rel", variant="noexact")),
                (FS.lr_config_id("R5", "rel"), FS.lr_config_id("R5", "rel", variant="noexact")),
                (FS.lr_config_id("R0", "rel"), FS.lr_config_id("R0", "rel", variant="norc")),
                (FS.lr_config_id("R5", "rel"), FS.lr_config_id("R5", "rel", variant="norc"))]
    if family in ("B", "BL", "D", "C", "H", "L", "W"):
        lad = [c for c in rel(FS.LADDER5) if c in cfgs]
        pairs = list(zip(lad[:-1], lad[1:])) + [(lad[0], lad[-1])]
        it20 = FS.lr_config_id("R0", "rel", variant="it20")
        pairs += [(lad[0], it20)] + [(it20, c) for c in lad[1:]]
        pairs += [(g, FS.lr_config_id("R5", "rel")) for g in cfgs if g.startswith("GRB-")]
        return pairs
    if family == "F":
        t = lambda r, s="dw": FS.lr_config_id("R5", r, s)
        pairs = [(FS.lr_config_id("R5", "rmst"), t("rfrac")), (FS.lr_config_id("R5", "rmst"),
                 t("rfrac", "avg")), (t("rfrac"), t("rfrac", "avg")),
                 (FS.lr_config_id("R5", "sbmst"), t("sbf")),
                 (FS.lr_config_id("R5", "sbmst"), t("sbf", "avg"))]
        pairs += [(t(r), t(r, "avg")) for r in ("mf", "pc", "sbf", "rel", "hyb")]
        pairs += [(t("rel"), t(r)) for r in ("mf", "pc", "sbf", "hyb")]
        return pairs
    if family == "X":
        c = lambda r, v: FS.lr_config_id(r, "rel", variant=v)
        return [(FS.lr_config_id("R5", "rel"), c("R5", "it20")),   # does the budget fix R5
                (c("R0", "it20"), c("R0", "it40")), (c("R0", "it40"), c("R0", "it80")),
                (c("R0", "it20"), c("R5", "it20")),                # cuts on top of the budget
                (c("R0", "it80"), c("R2", "it20")),                # equal dual effort
                (c("R0", "it80"), c("R5", "it20")),
                (c("R2", "it20"), c("R5", "it20"))]                # strengthening, high budget
    if family == "AGRB":
        return [(g, FS.lr_config_id("R5", "rel")) for g in cfgs]
    return []


def fmt(df):
    try:
        return df.to_markdown(index=False, floatfmt=".3g")
    except ImportError:          # `tabulate` not installed
        return "```\n" + df.to_string(index=False) + "\n```"



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--profile", default="final")
    ap.add_argument("--family", default="core")
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    time_limit = FS.PROFILES[a.profile]["time_limit"]
    out = os.path.join(root, "analysis")
    os.makedirs(out, exist_ok=True)
    fams = FS.build_families(a.profile, root)
    loaded = {}
    # Every table on disk, for paired comparisons whose baseline belongs to
    # another family on the same cell (E's R2 / R5 live in A).
    tables = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(root, "tables", "*_results.csv")))
              if not os.path.basename(p).startswith(("core_", "all_"))
              and "_all_results" not in os.path.basename(p)]
    ALL = (prep(pd.concat(tables, ignore_index=True)
                .drop_duplicates(["cell_id", "config_id", "idx"]), time_limit)
           if tables else None)
    for fam in FS.parse_families(a.family):
        path = os.path.join(root, "tables", f"{fam}_results.csv")
        if not os.path.exists(path):
            print(f"{fam}: no table (run collect)")
            continue
        df = prep(pd.read_csv(path), time_limit)
        loaded[fam] = df
        if fam == "CAL":
            continue
        rng = np.random.default_rng(RNG_SEED)
        md = [f"# Family {fam}: {FS.FAMILY_INFO[fam][1]}\n",
              f"time limit {time_limit:.0f} s; SGM shifts: time {SHIFT_T}, nodes {SHIFT_N}\n"]
        pair_rows = []
        for cell, idxs, cfgs in fams[fam]:
            cdf = df[df["cell_id"] == cell["id"]]
            if cdf.empty:
                continue
            lr_cfgs = [c for c in cfgs if not c.startswith("GRB-")]
            md.append(f"\n## Cell {cell['id']}  (n={cell['n']}, d={cell['density']}, "
                      f"beta={cell['beta']}, knob={cell['knob']}, {len(idxs)} instances)\n")
            summ = summary(cdf, lr_cfgs)
            if len(lr_cfgs) < len(cfgs):
                summ = pd.concat([summ, summary(cdf, [c for c in cfgs if c.startswith("GRB-")])])
            summ.insert(0, "cell", cell["id"])
            summ.to_csv(os.path.join(out, f"{fam}_summary_{cell['id']}.csv"), index=False)
            md.append(fmt(summ.drop(columns=["cell"])) + "\n")
            pool = ALL[ALL["cell_id"] == cell["id"]] if ALL is not None else cdf
            have = set(pool["config_id"])
            for x, y in comparisons(fam, cfgs):
                if (x in cfgs or y in cfgs) and x in have and y in have:
                    row = paired(pool, x, y, a.boot, rng)
                    row["cell"] = cell["id"]
                    pair_rows.append(row)
        if pair_rows:
            P = pd.DataFrame(pair_rows)
            P["time_p_holm"] = holm(P["time_p"])
            P["node_p_holm"] = holm(P["node_p"].fillna(1.0))
            P.to_csv(os.path.join(out, f"{fam}_pairs.csv"), index=False)
            md.append("\n## Paired comparisons (ratio = b / a; < 1 means b is better)\n")
            md.append(fmt(P[["cell", "a", "b", "n", "time_ratio", "time_ci_lo", "time_ci_hi",
                             "time_p_holm", "node_ratio", "node_ci_lo", "node_ci_hi",
                             "node_p_holm", "time_wins_b", "time_ties", "time_losses_b",
                             "solved_only_a", "solved_only_b", "mcnemar_p",
                             "time_noninferior_10pct"]]) + "\n")
        with open(os.path.join(out, f"{fam}.md"), "w") as f:
            f.write("\n".join(md))
        print(f"{fam}: analysis/{fam}.md")

    # channel-removal interactions: A vs G (probe channel), A vs ACUT (primal)
    rows = []
    rng = np.random.default_rng(RNG_SEED)
    r0, r5 = FS.lr_config_id("R0", "rel"), FS.lr_config_id("R5", "rel")
    for other, a0, a5, label in (("G", FS.lr_config_id("R0", "mf"), FS.lr_config_id("R5", "mf"),
                                  "probe channel (A vs G)"),
                                 ("ACUT", FS.lr_config_id("R0", "rel", variant="cutoff"),
                                  FS.lr_config_id("R5", "rel", variant="cutoff"),
                                  "primal channel (A vs ACUT)")):
        if "A" in loaded and other in loaded:
            for metric in ("nodes", "capped_time"):
                r = interaction(loaded["A"], r0, r5, loaded[other], a0, a5, metric, a.boot, rng)
                rows.append({"contrast": label, "metric": metric, **r})
    if "E" in loaded and "A" in loaded:
        EA = pd.concat([loaded["A"], loaded["E"]]).drop_duplicates(["cell_id", "config_id", "idx"])
        for metric in ("nodes", "capped_time"):
            r = interaction(EA, FS.lr_config_id("R5", "rel", variant="noexact"), r5,
                            EA, FS.lr_config_id("R2", "rel", variant="noexact"),
                            FS.lr_config_id("R2", "rel"), metric, a.boot, rng)
            rows.append({"contrast": "exact dual: effect at R5 / effect at R2", "metric": metric, **r})
            r = interaction(EA, FS.lr_config_id("R5", "rel", variant="norc"), r5,
                            EA, FS.lr_config_id("R0", "rel", variant="norc"), r0, metric,
                            a.boot, rng)
            rows.append({"contrast": "RC fixing: effect at R5 / effect at R0", "metric": metric, **r})
    if rows:
        I = pd.DataFrame(rows)
        I.to_csv(os.path.join(out, "interactions.csv"), index=False)
        with open(os.path.join(out, "interactions.md"), "w") as f:
            f.write("# Interactions (ratio of paired ratios; 1 = no interaction)\n\n" + fmt(I) + "\n")
        print("interactions: analysis/interactions.md")


if __name__ == "__main__":
    sys.exit(main())
