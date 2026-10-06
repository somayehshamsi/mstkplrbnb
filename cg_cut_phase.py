"""Exact cut dual by column generation (configuration variant `cg`).

At a node with cover cuts, the cut rungs (R1-R5) run a subgradient cut phase
on (lambda, mu) after the exact plain dual, followed by the exact cut-dual
step.  Variant `cg` replaces that cut phase at the node by column generation
on the node's cut-augmented Dantzig-Wolfe master:

    min   sum_T w(T) z_T
    s.t.  sum_T z_T             =  1          (pi0)
          sum_T l(T) z_T        <= B          (lambda >= 0)
          sum_T |T & S_k| z_T   <= r_k        (mu_k >= 0), one row per active cover
          z >= 0,

over the spanning trees T of the node's reduced graph (F+ forced in, F-
removed).  Its LP value is the node's Lagrangian dual
max_{lambda, mu >= 0} L(lambda, mu), and pricing is one MST under the weights
w + lambda l + sum_k mu_k 1[S_k].  Every pricing step yields a valid bound
L(lambda, mu) = MST - lambda B - sum_k mu_k r_k; the node keeps the best one,
so the bound is valid whether or not the master has converged.

Separation (one round per node by default): once the master has converged,
the budget-violating tree with the largest weight z_T in the master's
solution is separated exactly as in the other cut rungs (generate_cover_cuts,
Section 6).  Covers the master's solution violates enter as new rows, at most
`max_active_cuts` in the pool (covers whose multiplier is zero make room
first), and the master is solved again.

The master's solution, summed over the trees that contain each edge, is the
node's LP solution under the active covers; it replaces the Dantzig-Wolfe
indicator for branching at this node.

Strong-branching probes are unchanged: they never run this phase (the probe
solvers do not receive `cut_dual`), so the variant differs from its base
configuration only in the cut phase of the node solves.

Requires highspy (pip install highspy).  The master is tiny (at most
1 + 1 + max_active_cuts rows); HiGHS re-solves it from the previous basis
after every added column.
"""
from time import time

import numpy as np

try:
    import highspy
except ImportError:          # reported by the caller when the variant is used
    highspy = None

# Frozen settings of variant `cg`.  A solver attribute of the same name
# overrides them (none of the benchmark configurations does).
DEFAULTS = {
    "cg_max_rounds": 1,       # separation rounds per node
    "cg_max_pricing": 50,     # pricing steps per master solve
    "cg_sep_one": True,       # separate only the heaviest violating support tree
    "cg_early_stop": True,    # stop once the bound prunes the node
    "cg_tol": 1e-4,           # reduced-cost tolerance
}


def _opt(solver, name):
    return getattr(solver, name, DEFAULTS[name])


class _Master:
    """The restricted master in HiGHS: row 0 convexity, row 1 budget, then
    one row per cover."""

    def __init__(self, budget):
        if highspy is None:
            raise RuntimeError("variant `cg` needs highspy: pip install highspy")
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        inf = highspy.kHighsInf
        h.addRows(2, np.array([1.0, -inf]), np.array([1.0, float(budget)]), 0,
                  np.array([], dtype=np.int32), np.array([], dtype=np.int32),
                  np.array([], dtype=float))
        self.h = h
        self.inf = inf

    def add_col(self, cost, coefs):
        coefs = np.asarray(coefs, dtype=float)
        self.h.addCol(float(cost), 0.0, self.inf, len(coefs),
                      np.arange(len(coefs), dtype=np.int32), coefs)

    def add_row(self, rhs, values):
        vals = np.asarray(values, dtype=float)
        nz = np.nonzero(vals)[0].astype(np.int32)
        self.h.addRow(-self.inf, float(rhs), len(nz), nz, vals[nz])

    def solve(self):
        """(objective, lambda, mu, pi0, z) or None if not optimal."""
        t0 = time()
        self.h.run()
        from lagrangianrelaxation import LagrangianMST
        LagrangianMST.cg_lp_time += time() - t0
        LagrangianMST.cg_lp_solves += 1
        if self.h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            return None
        sol = self.h.getSolution()
        rd = np.asarray(sol.row_dual, dtype=float)
        return (float(self.h.getInfo().objective_function_value),
                max(0.0, -rd[1]), np.maximum(0.0, -rd[2:]), float(rd[0]),
                np.asarray(sol.col_value, dtype=float))


def cg_cut_phase(solver, plain_trees, can_separate, cap):
    """Run the cut phase of `solver` (a LagrangianMST inside solve()) by
    column generation.

    plain_trees   the trees of the exact plain dual, (edges, idx, w, l)
    can_separate  whether new covers may be separated at this node
    cap           the active-pool cap (max_active_cuts)

    Writes the node's dual state back onto the solver (bound, lambda, cover
    pool, multipliers, priced tree, LP solution).  Returns False if a
    separated cover proves the node infeasible, True otherwise.
    """
    from lagrangianrelaxation import LagrangianMST

    LagrangianMST.cg_nodes += 1
    W = solver.edge_weights
    Lg = solver.edge_lengths
    B = float(solver.budget)
    m = W.shape[0]
    rounds = int(_opt(solver, "cg_max_rounds")) if can_separate else 0
    max_pricing = int(_opt(solver, "cg_max_pricing"))
    sep_one = bool(_opt(solver, "cg_sep_one"))
    early = bool(_opt(solver, "cg_early_stop"))
    tol = float(_opt(solver, "cg_tol"))
    gran = float(getattr(solver, "objective_granularity", 1.0))
    F_in = solver.fixed_edges
    F_out = solver.excluded_edges
    free_mask = solver._get_free_mask()

    def free_idx(support):
        arr = solver._cut_index_array(support)
        return arr if free_mask is None else arr[free_mask[arr]]

    def row_mask(idx):
        mk = np.zeros(m, dtype=bool)
        mk[idx] = True
        return mk

    # The inherited pool, already projected onto this node's fixings at the
    # top of solve(): supports are free edges, right-hand sides reduced.
    rhs_eff = getattr(solver, "_rhs_eff", {}) or {}
    cuts = []                      # [support, rhs, free index array, mask]
    for i, (c, r) in enumerate(solver.best_cuts or []):
        c = c if isinstance(c, frozenset) else frozenset(c)
        ix = free_idx(c)
        cuts.append([c, int(rhs_eff.get(i, r)), ix, row_mask(ix)])

    cols, keys = [], set()

    def coefs(idx):
        return [1.0, float(Lg[idx].sum())] + [float(c[3][idx].sum()) for c in cuts]

    def build_master():
        M = _Master(B)
        for c in cuts:
            M.add_row(c[1], [])
        for idx in cols:
            M.add_col(float(W[idx].sum()), coefs(idx))
        return M

    master = [build_master()]

    def add_col(idx):
        idx = np.sort(np.asarray(idx, dtype=np.int64))
        k = idx.tobytes()
        if k in keys:
            return False
        keys.add(k)
        cols.append(idx)
        master[0].add_col(float(W[idx].sum()), coefs(idx))
        return True

    for _edges, idx, _w, _l in plain_trees:
        if idx is not None:
            add_col(idx)

    best = {"lb": float(solver.best_lower_bound), "lam": float(solver.best_lambda),
            "mu": {}, "edges": None, "idx": None}
    start_lb = best["lb"]
    stopped = [False]

    def ub_now():
        return min(float(getattr(solver, "incumbent_ub", float("inf"))),
                   float(solver.best_upper_bound))

    def price():
        """Column generation on the current pool; the last master solution."""
        out = None
        for _ in range(max_pricing):
            r = master[0].solve()
            if r is None:
                return out
            z, lam, mu, pi0, x = r
            w = W + lam * Lg
            for c, mk in zip(cuts, mu):
                if mk > 0.0:
                    w[c[2]] += mk
            cost, length, edges, idx = solver._mst_core(w)
            LagrangianMST.cg_msts += 1
            if not edges or idx is None:
                return out
            lb = cost - lam * B - sum(mk * c[1] for c, mk in zip(cuts, mu))
            feasible = float(length) <= B + 1e-9
            solver._record_primal_solution(edges, feasible)
            if feasible:
                wt = float(W[idx].sum())
                if wt < solver.best_upper_bound:
                    solver.best_upper_bound = wt
                    solver.best_feasible_edges = list(edges)
            if lb > best["lb"] + 1e-9:
                best.update(lb=lb, lam=lam, edges=edges, idx=idx,
                            mu={(c[0], c[1]): float(v) for c, v in zip(cuts, mu)})
            out = (z, lb, lam, mu, x)
            if early and best["lb"] > ub_now() - gran + 1e-6:
                LagrangianMST.cg_early_stops += 1
                stopped[0] = True
                break
            if cost - pi0 >= -tol or z - lb <= tol:
                break                      # no improving column: master optimal
            if not add_col(idx):
                break
        return out

    last = price()
    seen = {(c[0], c[1]) for c in cuts}
    for _round in range(rounds):
        if last is None or stopped[0]:
            break
        z, lb, lam, mu, x = last
        support = np.nonzero(x > 1e-9)[0].tolist()
        violating = [k for k in support if float(Lg[cols[k]].sum()) > B + 1e-9]
        if not violating:
            break
        if sep_one:
            violating = [max(violating, key=lambda k: x[k])]
        cand = []
        LagrangianMST.cut_nodes += 1
        for k in violating:
            edges = [solver.idx_to_edge[int(j)] for j in cols[k]]
            for S, r in (solver.generate_cover_cuts(edges) or []):
                S = S if isinstance(S, frozenset) else frozenset(S)
                # Project onto the node's fixings, as the subgradient phase does.
                S_fixed = S & F_in
                S_free = (S - F_in - F_out) if (S_fixed or (S & F_out)) else S
                r_eff = int(r) - len(S_fixed)
                if r_eff < 0:
                    LagrangianMST.cuts_infeasible += 1
                    return False
                if len(S_free) <= r_eff:
                    continue
                key = (frozenset(S_free), r_eff)
                if key in seen:
                    continue
                ix = free_idx(key[0])
                mk = row_mask(ix)
                viol = sum(x[j] * float(mk[cols[j]].sum()) for j in support) - r_eff
                if viol > 1e-6:
                    seen.add(key)
                    cand.append((viol, len(key[0]), [key[0], r_eff, ix, mk]))
        if not cand:
            break
        cand.sort(key=lambda t: (-t[0], t[1]))
        rebuild = False
        if len(cuts) + len(cand) > cap:
            keep = [c for c, v in zip(cuts, mu) if v > 1e-9]
            if len(keep) < len(cuts):
                cuts[:] = keep
                rebuild = True
        slots = max(0, int(cap) - len(cuts))
        if slots == 0:
            break
        added = [c for _v, _s, c in cand[:slots]]
        for c in added:
            cuts.append(c)
            if not rebuild:
                master[0].add_row(c[1], [float(c[3][idx].sum()) for idx in cols])
        if rebuild:
            master[0] = build_master()
        LagrangianMST.cuts_separated += len(added)
        LagrangianMST.cg_sep_rounds += 1
        res = price()
        if res is not None:
            last = res

    # ---- write back the node state that MSTNode and RC fixing read ----------
    solver.best_cuts = [(c[0], int(c[1])) for c in cuts]
    solver._rhs_eff = {i: int(c[1]) for i, c in enumerate(cuts)}
    solver._cut_edge_idx = [c[2] for c in cuts]
    solver._cut_edge_idx_all = [solver._cut_index_array(c[0]) for c in cuts]
    if best["edges"] is not None and best["lb"] > start_lb + 1e-6:
        LagrangianMST.cg_gain_nodes += 1
        LagrangianMST.cg_gain += best["lb"] - start_lb
        solver.best_lower_bound = best["lb"]
        solver.best_lambda = best["lam"]
        solver.best_mst_edges = best["edges"]
        solver.last_mst_edges = best["edges"]
        solver._last_mst_idx = best["idx"]
        solver._last_mst_list = best["edges"]
        mu = {i: best["mu"].get((c[0], c[1]), 0.0) for i, c in enumerate(cuts)}
    else:
        mu = {i: 0.0 for i in range(len(cuts))}
    solver.best_cut_multipliers_for_best_bound = dict(mu)
    solver.best_cut_multipliers = dict(mu)
    solver.lmbda = solver.best_lambda
    solver._invalidate_weight_cache()
    if last is not None and cuts:
        x = last[4]
        frac = {}
        for k in np.nonzero(x > 1e-12)[0].tolist():
            for j in cols[k].tolist():
                e = solver.idx_to_edge[int(j)]
                frac[e] = frac.get(e, 0.0) + float(x[k])
        solver._cg_frac = {e: v for e, v in frac.items() if v > 1e-6}
    return True
