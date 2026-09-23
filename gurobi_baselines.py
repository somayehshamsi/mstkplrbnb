"""Gurobi MIP baselines for the frozen MSTKP benchmark.

Every formulation shares the MSTKP objective, the length budget and
sum_e x_e = n - 1; they differ only in how spanning-tree connectivity is
enforced.

    SCF          single-commodity flow, f_ij + f_ji <= (n-1) x_e.
                 Compact, weak LP relaxation.
    DMCF         DIRECTED multi-commodity flow: y_ij + y_ji = x_e and
                 f^k_ij <= y_ij for every commodity k.  Its LP projects onto
                 the spanning-tree polytope (Magnanti and Wolsey), so its root
                 LP equals the plain Lagrangian dual bound.  The undirected
                 version (f^k_ij + f^k_ji <= x_e, the paper's Appendix A.2)
                 only reaches the weaker cut-set LP.
    DCUT         directed cut-set on arc variables y with one entering arc per
                 non-root vertex.  y(delta^-(S)) >= 1 is separated at
                 FRACTIONAL node relaxations (max-flow user cuts) and at
                 integer incumbents (lazy constraints) -- the standard LP
                 branch-and-cut; after separation its LP is again the
                 spanning-tree polytope.
    CUTSETLAZY   undirected cut-set x(delta(S)) >= 1 separated only at integer
                 incumbents.  The previous version's baseline, kept only as a
                 continuity row.

Fairness settings, identical for every formulation: Threads = 1, MIPGap = 0,
MIPGapAbs = 0.999 (objective is integral, so a gap below one unit is a
proof, exactly the rule LR-BnB prunes with), the minimum-length spanning
tree as MIP start (the incumbent LR-BnB starts from), model build time
counted against the limit, SoftMemLimit so an oversized model ends with a
recorded MEM_LIMIT status instead of taking the machine down.
"""

import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import maximum_flow, minimum_spanning_tree

FORMULATIONS = ("SCF", "DMCF", "DCUT", "CUTSETLAZY")


def _arrays(instance):
    E = instance.edges
    U = np.fromiter((e[0] for e in E), dtype=np.int64, count=len(E))
    V = np.fromiter((e[1] for e in E), dtype=np.int64, count=len(E))
    W = np.fromiter((e[2] for e in E), dtype=np.float64, count=len(E))
    L = np.fromiter((e[3] for e in E), dtype=np.float64, count=len(E))
    return U, V, W, L


def _min_length_tree(n, U, V, W, L):
    """Minimum-length spanning tree, ties broken by weight (edge indices)."""
    cost = L + W * 1e-9 + 1.0  # strictly positive: csgraph drops zeros
    M = sp.coo_matrix((cost, (U, V)), shape=(n, n)).tocsr()
    T = minimum_spanning_tree(M).tocoo()
    key = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(U, V))}
    out = []
    for a, b in zip(T.row.tolist(), T.col.tolist()):
        out.append(key[(min(a, b), max(a, b))])
    return out


def _incidence(n, U, V):
    """Node-arc incidence (out = +1, in = -1) for arcs 2e = (u,v), 2e+1 = (v,u)."""
    m = len(U)
    tails = np.empty(2 * m, dtype=np.int64)
    heads = np.empty(2 * m, dtype=np.int64)
    tails[0::2], heads[0::2] = U, V
    tails[1::2], heads[1::2] = V, U
    cols = np.arange(2 * m)
    A = sp.csr_matrix(
        (np.concatenate([np.ones(2 * m), -np.ones(2 * m)]),
         (np.concatenate([tails, heads]), np.concatenate([cols, cols]))),
        shape=(n, 2 * m))
    return A, tails, heads


def estimate_memory_gb(formulation, n, m):
    """Conservative memory reservation for the scheduler and SoftMemLimit."""
    if formulation == "DMCF":
        nvars = 2.0 * m * (n - 1)
        return float(min(40.0, 2.0 + 2.5e-6 * nvars))
    return float(min(12.0, 1.5 + 4e-6 * m * 8))


def run_gurobi(instance, formulation, time_limit, mem_limit_gb=None, seed=0,
               extra_params=None):
    """Solve one instance with one formulation.  Returns a flat metrics dict."""
    import gurobipy as gp
    from gurobipy import GRB

    if formulation not in FORMULATIONS:
        raise ValueError(f"unknown formulation {formulation!r}")

    t0 = time.time()
    n = int(instance.num_nodes)
    B = float(instance.budget)
    U, V, W, L = _arrays(instance)
    m = len(U)
    root = 0

    out = {
        "formulation": formulation, "grb_version": ".".join(map(str, gp.gurobi.version())),
        "status": "error", "solved": False, "obj": None, "final_lb": None,
        "root_lb": None, "nodes": None, "build_time": None, "grb_runtime": None,
        "user_cuts": 0, "lazy_cuts": 0, "solution_edges": None,
    }

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model(env=env)
    try:
        p = model.Params
        p.Threads = 1
        p.MIPGap = 0.0
        p.MIPGapAbs = 0.999
        p.Seed = int(seed)
        if mem_limit_gb:
            p.SoftMemLimit = float(mem_limit_gb)
        # Tests only (e.g. to switch Gurobi's own cuts off and watch the
        # separator work).  The benchmark never passes this.
        for _k, _v in (extra_params or {}).items():
            model.setParam(_k, _v)

        x = model.addMVar(m, vtype=GRB.BINARY, name="x")
        model.setObjective(W @ x, GRB.MINIMIZE)
        model.addConstr(L @ x <= B, name="budget")
        model.addConstr(np.ones(m) @ x == n - 1, name="card")

        A, tails, heads = _incidence(n, U, V)
        # arc -> edge aggregation: S2[e, 2e] = S2[e, 2e+1] = 1
        S2 = sp.csr_matrix((np.ones(2 * m), (np.repeat(np.arange(m), 2), np.arange(2 * m))),
                           shape=(m, 2 * m))

        y = None
        if formulation == "SCF":
            f = model.addMVar(2 * m, lb=0.0, name="f")
            b = -np.ones(n)
            b[root] = n - 1
            model.addConstr(A @ f == b, name="flow")
            model.addConstr(S2 @ f - (n - 1) * x <= 0, name="cap")

        elif formulation == "DMCF":
            y = model.addMVar(2 * m, lb=0.0, ub=1.0, name="y")
            model.addConstr(S2 @ y - x == 0, name="orient")
            K = n - 1
            f = model.addMVar(K * 2 * m, lb=0.0, ub=1.0, name="f")
            keep = np.array([i for i in range(n) if i != root])
            Ak = A[keep, :]                       # drop the redundant root row
            model.addConstr(sp.kron(sp.identity(K, format="csr"), Ak, format="csr") @ f
                            == np.concatenate([-(np.arange(n)[keep] == k).astype(float)
                                               for k in keep]),
                            name="cons")
            link = sp.kron(sp.identity(K, format="csr"), sp.identity(2 * m, format="csr"),
                           format="csr")
            ylink = sp.kron(np.ones((K, 1)), sp.identity(2 * m, format="csr"), format="csr")
            model.addConstr(link @ f - ylink @ y <= 0, name="link")

        elif formulation == "DCUT":
            y = model.addMVar(2 * m, vtype=GRB.BINARY, name="y")
            model.addConstr(S2 @ y - x == 0, name="orient")
            Hin = sp.csr_matrix((np.ones(2 * m), (heads, np.arange(2 * m))), shape=(n, 2 * m))
            rhs = np.ones(n)
            rhs[root] = 0.0
            model.addConstr(Hin @ y == rhs, name="indeg")
            p.LazyConstraints = 1
            p.PreCrush = 1

        elif formulation == "CUTSETLAZY":
            p.LazyConstraints = 1

        # MIP start: the minimum-length tree (x only; Gurobi completes the rest)
        start = np.zeros(m)
        ml_tree = _min_length_tree(n, U, V, W, L)
        start[ml_tree] = 1.0
        x.Start = start

        model.update()
        build_time = time.time() - t0
        out["build_time"] = build_time
        remaining = float(time_limit) - build_time
        if remaining <= 1.0:
            out["status"] = "timeout"
            return out
        p.TimeLimit = remaining

        xs = x.tolist()
        ys = y.tolist() if y is not None else None
        state = {"root_bnd": None, "user": 0, "lazy": 0}
        SCALE = 10 ** 6
        tails_l, heads_l = tails.tolist(), heads.tolist()
        U_l, V_l = U.tolist(), V.tolist()

        def components(sel_edges):
            parent = list(range(n))

            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a
            for e in sel_edges:
                ra, rb = find(U_l[e]), find(V_l[e])
                if ra != rb:
                    parent[ra] = rb
            comp = {}
            for v in range(n):
                comp.setdefault(find(v), []).append(v)
            return list(comp.values())

        def dcut_expr(S):
            inS = np.zeros(n, dtype=bool)
            inS[list(S)] = True
            arcs = np.nonzero(inS[heads] & ~inS[tails])[0]
            return gp.quicksum(ys[a] for a in arcs.tolist()), arcs

        def undirected_expr(S):
            inS = np.zeros(n, dtype=bool)
            inS[list(S)] = True
            es = np.nonzero(inS[U] ^ inS[V])[0]
            return gp.quicksum(xs[e] for e in es.tolist())

        def cb(model_, where):
            if where == GRB.Callback.MIP:
                if model_.cbGet(GRB.Callback.MIP_NODCNT) < 0.5:
                    state["root_bnd"] = model_.cbGet(GRB.Callback.MIP_OBJBND)
                return
            if where == GRB.Callback.MIPSOL and formulation in ("DCUT", "CUTSETLAZY"):
                xv = model_.cbGetSolution(xs)
                sel = [e for e in range(m) if xv[e] > 0.5]
                comps = components(sel)
                if len(comps) <= 1:
                    return
                for S in comps:
                    if root in S:
                        continue
                    if formulation == "DCUT":
                        expr, _ = dcut_expr(S)
                    else:
                        expr = undirected_expr(S)
                    model_.cbLazy(expr >= 1)
                    state["lazy"] += 1
                return
            if where == GRB.Callback.MIPNODE and formulation == "DCUT":
                if model_.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
                    return
                yv = np.asarray(model_.cbGetNodeRel(ys), dtype=float)
                cap = np.floor(np.clip(yv, 0.0, 1.0) * SCALE).astype(np.int32)
                nz = cap > 0
                C = sp.csr_matrix((cap[nz], (tails[nz], heads[nz])), shape=(n, n), dtype=np.int32)
                covered = np.zeros(n, dtype=bool)
                seen = set()
                for t in range(n):
                    if t == root or covered[t]:
                        continue
                    res = maximum_flow(C, root, t)
                    if res.flow_value >= SCALE * (1.0 - 1e-4):
                        continue
                    F = getattr(res, "flow", None)
                    if F is None:
                        F = res.residual
                    R = (C - F).tocsr()
                    R.data[R.data <= 0] = 0
                    R.eliminate_zeros()
                    reach = np.zeros(n, dtype=bool)
                    reach[root] = True
                    stack = [root]
                    while stack:
                        a = stack.pop()
                        for b in R.indices[R.indptr[a]:R.indptr[a + 1]].tolist():
                            if not reach[b]:
                                reach[b] = True
                                stack.append(b)
                    S = tuple(np.nonzero(~reach)[0].tolist())
                    if not S or S in seen:
                        continue
                    seen.add(S)
                    expr, arcs = dcut_expr(S)
                    if float(yv[arcs].sum()) < 1.0 - 1e-6:
                        model_.cbCut(expr >= 1)
                        state["user"] += 1
                        covered[list(S)] = True

        model.optimize(cb)

        st = model.Status
        out["grb_runtime"] = float(model.Runtime)
        out["nodes"] = float(model.NodeCount)
        out["user_cuts"], out["lazy_cuts"] = state["user"], state["lazy"]
        try:
            out["final_lb"] = float(model.ObjBound)
        except Exception:
            pass
        # Root bound = the bound on leaving the root.  Callbacks are periodic,
        # so the last one at node 0 can precede the end of root processing;
        # when the search never left the root the final bound IS the root
        # bound (a solve closed at the root used to report its pre-cut bound).
        if model.NodeCount <= 1 or state["root_bnd"] is None:
            out["root_lb"] = out["final_lb"]
        else:
            out["root_lb"] = state["root_bnd"]
        if model.SolCount > 0:
            # The objective is the integral weight of the returned tree, not
            # ObjVal: with integrality tolerances ObjVal can read 36562.9999
            # for a tree of weight 36563.
            xv = x.X
            sel = [e for e in range(m) if xv[e] > 0.5]
            out["obj"] = float(W[sel].sum())
            out["grb_objval"] = float(model.ObjVal)
            out["solution_edges"] = [[int(U[e]), int(V[e])] for e in sel]
        if st == GRB.OPTIMAL:
            out["status"], out["solved"] = "optimal", True
        elif st == GRB.TIME_LIMIT:
            out["status"] = "timeout"
        elif st == GRB.MEM_LIMIT:
            out["status"] = "memory"
        elif st == GRB.INFEASIBLE:
            out["status"] = "infeasible"
        else:
            out["status"] = f"grb_status_{st}"
        return out
    except gp.GurobiError as exc:
        if getattr(exc, "errno", None) == 10001 or "memory" in str(exc).lower():
            out["status"] = "memory"
            out["error"] = str(exc)
            return out
        raise
    finally:
        out["wall_time"] = time.time() - t0
        try:
            model.dispose()
            env.dispose()
        except Exception:
            pass
