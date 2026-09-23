"""Single-run engine for the frozen MSTKP benchmark.

run_lrbnb() runs ONE LR-BnB configuration on ONE instance and returns a flat
dict of the metrics the paper uses.  It is called inside a fresh worker
process by final_suite.py, which is the only supported driver for the final
suite: every run gets its own interpreter, so the class-level state the
solver keeps (incumbent, caches, pseudocost priors, counters) can never leak
from one run into the next.

What changed against the previous driver in this file:
  * The hard-coded configuration list, the per-rule intermediate CSVs (which
    every ladder rung shared, because they were keyed by branching rule), the
    shared instances.pkl (overwritten by the next run) and the summary/LaTeX
    helpers are gone; configurations are declared once in final_suite.py.
  * Every knob is explicit in the config and written back as read FROM THE
    SOLVER OBJECTS after construction (effective_params), for the node solver
    and for the strong-branching probe solver separately.
  * Objective-cutoff mode for the primal-free replicate (see MSTNode).
  * Counters that could not be analysed (constant or outcome-only) are no
    longer recorded; the time decomposition, probe effort, indicator pool
    and exact-dual counters are new.
The construction of the root node and the B&B call are unchanged, so a
configuration that matches an old one runs the same algorithm.
"""

import gc
import math
import random
import time

from lagrangianrelaxation import LagrangianMST
from mstkpbranchandbound import MSTNode
from branchandbound import RandomBranchingRule, BranchAndBound

# Solver attributes read back after construction.  (name, library default)
_SOLVER_KEYS = [
    ("cut_strengthening", "full"), ("max_active_cuts", 5),
    ("max_cut_depth", float("inf")), ("rank_lift", True),
    ("exact_cut_dual", True), ("use_rc_fixing", True), ("frac_source", "dw"),
    ("lift_cuts", True), ("dual_seed", False), ("dw_harvest", 0),
    ("frac_top_k", 0), ("use_cover_cuts", False), ("max_iter", None),
    ("mu_init", 0.0), ("cut_phase_frac", 3.0), ("root_max_iter", None),
]
# Probes only need the cut-shaping ones.
_PROBE_KEYS = ["cut_strengthening", "max_active_cuts", "max_cut_depth",
               "rank_lift", "exact_cut_dual", "lift_cuts", "dual_seed",
               "use_cover_cuts", "cut_phase_frac"]


def _json_num(v):
    if isinstance(v, float) and math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return v


def _reset_class_state():
    for attr in ("global_edges", "global_graph"):
        if hasattr(MSTNode, attr):
            delattr(MSTNode, attr)
    MSTNode._solver_pool = None
    MSTNode.global_upper_bound = float("inf")
    MSTNode.global_best_edges = None
    MSTNode.objective_cutoff = None
    MSTNode.step_reference = None
    for attr in ("_edge_key", "_edge_list", "_edge_indices", "_edge_weights",
                 "_edge_lengths", "_edge_attributes"):
        setattr(LagrangianMST, attr, None)
    LagrangianMST._sep_cache = None
    LagrangianMST._cut_idx_cache = {}
    LagrangianMST.total_compute_time = 0.0
    LagrangianMST.reset_cut_stats()
    MSTNode.reset_rc_stats()
    MSTNode.reset_pseudocost_stats()


def _pool_summary(hist):
    total = sum(hist.values())
    if not total:
        return None, None, None
    acc, median = 0, None
    for size in sorted(hist):
        acc += hist[size]
        if acc * 2 >= total:
            median = size
            break
    return (median, hist.get(1, 0) / total, hist.get(0, 0) / total)


def build_overrides(config):
    """solver_overrides exactly as the node solvers will receive them."""
    ov = {
        "rank_lift": bool(config["rank_lift"]),
        "exact_cut_dual": bool(config["exact_cut_dual"]),
        "use_rc_fixing": bool(config["use_rc_fixing"]),
        "frac_source": str(config["frac_source"]),
    }
    if config.get("root_max_iter") is not None:
        ov["root_max_iter"] = int(config["root_max_iter"])
    if config["cover_cuts"]:
        ov["cut_strengthening"] = config["cut_strengthening"]
        ov["max_active_cuts"] = int(config["max_active_cuts"])
        if config["cut_root_only"]:
            ov["max_cut_depth"] = 0
    return ov


def run_lrbnb(instance, config, time_limit, instance_seed, cutoff=None):
    """One LR-BnB run.  `instance` needs .edges, .num_nodes, .budget."""
    _reset_class_state()
    gc.collect()
    random.seed(int(instance_seed))

    overrides = build_overrides(config)
    MSTNode.objective_cutoff = float(cutoff) if cutoff is not None else None

    start_total = time.time()
    cpu0 = time.process_time()

    root = MSTNode(
        instance.edges, instance.num_nodes, instance.budget,
        initial_lambda=0.5,
        inherit_lambda=bool(config["inherit_lambda"]),
        inherit_step_size=bool(config["inherit_step_size"]),
        max_iter=int(config["max_iter"]),
        solver_overrides=overrides,
        branching_rule=config["branching_rule"],
        step_size=0.001,
        use_cover_cuts=bool(config["cover_cuts"]),
        cut_frequency=5,
        node_cut_frequency=10,
        parent_cover_cuts=None,
        parent_cover_multipliers=None,
        use_bisection=False,
        verbose=False,
        pseudocosts_up=None, pseudocosts_down=None,
        counts_up=None, counts_down=None,
        reliability_eta=3,
        lookahead_lambda=4,
    )
    root_time = time.time() - start_total
    root_lb = float(getattr(root, "local_lower_bound", float("nan")))
    root_lr_iterations = int(LagrangianMST.lr_iterations)

    solver = root.lagrangian_solver
    effective = {k: _json_num(getattr(solver, k, d)) for k, d in _SOLVER_KEYS}
    effective["branching_rule"] = root.branching_rule
    effective["reliability_eta"] = root.reliability_eta
    effective["sb_max_candidates"] = root._node_opt("sb_max_candidates", 3)
    probe_eff = None
    pool = getattr(MSTNode, "_solver_pool", None)
    if pool is not None and getattr(pool, "_objs", None):
        ps = pool._objs[0]
        _defaults = dict(_SOLVER_KEYS)
        probe_eff = {k: _json_num(getattr(ps, k, _defaults.get(k))) for k in _PROBE_KEYS}
        probe_eff["_is_probe"] = bool(getattr(ps, "_is_probe", False))

    bnb = BranchAndBound(
        RandomBranchingRule(), verbose=False,
        duality_gap_threshold=float(config["duality_gap_threshold"]),
        config={"branching_rule": config["branching_rule"],
                "objective_granularity": float(config["objective_granularity"])},
        instance_seed=int(instance_seed))

    remaining = max(0.0, float(time_limit) - (time.time() - start_total))
    best_solution, best_ub = bnb.solve(root, time_limit_s=remaining)
    wall = time.time() - start_total
    cpu = time.process_time() - cpu0

    timed_out = bool(bnb.timed_out) or wall > float(time_limit)
    ub = float(best_ub)
    gap = float(getattr(bnb, "final_duality_gap", float("inf")))
    final_lb = ub - gap if math.isfinite(gap) and math.isfinite(ub) else None

    if timed_out:
        status = "timeout"
    elif math.isfinite(ub):
        status = "optimal"
    else:
        status = "no_incumbent"

    sol = None
    if cutoff is None:
        edges = MSTNode.global_best_edges
        if not edges and best_solution is not None:
            edges = (getattr(best_solution, "best_feasible_edges", None)
                     or getattr(best_solution, "mst_edges", None))
        sol = [[int(u), int(v)] for u, v in (edges or [])]

    med, single, empty = _pool_summary(LagrangianMST.pool_hist)
    res = {
        "status": status,
        "solved": status == "optimal",
        "obj": ub if math.isfinite(ub) else None,
        "final_lb": final_lb,
        "root_lb": root_lb,
        "wall_time": wall,
        "cpu_time": cpu,
        "root_time": root_time,
        "nodes": int(bnb.total_nodes_solved),
        "lr_iterations": int(LagrangianMST.lr_iterations),
        "root_lr_iterations": root_lr_iterations,
        "probes": int(MSTNode.probe_calls),
        "probe_time": float(MSTNode.probe_time),
        "forced_decisions": int(MSTNode.forced_decisions),
        "forced_empty": int(MSTNode.forced_empty),
        "sep_calls": int(LagrangianMST.sep_calls),
        "sep_time": float(LagrangianMST.sep_time),
        "cuts_separated": int(LagrangianMST.cuts_separated),
        "cuts_infeasible": int(LagrangianMST.cuts_infeasible),
        "cut_forced_exclusions": int(MSTNode.cut_forced_exclusions),
        "exact_dual_nodes": int(LagrangianMST.exact_dual_nodes),
        "exact_dual_gain": float(LagrangianMST.exact_dual_gain),
        "rc_edges_excluded": int(MSTNode.rc_edges_excluded),
        "rc_edges_fixed": int(MSTNode.rc_edges_fixed),
        "indicator_calls": int(LagrangianMST.indicator_calls),
        "indicator_none": int(LagrangianMST.indicator_none),
        "indicator_time": float(LagrangianMST.indicator_time),
        "pool_median": med,
        "pool_singleton_share": single,
        "pool_empty_share": empty,
        "solution_edges": sol,
        "cutoff": float(cutoff) if cutoff is not None else None,
        "cutoff_violated": (cutoff is not None and math.isfinite(ub)
                            and ub < float(cutoff) - 1e-6),
        # --- verification-only counters (diagnostics table) ---
        "diag": {
            "effective_params": effective,
            "effective_probe_params": probe_eff,
            "pool_hist": {str(k): v for k, v in sorted(LagrangianMST.pool_hist.items())},
            "sep_probe_calls": int(LagrangianMST.sep_probe_calls),
            "sep_max_depth": int(LagrangianMST.sep_max_depth),
            "rank_lift_calls": int(LagrangianMST.rank_lift_calls),
            "exact_dual_probe_nodes": int(LagrangianMST.exact_dual_probe_nodes),
            "nodes_pruned_lower_bound": int(bnb.nodes_pruned_lower_bound),
            "nodes_pruned_budget": int(bnb.nodes_pruned_budget),
            "nodes_pruned_gap": int(bnb.nodes_pruned_gap),
            "step_reference": MSTNode.step_reference,
        },
    }
    MSTNode.objective_cutoff = None
    MSTNode.step_reference = None
    return res


if __name__ == "__main__":
    raise SystemExit("Use final_suite.py (run / run-one) to execute the final benchmark.")
