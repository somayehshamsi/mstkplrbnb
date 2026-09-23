

import heapq
import time
# import random
import networkx as nx
from lagrangianrelaxation import LagrangianMST
from branchandbound import Node
# , BranchAndBound, RandomBranchingRule
import math
from collections import defaultdict  # Add at top if not imported
import random

from contextlib import contextmanager


class SolverPool:
    """
    Keeps a small pool of LagrangianMST objects that share the same static
    graph/arrays. Each probe 'borrows' a solver, resets it, runs a few iters,
    then returns it to the pool.
    """
    def __init__(self, factory, size=3):
        self._objs = [factory() for _ in range(size)]
        self._free = self._objs.copy()

    @contextmanager
    def borrow(self):
        # If all are busy, reuse the first one; strong-branching is sequential here
        obj = self._free.pop() if self._free else self._objs[0]
        try:
            yield obj
        finally:
            # Make sure per-iteration buffers are cleared between probes
            cleanup = getattr(obj, "clear_iteration_state", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception:
                    pass
            if obj not in self._free:
                self._free.append(obj)






class MSTNode(Node):
    _solver_pool = None  # shared across all nodes in one solve
    global_upper_bound = float("inf")  # incumbent shared with node solvers
    # The tree behind global_upper_bound.  A node can find an incumbent and
    # then be discarded -- pruned on its own bound, or filtered out as a
    # child -- so the search has to read the incumbent from here rather than
    # from the nodes it happens to pop; see BranchAndBound.solve.
    global_best_edges = None

    # Reduced-cost fixing statistics for one run; see
    # LagrangianMST.reduced_cost_fixing.
    rc_edges_excluded = 0
    rc_edges_fixed = 0
    rc_nodes = 0
    # Edges forced out by a saturated cover; see _project_and_remap_for_child.
    cut_forced_exclusions = 0

    # Strong-branching effort and forced single-child decisions (reset per
    # run by reset_rc_stats).  A probe is one simulate_branching_bound call,
    # i.e. one tentative child.
    probe_calls = 0
    probe_time = 0.0
    forced_decisions = 0      # probes proved one side infeasible
    forced_empty = 0          # probes proved both sides infeasible

    # Objective-cutoff mode (experiment A-cut only; None everywhere else).
    # The search is handed UB = objective_cutoff (the known optimum) for
    # pruning and reduced-cost fixing.  The node dual's Polyak step is ALSO
    # sized by the incumbent (see the note on _root_primal_heuristic: a tight
    # incumbent collapses the step), so in cutoff mode the step is formed
    # against `step_reference` instead -- the minimum-length-tree incumbent
    # every run starts from -- which keeps the dual rule identical across
    # configurations and independent of what the search finds.
    objective_cutoff = None
    step_reference = None

    # Running totals of the observed PER-UNIT pseudocosts, used as the prior
    # for a direction that has never been observed.  The rules used to fall
    # back on the constant 1.0, which is meaningless here: objective values
    # run to 10^6 and observed bound changes to 10^3, so a prior of 1.0 is
    # numerically zero and an edge with no history could never outscore one
    # with any history -- the estimates then never spread beyond the handful
    # of edges the search happened to observe first.  The average of what has
    # been observed is the standard choice and is scale-free.
    pc_sum_up = 0.0
    pc_cnt_up = 0
    pc_sum_down = 0.0
    pc_cnt_down = 0

    # Floor on a branching distance, so a per-unit pseudocost stays bounded.
    PC_MIN_DIST = 1e-2

    @classmethod
    def note_pseudocost(cls, up, value):
        if value is None or math.isnan(value) or math.isinf(value):
            return
        if up:
            cls.pc_sum_up += float(value)
            cls.pc_cnt_up += 1
        else:
            cls.pc_sum_down += float(value)
            cls.pc_cnt_down += 1

    @classmethod
    def pseudocost_prior(cls, up):
        """Average observed per-unit pseudocost, or 0 before any observation."""
        if up:
            return cls.pc_sum_up / cls.pc_cnt_up if cls.pc_cnt_up else 0.0
        return cls.pc_sum_down / cls.pc_cnt_down if cls.pc_cnt_down else 0.0

    @classmethod
    def reset_pseudocost_stats(cls):
        cls.pc_sum_up = 0.0
        cls.pc_cnt_up = 0
        cls.pc_sum_down = 0.0
        cls.pc_cnt_down = 0

    @classmethod
    def reset_rc_stats(cls):
        cls.rc_edges_excluded = 0
        cls.rc_edges_fixed = 0
        cls.rc_nodes = 0
        cls.cut_forced_exclusions = 0
        cls.probe_calls = 0
        cls.probe_time = 0.0
        cls.forced_decisions = 0
        cls.forced_empty = 0

    @classmethod
    def incumbent_for_step(cls):
        """The incumbent a node dual forms its Polyak gap against."""
        if cls.objective_cutoff is not None and cls.step_reference is not None:
            return cls.step_reference
        return getattr(cls, "global_upper_bound", float("inf"))

    def __init__(self, edges, num_nodes, budget, fixed_edges=set(), excluded_edges=set(), branched_edges=set(),
                 initial_lambda=0.05, inherit_lambda=False, branching_rule="random_mst",
                 step_size= 0.001, inherit_step_size=False, use_cover_cuts=False, cut_frequency=5,
                 node_cut_frequency=10, parent_cover_cuts=None, parent_cover_multipliers=None,
                 use_bisection=False, max_iter=5, verbose=False, depth=0,
                 pseudocosts_up=None, pseudocosts_down=None, counts_up=None, counts_down=None,
                 reliability_eta=3, lookahead_lambda=4, solver_overrides=None,
                 parent_lower_bound=None, fixed_idx=None, excluded_idx=None):
        if depth == 0:
            MSTNode.global_edges = [(min(u, v), max(u, v), w, l) for u, v, w, l in edges]
            MSTNode.global_graph = nx.Graph()
            MSTNode.global_graph.add_edges_from(
                [(u, v, {"w": w, "l": l}) for u, v, w, l in MSTNode.global_edges]
            )
            MSTNode._solver_pool = None  # reset pool for a fresh instance
            MSTNode.reset_pseudocost_stats()

            # Seed the incumbent with the minimum-LENGTH spanning tree, which is
            # budget-feasible whenever the instance is.  Without a finite
            # incumbent the Polyak step of (28) cannot be formed and the node
            # multiplier loop silently falls back to a constant step size.
            MSTNode.global_upper_bound = float("inf")
            try:
                MSTNode.global_best_edges = None
                _len_mst = nx.minimum_spanning_tree(MSTNode.global_graph, weight="l")
                _tot_l = sum(d["l"] for _, _, d in _len_mst.edges(data=True))
                if _tot_l <= budget:
                    MSTNode.global_upper_bound = float(
                        sum(d["w"] for _, _, d in _len_mst.edges(data=True))
                    )
                    MSTNode.global_best_edges = [
                        (min(u, v), max(u, v)) for u, v in _len_mst.edges
                    ]
            except Exception:
                pass

            # Cutoff mode: remember the seeded incumbent as the fixed step
            # reference, then hand the search the known optimum.
            if MSTNode.objective_cutoff is not None:
                MSTNode.step_reference = MSTNode.global_upper_bound
                MSTNode.global_upper_bound = min(
                    MSTNode.global_upper_bound, float(MSTNode.objective_cutoff))
            else:
                MSTNode.step_reference = None

        # self.pseudocosts_up = pseudocosts_up or defaultdict(float)
        # self.pseudocosts_down = pseudocosts_down or defaultdict(float)
        # self.counts_up = counts_up or defaultdict(int)
        # self.counts_down = counts_down or defaultdict(int)
        self.pseudocosts_up = (
        pseudocosts_up if pseudocosts_up is not None else defaultdict(float)
        )
        self.pseudocosts_down = (
            pseudocosts_down if pseudocosts_down is not None else defaultdict(float)
        )
        self.counts_up = (
            counts_up if counts_up is not None else defaultdict(int)
        )
        self.counts_down = (
            counts_down if counts_down is not None else defaultdict(int)
        )
        self.reliability_eta = reliability_eta


        self.lookahead_lambda = lookahead_lambda

        self.depth = depth
        # create_children and create_single_child hand these down as
        # frozensets of already-normalised edges, so re-normalising them was
        # an O(depth) rebuild of three sets per node for nothing.
        self.fixed_edges = (
            fixed_edges if isinstance(fixed_edges, frozenset)
            else frozenset(tuple(sorted((u, v))) for u, v in fixed_edges)
        )
        self.excluded_edges = (
            excluded_edges if isinstance(excluded_edges, frozenset)
            else frozenset(tuple(sorted((u, v))) for u, v in excluded_edges)
        )
        self.branched_edges = (
            branched_edges if isinstance(branched_edges, frozenset)
            else frozenset(tuple(sorted((u, v))) for u, v in branched_edges)
        )
        if not hasattr(MSTNode, "global_edges"):
            MSTNode.global_edges = edges
        self.edges = MSTNode.global_edges

        self.num_nodes = num_nodes
        self.budget = budget

        self.inherit_lambda = inherit_lambda
        self.initial_lambda = initial_lambda if initial_lambda is not None else 0.05
        self.branching_rule = branching_rule
        self.step_size = step_size
        self.inherit_step_size = inherit_step_size

        self.use_cover_cuts = use_cover_cuts
        self.cut_frequency = cut_frequency
        self.node_cut_frequency = node_cut_frequency
        self.use_bisection = use_bisection
        self.verbose = verbose

        self.lagrangian_solver = LagrangianMST(
            MSTNode.global_edges, num_nodes, budget, self.fixed_edges, self.excluded_edges,
            fixed_idx=fixed_idx, excluded_idx=excluded_idx,
            initial_lambda=self.initial_lambda if not inherit_lambda else initial_lambda,
            step_size=self.step_size, max_iter=max_iter, 
            use_cover_cuts=self.use_cover_cuts, cut_frequency=self.cut_frequency,
            use_bisection=self.use_bisection, verbose=self.verbose,
            shared_graph=MSTNode.global_graph
        )
        self.lagrangian_solver.graph = MSTNode.global_graph

        # Index sets, carried down the tree instead of being re-derived from
        # the edge tuples at every node and at every strong-branching probe
        # (with reduced-cost fixing the excluded set reaches thousands of
        # edges).  The solver derives them when a caller has none to hand --
        # which is only the root, where both are empty.
        self.fixed_idx = (
            fixed_idx if fixed_idx is not None
            else frozenset(self.lagrangian_solver.fixed_edge_indices)
        )
        self.excluded_idx = (
            excluded_idx if excluded_idx is not None
            else frozenset(self.lagrangian_solver.excluded_edge_indices)
        )

        # Correlation-aware widened tunables (empty/None for non-negative corr,
        # so default behaviour is preserved). Applied BEFORE solve() below, and
        # stashed so child nodes inherit the same overrides.
        self.solver_overrides = solver_overrides or {}
        if self.solver_overrides:
            for _k, _v in self.solver_overrides.items():
                setattr(self.lagrangian_solver, _k, _v)

        if MSTNode._solver_pool is None:
            # Only the cut-shaping overrides belong on the probe.  The
            # iteration-budget and primal-repair ones (child_iter_decay,
            # child_min_iter, root_max_iter, enable_primal_repair,
            # use_budget_repair, ...) would silently make every
            # strong-branching probe many times more expensive than the two
            # iterations simulate_branching_bound asks for -- a probe is meant
            # to be a cheap estimate, not a second solve.
            _SB_OVERRIDE_KEYS = {
                "cut_strengthening",
                "max_active_cuts",
                "max_new_cuts_per_node",
                "max_cut_depth",
                "max_mu_depth",
                "mu_step_mode",
                "mu_step_frac",
                "mu_step_decay",
                "mu_cap_frac",
                "mu_increment_cap",
                "mu_init",
                "gamma_mu",
                "cut_phase_frac",
                "lam_phase_frac",
                "lift_cuts",
                "min_new_cut_iters",
                "cut_rank_mode",
                "min_cut_violation_for_add",
                "dead_mu_threshold",
                "use_fast_kruskal",
                # Components that rungs R4 / E switch off.  They were missing,
                # so every probe ran with rank lifting and the exact cut dual
                # ON whatever the run was configured to do -- R4 was only R4
                # in its real node solves, and the exact-dual ablation still
                # priced every probe exactly.
                "rank_lift",
                "exact_cut_dual",
                "exact_cut_rounds",
                "exact_cut_max",
                "dual_seed",
            }
            _sb_overrides = {
                k: v for k, v in self.solver_overrides.items()
                if k in _SB_OVERRIDE_KEYS
            }

            def _factory():
                solver = LagrangianMST(
                    MSTNode.global_edges, self.num_nodes, self.budget,
                    fixed_edges=set(), excluded_edges=set(),
                    initial_lambda=self.initial_lambda,
                    step_size=self.step_size, max_iter=5,
                    use_cover_cuts=self.use_cover_cuts, cut_frequency=self.cut_frequency,
                    use_bisection=False, verbose=False, shared_graph=MSTNode.global_graph
                )
                # The strong-branching solvers were built without the node's
                # solver_overrides, so every simulation ran on library defaults
                # no matter how the run was configured -- `cut_strengthening`
                # included, which meant the attribution ladder's rungs all did
                # the SAME thing inside strong branching and differed only in
                # their real solves.  reset() does not clear these, so setting
                # them once at construction is enough.
                for _k, _v in _sb_overrides.items():
                    setattr(solver, _k, _v)
                solver._is_probe = True

                # A probe is a RANKING device, not a second solve.  With the
                # cut phase at its node setting a two-iteration probe runs
                # 2 + cut_phase_frac*2 dual iterations, so at the shipped
                # setting the two probes behind one branching decision cost
                # more MSTs than the node itself -- for an estimate whose
                # only job is to order a handful of candidate edges.
                solver.cut_phase_frac = float(
                    self.solver_overrides.get("sb_cut_phase_frac", 0.0)
                )
                return solver

            MSTNode._solver_pool = SolverPool(_factory, size=1)
        self._sb_pool = MSTNode._solver_pool

        self.active_cuts = []
        self.cut_multipliers = {}
        if parent_cover_cuts:
            for cut_idx, (cut, rhs) in enumerate(parent_cover_cuts):
                normalized_cut = (
                    cut if isinstance(cut, frozenset)
                    else frozenset(tuple(sorted((u, v))) for u, v in cut)
                )
                new_idx = len(self.active_cuts)
                self.active_cuts.append((normalized_cut, rhs))
                self.cut_multipliers[new_idx] = (
                    parent_cover_multipliers.get(cut_idx, 0.0)
                    if parent_cover_multipliers else 0.0
                )

        # The node solver keeps its own incumbent and starts it at +inf, so
        # without this every node runs with gap = None and alpha pinned to
        # fallback_alpha.  The nodes that generate cuts are exactly the ones
        # whose trees break the budget, so they are the least likely to find an
        # incumbent of their own -- which is why mu never grew and the cover
        # cuts never reached the MST.
        self.lagrangian_solver.incumbent_ub = MSTNode.incumbent_for_step()

        self.local_lower_bound, self.best_upper_bound, self.new_cuts = self.lagrangian_solver.solve(
            # active_cuts already holds (frozenset of normalised edges, rhs);
            # rebuilding each support here re-sorted every edge of every cut
            # on every node, and a lifted support runs to thousands of edges.
            inherited_cuts=self.active_cuts,
            inherited_multipliers=self.cut_multipliers,
            depth=self.depth
        )

        # A child's feasible region is a subset of its parent's, so every lower
        # bound valid at the parent is valid here: the node bound is the better
        # of the two.  Without this the reported bound can move BACKWARDS down a
        # branch, which breaks the one invariant best-first search relies on --
        # that the sequence of popped bounds is non-decreasing.
        #
        # With cuts off it held by accident: the child starts at the parent's
        # best lambda and prices it on the first iteration, and
        # L_child(lambda) >= L_parent(lambda) because the child minimises over
        # fewer trees.  With cuts on the parent's bound also depends on mu, the
        # child re-derives lambda with mu parked at zero, and the parent's
        # (lambda, mu) point is never re-priced -- so the child can report less
        # than its parent.  Measured on n=50, density 0.2, seed 7: 0 of 920
        # children below their parent with cuts off, 24 of 288 with the
        # literature cuts, dropping 1.12 on average.  Those inversions put
        # loose nodes at the head of the queue and cost the search far more
        # than the cuts were winning.
        self.parent_lower_bound = parent_lower_bound

        # The node's OWN dual bound, before the parent clamp below.  Every
        # strong-branching estimate is derived from this node's dual solution
        # (its priced weights, its best_mst_edges) and every probe re-solves
        # from it, so the deltas they produce are commensurate with THIS
        # number, not the clamped one.  Measuring a probe against a clamped
        # bound drives fix/exc deltas negative on the 8-20% of children where
        # the clamp binds, collapsing every branching score to the same floor
        # and feeding zeros into the pseudocost table.
        self.own_lower_bound = self.local_lower_bound
        if (
            parent_lower_bound is not None
            and not math.isnan(parent_lower_bound)
            and self.local_lower_bound < parent_lower_bound
        ):
            self.local_lower_bound = parent_lower_bound

        if self.best_upper_bound < getattr(MSTNode, "global_upper_bound", float("inf")):
            MSTNode.global_upper_bound = self.best_upper_bound
            _inc = getattr(self.lagrangian_solver, "best_feasible_edges", None)
            if _inc:
                MSTNode.global_best_edges = list(_inc)

        # The tree behind best_upper_bound.  mst_edges below is the node's final
        # Lagrangian tree and is usually over budget, so the incumbent solution
        # has to be reported from here instead.
        self.best_feasible_edges = getattr(
            self.lagrangian_solver, "best_feasible_edges", None
        )
        # --- SYNC cuts and multipliers with solver's final state ---
        # self.active_cuts = [
        #     (set(cut), rhs) for (cut, rhs) in self.lagrangian_solver.best_cuts
        # ]
        # self.cut_multipliers = self.lagrangian_solver.best_cut_multipliers_for_best_bound.copy()
        # self.new_cuts = []  # best_cuts already includes surviving cuts
        # -----------------------------------------------------------

        # --- SYNC cuts and multipliers with solver's final state ---
        self.active_cuts = [
            (cut if isinstance(cut, frozenset) else frozenset(cut), int(rhs))
            for (cut, rhs) in (getattr(self.lagrangian_solver, "best_cuts", []) or [])
        ]

        self.cut_multipliers = dict(
            getattr(
                self.lagrangian_solver,
                "best_cut_multipliers_for_best_bound",
                getattr(self.lagrangian_solver, "best_cut_multipliers", {})
            ) or {}
        )

        # Important: once the solver has decided its final pool for this node,
        # do not keep a second parallel "new_cuts" list alive.
        self.new_cuts = []


        # self.mst_edges = [tuple(sorted((u, v))) for u, v in self.lagrangian_solver.last_mst_edges]
        raw_edges = self.lagrangian_solver.last_mst_edges
        if not raw_edges:
            if self.verbose:
                print("Warning: solver returned no MST; treating as empty node.")
            raw_edges = []
            self.lagrangian_solver.last_mst_edges = []

        # _mst_core reads its edges out of edge_list, which is normalised,
        # so this was n-1 redundant two-element sorts per node.
        self.mst_edges = list(raw_edges)
        self._mst_edge_set = frozenset(self.mst_edges)
        self._length_violation = None

        # Cached once: is_feasible() and compute_upper_bound() used to
        # re-derive these from the edge list on every call.
        self.actual_cost, self.actual_length = (
            self.lagrangian_solver.compute_real_weight_length()
        )

        # One strong primal pass at the root.  Everything the search does
        # afterwards is measured against the incumbent -- the pruning test,
        # the Polyak gap and, above all, the reduced-cost fixing, whose
        # threshold is exactly (incumbent - granularity - bound).  The dual
        # iterations only ever produce an incumbent when their own priced
        # tree happens to fit the budget, which at the root leaves it tens of
        # units loose; a parametric-MST sweep costs ~50 Kruskals ONCE and
        # normally lands on or near the optimum.
        # OFF by default.  It does find a near-optimal incumbent in one pass,
        # but the node dual is a Polyak subgradient whose step is sized by the
        # gap to that incumbent: hand it a tight one and alpha collapses, the
        # multiplier stops travelling within the few iterations a node gets,
        # and the child bounds get WORSE.  Measured at n = 60, density 0.2,
        # seed 43693 with cuts off: 60 nodes with the loose incumbent against
        # 2102 with the tight one.  Enable it only together with a step rule
        # that does not depend on the incumbent gap.
        if depth == 0 and getattr(self.lagrangian_solver, "root_primal", False):
            self._root_primal_heuristic()

        # Lagrangian reduced-cost fixing.  Must come last: it re-prices the
        # node at its best dual point, which resets the solver's tree cache.
        self._apply_reduced_cost_fixing()

        super().__init__(self.local_lower_bound)

        if self.verbose:
            print(
                f"Node initialized: lower_bound={self.local_lower_bound}, "
                f"upper_bound={self.best_upper_bound}, lambda={self.lagrangian_solver.best_lambda}, "
                f"fixed={self.fixed_edges}, excluded={self.excluded_edges}"
            )


    def _root_primal_heuristic(self):
        """Budget-feasible incumbent from a parametric MST, at the root only.

        Validity is checked here rather than trusted: the tree must span, be
        acyclic and fit the budget before it is allowed to become the
        incumbent every prune is measured against.
        """
        solver = self.lagrangian_solver

        try:
            w, l, edges = solver.primal_repair_budget()
        except Exception:
            return

        if not edges or math.isnan(w) or math.isinf(w):
            return

        norm = {tuple(sorted(e)) for e in edges}

        if len(norm) != self.num_nodes - 1:
            return

        attrs = solver.edge_attributes
        total_w = 0
        total_l = 0
        parent = list(range(self.num_nodes))

        for e in norm:
            wl = attrs.get(e)
            if wl is None:
                return
            total_w += wl[0]
            total_l += wl[1]

            a, b = e
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            while parent[b] != b:
                parent[b] = parent[parent[b]]
                b = parent[b]
            if a == b:
                return          # not a tree
            parent[b] = a

        if total_l > self.budget:
            return

        if total_w < self.best_upper_bound:
            self.best_upper_bound = float(total_w)
            self.best_feasible_edges = list(norm)
            solver.best_upper_bound = float(total_w)
            solver.best_feasible_edges = list(norm)

        if total_w < MSTNode.global_upper_bound:
            MSTNode.global_upper_bound = float(total_w)
            MSTNode.global_best_edges = list(norm)

    # ------------------------------------------------------------------
    # Shared helpers for the strong-branching-based rules.
    # ------------------------------------------------------------------
    def _forced_decision(self, edges_to_fix, edges_to_exclude):
        """Turn proved one-sided infeasibilities into a single child.

        Returns the (representative edge, child) pair the search expects, with
        child None when the node is proved empty.

        An edge that both probes proved infeasible is a proof that the NODE is
        empty: no completion contains it and none omits it.  The rules used to
        put such an edge into both sets, so create_single_child built a child
        carrying it as fixed AND excluded -- a contradictory node that
        _mst_core still spans (it forces the fixed edges in before consulting
        the exclusions), so the search went on exploring a subtree that
        represents nothing.
        """
        both = set(edges_to_fix) & set(edges_to_exclude)
        MSTNode.forced_decisions += 1

        if both:
            MSTNode.forced_empty += 1
            rep = sorted(both)[0]
            return ([rep], None)

        child = self.create_single_child(edges_to_fix, edges_to_exclude)
        rep = (sorted(edges_to_fix)[0] if edges_to_fix
               else sorted(edges_to_exclude)[0])
        return ([rep], child)

    def _branching_indicator(self, edge, indicator=None):
        """The LR-derived inclusion indicator for `edge`, clamped away from
        0 and 1.

        One accessor for every rule, so the value that normalises a pseudocost
        update is the same value the score is later computed from.  The
        pseudocost update in the search loop used to read
        get_fractional_value while the rules scored from the Dantzig-Wolfe
        indicator, so the rate and the distance it was divided by came from
        two different signals.
        """
        if indicator is None:
            # The indicator this node was scored with, cached by
            # get_branching_candidates, so a later pseudocost update
            # normalises by the same signal the score was built from.
            indicator = getattr(self, "_last_indicator", None)

        f = None
        if indicator is not None:
            f = indicator.get(edge)
        if f is None:
            f = self.get_fractional_value(edge)

        try:
            f = float(f)
        except (TypeError, ValueError):
            return 0.5

        if math.isnan(f):
            return 0.5

        return max(MSTNode.PC_MIN_DIST, min(1.0 - MSTNode.PC_MIN_DIST, f))

    def _node_opt(self, name, default):
        """A node-level knob, overridable through `solver_overrides`.

        The overrides dict is handed to the LagrangianMST, so a knob read off
        the NODE (sb_rank, sb_max_candidates) was unreachable from the
        benchmark harness and silently kept its default -- which is how a
        run configured for indicator ranking quietly measured margin ranking
        instead.
        """
        ov = getattr(self, "solver_overrides", None) or {}

        if name in ov:
            return ov[name]

        return getattr(self, name, default)

    def _rank_for_strong_branching(self, candidates, indicator=None,
                                   margins=None):
        """Order the probe candidates, most uncertain first.

        `margins` ranks by the priced swap margin (smallest first): the
        relaxation is nearly indifferent about a tree edge whose cheapest
        replacement barely costs more, which is the LR-native notion of an
        uncertain variable and uses NO fractional construction.  `indicator`
        ranks by distance from 0.5 on the Dantzig-Wolfe or running-average
        value.  Falling through to get_fractional_value is a last resort: it
        returns a constant 0.5 for every edge whenever the node's tree
        respects the budget, which leaves the order to the tie-break.
        """
        if margins is not None:
            big = float("inf")
            return sorted(candidates, key=lambda e: (margins.get(e, big), e))

        if indicator is not None:
            return sorted(candidates,
                          key=lambda e: (abs(indicator.get(e, 0.5) - 0.5), e))

        return sorted(candidates,
                      key=lambda e: (abs(self.get_fractional_value(e) - 0.5), e))

    def _apply_reduced_cost_fixing(self):
        """Tighten this node's fixings from its own dual solution.

        Any edge whose Lagrangian penalty already carries the node bound past
        the incumbent cannot appear in an improving tree, so it is excluded
        here and, through create_children, in the whole subtree; the mirror
        test fixes tree edges no improving tree can drop.  Both are exact --
        see the derivation on LagrangianMST.reduced_cost_fixing -- and the
        priced tree survives the test by construction, so the node's graph
        stays connected.
        """
        solver = self.lagrangian_solver

        if not getattr(solver, "use_rc_fixing", True):
            return

        ub = getattr(MSTNode, "global_upper_bound", float("inf"))

        if not (ub < float("inf")):
            return

        if math.isnan(self.local_lower_bound) or math.isinf(self.local_lower_bound):
            return

        try:
            ex_idx, fix_idx, _lb = solver.reduced_cost_fixing(
                ub,
                granularity=float(getattr(solver, "objective_granularity", 1.0)),
                want_fix=bool(getattr(solver, "rc_fix_edges", True)),
            )
        except Exception:
            return

        if ex_idx is None and fix_idx is None:
            return

        edge_list = solver.edge_list
        edge_indices = solver.edge_indices
        MSTNode.rc_nodes += 1

        if ex_idx is not None and ex_idx.size:
            add = {edge_list[i] for i in ex_idx.tolist()}
            add -= self.fixed_edges
            add -= self.excluded_edges

            if add:
                # The edge set and the index set must stay in step, so both
                # are derived from the same filtered `add`.
                self.excluded_edges = self.excluded_edges | add
                self.excluded_idx = self.excluded_idx | {edge_indices[e] for e in add}
                MSTNode.rc_edges_excluded += len(add)

        if fix_idx is not None and fix_idx.size:
            add = {edge_list[i] for i in fix_idx.tolist()}
            add -= self.fixed_edges
            add -= self.excluded_edges

            if add:
                self.fixed_edges = self.fixed_edges | add
                self.fixed_idx = self.fixed_idx | {edge_indices[e] for e in add}
                MSTNode.rc_edges_fixed += len(add)

    def __lt__(self, other):
        return self.local_lower_bound < other.local_lower_bound

    def is_child_likely_feasible(self):
        """
        Necessary condition for the node to contain a budget-feasible spanning
        tree: the fixed-in length plus the cheapest conceivable completion must
        fit the budget.  Returning False prunes the node, so the estimate has to
        stay a genuine LOWER bound on any completion, or feasible subtrees are
        lost.

        A spanning tree has exactly num_nodes - 1 edges, so with num_fixed_edges
        already fixed in, exactly num_nodes - 1 - num_fixed_edges remain.

        The shortest admissible edge is read off the instance-wide length
        order instead of sweeping E: the first entry that this node has not
        fixed or excluded is it, and both sets are small, so the scan almost
        always stops on the first entry.  The sweep was 14% of a whole run at
        n = 400 -- it ran twice per node over all 16k edges.
        """
        solver = self.lagrangian_solver
        attrs = solver.edge_attributes

        fixed_length = sum(attrs[e][1] for e in self.fixed_edges)
        edges_needed = self.num_nodes - 1 - len(self.fixed_edges)

        if edges_needed < 0:
            return False

        if edges_needed == 0:
            return fixed_length <= self.budget

        excluded = self.excluded_edges
        fixed = self.fixed_edges
        min_edge_length = None

        for e, l in solver._len_sorted_edges:
            if e in excluded or e in fixed:
                continue
            min_edge_length = l
            break

        if min_edge_length is None:
            return False

        estimated_length = fixed_length + edges_needed * min_edge_length
        return estimated_length <= self.budget

    def create_children(self, branched_edge):
        """
        Branch on `branched_edge` and build children with cover cuts inherited correctly.

        Correctness per cover cut (S, rhs):
        - S_fixed = S ∩ Fixed_child
        - S_excl  = S ∩ Excluded_child
        - S_free  = S \\ (Fixed_child ∪ Excluded_child)
        - rhs'    = rhs - |S_fixed|
        - If rhs' < 0          -> child infeasible (prune)
        - If |S_free| <= rhs'  -> cut redundant (drop)
        - Else pass (S_free, rhs') to child

        Multipliers are remapped 1:1 (no damping/caps).
        We additionally LIMIT the number of cuts passed to each child
        to at most `max_child_cuts`, keeping the strongest ones.
        """
        import numpy as np  # (note: currently unused, safe to remove if you like)
        # how many cuts to keep per child (you can tune this)
        max_child_cuts = getattr(self, "max_child_cuts", 25)

        # --- normalize branched edge ---
        u, v = branched_edge
        normalized_edge = (u, v) if u <= v else (v, u)
        new_branched_edges = self.branched_edges | {normalized_edge}

        # --- robustly merge & normalize active_cuts + new_cuts (accept pairs or edge indices) ---
        solver = self.lagrangian_solver
        edge_indices = solver.edge_indices                    # {(u,v): idx}
        known_edges = set(edge_indices.keys())
        idx_to_edge = getattr(solver, "idx_to_edge", None)
        if idx_to_edge is None:
            idx_to_edge = {j: e for e, j in edge_indices.items()}
            solver.idx_to_edge = idx_to_edge

        def _norm_edge(e):
            if not (isinstance(e, tuple) and len(e) == 2):
                return None
            a, b = e
            t = (a, b) if a <= b else (b, a)
            return t if t in known_edges else None

        def _iter_edges_any(cut_like):
            # single (u,v)
            if isinstance(cut_like, tuple) and len(cut_like) == 2:
                e = _norm_edge(cut_like)
                if e is not None:
                    yield e
                return
            # single index
            if isinstance(cut_like, int):
                e_raw = idx_to_edge.get(int(cut_like))
                e = _norm_edge(e_raw)
                if e is not None:
                    yield e
                return
            # iterable
            try:
                for item in cut_like:
                    if isinstance(item, int):
                        e_raw = idx_to_edge.get(int(item))
                        e = _norm_edge(e_raw)
                    elif isinstance(item, tuple) and len(item) == 2:
                        e = _norm_edge(item)
                    elif isinstance(item, (list, set, frozenset)) and len(item) == 2:
                        a, b = tuple(item)
                        e = _norm_edge((a, b))
                    else:
                        e = None
                    if e is not None:
                        yield e
            except TypeError:
                return

        def _norm_pair(pair):
            cut_like, rhs_like = pair
            return (set(_iter_edges_any(cut_like)), int(rhs_like))


        # 1) Build merged cuts.  Both lists already hold
        # (frozenset of normalised edges, rhs); _norm_pair is kept for
        # anything handed in from outside, but running it over the pool
        # rebuilt every support edge by edge, twice per node.
        def _keep(p):
            cut_like, rhs_like = p
            if isinstance(cut_like, frozenset):
                return (cut_like, int(rhs_like))
            return _norm_pair(p)

        all_cuts = [_keep(p) for p in (self.active_cuts or [])]
        all_cuts.extend(_keep(p) for p in (getattr(self, "new_cuts", []) or []))

        # 2) Build a map from support -> μ using solver.best_cuts
        best_cuts = getattr(solver, "best_cuts", []) or []
        best_mu   = getattr(solver, "best_cut_multipliers_for_best_bound", {}) or {}

        support_to_mu = {}
        for i, (cut_i, rhs_i) in enumerate(best_cuts):
            key = (frozenset(cut_i), rhs_i)
            support_to_mu[key] = float(best_mu.get(i, 0.0))

        # 3) Assign multipliers to all_cuts by support, default small μ for unseen cuts
        current_multipliers = {}
        for idx, (cut, rhs) in enumerate(all_cuts):
            key = (frozenset(cut), rhs)
            current_multipliers[idx] = support_to_mu.get(key, 0.0)

        # Quick prune for fixed child
        F_fixed = self.fixed_edges | {normalized_edge}
        _bi = edge_indices.get(normalized_edge)
        F_fixed_idx = self.fixed_idx | ({_bi} if _bi is not None else frozenset())
        must_prune_fixed = any((len(cut_set & F_fixed) > rhs) for (cut_set, rhs) in all_cuts)

        T_parent = set(self.mst_edges or [])
        
        def _project_and_remap_for_child(fixed_child_edges, excluded_child_edges):
            """Project the pool onto a child, and read off what the projection
            forces.

            A cover (S, rhs) whose reduced right-hand side has reached zero
            says that NONE of its remaining free edges can be in a feasible
            tree of that child -- rhs of them are already fixed in and at most
            rhs may be used.  Excluding them all is exact, and it is the point
            at which a lifted support pays: the seed cover forces out the
            handful of edges it was built from, while the lifted one forces
            out every admissible edge long enough to have been lifted into it.
            The scan is repeated until it reaches a fixpoint, since one
            cover's forced exclusions can drive another's support down to its
            own right-hand side.
            """
            infeasible = False
            projected = {}
            forced_out = set()

            for old_i, (S, rhs) in enumerate(all_cuts):
                # Every support edge is in the instance by construction, so
                # the old S_known rebuild only copied the set.  The branch
                # below also avoids copying a support the child does not
                # touch, which is the common case and the expensive one.
                S_fixed = S & fixed_child_edges
                S_excl = S & excluded_child_edges

                if S_fixed or S_excl or forced_out:
                    S_free = S - fixed_child_edges - excluded_child_edges - forced_out
                else:
                    S_free = S

                rhs_prime = int(rhs) - len(S_fixed)
                if rhs_prime < 0:
                    infeasible = True
                    break
                if len(S_free) <= rhs_prime:
                    continue

                if rhs_prime == 0:
                    # Saturated: every remaining free edge of the support is
                    # forced out, and the cover itself becomes redundant.
                    forced_out |= S_free
                    continue

                # approximate usefulness using current parent's tree
                lhs_est = len(T_parent & S_free)
                viol_est = lhs_est - rhs_prime

                key = S_free if isinstance(S_free, frozenset) else frozenset(S_free)
                mu_old = float(current_multipliers.get(old_i, 0.0))
                prev = projected.get(key)

                if (
                    prev is None
                    or rhs_prime < prev["rhs"]
                    or (rhs_prime == prev["rhs"] and viol_est > prev["viol"])
                ):
                    projected[key] = {
                        "rhs": rhs_prime,
                        "mu": mu_old,
                        "viol": viol_est,
                    }

            if infeasible:
                return None, None, True, None

            if forced_out:
                # A cover that was saturated may have driven another's
                # support down to ITS right-hand side, so the pass is
                # repeated until nothing more is forced.  The recursion is
                # bounded: every pass strictly grows `forced_out`.
                kept, mu, bad, more = _project_and_remap_for_child(
                    fixed_child_edges,
                    excluded_child_edges | forced_out,
                )
                if bad:
                    return None, None, True, None
                return kept, mu, False, forced_out | (more or set())

            # No tuple(sorted(support)) tie-break: `projected` is a dict, so
            # it iterates in insertion order, which follows all_cuts and is
            # therefore already deterministic -- and sorting a lifted support
            # of several thousand edges, once per distinct projection and
            # twice per node, was one of the most expensive lines in the
            # strengthened rungs.
            ordered = sorted(
                projected.items(),
                key=lambda kv: (-kv[1]["viol"], len(kv[0]), kv[1]["rhs"])
            )

            if len(ordered) > max_child_cuts:
                ordered = ordered[:max_child_cuts]

            kept_cuts, kept_mu = [], {}
            for new_idx, (sfree_key, info) in enumerate(ordered):
                kept_cuts.append((sfree_key, int(info["rhs"])))
                kept_mu[new_idx] = float(info["mu"])

            return kept_cuts, kept_mu, False, None
        
        
        # ---- children ----
        fixed_child = None
        if not must_prune_fixed:
            (kept_cuts_fixed, kept_mu_fixed, prune_fixed,
             forced_fixed) = _project_and_remap_for_child(
                fixed_child_edges=F_fixed,
                excluded_child_edges=self.excluded_edges,
            )
            child_excl_for_fixed = self.excluded_edges
            child_excl_idx_for_fixed = self.excluded_idx
            if forced_fixed:
                child_excl_for_fixed = self.excluded_edges | forced_fixed
                child_excl_idx_for_fixed = self.excluded_idx | {
                    edge_indices[e] for e in forced_fixed if e in edge_indices
                }
                MSTNode.cut_forced_exclusions += len(forced_fixed)
            if not prune_fixed:
                fixed_child = MSTNode(
                    self.edges, self.num_nodes, self.budget,
                    F_fixed, child_excl_for_fixed, new_branched_edges,
                    initial_lambda=solver.best_lambda if self.inherit_lambda else 0.05,
                    inherit_lambda=self.inherit_lambda, branching_rule=self.branching_rule,
                    step_size=solver.step_size if self.inherit_step_size else 0.001,
                    inherit_step_size=self.inherit_step_size, use_cover_cuts=self.use_cover_cuts,
                    cut_frequency=self.cut_frequency, node_cut_frequency=self.node_cut_frequency,
                    parent_cover_cuts=kept_cuts_fixed,
                    parent_cover_multipliers=kept_mu_fixed,
                    fixed_idx=F_fixed_idx, excluded_idx=child_excl_idx_for_fixed,
                    use_bisection=self.use_bisection, max_iter=solver.max_iter,
                    verbose=self.verbose, depth=self.depth + 1,
                    pseudocosts_up=self.pseudocosts_up, pseudocosts_down=self.pseudocosts_down,
                    counts_up=self.counts_up, counts_down=self.counts_down,
                    reliability_eta=self.reliability_eta, lookahead_lambda=self.lookahead_lambda,
                    solver_overrides=self.solver_overrides,
                    parent_lower_bound=self.local_lower_bound
                )

        F_excluded = self.excluded_edges | {normalized_edge}
        F_excluded_idx = self.excluded_idx | ({_bi} if _bi is not None else frozenset())
        (kept_cuts_excl, kept_mu_excl, _prune_excl,
         forced_excl) = _project_and_remap_for_child(
            fixed_child_edges=self.fixed_edges,
            excluded_child_edges=F_excluded,
        )
        if _prune_excl:
            return fixed_child, None
        if forced_excl:
            F_excluded = F_excluded | forced_excl
            F_excluded_idx = F_excluded_idx | {
                edge_indices[e] for e in forced_excl if e in edge_indices
            }
            MSTNode.cut_forced_exclusions += len(forced_excl)
        excluded_child = MSTNode(
            self.edges, self.num_nodes, self.budget,
            self.fixed_edges, F_excluded, new_branched_edges,
            initial_lambda=solver.best_lambda if self.inherit_lambda else 0.05,
            inherit_lambda=self.inherit_lambda, branching_rule=self.branching_rule,
            step_size=solver.step_size if self.inherit_step_size else 0.001,
            inherit_step_size=self.inherit_step_size, use_cover_cuts=self.use_cover_cuts,
            cut_frequency=self.cut_frequency, node_cut_frequency=self.node_cut_frequency,
            parent_cover_cuts=kept_cuts_excl,
            parent_cover_multipliers=kept_mu_excl,
            fixed_idx=self.fixed_idx, excluded_idx=F_excluded_idx,
            use_bisection=self.use_bisection, max_iter=solver.max_iter,
            verbose=self.verbose, depth=self.depth + 1,
            pseudocosts_up=self.pseudocosts_up, pseudocosts_down=self.pseudocosts_down,
            counts_up=self.counts_up, counts_down=self.counts_down,
            reliability_eta=self.reliability_eta, lookahead_lambda=self.lookahead_lambda,
            solver_overrides=self.solver_overrides,
            parent_lower_bound=self.local_lower_bound
        )

        return fixed_child, excluded_child

  

    def is_feasible(self):
        real_length = self.actual_length
        if real_length > self.budget:
            return False, "MST length exceeds budget"

        # Kruskal only ever joins two distinct components, so what it returns
        # is a forest; a forest with num_nodes - 1 edges on num_nodes vertices
        # is a spanning tree.  The size test therefore already carries the
        # connectivity test, and building a networkx graph per node to re-run
        # is_connected was pure overhead.
        if len(self.mst_edges) != self.num_nodes - 1:
            return False, "MST does not include all nodes"

        return True, "MST is feasible"

    def compute_upper_bound(self):
        return self.actual_cost
    

    def get_branching_candidates(self):


        if self.branching_rule in ["strong_branching", "strong_branching_all", "sb_fractional"]:
            # The indicator ranks the candidates for EVERY strong-branching
            # variant, not only the fractional one.
            #
            # Only sb_fractional used to build it, so strong_branching and
            # strong_branching_all fell back to get_fractional_value -- which
            # returns a constant 0.5 for every edge whenever the node's
            # Lagrangian tree respects the budget.  Measured at n = 300,
            # density 0.05: the indicator was absent at 100% of ranking calls
            # and every candidate tied at half of them, so the sb_max_candidates
            # edges actually probed were the lexicographically smallest of
            # about ninety.  That is not strong branching, and it is the
            # reason the MST-candidate variants trailed the fractional one.
            #
            # The variants still differ where the paper says they differ -- in
            # the CANDIDATE SET (tree edges, all free edges, or the strictly
            # fractional support) -- but they now all rank that set by the same
            # LR fractionality signal.
            # Which signal orders the probe candidates.
            #
            # "auto" keeps the comparison in the paper honest: the
            # fractional variant ranks on the indicator it is defined by,
            # while the MST-candidate variants rank on the swap margin, which
            # is LR-native and independent of Section 5.  Giving the MST
            # variants the indicator instead ("indicator") is a legitimate
            # measurement -- it isolates how much the fractional construction
            # is worth as a RANKING, separately from how much it is worth as
            # a candidate SET -- but it is not the baseline.
            sb_rank = str(self._node_opt("sb_rank", "auto")).lower()

            if sb_rank == "auto":
                sb_rank = ("indicator"
                           if self.branching_rule == "sb_fractional"
                           else "margin")

            sb_margins = None
            normalized_edge_weights = None

            if sb_rank == "margin":
                sb_margins = self.lagrangian_solver.branching_margins()

            if sb_rank != "margin" or self.branching_rule == "sb_fractional":
                normalized_edge_weights = (
                    self.lagrangian_solver.compute_fractional_solution(self))
                self._last_indicator = normalized_edge_weights

            # Select candidate edges based on branching rule
            if self.branching_rule == "strong_branching_all":
                candidate_edges = [
                    (u, v) for u, v, _, _ in self.edges
                    if (u, v) not in self.fixed_edges and
                    (u, v) not in self.excluded_edges and
                    (u, v) not in self.branched_edges
                ]
            elif self.branching_rule == "strong_branching":
                mst_edges = [tuple(sorted((u, v))) for u, v in self.lagrangian_solver.best_mst_edges]
                candidate_edges = [
                    e for e in mst_edges
                    if e not in self.fixed_edges and
                    e not in self.excluded_edges and
                    e not in self.branched_edges

                ]
            else:  # sb_fractional
                shor_primal_solution = normalized_edge_weights
                if shor_primal_solution is None:
                    # No fractional info available -> fall back to MST edges
                    mst_edges = [tuple(sorted((u, v))) for u, v in self.lagrangian_solver.best_mst_edges]
                    candidate_edges = [
                        e for e in mst_edges
                        if e not in self.fixed_edges
                        and e not in self.excluded_edges
                        and e not in self.branched_edges
                    ]
                else:
                    tolerance = 1e-6
                    candidate_edges = [
                        e for e in normalized_edge_weights
                        if e not in self.fixed_edges and
                        e not in self.excluded_edges and
                        e not in self.branched_edges and
                        normalized_edge_weights[e] > tolerance and
                        normalized_edge_weights[e] < 1.0 - tolerance
                    ]
                    if not candidate_edges:
                        mst_edges = [tuple(sorted((u, v))) for u, v in self.lagrangian_solver.best_mst_edges]
                        candidate_edges = [
                        e for e in mst_edges
                        if e not in self.fixed_edges and
                        e not in self.excluded_edges and
                        e not in self.branched_edges
                        ]

            if not candidate_edges:
                if self.verbose:
                    print(f"No {self.branching_rule} candidates available")
                return None

            if self.verbose:
                print(f"Node {id(self)}: {self.branching_rule} evaluating {len(candidate_edges)} edges: {candidate_edges}")

            max_sb_candidates = int(self._node_opt("sb_max_candidates", 3))
            candidate_edges = self._rank_for_strong_branching(
                candidate_edges,
                normalized_edge_weights if sb_margins is None else None,
                sb_margins,
            )[:max_sb_candidates]
            # Collect edges that lead to pruning
            edges_to_fix = set()
            edges_to_exclude = set()
            best_edge = None
            best_score = -float('inf')
            scores = []
            for edge in candidate_edges:                
                score,_,_, fix_infeasible, exclude_infeasible = self.calculate_strong_branching_score(edge)
                scores.append((edge, score))
                if fix_infeasible:
                    edges_to_exclude.add(edge)
                if exclude_infeasible:
                    edges_to_fix.add(edge)
                if not (fix_infeasible or exclude_infeasible):
                    if score > best_score:
                        best_score = score
                        best_edge = edge

            if edges_to_fix or edges_to_exclude:
                if self.verbose:
                    print(f"Creating single child with fixed edges: {edges_to_fix}, excluded edges: {edges_to_exclude}")
                return self._forced_decision(edges_to_fix, edges_to_exclude)

            # No pruning edges found, proceed with standard strong branching
            if not best_edge:
                if self.verbose:
                    print(f"No viable branching edge found after scoring")
                return None

            scores.sort(key=lambda x: x[1], reverse=True)
            if self.verbose:
                print(f"Selected best edge {best_edge} with score {best_score}")
            return [best_edge]

        elif self.branching_rule == "strong_branching_sim":
            mst_edges = [tuple(sorted((u, v))) for u, v in self.lagrangian_solver.best_mst_edges]
            candidate_edges = [
                e for e in mst_edges
                if e not in self.fixed_edges and
                e not in self.excluded_edges and
                e not in self.branched_edges
            ]
            if not candidate_edges:
                if self.verbose:
                    print("No strong branching sim candidates available")
                return None

            if self.verbose:
                print(f"Node {id(self)}: Strong branching sim evaluating {len(candidate_edges)} MST edges: {candidate_edges}")

            # Rank and cap exactly as the other strong-branching variants do.
            # This rule evaluated EVERY tree edge, and each evaluation builds a
            # networkx graph and searches it for a cycle, so a node cost O(n)
            # swap estimates x O(n) each -- 26s of shifted-geomean CPU at
            # n = 300 against 2-3s for every other rule, for the worst node
            # counts of the lot.  The estimates are a surrogate for a probe;
            # spending a quadratic sweep on them defeats the point.
            _sim_rank = str(self._node_opt("sb_rank", "auto")).lower()
            _sim_ind = None
            _sim_mg = None

            if _sim_rank in ("auto", "margin"):
                _sim_mg = self.lagrangian_solver.branching_margins()
            else:
                _sim_ind = self.lagrangian_solver.compute_fractional_solution(self)
                self._last_indicator = _sim_ind

            candidate_edges = self._rank_for_strong_branching(
                candidate_edges, _sim_ind, _sim_mg
            )[:int(self._node_opt("sb_max_candidates", 3))]

            best_edge = None
            best_score = -float('inf')

            for edge in candidate_edges:
                u, v = edge
                fixed_lower_bound = self.simulate_fix_edge(u, v)
                excluded_lower_bound = self.simulate_exclude_edge(u, v)

                # +inf from these swap estimates means "no estimate available"
                # -- every edge on the cycle is fixed, or the cut has no
                # replacement -- not "infeasible".  Scoring it +inf put the
                # edge at the top of the ranking on the strength of missing
                # information; it is now scored as no gain instead.
                if fixed_lower_bound == float('inf') or excluded_lower_bound == float('inf'):
                    score = 0.0
                else:
                    fix_score = fixed_lower_bound - self.own_lower_bound
                    exc_score = excluded_lower_bound - self.own_lower_bound
                    score = max(fix_score, 1e-6) * max(exc_score, 1e-6)

                if self.verbose:
                    print(f"Edge {edge}: Score {score}")

                if score > best_score:
                    best_score = score
                    best_edge = edge

            return [best_edge] if best_edge is not None else None

        elif self.branching_rule == "most_fractional":
            # shor_primal_solution = self.lagrangian_solver.compute_weighted_average_solution()
            shor_primal_solution = self.lagrangian_solver.compute_fractional_solution(self)
            self._last_indicator = shor_primal_solution


            if shor_primal_solution is None:
                candidates = [
                    tuple(sorted((u, v))) for u, v in self.lagrangian_solver.best_mst_edges
                    if tuple(sorted((u, v))) not in self.fixed_edges and
                    tuple(sorted((u, v))) not in self.excluded_edges and
                    tuple(sorted((u, v))) not in self.branched_edges
                ]

                if not candidates:
                    return None
                return [candidates[0]]

            # Candidate policy (iii): strictly fractional indicators only.
            #
            # The filter had been commented out, so every edge of the
            # Dantzig-Wolfe support was a candidate, including the ones the
            # master gives weight exactly one.  Such an edge lies in every
            # weighted tree, so fixing it in is close to a null move: the
            # include child reproduces the parent's relaxation and the search
            # spends a level to learn nothing.  The unfiltered set is kept as
            # a fallback for the case where the master returns an integral
            # point and nothing is strictly fractional.
            tol = 1e-6
            candidates = [
                e for e in shor_primal_solution
                if e not in self.fixed_edges
                and e not in self.excluded_edges
                and e not in self.branched_edges
                and tol < shor_primal_solution[e] < 1.0 - tol
            ]

            if not candidates:
                candidates = [
                    e for e in shor_primal_solution
                    if e not in self.fixed_edges
                    and e not in self.excluded_edges
                    and e not in self.branched_edges
                ]

            if not candidates:
                mst_edges = [tuple(sorted((u, v))) for u, v in self.lagrangian_solver.best_mst_edges]
                candidates = [
                    e for e in mst_edges
                    if e not in self.fixed_edges
                    and e not in self.excluded_edges
                    and e not in self.branched_edges
                ]

            branching_scores = []
            for e in candidates:
                w = shor_primal_solution.get(e, 0)
                distance_score = -abs(w - 0.5)
                branching_scores.append((e, distance_score))

            branching_scores.sort(key=lambda x: x[1], reverse=True)

            return [branching_scores[0][0]] if branching_scores else None


        
        elif self.branching_rule == "random_fractional":
            # shor_primal_solution = self.lagrangian_solver.compute_weighted_average_solution()
            shor_primal_solution = self.lagrangian_solver.compute_fractional_solution(self)
            self._last_indicator = shor_primal_solution
            if shor_primal_solution is None:
                candidates = [
                    tuple(sorted((u, v))) for u, v in self.lagrangian_solver.best_mst_edges
                    if tuple(sorted((u, v))) not in self.fixed_edges and
                    tuple(sorted((u, v))) not in self.excluded_edges and
                    tuple(sorted((u, v))) not in self.branched_edges
                ]
                if not candidates:
                    return None
                return [candidates[0]]

            normalized_edge_weights = shor_primal_solution
            candidates = [
                e for e in shor_primal_solution
                if e not in self.fixed_edges and
                e not in self.excluded_edges and
                e not in self.branched_edges and
                abs(normalized_edge_weights[e]) > 1e-6 and
                abs(normalized_edge_weights[e] - 1.0) > 1e-6
            ]
                            
            if not candidates:
                candidates = [
                e for e in shor_primal_solution
                if e not in self.fixed_edges and
                e not in self.excluded_edges and
                e not in self.branched_edges
                ]

            return candidates if candidates else None            

        
        elif self.branching_rule == "most_violated":
            candidate_edges = sorted(
                [(u, v, w, l) for u, v, w, l in self.edges if (u, v) not in self.fixed_edges and (u, v) not in self.excluded_edges],
                key=lambda x: x[2] / x[3],
                reverse=True,
            )
            # Return the TOP edge, not the whole sorted list.  The caller
            # picks uniformly at random from whatever comes back, so returning
            # everything threw the ordering away and made this rule a uniform
            # draw over all free edges -- which is what it measured as
            # (165 nodes, against 140 for random over the tree edges).
            #
            # It also never filtered out already-branched edges, and the key
            # w/l is a static edge property with no dependence on the node, so
            # this rule re-ranks identically at every node of the search.
            candidate_edges = [
                (u, v) for u, v, _, _ in candidate_edges
                if (u, v) not in self.branched_edges
            ]

            return [candidate_edges[0]] if candidate_edges else None

        elif self.branching_rule == "random_mst":
            candidate_edges = [e for e in self.mst_edges if e not in self.fixed_edges and
                            e not in self.excluded_edges and
                            e not in self.branched_edges]
            return candidate_edges if candidate_edges else None
            # return [random.choice(candidate_edges)] if candidate_edges else None


        elif self.branching_rule == "random_all":
            candidate_edges = [(u, v) for u, v, _, _ in self.edges if (u, v) not in self.fixed_edges and
                            (u, v) not in self.excluded_edges and
                            (u, v) not in self.branched_edges]
            return candidate_edges if candidate_edges else None
        
        
        elif self.branching_rule == "reliability":
            # 1) Candidate edges: current MST edges (no fractional needed)
            # mst_edges = [tuple(sorted((u, v))) for u, v in self.lagrangian_solver.best_mst_edges]
            # candidate_edges = [
            #     e for e in mst_edges
            #     if e not in self.fixed_edges
            #     and e not in self.excluded_edges
            #     and e not in self.branched_edges
            # ]

            # if not candidate_edges:
            #     return None

            # # 2) Split into unhistoried vs reliable, based on TOTAL observations
            # unhistoried = []
            # reliable_candidates = []
            # for e in candidate_edges:
            #     cu = self.counts_up.get(e, 0)
            #     cd = self.counts_down.get(e, 0)
            #     total = cu + cd
            #     if total >= self.reliability_eta:
            #         reliable_candidates.append(e)
            #     else:
            #         unhistoried.append(e)

            # # 3) Adaptive lookahead: how many edges to strong-branch
            # if self.depth < 5:
            #     max_sb_evals = self.lookahead_lambda
            # else:
            #     max_sb_evals = max(2, self.lookahead_lambda - 1)

            # # You can optionally order unhistoried by some heuristic, e.g. by edge weight/length.
            # # For now, we just take them as they come from MST.
            # unhistoried = unhistoried[:max_sb_evals]
            # 1) Fractional solution for prioritization
            # shor_primal_solution = self.lagrangian_solver.compute_weighted_average_solution()
            shor_primal_solution = self.lagrangian_solver.compute_fractional_solution(self)
            self._last_indicator = shor_primal_solution


            candidate_edges = []
            if shor_primal_solution is not None:
                tolerance = 1e-6
                candidate_edges = [
                    e for e in shor_primal_solution
                    if e not in self.fixed_edges
                    and e not in self.excluded_edges
                    and e not in self.branched_edges
                    and shor_primal_solution[e] > tolerance
                    and shor_primal_solution[e] < 1.0 - tolerance
                ]
                # most fractional first
                candidate_edges.sort(key=lambda e: abs(shor_primal_solution.get(e, 0.5) - 0.5))

            if shor_primal_solution is None or not candidate_edges:
                mst_edges = [tuple(sorted((u, v))) for u, v in self.lagrangian_solver.best_mst_edges]
                candidate_edges = [
                    e for e in mst_edges
                    if e not in self.fixed_edges
                    and e not in self.excluded_edges
                    and e not in self.branched_edges
                ]

            if not candidate_edges:
                return None

            # 2) Split into unhistoried vs reliable.
            #
            # Reliability is PER DIRECTION: an edge counts as reliable only
            # once both the include and the exclude side have been observed
            # eta times.  The test used to be on the SUM, so an edge seen
            # three times on one side and never on the other was declared
            # reliable and then scored with a zero pseudocost on the unseen
            # side -- which pushed it to the bottom of the ranking and denied
            # it the one probe that would have filled the gap in.
            _eta = max(1, int(self._node_opt("reliability_eta",
                                             self.reliability_eta)))
            unhistoried = []
            reliable_candidates = []
            for e in candidate_edges:
                cu = self.counts_up.get(e, 0)
                cd = self.counts_down.get(e, 0)

                if min(cu, cd) >= _eta:
                    reliable_candidates.append(e)
                else:
                    unhistoried.append(e)

            # 3) Adaptive lookahead: how many edges to strong-branch
            # The probe budget: how deep the expensive phase runs, how many
            # candidates it probes there, and how many below it.  These were
            # three literals, which made the probe cost of reliability and
            # hybrid impossible to tune from the harness.  Defaults are the
            # old literals, so an unset run is unchanged.
            if self.depth < int(self._node_opt("sb_depth_gate", 5)):
                max_sb_evals = int(
                    self._node_opt("lookahead_lambda", self.lookahead_lambda))
                # max_sb_evals = 1

            else:
                # max_sb_evals = max(2, self.lookahead_lambda - 1)
                max_sb_evals = int(self._node_opt("sb_deep_evals", 1))


            # Strong branching only on the most fractional unhistoried edges
            if shor_primal_solution is not None:
                unhistoried.sort(key=lambda e: abs(shor_primal_solution.get(e, 0.5) - 0.5))

            else:
                unhistoried.sort(key=lambda e: abs(self.get_fractional_value(e) - 0.5))


            unhistoried = unhistoried[:max_sb_evals]
            edges_to_fix = set()
            edges_to_exclude = set()
            scores = []

            # 4) Strong branching on unhistoried edges (also updates pseudocosts)
            for edge in unhistoried:
                # Strong branching returns LB improvements for fix/exclude
                sb_score, fix_delta, exc_delta, fix_inf, exc_inf = self.calculate_strong_branching_score(edge)
                if self.verbose:
                    print("unhistoriedscore", edge, sb_score)

                # Get current counts
                count_up = self.counts_up.get(edge, 0)
                count_down = self.counts_down.get(edge, 0)

                # Learning rate (EMA over LB gains)
                if count_up == 0 and count_down == 0:
                    alpha = 0.5
                elif count_up < 3 or count_down < 3:
                    alpha = 0.3
                else:
                    alpha = 0.1

                # Pseudocosts are stored PER UNIT of the branching distance,
                # as in Benichou et al. and as Section 5.4 defines them, so
                # that the score below can predict a bound change by
                # multiplying the stored rate back by the distance.
                #
                # The probe used to store the raw improvement and the score
                # then multiplied by the distance again, applying the same
                # factor twice and giving these dictionaries a different
                # meaning from the one the pseudocost rule writes into them.
                f_e = self._branching_indicator(edge, shor_primal_solution)
                dist_up = max(1.0 - f_e, MSTNode.PC_MIN_DIST)
                dist_dn = max(f_e, MSTNode.PC_MIN_DIST)

                if not fix_inf:
                    new_pc_up = max(0.0, fix_delta) / dist_up
                    old = self.pseudocosts_up.get(edge, None)
                    if old is None or math.isnan(old) or math.isinf(old):
                        self.pseudocosts_up[edge] = new_pc_up
                    else:
                        self.pseudocosts_up[edge] = (1 - alpha) * old + alpha * new_pc_up
                    self.counts_up[edge] = count_up + 1
                    MSTNode.note_pseudocost(True, new_pc_up)

                if not exc_inf:
                    new_pc_down = max(0.0, exc_delta) / dist_dn
                    old = self.pseudocosts_down.get(edge, None)
                    if old is None or math.isnan(old) or math.isinf(old):
                        self.pseudocosts_down[edge] = new_pc_down
                    else:
                        self.pseudocosts_down[edge] = (1 - alpha) * old + alpha * new_pc_down
                    self.counts_down[edge] = count_down + 1
                    MSTNode.note_pseudocost(False, new_pc_down)

                # Store SB score if both sides feasible
                if not fix_inf and not exc_inf:
                    scores.append((sb_score, edge, False, False))
                else:
                    if fix_inf:
                        edges_to_exclude.add(edge)
                    if exc_inf:
                        edges_to_fix.add(edge)

            # 5) Pseudocost scoring for reliable edges (no fractional x)
            # for edge in reliable_candidates:
            #     pc_up = self.pseudocosts_up.get(edge, 0.0)
            #     pc_down = self.pseudocosts_down.get(edge, 0.0)

            #     cu = self.counts_up.get(edge, 0)
            #     cd = self.counts_down.get(edge, 0)
            #     confidence_up = min(1.0, cu / (2 * self.reliability_eta))
            #     confidence_down = min(1.0, cd / (2 * self.reliability_eta))
            #     confidence = 0.5 * (confidence_up + confidence_down)

            #     gain_up = max(pc_up, 0.0)
            #     gain_down = max(pc_down, 0.0)

            #     score = max(gain_up, 1e-6) * max(gain_down, 1e-6)
            #     score *= (0.9 + 0.1 * confidence)

            #     if self.verbose:
            #         print("reliablescore", edge, score)

            #     scores.append((score, edge, False, False))
            for edge in reliable_candidates:
                f = self._branching_indicator(edge, shor_primal_solution)

                pc_up = max(0.0, self.pseudocosts_up.get(edge, 0.0))
                pc_down = max(0.0, self.pseudocosts_down.get(edge, 0.0))

                # distance-scaled predicted gains
                delta_up = pc_up * (1.0 - f)   # include/fix-to-1 move
                delta_down = pc_down * f       # exclude/fix-to-0 move

                cu = self.counts_up.get(edge, 0)
                cd = self.counts_down.get(edge, 0)
                confidence_up = min(1.0, cu / (2 * _eta))
                confidence_down = min(1.0, cd / (2 * _eta))
                confidence = 0.5 * (confidence_up + confidence_down)

                score = max(delta_up, 1e-6) * max(delta_down, 1e-6)
                score *= (0.9 + 0.1 * confidence)

                scores.append((score, edge, False, False))

            if not scores and not (edges_to_fix or edges_to_exclude):
                return None

            # 6) Forced decisions from infeasible SB sides
            if edges_to_fix or edges_to_exclude:
                if self.verbose:
                    print(f"[RLB] FORCED CHILD: fix={edges_to_fix}, exclude={edges_to_exclude}")
                return self._forced_decision(edges_to_fix, edges_to_exclude)

            # 7) Normal case: choose best score
            scores.sort(key=lambda x: x[0], reverse=True)
            best_score, best_edge, _, _ = scores[0]

            if self.verbose:
                origin = "reliable" if best_edge in reliable_candidates else "unhistoried"
                print(f"[RLB] SELECTED best_edge={best_edge} from {origin} score={best_score:.4f}")

            return [best_edge]


        elif self.branching_rule == "pseudocost":
            # Use fractional info (weighted-average solution) to estimate f in [0,1]
            # shor_primal_solution = self.lagrangian_solver.compute_weighted_average_solution()
            shor_primal_solution = self.lagrangian_solver.compute_fractional_solution(self)
            self._last_indicator = shor_primal_solution


            # Candidate set: the STRICTLY FRACTIONAL indicator entries, with
            # the same fallback chain most-fractional uses.
            #
            # This used to take the whole indicator support.  The
            # Dantzig-Wolfe point is near-integral -- a median of one strictly
            # fractional entry at n = 300 -- so the support is ~n edges almost
            # all sitting at exactly one, where f clamps to 1 - PC_MIN_DIST
            # and the product score collapses to prior_up * prior_dn * 0.0099
            # for every unobserved edge.  That is a tie across the whole
            # candidate set, decided by iteration order, and it is why
            # pseudo-cost trailed the other indicator rules on every instance
            # family measured (1118 nodes against 650 for most-fractional on
            # the anti-correlated family).
            tol = 1e-6
            if shor_primal_solution is not None:
                candidates = [
                    tuple(sorted(e)) for e, val in shor_primal_solution.items()
                    if tuple(sorted(e)) not in self.fixed_edges
                    and tuple(sorted(e)) not in self.excluded_edges
                    and tuple(sorted(e)) not in self.branched_edges
                    and tol < val < 1.0 - tol
                ]

                if not candidates:
                    candidates = [
                        tuple(sorted(e)) for e in shor_primal_solution.keys()
                        if tuple(sorted(e)) not in self.fixed_edges
                        and tuple(sorted(e)) not in self.excluded_edges
                        and tuple(sorted(e)) not in self.branched_edges
                    ]
            else:
                mst_edges = [tuple(sorted((u, v))) for u, v in self.lagrangian_solver.best_mst_edges]
                candidates = [
                    e for e in mst_edges
                    if e not in self.fixed_edges
                    and e not in self.excluded_edges
                    and e not in self.branched_edges
                ]

            if not candidates:
                return None

            EPS = 1e-9

            best_edge = None
            best_score = float("-inf")

            prior_up = MSTNode.pseudocost_prior(True)
            prior_dn = MSTNode.pseudocost_prior(False)

            for e in candidates:
                f = self._branching_indicator(e, shor_primal_solution)

                # Pseudocosts are per-unit rates; see note_pseudocost.
                pc_up = float(self.pseudocosts_up.get(e, 0.0))
                pc_dn = float(self.pseudocosts_down.get(e, 0.0))

                cu = int(self.counts_up.get(e, 0))
                cd = int(self.counts_down.get(e, 0))

                # A direction with no history takes the average of the rates
                # observed so far, so an unobserved edge is ranked as an
                # average edge rather than as a worthless one.
                if cu == 0:
                    pc_up = prior_up
                if cd == 0:
                    pc_dn = prior_dn

                # expected gains for fixing vs excluding
                gain_fix = max(EPS, pc_up * (1.0 - f))
                gain_exc = max(EPS, pc_dn * f)

                # classic product score (like strong branching proxy)
                score = gain_fix * gain_exc

                if score > best_score:
                    best_score = score
                    best_edge = e

            return [best_edge] if best_edge is not None else None
              
        elif self.branching_rule == "hybrid_strong_fractional":

            # --- Adaptive criteria for choosing strong vs fractional branching ---
            # bu = self.best_upper_bound
            # lb = self.local_lower_bound
            # if math.isfinite(bu) and math.isfinite(lb):
            #     denom = max(abs(bu), abs(lb), 1.0)
            #     gap_ratio = (bu - lb) / denom
            # else:
            #     # treat as large gap early on when bounds aren't finite yet
            #     gap_ratio = 1.0

            use_strong_branching = (
                self.depth < int(self._node_opt("sb_depth_gate", 5))
                # or (self.depth < 10 and (len(self.fixed_edges) + len(self.excluded_edges)) < 0.1 * len(self.edges))
                # or (gap_ratio > 0.99)  # large gap remaining (keep your threshold)
            )
            if use_strong_branching:
                # Strong branching phase with computational limits
                # shor_primal_solution = self.lagrangian_solver.compute_weighted_average_solution()
                shor_primal_solution = self.lagrangian_solver.compute_fractional_solution(self)
                self._last_indicator = shor_primal_solution


                if shor_primal_solution is not None:
                    # Normalize keys and drop non-finite weights
                    tolerance = 1e-6
                    candidate_edges = [
                        e for e in shor_primal_solution
                        if e not in self.fixed_edges
                        and e not in self.excluded_edges
                        and e not in self.branched_edges
                        and shor_primal_solution[e] > tolerance
                        and shor_primal_solution[e] < 1.0 - tolerance
                    ]

                    candidate_edges.sort(key=lambda e: abs(shor_primal_solution.get(e, 0.5) - 0.5))
                if shor_primal_solution is None or not candidate_edges:
                    mst_edges = [tuple(sorted((u, v))) for u, v in self.lagrangian_solver.best_mst_edges]
                    candidate_edges = [
                        e for e in mst_edges
                        if e not in self.fixed_edges
                        and e not in self.excluded_edges
                        and e not in self.branched_edges
                    ]
                
                if not candidate_edges:
                    if self.verbose:
                        print("No hybrid strong candidates available")
                    return None

                # Probe budget shrinks with depth; candidates are ordered by
                # fractionality first so a truncated budget is spent on the
                # most uncertain edges even when the fallback MST candidate
                # set is in use.
                max_sb_evals = max(
                    2, int(self._node_opt("sb_width", 8)) - self.depth)

                candidate_edges = self._rank_for_strong_branching(
                    candidate_edges,
                    shor_primal_solution if shor_primal_solution else None,
                )[:max_sb_evals]

                if self.verbose:
                    print(f"Hybrid (strong): evaluating {len(candidate_edges)} edges at depth {self.depth}")

                edges_to_fix = set()
                edges_to_exclude = set()
                best_edge = None
                best_score = -float('inf')
                scores = []

                for edge in candidate_edges:
                    score, _, _, fix_infeasible, exclude_infeasible = self.calculate_strong_branching_score(edge)
                    scores.append((edge, score))

                    if fix_infeasible:
                        edges_to_exclude.add(edge)
                    if exclude_infeasible:
                        edges_to_fix.add(edge)
                    if not (fix_infeasible or exclude_infeasible):
                        if score > best_score:
                            best_score = score
                            best_edge = edge

                # Handle forced decisions
                if edges_to_fix or edges_to_exclude:
                    if self.verbose:
                        print(f"Hybrid: forced decisions - fix: {edges_to_fix}, exclude: {edges_to_exclude}")
                    return self._forced_decision(edges_to_fix, edges_to_exclude)

                if not best_edge:
                    if self.verbose:
                        print("No viable edge found in strong branching phase")
                    # Fall through to fractional branching
                else:
                    if self.verbose:
                        print(f"Hybrid (strong) selected edge {best_edge} with score {best_score}")
                    return [best_edge]

            # Fractional branching phase (either chosen initially or fallback)
            if self.verbose and use_strong_branching:
                print("Falling back to fractional branching")
            elif self.verbose:
                print(f"Hybrid (fractional): using fractional at depth {self.depth}")

            # shor_primal_solution = self.lagrangian_solver.compute_weighted_average_solution()
            shor_primal_solution = self.lagrangian_solver.compute_fractional_solution(self)
            self._last_indicator = shor_primal_solution


            if shor_primal_solution is None:
                # Final fallback to MST edges
                mst_edges = [tuple(sorted((u, v))) for u, v in self.lagrangian_solver.best_mst_edges]
                candidate_edges = [
                    e for e in mst_edges
                    if e not in self.fixed_edges
                    and e not in self.excluded_edges
                    and e not in self.branched_edges
                ]
                return [candidate_edges[0]] if candidate_edges else None

            # Candidate policy (iii), with the same fallback chain as the
            # most-fractional rule: strictly fractional first, then the whole
            # master support, then the Lagrangian tree.
            tol = 1e-6
            candidates = [
                e for e in shor_primal_solution
                if e not in self.fixed_edges
                and e not in self.excluded_edges
                and e not in self.branched_edges
                and tol < shor_primal_solution[e] < 1.0 - tol
            ]

            if not candidates:
                candidates = [
                    e for e in shor_primal_solution
                    if e not in self.fixed_edges
                    and e not in self.excluded_edges
                    and e not in self.branched_edges
                ]

            if not candidates:
                mst_edges = [tuple(sorted((u, v))) for u, v in self.lagrangian_solver.best_mst_edges]
                candidates = [
                    e for e in mst_edges
                    if e not in self.fixed_edges
                    and e not in self.excluded_edges
                    and e not in self.branched_edges
                ]

            if not candidates:
                return None

            # Score by distance from 0.5 (most fractional first)
            branching_scores = []
            for e in candidates:
                distance_score = -abs(shor_primal_solution.get(e, 0.5) - 0.5)  # Negative so sorting descending gives smallest distance first
                branching_scores.append((e, distance_score))

            branching_scores.sort(key=lambda x: x[1], reverse=True)

            return [branching_scores[0][0]]

            
       
        else:
            raise ValueError(f"Unknown branching rule: {self.branching_rule}")

   
    def create_single_child(self, edges_to_fix, edges_to_exclude):
        """
        Fully consistent with create_children:
        - Correct (S, rhs) projection: S_free, rhs'
        - Redundant cut removal
        - Infeasibility detection
        - Support-based μ remapping
        - max_child_cuts limiting
        - Proper state propagation (depth, pseudocosts, reliability, etc.)
        """

        solver = self.lagrangian_solver
        edge_indices = solver.edge_indices
        known_edges = set(edge_indices.keys())

        idx_to_edge = getattr(solver, "idx_to_edge", None)
        if idx_to_edge is None:
            idx_to_edge = {j: e for e, j in edge_indices.items()}
            solver.idx_to_edge = idx_to_edge

        max_child_cuts = getattr(self, "max_child_cuts", 25)

        # --- Normalize edges ---
        def _norm_edge(e):
            if not (isinstance(e, tuple) and len(e) == 2):
                return None
            a, b = e
            t = (a, b) if a <= b else (b, a)
            return t if t in known_edges else None

        # --- Normalize any cut representation into edge set ---
        def _iter_edges_any(x):
            if isinstance(x, tuple) and len(x) == 2:
                e = _norm_edge(x)
                if e: yield e
                return
            if isinstance(x, int):
                e_raw = idx_to_edge.get(int(x))
                e = _norm_edge(e_raw)
                if e: yield e
                return
            try:
                for item in x:
                    if isinstance(item, int):
                        e_raw = idx_to_edge.get(item)
                        e = _norm_edge(e_raw)
                    elif isinstance(item, tuple) and len(item) == 2:
                        e = _norm_edge(item)
                    elif isinstance(item, (list, set, frozenset)) and len(item) == 2:
                        e = _norm_edge(tuple(item))
                    else:
                        e = None
                    if e: yield e
            except TypeError:
                return

        def _norm_pair(pair):
            cut_like, rhs_like = pair
            return (set(_iter_edges_any(cut_like)), int(rhs_like))

        # ---- Merge active cuts + new cuts (already normalised) ----
        def _keep(p):
            cut_like, rhs_like = p
            if isinstance(cut_like, frozenset):
                return (cut_like, int(rhs_like))
            return _norm_pair(p)

        all_cuts = [_keep(p) for p in (self.active_cuts or [])]
        all_cuts.extend(_keep(p) for p in (getattr(self, "new_cuts", []) or []))

        # ---- μ mapping from solver.best_cuts ----
        best_cuts = getattr(solver, "best_cuts", []) or []
        best_mu   = getattr(solver, "best_cut_multipliers_for_best_bound", {}) or {}

        support_to_mu = {}
        for i, (cut_i, rhs_i) in enumerate(best_cuts):
            support_to_mu[(frozenset(cut_i), rhs_i)] = float(best_mu.get(i, 0.0))

        # each cut in all_cuts receives μ by support
        current_multipliers = {}
        for idx, (cut, rhs) in enumerate(all_cuts):
            key = (frozenset(cut), rhs)
            current_multipliers[idx] = support_to_mu.get(key, 0.0)

        # --- Final fixed / excluded sets ---
        child_fixed = set(self.fixed_edges)
        for e in edges_to_fix:
            ne = _norm_edge(e)
            if ne: child_fixed.add(ne)

        child_excl = set(self.excluded_edges)
        for e in edges_to_exclude:
            ne = _norm_edge(e)
            if ne: child_excl.add(ne)

        child_fixed = frozenset(child_fixed)
        child_excl = frozenset(child_excl)
        child_fixed_idx = self.fixed_idx | {
            edge_indices[e] for e in child_fixed - self.fixed_edges if e in edge_indices
        }
        child_excl_idx = self.excluded_idx | {
            edge_indices[e] for e in child_excl - self.excluded_edges if e in edge_indices
        }
        new_branched_edges = self.branched_edges | child_fixed | child_excl

        # --- Project cuts exactly like create_children ---
        # def _project_for_child(fixed_child, excluded_child):
        #     infeasible = False
        #     proj = {}

        #     for old_i, (S, rhs) in enumerate(all_cuts):
        #         S_known = {e for e in S if e in known_edges}
        #         S_fixed = S_known & fixed_child
        #         S_free  = S_known - fixed_child - excluded_child

        #         rhs_prime = rhs - len(S_fixed)
        #         if rhs_prime < 0:
        #             infeasible = True
        #             break
        #         if len(S_free) <= rhs_prime:
        #             continue

        #         key = frozenset(S_free)
        #         mu_old = float(current_multipliers.get(old_i, 0.0))

        #         prev = proj.get(key)
        #         if (prev is None or
        #             rhs_prime < prev[0] or
        #             (rhs_prime == prev[0] and abs(mu_old) > abs(prev[1]))):
        #             proj[key] = (rhs_prime, mu_old)

        #     if infeasible:
        #         return None, None, True

        #     # sort like create_children
        #     def _key(sfree, rhs_, mu_):
        #         return (-len(sfree), rhs_, tuple(sorted(sfree)))

        #     ordered = sorted(
        #         ((sfree, rhs_mu[0], rhs_mu[1]) for sfree, rhs_mu in proj.items()),
        #         key=lambda x: _key(x[0], x[1], x[2])
        #     )

        #     if len(ordered) > max_child_cuts:
        #         ordered = ordered[:max_child_cuts]

        #     kept_cuts = [(set(sfree), rhs) for (sfree, rhs, mu) in ordered]
        #     kept_mu   = {i: float(mu) for i, (_, _, mu) in enumerate(ordered)}

        #     return kept_cuts, kept_mu, False
        T_parent = set(self.mst_edges or [])
        def _project_for_child(fixed_child, excluded_child):
            infeasible = False
            proj = {}
            forced_out = set()

            for old_i, (S, rhs) in enumerate(all_cuts):
                S_fixed = S & fixed_child
                S_excl = S & excluded_child

                if S_fixed or S_excl or forced_out:
                    S_free = S - fixed_child - excluded_child - forced_out
                else:
                    S_free = S

                rhs_prime = rhs - len(S_fixed)
                if rhs_prime < 0:
                    infeasible = True
                    break
                if len(S_free) <= rhs_prime:
                    continue

                if rhs_prime == 0:
                    forced_out |= S_free
                    continue

                lhs_est = len(T_parent & S_free)
                viol_est = lhs_est - rhs_prime
                key = S_free if isinstance(S_free, frozenset) else frozenset(S_free)
                mu_old = float(current_multipliers.get(old_i, 0.0))

                prev = proj.get(key)
                if (
                    prev is None
                    or rhs_prime < prev["rhs"]
                    or (rhs_prime == prev["rhs"] and viol_est > prev["viol"])
                ):
                    proj[key] = {
                        "rhs": rhs_prime,
                        "mu": mu_old,
                        "viol": viol_est,
                    }

            if infeasible:
                return None, None, True, None

            if forced_out:
                kept, mu, bad, more = _project_for_child(
                    fixed_child, excluded_child | forced_out
                )
                if bad:
                    return None, None, True, None
                return kept, mu, False, forced_out | (more or set())

            ordered = sorted(
                proj.items(),
                key=lambda kv: (-kv[1]["viol"], len(kv[0]), kv[1]["rhs"])
            )

            if len(ordered) > max_child_cuts:
                ordered = ordered[:max_child_cuts]

            kept_cuts = [(sfree, info["rhs"]) for sfree, info in ordered]
            kept_mu   = {i: float(info["mu"]) for i, (_, info) in enumerate(ordered)}

            return kept_cuts, kept_mu, False, None

        kept_cuts, kept_mu, prune, forced = _project_for_child(child_fixed, child_excl)
        if prune:
            return None
        if forced:
            child_excl = child_excl | forced
            child_excl_idx = child_excl_idx | {
                edge_indices[e] for e in forced if e in edge_indices
            }
            MSTNode.cut_forced_exclusions += len(forced)

        # ---- Create the child (identical arguments as create_children) ----
        child = MSTNode(
            self.edges, self.num_nodes, self.budget,
            fixed_edges=child_fixed,
            excluded_edges=child_excl,
            branched_edges=new_branched_edges,
            initial_lambda=solver.best_lambda if self.inherit_lambda else 0.05,
            inherit_lambda=self.inherit_lambda,
            branching_rule=self.branching_rule,
            step_size=solver.step_size if self.inherit_step_size else 0.001,
            inherit_step_size=self.inherit_step_size,
            use_cover_cuts=self.use_cover_cuts,
            cut_frequency=self.cut_frequency,
            node_cut_frequency=self.node_cut_frequency,
            parent_cover_cuts=kept_cuts,
            parent_cover_multipliers=kept_mu,
            fixed_idx=child_fixed_idx,
            excluded_idx=child_excl_idx,
            use_bisection=self.use_bisection,
            max_iter=solver.max_iter,
            verbose=self.verbose,
            depth=self.depth + 1,
            pseudocosts_up=self.pseudocosts_up,
            pseudocosts_down=self.pseudocosts_down,
            counts_up=self.counts_up,
            counts_down=self.counts_down,
            reliability_eta=self.reliability_eta,
            lookahead_lambda=self.lookahead_lambda,
            solver_overrides=self.solver_overrides,
            parent_lower_bound=self.local_lower_bound,
        )

        return child



   

    def get_modified_weight(self, edge):
        u, v = tuple(sorted(edge))
        w, l = self.lagrangian_solver.edge_attributes[(u, v)]

        modified = w + self.lagrangian_solver.best_lambda * l
        
        for cut_idx, (cut, _) in enumerate(self.active_cuts):
            if (u, v) in cut:
                modified += self.cut_multipliers.get(cut_idx, 0)
        
        return modified

   
    

   

    def simulate_branching_bound(self, edge, fix_edge: bool = True, max_iters: int = 10):
        """
        Behavior-preserving strong-branching probe using a pooled LagrangianMST.
        No unsupported kwargs are passed to solve().
        """
        _t_probe = time.time()
        try:
            return self._simulate_branching_bound_impl(edge, fix_edge, max_iters)
        finally:
            MSTNode.probe_calls += 1
            MSTNode.probe_time += time.time() - _t_probe

    def _simulate_branching_bound_impl(self, edge, fix_edge, max_iters):
        # --- Normalize edge ---
        try:
            u, v = edge
        except Exception:
            raise ValueError(f"simulate_branching_bound: invalid edge {edge!r}")
        branched_edge = tuple(sorted((u, v)))

        # --- Build hypothetical fixed/excluded sets (avoid deep copies) ---
        _bi = self.lagrangian_solver.edge_indices.get(branched_edge)
        _bs = {_bi} if _bi is not None else frozenset()

        if fix_edge:
            new_fixed = self.fixed_edges | {branched_edge}
            new_excluded = self.excluded_edges
            new_fixed_idx = self.fixed_idx | _bs
            new_excluded_idx = self.excluded_idx
        else:
            new_fixed = self.fixed_edges
            new_excluded = self.excluded_edges | {branched_edge}
            new_fixed_idx = self.fixed_idx
            new_excluded_idx = self.excluded_idx | _bs

        # --- Inherit cuts (keep stable order) ---
        if getattr(self, "new_cuts", None):
            all_cuts = list(self.active_cuts) + list(self.new_cuts)
        else:
            all_cuts = list(self.active_cuts)

        # Multipliers aligned to all_cuts indices
        cut_multipliers = {}
        if getattr(self, "cut_multipliers", None) is not None and self.active_cuts:
            parent_index = {}
            for idx, (cset, rhs) in enumerate(self.active_cuts):
                parent_index[(frozenset(cset), rhs)] = idx
            for idx, (cset, rhs) in enumerate(all_cuts):
                pidx = parent_index.get((frozenset(cset), rhs))
                cut_multipliers[idx] = self.cut_multipliers.get(pidx, 0.0) if pidx is not None else 0.0
        else:
            for idx in range(len(all_cuts)):
                cut_multipliers[idx] = 0.0

        # --- Borrow, reset, (optionally) set warm-start state via attributes, then solve ---
        with self._sb_pool.borrow() as sim_solver:
                # --- Sync cut-related parameters from the main solver to the SB solver ---
            if hasattr(self.lagrangian_solver, "max_cut_depth"):
                sim_solver.max_cut_depth = getattr(self.lagrangian_solver, "max_cut_depth")
            if hasattr(self.lagrangian_solver, "extra_iter_for_cuts"):
                sim_solver.extra_iter_for_cuts = getattr(self.lagrangian_solver, "extra_iter_for_cuts")
            if hasattr(self.lagrangian_solver, "min_cut_violation_for_add"):
                sim_solver.min_cut_violation_for_add = getattr(
                    self.lagrangian_solver, "min_cut_violation_for_add"
                )
            # Belt and braces: the pool factory already applies the
            # cut-shaping overrides at construction, but the probe is also
            # reused across nodes, so re-assert the two that decide which
            # ladder rung it is simulating.
            for _attr in ("cut_strengthening", "max_active_cuts"):
                if hasattr(self.lagrangian_solver, _attr):
                    setattr(sim_solver, _attr,
                            getattr(self.lagrangian_solver, _attr))

            sim_solver.reset(
                fixed_edges=new_fixed,
                excluded_edges=new_excluded,
                fixed_idx=new_fixed_idx,
                excluded_idx=new_excluded_idx,
                initial_lambda=getattr(self.lagrangian_solver, "best_lambda", getattr(self, "initial_lambda", 0.05)),
                step_size=getattr(self.lagrangian_solver, "step_size", getattr(self, "step_size", 0.001)),
                max_iter=int(max_iters),
                use_cover_cuts=bool(getattr(self, "use_cover_cuts", False)),
                cut_frequency=int(getattr(self, "cut_frequency", 10)),
                use_bisection=False,
                verbose=False,
            )

            # reset() clears it, so hand the incumbent over afterwards: the
            # probe needs a finite gap to form a Polyak step just as a real
            # node does.
            sim_solver.incumbent_ub = MSTNode.incumbent_for_step()

            # (Optional) Warm-start by assigning fields directly if your solver honors them.
            # These two lines are safe; they won't break if attrs don't exist.
            if getattr(self, "mst_edges", None) is not None and hasattr(sim_solver, "best_mst_edges"):
                sim_solver.best_mst_edges = list(self.mst_edges)
            if hasattr(self.lagrangian_solver, "best_lambda") and hasattr(sim_solver, "lmbda"):
                sim_solver.lmbda = float(self.lagrangian_solver.best_lambda)

            # Call solve WITHOUT unsupported kwargs
            # lower_bound, upper_bound, _info = sim_solver.solve(
            #     inherited_cuts=all_cuts,
            #     inherited_multipliers=cut_multipliers
            # )
            # Depth for the probe: children of this node → depth + 1
            probe_depth = getattr(self, "depth", 0) + 1

            lower_bound, upper_bound, _info = sim_solver.solve(
                inherited_cuts=all_cuts,
                inherited_multipliers=cut_multipliers,
                depth=probe_depth,          # <<< IMPORTANT
            )


        return float(lower_bound)


    def calculate_strong_branching_score(self, edge):
        """
        Fast strong branching using simulation instead of full child creation.

        Returns:
            (score, fix_delta, exc_delta, fix_infeasible, exclude_infeasible)
        """
        u, v = tuple(sorted(edge))

        # Only +inf proves infeasibility.
        #
        # The caller turns `fix_infeasible` into a FORCED single child that
        # excludes the edge for good, with no sibling -- so this flag has to be
        # a proof, not a guess.  Every `return float("inf")` inside solve() is
        # one: a cycle among the fixed edges, a graph that cannot be spanned
        # under the exclusions, or a valid cut whose reduced right-hand side
        # has gone negative.
        #
        # -inf is NOT.  solve() hands back `best_lower_bound`, which reset()
        # initialises to -inf and which stays there if no iteration produced a
        # usable bound -- that means "no bound computed", not "no solution".
        # NaN is a numerical failure and proves nothing either.  The old test
        # was `isnan(lb) or isinf(lb)`, which swept both in and could throw the
        # optimum away.  They are now treated as "no information": nothing is
        # forced and the edge simply carries no strong-branching signal.
        def _probe(lb):
            if lb == float("inf"):
                return True, False            # proven infeasible
            if math.isnan(lb) or math.isinf(lb):
                return False, True            # unusable (-inf / NaN)
            return False, False

        # --- Strong-branching probe: FIX edge ---
        fixed_lower_bound = self.simulate_branching_bound(edge, fix_edge=True, max_iters=2)
        fix_infeasible, fix_unusable = _probe(fixed_lower_bound)

        if self.verbose and (fix_infeasible or fix_unusable):
            print(f"Fixed simulation for edge {edge}: "
                  f"{'infeasible' if fix_infeasible else 'unusable'} (LB={fixed_lower_bound})")

        # --- Strong-branching probe: EXCLUDE edge ---
        excluded_lower_bound = self.simulate_branching_bound(edge, fix_edge=False, max_iters=2)
        exclude_infeasible, exc_unusable = _probe(excluded_lower_bound)

        if self.verbose and (exclude_infeasible or exc_unusable):
            print(f"Excluded simulation for edge {edge}: "
                  f"{'infeasible' if exclude_infeasible else 'unusable'} (LB={excluded_lower_bound})")

        # An unusable probe yields no delta and no forcing.
        if fix_unusable or exc_unusable:
            return 0.0, 0.0, 0.0, fix_infeasible, exclude_infeasible

        # If both directions are infeasible, this edge is useless as a branching candidate
        if fix_infeasible and exclude_infeasible:
            if self.verbose:
                print(f"Edge {edge}: both branches infeasible in strong branching probe")
            # Worst possible score, deltas 0 (won't affect pseudocosts either)
            return -float("inf"), 0.0, 0.0, True, True

        # --- Compute LB improvements (relative to current node LB) ---
        # For infeasible side, treat delta as 0 for logging/pseudocosts (we don't use it when *_infeasible is True).
        fix_delta = (fixed_lower_bound - self.own_lower_bound) if not fix_infeasible else 0.0
        exc_delta = (excluded_lower_bound - self.own_lower_bound) if not exclude_infeasible else 0.0

        # Guard against a non-finite delta reaching the score or the
        # pseudocosts: max(nan, 0.0) returns nan in Python, which would then
        # propagate through the product score and poison the EMA.
        if math.isnan(fix_delta) or math.isinf(fix_delta):
            fix_delta = 0.0
        if math.isnan(exc_delta) or math.isinf(exc_delta):
            exc_delta = 0.0

        # Only positive improvements should contribute to the score
        fix_gain = max(fix_delta, 0.0)
        exc_gain = max(exc_delta, 0.0)

        # --- Score ---
        # If one side is infeasible and the other is feasible, we want a "forced" decision.
        # The hybrid / reliability code already treats `fix_infeasible` / `exclude_infeasible`
        # as forcing edges_to_exclude / edges_to_fix, so we can just give any finite score here.
        if not fix_infeasible and not exclude_infeasible:
            # Your original product-based score, but using gains and a small epsilon
            score = max(fix_gain, 1e-6) * max(exc_gain, 1e-6)
        else:
            # One side infeasible → the calling code will handle the forcing;
            # we don't rely on the numeric score for ranking in that case.
            score = float("inf")

        if self.verbose:
            print(
                f"Edge {edge}: "
                f"score={score:.6g}, "
                f"fix_LB={fixed_lower_bound:.6g}, exc_LB={excluded_lower_bound:.6g}, "
                f"Δfix={fix_delta:.6g}, Δexc={exc_delta:.6g}, "
                f"fix_inf={fix_infeasible}, exc_inf={exclude_infeasible}"
            )

        return score, fix_delta, exc_delta, fix_infeasible, exclude_infeasible

    
    def simulate_fix_edge(self, u, v):
        normalized_edge = tuple(sorted((u, v)))
        mst_edges = [tuple(sorted((x, y))) for x, y in self.lagrangian_solver.best_mst_edges]
        if normalized_edge in mst_edges:
            return self.own_lower_bound

        mst_graph = nx.Graph(mst_edges)
        mst_graph.add_edge(u, v)

        try:
            cycle = nx.find_cycle(mst_graph, source=u)
        except nx.NetworkXNoCycle:
            return self.own_lower_bound

        cycle_without_fixed = [edge for edge in cycle if edge not in self.fixed_edges]
        heaviest_edge = None
        max_weight = float('-inf')
        for edge in cycle_without_fixed:
            if edge not in self.fixed_edges:
                edge_weight = self.get_modified_weight(edge)
                if edge_weight > max_weight:
                    max_weight = edge_weight
                    heaviest_edge = edge

        if not heaviest_edge:
            return float('inf')

        fixed_edge_weight = self.get_modified_weight(normalized_edge)
        heaviest_edge_weight = self.get_modified_weight(heaviest_edge)
        new_lower_bound = self.own_lower_bound + fixed_edge_weight - heaviest_edge_weight
        return new_lower_bound

    def simulate_exclude_edge(self, u, v):
        normalized_edge = tuple(sorted((u, v)))
        mst_edges = [tuple(sorted((x, y))) for x, y in self.lagrangian_solver.best_mst_edges]
        if normalized_edge not in mst_edges:
            return self.own_lower_bound

        mst_graph = nx.Graph(mst_edges)
        mst_graph.remove_edge(u, v)

        components = list(nx.connected_components(mst_graph))
        if len(components) != 2:
            return float('inf')

        cheapest_edge = None
        min_weight = float('inf')
        for x, y, w, l in self.edges:
            normalized = tuple(sorted((x, y)))
            if normalized == normalized_edge:
                continue
            if (x in components[0] and y in components[1]) or (x in components[1] and y in components[0]):
                if normalized not in self.excluded_edges:
                    edge_weight = self.get_modified_weight(normalized)
                    if edge_weight < min_weight:
                        min_weight = edge_weight
                        cheapest_edge = normalized

        if not cheapest_edge:
            return float('inf')

        excluded_edge_weight = self.get_modified_weight(normalized_edge)
        replacement_edge_weight = self.get_modified_weight(cheapest_edge)
        new_lower_bound = self.own_lower_bound - excluded_edge_weight + replacement_edge_weight
        return new_lower_bound

    def print_cut_info(self):
        if self.verbose:
            print(f"\nNode Cut Status (Fixed: {self.fixed_edges}, Excluded: {self.excluded_edges})")
            print("Active Cuts:")
            for i, (cut, rhs) in enumerate(self.active_cuts):
                mult = self.cut_multipliers.get(i, 0)
                print(f"Cut {i}: Cut {cut} (RHS: {rhs}, Multiplier: {mult:.3f})")
            
            print("\nInherited Cuts Breakdown:")
            inherited_from_parent = 0
            new_generated = 0
            for cut, rhs in self.lagrangian_solver.best_cuts:
                if (cut, rhs) in self.active_cuts:
                    inherited_from_parent += 1
                else:
                    new_generated += 1
            print(f"Total cuts: {len(self.lagrangian_solver.best_cuts)}")
            print(f" - Inherited: {inherited_from_parent}")
            print(f" - New: {new_generated}")

    def get_fractional_value(self, edge):
        normalized = tuple(sorted(edge))

        # The tree's budget violation is a node constant, not a property of
        # the edge being scored.  Recomputing it per candidate was an O(n)
        # sum per call and 11% of a whole run at n = 400.
        violation = self._length_violation
        if violation is None:
            violation = self.actual_length - self.budget
            self._length_violation = violation

        if violation <= 0:
            return 0.5  # Feasible or under; neutral uncertainty
        if normalized in self._mst_edge_set:
            edge_contrib = self.lagrangian_solver.edge_attributes[normalized][1] / violation if violation > 0 else 0.5
            f = 1 - min(1.0, max(0.0, edge_contrib))  # High contrib = more fractional (likely to flip out)
        else:
            sim_lb = self.simulate_fix_edge(*edge)
            delta = max(0, sim_lb - self.own_lower_bound)
            f = min(1.0, max(0.0, delta / (self.lagrangian_solver.best_lambda or 1.0)))  # Delta normalized by lambda
        if self.verbose:
            print(f"Slackness-based f={f:.2f} for edge {edge}")
        return max(0.01, min(0.99, f))  # Clamp away from 0/1 to avoid div-by-zero
    #this version

