
import networkx as nx
import numpy as np
from time import time
from collections import defaultdict, OrderedDict
from scipy.optimize import linprog  
import math
import heapq
import hashlib
import bisect

CUT_STRENGTHENINGS = ("literature", "lemma1", "full")



class UnionFind:
    __slots__ = ['parent', 'rank', 'size']
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
    
    def find(self, u):
        # Iterative with path halving.  This is the single hottest function in
        # the solver -- 79M calls in one n=200 run -- and the recursive form
        # paid a Python frame per level of the tree.
        parent = self.parent
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.size[pu] < self.size[pv]:
            pu, pv = pv, pu
        self.parent[pv] = pu
        self.size[pu] += self.size[pv]
        self.rank[pu] = max(self.rank[pu], self.rank[pv] + 1)
        return True
    
    def connected(self, u, v):
        return self.find(u) == self.find(v)
    
    def count_components(self):
        return len(set(self.find(i) for i in range(len(self.parent))))

class LRUCache:
    __slots__ = ['cache', 'capacity']
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)

        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

class LagrangianMST:
    total_compute_time = 0

    # Separation statistics, so a run can report what the cuts actually did
    # and not only the node count they leave behind.  Zero them per run with
    # LagrangianMST.reset_cut_stats().
    cuts_separated = 0     # covers accepted into the active pool
    cuts_infeasible = 0    # nodes proved infeasible by rhs_eff < 0
    cut_nodes = 0          # nodes where separation actually ran
    exact_dual_nodes = 0   # nodes the exact cut dual raised
    exact_dual_gain = 0.0  # total bound raised by the exact cut dual
    lr_iterations = 0      # subgradient iterations actually executed,
                           # node solves and strong-branching probes alike

    # Frozen-benchmark instrumentation (all reset per run by reset_cut_stats).
    # Paper metrics: separation and indicator time/volume and the indicator
    # candidate-pool histogram.  Usage counters (probe split, rank lifting)
    # exist so a smoke test can prove each configuration really ran the
    # components it claims to -- they change nothing in the algorithm.
    sep_calls = 0          # generate_cover_cuts invocations (nodes + probes)
    sep_probe_calls = 0    # ... of which inside strong-branching probes
    sep_time = 0.0         # seconds inside generate_cover_cuts
    sep_max_depth = -1     # deepest node at which separation ran
    rank_lift_calls = 0    # rank-lifting certificate invocations
    exact_dual_probe_nodes = 0  # exact-cut-dual gains inside probes
    indicator_calls = 0    # compute_fractional_solution invocations
    indicator_none = 0     # ... that returned no indicator
    indicator_time = 0.0   # seconds building the DW / average indicator
    pool_hist = {}         # strictly-fractional free entries -> count
    # Validation only (validate_cuts.py): when a list, every generated cut is
    # appended with its node context.  None in every benchmark run.
    cut_log = None

    # Single slot for the node-invariant separation scaffold; see
    # generate_cover_cuts.  Held on the class so it is not retained per open
    # node, and dropped when the instance changes.
    _sep_cache = None

    # Cut support -> edge-index array, shared by every node carrying the cut.
    _cut_idx_cache = {}
    # Monotone tag for a node's free mask; see _rebuild_candidates.
    _free_mask_serial = 0


    def __init__(self, edges, num_nodes, budget, fixed_edges=None, excluded_edges=None,
                 initial_lambda=0.05, step_size=0.001, max_iter=10, 
                 use_cover_cuts=False, cut_frequency=5, use_bisection=False,
                 verbose=False, shared_graph=None,
                 fixed_idx=None, excluded_idx=None):
        start_time = time()
        self.edges = edges
        self.num_nodes = num_nodes
        self.budget = budget
        # MSTNode hands these down as frozensets of already-normalised
        # edges; re-deriving them cost an O(depth) rebuild per node and per
        # strong-branching probe.
        self.fixed_edges = (
            fixed_edges if isinstance(fixed_edges, frozenset)
            else frozenset(tuple(sorted((u, v))) for u, v in (fixed_edges or ()))
        )
        self.excluded_edges = (
            excluded_edges if isinstance(excluded_edges, frozenset)
            else frozenset(tuple(sorted((u, v))) for u, v in (excluded_edges or ()))
        )

        edge_key = id(edges)
        if getattr(LagrangianMST, "_edge_key", None) != edge_key:
            LagrangianMST._edge_key = edge_key
            LagrangianMST._sep_cache = None
            edge_list = [tuple(sorted((u, v))) for u, v, _, _ in edges]
            LagrangianMST._edge_list = edge_list
            LagrangianMST._edge_indices = {edge: idx for idx, edge in enumerate(edge_list)}
            # float64, not float32.  A dual bound is a sum of n-1 priced
            # weights and is compared against the incumbent with a one-unit
            # integrality margin; float32 carries ~1e-4 of absolute error per
            # term, which at n = 400 accumulates to a few hundredths and puts
            # a rounding error inside the margin the pruning test relies on.
            # The two arrays are 16k doubles -- a tenth of a megabyte.
            LagrangianMST._edge_weights = np.array([w for _, _, w, _ in edges], dtype=np.float64)
            LagrangianMST._edge_lengths = np.array([l for _, _, _, l in edges], dtype=np.float64)
            LagrangianMST._edge_attributes = {
                edge: (w, l) for (edge, (_, _, w, l)) in zip(edge_list, edges)
            }
            # Node-invariant scaffolding shared by every solver on this
            # instance.  All of it used to be rebuilt per node (idx_to_edge
            # alone is an m-entry dict, 1.3ms at n = 400 and 4% of a whole
            # run) or per call (the length order, the endpoint arrays).
            LagrangianMST._idx_to_edge = dict(enumerate(edge_list))
            LagrangianMST._edge_key_set = set(edge_list)
            LagrangianMST._edge_u = np.fromiter(
                (e[0] for e in edge_list), dtype=np.int32, count=len(edge_list)
            )
            LagrangianMST._edge_v = np.fromiter(
                (e[1] for e in edge_list), dtype=np.int32, count=len(edge_list)
            )
            # Edges in nondecreasing length, as indices: the cheapest-
            # completion test walks this instead of sweeping E.
            LagrangianMST._len_order = np.argsort(
                LagrangianMST._edge_lengths, kind="stable"
            ).astype(np.int64)
            LagrangianMST._all_idx = np.arange(len(edge_list), dtype=np.int64)
            LagrangianMST._cut_idx_cache = {}
            # (edge, length) in nondecreasing length, as plain Python objects:
            # the shortest-admissible-edge scan runs per child node and wants
            # tuples, not numpy scalars.
            _lens = LagrangianMST._edge_lengths
            LagrangianMST._len_sorted_edges = [
                (edge_list[i], float(_lens[i]))
                for i in LagrangianMST._len_order.tolist()
            ]

        self.edge_list = LagrangianMST._edge_list
        self.edge_indices = LagrangianMST._edge_indices
        self.edge_weights = LagrangianMST._edge_weights
        self.edge_lengths = LagrangianMST._edge_lengths
        self.edge_attributes = LagrangianMST._edge_attributes
        self.idx_to_edge = LagrangianMST._idx_to_edge
        self._edge_key_set = LagrangianMST._edge_key_set
        self._edge_u = LagrangianMST._edge_u
        self._edge_v = LagrangianMST._edge_v
        self._len_order = LagrangianMST._len_order
        self._len_sorted_edges = LagrangianMST._len_sorted_edges

        self.lmbda = initial_lambda
        self.step_size = step_size
        # self.p = p
        self.max_iter = max_iter
        self.use_bisection = use_bisection
        self.verbose = verbose

        self.best_lower_bound = float('-inf')
        self.best_upper_bound = float('inf')
        # Incumbent supplied from outside (the search's best known solution).
        # It feeds the Polyak gap only, and is deliberately kept apart from
        # best_upper_bound so that a node still reports just what it found
        # itself -- the search attributes the solution to the reporting node.
        self.incumbent_ub = float('inf')
        # The tree that actually achieves best_upper_bound.  last_mst_edges is
        # the final Lagrangian tree, which is normally over budget, so it must
        # not be used to report the incumbent solution.
        self.best_feasible_edges = None
        self.last_mst_edges = []
        self.primal_solutions = []
        self.avg_trees = []
        self.fractional_solutions = []
        self.step_sizes = []
        self.subgradients = []
        self._MAX_HISTORY = 100
        self._primal_history_cap = 30
        self._fractional_history_cap = 50
        self._subgradient_history_cap = 20

        self.best_lambda = self.lmbda
        self.best_mst_edges = None
        self.best_cost = 0

        self.use_cover_cuts = use_cover_cuts
        self.cut_frequency = cut_frequency
        self.best_cuts = []
        self.best_cut_multipliers = {}

        self.multipliers = []

        # The caller normally maintains the index sets incrementally, since
        # a child differs from its parent by a single edge.  Deriving them
        # here means hashing every fixed/excluded edge tuple twice, and with
        # reduced-cost fixing the excluded set reaches thousands of edges --
        # 5ms per node at n = 400, rebuilt again for every strong-branching
        # probe.
        self.fixed_edge_indices = (
            set(fixed_idx) if fixed_idx is not None
            else {self.edge_indices[e] for e in self.fixed_edges if e in self.edge_indices}
        )
        self.excluded_edge_indices = (
            set(excluded_idx) if excluded_idx is not None
            else {self.edge_indices[e] for e in self.excluded_edges if e in self.edge_indices}
        )
        self.cache_tolerance = 1e-6 if num_nodes > 100 else 1e-8
        self.mst_cache = LRUCache(capacity=64)

        self._last_mst_idx = None
        self._last_mst_list = None
        # Filter-Kruskal block size.  One block connects all but a handful of
        # components on these graphs, so the second block is tiny.
        self._kruskal_block = max(4 * num_nodes, 1024)

        self.last_mst_edges = None

        if shared_graph is not None:
            self.graph = shared_graph
        else:
            self.graph = nx.Graph()
            self.graph.add_edges_from(self.edge_list)

        self._mw_cached = None
        self._mw_lambda = None
        self._mw_mu = None
        self._mw_free_mask_key = None
        self._rebuild_candidates()

        end_time = time()
        LagrangianMST.total_compute_time += end_time - start_time


    

    def reset(self, *, fixed_edges=None, excluded_edges=None, initial_lambda=None,
              step_size=None, max_iter=None, use_cover_cuts=None, cut_frequency=None,
              use_bisection=None, verbose=None, fixed_idx=None, excluded_idx=None):
        if fixed_edges is None:
            self.fixed_edges = frozenset()
        elif isinstance(fixed_edges, frozenset):
            self.fixed_edges = fixed_edges
        else:
            self.fixed_edges = frozenset(tuple(sorted((u, v))) for u, v in fixed_edges)

        if excluded_edges is None:
            self.excluded_edges = frozenset()
        elif isinstance(excluded_edges, frozenset):
            self.excluded_edges = excluded_edges
        else:
            self.excluded_edges = frozenset(tuple(sorted((u, v))) for u, v in excluded_edges)

        self.fixed_edge_indices = (
            set(fixed_idx) if fixed_idx is not None
            else {self.edge_indices[e] for e in self.fixed_edges if e in self.edge_indices}
        )
        self.excluded_edge_indices = (
            set(excluded_idx) if excluded_idx is not None
            else {self.edge_indices[e] for e in self.excluded_edges if e in self.edge_indices}
        )
        self._rebuild_candidates()
        self._last_mst_idx = None
        self._last_mst_list = None

        if initial_lambda is not None:
            self.lmbda = float(initial_lambda)
        else:
            self.lmbda = getattr(self, "lmbda", 0.05)

        if step_size is not None:
            self.step_size = float(step_size)
        if max_iter is not None:
            self.max_iter = int(max_iter)
        if use_cover_cuts is not None:
            self.use_cover_cuts = bool(use_cover_cuts)
        if cut_frequency is not None:
            self.cut_frequency = int(cut_frequency)
        if use_bisection is not None:
            self.use_bisection = bool(use_bisection)
        if verbose is not None:
            self.verbose = bool(verbose)

        self.best_lower_bound = float("-inf")
        self.best_upper_bound = float("inf")
        self.incumbent_ub = float("inf")
        self.best_feasible_edges = None

        self.best_lambda = float(self.lmbda)
        self.best_mst_edges = []
        self.best_cost = 0

        self.best_cuts = []
        self.best_cut_multipliers = {}
        self.best_cut_multipliers_for_best_bound = {}

        self.multipliers = []

        # Important when reusing solver objects in strong branching
        self._v_lambda = 0.0

        self.last_mst_edges = None

        try:
            cap = self.mst_cache.capacity
        except Exception:
            cap = max(20, self.num_nodes * 2)
        self.mst_cache = LRUCache(capacity=cap)

        self._invalidate_weight_cache()


    def _rebuild_candidates(self):
        """Indices Kruskal may choose from, plus the node's free mask.

        A node's fixings do not move while it solves, so both are node
        constants.  They used to be re-derived from an m-long boolean mask on
        every subgradient iteration and, for the mask, keyed by a frozenset of
        the fixings rebuilt on every call.
        """
        skip = self.fixed_edge_indices | self.excluded_edge_indices

        LagrangianMST._free_mask_serial += 1
        self._free_mask_key = LagrangianMST._free_mask_serial

        if not skip:
            # Shared and read-only; nothing writes through it.
            self._cand_idx = LagrangianMST._all_idx
            self._free_mask_cache = None
            return

        mask = np.ones(len(self.edge_list), dtype=bool)
        mask[np.fromiter(skip, dtype=np.int64, count=len(skip))] = False
        self._cand_idx = np.flatnonzero(mask)
        self._free_mask_cache = mask

    def _cut_index_array(self, cut):
        """Edge indices of a cut support, cached by support.

        A lifted support runs to thousands of edges and every node rebuilds
        the index arrays of its whole pool.  A frozenset caches its own hash,
        so the repeat lookups are O(1) and the arrays are shared by every node
        that inherits the cut.
        """
        cache = LagrangianMST._cut_idx_cache
        arr = cache.get(cut)

        if arr is None:
            ei = self.edge_indices
            arr = np.fromiter(
                (ei[e] for e in cut if e in ei), dtype=np.int64, count=-1
            )
            if len(cache) > 4096:
                cache.clear()
            cache[cut] = arr

        return arr

    def clear_iteration_state(self):
        """Clear per-iteration buffers"""
        self.primal_solutions = []
        self.avg_trees = []
        self.fractional_solutions = []
        self.subgradients = []
        self.step_sizes = []
        self.multipliers = []
        self._v_lambda = 0.0
        # self.last_modified_weights = None
        # self.last_mst_edges = None
        self._invalidate_weight_cache()
        if hasattr(self, 'mst_cache'):
            self.mst_cache = LRUCache(capacity=5)
   
    def generate_cover_cuts(self, mst_edges):
        """Timed, counted entry point to the Section 6 separation below.

        Pure instrumentation: the cuts returned are exactly those of
        _generate_cover_cuts_impl.  When LagrangianMST.cut_log is a list (the
        validation harness only) each call's node context and cuts are
        recorded so every cut can be checked against brute force.
        """
        t0 = time()
        try:
            out = self._generate_cover_cuts_impl(mst_edges)
        finally:
            LagrangianMST.sep_time += time() - t0
            LagrangianMST.sep_calls += 1
            if getattr(self, "_is_probe", False):
                LagrangianMST.sep_probe_calls += 1
            d = int(getattr(self, "depth", 0) or 0)
            if d > LagrangianMST.sep_max_depth:
                LagrangianMST.sep_max_depth = d
        if LagrangianMST.cut_log is not None:
            LagrangianMST.cut_log.append((
                frozenset(getattr(self, "fixed_edges", ()) or ()),
                frozenset(getattr(self, "excluded_edges", ()) or ()),
                float(self.budget),
                [(frozenset(c), int(r)) for c, r in (out or [])],
                bool(getattr(self, "_is_probe", False)),
            ))
        return out

    def _generate_cover_cuts_impl(self, mst_edges):
        """
        Node-local cover-cut separation (Section 6).

        From the current Lagrangian tree T^k at node (F+, F-) the procedure
        builds one seed cover and turns it into up to two strengthened
        candidates: a lifted residual cover and a lifted tree-completion cover.

        6.1 Seed cover.
            A := E \\ (F+ u F-) and B' := B - sum_{e in F+} l_e. The edges of
            T^k n A are sorted in nonincreasing order of length and accumulated
            until their total length exceeds B'. The resulting set S gives the
            residual cover inequality sum_{e in S} x_e <= |S| - 1, which T^k
            violates by one unit.

        6.2 Lightweight unit lifting (Lemma 1).
            With k := |S| and sigma_{k-1}(H) the sum of the k-1 shortest lengths
            in the current lifted support H, candidates f in A \\ H are scanned
            in nonincreasing order of length and accepted whenever
                l_f > B' - sigma_{k-1}(H).
            The right-hand side stays |S| - 1.

        6.3 Tree-completion-aware refinement (Proposition 1).
            C(Q) is the minimum additional length needed to complete F+ u Q to a
            spanning tree using admissible edges of A \\ Q, obtained by
            contracting F+ u Q and running Kruskal on lengths, and +inf when
            F+ u Q has a cycle or admits no completion. Q is certified when
                sum_{e in Q} l_e + C(Q) > B'.
            Starting from P := S, which is certified because its own length
            already exceeds B', edges are scanned in nondecreasing order of
            length -- so the ones contributing least to the violation are tested
            first -- and e is deleted whenever P \\ {e} is still certified. The
            scan repeats until a full pass deletes no edge, giving Q*.

        6.4 Completion-aware unit lifting (Lemma 2).
            With k* := |Q*| and B* := B' - C(Q*), candidates are drawn from
            A(Q*), the admissible edges internal to a component of the forest
            F+ u Q*, scanned in nonincreasing order of length and accepted
            whenever
                l_f > B* - sigma_{k*-1}(H).
            The right-hand side stays |Q*| - 1.

        Returns a list of (support, rhs) pairs. Reduction under the node fixings
        (16), ranking by violation on the generating tree, and the active-pool
        cap are applied by the caller.
        """
        if not mst_edges:
            return []

        EPS = 1e-12

        # Which rung of the attribution ladder this run sits on; see
        # CUT_STRENGTHENINGS.
        strengthening = str(getattr(self, "cut_strengthening", "full")).lower()

        # ------------------------------------------------------------
        # Normalize edges
        # ------------------------------------------------------------
        def norm(e):
            u, v = e
            return (u, v) if u <= v else (v, u)

        mst_norm = [norm(e) for e in mst_edges]

        # ------------------------------------------------------------
        # Node data: A = E \ (F+ u F-) and B' = B - length(F+)
        # ------------------------------------------------------------
        edge_attr = self.edge_attributes  # edge -> (weight, length)

        def get_len(e):
            return edge_attr[e][1]

        fixed = set(getattr(self, "fixed_edges", set()))
        excluded = set(getattr(self, "excluded_edges", set()))

        L_fix = sum(get_len(e) for e in fixed if e in edge_attr)
        Bp = self.budget - L_fix

        # ------------------------------------------------------------
        # Union-find over vertex positions, shared by every structure below.
        #
        # Nodes are addressed by position so every union-find here runs on
        # plain lists.  The dict-of-nodes version copied two dicts per C(.)
        # call, which at 285 calls per separation cost more than the Kruskal
        # walk itself; `list[:]` copies at C speed.
        # ------------------------------------------------------------
        def _find(parent, x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(parent, rank, x, y):
            rx, ry = _find(parent, x), _find(parent, y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                parent[rx] = ry
            elif rank[rx] > rank[ry]:
                parent[ry] = rx
            else:
                parent[ry] = rx
                rank[rx] += 1
            return True

        # ------------------------------------------------------------
        # Node list for the local union-find
        # ------------------------------------------------------------
        def get_nodes():
            if hasattr(self, "graph") and hasattr(self.graph, "nodes"):
                try:
                    return list(self.graph.nodes)
                except Exception:
                    pass

            nodes = set()
            for (u, v) in edge_attr.keys():
                nodes.add(u)
                nodes.add(v)
            for (u, v) in fixed:
                nodes.add(u)
                nodes.add(v)
            return list(nodes)

        # ------------------------------------------------------------
        # Node-invariant scaffolding, cached across the separations at a node.
        #
        # A = E \ (F+ u F-), the vertex indexing and the contraction of F+
        # depend only on (F+, F-) -- not on the tree being separated.  One node
        # separates several trees (one per cut round, plus the strong-branching
        # probes), and rebuilding all of it per call cost an O(m) sweep and an
        # O(m log m) sort every time.  The key is the CONTENTS of F+ and F-,
        # not their identity: a pooled strong-branching solver is handed a new
        # pair of sets per probe and would otherwise keep a stale scaffold.
        # The identity checks on the shared edge arrays catch a move to a
        # different instance.
        #
        # ONE slot, on the class rather than the instance.  Every node keeps
        # its solver alive while it sits in the open list, so an instance
        # attribute would pin a copy of A and its length order -- about a
        # megabyte at n = 400 -- per open node, for a scaffold that is only
        # ever read during that node's own solve.  Separation is strictly
        # sequential (one node at a time, and the strong-branching probes run
        # one after another), so a single slot has the same hit rate at
        # constant total memory, and the scaffold is released as soon as the
        # next node separates.  The key carries no solver identity because it
        # does not need to: two solvers at the same fixings over the same edge
        # arrays have the same scaffold.
        # ------------------------------------------------------------
        edge_list_ref = getattr(self, "edge_list", [])
        cache_key = (frozenset(fixed), frozenset(excluded), self.budget)
        SC = LagrangianMST._sep_cache

        if (SC is None
                or SC["attr"] is not edge_attr
                or SC["elist"] is not edge_list_ref
                or SC["key"] != cache_key):

            # A is not materialised.  Every scan below walks the
            # instance-wide length order and skips what this node has fixed
            # or excluded, through the free mask; building the list, the set
            # and the (length, edge) pairs cost three O(m) passes per node
            # for views that the scans mostly stop early on anyway.
            free_mask_new = self._get_free_mask()
            A_empty = self._cand_idx.size == 0

            NODES_new = get_nodes()
            pos_new = {v: i for i, v in enumerate(NODES_new)}
            n_new = len(NODES_new)

            # F+ is contracted once; every C(.) evaluation starts from this
            # state, and no consumer mutates it -- they all copy first.
            bp_new = list(range(n_new))
            br_new = [0] * n_new
            bc_new = n_new
            dead = A_empty

            if not dead:
                for e in fixed:
                    if e not in edge_attr:
                        dead = True
                        break

                    iu = pos_new.get(e[0])
                    iv = pos_new.get(e[1])
                    if iu is None or iv is None:
                        dead = True
                        break

                    # A cycle inside F+ means no spanning tree contains it; the
                    # node is infeasible and there is nothing to separate.
                    if not _union(bp_new, br_new, iu, iv):
                        dead = True
                        break

                    bc_new -= 1

            SC = {
                "key": cache_key,
                "attr": edge_attr,
                "elist": edge_list_ref,
                "dead": dead,
                "free_mask": free_mask_new,
                "A_desc": None,
                "node_pos": pos_new,
                "n": n_new,
                "bp": bp_new,
                "br": br_new,
                "bc": bc_new,
                # Built on first use only, and only by the tree-completion
                # rung: `literature` and `lemma1` return before ever asking
                # for a completion, and used to pay a Kruskal over all of A on
                # every separation for nothing.
                "pool": None,
            }
            LagrangianMST._sep_cache = SC

        if SC["dead"]:
            return []

        free_mask = SC["free_mask"]
        node_pos = SC["node_pos"]
        NUM_NODES_LOCAL = SC["n"]
        base_parent = SC["bp"]
        base_rank = SC["br"]
        base_components = SC["bc"]

        # Current Lagrangian tree restricted to the admissible unfixed edges.
        _ei = self.edge_indices

        if free_mask is None:
            TcapA = [e for e in mst_norm if e in _ei]
        else:
            TcapA = [
                e for e in mst_norm
                if (_ei.get(e) is not None and free_mask[_ei[e]])
            ]

        # Nothing to separate: the residual tree already respects B'.
        if sum(get_len(e) for e in TcapA) <= Bp + EPS:
            return []

        # ------------------------------------------------------------
        # A in nonincreasing length order.
        #
        # Both liftings want the long edges first and stop at a threshold
        # that rises as edges enter, so they walk this one cached list and
        # break, instead of sorting a candidate set of their own per call: at
        # n = 400 that is one sort of 16k edges per node in place of two per
        # separation, and the scans themselves touch only the tail above the
        # threshold rather than all of A.
        #
        # Stable sort of A in edge_list order, so equal lengths come out in a
        # fixed order rather than in the iteration order of whichever set the
        # caller happened to build.
        # ------------------------------------------------------------
        def _A_desc():
            """A as (length, edge) pairs, nonincreasing in length.

            Ordered off the node's own admissible index array rather than off
            E: after reduced-cost fixing a node at n = 400 typically has a few
            hundred free edges against 16k in the instance, so sorting A costs
            almost nothing while filtering the instance-wide order would walk
            all 16k to yield those few hundred -- on every scan.

            The order is an ascending STABLE sort reversed, so equal lengths
            come out in a fixed order and every scan is reproducible.
            """
            desc = SC["A_desc"]

            if desc is None:
                cand = self._cand_idx
                lengths = self.edge_lengths
                elist = self.edge_list
                asc = cand[np.argsort(lengths[cand], kind="stable")]
                desc = [
                    (float(lengths[i]), elist[i])
                    for i in asc.tolist()[::-1]
                ]
                SC["A_desc"] = desc

            return desc

        # ------------------------------------------------------------
        # Pool for the C(.) completions.
        #
        # C(Q) is a Kruskal over the admissible edges with F+ u Q contracted.
        # Running it over all of A is wasted work: every edge outside the
        # minimum spanning FOREST of (V, A) is rejected by it anyway.  Take
        # f not in that forest.  Global Kruskal skipped f because its ends
        # were already joined by a path P of edges no longer than f, and P
        # lies in the forest.  The completion walks the same lengths in the
        # same order and starts from strictly more merging (F+ u Q is
        # contracted first), so by the time it reaches f the ends of P -- and
        # hence of f -- are connected there too, and f is rejected again.  An
        # edge of P that sits in Q is contracted from the start, which only
        # merges them sooner.
        #
        # So the completion pool is the spanning forest, which is at most
        # n - 1 edges against |A| = O(n^2): at n = 400, density 0.2 that is
        # 399 instead of 15960, for exactly the same C(Q).  Section 6.3 tests
        # one deletion per cover edge per pass, so this is the difference
        # between the `full` rung costing 72% of the solve and costing a few
        # percent.  The lifting steps still range over all of A.
        #
        # Only the tree-completion rung evaluates C(.), so the forest is built
        # on demand: `literature` and `lemma1` return before ever asking, and
        # used to pay a Kruskal over all of A on every separation for nothing.
        # ------------------------------------------------------------
        def _pool():
            pool = SC["pool"]

            if pool is None:
                # Filtering Kruskal on lengths, in C wherever the ordering is
                # done; the plain Python walk of A was one of the larger
                # fixed costs the tree-completion rung paid per node.
                lengths = self.edge_lengths
                elist = self.edge_list
                pool = []

                for i in self.length_forest():
                    e = elist[i]
                    iu = node_pos.get(e[0])
                    iv = node_pos.get(e[1])

                    if iu is None or iv is None:
                        continue

                    pool.append((iu, iv, float(lengths[i]), e))

                SC["pool"] = pool

            return pool

        # ------------------------------------------------------------
        # C(Q): minimum additional LENGTH needed to complete F+ u Q to a
        # spanning tree using admissible edges of A \ Q.
        #
        # `stop_above` is a decision threshold: Kruskal accumulates lengths in
        # nondecreasing order, so once the partial total passes the threshold
        # the final C(Q) does too, and the walk can stop.  The value returned
        # then only certifies "greater than stop_above" -- callers that need the
        # exact C(Q) must omit the threshold.
        # ------------------------------------------------------------
        def completion_mst_cost(Q, stop_above=None):
            Qset = Q if isinstance(Q, (set, frozenset)) else set(Q)

            parent = base_parent[:]
            rank = base_rank[:]
            components = base_components

            # Contract Q. A cycle in F+ u Q means no spanning tree contains it.
            for e in Qset:
                if e not in edge_attr:
                    return float("inf")

                u, v = e
                iu = node_pos.get(u)
                iv = node_pos.get(v)
                if iu is None or iv is None:
                    return float("inf")

                if not _union(parent, rank, iu, iv):
                    return float("inf")

                components -= 1

            if components == 1:
                return 0.0

            total = 0.0

            for iu, iv, le, e in _pool():
                if e in Qset:
                    continue

                if _union(parent, rank, iu, iv):
                    total += le
                    components -= 1

                    if components == 1:
                        return total

                    if stop_above is not None and total > stop_above + EPS:
                        return total

            # F+ u Q cannot be completed to a spanning tree at this node.
            return float("inf")

        # ------------------------------------------------------------
        # Deletion scan in one pass instead of one completion per candidate.
        #
        # `deletable_after` returns, for the current cover Q, the exact
        # M(Q \ {e}) of EVERY e in Q at once.
        #
        # T is the shortest spanning tree containing F+ u Q, i.e. the MST of
        # the graph with those edges priced at -infinity.  Dropping e from the
        # mandatory set only raises one weight, from -infinity back to l_e, and
        # the MST update for raising a tree edge's weight is the textbook one:
        # the tree either keeps e or swaps it for the shortest edge crossing
        # the cut that removing e leaves behind.  So
        #
        #     M(Q \ {e}) = M(Q) - max(0, l_e - repl(e))
        #
        # with repl(e) the shortest non-tree edge across that cut, +inf when e
        # is a bridge.  repl is attained inside COMPLETION_POOL for the same
        # reason the completions are: an edge outside the spanning forest has
        # its ends joined there by shorter edges, and that path has to cross
        # the cut somewhere.
        #
        # All the repl values come from one ascending sweep of the pool with a
        # union-find that climbs T and collapses each tree edge as it is
        # settled -- the standard offline MST-sensitivity pass, O(n a(n)).
        # That replaces |Q| completion Kruskals per pass, which is what made
        # the `full` rung quadratic in n and 72% of the solve at n = 400.
        # ------------------------------------------------------------
        def deletable_after(Q, sumQ):
            Qset = Q if isinstance(Q, (set, frozenset)) else set(Q)

            parent = base_parent[:]
            rank = base_rank[:]
            components = base_components

            # T starts as F+ (already contracted into base) plus Q.
            tree_edges = []

            for e in Qset:
                if e not in edge_attr:
                    return None, float("inf")

                u, v = e
                iu = node_pos.get(u)
                iv = node_pos.get(v)
                if iu is None or iv is None:
                    return None, float("inf")

                if not _union(parent, rank, iu, iv):
                    return None, float("inf")

                components -= 1
                tree_edges.append((iu, iv, get_len(e), e))

            completion_total = 0.0
            in_tree = set(Qset)

            if components > 1:
                for iu, iv, le, e in _pool():
                    if e in Qset:
                        continue

                    if _union(parent, rank, iu, iv):
                        completion_total += le
                        components -= 1
                        tree_edges.append((iu, iv, le, e))
                        in_tree.add(e)

                        if components == 1:
                            break

            if components > 1:
                # F+ u Q cannot be completed to a spanning tree at this node.
                return None, float("inf")

            M_cur = sumQ + completion_total

            # F+ edges are mandatory everywhere and are never scan candidates,
            # but they are part of T and must be climbed through, so add them.
            for e in fixed:
                iu = node_pos.get(e[0])
                iv = node_pos.get(e[1])
                if iu is not None and iv is not None:
                    tree_edges.append((iu, iv, get_len(e), e))
                    in_tree.add(e)

            # Root T and record, for each vertex, the tree edge to its parent.
            adj = [[] for _ in range(NUM_NODES_LOCAL)]
            for iu, iv, le, e in tree_edges:
                adj[iu].append((iv, e))
                adj[iv].append((iu, e))

            par = [-1] * NUM_NODES_LOCAL
            par_edge = [None] * NUM_NODES_LOCAL
            depth = [0] * NUM_NODES_LOCAL
            seen = [False] * NUM_NODES_LOCAL

            for root in range(NUM_NODES_LOCAL):
                if seen[root]:
                    continue
                seen[root] = True
                stack = [root]
                while stack:
                    x = stack.pop()
                    for y, e in adj[x]:
                        if not seen[y]:
                            seen[y] = True
                            par[y] = x
                            par_edge[y] = e
                            depth[y] = depth[x] + 1
                            stack.append(y)

            # `up` collapses vertices whose parent edge already has its repl.
            up = list(range(NUM_NODES_LOCAL))

            def _up_find(x):
                while up[x] != x:
                    up[x] = up[up[x]]
                    x = up[x]
                return x

            repl = {}
            remaining = len(Qset)

            for iu, iv, le, e in _pool():
                if remaining <= 0:
                    break
                if e in in_tree:
                    continue

                a = _up_find(iu)
                b = _up_find(iv)

                while a != b:
                    if depth[a] < depth[b]:
                        a, b = b, a

                    pe = par_edge[a]
                    if pe is None:
                        break

                    if pe in Qset and pe not in repl:
                        repl[pe] = le
                        remaining -= 1

                    up[a] = _up_find(par[a])
                    a = _up_find(a)

            out = {}
            for e in Qset:
                r = repl.get(e)
                le = get_len(e)
                drop = 0.0 if r is None else max(0.0, le - r)
                out[e] = M_cur - drop

            return out, M_cur


        # ------------------------------------------------------------
        # Sequential unit lifting, shared by Lemma 1 and Lemma 2.
        #
        # H starts at the cover and k = |cover| stays fixed. An edge f is
        # accepted when l_f > cap - sigma_{k-1}(H), and sigma_{k-1}(H) is
        # refreshed after every acceptance so that the threshold tracks the
        # growing support. sigma never increases, so the threshold never
        # decreases: once a candidate fails, every shorter one fails too.
        #
        # `candidates` is an iterable of edges in nonincreasing length order,
        # and the scan stops at the first failure, so callers stream the
        # shared descending view of A instead of sorting a candidate set of
        # their own.  Edges already in H are skipped without testing the
        # threshold, exactly as when they were excluded from the candidate set
        # up front.
        #
        # Candidates of equal length are not interchangeable: an acceptance
        # can only lower sigma, so the threshold only rises, and a block of
        # equal-length candidates is cut off after however many the budget
        # allows.  Which ones those are is a tie-break, and the shared view
        # fixes it by edge index so it is the same on every run -- the
        # candidate sets built per call used to leave it to set iteration
        # order, which is why two runs could lift different edges of the same
        # length into a support of the same size.
        # ------------------------------------------------------------
        def unit_lift(cover, cap, candidates):
            H = set(cover)
            k = len(H)

            if k < 1:
                return H

            # Escape hatch for measuring what the lifting is worth to a
            # DUALIZED cut, which contributes only mu * (|T n S| - rhs).
            # Lifting leaves rhs alone and admits the longest admissible
            # edges, which is what the priced tree already avoids, so the
            # suspicion is that |T n S| -- and hence the bound -- does not
            # move while the support, and the cost of carrying it, grows.
            # Off by default: this only exists so the claim can be tested
            # rather than argued.
            if not getattr(self, "lift_cuts", True):
                return H

            lens = sorted(get_len(e) for e in H)
            small = lens[:k - 1]
            sigma = sum(small)

            # sigma_{k-1}(H) is maintained through a MAX-HEAP over the k-1
            # shortest lengths in H, so an acceptance costs O(log k).  The
            # sorted-list form inserted into a list that grows with H, and a
            # lifted support reaches thousands of edges, which made the scan
            # quadratic: 10.6M list.insert calls in one n = 400 run.  An
            # acceptance changes sigma exactly when the new length is
            # strictly below the current (k-1)-th smallest, which is what
            # bisect_left decided before.
            heap = [-x for x in small]
            heapq.heapify(heap)

            for lf, f in candidates:
                if f in H:
                    continue

                if lf <= cap - sigma + EPS:
                    break

                H.add(f)

                if k >= 2:
                    top = -heap[0]
                    if lf < top:
                        sigma += lf - top
                        heapq.heapreplace(heap, -lf)

            return H

        # ------------------------------------------------------------
        # A(Q): admissible edges internal to a component of the forest F+ u Q.
        # They cannot join two components, so they cannot lower the minimum
        # completion length below C(Q).
        #
        # Returned as a membership test rather than a set.  Its only consumer
        # is the Lemma 2 lifting, which walks A from the longest edge down and
        # stops at a length threshold, so materializing A(Q*) meant an O(|A|)
        # union-find sweep per separation to build a set whose short half was
        # never looked at.  The filter is applied lazily along that walk
        # instead; the forest is still built once per call.
        # ------------------------------------------------------------
        def internal_admissible_test(Q):
            parent = base_parent[:]
            rank = base_rank[:]

            for (u, v) in Q:
                iu = node_pos.get(u)
                iv = node_pos.get(v)
                if iu is not None and iv is not None:
                    _union(parent, rank, iu, iv)

            # Flattened once: the Lemma 2 scan tests thousands of candidates
            # per separation, and a list index beats two path-compressing
            # finds per test.
            comp = [_find(parent, i) for i in range(NUM_NODES_LOCAL)]

            def _is_internal(e):
                iu = node_pos.get(e[0])
                iv = node_pos.get(e[1])

                if iu is None or iv is None:
                    return False

                return comp[iu] == comp[iv]

            return _is_internal

        # ------------------------------------------------------------
        # Exact rank lifting.
        #
        # Proposition (exact support certificate).  Let H be admissible with
        # the same vertex components as F+ u Q, and let MSF(H) be a minimum-
        # LENGTH spanning forest of H.  Then
        #
        #     sum_{e in H} x_e <= rank(H) - 1
        #
        # is valid for the node IF AND ONLY IF  l(MSF(H)) + C(H) > B'.
        #
        #   (<=)  a tree with rank(H) edges of H contains a basis of H; every
        #         basis of H spans the same components, so its length is at
        #         least l(MSF(H)) and completing it costs at least C(H).
        #   (=>)  MSF(H) plus its cheapest completion IS a spanning tree with
        #         rank(H) edges of H and length exactly l(MSF(H)) + C(H).
        #
        # With H = Q this is Proposition 1.  Used as a LIFTING rule it is
        # strictly finer than Lemma 2: that lemma asks l_f > B* - sigma_{k-1},
        # a GLOBAL threshold which makes an edge earn its place against the
        # LONGEST edge of the cover, whereas the certificate above only
        # charges f what it would actually cost the forest,
        #
        #     d(f) = max(0, pi(f) - l_f),   pi(f) = longest REMOVABLE edge on
        #                                           f's path in F+ u Q,
        #
        # and the certificate survives as long as the accepted d(f) sum to
        # less than the initial slack l(Q) + C(Q) - B'.  Charging the path
        # maximum instead of the support maximum admits every edge whose own
        # path is cheap -- the SHORT internal edges Lemma 2 always rejects,
        # which are exactly the edges a priced tree would otherwise reconnect
        # through, so they are the ones that raise the cut's dual price (see
        # exact_cut_dual_ascent).
        #
        # The forest is NOT re-optimised as edges are accepted, which is
        # conservative and therefore safe.  Adding one edge f to a graph drops
        # the minimum spanning forest by exactly max(0, pi'(f) - l_f) with
        # pi'(f) the path maximum in the CURRENT forest, and a minimum
        # spanning forest is also a minimum-bottleneck one, so pi'(f) is the
        # smallest path maximum any subgraph of H can offer and in particular
        # pi'(f) <= pi_Q(f).  Each accepted edge therefore costs the forest at
        # most the d(f) charged here, and by induction over the accepted edges
        #
        #     l(MSF_l(H))  >=  l(Q) - sum_f d(f),
        #
        # which is what the certificate needs.  (Brualdi's basis-exchange
        # bijection gives the same bound; the bottleneck argument is used
        # because it needs nothing beyond the MST optimality condition.)
        #
        # The strictness matters: accepting on d <= slack instead of
        # d < slack makes the certificate hold with equality, l(MSF) + C = B',
        # and produces genuinely invalid cuts -- n = 8, F+ = {(0,6),(3,6),
        # (4,6)}, F- = {(1,5)}, B = 13, Q = {(0,1),(2,7),(6,7)} with
        # l(Q) = 8, C(Q) = 1, B' = 8: charging (2,4) its d = 1 against a
        # slack of 1 admits it, and the budget-feasible tree
        # F+ u {(0,1),(6,7),(2,4),(2,5)} of length exactly 13 then carries
        # three support edges against a right-hand side of two.
        # ------------------------------------------------------------
        def rank_lift(Q, C_Q, candidates):
            LagrangianMST.rank_lift_calls += 1
            H = set(Q)

            if not Q:
                return H

            # Same escape hatch as unit_lift.
            if not getattr(self, "lift_cuts", True):
                return H

            slack = float(sum(get_len(e) for e in Q)) + float(C_Q) - float(Bp)

            if not (slack > 0.0) or math.isinf(slack):
                return H

            NEG = float("-inf")
            adj = {}

            def _adj_add(u, v, w):
                adj.setdefault(u, []).append((v, w))
                adj.setdefault(v, []).append((u, w))

            # F+ edges carry -inf: a forced edge cannot be the one that
            # leaves, so it never sets pi(f).
            for e in fixed:
                if e in edge_attr:
                    _adj_add(e[0], e[1], NEG)

            for e in Q:
                _adj_add(e[0], e[1], get_len(e))

            par = {}
            pw = {}
            depth = {}

            for root in adj:
                if root in depth:
                    continue

                depth[root] = 0
                par[root] = None
                pw[root] = NEG
                stack = [root]

                while stack:
                    x = stack.pop()
                    for (y, w) in adj[x]:
                        if y not in depth:
                            depth[y] = depth[x] + 1
                            par[y] = x
                            pw[y] = w
                            stack.append(y)

            l_min = min(get_len(e) for e in Q)

            for lf, f in candidates:
                # pi(f) >= l_min whenever f's path carries a Q edge, so once
                # l_f drops this far no candidate can be charged less than the
                # slack that is left.
                if lf <= l_min - slack + EPS:
                    break

                if f in H:
                    continue

                a = f[0]
                b = f[1]

                if a not in depth or b not in depth:
                    continue

                pi = NEG
                broke = False

                while a != b:
                    if depth[a] < depth[b]:
                        a, b = b, a

                    w = pw[a]
                    if w > pi:
                        pi = w

                    nxt = par[a]

                    if nxt is None:
                        broke = True
                        break

                    a = nxt

                if broke:
                    # Different components: not internal to F+ u Q.
                    continue

                d = pi - lf

                if d < 0.0 or pi == NEG:
                    d = 0.0

                if d < slack - EPS:
                    H.add(f)
                    slack -= d

            return H

        # ------------------------------------------------------------
        # Static dominance-based extension of Agra et al. [5, 6], used by the
        # "literature" rung: admissible edges at least as long as the longest
        # cover edge that close a cycle with the cover.
        #
        # The cycle test is taken with respect to the forest the cover induces
        # on its own, not F+ u cover.  At the root, where [5, 6] operate, the
        # two coincide; deeper in the tree this is the stricter of the two, so
        # it cannot flatter the baseline.  Every edge it admits is internal to a
        # component of F+ u cover as well, so Lemma 2 still covers validity.
        #
        # By Proposition 2 these are exactly the edges (23) accepts at its first
        # step, and admitting them leaves sigma_{k-1} unchanged, which is why
        # the whole set can be added at once without re-evaluating a threshold.
        # ------------------------------------------------------------
        def dominance_extend(cover):
            C = set(cover)

            if not C:
                return C

            l_max = max(get_len(e) for e in C)

            cover_parent = {x: x for e in C for x in e}

            def cover_find(x):
                while cover_parent[x] != x:
                    cover_parent[x] = cover_parent[cover_parent[x]]
                    x = cover_parent[x]
                return x

            for (u, v) in C:
                ru, rv = cover_find(u), cover_find(v)
                if ru != rv:
                    cover_parent[ru] = rv

            # The threshold is fixed at l_max, so the descending view can be
            # cut off the moment it drops below it: only the long-edge prefix
            # is ever looked at, where the old flat sweep read all of A.
            out = set(C)

            for _le, e in _A_desc():
                if _le < l_max - EPS:
                    break

                if (e not in C
                        and e[0] in cover_parent
                        and e[1] in cover_parent
                        and cover_find(e[0]) == cover_find(e[1])):
                    out.add(e)

            return out

        # ------------------------------------------------------------
        # Deduplication and dominance filtering, shared by every rung.
        # ------------------------------------------------------------
        def _finalize(cand):
            uniq = {}

            for cset, rhs in cand:
                if rhs < 0 or len(cset) <= rhs:
                    # Redundant: the support cannot exceed the right-hand side.
                    continue

                # A frozenset is hashable and caches its hash, so it keys the
                # deduplication directly; sorting a support of several
                # thousand edges to build a tuple key was wasted work, and
                # `uniq` iterates in insertion order either way.
                key = cset if isinstance(cset, frozenset) else frozenset(cset)
                best = uniq.get(key)

                if best is None or rhs < best[1]:
                    uniq[key] = (key, int(rhs))

            # sum_{S} x <= b implies sum_{S'} x <= b' whenever S' is a subset of
            # S and b <= b', so the strongest candidates are offered first:
            # smallest right-hand side, then largest support.
            out = []

            for cset, rhs in sorted(uniq.values(), key=lambda t: (t[1], -len(t[0]))):
                implied = any(
                    cset <= dset and drhs <= rhs
                    for dset, drhs in out
                )

                if not implied:
                    out.append((cset, rhs))

            return out

        # ============================================================
        # 6.1 Seed cover
        # ============================================================
        S_seed = []
        sum_seed = 0.0

        # Nonincreasing length, ties in the order T n A presents them -- the
        # same order `sorted(key=get_len, reverse=True)` gave, since a reverse
        # sort keeps equal elements in their original relative order.
        _tsorted = [(edge_attr[e][1], i, e) for i, e in enumerate(TcapA)]
        _tsorted.sort(key=lambda t: (-t[0], t[1]))

        for le, _i, e in _tsorted:
            S_seed.append(e)
            sum_seed += le

            if sum_seed > Bp + EPS:
                break

        if not S_seed or sum_seed <= Bp + EPS:
            return []

        # ------------------------------------------------------------
        # Dual-gain-maximising seed (optional, `dual_seed`).
        #
        # A cover made of TREE edges has T n S = S, so its exact dual price
        # (see exact_cut_dual_ascent) is decided by the WORST edge in S:
        #
        #     mu*(S) = min_{e in S} (cheapest replacement outside S) - c_e.
        #
        # Longest-first minimises |S|, which is what the propagation wants,
        # but it says nothing about that minimum -- a single tree edge with an
        # alternative optimal swap drags mu* to zero and the cover cannot move
        # the bound at all.  Sorting the same candidates by margin instead
        # maximises min_{e in S} margin(e) subject to the very same residual-
        # cover condition, and one sweep of tree_edge_margins scores both
        # orders, so the stronger seed can simply be taken.
        # ------------------------------------------------------------
        if bool(getattr(self, "dual_seed", False)) and len(TcapA) > 1:
            try:
                _tidx = getattr(self, "_last_mst_idx", None)
                _cvec = getattr(self, "_last_mw", None)

                if _tidx is not None and _cvec is not None:
                    _mg = self.tree_edge_margins(np.asarray(_cvec, dtype=float), _tidx)
                else:
                    _mg = None

                if _mg:
                    _ei = self.edge_indices
                    _keyed = []

                    for e in TcapA:
                        j = _ei.get(e)
                        if j is None or j not in _mg:
                            _keyed = None
                            break
                        _keyed.append((_mg[j], edge_attr[e][1], e))

                    if _keyed:
                        # Margin-first prefix that still exceeds B'.
                        _keyed.sort(key=lambda t: (-t[0], -t[1]))
                        S_alt = []
                        _tot = 0.0

                        for _m, _le, e in _keyed:
                            S_alt.append(e)
                            _tot += _le

                            if _tot > Bp + EPS:
                                break

                        if _tot > Bp + EPS:
                            _lo_seed = min(_mg[_ei[e]] for e in S_seed)
                            _lo_alt = min(_mg[_ei[e]] for e in S_alt)

                            if _lo_alt > _lo_seed + EPS:
                                S_seed = S_alt
                                sum_seed = _tot
            except Exception:
                # A better seed is an optimisation, never a requirement.
                pass

        cuts = []

        if strengthening == "literature":
            # [39] + [5, 6]: seed cover with the static dominance extension.
            # The seed passes the tree-completion membership test outright,
            # since its own length already exceeds B'.
            cuts.append((dominance_extend(S_seed), len(S_seed) - 1))
            return _finalize(cuts)

        # ============================================================
        # 6.2 Lifted residual cover: rhs stays |S| - 1
        # ============================================================
        # S_seed is already in H, so streaming all of A is the same
        # candidate set as A \ S_seed.
        cuts.append((unit_lift(S_seed, Bp, _A_desc()), len(S_seed) - 1))

        if strengthening == "lemma1":
            # Sequential unit lifting only: no contraction, no Lemma 2.
            return _finalize(cuts)

        # ============================================================
        # 6.3 / 6.4 Lifted tree-completion cover: rhs stays |Q*| - 1
        # ============================================================
        Q_star = set(S_seed)

        # Running state for the deletion scan.  `sum_Q` is kept incrementally,
        # and the certificate below is carried between candidates.
        #
        # M is monotone and 1-Lipschitz downwards in a deletion:
        #   M(Q \ {e}) >= M(Q) - l_e.
        # (Take the shortest spanning tree T' for Q \ {e}.  If e is in T' then
        # M(Q) <= M(Q\{e}) outright; otherwise T' + e closes a cycle carrying
        # an edge f outside F+ u Q, and T' + e - f contains F+ u Q with length
        # at most len(T') + l_e.)  So M(Q_star) - l_e > B' already certifies
        # Q_star \ {e}, and the completion walk can be skipped for it -- which
        # is exactly the short edges the scan tries first.
        sum_Q = float(sum(get_len(e) for e in Q_star))

        # One pass of `deletable_after` gives the exact M(Q0 \ {x}) for every
        # x in the Q0 it ran at.  A deletion moves Q_star off Q0, and
        # recomputing after each one made this the single most expensive
        # function in the solver (21 sensitivity passes per node at n = 200).
        #
        # Most of those passes are avoidable, because the pass's values stay
        # usable after the set moves.  Writing D for the edges deleted since
        # the pass and D' = D u {e} for the deletion under test, repeated
        # application of 1-Lipschitz from any single anchor x in D' gives
        #
        #     M(Q0 \ D') >= M(Q0 \ {x}) - sum_{y in D', y != x} l_y
        #                 = exact[x] + l_x - sum_{y in D'} l_y,
        #
        # so the best certificate the pass supports is the largest anchor:
        #
        #     M(Q_star \ {e}) >= max_{x in D'} (exact[x] + l_x)
        #                        - (sum_{y in D} l_y + l_e).
        #
        # Since exact[x] + l_x = M(Q0) + min(l_x, repl(x)) >= M(Q0), every
        # anchor is at least as strong as chaining plain M(Q) - l_e steps off
        # M(Q0), which is what the carried bound used to do -- it effectively
        # pinned x to the first deletion after the pass and threw the rest of
        # the pass away.  `anchor` keeps the running maximum instead, so each
        # further deletion can only strengthen the certificate.  A pass is
        # needed only when even that fails, and the scan tries the shortest
        # edges first, which is exactly where the certificate bites.
        exact = None            # x -> M(Q0 \ {x}) from the last pass
        M0 = None               # M(Q0) at that pass
        del_sum = 0.0           # total length deleted since that pass
        anchor = float("-inf")  # max over deleted x of exact[x] + l_x
        no_tree = False

        # Effort cap on the deletion scan.  Every pass is an O(n) MST
        # sensitivity computation and the scan needs one whenever the carried
        # certificate is not enough; at n = 400 that reached 15 passes per
        # node and made this the single most expensive function in the
        # solver.  Stopping early only leaves Q* LARGER -- the cut is then
        # weaker, never invalid, since every Q the scan certifies stays
        # certified -- so this trades a little cut strength for the time to
        # separate more nodes.  Set it to None for the uncapped scan.
        _refresh_cap = getattr(self, "max_completion_refresh", 6)
        _refresh_budget = [float("inf") if _refresh_cap is None else int(_refresh_cap)]

        def _refresh():
            nonlocal exact, M0, del_sum, anchor, no_tree
            exact, M_val = deletable_after(Q_star, sum_Q)
            no_tree = exact is None
            M0 = M_val
            del_sum = 0.0
            anchor = float("-inf")
            _refresh_budget[0] -= 1

        def _lb_after(e, le):
            """Lower bound on M(Q_star \\ {e}); exact while no deletion has
            happened since the last pass."""
            if exact is None:
                return float("-inf")

            ex = exact.get(e)

            if del_sum == 0.0:
                # Q_star is still the Q0 of the pass: the value is exact.
                return float("-inf") if ex is None else ex

            best = anchor

            if ex is not None and ex + le > best:
                best = ex + le

            return best - (del_sum + le)

        _refresh()

        changed = True
        while changed and len(Q_star) > 1:
            changed = False

            for e in sorted(Q_star, key=get_len):
                if len(Q_star) <= 1:
                    break

                if e not in Q_star:
                    continue

                le = get_len(e)

                # `no_tree` means F+ u Q_star admits no spanning tree, so
                # M = +inf and the deletion is certified.  It must be
                # re-established after each deletion rather than latched: the
                # claim "no spanning tree for Q implies none for Q \ {e}" is
                # only sound when the obstruction is connectivity (dropping a
                # mandatory edge returns it to the completion pool, which
                # leaves reachability unchanged).  If the obstruction were a
                # cycle inside F+ u Q, removing e could break it and the
                # deletion would NOT be certified -- so the flag is refreshed
                # with the rest of the state instead of being trusted forever.
                if no_tree:
                    _refresh()

                if no_tree:
                    ok = True
                else:
                    lb = _lb_after(e, le)

                    if lb > Bp + EPS:
                        # Certified with no pass: either the exact value from
                        # a current pass, or the anchored bound above.
                        ok = True
                    elif del_sum == 0.0:
                        # The pass is current, so `lb` was the exact value and
                        # this edge genuinely cannot be deleted.
                        ok = False
                    elif _refresh_budget[0] <= 0:
                        # Out of passes: keep the edge rather than pay for a
                        # certificate.  Q* stays valid, just larger.
                        ok = False
                    else:
                        _refresh()
                        ok = (
                            True if no_tree
                            else exact.get(e, float("-inf")) > Bp + EPS
                        )

                if not ok:
                    continue

                Q_star = Q_star - {e}
                sum_Q -= le

                # The pass's values stay usable; only the anchor and the
                # deleted total move with the set.
                del_sum += le

                if exact is not None:
                    ex = exact.get(e)

                    if ex is not None and ex + le > anchor:
                        anchor = ex + le

                changed = True

        if len(Q_star) > 1:
            # When the last pass ran at the Q_star the scan ended on -- no
            # deletion since, i.e. del_sum == 0 -- it already computed
            # M(Q_star) = sum_Q + C(Q_star) exactly, so C_star can be read off
            # it instead of contracting Q_star and walking the pool a second
            # time.  After a certified deletion only a lower bound is in hand,
            # so fall back to the full computation there.
            if del_sum == 0.0 and M0 is not None and not math.isinf(M0):
                C_star = M0 - sum_Q
            else:
                C_star = completion_mst_cost(Q_star)

            B_star = Bp - C_star

            if C_star != float("inf") and B_star > -EPS:
                _is_internal = internal_admissible_test(Q_star)

                if bool(getattr(self, "rank_lift", True)):
                    # Exact support certificate instead of Lemma 2's global
                    # sigma threshold; see rank_lift.
                    H_tc = rank_lift(
                        Q_star,
                        C_star,
                        ((lf, f) for lf, f in _A_desc() if _is_internal(f)),
                    )
                else:
                    H_tc = unit_lift(
                        Q_star,
                        B_star,
                        ((lf, f) for lf, f in _A_desc() if _is_internal(f)),
                    )
            else:
                # C(Q*) alone already exhausts B': the node is infeasible and
                # lifting would accept every internal edge, so keep Q* as is.
                H_tc = set(Q_star)

            cuts.append((H_tc, len(Q_star) - 1))

        return _finalize(cuts)

    
    
    @classmethod
    def reset_cut_stats(cls):
        cls.cuts_separated = 0
        cls.cuts_infeasible = 0
        cls.cut_nodes = 0
        cls.exact_dual_nodes = 0
        cls.exact_dual_gain = 0.0
        cls.lr_iterations = 0
        cls.sep_calls = 0
        cls.sep_probe_calls = 0
        cls.sep_time = 0.0
        cls.sep_max_depth = -1
        cls.rank_lift_calls = 0
        cls.exact_dual_probe_nodes = 0
        cls.indicator_calls = 0
        cls.indicator_none = 0
        cls.indicator_time = 0.0
        cls.pool_hist = {}

    def compute_modified_weights(self):
        """Edge weights priced by the current multipliers:
        w_e + lambda*l_e + sum_i mu_i [e in S_i]."""
        lam = max(0.0, min(getattr(self, "lmbda", 0.0), 1e4))

        if lam:
            base = self.edge_weights + lam * self.edge_lengths
        else:
            base = self.edge_weights.copy()

        cuts = self.best_cuts if self.use_cover_cuts else None

        if not cuts:
            self._mw_cached = None
            self._mw_lambda = lam
            self._mw_mu = None
            self._mw_free_mask_key = None
            return base

        # No cut carries a price -- which is what the whole lambda phase runs
        # with -- so the priced vector IS the lambda-priced base.
        _mults = self.best_cut_multipliers
        if not _mults or not any(v > 0.0 for v in _mults.values()):
            self._mw_cached = None
            self._mw_lambda = lam
            self._mw_mu = None
            self._mw_free_mask_key = None
            return base

        cut_idxs_free = getattr(self, "_cut_edge_idx", None)  # FREE indices only
        mu_len = len(cuts)

        mu = np.fromiter(
            (max(0.0, min(self.best_cut_multipliers.get(i, 0.0), 1e4))
             for i in range(mu_len)),
            dtype=np.float64,
            count=mu_len,
        )

        free_mask_key = self._free_mask_key

        if (self._mw_cached is not None
                and self._mw_lambda == lam
                and self._mw_mu is not None
                and self._mw_mu.shape == mu.shape
                and np.array_equal(self._mw_mu, mu)
                and self._mw_free_mask_key == free_mask_key):
            return self._mw_cached

        # `base` is already a private array, so the cut prices go straight in.
        if cut_idxs_free is not None:
            for i, idxs in enumerate(cut_idxs_free):
                m = mu[i]
                if m > 0.0 and idxs.size:
                    base[idxs] += m
        else:
            for i, (cut, _) in enumerate(cuts):
                m = mu[i]
                if m <= 0.0:
                    continue
                for e in cut:
                    j = self.edge_indices.get(e)
                    if j is not None:
                        base[j] += m

        self._mw_cached = base
        self._mw_lambda = lam
        self._mw_mu = mu
        self._mw_free_mask_key = free_mask_key
        return base

    def _invalidate_weight_cache(self):
        # The free mask is NOT cleared here: it is a function of the node's
        # fixings, which _rebuild_candidates owns, not of the multipliers.
        self._mw_cached = None
        self._mw_lambda = None
        self._mw_mu = None
        self._mw_free_mask_key = None

   
    def _get_free_mask(self):
        """Boolean mask over E: True where the edge is free at this node.

        None means nothing is fixed or excluded, i.e. every edge is free.
        Maintained by _rebuild_candidates, which runs exactly when the
        fixings change.
        """
        return self._free_mask_cache

    def _append_with_cap(self, bucket, item, cap):
        bucket.append(item)
        overflow = len(bucket) - cap
        if overflow > 0:
            del bucket[:overflow]

    def _record_primal_solution(self, mst_edges, feasible):
        # snapshot = tuple(sorted(mst_edges)) if mst_edges else ()
        snapshot = tuple(mst_edges) if mst_edges else ()

        self._append_with_cap(
            self.primal_solutions,
            (snapshot, bool(feasible)),
            self._primal_history_cap,
        )

    
    def _record_fractional_solution(self, fractional_solution):
        if not fractional_solution:
            lightweight = ()
        else:
            lightweight = tuple(
                heapq.nlargest(20, fractional_solution.items(), key=lambda kv: abs(kv[1]))
            )
        self._append_with_cap(self.fractional_solutions, lightweight, self._fractional_history_cap)


    def _record_subgradient(self, value):
        self._append_with_cap(
            self.subgradients,
            float(value),
            self._subgradient_history_cap,
        )




    def _mst_core(self, weights):
        """Minimum spanning tree over the admissible edges, F+ forced in.

        Filter-Kruskal.  Instead of ordering all |A| admissible edges, the
        cheapest block is taken, Kruskal is run over it, and then ONE
        vectorised component-label test drops every remaining edge whose ends
        that block has already joined -- exactly the edges Kruskal would
        reject, never looked at again.  Two or three blocks finish the tree.

        The edges are still consumed in the global (weight, index) order a
        stable full sort gives, so the tree, its cost and its tie-breaking are
        what the full sort produced:

          * np.partition splits the candidates at the (block+1)-th smallest
            weight t.  Everything strictly below t goes into the block, and
            the block is topped up from the edges EQUAL to t in ascending
            index order -- the order a stable sort would have put them in.
          * So every block edge precedes every remaining edge under
            (weight, index), and argsort(kind="stable") orders within the
            block by the same key.

        At n = 400, density 0.2 this walks ~2k edges in Python per MST where
        the full sort walked 16k, and sorts 1.6k instead of 16k -- and the
        dual runs one MST per subgradient iteration.

        Returns (cost, length, edges, indices), or (inf, inf, [], None) when
        the node admits no spanning tree.
        """
        n = self.num_nodes
        need = n - 1
        parent = list(range(n))
        size = [1] * n
        mst_idx = []
        edge_list = self.edge_list

        # F+ first.  A cycle among the fixed edges means the node is empty.
        for i in self.fixed_edge_indices:
            u, v = edge_list[i]
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            if u == v:
                self._last_mst_idx = None
                self._last_mst_list = None
                return float("inf"), float("inf"), [], None
            if size[u] < size[v]:
                u, v = v, u
            parent[v] = u
            size[u] += size[v]
            mst_idx.append(i)

        cand = self._cand_idx
        block = self._kruskal_block
        EU = self._edge_u
        EV = self._edge_v

        while len(mst_idx) < need and cand.size:
            if cand.size <= block:
                order = cand[np.argsort(weights[cand], kind="stable")]
                rest = None
            else:
                # No fancy-index copy when nothing is fixed or excluded: the
                # candidate array IS the whole index range there, and this
                # runs once per subgradient iteration.
                wc = weights if cand is LagrangianMST._all_idx else weights[cand]
                t = np.partition(wc, block)[block]
                sel = wc < t
                nsel = int(np.count_nonzero(sel))
                if nsel < block:
                    eq = np.flatnonzero(wc == t)
                    sel[eq[: block - nsel]] = True
                sub = cand[sel]
                rest = cand[~sel]
                order = sub[np.argsort(weights[sub], kind="stable")]

            for i in order.tolist():
                u, v = edge_list[i]
                while parent[u] != u:
                    parent[u] = parent[parent[u]]
                    u = parent[u]
                while parent[v] != v:
                    parent[v] = parent[parent[v]]
                    v = parent[v]
                if u == v:
                    continue
                if size[u] < size[v]:
                    u, v = v, u
                parent[v] = u
                size[u] += size[v]
                mst_idx.append(i)
                if len(mst_idx) == need:
                    break

            if rest is None or len(mst_idx) == need:
                break

            labels = [0] * n
            for x in range(n):
                r = x
                while parent[r] != r:
                    parent[r] = parent[parent[r]]
                    r = parent[r]
                labels[x] = r

            lab = np.asarray(labels, dtype=np.int32)
            cand = rest[lab[EU[rest]] != lab[EV[rest]]]

        if len(mst_idx) != need:
            self._last_mst_idx = None
            self._last_mst_list = None
            return float("inf"), float("inf"), [], None

        idx = np.asarray(mst_idx, dtype=np.int64)
        tree = [edge_list[i] for i in mst_idx]

        # Identity tag, so compute_real_weight_length can price this tree from
        # `idx` instead of walking it back through the edge-index dict.
        self._last_mst_idx = idx
        self._last_mst_list = tree

        cost = float(weights[idx].sum())
        length = float(self.edge_lengths[idx].sum())

        return cost, length, tree, idx

    def length_forest(self):
        """Minimum spanning FOREST of (V, A) by LENGTH, as edge indices in
        nondecreasing length.

        This is the completion pool of Section 6.3: every admissible edge
        outside it is rejected by every C(.) evaluation, so the completions
        only ever need these at most n-1 edges.  Same filtering Kruskal as
        _mst_core -- on lengths, with nothing forced in and the search
        allowed to end on a forest -- instead of a Python walk of all of A.
        """
        n = self.num_nodes
        need = n - 1
        parent = list(range(n))
        size = [1] * n
        out = []

        cand = self._cand_idx
        L = self.edge_lengths
        block = self._kruskal_block
        EU = self._edge_u
        EV = self._edge_v
        edge_list = self.edge_list

        while len(out) < need and cand.size:
            if cand.size <= block:
                order = cand[np.argsort(L[cand], kind="stable")]
                rest = None
            else:
                wc = L if cand is LagrangianMST._all_idx else L[cand]
                t = np.partition(wc, block)[block]
                sel = wc < t
                nsel = int(np.count_nonzero(sel))
                if nsel < block:
                    eq = np.flatnonzero(wc == t)
                    sel[eq[: block - nsel]] = True
                sub = cand[sel]
                rest = cand[~sel]
                order = sub[np.argsort(L[sub], kind="stable")]

            for i in order.tolist():
                u, v = edge_list[i]
                while parent[u] != u:
                    parent[u] = parent[parent[u]]
                    u = parent[u]
                while parent[v] != v:
                    parent[v] = parent[parent[v]]
                    v = parent[v]
                if u == v:
                    continue
                if size[u] < size[v]:
                    u, v = v, u
                parent[v] = u
                size[u] += size[v]
                out.append(i)
                if len(out) == need:
                    break

            if rest is None or len(out) == need:
                break

            labels = [0] * n
            for x in range(n):
                r = x
                while parent[r] != r:
                    parent[r] = parent[parent[r]]
                    r = parent[r]
                labels[x] = r

            lab = np.asarray(labels, dtype=np.int32)
            cand = rest[lab[EU[rest]] != lab[EV[rest]]]

        return out

    def custom_kruskal(self, modified_weights):
        cost, length, edges, _ = self._mst_core(np.asarray(modified_weights))
        return cost, length, edges

    def incremental_kruskal(self, prev_weights, prev_mst_edges, current_weights):
        # The name is historical.  Lambda moves on every subgradient
        # iteration, so EVERY edge weight changes and the "incremental"
        # candidate set was in fact the whole edge list; the filtering Kruskal
        # gives the same tree for a fraction of the Python work.
        cost, length, edges, _ = self._mst_core(current_weights)
        return cost, length, edges

    def _argsort_kruskal(self, weights):
        cost, length, edges, _ = self._mst_core(np.asarray(weights))
        return cost, length, edges

    def compute_mst(self, modified_edges=None):
        start_time = time()

        if modified_edges is not None:
            weights = np.array([w for _, _, w in modified_edges], dtype=np.float64)
        else:
            weights = self.compute_modified_weights()

        mst_cost, mst_length, mst_edges, _ = self._mst_core(weights)

        if self.verbose:
            print(f"MST computed (no cache): length={mst_length:.2f}")

        self.last_mst_edges = mst_edges

        LagrangianMST.total_compute_time += time() - start_time
        return mst_cost, mst_length, mst_edges

    def compute_mst_incremental(self, prev_weights, prev_mst_edges):
        # Current modified weights, computed ONCE and handed back through
        # _last_mw so solve() does not recompute them for the next step.
        current_weights = self.compute_modified_weights()
        self._last_mw = current_weights

        if prev_weights is None or prev_mst_edges is None:
            if self.verbose:
                print("Incremental MST: no previous MST, using full Kruskal")
            return self.custom_kruskal(current_weights)

        # Lambda and mu sometimes do not move -- a clamped step, or a phase
        # that has converged.  Re-pricing the previous tree is O(n) against a
        # whole Kruskal.
        if current_weights is prev_weights or np.array_equal(current_weights, prev_weights):
            idx = self._last_mst_idx
            if idx is not None and len(idx) == self.num_nodes - 1:
                mst_cost = float(current_weights[idx].sum())
                mst_length = float(self.edge_lengths[idx].sum())
                if self.verbose:
                    print(f"Incremental MST: reusing previous MST, length={mst_length:.2f}")
                return mst_cost, mst_length, prev_mst_edges

        return self.incremental_kruskal(prev_weights, prev_mst_edges, current_weights)

    def solve(self, inherited_cuts=None, inherited_multipliers=None, depth=0, node=None):
        start_time = time()
        self.depth = depth
        
        # --- robust normalization of inherited_cuts (accept pairs or indices) ---
        edge_indices = self.edge_indices
        # Node-invariant, built once per instance in __init__.  Rebuilding it
        # here cost an m-entry dict per node AND per strong-branching probe:
        # 1.3ms at n = 400, 4% of a whole run.
        idx_to_edge = self.idx_to_edge

        def _norm_edge(e):
            if not (isinstance(e, tuple) and len(e) == 2):
                return None
            u, v = e
            t = (u, v) if u <= v else (v, u)
            return t if t in edge_indices else None

        def _iter_edges_any(cut_like):
            if isinstance(cut_like, tuple) and len(cut_like) == 2:
                e = _norm_edge(cut_like); 
                if e is not None: yield e
                return
            if isinstance(cut_like, int):
                e = _norm_edge(idx_to_edge.get(int(cut_like)))
                if e is not None: yield e
                return
            try:
                for item in cut_like:
                    if isinstance(item, int):
                        e = _norm_edge(idx_to_edge.get(int(item)))
                    elif isinstance(item, tuple) and len(item) == 2:
                        e = _norm_edge(item)
                    elif isinstance(item, (list, set, frozenset)) and len(item) == 2:
                        a, b = tuple(item); e = _norm_edge((a, b))
                    else:
                        e = None
                    if e is not None: 
                        yield e
            except TypeError:
                return

        # Inherited cuts arrive from MSTNode.create_children, which has already
        # normalised every edge and dropped anything outside the edge list, so
        # the general walk above re-derives what it is handed.  That cost 9% of
        # the whole solve at n=200 (4.6M _iter_edges_any calls in one run).
        # A set intersection settles it at C speed, and when it does not come
        # out whole the input was not already normalised and the slow path
        # still runs.
        edge_key_set = self._edge_key_set

        def _norm_pair(pair):
            cut_like, rhs_like = pair

            # Every cut this solver sees was built by generate_cover_cuts or
            # projected from one, so it is already a frozenset of normalised
            # edges belonging to this instance.  Taking that at face value
            # removes an O(|S|) rebuild per cut per node, and a lifted
            # support runs to thousands of edges.
            if isinstance(cut_like, frozenset):
                return (cut_like, int(rhs_like))

            if isinstance(cut_like, set):
                kept = cut_like & edge_key_set
                if len(kept) == len(cut_like):
                    return (frozenset(kept), int(rhs_like))

            return (frozenset(_iter_edges_any(cut_like)), int(rhs_like))

        if inherited_cuts:
            self.best_cuts = [_norm_pair(p) for p in inherited_cuts]
            self.best_cut_multipliers = (inherited_multipliers or {}).copy()
        else:
            self.best_cuts = []
            self.best_cut_multipliers = {}
        self.best_cut_multipliers_for_best_bound = self.best_cut_multipliers.copy()


        # --- robust normalization of inherited_cuts (accept pairs or indices) ---
    
            
        prev_weights = None
        prev_mst_edges = None

       
        if self.use_bisection:
        # Validate graph and edges
            if not self.edges or not nx.is_connected(self.graph):
                if self.verbose:
                    print(f"Error at depth {depth}: Empty edge list or disconnected graph in bisection path")
                return self.best_lower_bound, self.best_upper_bound, []
            

        # else:  # Subgradient method with Polyak hybrid + cover cuts (λ, μ), depth-based freezing
        #     # --- Tunables / safety limits ---
        #     MAX_SOLUTIONS    = getattr(self, "max_primal_solutions", 50)
        #     max_iter         = min(self.max_iter, 200)

        #     # Polyak / momentum for λ
        #     self.momentum_beta = getattr(self, "momentum_beta", 0.9)
        #     gamma_base         = getattr(self, "gamma_base", 0.1)

        #     # μ update parameters
        #     gamma_mu         = getattr(self, "gamma_mu", 0.30)
        #     mu_increment_cap = getattr(self, "mu_increment_cap", 1.0)
        #     eps              = 1e-12

        #     # Depth-based behaviour
        #     max_cut_depth = getattr(self, "max_cut_depth", 30)   # where we ADD cuts
        #     max_mu_depth  = getattr(self, "max_mu_depth", 50)    # where we UPDATE μ / use cuts in dual
        #     is_root       = (depth == 0)

        #     # Node-level separation parameters
        #     max_active_cuts           = getattr(self, "max_active_cuts", 5)
        #     max_new_cuts_per_node     = getattr(self, "max_new_cuts_per_node", 5)
        #     min_cut_violation_for_add = getattr(self, "min_cut_violation_for_add", 1.0)
        #     dead_mu_threshold         = getattr(self, "dead_mu_threshold", 1e-6)

        #     # Extra iterations allowed at root
        #     root_max_iter = int(getattr(self, "root_max_iter", max_iter * 2))

        #     # Ensure cut structures exist
        #     if not hasattr(self, "best_cuts"):
        #         self.best_cuts = []   # list of (set(edges), rhs)
        #     if not hasattr(self, "best_cut_multipliers"):
        #         self.best_cut_multipliers = {}  # μ_i for each cut
        #     if not hasattr(self, "best_cut_multipliers_for_best_bound"):
        #         self.best_cut_multipliers_for_best_bound = {}  # μ at best LB

        #     # Which behaviour at this node?
        #     cutting_active_here = self.use_cover_cuts and (depth <= max_cut_depth)   # can ADD cuts
        #     mu_dynamic_here     = self.use_cover_cuts and (depth <= max_mu_depth)    # can UPDATE μ / use in dual
        #     cuts_present_here   = self.use_cover_cuts and bool(self.best_cuts)

        #     # Ensure λ starts in a reasonable range (consistent with compute_modified_weights)
        #     self.lmbda = max(0.0, min(getattr(self, "lmbda", 0.05), 1e4))

        #     polyak_enabled = True

        #     # Collect newly generated cuts at this node
        #     node_new_cuts = []

        #     # --- Quick guards ---
        #     if not self.edge_list or self.num_nodes <= 1:
        #         if self.verbose:
        #             print(f"Error at depth {depth}: Empty edge list or invalid graph")
        #         end_time = time()
        #         LagrangianMST.total_compute_time += end_time - start_time
        #         return self.best_lower_bound, self.best_upper_bound, node_new_cuts

        #     # Fixed / forbidden edges
        #     F_in  = getattr(self, "fixed_edges", set())
        #     F_out = getattr(self, "excluded_edges", set())
        #     edge_idx = self.edge_indices
        #     if not hasattr(self, "_rhs_eff"):
        #         self._rhs_eff = {}

        #     # ------------------------------------------------------------------
        #     # Separation policy (FIXED):
        #     #   - DO NOT do objective-only pre-separation at root.
        #     #   - Always delay separation to the first violating MST inside the loop.
        #     #   - Still obey depth limits: only add cuts when cutting_active_here AND μ is dynamic.
        #     # ------------------------------------------------------------------
        #     pending_sep = bool(cutting_active_here and mu_dynamic_here)

        #     # ------------------------------------------------------------------
        #     # 2) Compute rhs_eff and detect infeasibility (fixed edges + cuts)
        #     #    rhs_eff = rhs - |cut ∩ F_in|
        #     # ------------------------------------------------------------------
        #     if self.use_cover_cuts and self.best_cuts:
        #         for idx_c, (cut, rhs) in enumerate(self.best_cuts):
        #             rhs_eff = int(rhs) - len(cut & F_in)
        #             self._rhs_eff[idx_c] = rhs_eff
        #             if rhs_eff < 0:
        #                 end_time = time()
        #                 LagrangianMST.total_compute_time += end_time - start_time
        #                 return float('inf'), self.best_upper_bound, node_new_cuts

        #     # ------------------------------------------------------------------
        #     # 3) Trim number of cuts (keep at most max_active_cuts)
        #     # ------------------------------------------------------------------
        #     if self.use_cover_cuts and self.best_cuts and len(self.best_cuts) > max_active_cuts:
        #         parent_mu_map = getattr(self, "best_cut_multipliers_for_best_bound", None)
        #         if not parent_mu_map:
        #             parent_mu_map = self.best_cut_multipliers

        #         idx_and_cut = list(enumerate(self.best_cuts))
        #         idx_and_cut.sort(
        #             key=lambda ic: abs(parent_mu_map.get(ic[0], 0.0)),
        #             reverse=True
        #         )
        #         idx_and_cut = idx_and_cut[:max_active_cuts]

        #         new_cuts_list = []
        #         new_mu       = {}
        #         new_mu_best  = {}
        #         new_rhs_eff  = {}

        #         for new_i, (old_i, cut_rhs) in enumerate(idx_and_cut):
        #             new_cuts_list.append(cut_rhs)
        #             new_mu[new_i]      = float(parent_mu_map.get(old_i, 0.0))
        #             new_mu_best[new_i] = float(parent_mu_map.get(old_i, 0.0))
        #             new_rhs_eff[new_i] = self._rhs_eff[old_i]

        #         self.best_cuts = new_cuts_list
        #         self.best_cut_multipliers = new_mu
        #         self.best_cut_multipliers_for_best_bound = new_mu_best
        #         self._rhs_eff = new_rhs_eff

        #     cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)

        #     # ------------------------------------------------------------------
        #     # 4) Build cut -> edge index arrays (for pricing/subgradients)
        #     # ------------------------------------------------------------------
        #     def _rebuild_cut_structures():
        #         nonlocal cut_edge_idx_free, cut_edge_idx_all, rhs_eff_vec

        #         cut_edge_idx_free = []
        #         cut_edge_idx_all  = []

        #         for cut, rhs in self.best_cuts:
        #             idxs_free = [
        #                 edge_idx[e] for e in cut
        #                 if (e not in F_in and e not in F_out) and (e in edge_idx)
        #             ]
        #             arr_free = (
        #                 np.fromiter(idxs_free, dtype=np.int32)
        #                 if idxs_free else np.empty(0, dtype=np.int32)
        #             )
        #             cut_edge_idx_free.append(arr_free)

        #             idxs_all = [edge_idx[e] for e in cut if e in edge_idx]
        #             arr_all  = (
        #                 np.fromiter(idxs_all, dtype=np.int32)
        #                 if idxs_all else np.empty(0, dtype=np.int32)
        #             )
        #             cut_edge_idx_all.append(arr_all)

        #         self._cut_edge_idx     = cut_edge_idx_free
        #         self._cut_edge_idx_all = cut_edge_idx_all

        #         rhs_eff_vec = (
        #             np.array([self._rhs_eff[i] for i in range(len(self.best_cuts))], dtype=float)
        #             if self.best_cuts else np.zeros(0, dtype=float)
        #         )

        #     cut_edge_idx_free = []
        #     cut_edge_idx_all  = []
        #     rhs_eff_vec       = np.zeros(0, dtype=float)

        #     if self.use_cover_cuts and self.best_cuts:
        #         _rebuild_cut_structures()

        #     # Track usefulness of cuts at this node
        #     max_cut_violation = [0.0 for _ in self.best_cuts]

        #     # Histories / caches
        #     self._mw_cached = None
        #     self._mw_lambda = None
        #     self._mw_mu     = np.zeros(len(cut_edge_idx_free), dtype=float)

        #     if not hasattr(self, "subgradients"):
        #         self.subgradients = []
        #     if not hasattr(self, "step_sizes"):
        #         self.step_sizes = []
        #     if not hasattr(self, "multipliers"):
        #         self.multipliers = []

        #     prev_weights   = None
        #     prev_mst_edges = None

        #     if not hasattr(self, "_mst_mask") or self._mst_mask.size != len(self.edge_weights):
        #         self._mst_mask = np.zeros(len(self.edge_weights), dtype=bool)
        #     mst_mask = self._mst_mask

        #     # Decide iteration limit for this node:
        #     if is_root:
        #         iter_limit = root_max_iter * 1.1 if self.use_cover_cuts else root_max_iter
        #     else:
        #         iter_limit = max_iter
        #     # ------------------------------------------------------------------
        #     # 5) Subgradient iterations
        #     # ------------------------------------------------------------------
        #     for iter_num in range(int(iter_limit)):
        #         # 1) MST with current λ, μ              
        #         try:
        #             mst_cost, mst_length, mst_edges = self.compute_mst_incremental(prev_weights, prev_mst_edges)
        #         except Exception:
        #             mst_cost, mst_length, mst_edges = self.compute_mst()

        #         self.last_mst_edges = mst_edges
        #         prev_mst_edges      = mst_edges
        #         cut_g_signed = []

        #         # 1a) ONE-SHOT delayed separation (root AND non-root)
        #         if (
        #             cutting_active_here
        #             and mu_dynamic_here
        #             and pending_sep
        #             and len(self.best_cuts) < max_active_cuts
        #             and mst_length > self.budget
        #         ):
        #             try:
        #                 cand_cuts_loop = self.generate_cover_cuts(mst_edges) or []
        #                 print("sss")

        #                 T_loop = set(mst_edges)
        #                 scored_loop = []
        #                 F_in_set = set(F_in)  # (already defined above)

        #                 for cut, rhs in cand_cuts_loop:
        #                     S_set   = set(cut)
        #                     S_free  = S_set - F_in_set                 # remove fixed edges from LHS set
        #                     lhs_free = len(T_loop & S_free)            # only MST edges that are NOT fixed
        #                     rhs_eff  = int(rhs) - len(S_set & F_in_set)
        #                     violation = lhs_free - rhs_eff

        #                     if violation >= min_cut_violation_for_add:
        #                         scored_loop.append((violation, S_set, rhs))

        #                 scored_loop.sort(reverse=True, key=lambda t: t[0])

        #                 remaining_slots = max(0, max_active_cuts - len(self.best_cuts))
        #                 if remaining_slots > 0:
        #                     scored_loop = scored_loop[:min(max_new_cuts_per_node, remaining_slots)]
        #                 else:
        #                     scored_loop = []

        #                 existing = {frozenset(c): rhs for (c, rhs) in self.best_cuts}
        #                 added_any = False

        #                 for violation, S, rhs in scored_loop:
        #                     fz = frozenset(S)
        #                     if fz in existing:
        #                         continue

        #                     self.best_cuts.append((set(S), rhs))
        #                     new_idx = len(self.best_cuts) - 1
        #                     MU0 = getattr(self, "mu_init", 0.0)  # safe default: 0 (avoid immediate decay overhead)
        #                     self.best_cut_multipliers[new_idx] = MU0
        #                     self.best_cut_multipliers_for_best_bound[new_idx] = MU0


        #                     # keep rhs_eff consistent
        #                     self._rhs_eff[new_idx] = int(rhs) - len(set(S) & F_in)
        #                     if self._rhs_eff[new_idx] < 0:
        #                         end_time = time()
        #                         LagrangianMST.total_compute_time += end_time - start_time
        #                         return float('inf'), self.best_upper_bound, node_new_cuts

        #                     max_cut_violation.append(0.0)
        #                     node_new_cuts.append((set(S), rhs))
        #                     added_any = True

        #                 if added_any:
        #                     _rebuild_cut_structures()
        #                     self._mw_cached = None
        #                     self._mw_mu     = np.zeros(len(cut_edge_idx_free), dtype=float)
        #                     cuts_present_here = True

        #             except Exception as e:
        #                 if self.verbose:
        #                     print(f"Error in delayed separation at depth {depth}, iter {iter_num}: {e}")
        #             finally:
        #                 pending_sep = False  # do at most once per node

        #         # Prepare weights for next iteration (cache)
        #         prev_weights = getattr(self, "_last_mw", prev_weights)

        #         # 2) Primal & UB
        #         is_feasible = (mst_length <= self.budget)
        #         self._record_primal_solution(self.last_mst_edges, is_feasible)

        #         if is_feasible:
        #             try:
        #                 real_weight, real_length = self.compute_real_weight_length()
        #                 if (
        #                     not math.isnan(real_weight)
        #                     and not math.isinf(real_weight)
        #                     and real_weight < self.best_upper_bound
        #                 ):
        #                     self.best_upper_bound = real_weight
        #             except Exception as e:
        #                 if self.verbose:
        #                     print(f"Error updating primal solution: {e}")

        #         if len(self.primal_solutions) > MAX_SOLUTIONS:
        #             self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]

        #         # 3) Dual value: L(λ, μ) = MST_cost - λ B - Σ μ_i rhs_eff_i
        #         lam_for_dual = max(0.0, min(self.lmbda, 1e4))

        #         if self.use_cover_cuts and len(rhs_eff_vec) > 0:
        #             mu_vec = np.fromiter(
        #                 (
        #                     max(0.0, min(self.best_cut_multipliers.get(i, 0.0), 1e4))
        #                     for i in range(len(rhs_eff_vec))
        #                 ),
        #                 dtype=float,
        #                 count=len(rhs_eff_vec),
        #             )
        #             cover_cut_penalty = float(mu_vec @ rhs_eff_vec)
        #         else:
        #             cover_cut_penalty = 0.0

        #         lagrangian_bound = mst_cost - lam_for_dual * self.budget - cover_cut_penalty
        #         # if cover_cut_penalty != 0.0:
        #             # print("ggg", cover_cut_penalty)
        #         # print("lagrangian bound:", lagrangian_bound)

        #         if (
        #             not math.isnan(lagrangian_bound)
        #             and not math.isinf(lagrangian_bound)
        #             and abs(lagrangian_bound) < 1e10
        #         ):
        #             if lagrangian_bound > self.best_lower_bound + 1e-6:
        #                 self.best_lower_bound = lagrangian_bound
        #                 self.best_lambda      = lam_for_dual
        #                 self.best_mst_edges   = self.last_mst_edges
        #                 self.best_cost        = mst_cost
        #                 self.best_cut_multipliers_for_best_bound = self.best_cut_multipliers.copy()

        #         # 4) Subgradients
        #         knapsack_subgradient = float(mst_length - self.budget)
        #         # print("fff", mst_length)
        #         # print("lala",self.lmbda)
        #         # print("wer", knapsack_subgradient)

        #         # Fast skip: if MST feasible and all μ are ~0, don't pay cut gradient cost
        #         all_mu_small = (not self.best_cut_multipliers) or \
        #                     (max(self.best_cut_multipliers.values()) <= dead_mu_threshold)

        #         if cuts_present_here and mu_dynamic_here and len(cut_edge_idx_all) > 0 and not (is_feasible and all_mu_small):
        #             mst_mask[:] = False
        #             for e in mst_edges:
        #                 j = self.edge_indices.get(e)
        #                 if j is not None:
        #                     mst_mask[j] = True

        #             cut_g_signed = []
        #             cut_g_pos    = []

        #             for i, idxs_free in enumerate(cut_edge_idx_free):
        #                 lhs_free = int(mst_mask[idxs_free].sum()) if idxs_free.size else 0
        #                 g_i = float(lhs_free) - float(rhs_eff_vec[i])
        #                 cut_g_signed.append(g_i)
        #                 cut_g_pos.append(g_i if g_i > 0.0 else 0.0)

        #                 if g_i > max_cut_violation[i]:
        #                     max_cut_violation[i] = g_i

        #             cut_subgradients = cut_g_pos
        #         else:
        #             cut_subgradients = []
        #             cut_g_signed = []
        #             cut_g_pos = []


        #         norm_sq = knapsack_subgradient ** 2
        #         for g in cut_subgradients:
        #             norm_sq += float(g) ** 2

        #         # Polyak step size
        #         if polyak_enabled and self.best_upper_bound < float('inf') and norm_sq > 0.0:
        #             gap   = max(0.0, self.best_upper_bound - lagrangian_bound)
        #             alpha = gamma_base * gap / (norm_sq + eps)
        #         else:
        #             alpha = getattr(self, "step_size", 0.001)

        #         # λ update with momentum, then clamp
        #         v_prev = getattr(self, "_v_lambda", 0.0)
        #         v_new  = self.momentum_beta * v_prev + (1.0 - self.momentum_beta) * knapsack_subgradient
        #         self._v_lambda = v_new
        #         self.lmbda     = self.lmbda + alpha * v_new
        #         # print("ooo", alpha)

        #         if self.lmbda < 0.0:
        #             self.lmbda = 0.0
        #         if self.lmbda > 1e4:
        #             self.lmbda = 1e4

        #         # μ updates: projected subgradient for constraints sum_{e in S} x_e <= rhs_eff
        #         if mu_dynamic_here and len(cut_g_pos) > 0:
        #             for i, g in enumerate(cut_g_pos):
        #                 g = float(g)
        #                 if g <= 0.0:
        #                     continue

        #                 delta = gamma_mu * alpha * g

        #                 # cap only positive increment
        #                 if mu_increment_cap is not None:
        #                     delta = min(mu_increment_cap, delta)

        #                 mu_old = float(self.best_cut_multipliers.get(i, 0.0))
        #                 mu_new = mu_old + delta

        #                 # projection + clamp
        #                 if mu_new > 1e4:
        #                     mu_new = 1e4

        #                 self.best_cut_multipliers[i] = mu_new


        #         self.step_sizes.append(alpha)
        #         self.multipliers.append((self.lmbda, self.best_cut_multipliers.copy()))

        #     # ------------------------------------------------------------------
        #     # 6) Drop "dead" cuts globally
        #     # ------------------------------------------------------------------
        #     if self.use_cover_cuts and self.best_cuts and mu_dynamic_here:
        #         keep_indices = []

        #         parent_mu_map = getattr(
        #             self,
        #             "best_cut_multipliers_for_best_bound",
        #             self.best_cut_multipliers,
        #         )

        #         for i, (cut, rhs) in enumerate(self.best_cuts):
        #             mu_i    = float(self.best_cut_multipliers.get(i, 0.0))
        #             mu_hist = float(parent_mu_map.get(i, 0.0))

        #             ever_useful = (i < len(max_cut_violation) and max_cut_violation[i] > 0.0) \
        #                         or (abs(mu_hist) >= dead_mu_threshold)

        #             if (not ever_useful) and abs(mu_i) < dead_mu_threshold and abs(mu_hist) < dead_mu_threshold:
        #                 continue
        #             keep_indices.append(i)

        #         if len(keep_indices) < len(self.best_cuts):
        #             new_best_cuts = []
        #             new_mu        = {}
        #             new_mu_best   = {}
        #             new_rhs_eff   = {}

        #             for new_idx, old_idx in enumerate(keep_indices):
        #                 new_best_cuts.append(self.best_cuts[old_idx])
        #                 new_mu[new_idx]      = float(self.best_cut_multipliers.get(old_idx, 0.0))
        #                 new_mu_best[new_idx] = float(self.best_cut_multipliers_for_best_bound.get(old_idx, 0.0))
        #                 new_rhs_eff[new_idx] = self._rhs_eff[old_idx]

        #             self.best_cuts = new_best_cuts
        #             self.best_cut_multipliers = new_mu
        #             self.best_cut_multipliers_for_best_bound = new_mu_best
        #             self._rhs_eff = new_rhs_eff

        #     # ------------------------------------------------------------------
        #     # 7) Restore best (λ, μ) to pass to children
        #     # ------------------------------------------------------------------
        #     if hasattr(self, "best_lambda"):
        #         self.lmbda = self.best_lambda

        #     if mu_dynamic_here and hasattr(self, "best_cut_multipliers_for_best_bound"):
        #         self.best_cut_multipliers = self.best_cut_multipliers_for_best_bound.copy()

        #     end_time = time()
        #     LagrangianMST.total_compute_time += end_time - start_time
        #     return self.best_lower_bound, self.best_upper_bound, node_new_cuts

        
        else:  # Subgradient method with Polyak hybrid + cover cuts (λ, μ), depth-based freezing
            import os

            # --- Tunables / safety limits ---
            MAX_SOLUTIONS = getattr(self, "max_primal_solutions", 50)
            max_iter = min(self.max_iter, 200)

            # Polyak / momentum for λ
            self.momentum_beta = getattr(self, "momentum_beta", 0.7)
            gamma_base = getattr(self, "gamma_base", 0.05)

            # Safety controls for λ update
            fallback_alpha = getattr(self, "fallback_alpha", 1e-5)
            max_lambda_delta = getattr(self, "max_lambda_delta", 0.02)

            # μ update parameters.
            #
            # The two families of dualized rows are measured in different units,
            # and that -- not the separation -- is what used to make the cover
            # cuts useless:
            #
            #   budget row    g_lambda = sum_e l_e x_e - B      (a LENGTH, 1e2..1e4)
            #   cover rows    g_mu_i   = |T n S_i| - rhs_i      (a COUNT,  0..10)
            #
            # "joint" forms one Polyak step alpha = gamma * gap / ||g||^2 over
            # the concatenated vector and reuses it for mu.  ||g||^2 is the
            # budget row squared to within a rounding error, so alpha is sized
            # for lambda and lands 3-4 orders of magnitude below what the cut
            # rows need.  Measured over a full 783-node run at n=50: the largest
            # mu ever reached was 2.0e-3 against modified edge weights of order
            # 5e2, i.e. a price the MST can never see.  Each cut then moved the
            # bound by mu * (lhs - rhs) ~ 1e-3 while still costing separation
            # time -- cuts on came out slower and no tighter than cuts off.
            #
            # "block" sizes the step on the cut rows alone.  That overshoots by
            # about as much as "joint" undershoots (gap/||g_mu||^2 with the
            # inflated incumbent gap gives mu ~ 1e2 where ~1 is wanted), mu then
            # dominates the modified weights and lambda stops converging.  On
            # n=30 roots it took the dual bound from 594 (no cuts) down to 327.
            #
            # "scaled" keeps the joint Polyak step but first puts the budget row
            # into the cut rows' units by dividing it by the mean edge length.
            # That removes the 1e6 imbalance, but a Polyak step is still the
            # wrong instrument for these rows: it sizes the step by the duality
            # gap, and along a cut coordinate the dual is nearly flat -- the
            # whole gain available from a cover cut at an n=50 root measures
            # +1.5 on a bound of 4.0e3, while the mu that collects it is ~1.
            # A gap-sized step therefore crawls (0.01 per iteration) exactly
            # where it needs to travel furthest.
            #
            # "normalized" drops the gap and measures the step in
            # the units mu is actually denominated in.  mu is a price added to
            # edge weights, so one step moves the multiplier vector a fixed
            # fraction `mu_step_frac` of the modified-weight scale along
            # g_mu/||g_mu||, shrinking geometrically over the cut phase so the
            # sequence settles instead of oscillating.  This is the standard
            # normalized subgradient step with a diminishing scale, and it is
            # safe here precisely because phase 1 has already banked the plain
            # bound: an overshoot costs cut-phase iterations, never bound.
            #
            # "normalized_inf" is the DEFAULT: the same fixed distance, taken
            # with the infinity norm, so a pool of k cuts does not slow each
            # multiplier by sqrt(k).  See the note at the alpha_mu branch.
            mu_step_mode = str(getattr(self, "mu_step_mode", "normalized_inf")).lower()
            gamma_mu = getattr(self, "gamma_mu", 0.25)

            # Row scale used by "scaled": the mean length of an admissible edge.
            len_scale = float(np.mean(np.abs(self.edge_lengths)))
            if not (len_scale > 0.0) or math.isnan(len_scale):
                len_scale = 1.0

            # The scale mu lives on: a typical modified edge weight.  Every mu
            # quantity below is a fraction of this, so nothing depends on how
            # the instance generator happens to scale weights and lengths.
            weight_scale = float(
                np.mean(np.abs(self.edge_weights))
                + max(0.0, float(getattr(self, "lmbda", 0.0))) * len_scale
            )
            if not (weight_scale > 0.0) or math.isnan(weight_scale):
                weight_scale = 1.0

            # Distance mu travels on the first cut-phase iteration, and the
            # per-iteration shrink applied to it.
            mu_step_frac = float(getattr(self, "mu_step_frac", 0.002))
            mu_step_decay = float(getattr(self, "mu_step_decay", 0.9))

            # mu prices edges, so a cap on its increment only means something
            # relative to the weights it is added to.  The old default was a
            # hard 0.002, which on these instances (modified weights ~5e2) was a
            # 4e-6 relative move and clamped the step to nothing on exactly the
            # nodes where the cut mattered.  Express it as a fraction of the
            # modified-weight scale instead.
            #
            # Note this is a SAFETY RAIL, not a tuning knob, under either
            # normalized mode: those take a step of at most
            # mu_step_frac * weight_scale (0.002 by default), a full order of
            # magnitude under mu_cap_frac * weight_scale, so the clamp never
            # binds.  It exists for "joint"/"block"/"scaled", whose Polyak
            # ratios are unbounded when the incumbent gap is loose.
            mu_cap_frac = getattr(self, "mu_cap_frac", 0.02)
            mu_increment_cap = getattr(self, "mu_increment_cap", None)
            if mu_increment_cap is None and mu_cap_frac is not None:
                mu_increment_cap = float(mu_cap_frac) * weight_scale

            eps = 1e-12

            # Depth-based behaviour.
            #
            # These used to stop separating below depth 30 and freeze mu below
            # depth 50, to bound what separation cost.  Trees on these
            # instances reach depth 230, so the large majority of nodes ran
            # with stale multipliers or no cuts at all -- and separation is now
            # far cheaper than when those numbers were picked, so the reason
            # for them is gone.  Lifting both is better on BOTH axes, measured
            # over 20 seeds at n=100, density 0.2 (nodes as a geometric mean
            # against the capped defaults, and total time):
            #
            #   literature   0.592   263.0s -> 215.9s
            #   lemma1       0.668   307.5s -> 268.3s
            #   full         0.706   395.9s -> 334.6s
            #
            # No run timed out and every root bound is unchanged, as a depth
            # cap cannot reach the root.  Set either attribute to restore a
            # cap; `max_cut_depth = 0` still gives the root-only rung.
            max_cut_depth = getattr(self, "max_cut_depth", float("inf"))
            max_mu_depth = getattr(self, "max_mu_depth", float("inf"))
            is_root = depth == 0

            # Node-level separation parameters
            max_active_cuts = getattr(self, "max_active_cuts", 5)
            max_new_cuts_per_node = getattr(self, "max_new_cuts_per_node", 5)
            min_cut_violation_for_add = getattr(self, "min_cut_violation_for_add", 1.0)
            dead_mu_threshold = getattr(self, "dead_mu_threshold", 1e-6)

            # Extra iterations allowed at root
            root_max_iter = int(getattr(self, "root_max_iter", max_iter * 2))

            # ------------------------------------------------------------------
            # DEBUG SETTINGS
            # ------------------------------------------------------------------
            debug_cuts = False
            debug_iter_every = 1       # change to 5 or 10 if the log becomes too large
            debug_cut_max_rows = 10

            debug_log_path = getattr(
                self,
                "debug_cut_log_path",
                os.path.join(os.path.expanduser("~/Desktop"), "cut_debug_log.txt"),
            )

            # Clear the log only once at the root node.  Guarded by
            # `debug_cuts`: with the debug log off nothing else in this method
            # touches the file, and opening it unconditionally made every root
            # solve crash on a machine without the hard-coded ~/Desktop.
            if debug_cuts and depth == 0:
                try:
                    with open(debug_log_path, "w") as f:
                        f.write("CUT DEBUG LOG\n")
                        f.write("=" * 100 + "\n")
                except OSError:
                    debug_cuts = False

            def _dbg(msg, iter_num=None, force=False):
                if not debug_cuts:
                    return

                if iter_num is not None and not force:
                    if iter_num % debug_iter_every != 0:
                        return

                if iter_num is None:
                    line = f"[CUTDBG depth={depth}] {msg}"
                else:
                    line = f"[CUTDBG depth={depth} iter={iter_num}] {msg}"

                with open(debug_log_path, "a") as f:
                    f.write(line + "\n")

            def _edge_len(e):
                try:
                    return float(self.edge_lengths[self.edge_indices[e]])
                except Exception:
                    return float("nan")

            def _cut_len(cut):
                return sum(_edge_len(e) for e in cut if e in self.edge_indices)

            def _cut_repr(cut, max_edges=6):
                cut_list = sorted(list(cut))
                shown = cut_list[:max_edges]
                suffix = "" if len(cut_list) <= max_edges else f", ... +{len(cut_list) - max_edges}"
                return f"{shown}{suffix}"

            def _print_cut_table(stage, iter_num=None, force=False):
                if not debug_cuts:
                    return

                if iter_num is not None and not force:
                    if iter_num % debug_iter_every != 0:
                        return

                _dbg(
                    f"{stage}: active cuts = {len(getattr(self, 'best_cuts', []))}",
                    iter_num,
                    force,
                )

                if not getattr(self, "best_cuts", []):
                    return

                for i, (cut, rhs) in enumerate(self.best_cuts[:debug_cut_max_rows]):
                    mu = float(getattr(self, "best_cut_multipliers", {}).get(i, 0.0))
                    mu_best = float(getattr(self, "best_cut_multipliers_for_best_bound", {}).get(i, 0.0))
                    rhs_eff = getattr(self, "_rhs_eff", {}).get(i, rhs)

                    _dbg(
                        f"  cut[{i}] size={len(cut)} rhs={rhs} rhs_eff={rhs_eff} "
                        f"mu={mu:.6g} mu_best={mu_best:.6g} "
                        f"len_sum={_cut_len(cut):.3f} edges={_cut_repr(cut)}",
                        iter_num,
                        force,
                    )

                if len(self.best_cuts) > debug_cut_max_rows:
                    _dbg(
                        f"  ... {len(self.best_cuts) - debug_cut_max_rows} more cuts not shown",
                        iter_num,
                        force,
                    )

            # ------------------------------------------------------------------
            # Ensure cut structures exist
            # ------------------------------------------------------------------
            if not hasattr(self, "best_cuts"):
                self.best_cuts = []

            if not hasattr(self, "best_cut_multipliers"):
                self.best_cut_multipliers = {}

            if not hasattr(self, "best_cut_multipliers_for_best_bound"):
                self.best_cut_multipliers_for_best_bound = {}

            # Which behaviour at this node?
            cutting_active_here = self.use_cover_cuts and depth <= max_cut_depth
            mu_dynamic_here = self.use_cover_cuts and depth <= max_mu_depth
            use_cuts_in_dual_here = self.use_cover_cuts and bool(self.best_cuts)

            # Ensure λ starts in a reasonable range
            self.lmbda = max(0.0, min(getattr(self, "lmbda", 0.05), 1e4))

            polyak_enabled = True
            node_new_cuts = []

            # Every objective value is an integer, so a node whose bound has
            # reached incumbent - granularity cannot hold an improving tree:
            # the search drops it the moment it is popped, and the remaining
            # dual iterations produce nothing it will ever read.  Set
            # `objective_granularity` to 0 for a fractional objective.
            #
            # `stop_when_dominated` is nevertheless OFF by default.  Those
            # iterations are not only computing a bound: they are also where
            # the node picks up a budget-feasible tree, and the incumbent it
            # would have found there is what the REST of the search prunes
            # and fixes against.  Measured over 11 instances at n = 400
            # (densities 0.1 and 0.2), stopping early cost more in nodes
            # elsewhere than it saved here, and it broke the ladder ordering
            # on both sets.  Turn it on with a primal heuristic that does not
            # depend on the dual trajectory.
            _prune_gran = float(getattr(self, "objective_granularity", 1.0))
            _stop_dominated = bool(getattr(self, "stop_when_dominated", False))

            _dbg(
                f"START NODE | use_cover_cuts={self.use_cover_cuts}, "
                f"cutting_active_here={cutting_active_here}, "
                f"mu_dynamic_here={mu_dynamic_here}, "
                f"use_cuts_in_dual_here={use_cuts_in_dual_here}, "
                f"lambda_start={self.lmbda:.6g}, "
                f"inherited_cuts={len(self.best_cuts)}, "
                f"log_file={debug_log_path}",
                force=True,
            )

            _print_cut_table("Inherited cuts before reduction", force=True)

            # ------------------------------------------------------------------
            # Quick guards
            # ------------------------------------------------------------------
            if not self.edge_list or self.num_nodes <= 1:
                _dbg("STOP: empty edge list or invalid graph", force=True)

                end_time = time()
                LagrangianMST.total_compute_time += end_time - start_time
                return self.best_lower_bound, self.best_upper_bound, node_new_cuts

            # Fixed / forbidden edges
            # Frozensets of normalised edges; no copy needed.
            F_in = self.fixed_edges
            F_out = self.excluded_edges
            edge_idx = self.edge_indices

            self._rhs_eff = {}

            _dbg(
                f"Node fixings: |F_in|={len(F_in)}, |F_out|={len(F_out)}, "
                f"fixed_length={sum(_edge_len(e) for e in F_in if e in edge_idx):.3f}, "
                f"budget={self.budget:.3f}",
                force=True,
            )

            # ------------------------------------------------------------------
            # 2) Reduce inherited cuts and remove redundant cuts
            # ------------------------------------------------------------------
            if self.use_cover_cuts and self.best_cuts:
                old_mu = dict(getattr(self, "best_cut_multipliers", {}) or {})
                old_mu_best = dict(getattr(self, "best_cut_multipliers_for_best_bound", {}) or {})

                reduced_cuts = []
                reduced_mu = {}
                reduced_mu_best = {}
                reduced_rhs_eff = {}

                kept_count = 0
                redundant_count = 0

                for old_i, (cut, rhs) in enumerate(self.best_cuts):
                    S = cut

                    # A support the node's fixings do not meet projects to
                    # itself, and at n = 400 a lifted support is thousands of
                    # edges -- this is the copy worth not making.
                    S_fixed = S & F_in
                    S_excluded = S & F_out
                    S_free = (S - F_in - F_out) if (S_fixed or S_excluded) else S
                    rhs_eff = int(rhs) - len(S_fixed)

                    _dbg(
                        f"Reduce old_cut[{old_i}]: old_size={len(S)}, old_rhs={rhs}, "
                        f"|S_fixed|={len(S_fixed)}, |S_excluded|={len(S_excluded)}, "
                        f"|S_free|={len(S_free)}, rhs_eff={rhs_eff}, "
                        f"mu_old={float(old_mu.get(old_i, 0.0)):.6g}",
                        force=True,
                    )

                    if rhs_eff < 0:
                        LagrangianMST.cuts_infeasible += 1
                        _dbg(
                            f"STOP: inherited cut[{old_i}] makes node infeasible "
                            f"because rhs_eff={rhs_eff}<0",
                            force=True,
                        )

                        end_time = time()
                        LagrangianMST.total_compute_time += end_time - start_time
                        return float("inf"), self.best_upper_bound, node_new_cuts

                    # Redundant at this node
                    if len(S_free) <= rhs_eff:
                        redundant_count += 1
                        _dbg(
                            f"Drop old_cut[{old_i}] as redundant: "
                            f"|S_free|={len(S_free)} <= rhs_eff={rhs_eff}",
                            force=True,
                        )
                        continue

                    new_i = len(reduced_cuts)
                    reduced_cuts.append((frozenset(S_free), int(rhs_eff)))

                    mu_val = float(old_mu.get(old_i, 0.0))
                    mu_best_val = float(old_mu_best.get(old_i, mu_val))

                    reduced_mu[new_i] = mu_val
                    reduced_mu_best[new_i] = mu_best_val
                    reduced_rhs_eff[new_i] = int(rhs_eff)
                    kept_count += 1

                self.best_cuts = reduced_cuts
                self.best_cut_multipliers = reduced_mu
                self.best_cut_multipliers_for_best_bound = reduced_mu_best
                self._rhs_eff = reduced_rhs_eff

                _dbg(
                    f"Cut reduction summary: kept={kept_count}, "
                    f"redundant_dropped={redundant_count}",
                    force=True,
                )

            _print_cut_table("Cuts after reduction", force=True)

            # ------------------------------------------------------------------
            # 3) Trim number of cuts
            # ------------------------------------------------------------------
            if self.use_cover_cuts and self.best_cuts and len(self.best_cuts) > max_active_cuts:
                parent_mu_map = getattr(self, "best_cut_multipliers_for_best_bound", None)

                if not parent_mu_map:
                    parent_mu_map = self.best_cut_multipliers

                idx_and_cut = list(enumerate(self.best_cuts))
                # No generating tree is available for inherited cuts, so the
                # multiplier magnitude stands in for the violation; ties are
                # broken in favour of smaller supports as in Section 6.4.
                idx_and_cut.sort(
                    key=lambda ic: (
                        -abs(parent_mu_map.get(ic[0], 0.0)),
                        len(ic[1][0]),
                    ),
                )

                kept_old_indices = [old_i for old_i, _ in idx_and_cut[:max_active_cuts]]
                dropped_old_indices = [old_i for old_i, _ in idx_and_cut[max_active_cuts:]]

                _dbg(
                    f"Trim cuts: max_active_cuts={max_active_cuts}, "
                    f"kept_old_indices={kept_old_indices}, "
                    f"dropped_old_indices={dropped_old_indices}",
                    force=True,
                )

                idx_and_cut = idx_and_cut[:max_active_cuts]

                new_cuts_list = []
                new_mu = {}
                new_mu_best = {}
                new_rhs_eff = {}

                for new_i, (old_i, cut_rhs) in enumerate(idx_and_cut):
                    new_cuts_list.append(cut_rhs)
                    new_mu[new_i] = float(self.best_cut_multipliers.get(old_i, 0.0))
                    new_mu_best[new_i] = float(parent_mu_map.get(old_i, new_mu[new_i]))
                    new_rhs_eff[new_i] = int(self._rhs_eff.get(old_i, cut_rhs[1]))

                self.best_cuts = new_cuts_list
                self.best_cut_multipliers = new_mu
                self.best_cut_multipliers_for_best_bound = new_mu_best
                self._rhs_eff = new_rhs_eff

            cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)
            use_cuts_in_dual_here = self.use_cover_cuts and bool(self.best_cuts)

            _print_cut_table("Cuts after trimming", force=True)

            # ------------------------------------------------------------------
            # 4) Build cut -> edge index arrays
            # ------------------------------------------------------------------
            def _rebuild_cut_structures():
                nonlocal cut_edge_idx_free, cut_edge_idx_all, rhs_eff_vec

                # The support -> index arrays are cached by support and the
                # free split is one boolean take, so a pool of lifted cuts
                # costs a handful of numpy calls instead of an O(|S|) Python
                # list comprehension per cut per node.  At n = 400 the lifted
                # supports run to thousands of edges and this alone was 9% of
                # the solve on the strengthened rungs.
                free_mask = self._get_free_mask()

                cut_edge_idx_free = []
                cut_edge_idx_all = []

                for i, (cut, rhs) in enumerate(self.best_cuts):
                    arr_all = self._cut_index_array(cut)

                    if free_mask is None:
                        arr_free = arr_all
                    else:
                        arr_free = arr_all[free_mask[arr_all]]

                    cut_edge_idx_free.append(arr_free)
                    cut_edge_idx_all.append(arr_all)

                    if i not in self._rhs_eff:
                        self._rhs_eff[i] = int(rhs)

                self._cut_edge_idx = cut_edge_idx_free
                self._cut_edge_idx_all = cut_edge_idx_all

                rhs_eff_vec = (
                    np.array(
                        [self._rhs_eff[i] for i in range(len(self.best_cuts))],
                        dtype=float,
                    )
                    if self.best_cuts
                    else np.zeros(0, dtype=float)
                )

                if debug_cuts:
                    _dbg(
                        f"Rebuilt cut structures: num_cuts={len(self.best_cuts)}, "
                        f"rhs_eff_vec={rhs_eff_vec.tolist()}, "
                        f"free_edge_counts={[len(a) for a in cut_edge_idx_free]}",
                        force=True,
                    )

            cut_edge_idx_free = []
            cut_edge_idx_all = []
            rhs_eff_vec = np.zeros(0, dtype=float)

            if self.use_cover_cuts and self.best_cuts:
                _rebuild_cut_structures()


            max_cut_violation = [0.0 for _ in self.best_cuts]

            # Histories / caches
            self._mw_cached = None
            self._mw_lambda = None
            self._mw_mu = np.zeros(len(cut_edge_idx_free), dtype=float)

            if not hasattr(self, "subgradients"):
                self.subgradients = []

            if not hasattr(self, "step_sizes"):
                self.step_sizes = []

            if not hasattr(self, "multipliers"):
                self.multipliers = []

            prev_weights = None
            prev_mst_edges = None

            if not hasattr(self, "_mst_mask") or self._mst_mask.size != len(self.edge_weights):
                self._mst_mask = np.zeros(len(self.edge_weights), dtype=bool)

            mst_mask = self._mst_mask

            # Decide iteration limit for this node
            if is_root:
                iter_limit = root_max_iter
            else:
                # Optional depth decay: with lambda inheritance, deep children
                # only need to REFINE the parent's near-optimal lambda, not
                # rediscover it, so fewer iterations suffice. Controlled by
                # `child_iter_decay` (per-level multiplier) and `child_min_iter`
                # (floor). Both default to no-op values, so when unset the cap
                # is exactly the old flat `max_iter` -> non-negative runs, which
                # set neither, are unaffected.
                decay = getattr(self, "child_iter_decay", 1.0)
                min_iter = getattr(self, "child_min_iter", max_iter)
                if decay < 1.0 and depth > 0:
                    decayed = int(round(max_iter * (decay ** depth)))
                    iter_limit = max(min_iter, decayed)
                else:
                    iter_limit = max_iter

            # ------------------------------------------------------------------
            # Two phases: lambda alone, then lambda and mu together.
            #
            # The node bound is a max over the multiplier TRAJECTORY, not a max
            # over (lambda, mu) space, and that is what used to make the cuts
            # counter-productive.  As soon as one cut carried mu > 0 the priced
            # MST changed, the budget subgradient changed with it and the lambda
            # sequence left the path it would have followed on its own.  It
            # never came back: on n=50 roots the cut run ended at lambda=0.131
            # where the plain run reached 0.1695, and the reported bound was
            # 3761.5 against 3785.9 -- cuts on, bound DOWN by 24, before a
            # single cut had done any work.
            #
            # So phase 1 reproduces the no-cut run exactly: every mu is held at
            # zero, nothing is separated, and lambda walks the same sequence it
            # would walk with `use_cover_cuts` off.  Phase 2 restarts from the
            # best lambda of phase 1 with the inherited multipliers restored and
            # spends `cut_phase_frac` of the budget separating and moving mu.
            #
            # Because `best_lower_bound` keeps the max over both phases and
            # phase 1 is the plain run, the node bound with cuts is now never
            # below the node bound without them -- the cuts can only add.  What
            # they cost is the phase-2 iterations, which is the honest price to
            # weigh them against.
            cuts_enabled_here = self.use_cover_cuts and (
                cutting_active_here or bool(self.best_cuts)
            )

            # Phase 1 is a full replay of the plain run, so a node with cuts
            # costs (1 + cut_phase_frac) times the dual iterations of a node
            # without them -- that, not separation, is where the cuts-on time
            # goes.  `lam_phase_frac` exposes phase 1's share so the floor it
            # buys can be weighed against what it costs; 1.0 is the replay in
            # full, which is what the monotonicity argument above assumes.
            if cuts_enabled_here:
                lam_phase_frac = float(getattr(self, "lam_phase_frac", 1.0))
                lam_phase_iters = int(round(lam_phase_frac * iter_limit))

                if lam_phase_frac < 1.0:
                    # A shortened replay still has to run: only the default,
                    # which is the replay in full, may round down to the
                    # `iter_limit` it was handed -- zero included.
                    lam_phase_iters = max(1, lam_phase_iters)

                # The cut phase gets three times the lambda phase's budget.
                # Measured over 3 instances at n = 400, density 0.2 (nodes,
                # geometric over the ladder): 2.0 leaves the multipliers
                # short of the range where a cover reprices the tree, 4.0
                # spends iterations after they have arrived.
                cut_phase_frac = float(getattr(self, "cut_phase_frac", 3.0))
                cut_phase_iters = max(1, int(round(cut_phase_frac * iter_limit)))
            else:
                lam_phase_iters = int(iter_limit)
                cut_phase_iters = 0

            total_iters = lam_phase_iters + cut_phase_iters

            # Multipliers inherited from the parent, parked until phase 2.
            parked_mu = {
                i: float(self.best_cut_multipliers.get(i, 0.0))
                for i in range(len(self.best_cuts))
            }

            # The lambda the node was handed.  With `inherit_lambda` that is the
            # parent's best lambda, so (lam_at_entry, parked_mu) is the exact
            # point the parent's own bound came from.  Phase 2 reopens there
            # whenever a multiplier survived, which re-prices that point on its
            # first iteration: the child minimises over a subset of the parent's
            # trees, so it cannot score below the parent there, and the cut
            # strength carries down the branch instead of having to be
            # rediscovered from mu = 0 at every node.  With no inherited
            # multiplier there is nothing to re-price and phase 2 opens at the
            # best lambda phase 1 found, as before.
            lam_at_entry = max(0.0, min(float(getattr(self, "lmbda", 0.0)), 1e4))
            have_inherited_mu = any(v > 0.0 for v in parked_mu.values())

            # Price the inherited point ONCE, before mu is parked.
            #
            # (lam_at_entry, parked_mu) is where the parent's own bound came
            # from, and this node minimises over a subset of the parent's
            # trees, so L_child at that point is >= the parent's bound.
            # Recording it here raises `best_lower_bound` AND the dual state
            # that goes with it -- best_lambda, best_cut_multipliers_for_best_bound,
            # best_mst_edges -- so the multipliers this node hands its own
            # children describe the stronger point too.  Without it the bound
            # could be repaired after the fact (MSTNode clamps it to the
            # parent's), but the dual solution behind it stayed weaker and the
            # inversion simply reappeared one level down.
            if cuts_enabled_here and have_inherited_mu and len(rhs_eff_vec) > 0:
                try:
                    self.lmbda = lam_at_entry
                    self._invalidate_weight_cache()

                    _c0, _l0, _e0 = self.compute_mst()

                    if _e0 and not math.isinf(_c0) and not math.isnan(_c0):
                        _mu0 = np.fromiter(
                            (
                                max(0.0, min(parked_mu.get(i, 0.0), 1e4))
                                for i in range(len(rhs_eff_vec))
                            ),
                            dtype=float,
                            count=len(rhs_eff_vec),
                        )
                        _lb0 = (
                            _c0
                            - lam_at_entry * self.budget
                            - float(_mu0 @ rhs_eff_vec)
                        )

                        if (
                            not math.isnan(_lb0)
                            and not math.isinf(_lb0)
                            and abs(_lb0) < 1e10
                            and _lb0 > self.best_lower_bound
                        ):
                            self.best_lower_bound = _lb0
                            self.best_lambda = lam_at_entry
                            self.best_mst_edges = _e0
                            self.best_cost = _c0
                            self.best_cut_multipliers_for_best_bound = dict(parked_mu)

                            _dbg(
                                f"Inherited point priced: lambda={lam_at_entry:.6g}, "
                                f"LB={_lb0:.6g}",
                                force=True,
                            )
                except Exception as _exc:
                    _dbg(f"Could not price the inherited point: {_exc}", force=True)

            if cuts_enabled_here:
                for i in parked_mu:
                    self.best_cut_multipliers[i] = 0.0
                self._invalidate_weight_cache()

            # Separation follows Algorithm 2 line 10: every budget-violating
            # tree produced by the multiplier sequence yields one seed cover.
            # Trees already separated on are skipped, and the active-pool cap
            # bounds the total separation work spent at the node.  Only phase-2
            # trees are separated on: those are priced at a lambda that is
            # already near the node's dual optimum, so their seed covers are the
            # ones that matter there, instead of covers read off a tree from the
            # middle of the lambda ramp that no longer violates anything by the
            # time lambda settles.
            sep_rounds = 0
            max_sep_rounds = int(getattr(self, "max_sep_rounds", int(iter_limit)))
            separated_trees = set()

            # Patience for the cut phase.  Measured over a whole n = 400 run,
            # only 38% of the nodes that reach phase 2 ever see their bound
            # move there; the rest spend the entire phase -- one Kruskal per
            # iteration -- re-deriving a bound phase 1 already had.  The
            # phase now stops once it has gone `cut_phase_patience`
            # iterations without improving, counted from the last
            # improvement, so a node that IS gaining keeps its full budget.
            # Separation resets the counter: a cut that has just entered has
            # not had a chance to move its multiplier yet.
            cut_phase_patience = int(getattr(self, "cut_phase_patience", 6))
            last_gain_iter = lam_phase_iters
            sep_last_iter = lam_phase_iters

            _dbg(
                f"Iteration setup: lambda_phase={lam_phase_iters}, "
                f"cut_phase={cut_phase_iters}, "
                f"max_sep_rounds={max_sep_rounds}, "
                f"parked_mu={parked_mu}",
                force=True,
            )

            # ------------------------------------------------------------------
            # 5) Subgradient iterations
            # ------------------------------------------------------------------
            for iter_num in range(total_iters):
                # Counted here rather than from total_iters: the loop has two
                # early exits (a feasible tree at the phase-2 start, and the
                # cut-phase patience test), so the budget overstates the work.
                LagrangianMST.lr_iterations += 1
                # Phase switch: rewind lambda to the best one phase 1 found,
                # put the inherited multipliers back and clear the momentum, so
                # phase 2 starts from the best point of the plain run.
                if cuts_enabled_here and iter_num == lam_phase_iters:
                    if have_inherited_mu:
                        self.lmbda = lam_at_entry
                    elif hasattr(self, "best_lambda"):
                        self.lmbda = max(0.0, min(float(self.best_lambda), 1e4))

                    # Re-denominate the mu step in the weights it is actually
                    # added to.  weight_scale was fixed from the node's ENTRY
                    # lambda, but the whole point of phase 1 is to move lambda,
                    # and the modified weights move with it -- so a step sized
                    # on entry can land well below what the MST can see by the
                    # time it is taken.  One np.mean at the switch.
                    _ws = float(
                        np.mean(np.abs(self.edge_weights))
                        + max(0.0, float(self.lmbda)) * len_scale
                    )
                    if _ws > 0.0 and not math.isnan(_ws):
                        weight_scale = _ws
                        if mu_cap_frac is not None:
                            mu_increment_cap = float(mu_cap_frac) * weight_scale

                    for i, mu_val in parked_mu.items():
                        if i < len(self.best_cuts):
                            self.best_cut_multipliers[i] = mu_val

                    self._v_lambda = 0.0
                    self._invalidate_weight_cache()
                    prev_weights = None
                    prev_mst_edges = None

                    _dbg(
                        f"Phase 2 starts: lambda={self.lmbda:.6g}, "
                        f"restored_mu={ {k: round(v, 6) for k, v in parked_mu.items()} }",
                        iter_num,
                        force=True,
                    )

                cuts_live_now = cuts_enabled_here and iter_num >= lam_phase_iters

                # --------------------------------------------------------------
                # 5.1) MST with current λ and μ
                # --------------------------------------------------------------
                try:
                    mst_cost, mst_length, mst_edges = self.compute_mst_incremental(
                        prev_weights,
                        prev_mst_edges,
                    )
                    mst_method = "incremental"

                except Exception as e:
                    _dbg(
                        f"Incremental MST failed: {e}. Falling back to full MST.",
                        iter_num,
                        force=True,
                    )

                    mst_cost, mst_length, mst_edges = self.compute_mst()
                    mst_method = "full"

                if (
                    not mst_edges
                    or math.isinf(mst_cost)
                    or math.isinf(mst_length)
                    or math.isnan(mst_cost)
                    or math.isnan(mst_length)
                ):
                    _dbg(
                        f"STOP: invalid MST. method={mst_method}, "
                        f"mst_cost={mst_cost}, mst_length={mst_length}, "
                        f"num_edges={len(mst_edges) if mst_edges else 0}",
                        iter_num,
                        force=True,
                    )

                    end_time = time()
                    LagrangianMST.total_compute_time += end_time - start_time
                    return float("inf"), self.best_upper_bound, node_new_cuts

                self.last_mst_edges = mst_edges
                prev_mst_edges = mst_edges

                _dbg(
                    f"MST: method={mst_method}, cost={mst_cost:.6g}, "
                    f"length={mst_length:.6g}, budget={self.budget:.6g}, "
                    f"budget_violation={mst_length - self.budget:.6g}, "
                    f"num_edges={len(mst_edges)}",
                    iter_num,
                )

                # The cut phase has nothing to do when it opens on a
                # budget-feasible tree and no cut is live: a cover inequality is
                # valid for every budget-feasible tree, so this tree violates
                # none of them, separation would return an empty list and there
                # is no mu to move.  The bound at this exact point was already
                # recorded in phase 1 -- same lambda, same zero multipliers,
                # same tree -- so stopping here costs nothing and saves the
                # whole phase.  On these instances that is most nodes.
                if (
                    cuts_live_now
                    and iter_num == lam_phase_iters
                    and mst_length <= self.budget
                    and (not self.best_cuts
                         or getattr(self, "skip_cut_phase_when_feasible", False))
                ):
                    # The cut phase has nothing to do when it opens on a
                    # budget-feasible tree.
                    #
                    # A cover inequality is valid for EVERY budget-feasible
                    # spanning tree of this node, and this tree is one -- it
                    # contains F+, avoids F- and fits the budget.  So it
                    # violates no cover: separation would return an empty
                    # list, every live cut is slack, and the subgradient is
                    # nonpositive in every dualized row, so the multipliers
                    # would only walk back toward the zero-mu point that
                    # phase 1 has already scored.  The whole phase is
                    # skipped, and over 62% of the nodes at n = 400 that is
                    # 15 Kruskals saved for a bound that does not move.
                    _dbg(
                        "Cut phase skipped: feasible tree at the phase-2 "
                        "starting point",
                        iter_num,
                        force=True,
                    )
                    break

                # --------------------------------------------------------------
                # 5.2) Separation (Algorithm 2, line 10)
                #
                # Every budget-violating tree the multiplier sequence produces
                # yields one seed cover.  A tree already separated on gives the
                # same seed again, so it is skipped, and once the active pool is
                # full no further separation runs at this node.
                # --------------------------------------------------------------
                # The tree's edge INDICES, which _mst_core already has as a
                # sorted-by-insertion array, identify it just as well as a
                # frozenset of its n-1 edge tuples and cost one memoryview
                # hash instead of n-1 tuple hashes per iteration.
                _tidx = self._last_mst_idx
                tree_key = (
                    np.sort(_tidx).tobytes() if _tidx is not None
                    else frozenset(mst_edges)
                )

                # A cut added with almost no cut-phase iterations left cannot
                # have its multiplier tuned: it takes a pool slot and distorts
                # the modified weights without ever earning a bound.  That is
                # the shape of the cliff the tree-completion rung falls off
                # between a pool of 8 and one of 12 -- the phase has
                # cut_phase_frac * iter_limit iterations to fit one multiplier
                # per cut, and past some pool size there are more multipliers
                # than iterations.  `min_new_cut_iters` refuses a cut that
                # arrives too late to be fitted; 0 is off, which is the
                # behaviour this replaces.
                min_new_cut_iters = int(getattr(self, "min_new_cut_iters", 0))

                _new_cut_this_iter = False

                should_separate = (
                    cuts_live_now
                    and cutting_active_here
                    and mu_dynamic_here
                    and sep_rounds < max_sep_rounds
                    and len(self.best_cuts) < max_active_cuts
                    and mst_length > self.budget
                    and tree_key not in separated_trees
                    and (total_iters - iter_num) >= min_new_cut_iters
                )

                _dbg(
                    f"Separation check: should_separate={should_separate}, "
                    f"sep_rounds={sep_rounds}/{max_sep_rounds}, "
                    f"active_cuts={len(self.best_cuts)}/{max_active_cuts}, "
                    f"budget_violated={mst_length > self.budget}",
                    iter_num,
                )

                if should_separate:
                    LagrangianMST.cut_nodes += 1
                    try:
                        cand_cuts_loop = self.generate_cover_cuts(mst_edges) or []

                        _dbg(
                            f"Generated candidate cuts: count={len(cand_cuts_loop)}",
                            iter_num,
                            force=True,
                        )

                        T_loop = set(mst_edges)
                        scored_loop = []

                        for cand_i, (cut, rhs) in enumerate(cand_cuts_loop):
                            S_set = cut

                            S_fixed = S_set & F_in
                            S_excluded = S_set & F_out
                            S_free = (
                                (S_set - F_in - F_out)
                                if (S_fixed or S_excluded) else S_set
                            )
                            rhs_eff_new = int(rhs) - len(S_fixed)

                            if rhs_eff_new < 0:
                                LagrangianMST.cuts_infeasible += 1
                                _dbg(
                                    f"STOP: candidate cut[{cand_i}] gives "
                                    f"rhs_eff_new={rhs_eff_new}<0",
                                    iter_num,
                                    force=True,
                                )

                                end_time = time()
                                LagrangianMST.total_compute_time += end_time - start_time
                                return float("inf"), self.best_upper_bound, node_new_cuts

                            if len(S_free) <= rhs_eff_new:
                                _dbg(
                                    f"Candidate cut[{cand_i}] dropped as redundant: "
                                    f"|S_free|={len(S_free)} <= rhs_eff={rhs_eff_new}",
                                    iter_num,
                                    force=True,
                                )
                                continue

                            lhs_free = len(T_loop & S_free)
                            violation = lhs_free - rhs_eff_new

                            # _dbg is a no-op with debugging off, but the
                            # f-string is built before the call either way,
                            # and _cut_len sums over the whole support: at a
                            # few thousand edges per candidate that was a few
                            # percent of the solve spent formatting a string
                            # nobody reads.
                            if debug_cuts:
                                _dbg(
                                    f"Candidate cut[{cand_i}]: orig_size={len(S_set)}, "
                                    f"|fixed|={len(S_fixed)}, "
                                    f"|excluded|={len(S_excluded)}, "
                                    f"|free|={len(S_free)}, "
                                    f"rhs={rhs}, rhs_eff={rhs_eff_new}, "
                                    f"lhs_on_current_MST={lhs_free}, "
                                    f"violation={violation}, "
                                    f"len_sum={_cut_len(S_free):.3f}",
                                    iter_num,
                                    force=True,
                                )

                            if violation >= min_cut_violation_for_add:
                                scored_loop.append(
                                    (float(violation), frozenset(S_free), int(rhs_eff_new))
                                )

                        # Most violated on the generating tree first, ties
                        # broken in favour of smaller supports.
                        #
                        # `cut_rank_mode="density"` ranks by violation per
                        # support edge instead.  A dualized cover puts the SAME
                        # mu on every edge of S, so within S the ordering the
                        # MST sees is untouched and only the S-versus-rest
                        # boundary moves: a large support spreads that push
                        # into something close to a uniform shift, while a
                        # small one concentrates it.  If that is what decides a
                        # cut's worth here, violation alone is the wrong
                        # ranking.  "violation" is the default and the
                        # behaviour this replaces.
                        if str(getattr(self, "cut_rank_mode", "violation")) == "density":
                            scored_loop.sort(
                                key=lambda t: (-t[0] / max(1, len(t[1])), len(t[1]))
                            )
                        else:
                            scored_loop.sort(key=lambda t: (-t[0], len(t[1])))

                        remaining_slots = max(0, max_active_cuts - len(self.best_cuts))

                        if remaining_slots > 0:
                            scored_loop = scored_loop[
                                : min(max_new_cuts_per_node, remaining_slots)
                            ]
                        else:
                            scored_loop = []

                        _dbg(
                            f"Candidate cuts after filtering: "
                            f"kept_for_addition={len(scored_loop)}, "
                            f"remaining_slots={remaining_slots}",
                            iter_num,
                            force=True,
                        )

                        existing = {
                            frozenset(c): (i, int(rhs))
                            for i, (c, rhs) in enumerate(self.best_cuts)
                        }

                        changed_any = False

                        for violation, S, rhs in scored_loop:
                            # S is already a frozenset; see the scoring loop.
                            fz = S

                            if fz in existing:
                                old_i, old_rhs = existing[fz]

                                if rhs < old_rhs:
                                    _dbg(
                                        f"Replace duplicate cut at index {old_i}: "
                                        f"old_rhs={old_rhs}, new_rhs={rhs}, "
                                        f"violation={violation}",
                                        iter_num,
                                        force=True,
                                    )

                                    self.best_cuts[old_i] = (frozenset(S), int(rhs))
                                    self._rhs_eff[old_i] = int(rhs)
                                    max_cut_violation[old_i] = max(
                                        max_cut_violation[old_i],
                                        violation,
                                    )
                                    changed_any = True

                                else:
                                    _dbg(
                                        f"Skip duplicate cut: existing_rhs={old_rhs}, "
                                        f"new_rhs={rhs}, violation={violation}",
                                        iter_num,
                                        force=True,
                                    )

                                continue

                            self.best_cuts.append((frozenset(S), int(rhs)))
                            LagrangianMST.cuts_separated += 1
                            new_idx = len(self.best_cuts) - 1

                            # Start at zero.  L(lambda, 0) is exactly the dual
                            # value without the cut, so a cut can never lower
                            # the bound at the moment it enters; the arbitrary
                            # 1e-3 seed used before could, and was also small
                            # enough to be invisible to the MST anyway.  A cut
                            # is only added when it is violated, so g_i > 0 on
                            # this very iteration and mu leaves 0 at the next
                            # update.
                            # A new cut starts at mu = 0 and the subgradient
                            # raises it.  Starting it higher, on the theory
                            # that the cut phase is short and a cut should not
                            # spend its few iterations climbing, is strictly
                            # worse -- monotonically so, over four seeds at
                            # n = 250 on the tree-completion rung:
                            #
                            #     mu_init     nodes     time
                            #     0.0           300     29.0s
                            #     6.0           394     37.7s
                            #    29.0           683     65.3s
                            #    58.0          1106    125.6s
                            #
                            # (6, 29 and 58 are 1%, 5% and 10% of the weight
                            # scale at this size.)  The dual is a MAXIMIZATION
                            # over mu >= 0, and zero is where a cut belongs
                            # before anything has shown it should be penalized:
                            # a positive start prices edges the tree has not
                            # been shown to overuse, which moves the MST off
                            # the Lagrangian optimum and LOWERS the bound, and
                            # the subgradient then spends iterations climbing
                            # back down.  The cut phase is not long because mu
                            # ramps up from zero; it is long because finding
                            # the right mu is the work.
                            MU0 = getattr(self, "mu_init", 0.0)

                            self.best_cut_multipliers[new_idx] = float(MU0)
                            self.best_cut_multipliers_for_best_bound[new_idx] = float(MU0)
                            self._rhs_eff[new_idx] = int(rhs)

                            max_cut_violation.append(max(0.0, violation))
                            node_new_cuts.append((frozenset(S), int(rhs)))

                            existing[fz] = (new_idx, int(rhs))
                            changed_any = True

                            # Same as above, and _cut_repr sorts the support
                            # on top of it.
                            if debug_cuts:
                                _dbg(
                                    f"ADD cut[{new_idx}]: size={len(S)}, rhs={rhs}, "
                                    f"initial_mu={MU0}, violation={violation}, "
                                    f"len_sum={_cut_len(S):.3f}, edges={_cut_repr(S)}",
                                    iter_num,
                                    force=True,
                                )

                        if changed_any:
                            _new_cut_this_iter = True
                            _rebuild_cut_structures()

                            self._mw_cached = None
                            self._mw_mu = np.zeros(len(cut_edge_idx_free), dtype=float)

                            cuts_present_here = True
                            use_cuts_in_dual_here = self.use_cover_cuts and bool(self.best_cuts)

                            _print_cut_table(
                                "Cuts after separation/addition",
                                iter_num,
                                force=True,
                            )

                    except Exception as e:
                        _dbg(
                            f"ERROR in delayed separation at depth={depth}, "
                            f"iter={iter_num}: {e}",
                            iter_num,
                            force=True,
                        )

                    finally:
                        separated_trees.add(tree_key)
                        sep_rounds += 1
                        sep_last_iter = iter_num

                # Seed the multipliers of the covers that have none at the
                # breakpoint where this tree would start respecting them.
                if (
                    cuts_live_now
                    and getattr(self, "mu_warm_start", False)
                    and len(cut_edge_idx_free) > 0
                    and (iter_num == lam_phase_iters or _new_cut_this_iter)
                ):
                    try:
                        _tidx2 = self._last_mst_idx
                        _c2 = getattr(self, "_last_mw", None)

                        if _tidx2 is not None and _c2 is not None:
                            _ests = self.cover_mu_warm_start(
                                _tidx2, _c2, cut_edge_idx_free, rhs_eff_vec
                            )
                            _frac = float(getattr(self, "mu_warm_frac", 1.0))
                            _touched = False

                            for _i, _e in enumerate(_ests):
                                if _e is None or _e <= 0.0:
                                    continue
                                if self.best_cut_multipliers.get(_i, 0.0) > dead_mu_threshold:
                                    continue
                                self.best_cut_multipliers[_i] = float(_e) * _frac
                                _touched = True

                            if _touched:
                                self._invalidate_weight_cache()
                                _dbg(
                                    f"mu warm start: {self.best_cut_multipliers}",
                                    iter_num,
                                    force=True,
                                )
                    except Exception as _exc:
                        _dbg(f"mu warm start failed: {_exc}", iter_num, force=True)

                # Prepare weights for next iteration
                prev_weights = getattr(self, "_last_mw", prev_weights)

                # --------------------------------------------------------------
                # 5.3) Primal and upper bound
                # --------------------------------------------------------------
                is_feasible = mst_length <= self.budget

                self._record_primal_solution(self.last_mst_edges, is_feasible)

                if is_feasible:
                    try:
                        real_weight, real_length = self.compute_real_weight_length()

                        if (
                            not math.isnan(real_weight)
                            and not math.isinf(real_weight)
                            and real_weight < self.best_upper_bound
                        ):
                            old_ub = self.best_upper_bound
                            self.best_upper_bound = real_weight
                            self.best_feasible_edges = list(self.last_mst_edges)

                            _dbg(
                                f"UB improved: old_UB={old_ub}, "
                                f"new_UB={self.best_upper_bound:.6g}, "
                                f"real_length={real_length:.6g}",
                                iter_num,
                                force=True,
                            )

                    except Exception as e:
                        _dbg(
                            f"ERROR updating primal solution: {e}",
                            iter_num,
                            force=True,
                        )

                # Repair fallback: when the natural Lagrangian MST is over
                # budget we still try to construct a feasible incumbent, so
                # B&B gets a finite UB to prune against. Gated by a flag that
                # defaults to OFF, so non-negative-correlation runs (which set
                # no overrides) are completely unaffected.
                elif getattr(self, "enable_primal_repair", False):
                    try:
                        # The budget-aware repair is strong but costly (a
                        # binary search of Kruskals). Running it every
                        # subgradient iteration is wasteful since the incumbent
                        # barely moves. Run the cheap min-length repair each
                        # iteration to guarantee a UB exists, but only run the
                        # expensive budget repair periodically and on the last
                        # iteration. Controlled by `budget_repair_every`.
                        every = getattr(self, "budget_repair_every", 25)
                        # Both the end of the run and the end of the lambda
                        # phase count: the latter keeps the incumbent -- and so
                        # the Polyak gap -- on the same schedule the plain run
                        # follows, which is what makes phase 1 reproduce it.
                        is_last = (
                            iter_num >= total_iters - 1
                            or (cuts_enabled_here and iter_num == lam_phase_iters - 1)
                        )
                        # The expensive budget repair (mu-grid of Kruskals) only
                        # needs to run where it can actually improve the GLOBAL
                        # incumbent: at shallow depth. Deep nodes almost never
                        # beat the root's incumbent, so run only the cheap
                        # min-length repair there. This keeps per-node cost low
                        # so far more nodes are explored. Controlled by
                        # `budget_repair_max_depth` (default: root + a few).
                        max_depth = getattr(self, "budget_repair_max_depth", 3)
                        shallow = (self.depth <= max_depth)
                        want_budget = (
                            getattr(self, "use_budget_repair", False)
                            and shallow
                            and (iter_num % every == 0 or is_last)
                        )
                        # Temporarily toggle budget path per-iteration.
                        saved = getattr(self, "use_budget_repair", False)
                        self.use_budget_repair = want_budget
                        rw, rl, rep_edges = self.primal_repair()
                        self.use_budget_repair = saved
                        # The incumbent is the answer this solver reports, so
                        # check the budget HERE rather than relying on
                        # primal_repair's internal guarantee.  It does hold
                        # (the min-length tree is rejected when it exceeds the
                        # budget, and every swap re-checks), but an incumbent
                        # that silently went over budget would be returned as
                        # the optimum, so the invariant belongs at the point of
                        # use.
                        if (
                            rep_edges is not None
                            and not math.isnan(rw)
                            and not math.isinf(rw)
                            and not math.isnan(rl)
                            and rl <= self.budget + 1e-9
                            and rw < self.best_upper_bound
                        ):
                            old_ub = self.best_upper_bound
                            self.best_upper_bound = rw
                            self.best_feasible_edges = list(rep_edges)
                            self._record_primal_solution(rep_edges, True)
                            _dbg(
                                f"UB improved via repair: old_UB={old_ub}, "
                                f"new_UB={self.best_upper_bound:.6g}, "
                                f"length={rl:.6g}",
                                iter_num,
                                force=True,
                            )
                    except Exception as e:
                        _dbg(f"ERROR in primal_repair: {e}", iter_num, force=True)

                if len(self.primal_solutions) > MAX_SOLUTIONS:
                    self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]

                # --------------------------------------------------------------
                # 5.4) Dual value
                # --------------------------------------------------------------
                lam_for_dual = max(0.0, min(self.lmbda, 1e4))

                if use_cuts_in_dual_here and len(rhs_eff_vec) > 0:
                    mu_vec = np.fromiter(
                        (
                            max(0.0, min(self.best_cut_multipliers.get(i, 0.0), 1e4))
                            for i in range(len(rhs_eff_vec))
                        ),
                        dtype=float,
                        count=len(rhs_eff_vec),
                    )

                    cover_cut_penalty = float(mu_vec @ rhs_eff_vec)

                else:
                    mu_vec = np.zeros(0, dtype=float)
                    cover_cut_penalty = 0.0

                lagrangian_bound = (
                    mst_cost
                    - lam_for_dual * self.budget
                    - cover_cut_penalty
                )

                _dbg(
                    f"Dual: lambda={lam_for_dual:.6g}, "
                    f"mst_cost={mst_cost:.6g}, "
                    f"lambdaB={lam_for_dual * self.budget:.6g}, "
                    f"cover_penalty={cover_cut_penalty:.6g}, "
                    f"LB_candidate={lagrangian_bound:.6g}, "
                    f"best_LB_before={self.best_lower_bound:.6g}, "
                    f"UB={self.best_upper_bound}",
                    iter_num,
                )

                if len(mu_vec) > 0:
                    _dbg(
                        f"mu_vec={mu_vec.tolist()}, "
                        f"rhs_eff_vec={rhs_eff_vec.tolist()}",
                        iter_num,
                    )

                if (
                    not math.isnan(lagrangian_bound)
                    and not math.isinf(lagrangian_bound)
                    and abs(lagrangian_bound) < 1e10
                ):
                    if lagrangian_bound > self.best_lower_bound + 1e-6:
                        old_lb = self.best_lower_bound
                        last_gain_iter = iter_num

                        self.best_lower_bound = lagrangian_bound
                        self.best_lambda = lam_for_dual
                        self.best_mst_edges = self.last_mst_edges
                        self.best_cost = mst_cost
                        self.best_cut_multipliers_for_best_bound = (
                            self.best_cut_multipliers.copy()
                        )

                        _dbg(
                            f"LB improved: old_LB={old_lb:.6g}, "
                            f"new_LB={self.best_lower_bound:.6g}, "
                            f"best_lambda={self.best_lambda:.6g}, "
                            f"saved_mu={self.best_cut_multipliers_for_best_bound}",
                            iter_num,
                            force=True,
                        )

                        _ub_now = min(
                            self.best_upper_bound,
                            getattr(self, "incumbent_ub", float("inf")),
                        )

                        if (
                            _stop_dominated
                            and _ub_now < float("inf")
                            and self.best_lower_bound > _ub_now - _prune_gran + 1e-6
                        ):
                            _dbg(
                                "Node already dominated; stopping the dual",
                                iter_num,
                                force=True,
                            )
                            break

                # --------------------------------------------------------------
                # 5.5) Subgradients
                # --------------------------------------------------------------
                knapsack_subgradient = float(mst_length - self.budget)

                all_mu_small = (
                    not self.best_cut_multipliers
                    or max(self.best_cut_multipliers.values()) <= dead_mu_threshold
                )

                if (
                    cuts_live_now
                    and cuts_present_here
                    and mu_dynamic_here
                    and len(cut_edge_idx_free) > 0
                    and not (is_feasible and all_mu_small)
                ):
                    mst_mask[:] = False

                    _tree_idx = self._last_mst_idx
                    if _tree_idx is not None and len(_tree_idx) == len(mst_edges):
                        mst_mask[_tree_idx] = True
                    else:
                        for e in mst_edges:
                            j = self.edge_indices.get(e)
                            if j is not None:
                                mst_mask[j] = True

                    cut_g_signed = []
                    cut_g_pos = []

                    for i, idxs_free in enumerate(cut_edge_idx_free):
                        lhs_free = int(mst_mask[idxs_free].sum()) if idxs_free.size else 0
                        g_i = float(lhs_free) - float(rhs_eff_vec[i])

                        cut_g_signed.append(g_i)
                        cut_g_pos.append(g_i if g_i > 0.0 else 0.0)

                        if i < len(max_cut_violation) and g_i > max_cut_violation[i]:
                            max_cut_violation[i] = g_i

                        _dbg(
                            f"Cut subgradient cut[{i}]: lhs_free={lhs_free}, "
                            f"rhs_eff={rhs_eff_vec[i]}, "
                            f"g_signed={g_i}, "
                            f"g_pos={cut_g_pos[-1]}, "
                            f"mu_before={self.best_cut_multipliers.get(i, 0.0):.6g}",
                            iter_num,
                        )

                    # Modified:
                    # Use the signed cut subgradient in the norm and μ update.
                    # This allows μ to decrease when the cut becomes slack.
                    cut_subgradients = cut_g_signed

                else:
                    cut_g_signed = []
                    cut_g_pos = []
                    cut_subgradients = []

                    _dbg(
                        f"Skip cut subgradients: cuts_live_now={cuts_live_now}, "
                        f"cuts_present={cuts_present_here}, "
                        f"mu_dynamic={mu_dynamic_here}, "
                        f"num_cut_arrays={len(cut_edge_idx_free)}, "
                        f"is_feasible={is_feasible}, "
                        f"all_mu_small={all_mu_small}",
                        iter_num,
                    )

                norm_sq = knapsack_subgradient ** 2

                for g in cut_subgradients:
                    norm_sq += float(g) ** 2

                # --------------------------------------------------------------
                # 5.6) Polyak step size
                # --------------------------------------------------------------
                # Either the node's own incumbent or one handed in from the
                # search will do to form the gap; without at least one of them
                # there is no Polyak step and alpha collapses to a constant.
                ub_for_gap = min(
                    self.best_upper_bound,
                    getattr(self, "incumbent_ub", float("inf")),
                )

                if (
                    polyak_enabled
                    and ub_for_gap < float("inf")
                    and norm_sq > 0.0
                ):
                    gap = max(0.0, ub_for_gap - lagrangian_bound)
                    alpha = gamma_base * gap / (norm_sq + eps)
                else:
                    gap = None
                    # Before we have a finite UB, avoid the huge first lambda jump.
                    alpha = fallback_alpha

                # Step size for the cut block.
                #
                # "normalized": a fixed distance in modified-weight units along
                # g_mu/||g_mu||, shrinking over the cut phase.  No duality gap
                # enters, which is the point -- see the note on the modes above.
                #
                # "scaled": the joint Polyak ratio taken after the budget row is
                # divided by the mean edge length, which brings the two row
                # families into the same units.  The budget row stays in the
                # norm -- so the step is still moderated by how far the tree is
                # from the budget -- but it no longer outweighs the cut rows by
                # the ~1e6 factor that pinned mu at its initial value.
                #
                # "block": the ratio over the cut subgradients alone.  Kept for
                # the ablation; it overshoots badly (see the note above).
                #
                # "joint": the historical behaviour, alpha_mu = alpha.
                cut_norm_sq = 0.0
                for g in cut_subgradients:
                    cut_norm_sq += float(g) ** 2

                if mu_step_mode == "normalized_inf" and cut_norm_sq > 0.0:
                    # Same fixed distance per iteration, but measured with the
                    # infinity norm instead of the 2-norm.  Under the 2-norm a
                    # pool of k cuts each violated by one has ||g|| = sqrt(k),
                    # so every multiplier advances at mu_travel/sqrt(k): the
                    # more cuts are live, the slower each of them reaches the
                    # range where it reprices the tree.  That penalises exactly
                    # the rung that separates the most -- `full` emits two cuts
                    # per tree where the others emit one.  Dividing by
                    # max|g_i| instead advances each violated cut by
                    # mu_travel regardless of how many are live.
                    cut_phase_k = max(0, iter_num - lam_phase_iters)
                    # Floor the decay at 1% of the opening step.  The decay
                    # horizon (~1/(1-decay) iterations) and the cut-phase
                    # length are set by two independent knobs, so a large
                    # iteration budget drives mu_travel below anything that can
                    # reprice an edge -- and below dead_mu_threshold -- while
                    # every one of those iterations still pays a full Kruskal.
                    # At the shipped budgets (cut phase 10-20 iterations,
                    # decay 0.9) the floor never binds; it only stops the tail
                    # of a long run from being pure overhead.
                    mu_travel = (
                        mu_step_frac
                        * weight_scale
                        * max(mu_step_decay ** cut_phase_k, 0.01)
                    )
                    g_inf = max(abs(float(g)) for g in cut_subgradients)
                    alpha_mu = mu_travel / (gamma_mu * g_inf + eps)
                elif mu_step_mode == "normalized" and cut_norm_sq > 0.0:
                    # Move ||dmu|| = mu_step_frac * weight_scale * decay^k along
                    # g_mu/||g_mu||.  Written as an alpha so the update below is
                    # shared with the other modes: alpha_mu * gamma_mu * g_i is
                    # the i-th component of exactly that vector.
                    cut_phase_k = max(0, iter_num - lam_phase_iters)
                    mu_travel = (
                        mu_step_frac
                        * weight_scale
                        * (mu_step_decay ** cut_phase_k)
                    )
                    alpha_mu = mu_travel / (gamma_mu * math.sqrt(cut_norm_sq) + eps)
                elif mu_step_mode == "block" and gap is not None and cut_norm_sq > 0.0:
                    alpha_mu = gamma_base * gap / (cut_norm_sq + eps)
                elif mu_step_mode == "scaled" and gap is not None and cut_norm_sq > 0.0:
                    g_budget_scaled = knapsack_subgradient / len_scale
                    alpha_mu = gamma_base * gap / (
                        g_budget_scaled ** 2 + cut_norm_sq + eps
                    )
                else:
                    alpha_mu = alpha

                _dbg(
                    f"Step: knapsack_g={knapsack_subgradient:.6g}, "
                    f"cut_g_signed={cut_g_signed}, "
                    f"cut_g_pos={cut_g_pos}, "
                    f"norm_sq={norm_sq:.6g}, "
                    f"gap={gap}, "
                    f"alpha={alpha:.6g}, "
                    f"cut_norm_sq={cut_norm_sq:.6g}, "
                    f"alpha_mu={alpha_mu:.6g}",
                    iter_num,
                )

                # --------------------------------------------------------------
                # 5.7) λ update
                # --------------------------------------------------------------
                lambda_before = self.lmbda

                v_prev = getattr(self, "_v_lambda", 0.0)
                v_new = (
                    self.momentum_beta * v_prev
                    + (1.0 - self.momentum_beta) * knapsack_subgradient
                )

                self._v_lambda = v_new

                delta_lambda = alpha * v_new
                delta_lambda = max(-max_lambda_delta, min(max_lambda_delta, delta_lambda))

                self.lmbda = self.lmbda + delta_lambda
                self.lmbda = max(0.0, min(self.lmbda, 1e4))

                _dbg(
                    f"Lambda update: before={lambda_before:.6g}, "
                    f"v_prev={v_prev:.6g}, "
                    f"v_new={v_new:.6g}, "
                    f"after={self.lmbda:.6g}",
                    iter_num,
                )

                # --------------------------------------------------------------
                # 5.8) μ updates
                #
                # Modified:
                # Signed projected update:
                #     μ_i <- max(0, μ_i + gamma_mu * alpha * g_i)
                #
                # If g_i > 0, the cut is violated and μ_i increases.
                # If g_i < 0, the cut is slack and μ_i decreases.
                # --------------------------------------------------------------
                if cuts_live_now and mu_dynamic_here and len(cut_g_signed) > 0:
                    for i, g in enumerate(cut_g_signed):
                        g = float(g)

                        delta = gamma_mu * alpha_mu * g

                        # Symmetric cap because delta can now be positive or negative.
                        if mu_increment_cap is not None:
                            delta = max(-mu_increment_cap, min(mu_increment_cap, delta))

                        mu_old = float(self.best_cut_multipliers.get(i, 0.0))
                        mu_new = mu_old + delta
                        mu_new = max(0.0, min(mu_new, 1e4))

                        self.best_cut_multipliers[i] = mu_new

                        _dbg(
                            f"Mu signed update cut[{i}]: g={g:.6g}, "
                            f"delta={delta:.6g}, "
                            f"mu_old={mu_old:.6g}, "
                            f"mu_new={mu_new:.6g}",
                            iter_num,
                            force=True,
                        )

                self.step_sizes.append(alpha)
                self.multipliers.append((self.lmbda, self.best_cut_multipliers.copy()))

                # T^(i) paired with the step size alpha_i actually used at
                # iteration i, which is what (11) averages.  ONE list of
                # pairs, not two parallel lists: averaging `primal_solutions`
                # against `step_sizes` drifted apart the moment the primal
                # repair recorded an extra tree (it has no step size) or the
                # primal cap trimmed one end (step_sizes was uncapped), and
                # the length guard then made the average return None for the
                # rest of the node.  The repaired primal is deliberately not
                # collected here: it is a heuristic tree, not an LR
                # subproblem solution, so it belongs in the Dantzig-Wolfe
                # column pool but not in the running average.
                if self.last_mst_edges:
                    self._append_with_cap(
                        self.avg_trees,
                        (tuple(self.last_mst_edges), float(alpha)),
                        self._primal_history_cap,
                    )

                if (
                    cuts_live_now
                    and cut_phase_patience > 0
                    and (iter_num - max(last_gain_iter, sep_last_iter))
                        >= cut_phase_patience
                ):
                    _dbg(
                        f"Cut phase stopped: no gain for {cut_phase_patience} "
                        f"iterations",
                        iter_num,
                        force=True,
                    )
                    break

            # ------------------------------------------------------------------
            # 5.9) Exact cut dual
            #
            # The mu walk above is a subgradient method on a trajectory; the
            # cut dual itself has a closed form (see exact_cut_dual_ascent).
            # Running it here, before the dead-cut sweep, both raises the
            # bound and tells that sweep which covers actually carry price.
            # ------------------------------------------------------------------
            if self.use_cover_cuts and self.best_cuts:
                self.exact_cut_dual_ascent()

            # ------------------------------------------------------------------
            # 6) Drop dead cuts
            # ------------------------------------------------------------------
            if self.use_cover_cuts and self.best_cuts and mu_dynamic_here:
                keep_indices = []

                best_mu_map = getattr(
                    self,
                    "best_cut_multipliers_for_best_bound",
                    self.best_cut_multipliers,
                )

                _dbg(
                    f"Dead-cut check starts: active_cuts={len(self.best_cuts)}, "
                    f"max_cut_violation={max_cut_violation}",
                    force=True,
                )

                for i, (cut, rhs) in enumerate(self.best_cuts):
                    mu_i = float(self.best_cut_multipliers.get(i, 0.0))
                    mu_best_i = float(best_mu_map.get(i, 0.0))

                    ever_useful = (
                        i < len(max_cut_violation)
                        and max_cut_violation[i] > 0.0
                    ) or abs(mu_best_i) >= dead_mu_threshold

                    keep = not (
                        not ever_useful
                        and abs(mu_i) < dead_mu_threshold
                        and abs(mu_best_i) < dead_mu_threshold
                    )

                    _dbg(
                        f"Dead-cut decision cut[{i}]: "
                        f"max_violation={max_cut_violation[i] if i < len(max_cut_violation) else None}, "
                        f"mu_current={mu_i:.6g}, "
                        f"mu_best={mu_best_i:.6g}, "
                        f"ever_useful={ever_useful}, "
                        f"keep={keep}",
                        force=True,
                    )

                    if keep:
                        keep_indices.append(i)

                if len(keep_indices) < len(self.best_cuts):
                    _dbg(
                        f"Dropping dead cuts: keep_indices={keep_indices}, "
                        f"drop_count={len(self.best_cuts) - len(keep_indices)}",
                        force=True,
                    )

                    new_best_cuts = []
                    new_mu = {}
                    new_mu_best = {}
                    new_rhs_eff = {}

                    for new_idx, old_idx in enumerate(keep_indices):
                        new_best_cuts.append(self.best_cuts[old_idx])
                        new_mu[new_idx] = float(
                            self.best_cut_multipliers.get(old_idx, 0.0)
                        )
                        new_mu_best[new_idx] = float(
                            self.best_cut_multipliers_for_best_bound.get(old_idx, 0.0)
                        )
                        new_rhs_eff[new_idx] = int(
                            self._rhs_eff.get(old_idx, self.best_cuts[old_idx][1])
                        )

                    self.best_cuts = new_best_cuts
                    self.best_cut_multipliers = new_mu
                    self.best_cut_multipliers_for_best_bound = new_mu_best
                    self._rhs_eff = new_rhs_eff

                    if self.best_cuts:
                        _rebuild_cut_structures()
                    else:
                        self._cut_edge_idx = []
                        self._cut_edge_idx_all = []
                        rhs_eff_vec = np.zeros(0, dtype=float)

            _print_cut_table("Final cuts before returning from node", force=True)

            # ------------------------------------------------------------------
            # 7) Restore best λ and μ to pass to children
            #
            # Unchanged strategy:
            # λ and μ are both restored to the values that gave the best lower bound.
            # ------------------------------------------------------------------
            if hasattr(self, "best_lambda"):
                _dbg(
                    f"Restore lambda: current={self.lmbda:.6g}, "
                    f"best_lambda={self.best_lambda:.6g}",
                    force=True,
                )
                self.lmbda = self.best_lambda

            if hasattr(self, "best_cut_multipliers_for_best_bound"):
                _dbg(
                    f"Restore best μ for children: "
                    f"{self.best_cut_multipliers_for_best_bound}",
                    force=True,
                )
                self.best_cut_multipliers = (
                    self.best_cut_multipliers_for_best_bound.copy()
                )

            _dbg(
                f"END NODE: best_LB={self.best_lower_bound:.6g}, "
                f"best_UB={self.best_upper_bound}, "
                f"return_new_cuts={len(node_new_cuts)}, "
                f"final_active_cuts={len(self.best_cuts)}",
                force=True,
            )

            end_time = time()
            LagrangianMST.total_compute_time += end_time - start_time
            return self.best_lower_bound, self.best_upper_bound, node_new_cuts


    # ------------------------------------------------------------------
    # Variable fixing by Lagrangian penalties (reduced-cost fixing).
    #
    # Let c = w + lambda*l + sum_i mu_i 1_{S_i} be the priced weights at the
    # dual point this node settled on, T* the priced MST and
    #
    #     LB = c(T*) - lambda*B - sum_i mu_i rhs_i.
    #
    # For ANY tree T that is feasible at this node (it respects the budget and
    # every active cover cut) the dualized terms are nonpositive, so
    #
    #     w(T) >= c(T) - lambda*B - sum_i mu_i rhs_i.                     (*)
    #
    # min{c(T) : T a spanning tree, e in T} = c(T*) + d_in(e) with
    #
    #     d_in(e) = c_e - max{c_f : f on the T*-path between e's ends}     (>= 0)
    #
    # -- the textbook MST sensitivity quantity -- so every feasible tree
    # through e has w(T) >= LB + d_in(e) by (*).  If that already reaches the
    # incumbent, no tree through e can improve on it and e is excluded from
    # this node and its whole subtree.  Symmetrically, for f in T*,
    # min{c(T) : f not in T} = c(T*) + d_out(f) with d_out(f) = repl(f) - c_f,
    # and when that reaches the incumbent f is fixed IN.
    #
    # This is what makes a stronger cut pay in a Lagrangian branch and bound.
    # The cuts move LB up, which lowers the threshold d_in has to clear, and
    # -- for a LIFTED cover -- they also raise c on every support edge the
    # tree does not use, which is precisely where d_in is measured.  So the
    # bound improvement AND the lifting both turn directly into fixed
    # variables, instead of the lifting being dual-neutral ballast.
    #
    # Nothing here is heuristic: T* is a spanning tree, its own edges have
    # d_in = 0, so the surviving graph always contains T* and stays connected.
    # ------------------------------------------------------------------
    def _tree_path_max(self, tree_idx, c):
        """For every pair of vertices, the largest priced weight on the
        T*-path between them.

        Vertices are renumbered by preorder position, so that when vertex x is
        reached every vertex already numbered lies OUTSIDE x's subtree and the
        path from x to it runs through x's parent:

            M[x][v] = max(c(parent edge of x), M[parent(x)][v]).

        One numpy row operation per vertex: 1.4ms at n = 400, exact.
        """
        n = self.num_nodes
        EU = self._edge_u
        EV = self._edge_v

        adj_head = [[] for _ in range(n)]
        for i in tree_idx.tolist():
            u = int(EU[i])
            v = int(EV[i])
            w = float(c[i])
            adj_head[u].append((v, w))
            adj_head[v].append((u, w))

        pos = np.full(n, -1, dtype=np.int64)
        par_pos = np.zeros(n, dtype=np.int64)
        par_cost = np.zeros(n, dtype=np.float64)

        pos[0] = 0
        par_pos[0] = -1
        k = 1
        stack = [0]

        while stack:
            x = stack.pop()
            for (y, w) in adj_head[x]:
                if pos[y] < 0:
                    pos[y] = k
                    par_pos[k] = pos[x]
                    par_cost[k] = w
                    k += 1
                    stack.append(y)

        if k != n:
            # T* does not span: nothing to say about any pair.
            return None, None

        M = np.zeros((n, n), dtype=np.float64)

        for kk in range(1, n):
            pk = par_pos[kk]
            row = M[kk]
            np.maximum(M[pk, :kk], par_cost[kk], out=row[:kk])
            M[:kk, kk] = row[:kk]

        return M, pos

    # ------------------------------------------------------------------
    # Exact cut dual: the cover multipliers in closed form.
    #
    # For ONE cover (S, b) dualised into the node relaxation the Lagrangian
    # dual has no gap.  {x in P_base(M) : x(S) <= b} is the intersection of
    # the graphic matroid with a partition matroid, hence an integral
    # polytope, so
    #
    #     max_{mu >= 0} L(lambda, mu) = min{ c_lambda(T) : |T n S| <= b }.
    #
    # ONE cover.  Two or more dualised together are a matroid under two
    # cardinality constraints, which is not integral in general, and the
    # coordinate ascent below is then a monotone heuristic rather than an
    # exact solve of the joint dual: over 1964 random tiny nodes it fell
    # short of the joint dual optimum on 135 of them (worst 27%), and the
    # joint dual itself fell short of the constrained primal on 6 (worst
    # 3.2%), always with three or more covers.  It never exceeded the
    # primal, which is the only property the search depends on.
    #
    # h(mu) := MST(c + mu 1_S) - mu b is concave and piecewise linear, and its
    # slope at mu = 0 is v := |T* n S| - b, the violation the separator
    # certified.  Up to the first breakpoint the priced tree does not move, so
    #
    #     h(mu) = c(T*) + v mu           for 0 <= mu <= mu1,
    #     mu1   = min{ c_f - c_e : e in T* n S, f not in S u T*,
    #                              e on the T*-path of f },
    #
    # which is the cheapest way to buy the cover one unit of slack: drop a
    # support edge of the tree and reconnect OUTSIDE the support.  When v = 1
    # -- the violation of every cover this separator produces -- the slope
    # past mu1 is zero and h(mu1) = c(T*) + mu1 is the exact optimum of the
    # cut dual.  So a subgradient walk on mu is not needed at all: one
    # ascending sweep of the non-support non-tree edges gives mu* outright.
    #
    # Three consequences the implementation below exploits:
    #
    #   * mu1 is a REPLACEMENT cost outside S, so widening the support --
    #     exactly what the Lemma 1 and Lemma 2 liftings do -- removes
    #     candidate replacements and can only raise mu1.  Lifting is what
    #     pays for the bound, not the right-hand side, which Proposition 1's
    #     contraction already leaves tight.
    #   * T* stays optimal at mu = mu1, so the covers can be priced one after
    #     another off the SAME tree: each cut sees the prices its predecessors
    #     put on its own replacement candidates, which is a monotone
    #     coordinate ascent on the joint dual and needs no MST in between.
    #   * a cover whose mu* is 0 cannot help the bound at this dual point, so
    #     the sweep doubles as an exact cut-selection test.
    #
    # The bound that is finally accepted is recomputed from a fresh MST at the
    # final prices, so it is a valid Lagrangian bound for ANY mu >= 0 the
    # breakpoint arithmetic produced -- rounding cannot make it wrong, only
    # weaker.
    # ------------------------------------------------------------------
    def _rooted_tree(self, tree_idx):
        """Root the priced tree at vertex 0.

        Returns (par, par_edge, depth, order) as plain lists, `order` in
        nondecreasing depth so a caller can collapse the tree upward in one
        pass.  Returns None when the tree does not span.
        """
        n = self.num_nodes
        EU = self._edge_u
        EV = self._edge_v

        adj_head = [-1] * n
        nxt = [-1] * (2 * len(tree_idx))
        dst = [0] * (2 * len(tree_idx))
        via = [0] * (2 * len(tree_idx))
        k = 0

        for i in tree_idx.tolist():
            u = int(EU[i])
            v = int(EV[i])
            dst[k] = v; via[k] = i; nxt[k] = adj_head[u]; adj_head[u] = k; k += 1
            dst[k] = u; via[k] = i; nxt[k] = adj_head[v]; adj_head[v] = k; k += 1

        par = [-1] * n
        par_edge = [-1] * n
        depth = [0] * n
        seen = [False] * n
        order = [0]
        seen[0] = True
        qi = 0

        while qi < len(order):
            x = order[qi]
            qi += 1
            a = adj_head[x]
            while a != -1:
                y = dst[a]
                if not seen[y]:
                    seen[y] = True
                    par[y] = x
                    par_edge[y] = via[a]
                    depth[y] = depth[x] + 1
                    order.append(y)
                a = nxt[a]

        if len(order) != n:
            return None

        return par, par_edge, depth, order

    def _first_swap_margin(self, c, tree_idx, in_tree, Smask,
                           par, par_edge, depth, order):
        """mu1: the cheapest priced swap that takes one edge out of T* n S.

        Every tree edge OUTSIDE the support is collapsed up front, so the
        upward walks only ever meet the support edges whose replacement is
        wanted; the candidates are scanned in nondecreasing price and the
        sweep stops as soon as no remaining one can beat the best margin.
        Fixed edges are never in a (reduced) support, so they are collapsed
        with the rest and can never be the edge that leaves -- which is what
        the node's own fixings require.
        """
        n = self.num_nodes
        EU = self._edge_u
        EV = self._edge_v

        up = list(range(n))

        def _up_find(x):
            while up[x] != x:
                up[x] = up[up[x]]
                x = up[x]
            return x

        targets = 0
        max_ce = float("-inf")

        for x in order:
            e = par_edge[x]
            if e < 0:
                continue
            if Smask[e]:
                targets += 1
                ce = float(c[e])
                if ce > max_ce:
                    max_ce = ce
            else:
                up[x] = _up_find(par[x])

        if targets == 0:
            return float("inf"), 0

        cand = self._cand_idx
        nt = cand[(~in_tree[cand]) & (~Smask[cand])]

        if nt.size == 0:
            return float("inf"), targets

        nt = nt[np.argsort(c[nt], kind="stable")]

        best = float("inf")
        left = targets

        for i in nt.tolist():
            if left == 0:
                break
            cf = float(c[i])
            if best < float("inf") and cf - max_ce >= best:
                # c is scanned in nondecreasing order, so no later candidate
                # can beat `best` either.
                break
            a = _up_find(int(EU[i]))
            b = _up_find(int(EV[i]))
            while a != b:
                if depth[a] < depth[b]:
                    a, b = b, a
                e = par_edge[a]
                if e >= 0 and Smask[e]:
                    m = cf - float(c[e])
                    if m < best:
                        best = m
                    left -= 1
                up[a] = _up_find(par[a])
                a = _up_find(a)

        return best, targets

    def tree_edge_margins(self, c, tree_idx):
        """margin(e) = (cheapest priced replacement of e) - c_e, per tree edge.

        The same offline MST-sensitivity sweep _first_swap_margin runs, but
        over ALL non-tree candidates and recording every tree edge instead of
        the minimum over one support.  Since the replacement pool here is not
        restricted to the complement of a support, the value is a LOWER bound
        on the margin any support containing e would see -- which is what a
        separator wants: max over candidate covers S of min_{e in S} margin(e)
        is then a lower bound on that cover's exact dual price mu*.

        Returns a dict tree-edge-index -> margin (+inf for a bridge), or None
        when the tree cannot be rooted.
        """
        rooted = self._rooted_tree(tree_idx)

        if rooted is None:
            return None

        par, par_edge, depth, order = rooted
        n = self.num_nodes
        EU = self._edge_u
        EV = self._edge_v

        in_tree = np.zeros(len(self.edge_list), dtype=bool)
        in_tree[tree_idx] = True

        cand = self._cand_idx
        nt = cand[~in_tree[cand]]
        out = {}

        if nt.size:
            nt = nt[np.argsort(c[nt], kind="stable")]
            up = list(range(n))

            def _up_find(x):
                while up[x] != x:
                    up[x] = up[up[x]]
                    x = up[x]
                return x

            left = int(tree_idx.size)

            for i in nt.tolist():
                if left == 0:
                    break

                cf = float(c[i])
                a = _up_find(int(EU[i]))
                b = _up_find(int(EV[i]))

                while a != b:
                    if depth[a] < depth[b]:
                        a, b = b, a

                    e = par_edge[a]

                    if e >= 0 and e not in out:
                        out[e] = cf - float(c[e])
                        left -= 1

                    up[a] = _up_find(par[a])
                    a = _up_find(a)

        for i in tree_idx.tolist():
            if i not in out:
                out[i] = float("inf")

        return out

    def exact_cut_dual_ascent(self, tol=1e-9):
        """Price the active covers at their exact dual optimum.

        Returns True when the node bound was raised.  Leaves every dual and
        cache untouched otherwise.
        """
        if not (self.use_cover_cuts and self.best_cuts):
            return False

        if not bool(getattr(self, "exact_cut_dual", True)):
            return False

        cut_idx = getattr(self, "_cut_edge_idx", None)
        ncuts = len(self.best_cuts)

        if not cut_idx or len(cut_idx) != ncuts:
            return False

        rounds = max(1, int(getattr(self, "exact_cut_rounds", 3)))
        max_cuts = max(1, int(getattr(self, "exact_cut_max", 5)))

        # _mst_core publishes the tree it just built for compute_real_weight_length
        # to price; these are private probes, so the node's own tree is restored.
        save_idx = getattr(self, "_last_mst_idx", None)
        save_list = getattr(self, "_last_mst_list", None)

        try:
            lam = max(0.0, min(float(getattr(self, "lmbda", 0.0)), 1e4))
            c = np.asarray(self.edge_weights, dtype=float) + lam * np.asarray(
                self.edge_lengths, dtype=float)

            cost0, _len0, _tree, idx = self._mst_core(c)

            if idx is None:
                return False

            rooted = self._rooted_tree(idx)

            if rooted is None:
                return False

            par, par_edge, depth, order = rooted

            m = len(self.edge_list)
            in_tree = np.zeros(m, dtype=bool)
            in_tree[idx] = True
            Smask = np.zeros(m, dtype=bool)

            rhs_eff = getattr(self, "_rhs_eff", {})
            mu = [0.0] * ncuts
            viol = []

            for i in range(ncuts):
                S = cut_idx[i]
                b = float(rhs_eff.get(i, self.best_cuts[i][1]))
                v = (float(in_tree[S].sum()) - b) if S.size else 0.0
                viol.append(v)

            live = [i for i in range(ncuts) if viol[i] > 0.0 and cut_idx[i].size]

            if not live:
                return False

            live.sort(key=lambda i: (-viol[i], i))
            live = live[:max_cuts]
            gain = 0.0

            for _rnd in range(rounds):
                moved = False

                for i in live:
                    S = cut_idx[i]
                    Smask[S] = True
                    try:
                        b = float(rhs_eff.get(i, self.best_cuts[i][1]))
                        v = float(in_tree[S].sum()) - b

                        if v <= 0.0:
                            continue

                        step, _targets = self._first_swap_margin(
                            c, idx, in_tree, Smask, par, par_edge, depth, order)
                    finally:
                        Smask[S] = False

                    if not (step > tol) or not math.isfinite(step):
                        continue

                    if mu[i] + step > 1e4:
                        step = 1e4 - mu[i]
                        if step <= tol:
                            continue

                    mu[i] += step
                    c[S] += step
                    gain += v * step
                    moved = True

                if not moved:
                    break

            if gain <= tol:
                return False

            # Valid for any mu >= 0: price the true minimum at the final mu.
            cost1, _len1, _tree1, idx1 = self._mst_core(c)

            if idx1 is None:
                return False

            penalty = 0.0
            for i in range(ncuts):
                if mu[i] > 0.0:
                    penalty += mu[i] * float(rhs_eff.get(i, self.best_cuts[i][1]))

            lb = cost1 - lam * self.budget - penalty

            if math.isnan(lb) or math.isinf(lb):
                return False

            if lb <= float(self.best_lower_bound) + 1e-9:
                return False

            improvement = lb - float(self.best_lower_bound)
            self.best_lower_bound = lb
            self.best_lambda = lam

            new_mu = {i: float(mu[i]) for i in range(ncuts)}
            self.best_cut_multipliers = new_mu
            self.best_cut_multipliers_for_best_bound = dict(new_mu)
            self._invalidate_weight_cache()

            LagrangianMST.exact_dual_nodes += 1
            LagrangianMST.exact_dual_gain += improvement
            if getattr(self, "_is_probe", False):
                LagrangianMST.exact_dual_probe_nodes += 1

            return True
        finally:
            self._last_mst_idx = save_idx
            self._last_mst_list = save_list

    def reduced_cost_fixing(self, incumbent_ub, granularity=1.0, tol=1e-6,
                            want_fix=True):
        """Edges this node can drop, and edges it must keep.

        Returns (excluded_indices, fixed_indices, lb_used), or (None, None,
        None) when the test cannot be formed (no finite incumbent, no priced
        tree, an instance too large for the pairwise table).
        """
        n = self.num_nodes

        if not (incumbent_ub < float("inf")) or n < 3:
            return None, None, None

        if n > int(getattr(self, "rc_fix_max_nodes", 3000)):
            # The pairwise table is n^2 doubles; past this it is not worth it.
            return None, None, None

        # Re-price at the dual point solve() restored (best lambda, best mu)
        # and recompute the tree there, so the bound, the tree and the
        # weights below are one consistent triple.  Any (lambda, mu >= 0)
        # gives a valid bound, so this is safe even if the pool moved after
        # the best bound was recorded.
        self._invalidate_weight_cache()
        c = self.compute_modified_weights()
        cost, length, tree, idx = self._mst_core(c)

        if idx is None or not tree:
            return None, None, None

        lam = max(0.0, min(float(getattr(self, "lmbda", 0.0)), 1e4))
        penalty = 0.0

        if self.use_cover_cuts and self.best_cuts:
            rhs_eff = getattr(self, "_rhs_eff", {})
            for i, (_cut, rhs) in enumerate(self.best_cuts):
                mu_i = max(0.0, min(float(self.best_cut_multipliers.get(i, 0.0)), 1e4))
                if mu_i > 0.0:
                    penalty += mu_i * float(rhs_eff.get(i, rhs))

        lb = cost - lam * self.budget - penalty

        if math.isnan(lb) or math.isinf(lb):
            return None, None, None

        # An improving tree must come in strictly under the incumbent, and
        # every objective value is an integer, so it must reach
        # incumbent - granularity.
        slack = float(incumbent_ub) - float(granularity) - lb

        if slack < -tol:
            # The node itself is already dominated; the caller prunes it.
            return None, None, lb

        M, pos = self._tree_path_max(idx, c)

        if M is None:
            return None, None, lb

        EU = self._edge_u
        EV = self._edge_v

        d_in = c - M[pos[EU], pos[EV]]
        ex_idx = np.flatnonzero(d_in > slack + tol)

        fix_idx = None

        if want_fix:
            fix_idx = self._fix_by_replacement(idx, c, slack, tol, M, pos)

        return ex_idx, fix_idx, lb

    def _fix_by_replacement(self, tree_idx, c, slack, tol, M, pos):
        """Tree edges no improving tree can do without.

        d_out(f) = repl(f) - c_f, with repl(f) the cheapest admissible
        non-tree edge whose T*-path crosses f (+inf when f is a bridge).  One
        ascending sweep of the non-tree edges with a collapsing union-find
        settles every tree edge the sweep reaches; a tree edge the sweep has
        not reached when it stops only has repl >= the last weight scanned,
        which is still a valid LOWER bound on d_out and can still fix.
        """
        n = self.num_nodes
        EU = self._edge_u
        EV = self._edge_v

        tree_list = tree_idx.tolist()
        in_tree = np.zeros(len(self.edge_list), dtype=bool)
        in_tree[tree_idx] = True

        # Root T* by the preorder the path-max table used, so `pos` doubles as
        # the depth order: a vertex always precedes its descendants.
        par = np.full(n, -1, dtype=np.int64)
        par_edge = np.full(n, -1, dtype=np.int64)
        order = np.empty(n, dtype=np.int64)
        order[pos] = np.arange(n, dtype=np.int64)

        adj = [[] for _ in range(n)]
        for i in tree_list:
            u = int(EU[i])
            v = int(EV[i])
            adj[u].append((v, i))
            adj[v].append((u, i))

        seen = [False] * n
        root = int(order[0])
        seen[root] = True
        stack = [root]
        depth = [0] * n

        while stack:
            x = stack.pop()
            for (y, i) in adj[x]:
                if not seen[y]:
                    seen[y] = True
                    par[y] = x
                    par_edge[y] = i
                    depth[y] = depth[x] + 1
                    stack.append(y)

        # Non-tree admissible edges, cheapest first.
        cand = self._cand_idx
        nt = cand[~in_tree[cand]]

        if nt.size == 0:
            return np.empty(0, dtype=np.int64)

        nt = nt[np.argsort(c[nt], kind="stable")]

        up = list(range(n))

        def _up_find(x):
            while up[x] != x:
                up[x] = up[up[x]]
                x = up[x]
            return x

        repl = {}
        remaining = len(tree_list)
        scan_cap = int(getattr(self, "rc_fix_scan_cap", 4)) * n
        last_c = float("inf")
        scanned = 0

        for i in nt.tolist():
            if remaining <= 0 or scanned >= scan_cap:
                break

            scanned += 1
            ci = float(c[i])
            last_c = ci

            a = _up_find(int(EU[i]))
            b = _up_find(int(EV[i]))

            while a != b:
                if depth[a] < depth[b]:
                    a, b = b, a

                pe = int(par_edge[a])
                if pe < 0:
                    break

                if pe not in repl:
                    repl[pe] = ci
                    remaining -= 1

                up[a] = _up_find(int(par[a]))
                a = _up_find(a)

        if remaining <= 0:
            # Every tree edge was settled, so nothing is left to bound.
            unreached = float("inf")
        else:
            # The sweep stopped early; anything it did not reach has
            # repl >= the last weight it looked at.
            unreached = last_c if scanned >= scan_cap else float("inf")

        out = []
        for i in tree_list:
            r = repl.get(i, unreached)
            if r - float(c[i]) > slack + tol:
                out.append(i)

        return np.asarray(out, dtype=np.int64)

    def cover_mu_warm_start(self, tree_idx, c, cut_arrays, rhs_eff):
        """Breakpoint estimate for each cover multiplier.

        Raising mu_i prices EVERY edge of S_i together, so a tree edge
        f in T n S_i can only leave the tree for a replacement OUTSIDE S_i,
        and it leaves once

            mu >= margin(f) = min{c_g : g crosses f's cut, g not in S_i} - c_f.

        The tree therefore stops violating the cover at about the
        (|T n S_i| - rhs_i)-th smallest margin, which is exactly the point a
        one-dimensional search over mu_i would find -- and exactly the point
        the subgradient otherwise has to walk to one capped step at a time.
        Starting there is what lets the cut phase spend its iterations
        refining the multiplier instead of ramping it.

        It also puts the strengthenings on a footing where their difference
        shows: a lifted support leaves fewer edges outside itself to replace
        with, so its margins -- and with them the multiplier the cover can
        carry -- are larger.

        This is a STARTING POINT only.  The subgradient continues from it and
        the node bound stays the maximum over the whole trajectory, so a poor
        estimate costs iterations, never bound.

        OFF by default (`mu_warm_start`).  The estimate is the breakpoint for
        ONE cover against the tree in front of it, and a pool of five priced
        together overshoots: measured over 12 instances at n = 400,
        density 0.2 it moved the ladder's node counts the wrong way on both
        seed sets (seed 42: literature 663 -> 605 but lemma1 589 -> 630;
        seed 101: every rung up, 342/313/299 -> 419/398/347) and cost time
        on both.  It is kept because it is the right instrument once the
        multipliers are fitted one at a time rather than jointly.

        All the margins come from one ascending sweep of the non-tree edges
        per cover with a collapsing union-find -- the standard offline MST
        sensitivity pass, O(|A| a(n)).
        """
        n = self.num_nodes
        EU = self._edge_u
        EV = self._edge_v
        tl = tree_idx.tolist()

        in_tree = np.zeros(len(self.edge_list), dtype=bool)
        in_tree[tree_idx] = True

        adj = [[] for _ in range(n)]
        for i in tl:
            u = int(EU[i])
            v = int(EV[i])
            adj[u].append((v, i))
            adj[v].append((u, i))

        par = [-1] * n
        par_edge = [-1] * n
        depth = [0] * n
        seen = [False] * n
        seen[0] = True
        stack = [0]

        while stack:
            x = stack.pop()
            for (y, i) in adj[x]:
                if not seen[y]:
                    seen[y] = True
                    par[y] = x
                    par_edge[y] = i
                    depth[y] = depth[x] + 1
                    stack.append(y)

        cand = self._cand_idx
        nt = cand[~in_tree[cand]]

        if nt.size:
            nt = nt[np.argsort(c[nt], kind="stable")]

        nt_list = nt.tolist()
        out = []

        for k, arr in enumerate(cut_arrays):
            if arr.size == 0:
                out.append(None)
                continue

            Sset = set(arr.tolist())
            inside = [i for i in tl if i in Sset]
            viol = len(inside) - int(rhs_eff[k])

            if viol <= 0:
                out.append(0.0)
                continue

            target = set(inside)
            remaining = len(target)
            up = list(range(n))

            def _up_find(x):
                while up[x] != x:
                    up[x] = up[up[x]]
                    x = up[x]
                return x

            margins = []

            for g in nt_list:
                if remaining <= 0:
                    break
                if g in Sset:
                    continue

                cg = float(c[g])
                a = _up_find(int(EU[g]))
                b = _up_find(int(EV[g]))

                while a != b:
                    if depth[a] < depth[b]:
                        a, b = b, a

                    pe = par_edge[a]
                    if pe < 0:
                        break

                    if pe in target:
                        margins.append(cg - float(c[pe]))
                        target.discard(pe)
                        remaining -= 1

                    up[a] = _up_find(par[a])
                    a = _up_find(a)

            if len(margins) >= viol:
                margins.sort()
                est = margins[viol - 1]
            elif margins:
                est = max(margins)
            else:
                # Nothing outside S_i can replace any of them: no finite
                # multiplier makes this tree respect the cover.
                est = None

            out.append(None if est is None else max(0.0, est))

        return out

    def compute_mst_for_lambda(self, lambda_val):
        modified_edges = []
        for i, (u, v) in enumerate(self.edge_list):
            modified_w = self.edge_weights[i] + lambda_val * self.edge_lengths[i]
            for cut_idx, (cut, _) in enumerate(self.best_cuts):
                if (u, v) in cut:
                    modified_w += self.best_cut_multipliers.get(cut_idx, 0)
            modified_edges.append((u, v, modified_w))
        return self.compute_mst(modified_edges)

    def _log_fractional_solution(self, method, edge_weights, msts, elapsed_time):
        if self.verbose:
            total_weight = sum(self.edge_weights[self.edge_indices[e]] * w for e, w in edge_weights.items())
            total_length = sum(self.edge_lengths[self.edge_indices[e]] * w for e, w in edge_weights.items())
            print(f"{method} solution: {len(edge_weights)} edges, "
                  f"weight={total_weight:.2f}, length={total_length:.2f}, time={elapsed_time:.2f}s")
            print(f"MSTs used: {len(msts)}")

    
    
    def harvest_columns(self, want=6):
        """Extra columns for the restricted master, by bracketing the budget.

        The multiplier sequence is a poor sampler of trees: it converges, so
        it revisits the same tree over and over.  Measured at n = 300,
        density 0.05 it stores about sixteen trees per node but only THREE
        distinct ones, and two or fewer on 42% of the nodes.  With the
        convexity and budget rows a vertex of (13)-(15) has at most two
        nonzero weights, so a two-column pool can only ever produce a point
        whose fractional entries are the symmetric difference of two trees --
        which is why the indicator came out near-integral, with a median of
        one strictly fractional entry.

        Sampling lambda instead of accepting whatever the subgradient walked
        past fixes the pool at its source.  A small lambda prices length
        cheaply and returns a long tree, a large one returns a short tree, so
        walking outwards from the incumbent lambda brackets the budget and
        gives the master columns it can actually interpolate between.  Each
        sample is one MST, and the walk stops as soon as the pool is big
        enough and the budget is bracketed.

        Returns the list of distinct trees harvested (possibly empty).
        """
        lam0 = max(0.0, min(float(getattr(self, "lmbda", 0.0)), 1e4))
        W = np.asarray(self.edge_weights, dtype=float)
        L = np.asarray(self.edge_lengths, dtype=float)
        B = float(self.budget)

        seen = set()
        need = self.num_nodes - 1

        for tree, _f in (getattr(self, "primal_solutions", None) or []):
            if tree and len(tree) == need:
                seen.add(frozenset(tree))

        under = over = False

        for t in seen:
            ln = sum(float(L[self.edge_indices[e]]) for e in t
                     if e in self.edge_indices)
            if ln <= B:
                under = True
            else:
                over = True

        scale = lam0 if lam0 > 1e-9 else 1.0
        ladder = [0.0, 0.25, 0.5, 0.75, 1.25, 2.0, 4.0, 16.0]

        save_idx = getattr(self, "_last_mst_idx", None)
        save_list = getattr(self, "_last_mst_list", None)
        out = []

        try:
            for f in ladder:
                if len(seen) >= want and under and over:
                    break

                lam = 0.0 if f == 0.0 else scale * f

                if lam > 1e4:
                    continue

                _c, ln, tree, idx = self._mst_core(W + lam * L)

                if idx is None or not tree or len(tree) != need:
                    continue

                key = frozenset(tree)

                if len(key) != need or key in seen:
                    continue

                seen.add(key)
                out.append(list(tree))

                if ln <= B:
                    under = True
                else:
                    over = True
        except Exception:
            return out
        finally:
            self._last_mst_idx = save_idx
            self._last_mst_list = save_list

        return out

    def compute_dantzig_wolfe_solution(self, node):
        """Fractional primal point: the best convex combination of the trees
        the multiplier sequence produced, under the budget and the active
        cover cuts.  Used only to rank branching candidates.
        """
        start_time = time()

        if not self.primal_solutions:
            if self.verbose:
                print("Insufficient primal solutions for Dantzig-Wolfe")
            return None

        need = self.num_nodes - 1

        # The stored trees already carry normalised (min, max) edges, so they
        # go straight into a frozenset -- the old form re-sorted every edge of
        # every stored tree on every node.  Duplicates are dropped: the dual
        # revisits the same tree for many iterations in a row, and a repeated
        # column adds nothing to the LP while costing a full pass in the
        # diversity scan below.
        seen = set()
        valid_msts = []

        pool = list(self.primal_solutions)

        if int(getattr(self, "dw_harvest", 0)) > 0:
            for t in self.harvest_columns(int(getattr(self, "dw_harvest", 0))):
                pool.append((t, None))

        for mst_edges, _is_feasible in pool:
            if not mst_edges or len(mst_edges) != need:
                continue

            key = frozenset(mst_edges)

            if len(key) != need or key in seen:
                continue

            seen.add(key)
            valid_msts.append(key)

        if not valid_msts:
            if self.verbose:
                print("No valid MSTs after filtering")
            return None

        if len(valid_msts) == 1:
            if self.verbose:
                print("Dantzig-Wolfe: single MST, returning as integral solution")
            return {e: 1.0 for e in valid_msts[0]}

        if self.verbose:
            print(f"Using {len(valid_msts)} valid MSTs for Dantzig-Wolfe")

        # Which columns the restricted master gets.
        #
        # `dw_columns` selects the policy.  "diverse" is the max-coverage
        # greedy below: it repeatedly takes the tree contributing the most
        # NEW edges, which is not in the paper and which actively selects
        # AGAINST the trees the multiplier sequence converged to -- those look
        # alike, so the greedy takes one and discards the rest in favour of
        # early, far-from-optimal trees.  "recent" keeps the last max_msts
        # distinct trees, i.e. the ones generated closest to the best dual
        # point, which is what (12)-(15) samples.  "all" keeps every distinct
        # tree up to the cap.
        max_msts = min(int(getattr(self, "dw_max_columns", 10)), len(valid_msts))
        policy = str(getattr(self, "dw_columns", "diverse")).lower()

        if policy in ("recent", "last"):
            selected_msts = valid_msts[-max_msts:]
        elif policy == "all":
            selected_msts = valid_msts[:max_msts]
        else:
            selected_msts = None

        if selected_msts is None:
            selected_msts = []
            covered = set()
            remaining = list(valid_msts)

            while remaining and len(selected_msts) < max_msts:
                best_i = -1
                best_score = -1

                for i, mst in enumerate(remaining):
                    score = len(mst - covered)
                    if score > best_score:
                        best_score = score
                        best_i = i

                if best_i < 0:
                    break

                chosen = remaining.pop(best_i)
                selected_msts.append(chosen)
                covered |= chosen

        if len(selected_msts) < 2:
            if self.verbose:
                print(f"Only {len(selected_msts)} diverse MSTs selected")
            return None

        num_msts = len(selected_msts)
        edge_indices = self.edge_indices
        W = self.edge_weights
        L = self.edge_lengths

        cols = [
            np.fromiter((edge_indices[e] for e in mst), dtype=np.int64, count=need)
            for mst in selected_msts
        ]

        # Objective: minimise total weight (the tie-break term keeps the LP
        # from being degenerate between identical-weight columns).
        c = [float(W[idx].sum()) + 0.1 * (1.0 / num_msts) for idx in cols]

        A_eq = [np.ones(num_msts)]
        b_eq = [1.0]

        lengths = [float(L[idx].sum()) for idx in cols]
        A_ub = [lengths]
        b_ub = [float(self.budget)]

        # Cover cuts (limited, to avoid an infeasible LP).
        if self.best_cuts and len(self.best_cuts) <= 20:
            for cut_support, rhs in self.best_cuts:
                row = [float(len(mst & cut_support)) for mst in selected_msts]
                A_ub.append(row)
                b_ub.append(float(rhs))

        bounds = [(0, None)] * num_msts

        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method="highs")

            if not res.success:
                if self.verbose:
                    print(f"LP with cuts failed: {res.message}, trying without cuts")
                res = linprog(c, A_ub=[lengths], b_ub=[float(self.budget)],
                              A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
                if not res.success:
                    if self.verbose:
                        print(f"LP without cuts also failed: {res.message}")
                    return None

            lambda_k = res.x
        except Exception as e:
            if self.verbose:
                print(f"LP solver error: {e}")
            return None

        # Aggregate over the COLUMNS, not over E.  Every edge outside the
        # selected trees has weight zero by construction, so the old sweep of
        # all m edges against every column (16k x 10 membership tests per
        # node, 18% of a whole run at n = 400) was computing zeros.
        edge_weights = {}

        for k, mst in enumerate(selected_msts):
            lk = float(lambda_k[k])

            if lk <= 1e-12:
                continue

            for e in mst:
                edge_weights[e] = edge_weights.get(e, 0.0) + lk

        if edge_weights:
            edge_weights = {e: w for e, w in edge_weights.items() if w > 1e-6}

        if self.verbose:
            print(f"Dantzig-Wolfe solution: {len(edge_weights)} edges")

        return edge_weights if edge_weights else None

    def compute_weighted_average_solution(self):
        """Section 5.2: the step-size-weighted running average of the LR trees.

            xbar = sum_i (alpha_i / sum_j alpha_j) * chi_{T^(i)}

        A convex combination of spanning-tree incidence vectors, so it lies in
        the spanning-tree polytope, and edges of F+ lie in every tree and come
        out at one.  Unlike the Dantzig-Wolfe indicator it is NOT budget-aware
        -- nothing stops the average from breaking the length budget -- which
        is the gap Section 5.3 exists to close.

        Returns an edge -> value dict, or None when no tree was recorded.
        """
        pairs = getattr(self, "avg_trees", None)

        if not pairs:
            return None

        total = 0.0

        for _tree, a in pairs:
            if a > 0.0:
                total += a

        if total <= 0.0:
            return None

        out = {}

        for tree, a in pairs:
            if a <= 0.0:
                continue

            w = a / total

            for e in tree:
                out[e] = out.get(e, 0.0) + w

        if not out:
            return None

        # Convexity makes this automatic; the clamp only absorbs rounding.
        for e in out:
            v = out[e]
            if v > 1.0:
                out[e] = 1.0
            elif v < 0.0:
                out[e] = 0.0

        return out

    def branching_margins(self):
        """Priced swap margin of every tree edge: an LR-native branching signal
        that borrows nothing from Section 5.

        For the priced tree T* the margin of e in T* is repl(e) - c_e >= 0,
        the extra cost of the cheapest replacement.  A small margin means the
        relaxation is nearly indifferent between keeping e and swapping it
        out, which is the Lagrangian analogue of a variable sitting at a
        fractional value -- and unlike the Dantzig-Wolfe master and the
        running average it needs no tree pool and no LP.

        That independence is the point: it lets the MST-candidate branching
        variants order their candidates sensibly WITHOUT using the fractional
        indicators, so a comparison against the fractional-candidate variants
        measures the indicators rather than the absence of any ordering at
        all.

        Returns {edge: margin} over the tree edges (+inf where the sweep found
        no replacement), or None when no tree is available.
        """
        save_idx = getattr(self, "_last_mst_idx", None)
        save_list = getattr(self, "_last_mst_list", None)

        try:
            c = np.asarray(self.compute_modified_weights(), dtype=float)
            _cost, _len, _tree, idx = self._mst_core(c)

            if idx is None:
                return None

            mg = self.tree_edge_margins(c, idx)

            if not mg:
                return None

            el = self.edge_list

            return {el[i]: float(v) for i, v in mg.items()}
        except Exception:
            return None
        finally:
            self._last_mst_idx = save_idx
            self._last_mst_list = save_list

    def lr_fractionality(self, gap=None):
        """A GRADED fractionality surrogate, defined on every free edge.

        The Dantzig-Wolfe point is a very sparse notion of uncertainty: a
        vertex of (13)-(15) has at most two nonzero weights, so it marks a
        median of one or two edges as strictly fractional and says nothing
        about the rest.  Every branching rule then chooses from the same pair
        and the rules cannot differ, however good or bad their scores are.

        The Lagrangian offers a continuous notion instead.  For e in the
        priced tree, the swap margin repl(e) - c_e is what it would cost to
        push e out; for f outside it, d_in(f) is what it would cost to bring
        f in.  Reduced-cost fixing already uses exactly these against the
        optimality gap: a margin at or above the gap PROVES the edge cannot
        move in an improving tree.  So normalising the margin by the gap
        grades an edge from "provably settled" to "free to go either way":

            e in T*   :  x = 1 - 0.5 * max(0, 1 - margin(e)/gap)
            f not in T*:  x =     0.5 * max(0, 1 - d_in(f)/gap)

        which is 1 or 0 for an edge reduced-cost fixing could settle, 0.5 for
        one it cannot touch at all, and graded in between.  The strictly
        fractional entries are then exactly the edges that survive fixing --
        a candidate set that is both meaningful and large enough for the
        scoring rules to disagree on.

        NOTE this is a per-edge uncertainty SCORE, not a point of the
        spanning-tree polytope: unlike the running average and the
        Dantzig-Wolfe master it is not a convex combination of trees, so its
        entries do not sum to n-1 and it must not be called a fractional
        solution.  Branching rules only ever need a per-variable uncertainty
        signal, which is what this is.

        Returns {edge: value}, or None when no gap or no tree is available.
        """
        if gap is None:
            ub = float(getattr(self, "incumbent_ub", float("inf")))
            lb = float(getattr(self, "best_lower_bound", float("-inf")))

            if math.isinf(ub) or math.isinf(lb) or math.isnan(ub) or math.isnan(lb):
                gap = None
            else:
                gap = ub - lb

        save_idx = getattr(self, "_last_mst_idx", None)
        save_list = getattr(self, "_last_mst_list", None)

        try:
            c = np.asarray(self.compute_modified_weights(), dtype=float)
            _cost, _len, _tree, idx = self._mst_core(c)

            if idx is None:
                return None

            mg = self.tree_edge_margins(c, idx)

            if not mg:
                return None

            M, pos = self._tree_path_max(idx, c)

            if M is None:
                return None

            d_in = c - M[pos[self._edge_u], pos[self._edge_v]]

            finite = [v for v in mg.values() if math.isfinite(v)]

            if gap is None or not (gap > 0.0) or math.isinf(gap):
                # No usable incumbent: fall back to the spread of the margins
                # themselves, which keeps the grading scale-free.
                gap = (max(finite) if finite else 0.0) or 1.0

            in_tree = np.zeros(len(self.edge_list), dtype=bool)
            in_tree[idx] = True

            el = self.edge_list
            fixed = self.fixed_edges
            out = {}

            for i in mg:
                e = el[i]

                # A forced edge is settled by the node, not by a margin: it is
                # in every tree here, so it is never a branching candidate.
                if e in fixed:
                    out[e] = 1.0
                    continue

                m = mg[i]
                u = 0.0 if math.isinf(m) else 1.0 - m / gap
                u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
                out[e] = 1.0 - 0.5 * u

            for i in self._cand_idx.tolist():
                if in_tree[i]:
                    continue

                d = float(d_in[i])

                if d < 0.0:
                    d = 0.0

                u = 1.0 - d / gap
                u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)

                if u > 0.0:
                    out[el[i]] = 0.5 * u

            return out or None
        except Exception:
            return None
        finally:
            self._last_mst_idx = save_idx
            self._last_mst_list = save_list

    def compute_fractional_solution(self, node=None):
        """The node's LR-derived branching indicator.

        `frac_source` picks which of the paper's two constructions is used:

            "dw"  (default)  Dantzig-Wolfe restricted master, Section 5.3
            "avg"            step-size-weighted running average, Section 5.2

        The two are kept strictly separate -- neither falls back to the other
        -- so a run labelled "avg" measures the running average and nothing
        else.  Every rule already handles a None indicator by falling back to
        its own candidate set.
        """
        src = str(getattr(self, "frac_source", "dw")).lower()
        _t0 = time()
        try:
            out = self._fractional_by_source(src, node)
        finally:
            LagrangianMST.indicator_time += time() - _t0
            LagrangianMST.indicator_calls += 1

        # Candidate-pool size: strictly fractional entries on free edges.
        # Fixed edges sit at one and excluded edges at zero in both
        # constructions, so this is the pool every indicator rule filters to.
        if not out:
            LagrangianMST.indicator_none += 1
            _pool = 0
        else:
            _fx = getattr(self, "fixed_edges", ()) or ()
            _ex = getattr(self, "excluded_edges", ()) or ()
            _pool = sum(1 for e, v in out.items()
                        if 1e-6 < v < 1.0 - 1e-6
                        and e not in _fx and e not in _ex)
        LagrangianMST.pool_hist[_pool] = LagrangianMST.pool_hist.get(_pool, 0) + 1

        # Optional cap on how many entries stay strictly fractional.
        #
        # The pool size is the knob that decides whether the branching rules
        # can differ at all.  The Dantzig-Wolfe point leaves a median of one
        # or two, so every rule picks from the same pair and they measure
        # almost identically; the graded margin leaves a median of eight and
        # the rules separate into the expected order, but a pool that wide
        # costs more than it returns.  `frac_top_k` keeps the k most
        # uncertain entries fractional and settles the rest, so the operating
        # point between those two ends can be chosen rather than inherited.
        k = int(getattr(self, "frac_top_k", 0))

        if k > 0 and out:
            tol = 1e-6
            frac = [(abs(v - 0.5), e) for e, v in out.items() if tol < v < 1.0 - tol]

            if len(frac) > k:
                frac.sort()
                for _d, e in frac[k:]:
                    out[e] = 1.0 if out[e] >= 0.5 else 0.0

        return out

    def _fractional_by_source(self, src, node=None):

        if src in ("avg", "average", "running_average", "weighted_average"):
            return self.compute_weighted_average_solution()

        if src == "lr":
            return self.lr_fractionality()

        if src in ("dw+lr", "union"):
            # The master's opinion where it has one, the graded margin
            # everywhere else.  Keeps the strictly fractional entries the
            # master identifies and widens the pool with the edges reduced
            # cost fixing could not settle.
            base = self.lr_fractionality() or {}
            dw = self.compute_dantzig_wolfe_solution(node)

            if dw:
                tol = 1e-6
                for e, v in dw.items():
                    if tol < v < 1.0 - tol:
                        base[e] = v

            return base or None

        return self.compute_dantzig_wolfe_solution(node)

    def recover_primal_solution(self, node):
        start_time = time()

        for mst_edges, is_feasible in self.primal_solutions:
            mst_edges_normalized = {tuple(sorted((u, v))) for u, v in mst_edges}
            if not all(e in mst_edges_normalized for e in node.fixed_edges):
                continue
            if any(e in mst_edges_normalized for e in node.excluded_edges):
                continue

            real_length = sum(self.edge_lengths[self.edge_indices[e]] 
                              for e in mst_edges_normalized)
            if real_length > self.budget:
                continue

            valid_cuts = True
            for cut, rhs in node.active_cuts:
                cut_count = sum(1 for e in mst_edges_normalized if e in cut)
                if cut_count > rhs:
                    valid_cuts = False
                    break
            if not valid_cuts:
                continue

            uf = UnionFind(self.num_nodes)
            for u, v in mst_edges_normalized:
                uf.union(u, v)
            if uf.count_components() != 1 or len(set(u for u, _ in mst_edges_normalized) | set(v for _, v in mst_edges_normalized)) < self.num_nodes:
                continue

            real_weight = sum(self.edge_weights[self.edge_indices[e]] 
                              for e in mst_edges_normalized)
            end_time = time()
            if self.verbose:
                print(f"Feasible primal solution found from primal_solutions: weight={real_weight:.2f}, length={real_length:.2f}")
            return list(mst_edges_normalized), real_weight, real_length

        uf = UnionFind(self.num_nodes)
        mst_edges = []
        total_length = 0.0
        total_weight = 0.0

        for edge_idx in self.fixed_edge_indices:
            u, v = self.edge_list[edge_idx]
            if uf.union(u, v):
                mst_edges.append((u, v))
                total_length += self.edge_lengths[edge_idx]
                total_weight += self.edge_weights[edge_idx]
            else:
                if self.verbose:
                    print(f"Fixed edge ({u}, {v}) creates cycle in greedy heuristic")
                return None, float('inf'), float('inf')

        edge_indices = [i for i in range(len(self.edges)) 
                        if i not in self.fixed_edge_indices and i not in self.excluded_edge_indices]
        sorted_edges = sorted(edge_indices, key=lambda i: self.edge_weights[i])

        for edge_idx in sorted_edges:
            u, v = self.edge_list[edge_idx]
            new_length = total_length + self.edge_lengths[edge_idx]
            if new_length > self.budget:
                continue

            temp_edges = mst_edges + [(u, v)]
            valid_cuts = True
            for cut, rhs in node.active_cuts:
                cut_count = sum(1 for e in temp_edges if e in cut)
                if cut_count > rhs:
                    valid_cuts = False
                    break
            if not valid_cuts:
                continue

            if uf.union(u, v):
                mst_edges.append((u, v))
                total_length = new_length
                total_weight += self.edge_weights[edge_idx]

        if uf.count_components() != 1 or len(set(u for u, _ in mst_edges) | set(v for _, v in mst_edges)) < self.num_nodes:
            if self.verbose:
                print("Greedy heuristic failed to produce a valid spanning tree")
            return None, float('inf'), float('inf')

        end_time = time()
        if self.verbose:
            print(f"Feasible primal solution found via greedy heuristic: weight={total_weight:.2f}, length={total_length:.2f}")
        return mst_edges, total_weight, total_length

    def compute_real_weight_length(self):
        """True (unpriced) weight and length of the node's current tree."""
        tree = self.last_mst_edges

        if not tree:
            return 0.0, 0.0

        # A tree that came out of _mst_core still has its index array, so the
        # common case is two numpy reductions instead of 2(n-1) dictionary
        # lookups.  This runs two or three times per node.
        idx = self._last_mst_idx
        if idx is not None and tree is getattr(self, "_last_mst_list", None):
            return float(self.edge_weights[idx].sum()), float(self.edge_lengths[idx].sum())

        ei = self.edge_indices
        jj = np.fromiter((ei[e] for e in tree), dtype=np.int64, count=len(tree))
        return float(self.edge_weights[jj].sum()), float(self.edge_lengths[jj].sum())

    def primal_repair_budget(self):
        """
        Strong budget-feasible primal heuristic via a parametric MST.

        The min-length tree (used by primal_repair) wastes the budget: it picks
        the shortest tree even when far more length is allowed, paying huge
        weight. Instead we compute the MST under a blended cost
            cost_mu(e) = weight(e) + mu * length(e)
        and binary-search mu >= 0 so the resulting tree's LENGTH lands just
        under the budget. Small mu -> min-weight tree (low weight, long); large
        mu -> min-length tree. The crossover tree spends the budget on length
        to buy low weight, which is exactly what the optimum does. Uses the
        fast argsort Kruskal, so it is cheap even at 500+ nodes.

        Returns (weight, length, edges) or (inf, inf, None) if infeasible.
        """
        ei = self.edge_indices
        W = self.edge_weights
        L = self.edge_lengths
        budget = self.budget

        def tree_at(mu):
            blended = W + mu * L
            # honor fixed/excluded via the existing argsort kruskal
            _, _, edges = self._argsort_kruskal(blended)
            if not edges:
                return None, float("inf"), float("inf")
            edges = [tuple(sorted(e)) for e in edges]
            w = float(sum(W[ei[e]] for e in edges))
            l = float(sum(L[ei[e]] for e in edges))
            return edges, w, l

        # Feasibility floor: even the min-length tree must fit, else infeasible.
        e_hi, w_hi, l_hi = tree_at(1e9)  # ~ min-length tree
        if e_hi is None or l_hi > budget:
            return float("inf"), float("inf"), None

        # If the min-weight tree already fits, it is optimal for this relaxation.
        e_lo, w_lo, l_lo = tree_at(0.0)
        if e_lo is not None and l_lo <= budget:
            return w_lo, l_lo, e_lo

        # The parametric MST length is NON-MONOTONIC in a way that defeats
        # binary search: feasible trees can appear in isolated mu-bands
        # separated by large breakpoints, so a bisection that pushes toward the
        # feasibility boundary often locks onto the wasteful min-length tree and
        # misses a far better feasible tree at moderate mu. Instead we SCAN a
        # geometric grid of mu values, keep the lowest-WEIGHT feasible tree, and
        # also remember the lowest-weight infeasible tree just over budget to
        # repair as a backup.
        # Find the feasibility BREAKPOINT by bisection on mu: the smallest mu
        # at which the parametric tree first fits the budget. The best low-
        # weight feasible incumbent lives right at this transition. This is much
        # cheaper than a dense grid (≈50 Kruskals via bisection vs grid*refine)
        # and finds an equal-or-better tree, because it targets the exact
        # breakpoint instead of sampling near it.
        best = (w_hi, l_hi, e_hi)              # feasible fallback (min-length)

        # `repair_tight` controls incumbent quality. Default True = the strong
        # near-optimal incumbent (few B&B nodes). Set False to return only a
        # VALID but looser feasible tree (the min-length tree), so B&B must
        # branch to close the gap -> larger, more informative node counts for
        # benchmarking. Either way the solve still reaches optimality; this only
        # changes how much of the work B&B does vs the primal heuristic.
        if not getattr(self, "repair_tight", True):
            return best[0], best[1], best[2]

        lo, hi = 0.0, 1e9                       # lo: over budget, hi: feasible
        for _ in range(getattr(self, "budget_repair_bisect_iters", 50)):
            mid = (lo + hi) / 2.0
            edges, w, l = tree_at(mid)
            if edges is None:
                lo = mid
                continue
            if l <= budget:
                if w < best[0]:
                    best = (w, l, edges)
                hi = mid
            else:
                lo = mid

        # The tree just BELOW the breakpoint (mu=lo) is over budget but has the
        # lowest weight near here; shorten it to budget for the best incumbent.
        e_over, w_over, l_over = tree_at(lo)
        if e_over is not None and l_over > budget:
            rep = self._shorten_to_budget(e_over)
            if rep is not None:
                rw, rl, redges = rep
                if rl <= budget and rw < best[0]:
                    best = (rw, rl, redges)

        # Backup: shorten the absolute min-weight tree (mu=0) too.
        if e_lo is not None and w_lo < best[0]:
            rep = self._shorten_to_budget(e_lo)
            if rep is not None:
                rw, rl, redges = rep
                if rl <= budget and rw < best[0]:
                    best = (rw, rl, redges)

        return best[0], best[1], best[2]

    def _shorten_to_budget(self, tree_edges):
        """
        Given a spanning tree slightly OVER budget on length, repeatedly swap
        its longest-length edges for shorter non-tree edges that reconnect the
        two components, choosing swaps that cut the most length per unit weight
        gained, until the tree's length <= budget. Returns (weight, length,
        edges) or None if it cannot be made feasible within the effort cap.

        This fixes the parametric-MST 'gap' case: the input tree has near-
        optimal (low) weight but is a few percent too long; a handful of swaps
        recover feasibility while keeping weight low.
        """
        ei = self.edge_indices
        W = self.edge_weights
        L = self.edge_lengths
        budget = self.budget

        tree = set(tuple(sorted(e)) for e in tree_edges)
        cur_len = float(sum(L[ei[e]] for e in tree))
        cur_w = float(sum(W[ei[e]] for e in tree))

        # Candidate replacement edges (non-tree, not excluded), shortest first.
        non_tree = [
            e for e in self.edge_list
            if e not in tree and ei[e] not in self.excluded_edge_indices
        ]
        non_tree.sort(key=lambda e: L[ei[e]])

        max_swaps = getattr(self, "shorten_max_swaps", 20 * self.num_nodes)
        swaps = 0

        # Build adjacency once; maintain it incrementally across swaps.
        adj = {}
        for (u, v) in tree:
            adj.setdefault(u, []).append((v, (u, v)))
            adj.setdefault(v, []).append((u, (u, v)))

        def path_edges(su, sv):
            # BFS path su->sv over current tree adjacency
            prev = {su: None}
            stack = [su]
            while stack:
                x = stack.pop()
                if x == sv:
                    break
                for (y, edge) in adj.get(x, []):
                    if y not in prev:
                        prev[y] = (x, edge)
                        stack.append(y)
            if sv not in prev:
                return []
            out = []
            node = sv
            while prev[node] is not None:
                px, edge = prev[node]
                out.append(tuple(sorted(edge)))
                node = px
            return out

        for add_e in non_tree:
            if cur_len <= budget or swaps >= max_swaps:
                break
            add_l = float(L[ei[add_e]])
            add_w = float(W[ei[add_e]])
            su, sv = add_e
            cyc = path_edges(su, sv)
            if not cyc:
                continue
            # Drop the LONGEST edge on the cycle that is longer than add_e,
            # to reduce length; among those pick the one giving best length cut.
            best_drop = None
            best_dl = 0.0
            for de in cyc:
                dl = float(L[ei[de]]) - add_l   # length reduction if we swap
                if dl > best_dl:
                    best_dl = dl
                    best_drop = de
            if best_drop is None or best_dl <= 0:
                continue
            # Apply swap: remove best_drop, add add_e
            tree.discard(best_drop); tree.add(add_e)
            cur_len += add_l - float(L[ei[best_drop]])
            cur_w += add_w - float(W[ei[best_drop]])
            # update adjacency
            du, dv = best_drop
            adj[du] = [(y, e) for (y, e) in adj.get(du, []) if tuple(sorted(e)) != best_drop]
            adj[dv] = [(y, e) for (y, e) in adj.get(dv, []) if tuple(sorted(e)) != best_drop]
            adj.setdefault(su, []).append((sv, add_e))
            adj.setdefault(sv, []).append((su, add_e))
            swaps += 1

        if cur_len <= budget:
            # Feasible now. Spend remaining slack to REDUCE weight: swap in
            # low-weight non-tree edges, dropping a heavier tree edge on the
            # induced cycle, as long as length stays within budget. This pulls
            # the incumbent down toward the optimum instead of stopping at the
            # first feasible tree.
            improve_cap = getattr(self, "shorten_improve_swaps", 0)
            cand = [
                e for e in self.edge_list
                if e not in tree and ei[e] not in self.excluded_edge_indices
            ]
            cand.sort(key=lambda e: W[ei[e]])  # cheapest weight first
            imp = 0
            for add_e in cand:
                if imp >= improve_cap:
                    break
                add_w = float(W[ei[add_e]]); add_l = float(L[ei[add_e]])
                su, sv = add_e
                cyc = path_edges(su, sv)
                if not cyc:
                    continue
                # drop the heaviest-weight cycle edge whose swap keeps budget
                best_drop = None; best_gain = 0.0
                for de in cyc:
                    new_len = cur_len - float(L[ei[de]]) + add_l
                    if new_len > budget:
                        continue
                    gain = float(W[ei[de]]) - add_w
                    if gain > best_gain:
                        best_gain = gain; best_drop = de
                if best_drop is None or best_gain <= 0:
                    continue
                tree.discard(best_drop); tree.add(add_e)
                cur_len += add_l - float(L[ei[best_drop]])
                cur_w += add_w - float(W[ei[best_drop]])
                du, dv = best_drop
                adj[du] = [(y, e) for (y, e) in adj.get(du, []) if tuple(sorted(e)) != best_drop]
                adj[dv] = [(y, e) for (y, e) in adj.get(dv, []) if tuple(sorted(e)) != best_drop]
                adj.setdefault(su, []).append((sv, add_e))
                adj.setdefault(sv, []).append((su, add_e))
                imp += 1
            return cur_w, cur_len, list(tree)
        return None

    def primal_repair(self):
        """
        Produce a budget-FEASIBLE spanning tree (an incumbent) regardless of
        whether the current Lagrangian MST is over budget.

        Strategy:
          1. Build the minimum-LENGTH spanning tree over the allowed edges
             (respecting fixed/excluded via custom_kruskal on edge_lengths).
             This is the shortest possible tree for this node; if its length
             still exceeds the budget, the node is genuinely infeasible.
          2. If feasible, try to lower its real weight with budget-preserving
             swaps: for each non-tree edge, if adding it and dropping the
             heaviest-weight edge on the induced cycle keeps length <= budget
             and reduces weight, do it. Cheap local improvement, optional.

        Returns (weight, length, edges) for a feasible tree, or
        (inf, inf, None) if no feasible tree exists at this node.
        """
        # Step 1: minimum-length tree using the existing Kruskal machinery.
        _, min_len, len_tree = self.custom_kruskal(self.edge_lengths)
        if not len_tree or min_len == float("inf") or min_len > self.budget:
            return float("inf"), float("inf"), None

        # Strong budget-aware path (opt-in, negative-correlation only): the
        # min-length tree wastes budget and pays huge weight; the parametric
        # MST spends the budget to buy low weight. Falls back to the min-length
        # tree below if it somehow fails.
        if getattr(self, "use_budget_repair", False):
            bw, bl, be = self.primal_repair_budget()
            if be is not None and bl <= self.budget:
                return bw, bl, be

        tree = [tuple(sorted(e)) for e in len_tree]
        tree_set = set(tree)
        ei = self.edge_indices
        W = self.edge_weights
        L = self.edge_lengths

        def tree_weight(edges):
            return float(sum(W[ei[e]] for e in edges))

        def tree_length(edges):
            return float(sum(L[ei[e]] for e in edges))

        cur_len = tree_length(tree)
        cur_w = tree_weight(tree)

        # Step 2: budget-preserving weight-reducing swaps (bounded effort).
        # The min-length tree is ALREADY a valid feasible incumbent, so the
        # swaps are pure optional improvement. They are O(candidates * n) with a
        # Python BFS per candidate, which is far too slow on large/dense graphs,
        # so we skip improvement entirely beyond a size threshold. B&B still
        # gets a finite UB from the min-length tree itself.
        repair_improve_cap = getattr(self, "repair_improve_max_nodes", 120)
        if self.num_nodes > repair_improve_cap:
            return cur_w, cur_len, tree

        # Non-tree candidate edges, cheapest weight first.
        non_tree = [
            e for e in self.edge_list
            if e not in tree_set
            and ei[e] not in self.excluded_edge_indices
        ]
        non_tree.sort(key=lambda e: W[ei[e]])

        max_swaps = min(len(non_tree), 2 * self.num_nodes)
        swaps_done = 0

        for add_e in non_tree:
            if swaps_done >= max_swaps:
                break
            # Find the cycle created by adding add_e: path between its endpoints
            # in the current tree.
            adj = {}
            for (u, v) in tree:
                adj.setdefault(u, []).append((v, (u, v)))
                adj.setdefault(v, []).append((u, (u, v)))
            su, sv = add_e
            # BFS for the path su -> sv
            prev = {su: None}
            stack = [su]
            found = False
            while stack:
                x = stack.pop()
                if x == sv:
                    found = True
                    break
                for (y, edge) in adj.get(x, []):
                    if y not in prev:
                        prev[y] = (x, edge)
                        stack.append(y)
            if not found:
                continue
            # Reconstruct cycle edges.
            cycle_edges = []
            node = sv
            while prev[node] is not None:
                px, edge = prev[node]
                cycle_edges.append(tuple(sorted(edge)))
                node = px
            if not cycle_edges:
                continue
            # Candidate to drop: the heaviest-weight tree edge on the cycle
            # whose removal keeps us within budget after adding add_e.
            add_w = float(W[ei[add_e]])
            add_l = float(L[ei[add_e]])
            best_drop = None
            best_gain = 0.0
            for drop_e in cycle_edges:
                new_len = cur_len - float(L[ei[drop_e]]) + add_l
                if new_len > self.budget:
                    continue
                gain = float(W[ei[drop_e]]) - add_w  # weight reduction
                if gain > best_gain:
                    best_gain = gain
                    best_drop = drop_e
            if best_drop is not None and best_gain > 0:
                tree_set.discard(best_drop)
                tree_set.add(add_e)
                tree = list(tree_set)
                cur_len = cur_len - float(L[ei[best_drop]]) + add_l
                cur_w = cur_w - float(W[ei[best_drop]]) + add_w
                swaps_done += 1

        return cur_w, cur_len, list(tree_set)