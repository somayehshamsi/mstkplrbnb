

import heapq
import networkx as nx
import math
import csv
from datetime import datetime
from typing import Optional, List, Tuple
import os
import time
import gc
import psutil
import logging
logger = logging.getLogger(__name__)


import abc
# from abc import abstractmethod
import random


    
class BranchingRule(abc.ABC):
    """
    An abstract branching rule.
    """
    def __init__(self):
        """
        The branching rule's initializer
        """
        pass

    @abc.abstractmethod
    def get_branching_variable(self):
        """
        Selects and returns an object to branch on.
        """
        pass

class RandomBranchingRule(BranchingRule):
    """
    A branching rule that picks a variable to branch on randomly.
    """
    def __init__(self):
        """
        The branching rule's initializer
        """
        pass

    def get_branching_variable(self, candidates: list):
        """
        Selects and returns an object to branch on randomly from a list of candidates.
        """
        assert candidates, "Cannot randomly draw from an empty list of candidates."
        return random.choice(candidates)



class Node(abc.ABC):
    """
    A node of the branch and bound.
    """
    __slots__ = ['local_lower_bound']
    def __init__(self, local_lower_bound: float):
        """
        Initialises a node of the branch and bound with a lower bound.
        """
        self.local_lower_bound = local_lower_bound

    @abc.abstractmethod
    def create_children(self, branched_object: object):
        """
        Creates and returns two children nodes that partition
        the space of solutions according to the given branching object.
        """
        pass

    @abc.abstractmethod
    def is_feasible(self):
        """
        Checks if the node represents a feasible solution.
        """
        pass

    @abc.abstractmethod
    def compute_upper_bound(self):
        """
        Computes an upper bound for the node.
        """
        pass



class _Incumbent:
    """A solution detached from the node that produced it.

    Nodes are released as soon as they are processed, so the search cannot
    hold on to one just to report its tree; this carries the edges and
    nothing else.
    """
    __slots__ = ("best_feasible_edges", "mst_edges", "local_lower_bound",
                 "fixed_edges", "excluded_edges")

    def __init__(self, edges):
        self.best_feasible_edges = list(edges)
        self.mst_edges = self.best_feasible_edges
        self.local_lower_bound = float("-inf")
        self.fixed_edges = ()
        self.excluded_edges = ()


def _release_solver(node):
    ls = getattr(node, "lagrangian_solver", None)
    if ls is None:
        return
    for attr in ("graph", "_mw_cached", "_free_mask_cache", "mst_cache"):
        if hasattr(ls, attr):
            setattr(ls, attr, None)
    node.lagrangian_solver = None

class LazyPriorityQueue:

    """A priority queue with lazy deletion using heapq."""

    def __init__(self, verbose=False):
        self.heap = []  # List of [priority, counter, node]
        self.counter = 0
        self.deleted = set()  # Track deleted nodes
        self.verbose = verbose

    def push(self, priority, node):
        """Push an item with given priority."""
        if node not in self.deleted:
            heapq.heappush(self.heap, [priority, self.counter, node])
            self.counter += 1

    def batch_push(self, items):
        """Push multiple items at once."""
        for priority, node in items:
            self.push(priority, node)

    def mark_deleted(self, node):
        """Mark a node as deleted without removing it immediately."""
        self.deleted.add(node)

    def pop(self):
        """Pop the lowest-priority node safely, skipping deleted nodes."""
        # if len(self.deleted) > len(self.heap) // 2:
        #     self.cleanup()  # Rebuild only when deleted dominate
        while self.heap:
            priority, count, node = heapq.heappop(self.heap)
            if node in self.deleted:
                self.deleted.remove(node)
                continue
            return priority, node
        raise IndexError("pop from an empty LazyPriorityQueue")

    def __len__(self):
        return max(0, len(self.heap) - len(self.deleted))
    def cleanup(self):
        """Periodically clean up deleted items to save memory."""
        if len(self.deleted) > len(self.heap) // 2:
            self.heap = [entry for entry in self.heap if entry[2] not in self.deleted]
            heapq.heapify(self.heap)
            self.deleted.clear()

class BranchAndBound:
    """
    An implementation of the Branch-and-Bound
    """
   
    __slots__ = ['branching_rule', 'verbose', 'duality_gap_threshold', 'objective_granularity', 'stagnation_limit', 'best_upper_bound',
             'best_lower_bound', 'min_lower_bound', 'max_lower_bound', 'sum_lower_bounds', 'count_lower_bounds',
             'sampled_lower_bounds', 'pruned_lower_bounds', 'best_solution', 'priority_queue',
             'config', 'instance_seed', 'node_log_file', 'node_log_handle', 'node_log_writer',
             'node_log_rows', 'total_nodes_solved', 'nodes_pruned_lower_bound',
             'nodes_pruned_feasible', 'nodes_pruned_invalid_mst', 'nodes_pruned_budget', 'nodes_pruned_gap',
             'timed_out', 'final_duality_gap']
    
    def __init__(self, branching_rule: BranchingRule, verbose=False, duality_gap_threshold=0.00, 
                 stagnation_limit=1000, config: dict = None, instance_seed: int = None):
        """
        Initializes the B&B with a branching rule, verbose flag, duality gap threshold, stagnation limit,
        configuration details, and instance seed.

        Args:
            branching_rule: The branching rule to use.
            verbose: Whether to print verbose output.
            duality_gap_threshold: Threshold for duality gap to stop branching.
            stagnation_limit: Limit for nodes without improvement before reducing max_nodes.
            config: Dictionary containing configuration details (e.g., branching_rule, use_2opt, use_shooting, use_bisection).
            instance_seed: Seed used to generate the instance.
        """
        self.branching_rule = branching_rule
        self.verbose = verbose
        self.duality_gap_threshold = duality_gap_threshold
        # Every edge weight is an integer, so every feasible tree has an
        # integral objective and a node can be dropped as soon as its bound
        # rules out an improvement of one full unit.  The plain LB >= UB test
        # explores the whole band LB in (UB-1, UB], which near the end of the
        # search is most of what is left.  Set it to 0 for fractional weights.
        self.objective_granularity = float(
            (config or {}).get("objective_granularity", 1.0)
        )
        self.stagnation_limit = stagnation_limit
        self.best_upper_bound = float("inf")
        self.best_lower_bound = float("-inf")
        # self.all_lower_bounds = []
        # self.pruned_lower_bounds = []
        self.min_lower_bound = float("inf")
        self.max_lower_bound = float("-inf")
        self.sum_lower_bounds = 0.0
        self.count_lower_bounds = 0
        self.sampled_lower_bounds = []  # keep small sample
        self.pruned_lower_bounds = []
        self.best_solution = None
        self.priority_queue = LazyPriorityQueue()

        # Configuration and instance details
        self.config = config or {}
        self.instance_seed = instance_seed

        # Per-node CSV trace.  It is OFF unless the config asks for it:
        # every row carried the node's whole fixed and excluded edge lists, so
        # at n = 400 a deep search wrote gigabytes and paid an open()/close()
        # per node for it.  Set config["log_nodes"] = True (and optionally
        # config["node_log_dir"]) to get the old trace back.
        self.node_log_handle = None
        self.node_log_writer = None
        self.node_log_rows = 0
        self.node_log_file = None

        if self.config.get("log_nodes", False):
            config_str = (f"{self.config.get('branching_rule', 'unknown')}_")
            seed_str = f"seed_{self.instance_seed}" if self.instance_seed is not None else "noseed"
            nf_path = self.config.get(
                "node_log_dir", os.path.join(os.path.expanduser("~/Desktop"), "nf")
            )
            os.makedirs(nf_path, exist_ok=True)
            self.node_log_file = os.path.join(nf_path, f"{config_str}_{seed_str}.csv")
            self._init_node_logger()

        # Statistics
        self.total_nodes_solved = 0
        self.nodes_pruned_lower_bound = 0
        self.nodes_pruned_feasible = 0
        self.nodes_pruned_invalid_mst = 0
        self.nodes_pruned_budget = 0
        self.nodes_pruned_gap = 0
        self.timed_out = False  # Track timeout status
        self.final_duality_gap = float("inf")  # Store final duality gap

    def _init_node_logger(self):
        headers = ["NodeID", "LowerBound", "UpperBound", "DualityGap", "BranchedVariable", 
                   "FixedEdges", "ExcludedEdges", "Reason", "Timestamp", "Config", "InstanceSeed"]
        # One handle for the whole search instead of an open/close per node.
        self.node_log_handle = open(self.node_log_file, 'w', newline='')
        self.node_log_writer = csv.writer(self.node_log_handle)
        self.node_log_writer.writerow(headers)

    def _close_node_logger(self):
        if self.node_log_handle is not None:
            try:
                self.node_log_handle.close()
            except Exception:
                pass
            self.node_log_handle = None
            self.node_log_writer = None

    def _peek_min_live_lb(self):
        """Smallest lower bound among live nodes, without mutating the heap.

        heap[0] is the minimum only while it is undeleted; once lazily deleted
        entries are present the array order is not sorted, so returning the
        first undeleted entry can overstate the bound.  Scan them all.
        """
        deleted = getattr(self.priority_queue, "deleted", set())
        best = float("inf")
        for priority, count, node in getattr(self.priority_queue, "heap", []):
            if node not in deleted:
                p = float(priority)
                if p < best:
                    best = p
        return best
    def _log_node(self, node_id, node, branched_variable, reason, effective_upper_bound):
        writer = self.node_log_writer
        if writer is None:
            return

        duality_gap = effective_upper_bound - node.local_lower_bound if effective_upper_bound < float("inf") else float("inf")
        fixed_edges = str(list(node.fixed_edges)) if hasattr(node, 'fixed_edges') else "[]"
        excluded_edges = str(list(node.excluded_edges)) if hasattr(node, 'excluded_edges') else "[]"
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        config_str = (f"branching={self.config.get('branching_rule', 'unknown')},"
                      f"2opt={self.config.get('use_2opt', False)},"
                      f"shooting={self.config.get('use_shooting', False)},"
                      f"bisection={self.config.get('use_bisection', False)}")
        writer.writerow([node_id, node.local_lower_bound, effective_upper_bound, duality_gap, 
                        str(branched_variable), fixed_edges, excluded_edges, reason, timestamp,
                        config_str, self.instance_seed])

        self.node_log_rows += 1
        if self.node_log_rows % 512 == 0:
            self.node_log_handle.flush()




    def compute_initial_upper_bound(self, root: 'Node'):
        """
        Computes an initial upper bound using a greedy MST heuristic.
        """
        from mstkpbranchandbound import MSTNode

        G = nx.Graph()
        G.add_weighted_edges_from([(u, v, w) for u, v, w, l in root.edges])
        mst = nx.minimum_spanning_tree(G)
        total_length = sum(root.lagrangian_solver.edge_attributes[(min(u, v), max(u, v))][1] for u, v in mst.edges)
        if total_length <= root.budget:
            total_weight = sum(root.lagrangian_solver.edge_attributes[(min(u, v), max(u, v))][0] for u, v in mst.edges)
            self.best_upper_bound = total_weight
            self.best_solution = root

            # networkx yields edges in adjacency order, not sorted, so these
            # have to be normalised: everything else in the solver keys edges
            # by (min, max), and edge_attributes[(7, 2)] is a KeyError.
            #
            # And only `best_feasible_edges` is set.  Overwriting
            # `root.mst_edges` replaced the root's LAGRANGIAN tree with this
            # greedy min-weight one, which is a different tree: is_feasible()
            # would then test the budget on one and the connectivity on the
            # other, and get_fractional_value would read lengths off a tree the
            # node never priced (raising KeyError on the unnormalised tuples
            # into the bargain).
            root.best_feasible_edges = [
                (min(u, v), max(u, v)) for u, v in mst.edges
            ]
            # The nodes read their incumbent from MSTNode.global_upper_bound
            # -- it feeds the Polyak gap and every reduced-cost fixing test --
            # so a better tree found here has to reach them too.
            if total_weight < MSTNode.global_upper_bound:
                MSTNode.global_upper_bound = float(total_weight)
                MSTNode.global_best_edges = list(root.best_feasible_edges)
            if self.verbose:
                print(f"Initial feasible MST found with weight: {total_weight}, length: {total_length}")

        # MEMORY: cleanup local heavy objects
        try:
            del mst
            del G
        except:
            pass
        gc.collect()


    # Rules whose scores are built from pseudocost estimates.  Every one of
    # them should learn from the branchings the search actually performs, not
    # only from strong-branching probes: the children are solved anyway, so
    # the observed bound change is free and is measured over the node's full
    # iteration budget rather than a two-iteration probe.
    _PSEUDOCOST_RULES = ("pseudocost", "reliability", "hybrid_strong_fractional")

    def _update_pseudocosts(self, node, branched_edge, children):
        """Record the observed bound change of a real branching.

        Per-unit rates, as in Benichou et al.: the improvement is divided by
        the branching distance and the score multiplies it back.  The node's
        own branching indicator supplies that distance, so the update and the
        rules' scores are normalised by the same signal.

        This used to sit inline on ONE of the two branching paths in solve()
        -- the one taken when the node's Lagrangian tree happens to respect
        the budget.  Nodes are branched precisely because their tree does not,
        so on the harder instances most branchings went unrecorded.  It also
        dereferenced every child without checking for None, and a child pruned
        by the cut projection is None, which made pseudocost branching crash
        outright whenever cover cuts were enabled.
        """
        rule = getattr(node, "branching_rule", None)
        if rule not in self._PSEUDOCOST_RULES:
            return

        get_ind = getattr(node, "_branching_indicator", None)
        if get_ind is None:
            return

        try:
            f = get_ind(branched_edge)
        except Exception:
            return

        min_dist = getattr(type(node), "PC_MIN_DIST", 1e-2)
        dist_up = max(1.0 - f, min_dist)
        dist_dn = max(f, min_dist)

        for child in children:
            if child is None:
                continue

            delta = max(0.0, child.local_lower_bound - node.local_lower_bound)
            if math.isnan(delta) or math.isinf(delta):
                continue

            up = len(child.fixed_edges) > len(node.fixed_edges)
            pc = delta / (dist_up if up else dist_dn)

            table = node.pseudocosts_up if up else node.pseudocosts_down
            counts = node.counts_up if up else node.counts_down
            count = counts[branched_edge]
            table[branched_edge] = (table[branched_edge] * count + pc) / (count + 1)
            counts[branched_edge] = count + 1

            noter = getattr(type(node), "note_pseudocost", None)
            if noter is not None:
                noter(up, pc)

    def batch_insert_nodes(self, nodes):
        """Batch insert nodes into the priority queue, filtering out those with high lower bounds."""
        items = []
        for node in nodes:
            if node is None:  # FIXED: Handle None children
                continue
            if math.isnan(node.local_lower_bound) or math.isinf(node.local_lower_bound):
                if self.verbose:
                    print(f"Skipping node with invalid lower bound: {node.local_lower_bound}")
                continue
            if node.local_lower_bound <= self.best_upper_bound:
                items.append((node.local_lower_bound, node))
        self.priority_queue.batch_push(items)    




    def solve(self, root: Node, time_limit_s: float = 1800.0):
        from mstkpbranchandbound import MSTNode

        self.compute_initial_upper_bound(root)

        self.priority_queue.push(root.local_lower_bound, root)
        node_counter = 0
        branched_variable = None
        start_time = time.time()

        try:
            process = psutil.Process()
        except Exception:
            process = None

        while len(self.priority_queue) > 0:
            if process and node_counter % 1000 == 0 and node_counter > 0:
                mem_mb = process.memory_info().rss / 1024 / 1024
                if mem_mb > 4000 and self.verbose:
                    print(f"WARNING: Memory at {mem_mb:.0f}MB at node {node_counter}")

            if (
                self.priority_queue.deleted
                and len(self.priority_queue.deleted) > max(64, len(self.priority_queue.heap) // 3)
            ):
                self.priority_queue.cleanup()
            elif node_counter % 100 == 0 and node_counter > 0:
                self.priority_queue.cleanup()

            if time.time() - start_time > time_limit_s:
                self.timed_out = True
                if self.verbose:
                    print("Stopped: Exceeded 3600-second time limit")
                break

            try:
                min_lower_bound, node = self.priority_queue.pop()
            except IndexError:
                break

            try:
                self.total_nodes_solved += 1
                node_counter += 1
                branched_variable = None
                if process and node_counter % 200 == 0:
                    mem_mb = process.memory_info().rss / 1024 / 1024
                    logger.info(f"RSS at node {node_counter}: {mem_mb:.0f} MB")

                if math.isnan(node.local_lower_bound) or math.isinf(node.local_lower_bound):
                    self._log_node(node_counter, node, branched_variable, "Invalid lower bound", self.best_upper_bound)
                    if self.verbose:
                        print(f"Warning: Invalid lower bound {node.local_lower_bound} for node {node_counter}, skipping")
                    continue

                lb = node.local_lower_bound
                self.min_lower_bound = min(self.min_lower_bound, lb)
                self.max_lower_bound = max(self.max_lower_bound, lb)
                self.sum_lower_bounds += lb
                self.count_lower_bounds += 1
                if self.total_nodes_solved % 100 == 0:
                    self.sampled_lower_bounds.append(lb)

                self.best_lower_bound = max(self.best_lower_bound, lb)

                duality_gap = self.best_upper_bound - node.local_lower_bound if self.best_upper_bound < float("inf") else float("inf")
                # threshold = min(self.duality_gap_threshold * self.best_upper_bound, 0.98) if self.best_upper_bound < float("inf") else float("inf")
                threshold = self.duality_gap_threshold * self.best_upper_bound if self.best_upper_bound < float("inf") else float("inf")


                upper_bound = node.best_upper_bound
                if upper_bound < self.best_upper_bound:
                    self.best_upper_bound = upper_bound
                    self.best_solution = node

                # Take the incumbent from wherever it was found, not only from
                # the nodes this loop pops.  A node whose own bound already
                # reached the incumbent, or a child filtered out before it was
                # queued, may still have produced the best feasible tree seen
                # so far, and every prune below -- and every reduced-cost
                # fixing test -- is measured against this number.  Leaving it
                # stale kept whole subtrees alive that the search had already
                # beaten: at n = 400 it was the difference between 1084 nodes
                # and 285 on the tree-completion rung.
                _g = getattr(MSTNode, "global_upper_bound", float("inf"))
                if _g < self.best_upper_bound:
                    self.best_upper_bound = _g
                    _ge = getattr(MSTNode, "global_best_edges", None)
                    if _ge:
                        self.best_solution = _Incumbent(_ge)

                # if (
                #     node_counter == 1
                #     and self.best_upper_bound < float("inf")
                #     and hasattr(node, "apply_peg_test")
                # ):
                #     n_fixed = node.apply_peg_test(
                #         incumbent_ub=self.best_upper_bound,
                #         max_tests=10,
                #         candidate_source="mst"
                #     )
                              
                if self.verbose:
                    print(f"\n--- Node {node_counter} ---")
                    print(f"Lower bound: {node.local_lower_bound}")
                    print(f"Upper bound: {self.best_upper_bound}")
                    print(f"Node duality gap: {duality_gap:.2f}")

                # 1e-6, not 1e-9: the bound is a sum of n-1 float64 terms, so
                # it carries rounding of that order, and the test must not let
                # a node be dropped on rounding alone.
                if node.local_lower_bound > self.best_upper_bound - self.objective_granularity + 1e-6:
                    self.nodes_pruned_lower_bound += 1
                    self._log_node(node_counter, node, branched_variable, "Lower bound >= best upper bound", self.best_upper_bound)
                    continue

                if duality_gap < threshold:
                    self.nodes_pruned_gap += 1
                    self.pruned_lower_bounds.append(node.local_lower_bound)
                    self._log_node(node_counter, node, branched_variable, "Duality gap too small", self.best_upper_bound)
                    continue

                # With no duality-gap tolerance the threshold is 0 and the
                # very first live node fails the test, so this whole scan can
                # only ever conclude "do not stop" -- at the cost of a
                # heappop/heappush per node.
                if self.duality_gap_threshold > 0.0 and self.best_upper_bound < float("inf"):
                    threshold = min(self.duality_gap_threshold * self.best_upper_bound, 1.0)
                    stop = True
                    temp_heap = []
                    while self.priority_queue.heap:
                        priority, count, item = heapq.heappop(self.priority_queue.heap)
                        if item not in self.priority_queue.deleted:
                            temp_heap.append((priority, count, item))
                            if self.best_upper_bound - priority > threshold:
                                stop = False
                                break
                    for priority, count, item in temp_heap:
                        heapq.heappush(self.priority_queue.heap, [priority, count, item])
                    if stop and self.pruned_lower_bounds:
                        min_pruned_lower_bound = min(self.pruned_lower_bounds)
                        final_duality_gap = self.best_upper_bound - min_pruned_lower_bound if self.best_upper_bound < float("inf") else float("inf")
                        if self.verbose:
                            print(f"Stopping early: No remaining nodes with duality gap >= {threshold:.2f}")
                            print(f"Final duality gap: {final_duality_gap:.2f}")
                        break

                is_feasible, reason = node.is_feasible()

                if is_feasible:
                    if node.local_lower_bound < self.best_upper_bound:
                        candidates = node.get_branching_candidates()
                        if not candidates:
                            self._log_node(node_counter, node, branched_variable, "No branching candidates", self.best_upper_bound)
                            if self.verbose:
                                print("Decision4: Prune (no candidates for branching)")
                            continue

                        # get_branching_candidates signals a forced
                        # single-child decision as (edges, child).  The
                        # child is None when a cover cut proved that side
                        # infeasible too, and `None` is not an MSTNode --
                        # so the old test let the TUPLE fall through as if
                        # it were the candidate edge list, and branching
                        # then picked `[rep]` or `None` out of it and blew
                        # up in create_children with
                        # "not enough values to unpack".  Both sides being
                        # infeasible is a proof that the node is empty, so
                        # the shape has to be recognised and the node
                        # pruned, not mistaken for an edge list.
                        if (
                            isinstance(candidates, tuple)
                            and len(candidates) == 2
                            and (candidates[1] is None
                                 or isinstance(candidates[1], MSTNode))
                        ):
                            candidate_edges, single_child = candidates
                            branched_variable = candidate_edges[0] if isinstance(candidate_edges, list) else candidate_edges
                            if single_child is not None:
                                if single_child.local_lower_bound <= self.best_upper_bound:
                                    if not hasattr(single_child, "is_child_likely_feasible") or single_child.is_child_likely_feasible():
                                        self.batch_insert_nodes([single_child])
                                        self._log_node(node_counter, node, branched_variable, "Single child from strong branching", self.best_upper_bound)
                                        if self.verbose:
                                            print(f"Decision: Insert single child from strong branching (LB={single_child.local_lower_bound})")
                                    else:
                                        self.nodes_pruned_budget += 1
                                        self._log_node(node_counter, single_child, branched_variable, "Child budget violation", self.best_upper_bound)
                                        if self.verbose:
                                            print("Decision: Prune child (budget violation likely)")
                                        _release_solver(single_child)
                                else:
                                    self.nodes_pruned_lower_bound += 1
                                    self._log_node(node_counter, single_child, branched_variable, "Single child pruned (LB > UB)", self.best_upper_bound)
                                    if self.verbose:
                                        print(f"Decision: Prune single child (LB {single_child.local_lower_bound} > UB {self.best_upper_bound})")
                                    _release_solver(single_child)
                            continue
                        else:
                            candidate_edges = candidates
                            if not candidate_edges:
                                self._log_node(node_counter, node, branched_variable, "No branching candidates", self.best_upper_bound)
                                if self.verbose:
                                    print("Decision3: Prune (no candidates for branching)")
                                continue
                            branched_variable = (
                                candidate_edges[0]
                                if len(candidate_edges) == 1
                                else self.branching_rule.get_branching_variable(candidate_edges)
                            )

                        children = node.create_children(branched_variable)
                        self._update_pseudocosts(node, branched_variable, children)

                        filtered_children = []
                        for child in children:
                            if child is None:
                                continue
                            if hasattr(child, "is_child_likely_feasible") and not child.is_child_likely_feasible():
                                self.nodes_pruned_budget += 1
                                self._log_node(node_counter, child, branched_variable, "Child budget violation", self.best_upper_bound)
                                if self.verbose:
                                    print("Decision: Prune child (budget violation likely)")
                                _release_solver(child)
                            else:
                                filtered_children.append(child)

                        if filtered_children:
                            self.batch_insert_nodes(filtered_children)

                        self._log_node(node_counter, node, branched_variable, "Branching", self.best_upper_bound)
                        if self.verbose:
                            print("Decision: Branch (children added to queue)")
                    else:
                        self.nodes_pruned_feasible += 1
                        self._log_node(node_counter, node, branched_variable, "Feasible solution", self.best_upper_bound)
                    continue
                else:
                    if reason == "MST length exceeds budget":
                        candidates = node.get_branching_candidates()
                        if not candidates:
                            self._log_node(node_counter, node, branched_variable, "No branching candidates", self.best_upper_bound)
                            if self.verbose:
                                print("Decision2: Prune (no candidates for branching)")
                            continue

                        # get_branching_candidates signals a forced
                        # single-child decision as (edges, child).  The
                        # child is None when a cover cut proved that side
                        # infeasible too, and `None` is not an MSTNode --
                        # so the old test let the TUPLE fall through as if
                        # it were the candidate edge list, and branching
                        # then picked `[rep]` or `None` out of it and blew
                        # up in create_children with
                        # "not enough values to unpack".  Both sides being
                        # infeasible is a proof that the node is empty, so
                        # the shape has to be recognised and the node
                        # pruned, not mistaken for an edge list.
                        if (
                            isinstance(candidates, tuple)
                            and len(candidates) == 2
                            and (candidates[1] is None
                                 or isinstance(candidates[1], MSTNode))
                        ):
                            candidate_edges, single_child = candidates
                            branched_variable = candidate_edges[0] if isinstance(candidate_edges, list) else candidate_edges
                            if single_child is not None:
                                if single_child.local_lower_bound <= self.best_upper_bound:
                                    if not hasattr(single_child, "is_child_likely_feasible") or single_child.is_child_likely_feasible():
                                        self.batch_insert_nodes([single_child])
                                        self._log_node(node_counter, node, branched_variable, "Single child from strong branching", self.best_upper_bound)
                                    else:
                                        self.nodes_pruned_budget += 1
                                        self._log_node(node_counter, single_child, branched_variable, "Child budget violation", self.best_upper_bound)
                                        _release_solver(single_child)
                                else:
                                    self.nodes_pruned_lower_bound += 1
                                    self._log_node(node_counter, single_child, branched_variable, "Single child pruned (LB > UB)", self.best_upper_bound)
                                    _release_solver(single_child)
                            continue
                        else:
                            candidate_edges = candidates
                            if not candidate_edges:
                                self._log_node(node_counter, node, branched_variable, "No branching candidates", self.best_upper_bound)
                                if self.verbose:
                                    print("Decision1: Prune (no candidates for branching)")
                                continue
                            branched_variable = (
                                candidate_edges[0]
                                if len(candidate_edges) == 1
                                else self.branching_rule.get_branching_variable(candidate_edges)
                            )

                        children = node.create_children(branched_variable)
                        self._update_pseudocosts(node, branched_variable, children)

                        filtered_children = []
                        for child in children:
                            if child is None:
                                continue
                            if hasattr(child, "is_child_likely_feasible") and not child.is_child_likely_feasible():
                                self.nodes_pruned_budget += 1
                                self._log_node(node_counter, child, branched_variable, "Child budget violation", self.best_upper_bound)
                                _release_solver(child)
                            else:
                                filtered_children.append(child)
                        if filtered_children:
                            self.batch_insert_nodes(filtered_children)

                        self._log_node(node_counter, node, branched_variable, reason, self.best_upper_bound)
                        if self.verbose:
                            print(f"Decision: Branch on infeasible MST ({reason})")
                    else:
                        self.nodes_pruned_invalid_mst += 1
                        self._log_node(node_counter, node, branched_variable, reason, self.best_upper_bound)
                        if self.verbose:
                            print(f"Decision: Prune ({reason})")
            finally:
                _release_solver(node)


        # (The final duality gap is computed once, further below.  An earlier
        # block here assigned it from the pruned bounds or from an AVERAGE of
        # node bounds -- an average is not a valid bound at all -- and was in any
        # case overwritten before being read.)

        if self.timed_out:
            self._log_node(node_counter, root, branched_variable, "Timeout", self.best_upper_bound)
            if self.verbose:
                print(f"Final duality gap on timeout: {self.final_duality_gap:.2f}")

        print("\n--- Statistics ---")
        print(f"Total nodes solved: {self.total_nodes_solved}")
        print(f"Nodes pruned due to lower bound: {self.nodes_pruned_lower_bound}")
        print(f"Nodes pruned due to feasible solution: {self.nodes_pruned_feasible}")
        print(f"Nodes pruned due to invalid MST: {self.nodes_pruned_invalid_mst}")
        print(f"Nodes pruned due to budget violation: {self.nodes_pruned_budget}")
        print(f"Nodes pruned due to duality gap: {self.nodes_pruned_gap}")
        print(f"Optimal MST Cost within Budget: {self.best_upper_bound}")

        if process:
            try:
                final_memory = process.memory_info().rss / 1024 / 1024
                print(f"Final memory usage: {final_memory:.1f} MB")
            except Exception:
                pass

        if self.timed_out:
            print("Process stopped due to timeout (3600 seconds)")

        # Final duality gap.
        #
        # A node still limits the bound only if it was never resolved: either it
        # is still queued, or it was cut by the duality-gap tolerance, which
        # proves nothing about what it contains.  Nodes pruned because their own
        # lower bound already reached the incumbent ARE resolved and must not
        # enter the bound.
        #
        # If nothing is left open, the search exhausted the tree and the
        # incumbent is proven optimal, so the gap is exactly zero.  The previous
        # version fell back to `min_lower_bound` -- the smallest bound seen at
        # any processed node, essentially the root -- and so reported a large
        # gap for instances it had in fact solved to optimality.
        open_lbs = [
            b for b in self.pruned_lower_bounds
            if not math.isnan(b) and not math.isinf(b)
        ]

        live_lb = self._peek_min_live_lb()
        if not math.isinf(live_lb):
            open_lbs.append(live_lb)

        if self.best_upper_bound >= float("inf"):
            self.final_duality_gap = float("inf")
        elif not open_lbs:
            self.final_duality_gap = 0.0
        else:
            self.final_duality_gap = max(0.0, self.best_upper_bound - min(open_lbs))

        # Optional: log a final "Timeout" row to your node CSV
        if getattr(self, "timed_out", False):
            try:
                self._log_node(node_id=node_counter, node=root,
                            branched_variable=None, reason="Timeout",
                            effective_upper_bound=self.best_upper_bound)
            except Exception:
                pass

        self._close_node_logger()


        return self.best_solution, self.best_upper_bound



    def solve_with_shooting(self, root: Node, initial_lower_bound: float, initial_upper_bound: float, initial_solution: Optional[List[Tuple[int, int]]]):
        """
        Solves the Branch-and-Bound problem using the shooting method, as described in Yamada et al.
        
        Args:
            root: The root node of the B&B tree
            initial_lower_bound: Initial lower bound (e.g., from Lagrangian or 2-opt)
            initial_upper_bound: Initial upper bound (e.g., from Lagrangian or 2-opt)
            initial_solution: Initial solution edges (e.g., from 2-opt or None)
        
        Returns:
            Tuple[Node, float]: Best solution and its upper bound
        """
        from mstkpbranchandbound import MSTNode  # Import here to avoid circular import

        alpha = 0.3  # Parameter from the paper
        lower_bound = initial_lower_bound
        upper_bound = initial_upper_bound if initial_upper_bound < float("inf") else math.floor(root.best_upper_bound)
        
        # Initialize best solution with initial_solution if provided
        if initial_solution:
            self.best_solution = MSTNode(
                root.edges,
                root.num_nodes,
                root.budget,
                fixed_edges=set(initial_solution)
            )
            self.best_upper_bound = initial_upper_bound
        else:
            self.best_solution = None
            self.best_upper_bound = upper_bound

        shooting_trials = 0
        start_time = time.time()

        while lower_bound < upper_bound:
            # Check for timeout
            if time.time() - start_time > 1800:
                self.timed_out = True
                if self.verbose:
                    print("Shooting method stopped: Exceeded 2000-second time limit")
                break

            shooting_trials += 1
            incumbent_guess = math.floor(alpha * lower_bound + (1 - alpha) * upper_bound)
            if self.verbose:
                print(f"\nShooting Method Trial {shooting_trials}: Guessing incumbent = {incumbent_guess}")
                print(f"Lower bound = {lower_bound}, Upper bound = {upper_bound}")

            # Reset B&B state for this trial
            self.priority_queue = LazyPriorityQueue()
            self.total_nodes_solved = 0
            self.nodes_pruned_lower_bound = 0
            self.nodes_pruned_feasible = 0
            self.nodes_pruned_invalid_mst = 0
            self.nodes_pruned_budget = 0
            self.nodes_pruned_gap = 0
            self.all_lower_bounds = []
            self.pruned_lower_bounds = []
            self.best_lower_bound = float("-inf")
            self.timed_out = False  # Reset timeout flag for this trial

            # Run B&B with the guessed incumbent
            solution, upper_bound = self.solve(root)

            if self.timed_out:
                self._log_node(0, root, None, f"Shooting trial {shooting_trials} timeout after 2000 seconds", self.best_upper_bound)
                break

            if solution and upper_bound <= incumbent_guess:
                if self.verbose:
                    print(f"Shooting method succeeded after {shooting_trials} trials.")
                return solution, upper_bound
            else:
                # Update upper bound and try again
                upper_bound = incumbent_guess
                if self.verbose:
                    print(f"Shooting method failed. Updating upper bound to {upper_bound}.")

        if self.timed_out:
            self._log_node(0, root, None, "Shooting method timeout after 2000 seconds", self.best_upper_bound)
            if self.verbose:
                print(f"Final duality gap on timeout: {self.final_duality_gap:.2f}")

        if self.verbose:
            print(f"Shooting method converged to bounds: Lower = {lower_bound}, Upper = {upper_bound}")
        return self.best_solution, self.best_upper_bound
