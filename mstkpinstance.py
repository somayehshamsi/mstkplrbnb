
import math
import random
import networkx as nx
# matplotlib is imported inside plot() only.  Every benchmark worker imports
# this module, and a top-level pyplot import in 20 fresh processes at once
# means 20 concurrent font-cache builds on a new machine for nothing.

MAX_EDGE_WEIGHT = 12000
MAX_EDGE_LENGTH = 12000
# CONVEX_BUDGET_FACTOR = 0.8
DEFAULT_BETA = 0.6515927038133658

# Target weight-length correlation knob in [-1, 1].
#   0.0  -> lengths drawn independently of weights (reproduces the original
#           instances bit-for-bit, since the draw is a single random.randint).
#   >0   -> positive correlation (cheap edges tend to be short): EASIER.  The
#           minimum-weight tree is then nearly minimum-length, so the budget
#           barely binds.  Measured at n = 300, density 0.05, beta = 0.15:
#           corr = +0.8 drops the cuts-off rung from 2548 to 191 nodes, and
#           the top of the ladder disappears (FULL/LEMMA1 = 0.99x).
#   <0   -> negative correlation (cheap edges tend to be long): HARDER, and
#           this is the classic hard-knapsack structure.  At the same cell
#           corr = -0.8 makes the cuts-off and root-only rungs time out at
#           300s (>12000 nodes) while the node-local cover rungs finish in
#           about 8s with ~450 nodes.
# Set this to 0.8 or -0.8 for correlated runs, then back to 0.0.
DEFAULT_CORR =  0.0


def knob_for_rho(rho):
    """Knob value whose blend has Pearson correlation `rho` (before the
    integer rounding and [1, MAX] clipping, which move it by < 0.01).

    With independent uniforms, r * base + (1 - r) * noise has correlation
    r / sqrt(r^2 + (1 - r)^2) with the weight, so r = rho / (rho + sqrt(1 -
    rho^2)); the negative branch is symmetric.  rho = 0.5 -> 0.366,
    rho = 0.9 -> 0.674, and the old knob -0.8 is rho = -0.97.
    """
    rho = float(rho)
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [-1, 1]")
    a = abs(rho)
    if a == 0.0:
        return 0.0
    r = a / (a + math.sqrt(max(0.0, 1.0 - a * a)))
    return math.copysign(r, rho)


class MSTKPInstance:
    """
    Represents an instance of the MST problem with an extra knapsack constraint.
    Each edge is encoded as a tuple (u, v, w, l) where u and v are the nodes connected by the edge, w is the weight of the edge, and l is the length of the edge.
    The graph is encoded as a list of edges.
    The graph is connected.
    In order to compute the budget, we compute two MSTs:
    - One with the weight of the edges as weights, called Tw
    - One with the length of the edges as weights, called Tl
    The budget is taken a weighted average between the weight of Tw and the weight (not length) of Tl.
    """
    def __init__(self, num_nodes, density, beta=None, corr=None):
        self.num_nodes = num_nodes
        self.budget = None
        self.edges = []

        self.density = density
        self.beta = DEFAULT_BETA if beta is None else float(beta)
        # An explicit argument, so a run can never inherit a module constant
        # someone forgot to set back.  None keeps the old behaviour.
        self.corr = float(DEFAULT_CORR if corr is None else corr)

        assert 0 <= self.beta <= 1, "beta must be between 0 and 1"
        assert -1.0 <= self.corr <= 1.0, "corr must be between -1 and 1"
        self.graph = None
        self.tw = None
        self.tl = None

        self.compute_instance()

    def _correlated_length(self, weight):
        """
        Draw an edge length given its weight, controlled by the module-level
        DEFAULT_CORR knob. At corr == 0.0 this is exactly a single
        random.randint(1, MAX_EDGE_LENGTH) call, so the original independent
        instances are reproduced bit-for-bit (same random call sequence).

        Note: `corr` is a control knob, not the exact Pearson correlation of
        the realized (weight, length) pairs. Report the empirical correlation
        measured over the generated edges if an exact figure is needed.
        """
        r = self.corr
        noise = random.randint(1, MAX_EDGE_LENGTH)
        if r == 0.0:
            return noise
        # Rescale weight onto the length range so the blend is well-scaled.
        base = weight * (MAX_EDGE_LENGTH / MAX_EDGE_WEIGHT)
        if r >= 0:
            # Positive correlation: blend weight with independent noise.
            val = r * base + (1.0 - r) * noise
        else:
            # Negative correlation: blend the *reversed* weight with noise.
            val = (-r) * (MAX_EDGE_LENGTH + 1 - base) + (1.0 + r) * noise
        return max(1, min(MAX_EDGE_LENGTH, int(round(val))))

    def compute_instance(self):
        # Step 1: Start with a spanning tree to ensure connectivity
        self.graph = nx.Graph()
        nodes = list(range(self.num_nodes))
        random.shuffle(nodes)

        # Create a random spanning tree
        for i in range(self.num_nodes - 1):
            u, v = nodes[i], nodes[i + 1]
            weight = random.randint(1, MAX_EDGE_WEIGHT)
            length = self._correlated_length(weight)
            self.graph.add_edge(u, v, weight=weight, length=length)

        # Step 2: Add extra edges based on density
        total_possible_edges = (self.num_nodes * (self.num_nodes - 1)) // 2  # Complete graph edges
        target_edges = max(self.num_nodes - 1, round(self.density * total_possible_edges))  # Ensure at least the tree edges
        edges_added = set(self.graph.edges)

        while len(edges_added) < target_edges:
            u, v = random.sample(nodes, 2)
            if (u, v) not in edges_added and (v, u) not in edges_added:
                weight = random.randint(1, MAX_EDGE_WEIGHT)
                length = self._correlated_length(weight)
                self.graph.add_edge(u, v, weight=weight, length=length)
                edges_added.add((u, v))

        # Store edges in list format
        # self.edges = [(u, v, d["weight"], d["length"]) for u, v, d in self.graph.edges(data=True)]
        self.edges = [(min(u, v), max(u, v), d["weight"], d["length"]) for u, v, d in self.graph.edges(data=True)]        


        # Step 3: Compute the two MSTs
        self.tw = nx.minimum_spanning_tree(self.graph, weight="weight")
        self.tl = nx.minimum_spanning_tree(self.graph, weight="length")

        # Step 4: Compute budget
        total_tw_weight = sum(d["length"] for _, _, d in self.tw.edges(data=True))
        total_tl_weight = sum(d["length"] for _, _, d in self.tl.edges(data=True))

        # assert 0 <= CONVEX_BUDGET_FACTOR <= 1
        # self.budget = int(CONVEX_BUDGET_FACTOR * total_tw_weight + (1 - CONVEX_BUDGET_FACTOR) * total_tl_weight)
        beta = self.beta
        self.budget = int(beta * total_tw_weight + (1 - beta) * total_tl_weight)

        # self.budget = self.num_nodes * 20 -20
        # self.budget=900]
        assert total_tl_weight <= self.budget <= total_tw_weight
        del self.graph

    def print_all_edges(self):
        print("All edges in the graph (u, v, weight, length):")
        for u, v, w, l in self.edges:
            print(f"Edge ({u}, {v}): Weight = {w}, Length = {l}")

    def empirical_correlation(self):
        """Pearson correlation of the realized (weight, length) pairs."""
        ws = [w for _, _, w, _ in self.edges]
        ls = [l for _, _, _, l in self.edges]
        n = len(ws)
        if n < 2:
            return 0.0
        mw = sum(ws) / n
        ml = sum(ls) / n
        cov = sum((w - mw) * (l - ml) for w, l in zip(ws, ls))
        vw = sum((w - mw) ** 2 for w in ws)
        vl = sum((l - ml) ** 2 for l in ls)
        if vw == 0 or vl == 0:
            return 0.0
        return cov / (vw ** 0.5 * vl ** 0.5)

    def plot(self):
        import matplotlib.pyplot as plt
        G = nx.Graph()
        G.add_weighted_edges_from([(u, v, w) for u, v, w, l in self.edges])
        pos = nx.spring_layout(G)
        edge_labels = {(u, v): f"{w}, {l}" for u, v, w, l in self.edges}
        nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_weight="bold")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.axis("off")
        plt.savefig("mstkp_instance.png", format="PNG")
        plt.show()

if __name__ == "__main__":
    instance = MSTKPInstance(100, 0.3)  # More edges with higher density
    instance.plot()
    print(f"Budget: {instance.budget}")
    print(f"Edges: {instance.edges}")
    print(f"Total edges in the graph: {len(instance.edges)}")
    print(f"Expected number of edges: {round(0.6 * ((10 * 9) // 2))}")  # Expected based on density
    print(f"Configured corr: {instance.corr}, empirical corr: {instance.empirical_correlation():.3f}")
    print(f"Tw: {instance.tw.edges(data=True)}")
    print(f"Tl: {instance.tl.edges(data=True)}")
    print(f"Total Tw weight: {sum([d['weight'] for _, _, d in instance.tw.edges(data=True)])}")
    print(f"Total Tl weight: {sum([d['weight'] for _, _, d in instance.tl.edges(data=True)])}")
    print(f"Total Tw length: {sum([d['length'] for _, _, d in instance.tw.edges(data=True)])}")
    print(f"Total Tl length: {sum([d['length'] for _, _, d in instance.tl.edges(data=True)])}")
