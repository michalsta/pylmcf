import operator
from typing import TYPE_CHECKING, Optional

import numpy as np

from pylmcf.pylmcf_cpp import CGraph

if TYPE_CHECKING:
    # networkx is an optional extra, imported lazily inside the functions that
    # need it; this import exists only so the "nx.DiGraph" annotations resolve.
    import networkx as nx


class Graph(CGraph):
    """
    Graph is a wrapper around the C++ class CGraph, providing additional functionality
    for working with directed graphs, including methods to convert to NetworkX format
    and to visualize the graph.

    The primary purpose of this class is to represent a directed graph with nodes and edges,
    where each edge can have associated costs and capacities, and nodes can have supply or demand values,
    making it suitable for solving network flow problems.

    Args:
        no_nodes (int): Number of nodes in the graph.
        edge_starts (np.ndarray): Array of starting node indices for each edge.
        edge_ends (np.ndarray): Array of ending node indices for each edge.

    Methods:
        as_nx() -> nx.DiGraph | nx.MultiDiGraph:
            Converts the internal C++ subgraph representation to a NetworkX directed graph,
            including node and edge attributes such as demand, capacity, weight, and flow.

        show() -> None:
            Visualizes the graph using matplotlib and NetworkX, displaying nodes and edges
            with labels indicating flow, capacity, and cost.
    """

    def __init__(
        self, no_nodes: int, edge_starts: np.ndarray, edge_ends: np.ndarray
    ) -> None:
        super().__init__(no_nodes, edge_starts, edge_ends)

    def as_nx(self) -> "nx.DiGraph | nx.MultiDiGraph":
        """
        Convert the C++ graph to a NetworkX graph.

        Attribute names are the ones FromNX() reads by default (and the ones
        networkx's own min-cost-flow functions use), so FromNX(G.as_nx())
        reproduces the problem: node "demand" (= -supply), edge "capacity",
        "lower_bound" and "weight" (cost), plus "flow" and "label" once solved.

        Returns a DiGraph, or a MultiDiGraph if the graph has parallel arcs
        (a DiGraph would merge them into one edge).
        """
        import networkx as nx

        edge_starts = self.edge_starts()
        edge_ends = self.edge_ends()
        # Edges are sorted by (start, end), so parallel arcs are adjacent.
        has_parallel = bool(
            np.any((edge_starts[1:] == edge_starts[:-1]) & (edge_ends[1:] == edge_ends[:-1]))
        )
        nx_graph = nx.MultiDiGraph() if has_parallel else nx.DiGraph()
        for node_id, supply in enumerate(self.get_node_supply()):
            nx_graph.add_node(node_id, demand=-supply)
        capacities = self.get_edge_capacities()
        minimums = self.get_edge_minimums()
        costs = self.get_edge_costs()
        try:
            flows = self.result()
        except RuntimeError:
            flows = None
        for i, (edge_start, edge_end, capacity, minimum, cost) in enumerate(
            zip(edge_starts, edge_ends, capacities, minimums, costs)
        ):
            attrs = dict(capacity=capacity, lower_bound=minimum, weight=cost)
            if flows is not None:
                flow = flows[i]
                attrs["flow"] = flow
                attrs["label"] = f"fl: {flow} / cap: {capacity} / min: {minimum} @ cost: {cost}"
            else:
                attrs["label"] = f"cap: {capacity} / min: {minimum} @ cost: {cost}"
            nx_graph.add_edge(edge_start, edge_end, **attrs)
        return nx_graph

    def show(self, filename: Optional[str] = None) -> None:
        """
        Show the C++ subgraph as a NetworkX graph.

        Args:
            filename (Optional[str]): If provided, the graph visualization will be saved to this file.
                                      If None, the graph will be displayed on screen.
        """
        show_graph(self.as_nx(), filename)

    @staticmethod
    def FromNX(
        nx_graph: "nx.DiGraph",
        demand: Optional[str] = "demand",
        capacity: Optional[str] = "capacity",
        lower_bound: Optional[str] = "lower_bound",
        weight: Optional[str] = "weight",
    ) -> "Graph":
        """
        Create a Graph from a NetworkX graph.

        Args:
            nx_graph (nx.DiGraph | nx.MultiDiGraph): The input NetworkX directed
                graph. Parallel arcs of a MultiDiGraph become separate edges.
            demand (str, optional):
                The node attribute name for supply/demand values. Defaults to "demand".
                If not present, the supply must be set later using set_node_supply().
            capacity (str, optional):
                The edge attribute name for capacities. Defaults to "capacity".
                If not present, capacities must be set later using set_edge_capacities().
            lower_bound (str, optional):
                The edge attribute name for minimum flow values. Defaults to "lower_bound".
                If not present, minimums default to zero (no lower bound constraint).
            weight (str, optional):
                The edge attribute name for costs. Defaults to "weight".
                If not present, costs must be set later using set_edge_costs().
        Returns:
            Graph: The created Graph instance.

        Raises:
            ValueError: If the graph is undirected, if nodes are not contiguous
                integers from 0 to n-1, or if an attribute value is not an
                integer (fractional values are rejected, not truncated).
        """
        if not nx_graph.is_directed():
            raise ValueError("FromNX requires a directed graph (DiGraph or MultiDiGraph)")
        no_nodes = nx_graph.number_of_nodes()
        if set(range(no_nodes)) != set(nx_graph.nodes()):
            raise ValueError(
                f"Graph nodes must be contiguous integers from 0 to {no_nodes - 1}, "
                f"got: {sorted(nx_graph.nodes())}"
            )

        # edges(data=True) yields each parallel arc of a MultiDiGraph
        # separately; sorted() is stable, so they keep their key order.
        sorted_edges = sorted(nx_graph.edges(data=True), key=lambda edge: (edge[0], edge[1]))
        no_edges = len(sorted_edges)

        # Edge-index arrays are built as int32 (LEMON's index type) so the
        # C++ constructor takes them without a conversion copy.
        edge_array = np.array([(u, v) for u, v, _ in sorted_edges], dtype=np.int32).reshape(no_edges, 2)
        edge_starts = np.ascontiguousarray(edge_array[:, 0])
        edge_ends   = np.ascontiguousarray(edge_array[:, 1])
        capacities   = np.zeros(no_edges, dtype=np.int64) if capacity    is not None else None
        minimums     = np.zeros(no_edges, dtype=np.int64) if lower_bound is not None else None
        costs        = np.zeros(no_edges, dtype=np.int64) if weight      is not None else None

        for i, (u, v, attrs) in enumerate(sorted_edges):
            if capacities is not None:
                capacities[i] = _integral(attrs.get(capacity,    0), capacity,    (u, v))
            if minimums   is not None:
                minimums[i]   = _integral(attrs.get(lower_bound, 0), lower_bound, (u, v))
            if costs      is not None:
                costs[i]      = _integral(attrs.get(weight,      0), weight,      (u, v))

        G = Graph(no_nodes, edge_starts, edge_ends)

        # Set node supply/demand
        if demand is not None:
            supply = np.zeros(no_nodes, dtype=np.int64)
            for node_id in nx_graph.nodes():
                supply[node_id] = -_integral(nx_graph.nodes[node_id].get(demand, 0), demand, node_id)
            G.set_node_supply(supply)

        if capacities is not None:
            G.set_edge_capacities(capacities)
        if minimums is not None and minimums.any():
            G.set_edge_minimums(minimums)
        if costs is not None:
            G.set_edge_costs(costs)

        return G


def _integral(value, attr: str, where) -> int:
    """value as an int, or ValueError if it is not integer-valued (assigning
    1.5 into an int64 array would silently truncate it to 1)."""
    try:
        return operator.index(value)
    except TypeError:
        pass
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    raise ValueError(f"Attribute {attr!r} of {where} must be an integer, got {value!r}")


def show_graph(nx_graph: "nx.DiGraph", filename: Optional[str] = None) -> None:
    """
    Show a NetworkX graph using matplotlib.
    Args:
        nx_graph (nx.DiGraph): The input NetworkX directed graph.
        filename (Optional[str]): If provided, the graph visualization will be saved to this file.
                                  If None, the graph will be displayed on screen.
    """
    import networkx as nx
    from matplotlib import pyplot as plt

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(nx_graph)

    # draw nodes and labels separately so edges can be drawn with custom styles
    nx.draw_networkx_nodes(nx_graph, pos, node_color="lightblue", node_size=500)
    node_labels = {
        node: f"{node}: {data['demand']}" for node, data in nx_graph.nodes(data=True)
    }
    nx.draw_networkx_labels(nx_graph, pos, labels=node_labels, font_size=10)

    nx.draw_networkx_edges(
        nx_graph,
        pos,
        arrowstyle="->",
        arrowsize=10,
        connectionstyle="arc3, rad=0.15",
    )

    edge_labels = nx.get_edge_attributes(nx_graph, "label")
    nx.draw_networkx_edge_labels(
        nx_graph,
        pos,
        edge_labels=edge_labels,
        connectionstyle="arc3, rad=0.15",
    )
    plt.axis("off")
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
