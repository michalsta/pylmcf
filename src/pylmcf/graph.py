from typing import Optional

import numpy as np

from pylmcf.pylmcf_cpp import CGraph


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
        as_nx() -> nx.DiGraph:
            Converts the internal C++ subgraph representation to a NetworkX directed graph,
            including node and edge attributes such as capacity, cost, and flow.

        show() -> None:
            Visualizes the graph using matplotlib and NetworkX, displaying nodes and edges
            with labels indicating flow, capacity, and cost.
    """

    def __init__(
        self, no_nodes: int, edge_starts: np.ndarray, edge_ends: np.ndarray
    ) -> None:
        super().__init__(no_nodes, edge_starts, edge_ends)

    def as_nx(self) -> "nx.DiGraph":
        """
        Convert the C++ graph to a NetworkX graph.
        """
        import networkx as nx

        nx_graph = nx.DiGraph()
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
            zip(self.edge_starts(), self.edge_ends(), capacities, minimums, costs)
        ):
            attrs = dict(capacity=capacity, minimum=minimum, cost=cost)
            if flows is not None:
                flow = flows[i]
                attrs["flow"] = flow
                attrs["label"] = f"fl: {flow} / cap: {capacity} / min: {minimum} @ cost: {cost}"
            else:
                attrs["label"] = f"cap: {capacity} / min: {minimum} @ cost: {cost}"
            nx_graph.add_edge(edge_start, edge_end, **attrs)
        # for edge_start, edge_end in zip(self.edge_starts(), self.edge_ends()):
        #    nx_graph.add_edge(
        #        edge_start,
        #        edge_end,
        #    )
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
            nx_graph (nx.DiGraph): The input NetworkX directed graph.
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
            ValueError: If nodes are not contiguous integers from 0 to n-1.
        """
        no_nodes = nx_graph.number_of_nodes()
        if set(range(no_nodes)) != set(nx_graph.nodes()):
            raise ValueError(
                f"Graph nodes must be contiguous integers from 0 to {no_nodes - 1}, "
                f"got: {sorted(nx_graph.nodes())}"
            )

        edge_starts = []
        edge_ends = []

        sorted_edges = sorted(nx_graph.edges(), key=lambda edge: (edge[0], edge[1]))

        for u, v in sorted_edges:
            edge_starts.append(u)
            edge_ends.append(v)
        G = Graph(no_nodes, np.array(edge_starts), np.array(edge_ends))

        # Set node supply/demand
        if demand is not None:
            supply = np.zeros(no_nodes, dtype=np.int64)
            for node_id in nx_graph.nodes():
                supply[node_id] = -nx_graph.nodes[node_id].get(demand, 0)
            G.set_node_supply(supply)

        # Set edge capacities
        if capacity is not None:
            capacities = np.zeros(G.no_edges(), dtype=np.int64)
            for i, (u, v) in enumerate(sorted_edges):
                capacities[i] = nx_graph[u][v].get(capacity, 0)
            G.set_edge_capacities(capacities)

        # Set edge lower bounds
        if lower_bound is not None:
            minimums = np.zeros(G.no_edges(), dtype=np.int64)
            for i, (u, v) in enumerate(sorted_edges):
                minimums[i] = nx_graph[u][v].get(lower_bound, 0)
            if minimums.any():
                G.set_edge_minimums(minimums)

        # Set edge costs
        if weight is not None:
            costs = np.zeros(G.no_edges(), dtype=np.int64)
            for i, (u, v) in enumerate(sorted_edges):
                costs[i] = nx_graph[u][v].get(weight, 0)
            G.set_edge_costs(costs)

        return G


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
