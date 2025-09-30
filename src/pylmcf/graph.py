import numpy as np

from pylmcf.pylmcf_cpp import CGraph


class Graph(CGraph):
    '''
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
    '''
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
        for node_id in range(self.no_nodes()):
            nx_graph.add_node(node_id)
        capacities = self.get_edge_capacities()
        costs = self.get_edge_costs()
        flows = self.result()
        for edge_start, edge_end, capacity, cost, flow in zip(
            self.edge_starts(), self.edge_ends(), capacities, costs, flows
        ):
            nx_graph.add_edge(
                edge_start,
                edge_end,
                capacity=capacity,
                cost=cost,
                flow=flow,
                label=f"fl: {flow} / cap: {capacity} @ cost: {cost}",
            )
        # for edge_start, edge_end in zip(self.edge_starts(), self.edge_ends()):
        #    nx_graph.add_edge(
        #        edge_start,
        #        edge_end,
        #    )
        return nx_graph

    def show(self) -> None:
        """
        Show the C++ subgraph as a NetworkX graph.
        """
        import networkx as nx
        from matplotlib import pyplot as plt

        nx_graph = self.as_nx()
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(nx_graph)
        nx.draw(nx_graph, pos, with_labels=True, node_color="lightblue", node_size=500)
        edge_labels = nx.get_edge_attributes(nx_graph, "label")
        nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels)
        plt.show()
