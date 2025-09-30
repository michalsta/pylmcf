import numpy as np

from pylmcf.pylmcf_cpp import CGraph


class Graph(CGraph):
    def __init__(
        self, no_nodes: int, edge_starts: np.ndarray, edge_ends: np.ndarray
    ) -> None:
        super().__init__(no_nodes, edge_starts, edge_ends)

    def as_nx(self) -> "nx.DiGraph":
        """
        Convert the C++ subgraph to a NetworkX graph.
        """
        import networkx as nx

        nx_graph = nx.DiGraph()
        for node_id in range(self.no_nodes()):
            nx_graph.add_node(node_id)
        for edge_start, edge_end in zip(self.edge_starts(), self.edge_ends()):
            nx_graph.add_edge(
                edge_start,
                edge_end,
            )
        return nx_graph

    def show(self) -> None:
        """
        Show the C++ subgraph as a NetworkX graph.
        """
        import networkx as nx
        from matplotlib import pyplot as plt

        nx_graph = self.as_nx()
        plt.figure(figsize=(8, 6))
        nx.draw(nx_graph, with_labels=True)
        plt.show()
