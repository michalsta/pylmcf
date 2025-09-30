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
