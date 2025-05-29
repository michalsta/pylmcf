from pylmcf_cpp import LemonGraph
import networkx as nx
from functools import cached_property
import numpy as np


class DecompositableGraphWrapper:
    def __init__(self, cgraph):
        self.cgraph = cgraph

    def no_nodes(self):
        return self.cgraph.no_nodes()

    def no_edges(self):
        return self.cgraph.no_edges()

    def get_subgraphs(self):
        for idx in range(self.cgraph.no_subgraphs()):
            yield FlowSubgraphWrapper(self.cgraph.get_subgraph(idx))


class FlowSubgraphWrapper:
    def __init__(self, subgraph):
        self.subgraph = subgraph

    def no_empirical_nodes(self):
        return self.subgraph.count_empirical_nodes()

    def no_theoretical_nodes(self):
        return self.subgraph.count_theoretical_nodes()

    def matching_density(self):
        return self.subgraph.matching_density()

    def total_cost(self):
        return self.subgraph.total_cost()

    def flows_for_spectrum(self, idx):
        empirical_peak_idx, theoretical_peak_idx, flow = (
            self.subgraph.flows_for_spectrum(idx)
        )
        return empirical_peak_idx, theoretical_peak_idx, flow

    def as_nx_graph(self):
        nx_graph = nx.DiGraph()
        cnodes = self.subgraph.nodes()
        cedges = self.subgraph.edges()
        for node in cnodes:
            nx_graph.add_node(node.id(), layer=node.layer(), type=node.type_str(), str=str(node))
        for edge in cedges:
            nx_graph.add_edge(
                edge.start_node_id(), edge.end_node_id(), str=str(edge),
            )
        return nx_graph

    def show(self):
        from matplotlib import pyplot as plt
        nx_graph = self.as_nx_graph()
        print()
        print()
        print(nx_graph.nodes(data=True))
        pos = nx.multipartite_layout(nx_graph, subset_key="layer")
        edge_labels = nx.get_edge_attributes(nx_graph, "label")
        nx.draw(nx_graph, with_labels=True, pos=pos)
        nx.draw_networkx_edge_labels(nx_graph, pos=pos, edge_labels=edge_labels)
        plt.show()

class GraphWrapper:
    def __init__(self, no_nodes, edge_starts, edge_ends, edge_costs):
        self.lemon_graph = LemonGraph(no_nodes, edge_starts, edge_ends, edge_costs)
        self.no_nodes = no_nodes
        self.edge_starts = edge_starts
        self.edge_ends = edge_ends
        self.edge_costs = edge_costs
        self.node_supply = None
        self.edge_capacities = None
        self.solved = False

    @property
    def nx_graph(self):
        self._nx_graph = nx.DiGraph()
        self._nx_graph.add_nodes_from(range(self.no_nodes))
        self._nx_graph.add_edges_from(
            [
                (self.edge_starts[i], self.edge_ends[i], {"cost": self.edge_costs[i]})
                for i in range(len(self.edge_starts))
            ]
        )
        if self.node_supply is not None:
            for i in range(self.no_nodes):
                self._nx_graph.nodes[i]["supply"] = self.node_supply[i]
        if self.edge_capacities is not None:
            for i in range(len(self.edge_starts)):
                self._nx_graph[self.edge_starts[i]][self.edge_ends[i]]["capacity"] = (
                    self.edge_capacities[i]
                )
        return self._nx_graph

    def set_edge_capacities(self, capacities):
        self.edge_capacities = capacities
        self.lemon_graph.set_edge_capacities(capacities)

    def set_node_supply(self, supply):
        self.node_supply = supply
        self.lemon_graph.set_node_supply(supply)

    def solve(self):
        self.lemon_graph.solve()
        self.solved = True

    def total_cost(self):
        if not self.solved:
            raise ValueError("Graph has not been solved yet.")
        return self.lemon_graph.total_cost()

    def result(self):
        if not self.solved:
            return np.zeros(len(self.edge_starts), dtype=np.int64)
        return self.lemon_graph.result()

    def plot(self):
        from matplotlib import pyplot as plt

        edge_labels = {
            (
                self.edge_starts[i],
                self.edge_ends[i],
            ): f"cost: {self.edge_costs[i]} cap: {self.edge_capacities[i]}"
            for i in range(len(self.edge_starts))
        }
        pos = nx.spring_layout(self.nx_graph)
        nx.draw(self.nx_graph, with_labels=True, pos=pos)
        nx.draw_networkx_edge_labels(self.nx_graph, edge_labels=edge_labels, pos=pos)
        plt.show()
