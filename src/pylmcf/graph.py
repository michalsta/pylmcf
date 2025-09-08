from dataclasses import dataclass
from typing import Union
from abc import ABC
from pprint import pprint
import numpy as np
import networkx as nx
from tqdm import tqdm

from pylmcf.pylmcf_cpp import CGraph

class Graph(CGraph):
    def __init__(self, no_nodes: int, edge_starts: np.ndarray, edge_ends: np.ndarray):
        super().__init__(no_nodes, edge_starts, edge_ends)

    def as_nx(self):
        """
        Convert the C++ subgraph to a NetworkX graph.
        """
        nx_graph = nx.DiGraph()
        for node_id in range(self.no_nodes()):
            nx_graph.add_node(node_id)
        for edge_start, edge_end in zip(self.edge_starts(), self.edge_ends()):
            nx_graph.add_edge(
                edge_start,
                edge_end,
            )
        return nx_graph

    def show(self):
        """
        Show the C++ subgraph as a NetworkX graph.
        """
        from matplotlib import pyplot as plt
        nx_graph = self.as_nx()
        plt.figure(figsize=(8, 6))
        nx.draw(nx_graph, with_labels=True)
        plt.show()
        # pos = nx.multipartite_layout(nx_graph, subset_key="layer")
        # edge_labels = nx.get_edge_attributes(nx_graph, "label")
        # nx.draw(nx_graph, with_labels=True, pos=pos)
        # nx.draw_networkx_edge_labels(nx_graph, pos=pos, edge_labels=edge_labels)
        # plt.show()

