from dataclasses import dataclass
from typing import Union
from abc import ABC
from pprint import pprint
import numpy as np
import networkx as nx
from tqdm import tqdm



class CSubgraph:
    def __init__(self, cobj):
        self.cobj = cobj

    def as_nx(self):
        """
        Convert the C++ subgraph to a NetworkX graph.
        """
        nx_graph = nx.DiGraph()
        for node in self.cobj.nodes():
            nx_graph.add_node(node.id(), layer=node.layer())
        for edge in self.cobj.edges():
            nx_graph.add_edge(
                edge.start_node_id(),
                edge.end_node_id(),
                obj=None,
            )
        return nx_graph

    def show(self):
        """
        Show the C++ subgraph as a NetworkX graph.
        """
        from matplotlib import pyplot as plt

        nx_graph = self.as_nx()
        pos = nx.multipartite_layout(nx_graph, subset_key="layer")
        edge_labels = nx.get_edge_attributes(nx_graph, "label")
        nx.draw(nx_graph, with_labels=True, pos=pos)
        nx.draw_networkx_edge_labels(nx_graph, pos=pos, edge_labels=edge_labels)
        plt.show()

