from abc import ABC, abstractmethod
import numpy as np
from pylmcf.graph_elements import *







class TrashFactory(ABC):
    @abstractmethod
    def dead_end_trash_cost(self, dead_end_nodes):
        pass
    @abstractmethod
    def add_to_subgraph(self, subgraph):
        pass


class TrashFactorySimple(TrashFactory):
    def __init__(self, trash_cost):
        self.trash_cost = trash_cost

    def dead_end_trash_cost(self, dead_end_nodes):
        return 0

    def add_to_subgraph(self, subgraph):
        subgraph.edges.append(SimpleTrashEdge(TODO_REMOVE_ME, subgraph.source, subgraph.sink, self.trash_cost))




class Trash(ABC):
    def apply_cost_scaling(self, scaling):
        self.scale = scaling

    @abstractmethod
    def set_edge_capacities(self):
        pass

    @abstractmethod
    def add_to_network(self, WNM):
        pass

    @abstractmethod
    def __init__(self, trash_cost):
        pass

    @abstractmethod
    def distance_limit(self):
        pass


class SimpleTrash(Trash):
    def __init__(self, trash_cost):
        self.scale = 1
        self.trash_cost = trash_cost

    def add_to_network(self, WNM):
        self.G = WNM.G
        self.WNM = WNM
        self.trash_edge = self.G.add_edge(self.WNM.source, self.WNM.sink, np.int64(self.trash_cost * self.scale))

    def set_edge_capacities(self):
        # print("Trash capacity", np.sum(self.WNM.empirical_spectrum.intensities))
        self.edge_capacity = self.WNM.total_supply
        self.G.set_edge_capacities(
            np.array([self.trash_edge]),
            np.array([self.edge_capacity]),
        )

    def distance_limit(self):
        return self.trash_cost * self.scale

    def print_summary(self):
        print("Simple trash")
        print("     Trash cost", self.trash_cost * self.scale)
        print("     Trash edge", self.trash_edge)
        print("     Trash capacity", np.sum(self.WNM.empirical_spectrum.intensities))
        print("     Trash flow", self.G.flows[self.trash_edge])

class NoTrash(Trash):
    def __init__(self, cost = None):
        pass

    def add_to_network(self, WNM):
        pass

    def set_edge_capacities(self):
        pass

    def distance_limit(self):
        return np.inf

    def print_summary(self):
        print("Trivial trash")

class EmpiricalTrash(Trash):
    def __init__(self, trash_cost):
        self.scale = 1
        self.trash_cost = trash_cost

    def add_to_network(self, WNM):
        self.G = WNM.G
        self.WNM = WNM
        self.matching_network = WNM.matching
        edge_starts = self.matching_network.empirical_node_ids
        edge_ends = np.full(len(edge_starts), WNM.sink)
        self.trash_edges = self.G.add_edges(
            edge_starts, edge_ends, np.full(len(edge_starts), self.trash_cost * self.scale)
        )

    def set_edge_capacities(self):
        self.G.set_edge_capacities(
            self.trash_edges, self.WNM.empirical_spectrum.intensities
        )

    def distance_limit(self):
        return self.trash_cost * self.scale

    def print_summary(self):
        print("Empirical trash")
        print("Trash cost", self.trash_cost * self.scale)
        print("Trash edges", self.trash_edges)
        print("Trash edge capacities", self.WNM.empirical_spectrum.intensities)
        print("Trash edge intensities", self.WNM.empirical_spectrum.intensities)



class TheoryTrash(Trash):
    def __init__(self, trash_cost):
        self.scale = 1
        self.trash_cost = trash_cost

    def add_to_network(self, WNM):
        self.G = WNM.G
        self.WNM = WNM
        self.matching_network = WNM.matching
        self.self_single_theory_trashes = [
            SingleTheoryTrash(WNM, self.trash_cost * self.scale, STM) for STM in self.matching_network.theory_matchers
        ]

    def set_edge_capacities(self):
        for single_theory_trash in self.self_single_theory_trashes:
            single_theory_trash.set_edge_capacities()

    def distance_limit(self):
        return self.trash_cost * self.scale


class SingleTheoryTrash:
    def __init__(self, WNM, trash_cost, STM):
        self.G = WNM.G
        self.WNM = WNM
        self.trash_cost = trash_cost
        self.STM = STM
        edge_starts = np.full(len(STM.theoretical_nodes), WNM.source)
        edge_ends = STM.theoretical_nodes
        self.trash_edges = self.G.add_edges(
            edge_starts, edge_ends, np.full(len(edge_starts), self.trash_cost)
        )

    def set_edge_capacities(self):
        self.G.set_edge_capacities(
            self.trash_edges, self.STM.theoretical_spectrum.intensities
        )



