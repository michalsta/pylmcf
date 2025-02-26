from abc import ABC, abstractmethod
import numpy as np


class Trash(ABC):
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
        self.trash_cost = trash_cost

    def add_to_network(self, WNM):
        self.G = WNM.G
        self.WNM = WNM
        self.trash_edge = self.G.add_edge(self.WNM.source, self.WNM.sink, self.trash_cost)

    def set_edge_capacities(self):
        # print("Trash capacity", np.sum(self.WNM.empirical_spectrum.intensities))
        self.G.set_edge_capacities(
            np.array([self.trash_edge]),
            np.array([np.sum(self.WNM.empirical_spectrum.intensities)]),
        )

    def distance_limit(self):
        return self.trash_cost

class NoTrash(Trash):
    def __init__(self, cost = None):
        pass

    def add_to_network(self, WNM):
        pass

    def set_edge_capacities(self):
        pass

    def distance_limit(self):
        return np.inf

class EmpiricalTrash:
    def __init__(self, WNM, trash_cost):
        self.trash_cost = trash_cost

    def add_to_network(self, WNM):
        self.G = WNM.G
        self.WNM = WNM
        edge_starts = WNM.empirical_node_ids
        edge_ends = np.full(len(edge_starts), WNM.sink)
        self.trash_edges = self.G.add_edges(
            edge_starts, edge_ends, np.full(len(edge_starts), self.trash_cost)
        )

    def set_edge_capacities(self):
        self.G.set_edge_capacities(
            self.trash_edges, self.WNM.empirical_spectrum.intensities
        )

    def distance_limit(self):
        return self.trash_cost


class TheoryTrash(Trash):
    def __init__(self, WNM, trash_cost):
        self.trash_cost = trash_cost

    def add_to_network(self, WNM):
        self.G = WNM.G
        self.WNM = WNM
        self.self_single_theory_trashes = [
            SingleTheoryTrash(WNM, trash_cost, STM) for STM in WNM.theory_matchers
        ]

    def set_edge_capacities(self):
        for single_theory_trash in self.self_single_theory_trashes:
            single_theory_trash.set_edge_capacities()


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
