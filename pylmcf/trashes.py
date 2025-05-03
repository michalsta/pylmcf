from abc import ABC, abstractmethod
import numpy as np
from pylmcf.graph_elements import *




class DeadEndTrash():
    def __init__(self, trash_cost, dead_end_nodes, no_theoretical_spectra):
        self.trash_cost = trash_cost
        self.empirical_intensity = 0
        self.theoretical_intensities = np.zeros(no_theoretical_spectra, dtype=np.int64)
        for node in dead_end_nodes:
            match node:
                case EmpiricalNode(id, peak_idx, intensity):
                    self.empirical_intensity += intensity
                case TheoreticalNode(id, spectrum_id, peak_idx,intensity):
                    self.theoretical_intensities[spectrum_id] += intensity
                case _:
                    raise ValueError("Unknown node type")

    def cost_at_point(self, point):
        empirical_cost = self.empirical_intensity
        theoretical_cost = np.sum(self.theoretical_intensities * point)
        print("Trash cost at point", point, "empirical cost", empirical_cost, "theoretical cost", theoretical_cost)
        return max(empirical_cost, theoretical_cost) * self.trash_cost


class TrashFactory(ABC):
    @abstractmethod
    def dead_end_trash(self, dead_end_nodes, no_theoretical_spectra):
        pass

    @abstractmethod
    def add_to_subgraph(self, subgraph):
        pass


class TrashFactorySimple(TrashFactory):
    def __init__(self, trash_cost):
        self.trash_cost = trash_cost

    def dead_end_trash(self, dead_end_nodes, no_theoretical_spectra):
        return DeadEndTrash(self.trash_cost, dead_end_nodes, no_theoretical_spectra)

    def add_to_subgraph(self, subgraph):
        subgraph.edges.append(SimpleTrashEdge(subgraph.source, subgraph.sink, self.trash_cost))



class TrashFactoryEmpirical(TrashFactory):
    def __init__(self, trash_cost):
        self.trash_cost = trash_cost

    def dead_end_trash(self, dead_end_nodes, no_theoretical_spectra):
        empirical_dead_ends = [node for node in dead_end_nodes if isinstance(node, EmpiricalNode)]
        return DeadEndTrash(self.trash_cost, empirical_dead_ends, no_theoretical_spectra)

    def add_to_subgraph(self, subgraph):
        for node in subgraph.nodes:
            match node:
                case EmpiricalNode(id, peak_idx, intensity):
                    subgraph.edges.append(EmpiricalTrashEdge(node, subgraph.sink, intensity, self.trash_cost))
                case _:
                    pass

class TrashFactoryTheory(TrashFactory):
    def __init__(self, trash_cost):
        self.trash_cost = trash_cost

    def dead_end_trash(self, dead_end_nodes, no_theoretical_spectra):
        theoretical_dead_ends = [node for node in dead_end_nodes if isinstance(node, TheoreticalNode)]
        return DeadEndTrash(self.trash_cost, theoretical_dead_ends, len(theoretical_dead_ends))

    def add_to_subgraph(self, subgraph):
        for node in subgraph.nodes:
            match node:
                case TheoreticalNode(peak_idx, intensity):
                    subgraph.edges.append(TheoryTrashEdge(subgraph.source, node, self.trash_cost))
                case _:
                    pass


# =========================== OBSOLETE CODE BELOW DO NOT USE ==========================


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



