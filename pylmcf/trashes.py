from abc import ABC, abstractmethod
import numpy as np
from pylmcf.graph_elements import *


class DeadEndTrash:
    def __init__(self, trash_cost, dead_end_nodes, no_theoretical_spectra):
        self.trash_cost = trash_cost
        self.empirical_intensity = 0
        self.theoretical_intensities = np.zeros(no_theoretical_spectra, dtype=np.int64)
        for node in dead_end_nodes:
            match node:
                case EmpiricalNode(id, peak_idx, intensity):
                    self.empirical_intensity += intensity
                case TheoreticalNode(id, spectrum_id, peak_idx, intensity):
                    self.theoretical_intensities[spectrum_id] += intensity
                case _:
                    raise ValueError("Unknown node type")

    def cost_at_point(self, point):
        empirical_cost = self.empirical_intensity
        theoretical_cost = np.sum(self.theoretical_intensities * point)
        print(
            "Trash cost at point",
            point,
            "empirical cost",
            empirical_cost,
            "theoretical cost",
            theoretical_cost,
        )
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
        subgraph.edges.append(
            SimpleTrashEdge(subgraph.source, subgraph.sink, self.trash_cost)
        )


class TrashFactoryEmpirical(TrashFactory):
    def __init__(self, trash_cost):
        self.trash_cost = trash_cost

    def dead_end_trash(self, dead_end_nodes, no_theoretical_spectra):
        empirical_dead_ends = [
            node for node in dead_end_nodes if isinstance(node, EmpiricalNode)
        ]
        return DeadEndTrash(
            self.trash_cost, empirical_dead_ends, no_theoretical_spectra
        )

    def add_to_subgraph(self, subgraph):
        for node in subgraph.nodes:
            match node:
                case EmpiricalNode(id, peak_idx, intensity):
                    subgraph.edges.append(
                        EmpiricalTrashEdge(
                            node, subgraph.sink, intensity, self.trash_cost
                        )
                    )
                case _:
                    pass


class TrashFactoryTheory(TrashFactory):
    def __init__(self, trash_cost):
        self.trash_cost = trash_cost

    def dead_end_trash(self, dead_end_nodes, no_theoretical_spectra):
        theoretical_dead_ends = [
            node for node in dead_end_nodes if isinstance(node, TheoreticalNode)
        ]
        return DeadEndTrash(
            self.trash_cost, theoretical_dead_ends, len(theoretical_dead_ends)
        )

    def add_to_subgraph(self, subgraph):
        for node in subgraph.nodes:
            match node:
                case TheoreticalNode(peak_idx, intensity):
                    subgraph.edges.append(
                        TheoryTrashEdge(subgraph.source, node, self.trash_cost)
                    )
                case _:
                    pass

