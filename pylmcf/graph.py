from pylmcf import pylmcf_cpp
from array import array
from functools import cached_property
from enum import Enum
from pylmcf.graph_wrapper import GraphWrapper
from pylmcf.graph_elements import *
from pylmcf.spectrum import Distribution
from pylmcf.trashes import TrashFactorySimple
from dataclasses import dataclass
from typing import Union
from abc import ABC
from pprint import pprint
import numpy as np
import networkx as nx
import time
import types
from tqdm import tqdm

# import codon

# Edge = Union[MatchingEdge, SrcToEmpEdge, TheoToSinkEdge, SimpleTrashEdge]


def compare_subgraphs(subgraph1, subgraph2):
    """
    Compare two subgraphs and return True if they are equal, False otherwise.
    """
    s1 = frozenset([frozenset(s) for s in subgraph1])
    s2 = frozenset([frozenset(s) for s in subgraph2])
    return s1 == s2


class DecompositableFlowGraph:
    def __init__(self, empirical_spectrum, theoretical_spectra, dist_fun, max_dist):
        assert isinstance(empirical_spectrum, Distribution)
        if not isinstance(theoretical_spectra, list):
            theoretical_spectra = list(theoretical_spectra)
        assert all(isinstance(ts, Distribution) for ts in theoretical_spectra)
        assert isinstance(dist_fun, types.FunctionType)
        assert isinstance(max_dist, int)

        def wrapped_dist(p, y):
            i = p[1]
            x = p[0][:, i : i + 1]
            return dist_fun(x[: np.newaxis], y)

        print("Creating C++ DecompositableFlowGraph")
        self.cobj = pylmcf_cpp.CDecompositableFlowGraph(
            empirical_spectrum.cspectrum,
            [ts.cspectrum for ts in theoretical_spectra],
            wrapped_dist,
            max_dist,
        )
        print("C++ DecompositableFlowGraph created")
        self.no_theoretical_spectra = 0
        self.graph = nx.DiGraph()
        self.nodes = [None, None]  # Reserve IDs for source and sink in subgraphs
        self.edges = []
        self.built = False

        self._add_empirical_spectrum(empirical_spectrum)

        for spectrum in theoretical_spectra:
            self._add_theoretical_spectrum(
                empirical_spectrum, spectrum, dist_fun, max_dist
            )

    def _add_empirical_spectrum(self, spectrum):
        for idx, peak_intensity in tqdm(
            enumerate(spectrum.intensities), desc="Adding empirical spectrum"
        ):
            node = EmpiricalNode(
                id=len(self.nodes), peak_idx=idx, intensity=peak_intensity
            )
            self.nodes.append(node)
            self.graph.add_node(node, layer=1)

    def _add_theoretical_spectrum(
        self, empirical_spectrum, spectrum, dist_fun, max_dist
    ):
        for idx in tqdm(range(len(spectrum.intensities))):
            theo_node = TheoreticalNode(
                id=len(self.nodes),
                spectrum_id=self.no_theoretical_spectra,
                peak_idx=idx,
                intensity=spectrum.intensities[idx],
            )
            self.nodes.append(theo_node)
            self.graph.add_node(theo_node, layer=2)

            emp_indexes, dists = empirical_spectrum.closer_than(
                spectrum.get_point(idx), max_dist, dist_fun
            )

            for emp_idx, dist in zip(emp_indexes, dists):
                edge = MatchingEdge(
                    start_node=self.nodes[emp_idx + 2],
                    end_node=theo_node,
                    emp_peak_idx=emp_idx,
                    theo_spectrum_id=self.no_theoretical_spectra,
                    theo_peak_idx=idx,
                    cost=dist,
                )
                self.edges.append(edge)
                self.graph.add_edge(
                    self.nodes[emp_idx + 2],
                    theo_node,
                    obj=edge,
                )
        self.no_theoretical_spectra += 1

    def build(self, trash_costructors=[]):
        self.built = True
        self.subgraphs = []

        for trash_costructor in trash_costructors:
            assert isinstance(trash_costructor, TrashFactorySimple)
            self.cobj.add_simple_trash(trash_costructor.trash_cost)

        self.cobj.build()

        print("C++ graph built")

        dead_end_nodes = [
            node for node, degree in dict(self.graph.degree()).items() if degree < 1
        ]

        # self.csubgraphs, self.cdead_end_nodes = self.cobj.subgraphs()
        self.dead_end_trashes = [
            tc.dead_end_trash(dead_end_nodes, self.no_theoretical_spectra)
            for tc in trash_costructors
        ]
        self.graph.remove_nodes_from(dead_end_nodes)

        # print(f"Dead end nodes: {len(dead_end_nodes)}")
        # print(f"Dead end nodes c++: {len(self.cdead_end_nodes)}")
        # assert [n.id for n in dead_end_nodes] == self.cdead_end_nodes, "Dead end nodes do not match with c++ dead end nodes"
        # dead_end_nodes = [self.nodes[i] for i in self.cdead_end_nodes]
        # print(f"Graph nodes: {self.graph.nodes}")

        for_comparison = []
        from tqdm import tqdm

        for subgraph in tqdm(
            list(nx.weakly_connected_components(self.graph)), desc="Building subgraphs"
        ):
            for_comparison.append([n.id for n in subgraph])
            subgraph = FlowSubgraph(self.graph.subgraph(subgraph), self)
            for trash_costructor in trash_costructors:
                trash_costructor.add_to_subgraph(subgraph)

            self.subgraphs.append(subgraph)

        # assert compare_subgraphs(self.csubgraphs, for_comparison), "Subgraphs do not match with c++ subgraphs"
        for subgraph in self.subgraphs:
            subgraph.build()

    def cgraph_as_nx(self):
        """
        Convert the C++ graph to a NetworkX graph.
        """
        nx_graph = nx.DiGraph()
        for node in self.cobj.nodes():
            nx_graph.add_node(node.id(), layer=node.layer())
        for edge in self.edges:
            nx_graph.add_edge(
                edge.start_node.id,
                edge.end_node.id,
                obj=None,
            )
        return nx_graph

    def show_cgraph(self):
        """
        Show the C++ graph as a NetworkX graph.
        """
        from matplotlib import pyplot as plt

        nx_graph = self.cgraph_as_nx()
        pos = nx.multipartite_layout(nx_graph, subset_key="layer")
        edge_labels = nx.get_edge_attributes(nx_graph, "label")
        nx.draw(nx_graph, with_labels=True, pos=pos)
        nx.draw_networkx_edge_labels(nx_graph, pos=pos, edge_labels=edge_labels)
        plt.show()

    def set_point(self, point):
        self.cobj.set_point(point)
        assert len(point) == self.no_theoretical_spectra
        self.point = point
        self.total_cost = 0
        for subgraph in self.subgraphs:
            print(self.total_cost)
            self.total_cost += subgraph.set_point(point)
        ret = self.total_cost + sum(
            trash.cost_at_point(point) for trash in self.dead_end_trashes
        )
        cret = self.cobj.total_cost()
        print(ret, cret)

        assert ret == cret
        return cret

    def show(self):
        from matplotlib import pyplot as plt

        pos = nx.multipartite_layout(self.graph, subset_key="layer")
        nx.draw(self.graph, with_labels=True, pos=pos)
        plt.show()

    def csubgraph_objs(self):
        """
        Convert the C++ subgraph to a NetworkX graph.
        """
        ret = []
        for idx in range(self.cobj.no_subgraphs()):
            ret.append(CSubgraph(self.cobj.get_subgraph(idx)))
        return ret


class FlowSubgraph:
    def __init__(self, nx_graph, parent):
        self.parent = parent
        self.source = SourceNode(0)
        self.sink = SinkNode(1)
        self.nodes = [self.source, self.sink]
        self.source_idx = 0
        self.sink_idx = 1
        self.nodes.extend(nx_graph.nodes)
        self.nodes.sort(key=lambda node: node.id)
        self.node_nx_id_to_lemon_id = {
            node.id: lemon_id for lemon_id, node in enumerate(self.nodes)
        }
        self.node_supply = np.zeros(len(self.nodes), dtype=np.int64)

        self.edges = []

        self.total_empirical_intensity = 0

        for node in self.nodes:
            match node:
                case EmpiricalNode(id, peak_idx, intensity):
                    edge = SrcToEmpEdge(self.source, node, intensity)
                    self.total_empirical_intensity += intensity
                    self.edges.append(edge)
                case TheoreticalNode(id, spectrum_id, peak_idx, intensity):
                    edge = TheoToSinkEdge(node, self.sink, spectrum_id, intensity)
                    self.edges.append(edge)
                case SourceNode(_):
                    pass
                case SinkNode(_):
                    pass
                case _:
                    raise ValueError(
                        f"Unexpected node type: {type(node)} (this shouldn't happen)"
                    )

        self.edges.extend(e[2]["obj"] for e in nx_graph.edges(data=True))

    def build(self):
        self.lemon_edge_starts = np.array(
            [self.node_nx_id_to_lemon_id[edge.start_node.id] for edge in self.edges],
            dtype=np.int64,
        )
        self.lemon_edge_ends = np.array(
            [self.node_nx_id_to_lemon_id[edge.end_node.id] for edge in self.edges],
            dtype=np.int64,
        )
        self.order = np.lexsort((self.lemon_edge_ends, self.lemon_edge_starts))
        self.lemon_edge_starts = np.asarray(self.lemon_edge_starts)[self.order]
        self.lemon_edge_ends = np.asarray(self.lemon_edge_ends)[self.order]
        self.edges = [self.edges[i] for i in self.order]

        def get_edge_cost(edge):
            try:
                return edge.cost
            except AttributeError:
                return 0

        self.lemon_edge_costs = np.array(
            [get_edge_cost(edge) for edge in self.edges], dtype=np.int64
        )
        self.lemon_edge_capacities = np.zeros(
            len(self.lemon_edge_starts), dtype=np.int64
        )

        self.lemon_graph = GraphWrapper(  # pylmcf_cpp.LemonGraph(
            len(self.nodes),
            self.lemon_edge_starts,
            self.lemon_edge_ends,
            self.lemon_edge_costs,
        )

    def nx_graph(self):
        bnx_graph = nx.DiGraph()
        flows = self.lemon_graph.result()
        for node in self.nodes:
            bnx_graph.add_node(node.id, layer=node.layer)
        for edge_id, edge in enumerate(self.edges):
            bnx_graph.add_edge(
                edge.start_node.id,
                edge.end_node.id,
                obj=edge,
                cost=edge.cost,
                capacity=self.lemon_edge_capacities[edge_id],
                flow=flows[edge_id],
                label=f"ca: {int(self.lemon_edge_capacities[edge_id])} co: {int(edge.cost)} f: {int(flows[edge_id])}",
            )
        return bnx_graph

    def show(self):
        from matplotlib import pyplot as plt

        nx_graph = self.nx_graph()
        pos = nx.multipartite_layout(nx_graph, subset_key="layer")
        edge_labels = nx.get_edge_attributes(nx_graph, "label")
        nx.draw(nx_graph, with_labels=True, pos=pos)
        nx.draw_networkx_edge_labels(nx_graph, pos=pos, edge_labels=edge_labels)
        plt.show()

    def set_point(self, point):
        assert len(point) == self.parent.no_theoretical_spectra
        self.total_theoretical_intensity = 0
        trash_edge_idx = None
        for edge_idx, edge in enumerate(self.edges):
            match edge:
                case TheoToSinkEdge(
                    start_node, end_node, theo_spectrum_id, theo_peak_intensity
                ):
                    new_cap = point[theo_spectrum_id] * edge.theo_peak_intensity
                    self.total_theoretical_intensity += new_cap
                    self.lemon_edge_capacities[edge_idx] = new_cap
                case SimpleTrashEdge() | TheoryTrashEdge() | EmpiricalTrashEdge():
                    self.lemon_edge_capacities[edge_idx] = BIGINT
                case MatchingEdge(
                    start_node,
                    end_node,
                    emp_peak_idx,
                    theo_spectrum_id,
                    theo_peak_idx,
                    cost,
                ):
                    self.lemon_edge_capacities[edge_idx] = BIGINT
                case SrcToEmpEdge(start_node, end_node, emp_peak_intensity):
                    print(f"empirical intensity: {emp_peak_intensity}")
                    self.lemon_edge_capacities[edge_idx] = emp_peak_intensity
                case _:
                    pass

        self.total_et_intensity = (
            self.total_empirical_intensity + self.total_theoretical_intensity
        )
        self.node_supply[self.source_idx] = max(
            self.total_empirical_intensity, self.total_theoretical_intensity
        )
        self.node_supply[self.sink_idx] = -1 * self.node_supply[self.source_idx]
        print(
            f"Total empirical intensity: {self.total_empirical_intensity}, total theoretical intensity: {self.total_theoretical_intensity}"
        )

        self.lemon_graph.set_edge_capacities(self.lemon_edge_capacities)
        self.lemon_graph.set_node_supply(self.node_supply)
        self.lemon_graph.solve()
        self.flows = self.lemon_graph.result()
        self.total_cost = self.lemon_graph.total_cost()
        return self.total_cost


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

