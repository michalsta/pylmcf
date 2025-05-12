from pylmcf import pylmcf_cpp
from array import array
from functools import cached_property
from enum import Enum
from pylmcf.graph_wrapper import GraphWrapper
from pylmcf.graph_elements import *
from pylmcf.spectrum import Spectrum
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

#Edge = Union[MatchingEdge, SrcToEmpEdge, TheoToSinkEdge, SimpleTrashEdge]


def compare_subgraphs(subgraph1, subgraph2):
    """
    Compare two subgraphs and return True if they are equal, False otherwise.
    """
    s1 = frozenset([frozenset(s) for s in subgraph1])
    s2 = frozenset([frozenset(s) for s in subgraph2])
    return s1 == s2

class DecompositableFlowGraph:
    def __init__(self, empirical_spectrum, theoretical_spectra, dist_fun, max_dist):
        assert isinstance(empirical_spectrum, Spectrum)
        if not isinstance(theoretical_spectra, list):
            theoretical_spectra = list(theoretical_spectra)
        assert all(isinstance(ts, Spectrum) for ts in theoretical_spectra)
        assert isinstance(dist_fun, types.FunctionType)
        assert isinstance(max_dist, int)

        def wrapped_dist(p, y):
            i = p[1]
            x = p[0][:, i:i+1]
            return dist_fun(x[: np.newaxis], y)
        print("Creating C++ DecompositableFlowGraph")
        self.cobj = pylmcf_cpp.CDecompositableFlowGraph(empirical_spectrum.cspectrum, [ts.cspectrum for ts in theoretical_spectra], wrapped_dist, max_dist)
        print("C++ DecompositableFlowGraph created")
        self.no_theoretical_spectra = 0
        self.graph = nx.DiGraph()
        self.nodes = [None, None] # Reserve IDs for source and sink in subgraphs
        self.edges = []
        self.built = False

        self._add_empirical_spectrum(empirical_spectrum)

        for spectrum in theoretical_spectra:
            self._add_theoretical_spectrum(empirical_spectrum, spectrum, dist_fun, max_dist)

    def _add_empirical_spectrum(self, spectrum):
        for idx, peak_intensity in tqdm(enumerate(spectrum.intensities), desc="Adding empirical spectrum"):
            node = EmpiricalNode(
                id=len(self.nodes), peak_idx=idx, intensity=peak_intensity
            )
            self.nodes.append(node)
            self.graph.add_node(node, layer=1)


    def _add_theoretical_spectrum(self, empirical_spectrum, spectrum, dist_fun, max_dist):
        for idx in tqdm(range(len(spectrum.intensities))):
            theo_node = TheoreticalNode(
                id=len(self.nodes),
                spectrum_id=self.no_theoretical_spectra,
                peak_idx=idx,
                intensity=spectrum.intensities[idx],
            )
            self.nodes.append(theo_node)
            self.graph.add_node(theo_node, layer=2)

            emp_indexes, dists = empirical_spectrum.closer_than(spectrum.get_point(idx), max_dist, dist_fun)

            for emp_idx, dist in zip(emp_indexes, dists):
                edge = MatchingEdge(
                    start_node=self.nodes[emp_idx+2],
                    end_node=theo_node,
                    emp_peak_idx=emp_idx,
                    theo_spectrum_id=self.no_theoretical_spectra,
                    theo_peak_idx=idx,
                    cost=dist,
                )
                self.edges.append(edge)
                self.graph.add_edge(
                    self.nodes[emp_idx+2],
                    theo_node,
                    obj=edge,
                )
        self.no_theoretical_spectra += 1

    def build(self, trash_costructors=[]):
        self.built = True
        self.subgraphs = []

        self.cobj.build()

        print("C++ graph built")

        dead_end_nodes = [
            node for node, degree in dict(self.graph.degree()).items() if degree < 1
        ]

        self.csubgraphs, self.cdead_end_nodes = self.cobj.subgraphs()
        self.dead_end_trashes = [tc.dead_end_trash(dead_end_nodes, self.no_theoretical_spectra) for tc in trash_costructors]
        self.graph.remove_nodes_from(dead_end_nodes)



        print(f"Dead end nodes: {len(dead_end_nodes)}")
        print(f"Dead end nodes c++: {len(self.cdead_end_nodes)}")
        assert [n.id for n in dead_end_nodes] == self.cdead_end_nodes, "Dead end nodes do not match with c++ dead end nodes"
        dead_end_nodes = [self.nodes[i] for i in self.cdead_end_nodes]
        print(f"Graph nodes: {self.graph.nodes}")

        for_comparison = []
        from tqdm import tqdm
        for subgraph in tqdm(list(nx.weakly_connected_components(self.graph)), desc="Building subgraphs"):
            for_comparison.append([n.id for n in subgraph])
            subgraph = FlowSubgraph(self.graph.subgraph(subgraph), self)
            for trash_costructor in trash_costructors:
                trash_costructor.add_to_subgraph(subgraph)

            self.subgraphs.append(subgraph)

        assert compare_subgraphs(self.csubgraphs, for_comparison), "Subgraphs do not match with c++ subgraphs"
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
        return self.total_cost + sum(trash.cost_at_point(point) for trash in self.dead_end_trashes)

    def show(self):
        from matplotlib import pyplot as plt
        pos = nx.multipartite_layout(self.graph, subset_key="layer")
        nx.draw(self.graph, with_labels=True, pos=pos)
        plt.show()


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
                    raise ValueError(f"Unexpected node type: {type(node)} (this shouldn't happen)")

        self.edges.extend(e[2]['obj'] for e in nx_graph.edges(data=True))


    def build(self):
        self.lemon_edge_starts = np.array(
            [self.node_nx_id_to_lemon_id[edge.start_node.id] for edge in self.edges], dtype=np.int64)
        self.lemon_edge_ends = np.array(
            [self.node_nx_id_to_lemon_id[edge.end_node.id] for edge in self.edges], dtype=np.int64)
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
        self.lemon_edge_capacities = np.zeros(len(self.lemon_edge_starts), dtype=np.int64)

        self.lemon_graph = GraphWrapper(  # pylmcf_cpp.LemonGraph(
            len(self.nodes), self.lemon_edge_starts, self.lemon_edge_ends, self.lemon_edge_costs
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
                case TheoToSinkEdge(start_node, end_node, theo_spectrum_id, theo_peak_intensity):
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

        self.total_et_intensity = self.total_empirical_intensity + self.total_theoretical_intensity
        self.node_supply[self.source_idx] = max(self.total_empirical_intensity, self.total_theoretical_intensity)
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


# =========================== OBSOLETE CODE BELOW DO NOT USE ==========================

class FlowGraph:
    def __init__(self):
        self.no_nodes = 0
        self.edge_starts = array("q")
        self.edge_ends = array("q")
        self.edge_costs = array("q")
        self.edge_ids = array("q")
        self.order = None
        self.flows = None
        self.source = self.add_nodes(1)[0]
        self.sink = self.add_nodes(1)[0]

    def add_nodes(self, no_nodes):
        assert self.order is None, "Cannot add nodes after building the graph"
        ret = self.no_nodes
        self.no_nodes += no_nodes
        ret = np.arange(ret, ret + no_nodes, dtype=np.int64)
        return ret

    def add_edges(self, starts, ends, costs):
        assert self.order is None, "Cannot add edges after building the graph"
        new_ids = range(len(self.edge_starts), len(self.edge_starts) + len(starts))
        self.edge_ids.extend(new_ids)
        self.edge_starts.extend(starts)
        self.edge_ends.extend(ends)
        self.edge_costs.extend(costs)
        ret = np.array(new_ids, dtype=np.int64)
        return ret

    def add_edge(self, start, end, cost):
        assert self.order is None, "Cannot add edges after building the graph"
        new_id = len(self.edge_starts)
        self.edge_ids.append(new_id)
        self.edge_starts.append(start)
        self.edge_ends.append(end)
        self.edge_costs.append(cost)
        return new_id

    def build(self):
        self.order = np.lexsort((self.edge_ends, self.edge_starts))
        self.edge_starts = np.asarray(self.edge_starts)[self.order]
        self.edge_ends = np.asarray(self.edge_ends)[self.order]
        self.edge_costs = np.asarray(self.edge_costs)[self.order]
        self.edge_ids = np.asarray(self.edge_ids)[self.order]
        self.edge_capacities = np.zeros(len(self.edge_starts), dtype=np.int64)
        self.node_supply = np.zeros(self.no_nodes, dtype=np.int64)
        self.edge_id_to_index = np.argsort(self.edge_ids)

        self.trim()
        self.lemon_graph = pylmcf_cpp.LemonGraph(
            self.no_nodes, self.edge_starts, self.edge_ends, self.edge_costs
        )

    def set_edge_capacities(self, edge_ids, capacities):
        assert (
            self.order is not None
        ), "Cannot set edge capacities before building the graph"
        assert len(edge_ids) == len(capacities)
        self.edge_capacities[self.edge_id_to_index[edge_ids]] = capacities

    def set_node_supply(self, node_ids, supply):
        self.node_supply[node_ids] = supply

    def solve(self):
        self.lemon_graph.set_edge_capacities(self.edge_capacities)
        self.lemon_graph.set_node_supply(self.node_supply)
        start = time.perf_counter()
        self.lemon_graph.solve()
        self.lemon_time = time.perf_counter() - start
        self.flows = self.lemon_graph.result()
        self.total_cost = self.lemon_graph.total_cost()

    def result(self, edge_ids):
        return self.flows[self.edge_id_to_index[edge_ids]]

    def __str__(self):
        return "Graph with %d nodes and %d edges" % (
            self.no_nodes,
            len(self.edge_starts),
        )

    def print(self):
        print(str(self))
        print("Edges:")
        for i in range(len(self.edge_starts)):
            print(
                "Edge %d: %d -> %d, cost %d, capacity %d, flow %d"
                % (
                    i,
                    self.edge_starts[i],
                    self.edge_ends[i],
                    self.edge_costs[i],
                    self.edge_capacities[i],
                    self.flows[i],
                )
            )

    def to_networkx(self):
        G = nx.DiGraph()
        for i in range(len(self.edge_starts)):
            G.add_edge(
                self.edge_starts[i],
                self.edge_ends[i],
                capacity=self.edge_capacities[i],
                # flow=self.flows[i],
                # label=f"ca: {int(self.edge_capacities[i])} co: {int(self.edge_costs[i])} f: {int(self.flows[i])}"
            )
        return G

    def trim(self):
        return
        G = self.to_networkx()
        dead_end_nodes = [
            node for node, degree in dict(G.degree()).items() if degree <= 1
        ]
        G.remove_nodes_from(dead_end_nodes)
        components_split = G.subgraph(range(2, self.no_nodes))
        components = nx.weakly_connected_components(components_split)
        subgraphs = [G.subgraph([0, 1] + list(c)) for c in components]
        raise Exception()

    def show(self):
        from matplotlib import pyplot as plt

        nxg = self.to_networkx()
        edge_labels = nx.get_edge_attributes(nxg, "label")
        ranks = {node: nx.shortest_path_length(nxg, 0, node) for node in nxg.nodes}
        ranks[1] = 4
        # Assign rank as 'layer' attribute
        nx.set_node_attributes(nxg, ranks, "layer")

        # Use multipartite_layout based on 'layer'
        pos = nx.multipartite_layout(nxg, subset_key="layer")

        nx.draw(nxg, pos=pos, with_labels=True)
        nx.draw_networkx_edge_labels(nxg, pos=pos, edge_labels=edge_labels)
        plt.show()
