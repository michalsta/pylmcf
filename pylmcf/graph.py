from pylmcf import pylmcf_cpp
from array import array
from functools import cached_property
from enum import Enum
from pylmcf.graph_wrapper import GraphWrapper
from dataclasses import dataclass
from typing import Union
from abc import ABC
import numpy as np
import networkx as nx
import time

# import codon


BIGINT = np.int64(2**56)


@dataclass(frozen=True)
class SourceNode:
    id: int


@dataclass(frozen=True)
class SinkNode:
    id: int


@dataclass(frozen=True)
class EmpiricalNode:
    id: int
    peak_idx: int
    intensity: int


@dataclass(frozen=True)
class TheoreticalNode:
    id: int
    peak_idx: int
    spectrum_id: int
    intensity: int


Node = Union[SourceNode, SinkNode, EmpiricalNode, TheoreticalNode]


@dataclass(frozen=True)
class MatchingEdge:
    id: int
    start_node: Node
    end_node: Node
    emp_peak_idx: int
    theo_spectrum_id: int
    theo_peak_idx: int

@dataclass(frozen=True)
class SrcToEmpEdge:
    id: int
    start_node: Node
    end_node: Node
    emp_peak_intensity: int

@dataclass(frozen=True)
class TheoToSinkEdge:
    id: int
    start_node: Node
    end_node: Node
    theo_spectrum_id: int
    theo_peak_intensity: int

@dataclass(frozen=True)
class SimpleTrashEdge:
    id: int
    start_node: Node
    end_node: Node

@dataclass(frozen=True)
class TheoryTrashEdge:
    id: int
    start_node: Node
    end_node: Node
    theo_spectrum_id: int
    theo_peak_intensity: int

@dataclass(frozen=True)
class EmpiricalTrashEdge:
    id: int
    start_node: Node
    end_node: Node
    emp_peak_intensity: int

Edge = Union[MatchingEdge, SrcToEmpEdge, TheoToSinkEdge, SimpleTrashEdge]




class DecompositableFlowGraph:
    def __init__(self):
        self.source = SourceNode(0)
        self.sink = SinkNode(1)
        self.empirical_spectrum = None
        self.empirical_spectrum_corresponding_nodes = []
        self.theoretical_spectra = []
        self.graph = nx.DiGraph()
        self.nodes = [self.source, self.sink]
        self.edges = []
        self.built = False

    def add_empirical_spectrum(self, spectrum):
        assert (
            self.empirical_spectrum is None
        ), "Cannot add more than one empirical spectrum"
        assert (
            len(self.nodes) == 2
        ), "Cannot add empirical spectrum after adding theoretical spectra"

        self.empirical_spectrum = spectrum

        for idx, peak_intensity in enumerate(spectrum.intensities):
            node = EmpiricalNode(
                id=len(self.nodes), peak_idx=idx, intensity=peak_intensity
            )
            self.nodes.append(node)
            self.empirical_spectrum_corresponding_nodes.append(node)
            self.graph.add_node(node)


    def add_theoretical_spectrum(self, spectrum, dist_fun, max_dist):
        assert (
            self.empirical_spectrum is not None
        ), "Cannot add theoretical spectrum before empirical spectrum"
        assert (
            not self.built
        ), "Cannot add theoretical spectrum after building the graph"

        self.theoretical_spectra.append(spectrum)

        for idx in range(len(spectrum.intensities)):
            theo_node = TheoreticalNode(
                id=len(self.nodes),
                peak_idx=idx,
                spectrum_id=len(self.theoretical_spectra) - 1,
                intensity=spectrum.intensities[idx],
            )
            self.nodes.append(theo_node)
            self.graph.add_node(theo_node)

            dists = np.int64(
                dist_fun(
                    spectrum.positions[:, idx : idx + 1][: np.newaxis],
                    self.empirical_spectrum.positions,
                )
            )
            print("dist", dists)
            emp_indexes = np.where(dists < max_dist)[0]
            print("emp_indexes", emp_indexes)

            for emp_idx in emp_indexes:
                edge = MatchingEdge(
                    id=len(self.edges),
                    start_node=self.empirical_spectrum_corresponding_nodes[idx],
                    end_node=theo_node,
                    emp_peak_idx=emp_idx,
                    theo_spectrum_id=len(self.theoretical_spectra) - 1,
                    theo_peak_idx=idx,
                )
                self.edges.append(edge)
                self.graph.add_edge(
                    self.empirical_spectrum_corresponding_nodes[emp_idx],
                    theo_node,
                    obj=edge,
                )

    def build(self):
        self.built = True
        self.fragment_graphs = []

        dead_end_nodes = [
            node for node, degree in dict(self.graph.degree()).items() if degree < 1
        ]
        self.graph.remove_nodes_from(dead_end_nodes)

        for subgraph in nx.weakly_connected_components(self.graph):
            print("Subgraph")
            self.fragment_graphs.append(FlowSubgraph(self.graph.subgraph(subgraph), self))

        for subgraph in self.fragment_graphs:
            subgraph.build()

    def set_point(self, point):
        assert len(point) == len(self.theoretical_spectra)
        self.point = point
        self.total_cost = 0
        for subgraph in self.fragment_graphs:
            self.total_cost += subgraph.set_point(point)
        return self.total_cost

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
                    edge = SrcToEmpEdge(len(self.edges), intensity)
                    self.edges.append(edge)
                case TheoreticalNode(id, peak_idx, spectrum_id, intensity):
                    edge = TheoToSinkEdge(len(self.edges), spectrum_id, intensity)
                    self.edges.append(edge)
                case _:
                    raise ValueError(f"Unexpected node type: {type(node)} (this shouldn't happen)")

        sdfvsdv
        #for edge_start, edge_end, edge_obj in nx_graph.edges(data=True):

        self.add_simple_trash(10)

    def build(self):
        self.order = np.lexsort((self.edge_ends, self.edge_starts))
        self.edge_starts = np.asarray(self.edge_starts)[self.order]
        self.edge_ends = np.asarray(self.edge_ends)[self.order]
        self.edge_costs = np.asarray(self.edge_costs)[self.order]
        self.edge_capacities = np.asarray(self.edge_capacities)[self.order]
        self.edges = [self.edges[i] for i in self.order]

        self.lemon_graph = GraphWrapper(  # pylmcf_cpp.LemonGraph(
            len(self.node_ids), self.edge_starts, self.edge_ends, self.edge_costs
        )

    def add_simple_trash(self, cost):
        self.edge_starts.append(self.source)
        self.edge_ends.append(self.sink)
        self.edge_costs.append(cost)
        self.edge_capacities.append(0)
        self.edges.append(SimpleTrashEdge())
        print("Trash:", self.edge_starts)


    @cached_property
    def nx_graph(self):
        bnx_graph = nx.DiGraph()
        for node_id, layer in zip(self.node_ids, self.node_layers):
            bnx_graph.add_node(node_id, layer=layer)
        for i in range(len(self.edge_starts)):
            bnx_graph.add_edge(
                self.node_ids[self.edge_starts[i]],
                self.node_ids[self.edge_ends[i]],
                cost=self.edge_costs[i],
            )
        return bnx_graph

    def show(self):
        from matplotlib import pyplot as plt

        pos = nx.multipartite_layout(self.nx_graph, subset_key="layer")
        edge_labels = nx.get_edge_attributes(self.nx_graph, "cost")
        nx.draw(self.nx_graph, with_labels=True, pos=pos)
        nx.draw_networkx_edge_labels(self.nx_graph, pos=pos, edge_labels=edge_labels)
        plt.show()

    def set_point(self, point):
        self.total_theoretical_intensity = 0
        trash_edge_idx = None
        for edge_idx, edge in enumerate(self.edges):
            match edge:
                case TheoToSinkEdge(theo_spectrum_id, theo_peak_idx):
                    new_cap = point[theo_spectrum_id] * edge.theo_peak_intensity
                    self.total_theoretical_intensity += new_cap
                    self.edge_capacities[edge_idx] = new_cap
                case SimpleTrashEdge():
                    trash_edge_idx = edge_idx
                case _:
                    pass
        self.total_et_intensity = (
            self.total_empirical_intensity + self.total_theoretical_intensity
        )
        print("ET:", self.total_et_intensity)
        print(self.edge_capacities)
        self.node_supply[self.source] = self.total_et_intensity
        self.node_supply[self.sink] = -self.total_et_intensity

        self.edge_capacities[trash_edge_idx] = self.total_et_intensity

        print(np.array(self.edge_capacities))
        self.lemon_graph.set_edge_capacities(self.edge_capacities)
        self.lemon_graph.set_node_supply(self.node_supply)
        self.lemon_graph.solve()
        self.flows = self.lemon_graph.result()
        self.total_cost = self.lemon_graph.total_cost()
        self.lemon_graph.plot()
        return self.total_cost


class FlowGraph:
    def __init__(self):
        self.no_nodes = 0
        self.edge_starts = array("q")
        self.edge_ends = array("q")
        self.edge_costs = array("q")
        self.edge_ids = array("q")
        self.order = None
        self.flows = None
        # self.source = self.add_nodes(1)[0]
        # self.sink = self.add_nodes(1)[0]

    def add_nodes(self, no_nodes):
        assert self.order is None, "Cannot add nodes after building the graph"
        ret = self.no_nodes
        self.no_nodes += no_nodes
        ret = np.arange(ret, ret + no_nodes, dtype=np.int64)
        # print("Added nodes", ret)
        return ret

    def add_edges(self, starts, ends, costs):
        assert self.order is None, "Cannot add edges after building the graph"
        new_ids = range(len(self.edge_starts), len(self.edge_starts) + len(starts))
        self.edge_ids.extend(new_ids)
        self.edge_starts.extend(starts)
        self.edge_ends.extend(ends)
        self.edge_costs.extend(costs)
        ret = np.array(new_ids)
        # for id, start, end, cost in zip(new_ids, starts, ends, costs):
        #     print("Added edge", "id:", id, "start:", start, "end:", end, "cost:", cost)
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
        # print("Setting edge capacities", self.edge_id_to_index[edge_ids], capacities)
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
        G = self.to_networkx()
        dead_end_nodes = [
            node for node, degree in dict(G.degree()).items() if degree <= 1
        ]
        G.remove_nodes_from(dead_end_nodes)
        components_split = G.subgraph(range(2, self.no_nodes))
        components = nx.weakly_connected_components(components_split)
        subgraphs = [G.subgraph([0, 1] + list(c)) for c in components]
        print(len(subgraphs))
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
