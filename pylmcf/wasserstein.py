import numpy as np
from numba import njit
from .graph import FlowGraph, DecompositableFlowGraph
from .spectrum import Spectrum
from .trashes import *
import time
import networkx as nx

# from .numba_helper import match_nodes
import pylmcf_cpp


# =========================== OBSOLETE CODE BELOW DO NOT USE ==========================


def XXX_solve(G):
    G.solve()


@njit
def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


@njit
def wasserstein_network(X1, Y1, intensities1, X2, Y2, intensities2, trash_cost):
    sum_intensities = np.sum(intensities1, dtype=np.int64)
    # Create a graph
    nodes_supply = np.zeros(len(intensities1) + len(intensities2) + 2, dtype=np.int64)
    SRC_IDX = 0
    SINK_IDX = len(nodes_supply) - 1
    MAX_COST = np.int64(2**30)
    SCALING_FACTOR = 1  # np.int64(MAX_COST / trash_cost)
    SCALED_TRASH_COST = np.int64(trash_cost * SCALING_FACTOR / 2.0)
    nodes_supply[SRC_IDX] = sum_intensities
    nodes_supply[SINK_IDX] = -sum_intensities

    edge_starts = []
    edge_ends = []
    edge_capacities = []
    edge_costs = []

    LAYER1_START_IDX = 1
    LAYER2_START_IDX = len(intensities1) + 1

    # The intensity-carrying edges:
    # Add edges from source to first layer
    for i in range(len(intensities1)):
        # Source to layer 1
        # print("Source to layer 1", SRC_IDX, LAYER1_START_IDX + i)
        edge_starts.append(SRC_IDX)
        edge_ends.append(LAYER1_START_IDX + i)
        edge_capacities.append(intensities1[i])
        edge_costs.append(np.int64(0))

    # Add edges from second layer to sink
    for i in range(len(intensities2)):
        # print("Layer 2 to sink", LAYER2_START_IDX + i, SINK_IDX)
        # Layer 2 to sink
        edge_starts.append(LAYER2_START_IDX + i)
        edge_ends.append(SINK_IDX)
        edge_capacities.append(intensities2[i])
        edge_costs.append(np.int64(0))

    # The matching edges:
    matches = 0
    for i in range(len(intensities1)):
        for j in range(len(intensities2)):
            # print("Layer 1 to layer 2: ", LAYER1_START_IDX + i, LAYER2_START_IDX + j)
            dist_val = dist(X1[i], Y1[i], X2[j], Y2[j])
            if dist_val < trash_cost:
                edge_starts.append(LAYER1_START_IDX + i)
                edge_ends.append(LAYER2_START_IDX + j)
                edge_capacities.append(min(intensities1[i], intensities2[j]))
                edge_costs.append(np.int64(SCALING_FACTOR * dist_val))
                matches += 1

    """
    # The trash edges:
    # Add edges from first layer to sink
    for i in range(len(intensities1)):
        # Layer 1 to sink
        print("Layer 1 to sink", LAYER1_START_IDX + i, SINK_IDX)
        edge_starts.append(LAYER1_START_IDX + i)
        edge_ends.append(SINK_IDX)
        edge_capacities.append(intensities1[i])
        edge_costs.append(SCALED_TRASH_COST)

    # Add edges from source to second layer
    for i in range(len(intensities2)):
        # Source to layer 2
        print("Source to layer 2", SRC_IDX, LAYER2_START_IDX + i)
        edge_starts.append(SRC_IDX)
        edge_ends.append(LAYER2_START_IDX + i)
        edge_capacities.append(intensities2[i])
        edge_costs.append(SCALED_TRASH_COST)
    """

    # The trash edges:
    nodes_supply = np.append(nodes_supply, 0)
    for i in range(len(intensities1)):
        # Layer 1 to trash
        # print("Layer 1 to trash", LAYER1_START_IDX + i, len(nodes_supply) - 1)
        edge_starts.append(LAYER1_START_IDX + i)
        edge_ends.append(len(nodes_supply) - 1)
        edge_capacities.append(intensities1[i])
        edge_costs.append(SCALED_TRASH_COST)

    for i in range(len(intensities2)):
        # Layer 2 to trash
        # print("Layer 2 to trash", len(nodes_supply) - 1, LAYER2_START_IDX + i)
        edge_starts.append(len(nodes_supply) - 1)
        edge_ends.append(LAYER2_START_IDX + i)
        edge_capacities.append(intensities2[i])
        edge_costs.append(SCALED_TRASH_COST)

    return (
        nodes_supply,
        np.asarray(edge_starts, dtype=np.int64),
        np.asarray(edge_ends, dtype=np.int64),
        np.asarray(edge_capacities, dtype=np.int64),
        np.asarray(edge_costs, dtype=np.int64),
    )


def wasserstein_integer(X1, Y1, intensities1, X2, Y2, intensities2, trash_cost):
    # assert all(np.issubdtype(x.dtype, np.integer) for x in [X1, Y1, intensities1, X2, Y2, intensities2, trash_cost]), "All arguments must be integer type"
    assert trash_cost % 2 == 0, "Trash cost must be even (divisible by 2)"
    nodes_supply, edge_starts, edge_ends, edge_capacities, edge_costs = (
        wasserstein_network(X1, Y1, intensities1, X2, Y2, intensities2, trash_cost)
    )
    flows = pylmcf_cpp.lmcf(
        nodes_supply, edge_starts, edge_ends, edge_capacities, edge_costs
    )
    src_trashed = flows[len(flows) - len(X1) - len(X2) : len(flows) - len(X2)]
    dst_trashed = flows[len(flows) - len(X2) :]
    sources = edge_starts[len(X1) + len(X2) : len(flows) - len(X1) - len(X2)] - 1
    sinks = edge_ends[len(X1) + len(X2) : len(flows) - len(X1) - len(X2)] - (
        1 + len(X1)
    )
    total_cost = np.sum(flows * edge_costs)
    flows = flows[len(X1) + len(X2) : len(flows) - len(X1) - len(X2)]
    mask = flows > 0

    return {
        "src_trashed": src_trashed.copy(),
        "dst_trashed": dst_trashed.copy(),
        "transport_source_idx": sources[mask],
        "transport_sink_idx": sinks[mask],
        "transport_flow": flows[mask],
        "total_cost": total_cost,
    }


def wasserstein_integer_new(X1, Y1, intensities1, X2, Y2, intensities2, trash_cost):
    assert trash_cost % 2 == 0, "Trash cost must be even (divisible by 2)"
    nodes_supply, edge_starts, edge_ends, edge_capacities, edge_costs = (
        wasserstein_network(X1, Y1, intensities1, X2, Y2, intensities2, trash_cost)
    )
    print("Wasserstein network: nodes_supply", nodes_supply)
    print("Wasserstein network: edge_starts     ", edge_starts)
    print("Wasserstein network: edge_ends       ", edge_ends)
    print("Wasserstein network: edge_capacities ", edge_capacities)
    print("Wasserstein network: edge_costs      ", edge_costs)
    G = pylmcf_cpp.LemonGraph(len(nodes_supply), edge_starts, edge_ends, edge_costs)
    G.set_edge_capacities(edge_capacities)
    G.set_node_supply(nodes_supply)
    G.solve()
    flows = G.result()
    total_cost = G.total_cost()
    src_trashed = flows[len(flows) - len(X1) - len(X2) : len(flows) - len(X2)]
    dst_trashed = flows[len(flows) - len(X2) :]
    sources = edge_starts[len(X1) + len(X2) : len(flows) - len(X1) - len(X2)] - 1
    sinks = edge_ends[len(X1) + len(X2) : len(flows) - len(X1) - len(X2)] - (
        1 + len(X1)
    )
    total_cost = np.sum(flows * edge_costs)
    flows = flows[len(X1) + len(X2) : len(flows) - len(X1) - len(X2)]
    mask = flows > 0

    return {
        "src_trashed": src_trashed.copy(),
        "dst_trashed": dst_trashed.copy(),
        "transport_source_idx": sources[mask],
        "transport_sink_idx": sinks[mask],
        "transport_flow": flows[mask],
        "total_cost": total_cost,
    }


def wasserstein(X1, Y1, intensities1, X2, Y2, intensities2, trash_cost):
    return wasserstein_integer_new(
        X1,
        Y1,
        intensities1.astype(np.int64),
        X2,
        Y2,
        intensities2.astype(np.int64),
        trash_cost,
    )


class SingleTheoryMatching:
    def __init__(self, WNM, theoretical_spectrum, max_dist, dist_fun):
        # print("SingleTheoryMatching")
        self.theoretical_spectrum = theoretical_spectrum
        self.G = WNM.G
        self.WNM = WNM
        self.theoretical_nodes = self.G.add_nodes(len(theoretical_spectrum))
        self.node_id_to_idx = lambda x: x - self.theoretical_nodes[0]
        self.theory_to_sink_edges = self.G.add_edges(
            self.theoretical_nodes,
            np.full(len(self.theoretical_nodes), WNM.network.sink),
            np.zeros(len(self.theoretical_nodes), dtype=np.int64),
        )

        self.matching_edge_start_nodes = []  # empirical nodes
        self.matching_edge_end_nodes = []  # theoretical nodes
        distances = []

        for ii, theoretical_node in enumerate(self.theoretical_nodes):
            dists = dist_fun(
                theoretical_spectrum.positions[:, ii : ii + 1][: np.newaxis],
                WNM.empirical_spectrum.positions,
            )
            mask = dists < max_dist
            distances.extend(dists[mask])
            self.matching_edge_start_nodes.extend(WNM.empirical_node_ids[mask])
            self.matching_edge_end_nodes.extend(np.full(np.sum(mask), theoretical_node))

        self.matching_edge_start_nodes = np.array(self.matching_edge_start_nodes)
        self.matching_edge_end_nodes = np.array(self.matching_edge_end_nodes)

        self.matching_edge_ids = self.G.add_edges(
            self.matching_edge_start_nodes,
            self.matching_edge_end_nodes,
            np.array(distances, dtype=np.int64),
        )

        self._matching_flow = None

        #self.print_summary()

    def set_edge_capacities(self, scale_factor: float):
        scaled_intensities = self.theoretical_spectrum.intensities * scale_factor
        self.G.set_edge_capacities(self.theory_to_sink_edges, scaled_intensities)
        if len(self.matching_edge_ids) == 0:
            return
        empirical_layer_intensities = self.WNM.empirical_spectrum.intensities[
            self.WNM.empirical_node_id_to_idx(self.matching_edge_start_nodes)
        ]
        theoretical_layer_intensities = scaled_intensities[
            self.node_id_to_idx(self.matching_edge_end_nodes)
        ]

        self.scaled_matching_capacities = np.minimum(
            empirical_layer_intensities, theoretical_layer_intensities
        )
        self.G.set_edge_capacities(self.matching_edge_ids, self.scaled_matching_capacities)
        self._matching_flow = None


    def matching_flow(self):
        if self._matching_flow is None:
            self._matching_flow = self.G.flows[self.matching_edge_ids]
        return self._matching_flow

    @property
    def no_edges(self):
        return len(self.matching_edge_ids)

    def result(self):
        return {
            'empirical_indexes': self.WNM.empirical_node_id_to_idx(self.matching_edge_start_nodes),
            'theoretical_indexes': self.node_id_to_idx(self.matching_edge_end_nodes),
            'flows': self.matching_flow()
        }

    @property
    def no_possible_edges(self):
        return len(self.theoretical_nodes) * len(self.WNM.empirical_node_ids)

    @property
    def density(self):
        return self.no_edges / self.no_possible_edges

    def print_summary(self):
        # print("     Theoretical nodes", self.theoretical_nodes)
        # print("     Theoretical node intensities", self.theoretical_spectrum.intensities)
        # print("     Theory to sink edges", self.theory_to_sink_edges)
        # print("     Theory to sink edge capacities", self.theoretical_spectrum.intensities)
        # print("     Matching edges", self.matching_edge_ids)
        # print("     Matching edge capacities", self.G.edge_capacities[self.matching_edge_ids])
        # print("     Matching edge intensities", self.WNM.empirical_spectrum.intensities)
        # print("     Matching edge flows", self.G.flows[self.matching_edge_ids])
        print("     No theory nodes", len(self.theoretical_nodes))
        print("     No edges", self.no_edges)
        print("     No possible edges", self.no_possible_edges)
        print("     Density", self.density)


class WassersteinMatching:
    def __init__(
        self, WN, empirical_spectrum, theoretical_spectra, distance_limit, dist_fun
    ):
        self.empirical_spectrum = empirical_spectrum
        self.theoretical_spectra = theoretical_spectra
        self.distance_limit = distance_limit
        self.network = WN
        self.G = WN.G
        self.empirical_node_ids = self.G.add_nodes(len(empirical_spectrum))
        self.empirical_node_id_to_idx = lambda x: x - self.empirical_node_ids[0]
        self.theory_matchers = [
            SingleTheoryMatching(self, theoretical_spectrum, distance_limit, dist_fun)
            for theoretical_spectrum in theoretical_spectra
        ]

        self.source_to_empirical_edge_ids = self.G.add_edges(
            np.full(len(self.empirical_node_ids), self.network.source),
            self.empirical_node_ids,
            np.zeros(len(self.empirical_node_ids), dtype=np.int64),
        )
        self._source_to_empirical_flow = None

    def set_edge_capacities(self, scale_factors):
        assert len(scale_factors) == len(self.theory_matchers)
        for scale_factor, theory_matcher in zip(scale_factors, self.theory_matchers):
            theory_matcher.set_edge_capacities(scale_factor)
        # print("Setting edge capacities for source to empirical")
        # print("Empirical intensities", self.empirical_spectrum.intensities)
        self.G.set_edge_capacities(
            self.source_to_empirical_edge_ids, self.empirical_spectrum.intensities
        )

    def print_summary(self):
        print("Empirical nodes", self.empirical_node_ids)
        print("Empirical node intensities", self.empirical_spectrum.intensities)
        for id, theory_matcher in enumerate(self.theory_matchers):
            print("Single Theory matcher", id)
            theory_matcher.print_summary()

    def short_summary(self):
        return {
            "no_edges": self.no_edges,
            "no_possible_edges": self.no_possible_edges,
            "density": self.density,
            "total_cost": self.network.total_cost(),
            "solve_time": self.network.solve_time,
        }

    @property
    def source_to_empirical_flow(self):
        if self._source_to_empirical_flow is None:
            self._source_to_empirical_flow = self.G.flows[self.source_to_empirical_edge_ids]
        return self._source_to_empirical_flow

    @property
    def no_edges(self):
        return sum([tm.no_edges for tm in self.theory_matchers])

    @property
    def no_possible_edges(self):
        return sum([tm.no_possible_edges for tm in self.theory_matchers])

    @property
    def density(self):
        return self.no_edges / self.no_possible_edges

class WassersteinNetwork:
    def __init__(self, empirical_spectrum, theoretical_spectra, trashes, dist_fun, distance_limit=None):
        self.empirical_spectrum = empirical_spectrum
        self.theoretical_spectra = theoretical_spectra
        if distance_limit is None:
            distance_limit = max((t.distance_limit() for t in trashes), default=np.inf)
        G = DecompositableFlowGraph()
        self.G = G
        self.source = G.source
        self.sink = G.sink
        self.matching = WassersteinMatching(
            self, empirical_spectrum, theoretical_spectra, distance_limit, dist_fun
        )
        self.trashes = trashes
        for trash in trashes:
            trash.add_to_network(self)
        G.build()


    def solve(self, scale_factors):
        # print("WassersteinNetwork: solve")
        self.matching.set_edge_capacities(scale_factors)
        self.total_supply = max(
            np.sum(self.empirical_spectrum.intensities),
            np.sum([sf*np.sum(spectrum.intensities) for sf, spectrum in zip(scale_factors, self.theoretical_spectra)]),
        )
        for trash in self.trashes:
            trash.set_edge_capacities()
        # print(self.empirical_spectrum.intensities)
        # print([s.intensities for s in self.theoretical_spectra])
        self.G.set_node_supply(self.source, self.total_supply)
        self.G.set_node_supply(self.sink, -self.total_supply)
        start = time.perf_counter()
        self.G.solve()
        self.solve_time = time.perf_counter() - start
        # print("Lemon solved in", self.solve_time)
        # print(self.matching.short_summary())

    def print_summary(self):
        print("Source", self.source, "flow:", self.total_supply)
        print("Sink", self.sink, "flow:", self.total_supply)
        for trash in self.trashes:
            trash.print_summary()
        self.matching.print_summary()


    def total_cost(self):
        return self.G.total_cost

    def result(self):
        return {
            "theory_matchings": [tm.result() for tm in self.matching.theory_matchers],
            "empirical_flows": self.matching.source_to_empirical_flow,
            "total_cost": self.total_cost(),
        }


class WassersteinSolver:
    def __init__(
        self,
        empirical_spectrum,
        theoretical_spectra,
        trashes,
        dist_fun=lambda x, y: np.linalg.norm(x - y, axis=0),
        intensity_scaling=1_000,
        costs_scaling=1_000,
        distance_limit=None,
    ):
        self.intensity_scaling = intensity_scaling
        self.costs_scaling = costs_scaling
        if not isinstance(empirical_spectrum, Spectrum):
            empirical_spectrum = Spectrum.FromMasserstein(empirical_spectrum)
            theoretical_spectra = [
                Spectrum.FromMasserstein(s) for s in theoretical_spectra
            ]

        scaled_dist_fun = lambda x, y: np.int64(self.costs_scaling * dist_fun(x, y))

        self.empirical_spectrum = empirical_spectrum.scaled(self.intensity_scaling)
        # print("Empirical spectrum", self.empirical_spectrum)
        self.theoretical_spectra = [
            s.scaled(self.intensity_scaling) for s in theoretical_spectra
        ]
        # print("Theoretical spectra", self.theoretical_spectra)

        for trash in trashes:
            trash.apply_cost_scaling(self.costs_scaling)

        self.WN = WassersteinNetwork(
            self.empirical_spectrum,
            self.theoretical_spectra,
            trashes,
            scaled_dist_fun,
            distance_limit=None if distance_limit is None else distance_limit * self.costs_scaling,
        )

    def run(self, point=None):
        print("Running with point", point)
        start_time = time.perf_counter()
        if point is None:
            point = np.full(len(self.theoretical_spectra), 1.0)
        # point = point / np.sum(point)
        self.WN.solve(point)
        # print("Total cost", self.WN.total_cost())
        self.total_time = time.perf_counter() - start_time
        # print("Total time", self.total_time)
        print("Total cost", self.WN.total_cost() / self.intensity_scaling / self.costs_scaling)
        return self.WN.total_cost() / self.intensity_scaling / self.costs_scaling

    def result(self):
        res = self.WN.result()
        res["empirical_flows"] = res["empirical_flows"] / self.intensity_scaling
        for theory_matching in res["theory_matchings"]:
            theory_matching["flows"] = theory_matching["flows"] / self.intensity_scaling
        return res

    def estimate_proportions(self):
        target_function = lambda x: self.run(x)
        from scipy.optimize import minimize

        simplex = np.zeros((len(self.theoretical_spectra) + 1, len(self.theoretical_spectra)))
        for ii in range(len(self.theoretical_spectra)):
            simplex[ii + 1, ii] = 1.0
        res = minimize(
            target_function,
            np.full(len(self.theoretical_spectra), 0.5),
            bounds=[(0, None)] * len(self.theoretical_spectra),
            method="Nelder-Mead",
            options={'initial_simplex':simplex}
            #method="Powell",
            #method="SLSQP",
            #method="trust-constr",
            #method="L-BFGS-B",
            #method="COBYLA",
            #method="TNC",
        )
        print(f'NM cost: {res.fun}')
        return res.x

    #def estimate_proportions(self):
    #    start_point = np.full(len(self.theoretical_spectra), 1.0)


def wasserstein_integer_compat(X1, Y1, intensities1, X2, Y2, intensities2, trash_cost):
    S1 = Spectrum(np.array([X1, Y1]), intensities1)
    S2 = Spectrum(np.array([X2, Y2]), intensities2)
    solver = WassersteinSolver(S1, [S2], [SimpleTrash(trash_cost)], intensity_scaling=1, costs_scaling=1)
    return {'total_cost': solver.run()}
