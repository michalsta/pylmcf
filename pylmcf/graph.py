from pylmcf import pylmcf_cpp
import numpy as np


class Graph:
    def __init__(self):
        self.no_nodes = 0
        self.edge_starts = []
        self.edge_ends = []
        self.edge_costs = []
        self.edge_ids = []
        self.order = None
        self.flows = None
        self.node_labels = {}
        self.edge_labels = {}

    def add_nodes(self, no_nodes, labels=None):
        assert self.order is None, "Cannot add nodes after building the graph"
        ret = self.no_nodes
        self.no_nodes += no_nodes
        ret = np.arange(ret, ret + no_nodes, dtype=np.int64)
        # print("Added nodes", ret)
        if labels is not None:
            for i, label in zip(ret, labels):
                self.node_labels[i] = label
        return ret

    def add_edges(self, starts, ends, costs, labels=None):
        assert self.order is None, "Cannot add edges after building the graph"
        new_ids = range(len(self.edge_starts), len(self.edge_starts) + len(starts))
        self.edge_ids.extend(new_ids)
        self.edge_starts.extend(starts)
        self.edge_ends.extend(ends)
        self.edge_costs.extend(costs)
        ret = np.array(new_ids)
        # for id, start, end, cost in zip(new_ids, starts, ends, costs):
        #     print("Added edge", "id:", id, "start:", start, "end:", end, "cost:", cost)
        if labels is not None:
            for i, label in zip(ret, labels):
                self.edge_labels[i] = label
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
        self.edge_starts = np.array(self.edge_starts)[self.order]
        self.edge_ends = np.array(self.edge_ends)[self.order]
        self.edge_costs = np.array(self.edge_costs)[self.order]
        self.edge_ids = np.array(self.edge_ids)[self.order]
        self.edge_capacities = np.zeros(len(self.edge_starts), dtype=np.int64)
        self.node_supply = np.zeros(self.no_nodes, dtype=np.int64)
        self.edge_id_to_index = np.argsort(self.edge_ids)

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
        self.lemon_graph.solve()
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
