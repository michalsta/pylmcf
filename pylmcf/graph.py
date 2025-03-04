from pylmcf import pylmcf_cpp
from array import array
import numpy as np
import networkx as nx

class Graph:
    def __init__(self):
        self.no_nodes = 0
        self.edge_starts = array("q")
        self.edge_ends = array("q")
        self.edge_costs = array("q")
        self.edge_ids = array("q")
        self.order = None
        self.flows = None

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

    def to_networkx(self):
        G = nx.DiGraph()
        for i in range(len(self.edge_starts)):
            G.add_edge(
                self.edge_starts[i],
                self.edge_ends[i],
                capacity=self.edge_capacities[i],
                flow=self.flows[i],
                label=f"ca: {int(self.edge_capacities[i])} co: {int(self.edge_costs[i])} f: {int(self.flows[i])}"
            )
        return G

    def show(self):
        from matplotlib import pyplot as plt
        nxg = self.to_networkx()
        edge_labels = nx.get_edge_attributes(nxg, 'label')
        ranks = {node: nx.shortest_path_length(nxg, 0, node) for node in nxg.nodes}
        ranks[1] = 4
        # Assign rank as 'layer' attribute
        nx.set_node_attributes(nxg, ranks, 'layer')

        # Use multipartite_layout based on 'layer'
        pos = nx.multipartite_layout(nxg, subset_key='layer')

        nx.draw(nxg, pos=pos, with_labels=True)
        nx.draw_networkx_edge_labels(nxg, pos=pos, edge_labels=edge_labels)
        plt.show()