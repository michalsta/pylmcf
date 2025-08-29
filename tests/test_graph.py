from pylmcf.pylmcf_cpp import LemonGraph
import numpy as np


def test_graph_simple():
    G = LemonGraph(3, np.array([0, 0, 1]), np.array([1, 2, 2]), np.array([1, 3, 5]))
    G.set_edge_capacities(np.array([1, 2, 3]))
    G.set_node_supply(np.array([5, 0, -5]))
    G.solve()
    assert all(G.result() == np.array([1, 2, 1]))
    assert G.total_cost() == 12
