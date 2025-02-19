from pylmcf_cpp import Graph
import numpy as np

G = Graph(3, np.array([0, 0, 1]), np.array([1, 2, 2]), np.array([1, 3, 5]))
G.set_edge_capacities(np.array([1, 2, 3]))
G.set_node_supply(np.array([5, 0, -5]))
G.solve()
print(G.result(), G.total_cost())
