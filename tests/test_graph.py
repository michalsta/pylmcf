from pylmcf_cpp import Graph
import numpy as np

G = Graph(3, np.array([0, 0, 1]), np.array([1, 2, 2]), np.array([1, 3, 5]))
print("AAA")
G.set_edge_capacities(np.array([1, 2, 3]))
print("BBB")
G