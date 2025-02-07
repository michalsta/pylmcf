import pylmcf_cpp
import numpy as np


INF = np.int64(2**30)
I = INF


node_supply     = np.array([10, 0, 0, 0, 0, -10], dtype=np.int64)
edge_starts     = np.array([0, 0, 1, 1, 2, 2, 3, 4], dtype=np.int64)
edge_ends       = np.array([1, 2, 3, 4, 3, 4, 5, 5], dtype=np.int64)
edge_capacities = np.array([3, 7, I, I, I, I, 5, 5], dtype=np.int64)
edge_costs      = np.array([0, 0, 2, 4, 5, 4, 0, 0], dtype=np.int64)
result          = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)

res = pylmcf_cpp.lmcf(node_supply, edge_starts, edge_ends, edge_capacities, edge_costs)
'''


node_supply     = np.array([10, -10], dtype=np.int64)
edge_starts     = np.array([0], dtype=np.int64)
edge_ends       = np.array([1], dtype=np.int64)
edge_capacities = np.array([7], dtype=np.int64)
edge_costs      = np.array([3], dtype=np.int64)
res = pylmcf_cpp.lmcf(node_supply, edge_starts, edge_ends, edge_capacities, edge_costs)
'''
print(res)
