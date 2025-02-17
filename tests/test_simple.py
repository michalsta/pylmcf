import pylmcf_cpp
from pylmcf import *
import numpy as np


INF = np.int64(2**30)
I = INF

'''
node_supply     = np.array([10, 0, 0, 0, 0, -10], dtype=np.int64)
edge_starts     = np.array([0, 0, 1, 1, 2, 2, 3, 4], dtype=np.int64)
edge_ends       = np.array([1, 2, 3, 4, 3, 4, 5, 5], dtype=np.int64)
edge_capacities = np.array([3, 7, I, I, I, I, 5, 5], dtype=np.int64)
edge_costs      = np.array([0, 0, 2, 4, 5, 4, 0, 0], dtype=np.int64)
result          = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)

res = pylmcf_cpp.lmcf(node_supply, edge_starts, edge_ends, edge_capacities, edge_costs)



node_supply     = np.array([10, -10], dtype=np.int64)
edge_starts     = np.array([0], dtype=np.int64)
edge_ends       = np.array([1], dtype=np.int64)
edge_capacities = np.array([7], dtype=np.int64)
edge_costs      = np.array([3], dtype=np.int64)
res = pylmcf_cpp.lmcf(node_supply, edge_starts, edge_ends, edge_capacities, edge_costs)
'''

# X1 = np.array([0, 10], dtype=np.double)
# Y1 = np.array([0, 5], dtype=np.double)
# X2 = np.array([0, 7], dtype=np.double)
# Y2 = np.array([1, 4], dtype=np.double)
# intensities1 = np.array([3, 7], dtype=np.int64)
# intensities2 = np.array([5, 5], dtype=np.int64)
# trash_cost = np.uint64(1000)

# X1 = np.array([0], dtype=np.double)
# Y1 = np.array([0], dtype=np.double)
# X2 = np.array([0], dtype=np.double)
# Y2 = np.array([1], dtype=np.double)
# intensities1 = np.array([5], dtype=np.uint64)
# intensities2 = np.array([5], dtype=np.uint64)
# trash_cost = np.uint64(1000)

SIZE = 100000
RANGE = 100000

# np.random.seed(345)
# X1 = np.random.randint(0, RANGE, SIZE)
# Y1 = np.random.randint(0, RANGE, SIZE)
# X2 = np.random.randint(0, RANGE, SIZE)
# Y2 = np.random.randint(0, RANGE, SIZE)
# intensities1 = np.random.randint(1, RANGE, SIZE)
# intensities2 = np.random.randint(1, RANGE, SIZE)
# trash_cost = np.uint64(10)

#print(type(res[0]), res[0], res[0].dtype)
#print(type(res[1]), res[1], res[1].dtype)
#print(type(res[2]), res[2], res[2].dtype)
#print(type(res[3]), res[3], res[3].dtype)
#print(type(res[4]), res[4], res[4].dtype)


X1 = np.array([0, 100], dtype=np.double)
Y1 = np.array([0, 100], dtype=np.double)
X2 = np.array([0], dtype=np.double)
Y2 = np.array([1], dtype=np.double)
intensities1 = np.array([5, 1], dtype=np.uint64)
intensities2 = np.array([6], dtype=np.uint64)
trash_cost = np.uint64(10)
res = wasserstein(X1, Y1, intensities1, X2, Y2, intensities2, trash_cost)

print(res)