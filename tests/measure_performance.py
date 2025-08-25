import numpy as np
from pylmcf import Distribution
from pylmcf import WassersteinSolver
import cProfile

np.random.seed(23456)
S1 = Distribution(np.random.rand(2, 1000) * 100, np.random.rand(1000))
S2 = Distribution(np.random.rand(2, 1000) * 100, np.random.rand(1000))
S3 = Distribution(np.random.rand(2, 1000) * 100, np.random.rand(1000))
S4 = Distribution(np.random.rand(2, 1000) * 100, np.random.rand(1000))
S5 = Distribution(np.random.rand(2, 1000) * 100, np.random.rand(1000))
S6 = Distribution(np.random.rand(2, 1000) * 100, np.random.rand(1000))


solver = WassersteinSolver(
    S1, [S2, S3, S4, S5, S6], trash_cost=1, intensity_scaling=10000, costs_scaling=10000
)


# print(solver.run())
def performance_test():
    print(solver.run())


cProfile.run("performance_test()", "perf_test.prof")
