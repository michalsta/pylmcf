from pylmcf.distribution import Distribution_1D
from pylmcf.solver import Solver
import numpy as np
import math

def test_matching():
    s1 = Distribution_1D([0], [10])
    s2 = Distribution_1D([1, 2], [5, 5])
    WN = Solver(s1, [s2], lambda x, y: np.linalg.norm(x - y, axis=0), 100, 10, 10000)
    WN.set_point([1])
    assert math.isclose(WN.total_cost(), 15.0)


def test_matching2():
    s1 = Distribution_1D([0], [10])
    s2 = Distribution_1D([1], [4])
    s3 = Distribution_1D([2], [6])
    WN = Solver(
        s1, [s2, s3], lambda x, y: np.linalg.norm(x - y, axis=0), 100, 10, 10000
    )
    WN.set_point([1, 1])
    assert math.isclose(WN.total_cost(), 16.0)


def test_matching3():
    s1 = Distribution_1D([0], [10])
    s2 = Distribution_1D([1], [4])
    s3 = Distribution_1D([200], [6])
    WN = Solver(
        s1, [s2, s3], lambda x, y: np.linalg.norm(x - y, axis=0), 10, 10, 100
    )
    WN.set_point([1, 1])
    WN.print_diagnostics()
    print(WN.flows())
    print("aaa",WN.total_cost())
    assert math.isclose(WN.total_cost(), 64.0)

if __name__ == "__main__":
    test_matching()
    test_matching2()
    test_matching3()
    print("Everything passed")
