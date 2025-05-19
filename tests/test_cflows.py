import numpy as np


from pylmcf.spectrum import Spectrum
from pylmcf.solver import Solver


def test_flows():
    S1 = Spectrum(np.array([[1, 2, 30]]), np.array([1, 4, 3]))
    S2 = Spectrum(np.array([[1, 4, 30, 31]]), np.array([5, 1, 1, 1]))

    dist_fun = lambda x, y: np.linalg.norm(x - y, axis=0)
    trash_cost = 10
    max_distance = 100
    solver = Solver(
        empirical_spectrum=S1,
        theoretical_spectra=[S2],
        distance_function=dist_fun,
        max_distance=max_distance,
        trash_cost=trash_cost,
        scale_factor=10000,
    )

    solver.set_point([1])

    # solver.print()

    print("Flows:")
    for flow in solver.flows():
        print(flow)


if __name__ == "__main__":
    test_flows()
    print("Everything passed")
