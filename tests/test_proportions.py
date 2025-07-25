import pylmcf
import numpy as np
from pylmcf.solver import DeconvolutionSolver

def test_proportions():
    T = pylmcf.Spectrum_1D([0, 10, 20, 30], [4, 4, 5, 5])
    S1 = pylmcf.Spectrum_1D([0, 11], [1, 2])
    S2 = pylmcf.Spectrum_1D([20, 21, 30], [2, 3, 5])

    solver = DeconvolutionSolver(
        T, [S1, S2], lambda x, y: np.linalg.norm(x - y, axis=0), 100, 10, scale_factor=1000
    )
    #solver = pylmcf.WassersteinSolver(
    #    T, [S1, S2], [SimpleTrash(10)], intensity_scaling=100000, costs_scaling=1000000
    #)
    print(solver.set_point([3, 2]))
    # solver.WN.G.show()
    # solver.WN.print_summary()
    print(solver.solve())


if __name__ == "__main__":
    test_proportions()
