import pylmcf
from pylmcf.trashes import SimpleTrash


def test_proportions():
    T = pylmcf.Spectrum_1D([0, 10, 20, 30], [4, 4, 5, 5])
    S1 = pylmcf.Spectrum_1D([0, 11], [1, 2])
    S2 = pylmcf.Spectrum_1D([20, 21, 30], [2, 3, 5])

    solver = pylmcf.WassersteinSolver(
        T, [S1, S2], [SimpleTrash(10)], intensity_scaling=100000, costs_scaling=1000000
    )
    print(solver.run([3, 2]))
    # solver.WN.G.show()
    # solver.WN.print_summary()
    print(solver.estimate_proportions())


if __name__ == "__main__":
    test_proportions()
