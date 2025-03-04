from pylmcf import Spectrum, WassersteinSolver, Spectrum_1D, EmpiricalTrash, TheoryTrash

def test_dual_trash():
    S1 = Spectrum_1D([0, 10, 20, 30], [4, 4, 5, 5])
    S2 = Spectrum_1D([0, 11], [8, 8])
    S3 = Spectrum_1D([20, 21, 30], [4, 6, 10])

    solver = WassersteinSolver(S1, [S2, S3], [EmpiricalTrash(1), TheoryTrash(1)], intensity_scaling=100000, costs_scaling=1000000)
    print(solver.run())
    solver.WN.G.show()

if __name__ == '__main__':
    test_dual_trash()