from pylmcf import Spectrum, WassersteinSolver, Spectrum_1D, EmpiricalTrash, TheoryTrash

def test_dual_trash1():
    S1 = Spectrum_1D([0, 10, 20, 30], [4, 4, 5, 5])
    S2 = Spectrum_1D([0, 11], [8, 8])
    S3 = Spectrum_1D([20, 21, 30], [4, 6, 10])

    solver = WassersteinSolver(S1, [S2, S3], [EmpiricalTrash(100000), TheoryTrash(10000)], intensity_scaling=1, costs_scaling=1)
    print(solver.run())
    from pprint import pprint
    pprint(solver.result())
    solver.WN.G.print()

def test_dual_trash2():
    S1 = Spectrum_1D([0], [2])
    S2 = Spectrum_1D([1], [1])
    S3 = Spectrum_1D([100], [2])

    solver = WassersteinSolver(S1, [S2, S3], [EmpiricalTrash(50), TheoryTrash(50)], intensity_scaling=1, costs_scaling=1)
    print(solver.run())
    from pprint import pprint
    pprint(solver.result())
    solver.WN.G.print()

if __name__ == '__main__':
    test_dual_trash2()