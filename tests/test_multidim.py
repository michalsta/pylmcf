from pylmcf.wasserstein import WassersteinSolver, Spectrum


def test_1d():
    E = Spectrum([1], [1])
    T = Spectrum([1], [1])
    solver = WassersteinSolver(E, [T], 10)
    solver.run()


test_1d()