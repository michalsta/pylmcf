from pylmcf.wasserstein import WassersteinSolver
from pylmcf.spectrum import Spectrum, Spectrum_1D
from pylmcf.trashes import SimpleTrash, TrashFactorySimple
from pylmcf.graph import DecompositableFlowGraph
from pylmcf.solver import DeconvolutionSolver
import numpy as np


def test_1d():
    E = Spectrum_1D([1], [1])
    T = Spectrum_1D([2], [1])
    solver = WassersteinSolver(E, [T], [SimpleTrash(10)])
    assert solver.run() == 1

def test_2d():
    s1_pos = np.array([[0, 1, 0], [0, 0, 1]])
    s1_int = np.array([1, 1, 1])
    s1 = Spectrum(s1_pos, s1_int)
    s2_pos = np.array([[1, 1, 0], [1, 0, 1]])
    s2_int = np.array([1, 1, 1])
    s2 = Spectrum(s2_pos, s2_int)
    solver = WassersteinSolver(
        s1,
        [s2],
        [SimpleTrash(1000000)],
        costs_scaling=1000
    )
    #print(solver.run())
    assert solver.run() == 1.414

    # new algo
    DG = DecompositableFlowGraph(s1, [s2], lambda x, y: 1000*np.linalg.norm(x - y, axis=0), 5000)
    DG.build([TrashFactorySimple(1000000)])

    #print(DG.set_point([1]))
    assert DG.set_point([1]) == 1414

    # new solver
    solver = DeconvolutionSolver(
        s1,
        [s2],
        distance_function=lambda x, y: np.linalg.norm(x - y, axis=0),
        max_distance=5,
        trash_cost=1000000,
        scale_factor=1000
    )
    ret = solver.set_point([1])
    assert ret == 1.414


if __name__ == "__main__":
    test_1d()
    test_2d()
    print("Everything passed")