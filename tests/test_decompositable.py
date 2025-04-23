from pylmcf.graph import DecompositableFlowGraph
from pylmcf.spectrum import Spectrum
from pylmcf.trashes import TrashFactorySimple
import numpy as np


def test_simple():
    S1 = Spectrum(np.array([[1, 2, 30]]), np.array([1, 4, 3]))
    S2 = Spectrum(np.array([[1, 4, 30, 31]]), np.array([1, 4, 1, 2]))

    G = DecompositableFlowGraph()
    G.add_empirical_spectrum(S1)
    G.add_theoretical_spectrum(S2, lambda x, y: np.linalg.norm(x - y, axis=0), 5)
    G.build([TrashFactorySimple(10)])
    #G.show()
    print(G.set_point([1]))
    for fr_gr in G.fragment_graphs:
        fr_gr.show()


if __name__ == "__main__":
    test_simple()