from pylmcf.wasserstein import *
from pylmcf.graph import Graph
from pylmcf.spectrum import Spectrum_1D


def test_matching():
    s1 = Spectrum_1D([0], [10])
    s2 = Spectrum_1D([1, 2], [5, 5])
    WN = WassersteinNetwork(s1, [s2], 10, lambda x, y: np.linalg.norm(x - y))
    WN.solve([1])
    # WN.G.print()
    assert WN.G.total_cost == 15


test_matching()


def test_matching2():
    s1 = Spectrum_1D([0], [10])
    s2 = Spectrum_1D([1], [4])
    s3 = Spectrum_1D([2], [6])
    WN = WassersteinNetwork(s1, [s2, s3], 10, lambda x, y: np.linalg.norm(x - y))
    WN.solve([1, 1])
    # WN.G.print()
    assert WN.G.total_cost == 16


test_matching2()


def test_matching3():
    s1 = Spectrum_1D([0], [10])
    s2 = Spectrum_1D([1], [4])
    s3 = Spectrum_1D([200], [6])
    WN = WassersteinNetwork(s1, [s2, s3], 10, lambda x, y: np.linalg.norm(x - y))
    WN.solve([1, 1])
    # WN.G.print()
    assert WN.G.total_cost == 64


#     assert WN

test_matching3()
