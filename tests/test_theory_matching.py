from pylmcf.wasserstein import *
from pylmcf.graph import Graph
from pylmcf.spectrum import Spectrum

def test_matching():
    s1 = Spectrum([0], [10])
    s2 = Spectrum([1, 2], [5, 5])
    WN = WassersteinNetwork(s1, [s2], 10)
    WN.solve([1])
    WN.G.print()
    assert WN.G.total_cost == 15

test_matching()

def test_matching2():
    s1 = Spectrum([0], [10])
    s2 = Spectrum([1], [4])
    s3 = Spectrum([2], [6])
    WN = WassersteinNetwork(s1, [s2, s3], 10)
    WN.solve([1, 1])
    WN.G.print()
    assert WN.G.total_cost == 16

test_matching2()

def test_matching3():
    s1 = Spectrum([0], [10])
    s2 = Spectrum([1], [4])
    s3 = Spectrum([200], [6])
    WN = WassersteinNetwork(s1, [s2, s3], 10)
    WN.solve([1, 1])
    WN.G.print()
    print(WN.G.total_cost)
#     assert WN

test_matching3()