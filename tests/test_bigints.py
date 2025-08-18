from pylmcf_cpp import CDecompositableFlowGraph
from pylmcf.spectrum import Spectrum_1D
import numpy as np


def test_scale():
    for exponent in range(0, 18):
        scale_factor = 10 ** exponent
        almost_max_int = 2**62
        empirical_spectrum = Spectrum_1D(np.array([1]), np.array([1])).scaled(scale_factor)
        theoretical_spectrum = Spectrum_1D(np.array([2]), np.array([1])).scaled(scale_factor)
        print(empirical_spectrum)
        dist_fun = lambda x, y: np.linalg.norm(x - y, axis=0)
        max_distance = 10
        def wrapped_dist(p, y):
            i = p[1]
            x = p[0][:, i : i + 1]
            return dist_fun(x[: np.newaxis], y)

        DG = CDecompositableFlowGraph(
            empirical_spectrum.cspectrum, [theoretical_spectrum.cspectrum], wrapped_dist, max_distance
        )
        DG.add_simple_trash(10)
        DG.build()
        point = [1.0] * len([theoretical_spectrum])
        print(DG.total_cost())
        #assert DG.total_cost() == scale_factor

if __name__ == "__main__":
    test_scale()
    print("Everything passed")

