from pylmcf.graph import DecompositableFlowGraph
from pylmcf.trashes import (
    TrashFactory,
    TrashFactorySimple,
    TrashFactoryEmpirical,
    TrashFactoryTheory,
)
from pylmcf.distribution import Distribution
import pylmcf_cpp
from tqdm import tqdm
from scipy.optimize import minimize
import numpy as np
from collections import namedtuple


class DeconvolutionSolver:
    def __init__(
        self,
        empirical_spectrum,
        theoretical_spectra,
        distance_function,
        max_distance,
        trash_cost,
        scale_factor=1000000.0,
    ):
        self.scale_factor = scale_factor
        self.empirical_spectrum = empirical_spectrum.scaled(scale_factor)
        self.theoretical_spectra = [t.scaled(scale_factor) for t in theoretical_spectra]
        dist_fun = lambda x, y: distance_function(x, y) * scale_factor
        self.DG = DecompositableFlowGraph(
            self.empirical_spectrum,
            self.theoretical_spectra,
            dist_fun,
            max_distance * scale_factor,
        )
        return self.DG.build([TrashFactorySimple(trash_cost * scale_factor)])

    def set_point(self, point):
        return self.DG.set_point(point) / self.scale_factor / self.scale_factor

    def solve(self, start_point=None, debug_prints=False):
        def opt_fun(point):
            ret = self.DG.set_point(point)
            if debug_prints:
                print(int(np.log10(ret)), ret)
            return ret

        if start_point is None:
            start_point = [1.0] * len(self.theoretical_spectra)
        start_point = self.scale_factor * np.array(start_point)

        return minimize(
            opt_fun,
            method="Nelder-Mead",
            x0=start_point,
            bounds=[(0, None)] * len(self.theoretical_spectra),
            options={"disp": True, "maxiter": 100000},
        )

