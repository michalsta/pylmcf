from pylmcf.graph import DecompositableFlowGraph
from pylmcf.trashes import (
    TrashFactory,
    TrashFactorySimple,
    TrashFactoryEmpirical,
    TrashFactoryTheory,
)
from pylmcf.spectrum import Distribution
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


class Solver:
    def __init__(
        self,
        empirical_spectrum,
        theoretical_spectra,
        distance_function,
        max_distance,
        trash_cost,
        scale_factor=None,
    ):
        assert isinstance(empirical_spectrum, Distribution)
        assert isinstance(theoretical_spectra, list)
        assert all(isinstance(t, Distribution) for t in theoretical_spectra)
        assert callable(distance_function)
        assert isinstance(max_distance, (int, float))
        assert isinstance(trash_cost, (int, float))
        assert scale_factor is None or isinstance(scale_factor, (int, float))

        if scale_factor is None:
            ALMOST_MAXINT = 2**60
            empirical_sum_intensity = empirical_spectrum.sum_intensities
            theoretical_sum_intensity = sum(
                t.sum_intensities for t in theoretical_spectra
            )
            max_sum_intensity = max(
                empirical_sum_intensity, theoretical_sum_intensity
            )
            scale_factor = np.sqrt(ALMOST_MAXINT / (max_sum_intensity * trash_cost))
            assert scale_factor > 0, "Can't auto-compute a sensible scale factor. You might have some luck with setting it manually, but it probably means something about your data or trash_cost is off."

        self.scale_factor = scale_factor
        self.empirical_spectrum = empirical_spectrum.scaled(scale_factor)
        self.theoretical_spectra = [t.scaled(scale_factor) for t in theoretical_spectra]

        def wrapped_dist(p, y):
            i = p[1]
            x = p[0][:, i : i + 1]
            return distance_function(x[: np.newaxis], y) * scale_factor

        self.graph = pylmcf_cpp.CDecompositableFlowGraph(
            self.empirical_spectrum.cspectrum,
            [ts.cspectrum for ts in self.theoretical_spectra],
            wrapped_dist,
            int(max_distance * scale_factor),
        )
        self.graph.add_simple_trash(int(trash_cost * scale_factor))
        self.graph.build()
        self.point = None

    def set_point(self, point):
        self.point = point
        self.graph.set_point(point)

    def total_cost(self):
        return self.graph.total_cost() / self.scale_factor / self.scale_factor

    def print(self):
        print(str(self.graph))

    def flows(self):
        result = []
        for i in range(len(self.theoretical_spectra)):
            empirical_peak_idx, theoretical_peak_idx, flow = (
                self.graph.flows_for_spectrum(i)
            )
            result.append(
                namedtuple(
                    "Flow", ["empirical_peak_idx", "theoretical_peak_idx", "flow"]
                )(empirical_peak_idx, theoretical_peak_idx, flow / self.scale_factor)
            )
        return result

    def solve(self, start_point=None, debug_prints=False):
        def opt_fun(point):
            self.graph.set_point(point)
            ret = self.graph.total_cost()
            if debug_prints:
                print(int(np.log10(ret + 1)), ret)
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

    def no_subgraphs(self):
        return self.graph.no_subgraphs()

    def print_diagnostics(self, subgraphs_too=False):
        print("Diagnostics:")
        print("No subgraphs:", self.graph.no_subgraphs())
        print("No empirical nodes:", self.graph.count_empirical_nodes())
        print("No theoretical nodes:", self.graph.count_theoretical_nodes())
        print("Matching density:", self.graph.matching_density())
        print("Scale factor:", self.scale_factor, f" log10: {np.log10(self.scale_factor)}")
        print("Total cost:", self.graph.total_cost())
        if not subgraphs_too:
            return
        for ii in range(self.graph.no_subgraphs()):
            s = self.graph.get_subgraph(ii)
            print("Subgraph", ii, ":")
            print("  No. empirical nodes:", s.count_empirical_nodes())
            print("  No. theoretical nodes:", s.count_theoretical_nodes())
            print("  Cost:", s.total_cost())
            print("  Matching density:", s.matching_density())
            print("  Theoretical spectra involved:", s.theoretical_spectra_involved())
