from pylmcf.graph import DecompositableFlowGraph
from pylmcf.trashes import TrashFactory, TrashFactorySimple, TrashFactoryEmpirical, TrashFactoryTheory
from pylmcf.spectrum import Spectrum
import pylmcf_cpp
from tqdm import tqdm
from scipy.optimize import minimize
import numpy as np
from collections import namedtuple

class DeconvolutionSolver:
    def __init__(self, empirical_spectrum, theoretical_spectra, distance_function, max_distance, trash_cost, scale_factor=1000000.0):
        self.scale_factor = scale_factor
        self.empirical_spectrum = empirical_spectrum.scaled(scale_factor)
        print("Empirical spectrum:", self.empirical_spectrum.intensities)
        self.theoretical_spectra = [t.scaled(scale_factor) for t in theoretical_spectra]
        print("Theoretical spectra:", [t.intensities for t in self.theoretical_spectra])
        dist_fun = lambda x, y: distance_function(x, y) * scale_factor
        self.DG = DecompositableFlowGraph(self.empirical_spectrum, self.theoretical_spectra, dist_fun, max_distance*scale_factor)

        #self.DG.build([TrashFactoryTheory(trash_cost*scale_factor), TrashFactoryEmpirical(trash_cost*scale_factor)])
        return self.DG.build([TrashFactorySimple(trash_cost*scale_factor)])

    def set_point(self, point):
        return self.DG.set_point(point) / self.scale_factor / self.scale_factor


    def solve(self, start_point = None):
        def opt_fun(point):
            ret = self.DG.set_point(point)
            #print("Optimizing with point:", point, "cost:", ret)
            print(int(np.log10(ret)), ret)
            return ret
        if start_point is None:
            start_point = [1.0] * len(self.theoretical_spectra)
        start_point = self.scale_factor * np.array(start_point)

        return minimize(opt_fun, method='Nelder-Mead', x0 = start_point, bounds=[(0, None)] * len(self.theoretical_spectra), options={'disp': True, 'maxiter':100000})




class Solver:
    def __init__(self, empirical_spectrum, theoretical_spectra, distance_function, max_distance, trash_cost, scale_factor=1000000.0):
        assert isinstance(empirical_spectrum, Spectrum)
        assert isinstance(theoretical_spectra, list)
        assert all(isinstance(t, Spectrum) for t in theoretical_spectra)
        assert callable(distance_function)
        assert isinstance(max_distance, (int, float))
        assert isinstance(trash_cost, (int, float))
        assert isinstance(scale_factor, (int, float))

        self.scale_factor = scale_factor
        self.empirical_spectrum = empirical_spectrum.scaled(scale_factor)
        del empirical_spectrum

        #print("Empirical spectrum:", str(self.empirical_spectrum.cspectrum_wrapper))
        self.theoretical_spectra = [t.scaled(scale_factor) for t in theoretical_spectra]
        del theoretical_spectra
        #print("Theoretical spectra:", [str(t.cspectrum_wrapper)+'\n' for t in self.theoretical_spectra])
        def wrapped_dist(p, y):
            i = p[1]
            x = p[0][:, i:i+1]
            return distance_function(x[: np.newaxis], y)*scale_factor
        self.graph = pylmcf_cpp.CDecompositableFlowGraph(self.empirical_spectrum.cspectrum, [ts.cspectrum for ts in self.theoretical_spectra], wrapped_dist, int(max_distance*scale_factor))

        self.graph.add_simple_trash(trash_cost*scale_factor)
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
            empirical_peak_idx, theoretical_peak_idx, flow = self.graph.flows_for_spectrum(i)
            result.append(namedtuple('Flow', ['empirical_peak_idx', 'theoretical_peak_idx', 'flow'])(empirical_peak_idx, theoretical_peak_idx, flow / self.scale_factor))
        return result