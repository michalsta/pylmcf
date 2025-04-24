from pylmcf.graph import DecompositableFlowGraph
from pylmcf.trashes import TrashFactory, TrashFactorySimple, TrashFactoryEmpirical, TrashFactoryTheory
from tqdm import tqdm
from scipy.optimize import minimize
import numpy as np

class DeconvolutionSolver:
    def __init__(self, empirical_spectrum, theoretical_spectra, distance_function, max_distance, trash_cost, scale_factor=1000000.0):
        self.scale_factor = scale_factor
        self.empirical_spectrum = empirical_spectrum.scaled(scale_factor)
        print("Empirical spectrum:", self.empirical_spectrum.intensities)
        self.theoretical_spectra = [t.scaled(scale_factor) for t in theoretical_spectra]
        print("Theoretical spectra:", [t.intensities for t in self.theoretical_spectra])
        self.DG = DecompositableFlowGraph()
        self.DG.add_empirical_spectrum(self.empirical_spectrum)
        dist_fun = lambda x, y: distance_function(x, y) * scale_factor
        for ts in tqdm(self.theoretical_spectra):
            self.DG.add_theoretical_spectrum(ts, dist_fun, max_distance*scale_factor)

        self.DG.build([TrashFactoryTheory(trash_cost*scale_factor), TrashFactoryEmpirical(trash_cost*scale_factor)])

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