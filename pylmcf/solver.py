from pylmcf.graph import DecompositableFlowGraph
from tqdm import tqdm
from scipy.optimize import minimize
import numpy as np

class DeconvolutionSolver:
    def __init__(self, empirical_spectrum, theoretical_spectra):
        self.empirical_spectrum = empirical_spectrum
        self.theoretical_spectra = theoretical_spectra[:10]
        self.DG = DecompositableFlowGraph()
        self.DG.add_empirical_spectrum(self.empirical_spectrum)
        for ts in tqdm(self.theoretical_spectra):
            self.DG.add_theoretical_spectrum(ts, lambda x, y: np.linalg.norm(10*x - y, axis=0), 10.0)

        self.DG.build()

    def solve(self, start_point = None):
        def opt_fun(point):
            print("XXX")
            return self.DG.set_point(point)
        if start_point is None:
            start_point = [1.0] * len(self.theoretical_spectra)

        return minimize(opt_fun, method='Nelder-Mead', x0 = start_point)