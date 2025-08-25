from pylmcf.graph import DecompositableFlowGraph
from pylmcf.distribution import Distribution
from pylmcf.solver import DeconvolutionSolver


import pandas as pd
import numpy as np
from tqdm import tqdm
import sys


distance_function = lambda x, y: np.linalg.norm(x - y, axis=0)
distance_threshold = 0.1
trash_cost = 10.0
detection_threshold = 0.002
# Easy
lib = "Workshop data/data/BMRB/BMRBlib.csv"
target = "Workshop data/data/MetaboMiner/csv/N925(2x4).csv"
# Moderate
#lib = "Workshop data/data/MetaboMiner/csv/MetaboMiner - Biofluid ( all ).csv"
#target = "Workshop data/data/MetaboMiner/csv/N925(2x4).csv"

lib = pd.read_csv(lib)
target = pd.read_csv(target)

def to_spectrum(df):
    A = np.array([df['1H'], df['13C'] / 10.0])
    W = np.array(df["weight"])
    W = np.maximum(W, 0)
    return Distribution(A, W)

lib = [(name, to_spectrum(group)) for name, group in lib.groupby("name")]
target = to_spectrum(target)


def prefilter(target, lib, threshold_intensity, distance_threshold):
    """
    Prefilter the library based on the target spectrum.
    """
    filtered_lib = []
    threshold_intensity = target.intensities.sum() * threshold_intensity
    for name, spectrum in tqdm(lib, desc="Prefiltering library"):
        min_intensity = np.inf
        for peak in spectrum.positions.T:
            if min_intensity < threshold_intensity:
                break
            dists = distance_function(target.positions, peak[:, np.newaxis])
            intensity = target.intensities[dists < distance_threshold].sum()
            min_intensity = min(min_intensity, intensity)
        if min_intensity >= threshold_intensity:
            filtered_lib.append((name, spectrum, min_intensity))
            print(f"Keeping {name} with min intensity {min_intensity}")
    return filtered_lib


lib = prefilter(target, lib, detection_threshold, distance_threshold)
print(f"Filtered library size: {len(lib)}")

try:
    lib_names, lib_spectra, min_ints = zip(*lib)
except ValueError:
    print("No spectra passed the prefiltering step. Terminating.")
    sys.exit(1)

solver = DeconvolutionSolver(target, lib_spectra, distance_function, distance_threshold, trash_cost=10.0, scale_factor=100.0)
print(solver.solve(start_point=min_ints))

