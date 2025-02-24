import numpy as np


class Spectrum:
    def __init__(self, positions, intensities):
        self.positions = np.array(positions)
        self.intensities = np.array(intensities)
        assert len(self.positions) == len(self.intensities)

    @staticmethod
    def FromMasserstein(masserstein_spectrum):
        locs, intensities = zip(*masserstein_spectrum.confs())
        return Spectrum(np.array(locs), np.array(intensities))

    def __len__(self):
        return len(self.positions)

    def scaled(self, factor):
        return Spectrum(self.positions.copy(),
                        self.intensities * factor)


