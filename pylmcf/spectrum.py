import numpy as np


class Spectrum:
    def __init__(self, positions, intensities):
        if not isinstance(positions, np.ndarray):
            raise ValueError("positions must be a numpy array")
        if not isinstance(intensities, np.ndarray):
            raise ValueError("intensities must be a numpy array")
        if len(positions.shape) != 2:
            raise ValueError(
                "positions must be a 2D array. If you have a 1D array, use Spectrum_1D"
            )
        if len(intensities.shape) != 1:
            raise ValueError("intensities must be a 1D array")
        if positions.shape[1] != len(intensities):
            raise ValueError("positions and intensities must have the same length")
        self.positions = positions
        self.intensities = intensities
        assert self.positions.shape[1] == len(self.intensities)

    @staticmethod
    def FromMasserstein(masserstein_spectrum):
        locs, intensities = zip(*masserstein_spectrum.confs)
        return Spectrum_1D(np.array(locs), np.array(intensities))

    @staticmethod
    def Concatenate(spectra):
        assert all([isinstance(s, Spectrum) for s in spectra])
        assert all([s.positions.shape[0] == spectra[0].positions.shape[0] for s in spectra])
        positions = np.concatenate([s.positions for s in spectra], axis=1)
        intensities = np.concatenate([s.intensities for s in spectra])
        return Spectrum(positions, intensities)

    def __len__(self):
        return len(self.intensities)

    def scaled(self, factor):
        return Spectrum(self.positions.copy(), self.intensities * factor)


class Spectrum_1D(Spectrum):
    def __init__(self, positions, intensities):
        if not isinstance(positions, np.ndarray):
            positions = np.array(positions)
        if not isinstance(intensities, np.ndarray):
            intensities = np.array(intensities)
        assert len(positions.shape) == 1
        assert len(intensities.shape) == 1
        assert positions.shape[0] == intensities.shape[0]
        self.positions_1d = positions
        super().__init__(positions[np.newaxis, :], intensities)
