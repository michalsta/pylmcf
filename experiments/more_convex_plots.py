from pylmcf import Distribution_1D, WassersteinSolver, Distribution
import pylmcf
import numpy as np
from tqdm import tqdm


def plot(E, T1, T2):
    import matplotlib.pyplot as plt

    XP = []
    YP = []
    outs = []
    solver = WassersteinSolver(
        E, [T1, T2], trash_cost=10, intensity_scaling=10000, costs_scaling=10000
    )
    for x in tqdm(np.arange(0.0, 1.0, 0.1)):
        for y in np.arange(0.0, 1.0, 0.1):
            val = solver.run([x, y])
            XP.append(x)
            YP.append(y)
            outs.append(val)

    fig, ax = plt.subplots()
    outs = np.log(outs)
    outs = np.clip(outs, 0, 7.4)
    sc = ax.scatter(XP, YP, c=outs, cmap="Greys")
    plt.colorbar(sc, label="Total Cost")
    plt.xlabel("Fraction 1")
    plt.ylabel("Fraction 2")
    plt.title("2D Color Plot of Total Cost")
    plt.show()


# E =  pylmcf.Spectrum_1D([0, 10, 20, 30], [4, 4, 5, 5])
# T1 = pylmcf.Spectrum_1D([0, 11], [8, 8])
# T2 = pylmcf.Spectrum_1D([20, 21, 30], [4, 6, 10])
# plot(E, T1, T2)


# from masserstein_example import convolved, spectra

# plot(convolved, spectra[0], spectra[1])

S1 = Distribution(np.random.rand(2, 1000) * 100, np.random.rand(1000))
S2 = Distribution(np.random.rand(2, 1000) * 100, np.random.rand(1000))
S3 = Distribution(np.random.rand(2, 1000) * 100, np.random.rand(1000))

plot(S1, S2, S3)
