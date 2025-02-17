from pylmcf import wasserstein_integer
from matplotlib import pyplot as plt
import numpy as np

def plot_convex():
    SIZE = 100
    RANGE = 100
    SCALE = 10000.0

    #np.random.seed(345)
    X1 = np.random.randint(0, RANGE, SIZE)
    Y1 = np.random.randint(0, RANGE, SIZE)
    X2 = np.random.randint(0, RANGE, SIZE)
    Y2 = np.random.randint(0, RANGE, SIZE)
    X3 = np.random.randint(0, RANGE, SIZE)
    Y3 = np.random.randint(0, RANGE, SIZE)
    intensities1 = np.random.randint(1, RANGE, SIZE)*SCALE
    intensities2 = np.random.randint(1, RANGE, SIZE)*SCALE
    intensities3 = np.random.randint(1, RANGE, SIZE)*SCALE*2
    trash_cost = np.uint64(10)

    params = np.arange(0.0, 1.0, 0.01)
    outs = []
    for fraction in params:
        intensities = np.concatenate((intensities1 * fraction, intensities2 * (1 - fraction)))
        X = np.concatenate((X1, X2))
        Y = np.concatenate((Y1, Y2))
        res = wasserstein_integer(X, Y, intensities, X3, Y3, intensities3, trash_cost)
        outs.append(res['total_cost'])

    outs = np.array(outs)
    if not np.all(np.diff(np.sign(np.diff(outs))) >= 0):
        plt.plot(params, outs)
        plt.show()
        print("Not convex")

while True:
    plot_convex()