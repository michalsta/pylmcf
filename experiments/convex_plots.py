from pylmcf import wasserstein_integer
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

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



def plot_3d_convex():
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
    X4 = np.random.randint(0, RANGE, SIZE)
    Y4 = np.random.randint(0, RANGE, SIZE)
    intensities1 = np.random.randint(1, RANGE, SIZE)*SCALE
    intensities2 = np.random.randint(1, RANGE, SIZE)*SCALE
    intensities3 = np.random.randint(1, RANGE, SIZE)*SCALE
    intensities4 = np.random.randint(1, RANGE, SIZE)*SCALE*3
    trash_cost = np.uint64(10)

    params1 = np.arange(0.0, 1.0, 0.01)
    params2 = np.arange(0.0, 1.0, 0.01)
    XP = []
    YP = []
    outs = []
    for fraction1 in tqdm(params1):
        for fraction2 in params2:
            if fraction1 + fraction2 > 1:
                continue
            intensities = np.concatenate((intensities1 * fraction1, intensities2 * fraction2, intensities3 * (1 - fraction1 - fraction2)))
            X = np.concatenate((X1, X2, X3))
            Y = np.concatenate((Y1, Y2, Y3))
            res = wasserstein_integer(X, Y, intensities, X4, Y4, intensities4, trash_cost)
            outs.append(res['total_cost'])
            XP.append(fraction1)
            YP.append(fraction2)

    outs = np.array(outs)
    fig, ax = plt.subplots()
    sc = ax.scatter(XP, YP, c=outs, cmap='viridis')
    plt.colorbar(sc, label='Total Cost')
    plt.xlabel('Fraction 1')
    plt.ylabel('Fraction 2')
    plt.title('2D Color Plot of Total Cost')
    plt.show()

while True:
    plot_3d_convex()