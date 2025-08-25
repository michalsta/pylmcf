from pylmcf import (
    wasserstein_integer,
    Distribution,
    WassersteinSolver,
    SimpleTrash,
    wasserstein_integer_compat,
)
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm


def plot_convex():
    SIZE = 100
    RANGE = 100
    SCALE = 10000.0

    # np.random.seed(345)
    X1 = np.random.randint(0, RANGE, SIZE)
    Y1 = np.random.randint(0, RANGE, SIZE)
    X2 = np.random.randint(0, RANGE, SIZE)
    Y2 = np.random.randint(0, RANGE, SIZE)
    X3 = np.random.randint(0, RANGE, SIZE)
    Y3 = np.random.randint(0, RANGE, SIZE)
    orig_intensities1 = np.random.randint(1, RANGE, SIZE)
    orig_intensities2 = np.random.randint(1, RANGE, SIZE)
    orig_intensities3 = np.random.randint(1, RANGE, SIZE) * 2
    trash_cost = np.uint64(10)
    intensities1 = orig_intensities1 * SCALE
    intensities2 = orig_intensities2 * SCALE
    intensities3 = orig_intensities3 * SCALE

    params = np.arange(0.0, 1.0, 0.01)
    outs = []
    new_outs = []
    Sp1 = Distribution(np.array([X1, Y1]), orig_intensities1)
    Sp2 = Distribution(np.array([X2, Y2]), orig_intensities2)
    Sp3 = Distribution(np.array([X3, Y3]), orig_intensities3)
    solver = WassersteinSolver(
        Sp1,
        [Sp2, Sp3],
        trashes=[SimpleTrash(trash_cost)],
        intensity_scaling=SCALE,
        costs_scaling=SCALE,
    )

    for fraction in params:
        intensities = np.concatenate(
            (intensities1 * fraction, intensities2 * (1 - fraction))
        )
        X = np.concatenate((X1, X2))
        Y = np.concatenate((Y1, Y2))
        res = wasserstein_integer(X, Y, intensities, X3, Y3, intensities3, trash_cost)
        outs.append(res["total_cost"])
        solver.run([fraction, 1 - fraction])
        new_outs.append(solver.total_cost)

    outs = np.array(outs)
    new_outs = np.array(new_outs)
    if not np.all(np.diff(np.sign(np.diff(outs))) >= 0):
        plt.plot(params, outs)
        plt.show()
        print("Not convex")


def plot_heatmap(XP, YP, outs):
    outs = np.array(outs)
    # outs = np.log(outs)
    outs = np.argsort(np.argsort(outs))
    fig, ax = plt.subplots()
    sc = ax.scatter(XP, YP, c=outs, cmap="viridis")
    plt.colorbar(sc, label="Total Cost")
    plt.xlabel("Fraction 1")
    plt.ylabel("Fraction 2")
    plt.title("2D Color Plot of Total Cost")
    plt.show()


def plot_3d_convex():
    SIZE = 100
    RANGE = 100
    SCALE = 10000.0

    # np.random.seed(345)
    X1 = np.random.randint(0, RANGE, SIZE)
    Y1 = np.random.randint(0, RANGE, SIZE)
    X2 = np.random.randint(0, RANGE, SIZE)
    Y2 = np.random.randint(0, RANGE, SIZE)
    X3 = np.random.randint(0, RANGE, SIZE)
    Y3 = np.random.randint(0, RANGE, SIZE)
    X4 = np.random.randint(0, RANGE, SIZE)
    Y4 = np.random.randint(0, RANGE, SIZE)
    intensities1 = np.random.randint(1, RANGE, SIZE)
    intensities2 = np.random.randint(1, RANGE, SIZE)
    intensities3 = np.random.randint(1, RANGE, SIZE)
    intensities4 = np.random.randint(1, RANGE, SIZE) * 3
    trash_cost = np.uint64(10)

    Sp1 = Distribution(np.array([X1, Y1]), intensities1)
    Sp2 = Distribution(np.array([X2, Y2]), intensities2)
    Sp3 = Distribution(np.array([X3, Y3]), intensities3)
    Sp4 = Distribution(np.array([X4, Y4]), intensities4)
    solver = WassersteinSolver(
        Sp1,
        [Sp2, Sp3, Sp4],
        trashes=[SimpleTrash(trash_cost)],
        intensity_scaling=1,
        costs_scaling=1,
    )

    intensities1 = intensities1 * SCALE
    intensities2 = intensities2 * SCALE
    intensities3 = intensities3 * SCALE
    intensities4 = intensities4 * SCALE

    params1 = np.arange(0.0, 1.0, 0.05)
    params2 = np.arange(0.0, 1.0, 0.05)
    XP = []
    YP = []
    outs1 = []
    outs2 = []
    outs3 = []

    for fraction1 in tqdm(params1):
        for fraction2 in params2:
            # if fraction1 + fraction2 > 1:
            #     continue
            intensities = np.concatenate(
                (
                    intensities1 * fraction1,
                    intensities2 * fraction2,
                    intensities3 * (2.0 - fraction1 - fraction2),
                )
            )
            X = np.concatenate((X1, X2, X3))
            Y = np.concatenate((Y1, Y2, Y3))
            res1 = wasserstein_integer_compat(
                X, Y, intensities, X4, Y4, intensities4, trash_cost
            )
            outs1.append(res1["total_cost"])
            res2 = wasserstein_integer(
                X, Y, intensities, X4, Y4, intensities4, trash_cost
            )
            outs2.append(res2["total_cost"])
            XP.append(fraction1)
            YP.append(fraction2)
            o = solver.run([fraction1, fraction2, 2.0 - fraction1 - fraction2])
            outs3.append(o)

    plot_heatmap(XP, YP, outs1)
    plot_heatmap(XP, YP, outs2)
    plot_heatmap(XP, YP, outs3)

    # outs = np.array(outs)
    # fig, ax = plt.subplots()
    # sc = ax.scatter(XP, YP, c=outs, cmap='viridis')
    # plt.colorbar(sc, label='Total Cost')
    # plt.xlabel('Fraction 1')
    # plt.ylabel('Fraction 2')
    # plt.title('2D Color Plot of Total Cost')
    # plt.show()


def plot_3d_convex():
    SIZE = 100
    RANGE = 100
    SCALE = 10000.0

    # np.random.seed(345)
    X1 = np.random.randint(0, RANGE, SIZE)
    Y1 = np.random.randint(0, RANGE, SIZE)
    X2 = np.random.randint(0, RANGE, SIZE)
    Y2 = np.random.randint(0, RANGE, SIZE)
    X3 = np.random.randint(0, RANGE, SIZE)
    Y3 = np.random.randint(0, RANGE, SIZE)
    X4 = np.random.randint(0, RANGE, SIZE)
    Y4 = np.random.randint(0, RANGE, SIZE)
    intensities1 = np.random.randint(1, RANGE, SIZE)
    intensities2 = np.random.randint(1, RANGE, SIZE)
    intensities3 = np.random.randint(1, RANGE, SIZE)
    intensities4 = np.random.randint(1, RANGE, SIZE) * 3
    trash_cost = np.uint64(10)

    Sp1 = Distribution(np.array([X1, Y1]), intensities1)
    Sp2 = Distribution(np.array([X2, Y2]), intensities2)
    Sp3 = Distribution(np.array([X3, Y3]), intensities3)
    Sp4 = Distribution(np.array([X4, Y4]), intensities4)
    solver = WassersteinSolver(
        Sp4,
        [Sp1, Sp2, Sp3],
        trashes=[SimpleTrash(trash_cost)],
        intensity_scaling=1,
        costs_scaling=1,
    )

    intensities1 = intensities1 * SCALE
    intensities2 = intensities2 * SCALE
    intensities3 = intensities3 * SCALE
    intensities4 = intensities4 * SCALE

    params1 = np.arange(0.0, 1.0, 0.05)
    params2 = np.arange(0.0, 1.0, 0.05)
    XP = []
    YP = []
    outs1 = []
    outs2 = []
    outs3 = []

    for fraction1 in tqdm(params1):
        for fraction2 in params2:
            # if fraction1 + fraction2 > 1:
            #     continue
            intensities = np.concatenate(
                (
                    intensities1 * fraction1,
                    intensities2 * fraction2,
                    intensities3 * (2.0 - fraction1 - fraction2),
                )
            )
            X = np.concatenate((X1, X2, X3))
            Y = np.concatenate((Y1, Y2, Y3))
            res1 = wasserstein_integer_compat(
                X, Y, intensities, X4, Y4, intensities4, trash_cost
            )
            outs1.append(res1["total_cost"])
            res2 = wasserstein_integer(
                X, Y, intensities, X4, Y4, intensities4, trash_cost
            )
            outs2.append(res2["total_cost"])
            XP.append(fraction1)
            YP.append(fraction2)
            o = solver.run([fraction1, fraction2, 2.0 - fraction1 - fraction2])
            outs3.append(o)

    plot_heatmap(XP, YP, outs1)
    plot_heatmap(XP, YP, outs2)
    plot_heatmap(XP, YP, outs3)

    # outs = np.array(outs)
    # fig, ax = plt.subplots()
    # sc = ax.scatter(XP, YP, c=outs, cmap='viridis')
    # plt.colorbar(sc, label='Total Cost')
    # plt.xlabel('Fraction 1')
    # plt.ylabel('Fraction 2')
    # plt.title('2D Color Plot of Total Cost')
    # plt.show()


def plot_nonscaled():
    SIZE = 100
    RANGE = 100
    SCALE = 10000.0

    # np.random.seed(345)
    X1 = np.random.randint(0, RANGE, SIZE)
    Y1 = np.random.randint(0, RANGE, SIZE)
    X2 = np.random.randint(0, RANGE, SIZE)
    Y2 = np.random.randint(0, RANGE, SIZE)
    X4 = np.random.randint(0, RANGE, SIZE)
    Y4 = np.random.randint(0, RANGE, SIZE)
    intensities1 = np.random.randint(1, RANGE, SIZE)
    intensities2 = np.random.randint(1, RANGE, SIZE)
    intensities4 = np.random.randint(1, RANGE, SIZE) * 3
    trash_cost = np.uint64(10)

    Sp1 = Distribution(np.array([X1, Y1]), intensities1)
    Sp2 = Distribution(np.array([X2, Y2]), intensities2)
    Sp4 = Distribution(np.array([X4, Y4]), intensities4)
    solver = WassersteinSolver(
        Sp4,
        [Sp1, Sp2],
        trashes=[SimpleTrash(trash_cost)],
        intensity_scaling=1,
        costs_scaling=1,
    )

    intensities1 = intensities1 * SCALE
    intensities2 = intensities2 * SCALE
    intensities4 = intensities4 * SCALE

    params1 = np.arange(0.0, 1.0, 0.05)
    params2 = np.arange(0.0, 1.0, 0.05)
    XP = []
    YP = []
    outs1 = []
    outs2 = []
    outs3 = []

    for fraction1 in tqdm(params1):
        for fraction2 in params2:
            # if fraction1 + fraction2 > 1:
            #     continue
            intensities = np.concatenate(
                (intensities1 * fraction1, intensities2 * fraction2)
            )
            X = np.concatenate((X1, X2))
            Y = np.concatenate((Y1, Y2))
            res1 = wasserstein_integer_compat(
                X, Y, intensities, X4, Y4, intensities4, trash_cost
            )
            outs1.append(res1["total_cost"])
            res2 = wasserstein_integer(
                X, Y, intensities, X4, Y4, intensities4, trash_cost
            )
            outs2.append(res2["total_cost"])
            XP.append(fraction1)
            YP.append(fraction2)
            o = solver.run([fraction1, fraction2])
            outs3.append(o)

    plot_heatmap(XP, YP, outs1)
    plot_heatmap(XP, YP, outs2)
    plot_heatmap(XP, YP, outs3)

    # outs = np.array(outs)
    # fig, ax = plt.subplots()
    # sc = ax.scatter(XP, YP, c=outs, cmap='viridis')
    # plt.colorbar(sc, label='Total Cost')
    # plt.xlabel('Fraction 1')
    # plt.ylabel('Fraction 2')
    # plt.title('2D Color Plot of Total Cost')
    # plt.show()


def plot_masserstein():
    from masserstein_example import (
        convolved,
        spectra,
        masserstein_convolved,
        masserstein_spectra,
    )

    solver = WassersteinSolver(
        convolved,
        spectra,
        trashes=[SimpleTrash(0.05)],
        intensity_scaling=10000000,
        costs_scaling=10000000,
    )
    # params = np.arange(0.0, 1.0, 0.03)
    # x = []
    # y = []
    # outs = []
    # for fraction1 in tqdm(params):
    #     for fraction2 in params:
    #         o = solver.run([fraction1, fraction2])
    #         x.append(fraction1)
    #         y.append(fraction2)
    #         outs.append(o)
    # plot_heatmap(x,y, outs)
    print(solver.estimate_proportions())
    import masserstein

    mp = masserstein.estimate_proportions(
        masserstein_convolved, masserstein_spectra, MTD=0.05
    )["proportions"]
    mp = np.array(mp)
    print("Cost at masserstein solution:", solver.run(mp))
    print("Cost at scaled masserstein solution:", solver.run(mp / np.sum(mp)))


# while True:
if __name__ == "__main__":
    plot_masserstein()
