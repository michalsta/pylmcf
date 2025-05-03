from pylmcf import Spectrum_1D, WassersteinSolver, Spectrum, wasserstein_integer, SimpleTrash, wasserstein_integer_compat, DecompositableFlowGraph, TrashFactorySimple
import numpy as np





def compare(E, T, trash_cost, fractions = None):
    if fractions is None:
        fractions = [1.0]*len(T)
    solver = WassersteinSolver(E, T, trashes=[SimpleTrash(trash_cost)], intensity_scaling=10000, costs_scaling=1000)
    val1 = solver.run(fractions)
    positions = np.concatenate([s.positions for s in T], axis=1)
    intensities = np.concatenate([s.intensities*f for s, f in zip(T, fractions)])
    val2 = wasserstein_integer(E.positions[0], E.positions[1], E.intensities, positions[0], positions[1], intensities, trash_cost)['total_cost']
    val3 = wasserstein_integer_compat(E.positions[0], E.positions[1], E.intensities, positions[0], positions[1], intensities, trash_cost)['total_cost']
    decomp_solver = DecompositableFlowGraph(E, T, [lambda x, y: np.linalg.norm(x - y, axis=0)]*len(T), [trash_cost]*len(T))
    decomp_solver.build([TrashFactorySimple(trash_cost)])
    val4 = decomp_solver.set_point(fractions)
    print(f"Solver: {val1}, Wasserstein: {val2}, Wasserstein_compat: {val3}, DecompositableFlowGraph: {val4}")
    #assert val1 == val2 # 2 uses diffrent trash so not really the same
    assert val1 == val3
    assert val1 == val4
    return val1, val2, val3, val4



def test_compare_1():
    S1 = Spectrum(np.array([[0], [0]]), np.array([1]))
    S2 = Spectrum(np.array([[1], [0]]), np.array([1]))

    print(compare(S1, [S2], 10))

def test_compare_2():
    S1 = Spectrum(np.array([[0], [0]]), np.array([1]))
    S2 = Spectrum(np.array([[1], [0]]), np.array([1]))
    S3 = Spectrum(np.array([[2], [0]]), np.array([1]))

    print(compare(S1, [S2, S3], 10))

def test_compare_3():
    S1 = Spectrum(np.array([[0], [0]]), np.array([1]))
    S2 = Spectrum(np.array([[1], [0]]), np.array([1]))
    S3 = Spectrum(np.array([[2], [0]]), np.array([1]))
    S4 = Spectrum(np.array([[3], [0]]), np.array([1]))

    print(compare(S1, [S2, S3, S4], 10))

'''
def test_compare_4():
    S1 = Spectrum(np.random.randint(0, 1000, (2,5)), np.random.randint(0, 1000, 5))
    S2 = Spectrum(np.random.randint(0, 1000, (2,5)), np.random.randint(0, 1000, 5))

    print(compare(S1, [S2], 10, [1.0]))


def test_compare_5():
    S1 = Spectrum(np.random.randint(0, 1000, (2,50)), np.random.randint(0, 1000, 50))
    S2 = Spectrum(np.random.randint(0, 1000, (2,50)), np.random.randint(0, 1000, 50))
    S3 = Spectrum(np.random.randint(0, 1000, (2,50)), np.random.randint(0, 1000, 50))

    print(compare(S1, [S2, S3], 10, [0.0, 1.0]))

def test_compare_6():
    S1 = Spectrum(np.random.randint(0, 1000, (2,50)), np.random.randint(0, 1000, 50))
    S2 = Spectrum(np.random.randint(0, 1000, (2,50)), np.random.randint(0, 1000, 50))
    S3 = Spectrum(np.random.randint(0, 1000, (2,50)), np.random.randint(0, 1000, 50))
    S4 = Spectrum(np.random.randint(0, 1000, (2,50)), np.random.randint(0, 1000, 50))

    print(compare(S1, [S2, S3, S4], 10, [0.0, 1.0, 1.0]))
'''
if __name__ == "__main__":
    test_compare_1()
    test_compare_2()
    test_compare_3()
    #test_compare_4()
    #test_compare_5()
    #test_compare_6()
