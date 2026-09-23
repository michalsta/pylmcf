"""Python bindings for the LCT and specialised chain solvers, against LEMON."""

import numpy as np
import pytest

import pylmcf
from pylmcf.pylmcf_cpp import lmcf


def arrays(*values):
    return tuple(np.asarray(v, dtype=np.int64) for v in values)


CLASSES = [pylmcf.NetworkSimplexLCT, pylmcf.NetworkSimplexLCTDyn]
FUNCTIONS = [pylmcf.lmcf_lct, pylmcf.lmcf_lct_dyn]


def assert_flow(supply, starts, ends, caps, costs, flow):
    assert flow.dtype == np.int64
    assert np.all(flow >= 0)
    assert np.all(flow <= caps)
    balance = np.zeros(len(supply), dtype=np.int64)
    np.add.at(balance, starts, flow)
    np.add.at(balance, ends, -flow)
    np.testing.assert_array_equal(balance, supply)
    reference = lmcf(supply, starts, ends, caps, costs)
    assert int(flow @ costs) == int(reference @ costs)


@pytest.mark.parametrize("fn", FUNCTIONS)
def test_lct_random_cold(fn):
    rng = np.random.default_rng(427)
    for _ in range(40):
        n, m = 9, 35
        starts = rng.integers(n, size=m, dtype=np.int64)
        ends = rng.integers(n, size=m, dtype=np.int64)
        caps = rng.integers(1, 20, size=m, dtype=np.int64)
        costs = rng.integers(0, 15, size=m, dtype=np.int64)
        seed_flow = rng.integers(0, caps + 1, dtype=np.int64)
        supply = np.zeros(n, dtype=np.int64)
        np.add.at(supply, starts, seed_flow)
        np.add.at(supply, ends, -seed_flow)
        args = supply, starts, ends, caps, costs
        assert_flow(*args, fn(*args))


@pytest.mark.parametrize("cls", CLASSES)
def test_lct_warm_and_recovery(cls):
    supply, starts, ends, caps, costs = arrays([5, 0, -5], [1, 0, 0], [2, 2, 1], [5, 3, 3], [5, 3, 1])
    solver = cls(supply, starts, ends, caps, costs)
    with pytest.raises(RuntimeError, match="solve"):
        solver.result()
    solver.solve()
    saved = solver.result()
    assert solver.cold_start_count() == 1
    solver.solve()
    assert solver.warm_start_count() == 1
    for amount in [4, 2, 5, 6, 1]:
        supply = arrays([amount, 0, -amount])[0]
        solver.set_node_supply(supply)
        with pytest.raises(RuntimeError, match="solve"):
            solver.total_cost()
        solver.solve()
        flow = solver.result()
        assert_flow(supply, starts, ends, caps, costs, flow)
        assert solver.total_cost() == int(flow @ costs)
    np.testing.assert_array_equal(saved, [2, 3, 2])
    solver.set_edge_capacities(np.zeros(3, dtype=np.int64))
    with pytest.raises(RuntimeError, match="INFEASIBLE"):
        solver.solve()
    with pytest.raises(RuntimeError, match="solve"):
        solver.result()
    solver.set_edge_capacities(caps)
    solver.solve()
    assert_flow(supply, starts, ends, caps, costs, solver.result())
    cold = solver.cold_start_count()
    solver.solve(warm=False)
    assert solver.cold_start_count() == cold + 1


@pytest.mark.parametrize("fn", FUNCTIONS)
@pytest.mark.parametrize("index,value,error", [
    (0, [1, 0, 0], ValueError),
    (1, [-1, 0, 1], ValueError),
    (2, [3, 2, 2], ValueError),
    (3, [-1, 3, 5], ValueError),
    (4, [-1, 3, 5], ValueError),
    (4, [1], ValueError),
    (3, [np.iinfo(np.int64).max, 3, 5], OverflowError),
    (4, [np.iinfo(np.int64).max, 3, 5], OverflowError),
])
def test_lct_invalid(fn, index, value, error):
    args = list(arrays([5, 0, -5], [0, 0, 1], [1, 2, 2], [3, 3, 5], [1, 3, 5]))
    args[index] = arrays(value)[0]
    with pytest.raises(error):
        fn(*args)


@pytest.mark.parametrize("fn", FUNCTIONS)
def test_lct_array_contract(fn):
    args = list(arrays([5, 0, -5], [0, 0, 1], [1, 2, 2], [3, 3, 5], [1, 3, 5]))
    for dtype in [np.float64, np.int32, np.uint64]:
        with pytest.raises(TypeError):
            fn(*(a.astype(dtype) for a in args))
    args[0] = np.array([5, 99, 0, 99, -5, 99], dtype=np.int64)[::2]
    with pytest.raises(ValueError, match="contiguous"):
        fn(*args)
    assert fn(*arrays([], [], [], [], [])).size == 0
    np.testing.assert_array_equal(fn(*arrays([0], [0], [0], [3], [0])), [0])


def chain_oracle(pos, emp, theo, kappa):
    n = len(pos)
    total = max(sum(map(int, emp)), sum(map(int, theo)))
    starts, ends, caps, costs = [], [], [], []
    for i in range(n):
        starts += [0, i + 2]
        ends += [i + 2, 1]
        caps += [emp[i], theo[i]]
        costs += [0, 0]
    for i in range(n - 1):
        starts += [i + 2, i + 3]
        ends += [i + 3, i + 2]
        caps += [total, total]
        costs += [pos[i + 1] - pos[i]] * 2
    starts += [0]
    ends += [1]
    caps += [total]
    costs += [kappa]
    args = arrays([total, -total] + [0] * n, starts, ends, caps, costs)
    return int(lmcf(*args) @ args[-1])


def test_chain_oracle_and_conservation():
    rng = np.random.default_rng(108)
    for n in [0, 1, 2, 5, 15]:
        for _ in range(15):
            pos = np.sort(rng.integers(-20, 30, n, dtype=np.int64))
            emp = rng.integers(0, 20, n, dtype=np.int64)
            theo = rng.integers(0, 20, n, dtype=np.int64)
            kappa = int(rng.integers(0, 30))
            result = pylmcf.solve_chain_1d(pos, emp, theo, kappa)
            assert result["total_cost"] == chain_oracle(pos, emp, theo, kappa)
            assert np.all((0 <= result["emp_in"]) & (result["emp_in"] <= emp))
            assert np.all((0 <= result["theo_out"]) & (result["theo_out"] <= theo))
            assert sum(result["emp_in"]) + result["trash"] == max(sum(emp), sum(theo))
            if n:
                np.testing.assert_array_equal(
                    result["emp_in"] - result["theo_out"],
                    np.diff(np.r_[0, result["gap"], 0]),
                )
            assert result["total_cost"] == int(abs(result["gap"]) @ np.diff(pos)) + kappa * result["trash"]


@pytest.mark.parametrize("pos,emp,theo,kappa,error", [
    ([2, 1], [1, 0], [0, 1], 2, ValueError),
    ([1], [-1], [0], 2, ValueError),
    ([1], [0], [-1], 2, ValueError),
    ([1], [0], [0], -1, ValueError),
    ([1], [], [0], 1, ValueError),
    ([0], [np.iinfo(np.int64).max], [0], 1, OverflowError),
    ([np.iinfo(np.int64).min, np.iinfo(np.int64).max], [1, 0], [0, 1], 1, OverflowError),
])
def test_chain_invalid(pos, emp, theo, kappa, error):
    with pytest.raises(error):
        pylmcf.solve_chain_1d(*arrays(pos, emp, theo), kappa)


def test_chain_array_contract():
    with pytest.raises(TypeError):
        pylmcf.solve_chain_1d(np.array([0.0]), *arrays([1], [1]), 1)
    with pytest.raises(ValueError, match="contiguous"):
        pylmcf.solve_chain_1d(np.arange(6, dtype=np.int64)[::2], *arrays([1]*3, [1]*3), 1)


@pytest.mark.parametrize("cls", CLASSES)
def test_lct_random_warm_chain(cls):
    rng = np.random.default_rng(832)
    n, m = 7, 25
    starts = rng.integers(n, size=m, dtype=np.int64)
    ends = rng.integers(n, size=m, dtype=np.int64)
    caps = rng.integers(1, 20, size=m, dtype=np.int64)
    costs = rng.integers(0, 10, size=m, dtype=np.int64)
    solver = cls(np.zeros(n, dtype=np.int64), starts, ends, caps, costs)
    solver.solve()
    for step in range(50):
        if step % 3 == 0:
            caps = rng.integers(1, 20, size=m, dtype=np.int64)
            solver.set_edge_capacities(caps)
        seed_flow = rng.integers(0, caps + 1, dtype=np.int64)
        supply = np.zeros(n, dtype=np.int64)
        np.add.at(supply, starts, seed_flow)
        np.add.at(supply, ends, -seed_flow)
        solver.set_node_supply(supply)
        solver.solve()
        assert_flow(supply, starts, ends, caps, costs, solver.result())
    assert solver.warm_start_count() + solver.cold_start_count() == 51


@pytest.mark.parametrize("cls", CLASSES)
def test_lct_owns_inputs_and_rejects_updates_atomically(cls):
    args = arrays([5, -5], [0], [1], [8], [2])
    solver = cls(*args)
    for arg in args:
        arg[:] = 0
    solver.solve()
    assert solver.total_cost() == 10
    with pytest.raises(ValueError):
        solver.set_edge_capacities(arrays([-1])[0])
    with pytest.raises(ValueError):
        solver.set_node_supply(arrays([1, 0])[0])
    assert solver.total_cost() == 10
    result = solver.result()
    result[:] = 0
    assert solver.result()[0] == 5
