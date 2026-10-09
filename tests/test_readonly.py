"""Read-only inputs work without weakening the existing array contract."""

import numpy as np
import pytest

import pylmcf
from pylmcf import pylmcf_cpp


def readonly(values, dtype=np.int64):
    array = np.array(values, dtype=dtype)
    array.flags.writeable = False
    return array


def inputs(dtype=np.int64):
    return tuple(readonly(x, dtype) for x in
                 ([5, 0, -5], [0, 0, 1], [1, 2, 2], [3, 3, 5], [1, 3, 5]))


FUNCTIONAL = [pylmcf_cpp.lmcf, pylmcf_cpp.lmcf_cycle_canceling,
              pylmcf_cpp.lmcf_cost_scaling, pylmcf_cpp.lmcf_capacity_scaling]
LCT_CLASSES = [pylmcf.NetworkSimplexLCT, pylmcf.NetworkSimplexLCTDyn]


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_graph_readonly_inputs(index_dtype):
    supply, starts, ends, caps, costs = inputs()
    graph = pylmcf.Graph(3, readonly(starts, index_dtype), readonly(ends, index_dtype))
    graph.set_node_supply(supply)
    graph.set_edge_capacities(caps)
    graph.set_edge_costs(costs)
    graph.set_edge_minimums(readonly([3, 0, 0]))
    graph.solve()
    np.testing.assert_array_equal(graph.result(), [3, 2, 3])
    assert graph.total_cost() == 24
    assert graph.result().flags.writeable and graph.potentials().flags.writeable
    graph.set_node_supply(readonly([4, 0, -4]))
    graph.solve()
    assert graph.total_cost() == 21


@pytest.mark.parametrize("fn,dtype", [
    (fn, dtype) for fn in FUNCTIONAL for dtype in
    ([np.int8, np.int16, np.int32, np.int64] if fn in FUNCTIONAL[:2]
     else [np.int32, np.int64])
])
@pytest.mark.parametrize("minimums", [False, True])
def test_functional_readonly_inputs(fn, dtype, minimums):
    args = list(inputs(dtype))
    if minimums:
        args.insert(4, readonly([3, 0, 0], dtype))
    before = [x.copy() for x in args]
    flow, potentials = fn(*args, return_potentials=True)
    np.testing.assert_array_equal(flow, [3, 2, 3] if minimums else [2, 3, 2])
    assert flow.dtype == dtype
    assert flow.flags.writeable and potentials.flags.writeable
    for array, original in zip(args, before):
        np.testing.assert_array_equal(array, original)
        assert not array.flags.writeable


@pytest.mark.parametrize("cls", LCT_CLASSES)
def test_lct_readonly_inputs_and_updates(cls):
    solver = cls(*inputs())
    solver.solve()
    assert solver.total_cost() == 21
    solver.set_node_supply(readonly([4, 0, -4]))
    solver.solve()
    assert solver.total_cost() == 15
    solver.set_edge_capacities(readonly([3, 4, 5]))
    solver.solve()
    np.testing.assert_array_equal(solver.result(), [0, 4, 0])
    assert solver.total_cost() == 12
    assert solver.result().flags.writeable


@pytest.mark.parametrize("fn", [pylmcf.lmcf_lct, pylmcf.lmcf_lct_dyn])
def test_lct_functional_readonly(fn):
    flow = fn(*inputs())
    np.testing.assert_array_equal(flow, [2, 3, 2])
    assert flow.flags.writeable


def test_chain_readonly_inputs():
    args = tuple(readonly(x) for x in ([0, 10], [5, 0], [0, 5]))
    expected = pylmcf.solve_chain_1d(*(x.copy() for x in args), kappa=20)
    actual = pylmcf.solve_chain_1d(*args, kappa=20)
    assert actual["total_cost"] == 50
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    for key in ("emp_in", "theo_out", "gap"):
        assert actual[key].flags.writeable


def test_readonly_memmap(tmp_path):
    path = tmp_path / "problem.npy"
    np.save(path, np.stack([x.copy() for x in inputs()]))
    mapped = np.load(path, mmap_mode="r")
    assert isinstance(mapped, np.memmap) and not mapped.flags.writeable
    np.testing.assert_array_equal(pylmcf_cpp.lmcf(*mapped), [2, 3, 2])


@pytest.mark.parametrize("fn", FUNCTIONAL + [pylmcf.lmcf_lct, pylmcf.lmcf_lct_dyn])
@pytest.mark.parametrize("bad,error", [
    ("strided", ValueError), ("misaligned", ValueError), ("dtype", TypeError),
])
def test_readonly_inputs_still_require_valid_layout_and_dtype(fn, bad, error):
    args = list(inputs())
    if bad == "strided":
        caps = readonly([3, 0, 3, 0, 5])[::2]
    elif bad == "misaligned":
        caps = np.ndarray((3,), dtype=np.int64, buffer=bytearray(25), offset=1)
        caps[:] = [3, 3, 5]
        caps.flags.writeable = False
    else:
        caps = readonly([3, 3, 5], np.float64)
    args[3] = caps
    with pytest.raises(error):
        fn(*args)
