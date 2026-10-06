# Input validation of the LEMON-backed APIs (Graph and the four functional
# lmcf* solvers).  Each guard here exists to turn a silently wrong answer —
# or undefined behaviour in C++ — into an exception, so each must be pinned:
# dtype truncation through nanobind's implicit conversion, out-of-range node
# ids fed to LEMON, unsorted edges in a StaticDigraph, and a minimum above
# its capacity (LEMON asserts that only in debug builds; release builds
# returned flows outside their bounds).

import numpy as np
import pytest

from pylmcf import pylmcf_cpp
from pylmcf.graph import Graph


def i64(*x):
    return np.array(x, dtype=np.int64)


SUPPLY, STARTS, ENDS, CAPS, COSTS = i64(5, 0, -5), i64(0, 0, 1), i64(1, 2, 2), i64(3, 3, 5), i64(1, 3, 5)

FUNCTIONAL = [pylmcf_cpp.lmcf, pylmcf_cpp.lmcf_cycle_canceling,
              pylmcf_cpp.lmcf_cost_scaling, pylmcf_cpp.lmcf_capacity_scaling]


# --- Graph ----------------------------------------------------------------

@pytest.mark.parametrize("n,starts,ends,match", [
    (3, i64(1, 0, 0), i64(2, 1, 2), "sorted"),
    (3, i64(-1, 0), i64(1, 2), "out of bounds"),
    (3, i64(0, 0), i64(1, 3), "out of bounds"),
    (3, i64(0, 2**33), i64(1, 2), "32-bit"),
    (3, i64(0, 0), i64(1), "same size"),
    (-1, i64(), i64(), "non-negative"),
])
def test_graph_rejects_bad_topology(n, starts, ends, match):
    with pytest.raises(ValueError, match=match):
        Graph(n, starts, ends)


@pytest.mark.parametrize("dtype", [np.float64, np.uint64])
def test_graph_rejects_non_signed_integer_edge_ids(dtype):
    with pytest.raises(TypeError):
        Graph(3, STARTS.astype(dtype), ENDS.astype(dtype))


@pytest.mark.parametrize("setter,values,match", [
    ("set_node_supply", i64(1, -1), "same size"),
    ("set_edge_capacities", i64(1), "same size"),
    ("set_edge_costs", i64(1, 2), "same size"),
    ("set_edge_minimums", i64(0), "same size"),
    ("set_edge_capacities", i64(1, -1, 1), "non-negative"),
    ("set_edge_costs", i64(1, -1, 1), "non-negative"),
    ("set_edge_minimums", i64(0, -1, 0), "non-negative"),
    ("set_node_supply", i64(5, 9, 0, 9, -5)[::2], "contiguous"),
])
def test_graph_setters_reject_bad_values(setter, values, match):
    g = Graph(3, STARTS, ENDS)
    with pytest.raises(ValueError, match=match):
        getattr(g, setter)(values)


@pytest.mark.parametrize("setter", ["set_node_supply", "set_edge_capacities",
                                    "set_edge_costs", "set_edge_minimums"])
@pytest.mark.parametrize("dtype", [np.float64, np.int32])
def test_graph_setters_reject_other_dtypes(setter, dtype):
    # The OO API is int64-only and must not convert: float64 -> int64 would
    # truncate silently.
    g = Graph(3, STARTS, ENDS)
    n = 3
    with pytest.raises(TypeError):
        getattr(g, setter)(np.zeros(n, dtype=dtype))


def test_graph_minimum_above_capacity():
    g = Graph(3, STARTS, ENDS)
    g.set_node_supply(SUPPLY)
    g.set_edge_costs(COSTS)
    # Either setting order must be caught, at solve time.
    g.set_edge_minimums(i64(4, 0, 0))
    g.set_edge_capacities(CAPS)
    with pytest.raises(ValueError, match="Edge 0 has minimum 4 above its capacity 3"):
        g.solve()
    with pytest.raises(ValueError, match="above its capacity"):
        g.infeasibility_cut()
    g.set_edge_minimums(i64(3, 0, 0))
    g.solve()
    assert g.result()[0] == 3


# --- functional API -------------------------------------------------------

@pytest.mark.parametrize("fn", FUNCTIONAL, ids=lambda f: f.__name__)
@pytest.mark.parametrize("args,error,match", [
    ((SUPPLY, i64(-1, 0, 1), ENDS, CAPS, COSTS), ValueError, "out of bounds"),
    ((SUPPLY, STARTS, i64(1, 2, 3), CAPS, COSTS), ValueError, "out of bounds"),
    ((SUPPLY, STARTS, ENDS, i64(3, 3), COSTS), ValueError, "same size"),
    ((SUPPLY, STARTS, ENDS, CAPS, i64(0, 0), COSTS), ValueError, "same size"),
    ((SUPPLY, STARTS, ENDS, i64(3, -3, 5), COSTS), ValueError, "non-negative"),
    ((SUPPLY, STARTS, ENDS, CAPS, i64(0, -1, 0), COSTS), ValueError, "non-negative"),
    ((SUPPLY, STARTS, ENDS, CAPS, i64(4, 0, 0), COSTS), ValueError, "above its capacity"),
    ((SUPPLY, STARTS, ENDS, CAPS, i64(1, 9, 3, 9, 5)[::2]), ValueError, "contiguous"),
    (tuple(x.astype(np.float64) for x in (SUPPLY, STARTS, ENDS, CAPS, COSTS)), TypeError, "no overload"),
    (tuple(x.astype(np.uint64) for x in (SUPPLY, STARTS, ENDS, CAPS, COSTS)), TypeError, "no overload"),
    ((SUPPLY.astype(np.int32), STARTS, ENDS, CAPS, COSTS), TypeError, "no overload"),
], ids=["neg-id", "id-range", "len", "min-len", "neg-cap", "neg-min", "min>cap",
        "noncontig", "float64", "uint64", "mixed-dtype"])
def test_functional_rejects(fn, args, error, match):
    with pytest.raises(error, match=match):
        fn(*args)


def test_lmcf_rejects_negative_costs():
    # Only the network simplex variant requires non-negative costs.
    with pytest.raises(ValueError, match="non-negative"):
        pylmcf_cpp.lmcf(SUPPLY, STARTS, ENDS, CAPS, i64(1, -3, 5))


# --- the empty problem ----------------------------------------------------

@pytest.mark.parametrize("fn", FUNCTIONAL, ids=lambda f: f.__name__)
def test_functional_empty_problem(fn):
    # LEMON rejects an empty node set as INFEASIBLE; an empty problem is
    # trivially optimal.
    e = i64()
    flows, pi = fn(e, e, e, e, e, return_potentials=True)
    assert flows.shape == (0,) and pi.shape == (0,)


def test_graph_empty_problem():
    g = Graph(0, i64(), i64())
    g.set_node_supply(i64())
    g.set_edge_capacities(i64())
    g.set_edge_costs(i64())
    g.solve()
    assert g.total_cost() == 0
    assert g.result().shape == (0,) and g.potentials().shape == (0,)
    assert g.infeasibility_cut() is None
