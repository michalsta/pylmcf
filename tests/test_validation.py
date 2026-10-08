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


GETTERS = ("get_node_supply", "get_edge_capacities", "get_edge_minimums", "get_edge_costs")


def _setter_graph():
    g = Graph(3, i64(0, 0, 1), i64(1, 2, 2))
    g.set_node_supply(i64(1, 0, -1))
    g.set_edge_capacities(i64(1, 1, 1))
    g.set_edge_costs(i64(1, 3, 1))
    return g


# A rejected update must change nothing: the negative entry comes *after*
# valid ones that differ from the current data, which a check-while-writing
# loop had already stored -- leaving the maps half-updated while the solver
# and the costs-changed flag still described the old problem, so the next
# warm solve returned a stale, suboptimal flow (cost 11 instead of 3 here).
@pytest.mark.parametrize("setter,bad", [
    ("set_edge_costs", i64(10, -1, 0)),
    ("set_edge_capacities", i64(0, -1, 2)),
    ("set_edge_minimums", i64(1, 0, -1)),
])
def test_graph_rejected_setter_changes_nothing(setter, bad):
    g = _setter_graph()
    g.solve()
    before = {name: getattr(g, name)().copy() for name in GETTERS}
    flows, cost = g.result().copy(), g.total_cost()
    with pytest.raises(ValueError, match="non-negative"):
        getattr(g, setter)(bad)
    for name in GETTERS:
        assert np.array_equal(getattr(g, name)(), before[name]), name
    assert np.array_equal(g.result(), flows) and g.total_cost() == cost

    # The next update of that setter must re-solve warm to the cold optimum.
    good = np.abs(bad)
    getattr(g, setter)(good)
    g.solve()
    cold = _setter_graph()
    getattr(cold, setter)(good)
    cold.solve()
    assert g.total_cost() == cold.total_cost()
    # And re-solving the unchanged graph must not move the optimum either.
    g2 = _setter_graph()
    g2.solve()
    with pytest.raises(ValueError):
        getattr(g2, setter)(bad)
    g2.solve()
    assert g2.total_cost() == cost


@pytest.mark.parametrize("setter", ["set_edge_costs", "set_edge_capacities", "set_edge_minimums"])
def test_graph_rejected_setter_on_fresh_graph(setter):
    # Nothing set yet: the rejected update must leave the zero defaults (not
    # LEMON's own defaults, 1 for costs and infinite for capacities).
    g = Graph(3, i64(0, 0, 1), i64(1, 2, 2))
    with pytest.raises(ValueError, match="non-negative"):
        getattr(g, setter)(i64(4, 5, -1))
    for name in GETTERS:
        assert np.array_equal(getattr(g, name)(), i64(0, 0, 0)), name


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


# --- alignment ------------------------------------------------------------
# A typed C++ pointer must be aligned to its element; numpy_to_span() used to
# accept any contiguous array, including views at an odd byte offset
# (flags.aligned False), and dereference them as int64_t* -- undefined
# behaviour that x86 merely tolerates.

def misaligned(values, dtype=np.int64):
    values = np.asarray(values, dtype=dtype)
    buf = np.zeros(values.nbytes + 1, dtype=np.uint8)
    out = np.ndarray(values.shape, dtype=dtype, buffer=buf, offset=1)
    out[:] = values
    # numpy calls every empty array aligned, whatever its address.
    assert out.ctypes.data % out.itemsize != 0 and out.flags.c_contiguous
    assert out.size == 0 or not out.flags.aligned
    return out


@pytest.mark.parametrize("setter,values", [
    ("set_node_supply", SUPPLY), ("set_edge_capacities", CAPS),
    ("set_edge_minimums", i64(0, 0, 0)), ("set_edge_costs", COSTS),
])
def test_graph_setters_reject_misaligned(setter, values):
    g = Graph(3, STARTS, ENDS)
    with pytest.raises(ValueError, match="aligned"):
        getattr(g, setter)(misaligned(values))


@pytest.mark.parametrize("dtype", [np.int64, np.int32])
@pytest.mark.parametrize("which", ["starts", "ends"])
def test_graph_constructor_rejects_misaligned(dtype, which):
    starts, ends = STARTS.astype(dtype), ENDS.astype(dtype)
    if which == "starts":
        starts = misaligned(starts, dtype)
    else:
        ends = misaligned(ends, dtype)
    with pytest.raises(ValueError, match="aligned"):
        Graph(3, starts, ends)


@pytest.mark.parametrize("fn", FUNCTIONAL, ids=lambda f: f.__name__)
@pytest.mark.parametrize("pos", range(6), ids=["supply", "starts", "ends", "caps", "minimums", "costs"])
def test_functional_rejects_misaligned(fn, pos):
    args = [SUPPLY, STARTS, ENDS, CAPS, i64(0, 0, 0), COSTS]
    args[pos] = misaligned(args[pos])
    with pytest.raises(ValueError, match="aligned"):
        fn(*args)


@pytest.mark.parametrize("fn", ["lmcf_lct", "lmcf_lct_dyn"])
def test_lct_rejects_misaligned(fn):
    import pylmcf
    with pytest.raises(ValueError, match="aligned"):
        getattr(pylmcf, fn)(SUPPLY, STARTS, ENDS, misaligned(CAPS), COSTS)


def test_misaligned_single_element_rejected_empty_accepted():
    g = Graph(2, i64(0), i64(1))
    with pytest.raises(ValueError, match="aligned"):
        g.set_edge_costs(misaligned([1]))
    e = Graph(1, i64(), i64())
    e.set_edge_costs(misaligned([]))  # nothing is dereferenced


def test_misaligned_fixes_suggested_by_the_error_work():
    bad = misaligned(SUPPLY)
    with pytest.raises(ValueError, match=r"arr\.copy\(\)"):
        Graph(3, STARTS, ENDS).set_node_supply(bad)
    for fixed in (bad.copy(), np.require(bad, requirements="CA")):
        g = Graph(3, STARTS, ENDS)
        g.set_node_supply(fixed)
        g.set_edge_capacities(CAPS)
        g.set_edge_costs(COSTS)
        g.solve()
        assert g.total_cost() == 21


# --- supply minimum and cost bound -------------------------------------------
# LEMON negates supplies (artificial-arc flows; Graph's LEQ cut too), so the
# dtype's minimum overflowed: the LEQ cut called a one-node INT64_MIN graph
# feasible.  NetworkSimplex's artificial arcs cost 2^62, so a real cost of
# 2^62 made a feasible problem come back INFEASIBLE.

I64_MIN, MAX_COST = np.iinfo(np.int64).min, 2**62 - 1


def test_graph_rejects_min_supply_and_changes_nothing():
    g = Graph(3, STARTS, ENDS)
    g.set_node_supply(SUPPLY)
    with pytest.raises(ValueError, match="greater than"):
        g.set_node_supply(i64(1, 2, I64_MIN))
    assert np.array_equal(g.get_node_supply(), SUPPLY)


def test_leq_cut_no_longer_sees_min_supply_as_feasible():
    g = Graph(1, i64(), i64())
    g.set_supply_type("leq")
    with pytest.raises(ValueError, match="greater than"):
        g.set_node_supply(i64(I64_MIN))
    g.set_node_supply(i64(I64_MIN + 1))  # the smallest accepted: LEQ-infeasible
    assert np.array_equal(g.infeasibility_cut(), [True])
    with pytest.raises(RuntimeError, match="INFEASIBLE"):
        g.solve()


@pytest.mark.parametrize("fn", FUNCTIONAL, ids=lambda f: f.__name__)
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
def test_functional_rejects_min_supply(fn, dtype):
    if dtype in (np.int8, np.int16) and fn in (pylmcf_cpp.lmcf_cost_scaling, pylmcf_cpp.lmcf_capacity_scaling):
        pytest.skip("cost/capacity scaling are bound for int32/int64 only")
    a = lambda *x: np.array(x, dtype=dtype)  # noqa: E731
    with pytest.raises(ValueError, match="greater than"):
        fn(a(0, 0, np.iinfo(dtype).min), a(0, 0, 1), a(1, 2, 2), a(3, 3, 5), a(1, 3, 5))


def test_graph_cost_bound():
    g = Graph(2, i64(0), i64(1))
    g.set_node_supply(i64(1, -1))
    g.set_edge_capacities(i64(1))
    g.set_edge_costs(i64(MAX_COST))
    with pytest.raises(ValueError, match="at most"):
        g.set_edge_costs(i64(MAX_COST + 1))
    with pytest.raises(ValueError, match="non-negative"):
        g.set_edge_costs(i64(-1))
    assert np.array_equal(g.get_edge_costs(), [MAX_COST])  # rejections changed nothing
    g.solve()  # 2^62 used to come back INFEASIBLE; 2^62 - 1 is the largest accepted
    assert g.total_cost() == MAX_COST


def test_graph_cost_bound_mid_array_rolls_back():
    g = Graph(3, STARTS, ENDS)
    g.set_edge_costs(COSTS)
    with pytest.raises(ValueError, match="at most"):
        g.set_edge_costs(i64(1, 2**62, 3))
    assert np.array_equal(g.get_edge_costs(), COSTS)


def test_functional_cost_bound():
    with pytest.raises(ValueError, match="at most"):
        pylmcf_cpp.lmcf(i64(1, -1), i64(0), i64(1), i64(1), i64(MAX_COST + 1))
    assert np.array_equal(pylmcf_cpp.lmcf(i64(1, -1), i64(0), i64(1), i64(1), i64(MAX_COST)), [1])
