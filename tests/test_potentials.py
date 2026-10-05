# Node potentials (the dual solution) exposed by Graph.potentials().
#
# Potentials are not unique, so they are never compared against golden
# values; each solve is certified from first principles instead (primal
# feasibility, complementary slackness, strong duality — see mcf_certify).

import numpy as np
import pytest

import mcf_certify
from mcf_certify import random_instance
from pylmcf.graph import Graph


def build(inst):
    g = Graph(inst["n"], inst["starts"], inst["ends"])
    push(g, inst)
    return g


def push(g, inst):
    g.set_node_supply(np.ascontiguousarray(inst["supply"]))
    g.set_edge_capacities(np.ascontiguousarray(inst["caps"]))
    g.set_edge_costs(np.ascontiguousarray(inst["costs"]))
    if inst["minimums"] is not None:
        g.set_edge_minimums(np.ascontiguousarray(inst["minimums"]))


def certify(g, inst):
    pi = g.potentials()
    mcf_certify.certify(inst["n"], inst["starts"], inst["ends"], inst["supply"],
                        inst["caps"], inst["costs"], g.result(), pi,
                        g.total_cost(), minimums=inst["minimums"])
    return pi


def test_potentials_simple():
    g = Graph(3, np.array([0, 0, 1]), np.array([1, 2, 2]))
    g.set_node_supply(np.array([5, 0, -5]))
    g.set_edge_capacities(np.array([3, 3, 3]))
    g.set_edge_costs(np.array([1, 3, 5]))
    g.solve()
    inst = {
        "n": 3,
        "starts": np.array([0, 0, 1]),
        "ends": np.array([1, 2, 2]),
        "supply": np.array([5, 0, -5]),
        "caps": np.array([3, 3, 3]),
        "costs": np.array([1, 3, 5]),
        "minimums": None,
    }
    pi = certify(g, inst)
    # Flows are [2, 3, 2]: the direct route (cost 3) is saturated, the
    # 0->1->2 route (cost 6) carries flow strictly inside its bounds, so both
    # of its arcs have rc == 0 and pin pi[2] - pi[0] to its cost.
    assert pi[2] - pi[0] == 6


def test_potentials_before_solve_raises():
    g = Graph(2, np.array([0]), np.array([1]))
    g.set_node_supply(np.array([1, -1]))
    g.set_edge_capacities(np.array([1]))
    g.set_edge_costs(np.array([1]))
    with pytest.raises(RuntimeError, match="solve"):
        g.potentials()


def test_potentials_invalidated_by_update():
    g = Graph(2, np.array([0]), np.array([1]))
    g.set_node_supply(np.array([1, -1]))
    g.set_edge_capacities(np.array([1]))
    g.set_edge_costs(np.array([1]))
    g.solve()
    g.potentials()
    g.set_edge_costs(np.array([2]))
    with pytest.raises(RuntimeError, match="solve"):
        g.potentials()


def test_potentials_returns_fresh_copy():
    g = Graph(2, np.array([0]), np.array([1]))
    g.set_node_supply(np.array([1, -1]))
    g.set_edge_capacities(np.array([1]))
    g.set_edge_costs(np.array([4]))
    g.solve()
    pi = g.potentials()
    pi[:] = 12345
    assert not np.array_equal(g.potentials(), pi)


def test_potentials_isolated_and_disconnected_nodes():
    # Node 2 has no edges; {3, 4} is a component carrying no flow.
    inst = {
        "n": 5,
        "starts": np.array([0, 3]),
        "ends": np.array([1, 4]),
        "supply": np.array([2, -2, 0, 0, 0]),
        "caps": np.array([5, 5]),
        "costs": np.array([7, 1]),
        "minimums": None,
    }
    g = build(inst)
    g.solve()
    pi = certify(g, inst)
    assert pi[1] - pi[0] == 7


@pytest.mark.parametrize("with_minimums", [False, True])
@pytest.mark.parametrize("seed", range(40))
def test_potentials_random_cold(seed, with_minimums):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 30))
    m = int(rng.integers(1, 4 * n))
    inst = random_instance(rng, n, m, with_minimums)
    g = build(inst)
    g.solve()
    certify(g, inst)


@pytest.mark.parametrize("seed", range(20))
def test_potentials_warm_chain(seed):
    # Re-solves go through warmRun(); the potentials it leaves behind must
    # certify the new optimum just as a cold solve's do.
    rng = np.random.default_rng(1000 + seed)
    n = int(rng.integers(3, 25))
    m = int(rng.integers(n, 4 * n))
    inst = random_instance(rng, n, m)
    g = build(inst)
    g.solve()
    certify(g, inst)
    for step in range(12):
        fresh = random_instance(rng, n, m)
        # Keep the topology; take new supplies/caps, and every third step new
        # costs too (exercising the costs_changed potential recomputation).
        fresh["starts"], fresh["ends"] = inst["starts"], inst["ends"]
        wit = rng.integers(0, 13, m)
        fresh["supply"] = np.zeros(n, dtype=np.int64)
        np.add.at(fresh["supply"], inst["starts"], wit)
        np.add.at(fresh["supply"], inst["ends"], -wit)
        fresh["caps"] = wit + rng.integers(0, 19, m) * (rng.integers(0, 3, m) != 0)
        if step % 3 != 2:
            fresh["costs"] = inst["costs"]
        inst = fresh
        push(g, inst)
        g.solve()
        certify(g, inst)
    resolves = (g.warm_start_count() + g.dual_repair_count()
            + g.primal_repair_count() + g.cold_start_count())
    assert resolves == 12


# --- Functional API: return_potentials=True ---------------------------------

from pylmcf import pylmcf_cpp  # noqa: E402

FUNCTIONAL = [
    (pylmcf_cpp.lmcf, [np.int8, np.int16, np.int32, np.int64]),
    (pylmcf_cpp.lmcf_cycle_canceling, [np.int8, np.int16, np.int32, np.int64]),
    (pylmcf_cpp.lmcf_cost_scaling, [np.int32, np.int64]),
    (pylmcf_cpp.lmcf_capacity_scaling, [np.int32, np.int64]),
]


def test_functional_default_returns_flows_only():
    out = pylmcf_cpp.lmcf(np.array([5, 0, -5]), np.array([0, 0, 1]), np.array([1, 2, 2]),
                          np.array([3, 3, 5]), np.array([1, 3, 5]))
    assert isinstance(out, np.ndarray)


@pytest.mark.parametrize("fn,dtypes", FUNCTIONAL, ids=lambda x: getattr(x, "__name__", ""))
def test_functional_potentials_small_dtypes(fn, dtypes):
    # The README instance fits int8; potentials still come back int64.
    for dt in dtypes:
        args = [np.array(a, dtype=dt) for a in
                ([5, 0, -5], [0, 0, 1], [1, 2, 2], [3, 3, 5], [1, 3, 5])]
        flows, pi = fn(*args, return_potentials=True)
        assert flows.dtype == dt
        assert pi.dtype == np.int64
        assert pi[2] - pi[0] == 6
        flows_only = fn(*args)
        assert np.array_equal(flows, flows_only)


@pytest.mark.parametrize("with_minimums", [False, True])
@pytest.mark.parametrize("seed", range(15))
@pytest.mark.parametrize("fn", [f for f, _ in FUNCTIONAL], ids=lambda f: f.__name__)
def test_functional_potentials_random(fn, seed, with_minimums):
    rng = np.random.default_rng(2000 + seed)
    n = int(rng.integers(2, 25))
    m = int(rng.integers(1, 4 * n))
    inst = random_instance(rng, n, m, with_minimums)
    # The functional API accepts arbitrary edge order; shuffle to exercise
    # its internal sort (potentials are per node, so must be unaffected).
    perm = rng.permutation(m)
    starts, ends = inst["starts"][perm], inst["ends"][perm]
    caps, costs = inst["caps"][perm], inst["costs"][perm]
    mins = inst["minimums"][perm] if with_minimums else None
    args = [inst["supply"], starts, ends, caps] + ([mins] if with_minimums else []) + [costs]
    flows, pi = fn(*args, return_potentials=True)
    mcf_certify.certify(n, starts, ends, inst["supply"], caps, costs, flows, pi,
                        int(np.dot(costs, flows)), minimums=mins)
    # Same optimum as the OO API.
    g = build(inst)
    g.solve()
    assert np.dot(costs, flows) == g.total_cost()
