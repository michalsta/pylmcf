# Node potentials (the dual solution) exposed by Graph.potentials().
#
# Potentials are not unique, so they are never compared against golden
# values.  Each solve is instead certified from first principles: with
# rc = cost + pi[start] - pi[end],
#   (a) complementary slackness against the returned flows:
#       rc > 0 => flow == minimum, rc < 0 => flow == capacity;
#   (b) strong duality: the dual objective computed from the potentials
#       alone equals total_cost().
# Given a primal-feasible flow, (a) is a full optimality certificate for the
# pair; (b) is what a caller recovering an objective from the duals relies on.

import numpy as np
import pytest

from pylmcf.graph import Graph


def random_instance(rng, n, m, with_minimums=False):
    """Random feasible instance; feasibility by witness construction."""
    starts = rng.integers(0, n, m)
    ends = rng.integers(0, n, m)
    ends = np.where(ends == starts, (ends + 1) % n, ends)
    order = np.lexsort((ends, starts))
    starts, ends = starts[order], ends[order]
    minimums = rng.integers(0, 4, m) if with_minimums else np.zeros(m, dtype=np.int64)
    wit = minimums + rng.integers(0, 13, m)
    supply = np.zeros(n, dtype=np.int64)
    np.add.at(supply, starts, wit)
    np.add.at(supply, ends, -wit)
    caps = wit + rng.integers(0, 19, m) * (rng.integers(0, 3, m) != 0)
    return {
        "n": n,
        "starts": starts.astype(np.int64),
        "ends": ends.astype(np.int64),
        "supply": supply,
        "caps": caps.astype(np.int64),
        "costs": rng.integers(0, 51, m).astype(np.int64),
        "minimums": minimums.astype(np.int64) if with_minimums else None,
    }


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
    assert pi.dtype == np.int64
    assert pi.shape == (inst["n"],)
    flows = g.result()
    lo = inst["minimums"] if inst["minimums"] is not None else np.zeros_like(flows)
    caps = inst["caps"]
    rc = inst["costs"] + pi[inst["starts"]] - pi[inst["ends"]]

    assert np.all(flows[rc > 0] == lo[rc > 0]), "rc > 0 on an arc above its minimum"
    assert np.all(flows[rc < 0] == caps[rc < 0]), "rc < 0 on an arc below capacity"

    dual = (-np.dot(inst["supply"], pi)
            + np.dot(np.minimum(rc, 0), caps)
            + np.dot(np.maximum(rc, 0), lo))
    assert dual == g.total_cost()
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
