"""Graph.solve_from_flow(): crash start from a caller-supplied flow.

Whatever flow is passed (optimal, feasible but poor, or not even feasible),
the result must be a certified optimum with the same cost as a plain solve().
"""
import numpy as np
import pytest

import mcf_certify
from mcf_certify import random_instance
from pylmcf.graph import Graph


def build(inst):
    g = Graph(inst["n"], inst["starts"], inst["ends"])
    g.set_node_supply(np.ascontiguousarray(inst["supply"]))
    g.set_edge_capacities(np.ascontiguousarray(inst["caps"]))
    g.set_edge_costs(np.ascontiguousarray(inst["costs"]))
    if inst["minimums"] is not None:
        g.set_edge_minimums(np.ascontiguousarray(inst["minimums"]))
    return g


def certify(g, inst):
    mcf_certify.certify(inst["n"], inst["starts"], inst["ends"], inst["supply"],
                        inst["caps"], inst["costs"], g.result(), g.potentials(),
                        g.total_cost(), minimums=inst["minimums"])


def starting_flows(rng, inst, optimal):
    m = len(inst["starts"])
    yield optimal
    yield np.zeros(m, np.int64)                                  # infeasible unless supply == 0
    yield np.minimum(optimal + rng.integers(0, 3, m), inst["caps"])  # perturbed, infeasible
    yield np.where(rng.random(m) < 0.5, optimal, 0).astype(np.int64)
    yield inst["caps"].astype(np.int64)                           # everything at capacity


@pytest.mark.parametrize("seed", range(200))
def test_crash_start_reaches_certified_optimum(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 30))
    m = int(rng.integers(1, 120))
    inst = random_instance(rng, n, m)
    ref = build(inst)
    ref.solve()
    opt = np.asarray(ref.result(), np.int64)
    for flow in starting_flows(rng, inst, opt):
        g = build(inst)
        g.solve_from_flow(np.ascontiguousarray(flow, np.int64))
        assert g.total_cost() == ref.total_cost()
        certify(g, inst)


def test_crash_start_with_minimums_runs_cold():
    rng = np.random.default_rng(1)
    inst = random_instance(rng, 12, 60, with_minimums=True)
    ref = build(inst)
    ref.solve()
    g = build(inst)
    g.solve_from_flow(np.zeros(60, np.int64))
    assert g.total_cost() == ref.total_cost()
    certify(g, inst)


def test_crash_start_flow_length_checked():
    g = Graph(2, np.array([0]), np.array([1]))
    g.set_node_supply(np.array([1, -1]))
    g.set_edge_capacities(np.array([1]))
    with pytest.raises(ValueError):
        g.solve_from_flow(np.zeros(2, np.int64))


@pytest.mark.parametrize("strategy", ["repair_only", "dual", "primal", "dual_ratio", "dual_greedy"])
def test_crash_start_then_warm_resolve(strategy):
    rng = np.random.default_rng(7)
    inst = random_instance(rng, 20, 90)
    g = build(inst)
    g.set_warm_repair(strategy)
    g.solve_from_flow(np.zeros(90, np.int64))
    certify(g, inst)
    # the crash-started basis must be reusable by the ordinary warm re-solve
    g.set_edge_costs(np.ascontiguousarray(inst["costs"][::-1]))
    inst2 = dict(inst, costs=inst["costs"][::-1].copy())
    g.solve()
    ref = build(inst2)
    ref.solve()
    assert g.total_cost() == ref.total_cost()
    certify(g, inst2)


@pytest.mark.parametrize("seed", range(100))
def test_crash_start_with_preferred_edges(seed):
    rng = np.random.default_rng(10_000 + seed)
    n = int(rng.integers(2, 30))
    m = int(rng.integers(1, 120))
    inst = random_instance(rng, n, m)
    ref = build(inst)
    ref.solve()
    opt = np.asarray(ref.result(), np.int64)
    for prefer in (np.arange(m, dtype=np.int64), rng.permutation(m).astype(np.int64)[: m // 2],
                   np.zeros(0, np.int64)):
        for flow in (np.zeros(m, np.int64), opt):
            g = build(inst)
            g.solve_from_flow(np.ascontiguousarray(flow), np.ascontiguousarray(prefer))
            assert g.total_cost() == ref.total_cost()
            certify(g, inst)


def test_preferred_edge_index_checked():
    g = Graph(2, np.array([0]), np.array([1]))
    g.set_node_supply(np.array([1, -1]))
    g.set_edge_capacities(np.array([1]))
    with pytest.raises(ValueError):
        g.solve_from_flow(np.zeros(1, np.int64), np.array([1], np.int64))
    with pytest.raises(ValueError):
        g.set_cold_start_flow(np.zeros(1, np.int64), np.array([-1], np.int64))
    with pytest.raises(ValueError):
        g.set_cold_start_flow(np.zeros(3, np.int64))


@pytest.mark.parametrize("strategy", ["repair_only", "dual", "primal", "dual_ratio", "dual_greedy"])
@pytest.mark.parametrize("seed", range(60))
def test_cold_start_flow_through_resolves(seed, strategy):
    """set_cold_start_flow(): the first solve crash-starts, warm re-solves stay exact."""
    rng = np.random.default_rng(20_000 + seed)
    n = int(rng.integers(3, 25))
    m = int(rng.integers(3, 100))
    inst = random_instance(rng, n, m)
    g = build(inst)
    g.set_warm_repair(strategy)
    g.set_cold_start_flow(np.zeros(m, np.int64), np.arange(m, dtype=np.int64))
    assert g.has_cold_start_flow()
    g.solve()
    assert g.crash_start_count() == 1
    certify(g, inst)
    for _ in range(4):
        # new feasible supplies: route a fresh witness flow along random edges
        wit = rng.integers(0, 5, m) * (rng.random(m) < 0.5)
        wit = np.minimum(wit, inst["caps"])
        sup = np.zeros(n, np.int64)
        np.add.at(sup, inst["starts"], wit)
        np.add.at(sup, inst["ends"], -wit)
        inst = dict(inst, supply=sup)
        g.set_node_supply(sup)
        g.solve()
        fresh = build(inst)
        fresh.solve()
        assert g.total_cost() == fresh.total_cost()
        certify(g, inst)
    g.clear_cold_start_flow()
    assert not g.has_cold_start_flow()


def test_cold_start_flow_cleared_runs_artificial():
    rng = np.random.default_rng(3)
    inst = random_instance(rng, 15, 60)
    g = build(inst)
    g.set_cold_start_flow(np.zeros(60, np.int64))
    g.clear_cold_start_flow()
    g.solve()
    assert g.crash_start_count() == 0
    certify(g, inst)
