# Warm-restart re-solve tests for the OO API (Graph).
#
# Graph.solve() warm-restarts from the retained basis on re-solves (LEMON
# warmRun); these tests pin its correctness against an independent cold
# oracle: a fresh Graph solving the same mutated instance from scratch.
# A warm result is certified by (a) exact total-cost equality with the cold
# oracle, (b) primal feasibility of the returned flows (bounds +
# conservation), (c) total_cost == sum(cost * flow).  A feasible flow whose
# cost equals the cold optimum is optimal, so no solver internals are
# trusted.
#
# Counter semantics (see Graph::solve): the first solve() and any solve()
# after a failure use plain run() and count in no counter; every subsequent
# successful re-solve goes through warmRun(), which increments exactly one
# of warm_start / dual_repair / primal_repair / cold_start.

import os

import numpy as np
import pytest

from pylmcf.graph import Graph


def make_instance(rng, n, m):
    """Random feasible instance; feasibility by witness construction."""
    starts = rng.integers(0, n, m)
    ends = rng.integers(0, n, m)
    ends = np.where(ends == starts, (ends + 1) % n, ends)
    order = np.lexsort((ends, starts))
    starts, ends = starts[order], ends[order]
    inst = {
        "n": n,
        "starts": starts,
        "ends": ends,
        "costs": rng.integers(0, 51, m),
        "minimums": None,
    }
    redraw_witness(rng, inst)
    mutate_caps(rng, inst)
    return inst


def redraw_witness(rng, inst):
    """New feasible witness flow -> new supply vector (sum stays 0)."""
    lo = inst["minimums"] if inst["minimums"] is not None else 0
    wit = lo + rng.integers(0, 13, len(inst["starts"]))
    supply = np.zeros(inst["n"], dtype=np.int64)
    np.add.at(supply, inst["starts"], wit)
    np.add.at(supply, inst["ends"], -wit)
    inst["wit"] = wit
    inst["supply"] = supply


def mutate_caps(rng, inst):
    """New caps >= witness; frequently exactly tight (breaks retained bases)."""
    wit = inst["wit"]
    slack = rng.integers(0, 19, len(wit))
    tight = rng.integers(0, 3, len(wit)) == 0
    inst["caps"] = np.where(tight, wit, wit + slack)


def mutate_costs(rng, inst):
    """Reprice a random subset of arcs; feasibility untouched."""
    m = len(inst["costs"])
    pick = rng.integers(0, 2, m) == 0
    inst["costs"] = np.where(pick, rng.integers(0, 51, m), inst["costs"])


def push(g, inst):
    g.set_edge_costs(np.ascontiguousarray(inst["costs"]))
    g.set_edge_capacities(np.ascontiguousarray(inst["caps"]))
    g.set_node_supply(np.ascontiguousarray(inst["supply"]))
    if inst["minimums"] is not None:
        g.set_edge_minimums(np.ascontiguousarray(inst["minimums"]))


def build(inst):
    g = Graph(inst["n"], np.ascontiguousarray(inst["starts"]),
              np.ascontiguousarray(inst["ends"]))
    push(g, inst)
    return g


def cold_cost(inst):
    g = build(inst)
    g.solve()
    return g.total_cost()


def certify(g, inst):
    ref = cold_cost(inst)
    assert g.total_cost() == ref
    flows = g.result()
    lo = inst["minimums"] if inst["minimums"] is not None else np.zeros_like(flows)
    assert np.all(flows >= lo)
    assert np.all(flows <= inst["caps"])
    assert np.dot(flows, inst["costs"]) == g.total_cost()
    bal = np.zeros(inst["n"], dtype=np.int64)
    np.add.at(bal, inst["starts"], flows)
    np.add.at(bal, inst["ends"], -flows)
    assert np.array_equal(bal, inst["supply"])


def warm_run_total(g):
    return (g.warm_start_count() + g.cold_start_count()
            + g.dual_repair_count() + g.primal_repair_count())


def test_identical_resolve_fires_warm():
    # The reachability guard: a re-solve with nothing changed must take the
    # warm fast path.  If warm machinery silently stops being wired into
    # solve(), this is the test that fails.
    rng = np.random.default_rng(7)
    inst = make_instance(rng, 6, 15)
    g = build(inst)
    g.solve()
    cost = g.total_cost()
    g.solve()
    assert g.total_cost() == cost
    assert g.warm_start_count() == 1
    assert g.cold_start_count() == 0
    certify(g, inst)


@pytest.mark.parametrize("seed", range(6))
def test_cap_supply_chains(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(4, 13))
    inst = make_instance(rng, n, int(rng.integers(n, 4 * n)))
    g = build(inst)
    g.solve()
    resolves = 0
    for step in range(12):
        kind = rng.integers(0, 3) if step % 4 else 3
        if kind == 0:
            redraw_witness(rng, inst)          # supply only
            inst["caps"] = np.maximum(inst["caps"], inst["wit"])
        elif kind == 1:
            mutate_caps(rng, inst)             # caps only
        elif kind == 2:
            redraw_witness(rng, inst)          # both
            mutate_caps(rng, inst)
        # kind == 3: no mutation — the basis stays feasible, so the cheap
        # repairTreeFlows fast path must succeed (a guaranteed warm start).
        push(g, inst)
        g.solve()
        resolves += 1
        certify(g, inst)
    # Every re-solve went through warmRun and took exactly one of its paths.
    assert warm_run_total(g) == resolves
    assert g.warm_start_count() > 0


@pytest.mark.parametrize("seed", range(4))
def test_cost_mutation_chains(seed):
    # Cost changes ride the costs_changed=true warm path: potentials are
    # recomputed and the basis is reoptimized (never the "already optimal"
    # fast path).  Interleave repricing with cap/supply mutations.
    rng = np.random.default_rng(100 + seed)
    n = int(rng.integers(4, 13))
    inst = make_instance(rng, n, int(rng.integers(n, 4 * n)))
    g = build(inst)
    g.solve()
    resolves = 0
    for step in range(10):
        mutate_costs(rng, inst)
        if step % 3 == 0:
            redraw_witness(rng, inst)
            mutate_caps(rng, inst)
        push(g, inst)
        g.solve()
        resolves += 1
        certify(g, inst)
    assert warm_run_total(g) == resolves


def test_minimums_force_cold_and_stay_correct():
    # Nonzero lower bounds make the retained basis unusable (init() folds
    # them into supplies/caps and the warm path does not); warmRun must fall
    # back to cold on every re-solve, and results must stay correct.
    rng = np.random.default_rng(42)
    n = 7
    inst = make_instance(rng, n, 20)
    inst["minimums"] = rng.integers(0, 4, 20)
    redraw_witness(rng, inst)                  # witness >= minimums
    mutate_caps(rng, inst)
    g = build(inst)
    g.solve()
    for _ in range(5):
        redraw_witness(rng, inst)
        mutate_caps(rng, inst)
        push(g, inst)
        g.solve()
        certify(g, inst)
    assert g.warm_start_count() == 0
    assert g.cold_start_count() == 5


def test_infeasible_then_feasible_recovery():
    # A failed solve must not poison the retained state: the next solve goes
    # cold (rebuilt basis), and the chain can continue warm afterwards.
    g = Graph(2, np.array([0]), np.array([1]))
    g.set_edge_costs(np.array([3]))
    g.set_edge_capacities(np.array([10]))
    g.set_node_supply(np.array([7, -7]))
    g.solve()
    assert g.total_cost() == 21

    g.set_edge_capacities(np.array([4]))
    with pytest.raises(RuntimeError, match="INFEASIBLE"):
        g.solve()

    g.set_edge_capacities(np.array([12]))
    g.solve()
    assert g.total_cost() == 21

    g.set_edge_capacities(np.array([9]))
    g.solve()
    assert g.total_cost() == 21
    assert g.warm_start_count() >= 1


@pytest.mark.skipif("PYLMCF_WARM_VIOLATION_LIMIT" in os.environ,
                    reason="env override preempts set_warm_violation_limit")
def test_warm_violation_limit_policy():
    # limit=0: whenever the cheap basis patch fails with any violated basic
    # arc, skip the simplex repair and go straight to cold.  Always correct,
    # and policy_cold_count records each forced cold.
    rng = np.random.default_rng(2026)
    inst = make_instance(rng, 10, 35)
    g = build(inst)
    g.set_warm_violation_limit(0)
    g.solve()
    resolves = 0
    for _ in range(15):
        redraw_witness(rng, inst)
        mutate_caps(rng, inst)
        push(g, inst)
        g.solve()
        resolves += 1
        certify(g, inst)
    assert warm_run_total(g) == resolves
    assert g.policy_cold_count() > 0
    assert g.dual_repair_count() == 0          # repair fully suppressed

    # Back to -1 (always attempt repair): chain continues, still correct.
    g.set_warm_violation_limit(-1)
    before = g.policy_cold_count()
    for _ in range(10):
        redraw_witness(rng, inst)
        mutate_caps(rng, inst)
        push(g, inst)
        g.solve()
        certify(g, inst)
    assert g.policy_cold_count() == before     # policy no longer forces cold
