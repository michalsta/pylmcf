"""Graph.solve_from_basis(): cold dual network simplex from a given basis.

Whatever basis is passed (from an optimum of other supplies, a random forest,
not a forest at all), the result must be a certified optimum with the same
cost as a plain solve().
"""
import numpy as np
import pytest

import mcf_certify
from mcf_certify import random_instance
from pylmcf.graph import Graph

INF = np.iinfo(np.int64).max


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


def instance(rng, uncapacitated):
    n = int(rng.integers(2, 30))
    inst = random_instance(rng, n, int(rng.integers(1, 120)))
    if uncapacitated:
        # Most edges uncapacitated (the packed scan), a few finite ones left
        # to be bound-flipped.
        caps = inst["caps"].copy()
        caps[rng.random(len(caps)) < 0.8] = INF
        inst["caps"] = caps
    return inst


def forest(rng, inst, edges):
    """Edges, in the given order, that keep a forest (union-find)."""
    uf = list(range(inst["n"]))

    def find(x):
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x
    out = []
    for e in edges:
        a, b = find(int(inst["starts"][e])), find(int(inst["ends"][e]))
        if a != b:
            uf[a] = b
            out.append(int(e))
    return np.array(out, np.int64)


def bases(rng, inst, other_opt):
    """Optimal-looking bases of other supplies, and random ones."""
    m = len(inst["starts"])
    finite = np.flatnonzero(inst["caps"] < INF)
    free = np.flatnonzero((other_opt > 0) & (other_opt < inst["caps"]))
    at_cap = np.flatnonzero((other_opt == inst["caps"]) & (inst["caps"] < INF))
    tree = forest(rng, inst, free)
    yield tree, np.setdiff1d(at_cap, tree).astype(np.int64)
    yield tree, np.zeros(0, np.int64)
    perm = rng.permutation(m)
    tree = forest(rng, inst, perm)
    rest = np.setdiff1d(finite, tree)
    yield tree, rest[rng.random(len(rest)) < 0.5].astype(np.int64)
    yield np.zeros(0, np.int64), np.zeros(0, np.int64)


@pytest.mark.parametrize("uncapacitated", [False, True])
@pytest.mark.parametrize("seed", range(150))
def test_dual_simplex_start_reaches_certified_optimum(seed, uncapacitated):
    rng = np.random.default_rng(seed)
    inst = instance(rng, uncapacitated)
    ref = build(inst)
    ref.solve()
    # A basis optimal for other supplies: dual feasible up to degenerate tree
    # edges, primal infeasible for these supplies.
    inst2 = dict(inst, supply=inst["supply"][rng.permutation(inst["n"])])
    alt = build(inst2)
    try:
        alt.solve()
        other_opt = np.asarray(alt.result(), np.int64)
    except RuntimeError:                       # infeasible permuted supplies
        other_opt = np.asarray(ref.result(), np.int64)
    for tree, upper in bases(rng, inst, other_opt):
        g = build(inst)
        g.solve_from_basis(np.ascontiguousarray(tree), np.ascontiguousarray(upper))
        assert g.total_cost() == ref.total_cost()
        certify(g, inst)


def test_dual_simplex_start_counts():
    rng = np.random.default_rng(3)
    inst = random_instance(rng, 12, 60)
    g = build(inst)
    g.solve_from_basis(np.zeros(0, np.int64), np.zeros(0, np.int64))
    assert g.dual_start_count() == 1
    certify(g, inst)


def test_dual_simplex_start_bad_basis_falls_back():
    # Not a forest: the basis is refused and the solve runs cold.
    g = Graph(3, np.array([0, 0, 1]), np.array([1, 2, 2]))
    g.set_node_supply(np.array([2, 0, -2]))
    g.set_edge_capacities(np.array([1, 5, 1]))
    g.set_edge_costs(np.array([1, 5, 1]))
    g.solve_from_basis(np.array([0, 1, 2], np.int64), np.zeros(0, np.int64))
    assert g.total_cost() == 2 + 5
    assert g.dual_start_count() == 0
    # An uncapacitated edge cannot start at its upper bound.
    g.set_edge_capacities(np.array([1, INF, 1]))
    g.solve_from_basis(np.array([0], np.int64), np.array([1], np.int64))
    assert g.total_cost() == 2 + 5
    assert g.dual_start_count() == 0


def test_dual_simplex_start_with_minimums_runs_cold():
    rng = np.random.default_rng(1)
    inst = random_instance(rng, 12, 60, with_minimums=True)
    ref = build(inst)
    ref.solve()
    g = build(inst)
    g.solve_from_basis(np.zeros(0, np.int64), np.zeros(0, np.int64))
    assert g.total_cost() == ref.total_cost()
    certify(g, inst)


def test_dual_simplex_start_edge_index_checked():
    g = Graph(2, np.array([0]), np.array([1]))
    g.set_node_supply(np.array([1, -1]))
    g.set_edge_capacities(np.array([1]))
    with pytest.raises(ValueError):
        g.solve_from_basis(np.array([1], np.int64), np.zeros(0, np.int64))
    with pytest.raises(ValueError):
        g.solve_from_basis(np.zeros(0, np.int64), np.array([-1], np.int64))


@pytest.mark.parametrize("strategy", ["repair_only", "dual", "primal", "dual_ratio", "dual_greedy"])
def test_dual_simplex_start_then_warm_resolve(strategy):
    rng = np.random.default_rng(7)
    inst = random_instance(rng, 20, 90)
    g = build(inst)
    g.set_warm_repair(strategy)
    g.solve_from_basis(np.zeros(0, np.int64), np.zeros(0, np.int64))
    certify(g, inst)
    # the dual-started basis must be reusable by the ordinary warm re-solve
    g.set_edge_costs(np.ascontiguousarray(inst["costs"][::-1]))
    inst2 = dict(inst, costs=inst["costs"][::-1].copy())
    g.solve()
    ref = build(inst2)
    ref.solve()
    assert g.total_cost() == ref.total_cost()
    certify(g, inst2)
