# LEMON NetworkSimplex options exposed on Graph: supply type (geq/leq), pivot
# rule, warm-repair strategy, warm-repair time budget, and the Circulation
# infeasibility certificate.  Every solve is certified independently
# (mcf_certify) or compared against an oracle; nothing trusts solver
# internals.

import os

import numpy as np
import pytest

import mcf_certify
from mcf_certify import random_instance
from pylmcf import pylmcf_cpp
from pylmcf.graph import Graph

PIVOT_RULES = ["first_eligible", "best_eligible", "block_search",
               "candidate_list", "altering_list"]
WARM_REPAIRS = ["repair_only", "dual", "primal", "dual_ratio", "dual_greedy"]


def build(inst, supply_type=None):
    g = Graph(inst["n"], inst["starts"], inst["ends"])
    if supply_type is not None:
        g.set_supply_type(supply_type)
    push(g, inst)
    return g


def push(g, inst):
    g.set_node_supply(np.ascontiguousarray(inst["supply"]))
    g.set_edge_capacities(np.ascontiguousarray(inst["caps"]))
    g.set_edge_costs(np.ascontiguousarray(inst["costs"]))
    if inst["minimums"] is not None:
        g.set_edge_minimums(np.ascontiguousarray(inst["minimums"]))


def certify(g, inst, supply_type="eq"):
    mcf_certify.certify(inst["n"], inst["starts"], inst["ends"], inst["supply"],
                        inst["caps"], inst["costs"], g.result(), g.potentials(),
                        g.total_cost(), minimums=inst["minimums"],
                        supply_type=supply_type)


def unbalance(rng, inst, supply_type):
    """Feasible geq (sum < 0) or leq (sum > 0) instance from a balanced one.

    The balanced witness flow stays feasible: under geq a node may now send
    out more than it needs to, under leq it may keep some supply back.
    """
    extra = rng.integers(0, 6, inst["n"]) * (rng.integers(0, 3, inst["n"]) == 0)
    extra[rng.integers(0, inst["n"])] += 1          # strictly unbalanced
    inst = dict(inst)
    inst["supply"] = inst["supply"] + (extra if supply_type == "leq" else -extra)
    return inst


def slack_oracle_cost(inst, supply_type):
    """Optimum of a geq/leq instance via an equality-form reformulation.

    One hub node absorbs the imbalance through free (cost 0, uncapacitated)
    edges: under geq, out - in = supply + t with t >= 0, so the hub feeds the
    surplus outflow t in (hub -> u); under leq the unused supply drains out
    (u -> hub).  Solved by the functional API.
    """
    n = inst["n"]
    big = int(np.abs(inst["supply"]).sum() + inst["caps"].sum() + 1)
    nodes = np.arange(n, dtype=np.int64)
    hub = np.full(n, n, dtype=np.int64)
    s_extra, e_extra = (hub, nodes) if supply_type == "geq" else (nodes, hub)
    supply = np.append(inst["supply"], -inst["supply"].sum())
    starts = np.concatenate([inst["starts"], s_extra])
    ends = np.concatenate([inst["ends"], e_extra])
    caps = np.concatenate([inst["caps"], np.full(n, big)])
    costs = np.concatenate([inst["costs"], np.zeros(n, dtype=np.int64)])
    args = [supply, starts, ends, caps]
    if inst["minimums"] is not None:
        args.append(np.concatenate([inst["minimums"], np.zeros(n, dtype=np.int64)]))
    args.append(costs)
    flows = pylmcf_cpp.lmcf(*args)
    return int(np.dot(costs, flows))


# --- supply type ----------------------------------------------------------

def test_supply_type_default_and_roundtrip():
    g = Graph(2, np.array([0]), np.array([1]))
    assert g.supply_type() == "geq"
    g.set_supply_type("leq")
    assert g.supply_type() == "leq"
    g.set_supply_type("geq")
    assert g.supply_type() == "geq"
    with pytest.raises(ValueError, match="geq"):
        g.set_supply_type("eq")


def test_geq_potentials_not_artificial():
    # Node 0 must ship exactly its supply of 2 over an edge of capacity 2:
    # tight, saturated, and its artificial arc stays in LEMON's tree at zero
    # flow — raw potentials put -2^62 on it.  Canonical: 0 -> -1 -> 0.
    g = Graph(3, np.array([0, 1]), np.array([1, 2]))
    g.set_node_supply(np.array([2, 0, -5]))
    g.set_edge_capacities(np.array([2, 9]))
    g.set_edge_costs(np.array([4, 1]))
    g.solve()
    assert list(g.result()) == [2, 2]
    pi = g.potentials()
    assert np.abs(pi).max() < 100
    certify(g, {"n": 3, "starts": np.array([0, 1]), "ends": np.array([1, 2]),
                "supply": np.array([2, 0, -5]), "caps": np.array([2, 9]),
                "costs": np.array([4, 1]), "minimums": None}, "geq")


def test_supply_type_simple():
    g = Graph(3, np.array([0, 0, 1]), np.array([1, 2, 2]))
    g.set_edge_costs(np.array([1, 3, 5]))
    g.set_edge_capacities(np.array([3, 3, 5]))
    # 9 supplied, 5 demanded: only leq (unused supply allowed) is feasible.
    g.set_node_supply(np.array([9, 0, -5]))
    with pytest.raises(RuntimeError, match="INFEASIBLE"):
        g.solve()
    g.set_supply_type("leq")
    g.solve()
    assert list(g.result()) == [2, 3, 2]
    # 3 supplied, 9 demanded: only geq (unmet demand allowed) is feasible.
    g.set_node_supply(np.array([3, 0, -9]))
    with pytest.raises(RuntimeError, match="INFEASIBLE"):
        g.solve()
    g.set_supply_type("geq")
    g.solve()
    assert list(g.result()) == [0, 3, 0]


def test_set_supply_type_invalidates_result():
    g = Graph(2, np.array([0]), np.array([1]))
    g.set_node_supply(np.array([1, -1]))
    g.set_edge_capacities(np.array([1]))
    g.set_edge_costs(np.array([1]))
    g.solve()
    g.set_supply_type("leq")
    with pytest.raises(RuntimeError, match="solve"):
        g.result()


@pytest.mark.parametrize("with_minimums", [False, True])
@pytest.mark.parametrize("supply_type", ["geq", "leq"])
@pytest.mark.parametrize("seed", range(25))
def test_supply_type_random(seed, supply_type, with_minimums):
    rng = np.random.default_rng(3000 + seed)
    n = int(rng.integers(2, 25))
    m = int(rng.integers(1, 4 * n))
    inst = unbalance(rng, random_instance(rng, n, m, with_minimums), supply_type)
    g = build(inst, supply_type)
    g.solve()
    certify(g, inst, supply_type)
    assert g.total_cost() == slack_oracle_cost(inst, supply_type)
    # LEMON's raw tree potentials carry its 2^62 artificial cost on
    # roughly 40% of these geq instances; canonical ones are path-bounded.
    assert np.abs(g.potentials()).max() <= inst["costs"].sum()


def redraw(rng, inst):
    """Same topology and costs, new witness flow -> new supplies and caps."""
    m = len(inst["starts"])
    wit = rng.integers(0, 13, m)
    inst = dict(inst)
    inst["supply"] = np.zeros(inst["n"], dtype=np.int64)
    np.add.at(inst["supply"], inst["starts"], wit)
    np.add.at(inst["supply"], inst["ends"], -wit)
    inst["caps"] = wit + rng.integers(0, 9, m)
    return inst


@pytest.mark.parametrize("supply_type", ["geq", "leq"])
def test_supply_type_resolve_chain(supply_type):
    # Unbalanced re-solves cannot warm-restart; they must still be right,
    # including when the chain alternates with balanced instances.
    rng = np.random.default_rng(3500)
    base = random_instance(rng, 15, 50)
    g = build(base, supply_type)
    for step in range(10):
        inst = redraw(rng, base)
        kind = "eq"
        if step % 2 == 0:
            inst, kind = unbalance(rng, inst, supply_type), supply_type
        push(g, inst)
        g.solve()
        # Certify under the graph's own supply type even on balanced steps:
        # it picks the canonical sign (and with zero total supply its slack
        # check already forces equality).
        certify(g, inst, supply_type)
        if kind == "eq":
            ref = build(inst)
            ref.solve()
            assert g.total_cost() == ref.total_cost()
        else:
            assert g.total_cost() == slack_oracle_cost(inst, supply_type)


# --- pivot rule -----------------------------------------------------------

def test_pivot_rule_default_and_roundtrip():
    g = Graph(2, np.array([0]), np.array([1]))
    assert g.pivot_rule() == "block_search"
    for rule in PIVOT_RULES:
        g.set_pivot_rule(rule)
        assert g.pivot_rule() == rule
    with pytest.raises(ValueError, match="block_search"):
        g.set_pivot_rule("bland")


@pytest.mark.parametrize("rule", PIVOT_RULES)
@pytest.mark.parametrize("seed", range(10))
def test_pivot_rule_warm_chain(rule, seed):
    rng = np.random.default_rng(4000 + seed)
    n = int(rng.integers(3, 25))
    m = int(rng.integers(n, 4 * n))
    inst = random_instance(rng, n, m)
    g = build(inst)
    g.set_pivot_rule(rule)
    for step in range(6):
        if step:
            inst = redraw(rng, inst)
            if step % 3 == 0:
                inst["costs"] = rng.integers(0, 51, m)
            push(g, inst)
        g.solve()
        certify(g, inst)
        ref = build(inst)
        ref.solve()
        assert g.total_cost() == ref.total_cost()


# --- warm repair strategy and budget --------------------------------------

def chain_graph(rng, n, m, steps, configure):
    """A balanced cap/supply re-solve chain (costs fixed); certifies each step."""
    inst = random_instance(rng, n, m)
    g = build(inst)
    configure(g)
    g.solve()
    for _ in range(steps):
        wit = rng.integers(0, 13, m)
        inst = dict(inst)
        inst["supply"] = np.zeros(n, dtype=np.int64)
        np.add.at(inst["supply"], inst["starts"], wit)
        np.add.at(inst["supply"], inst["ends"], -wit)
        inst["caps"] = wit + rng.integers(0, 5, m)
        push(g, inst)
        g.solve()
        certify(g, inst)
    return g


def counters(g):
    return {"warm": g.warm_start_count(), "dual": g.dual_repair_count(),
            "primal": g.primal_repair_count(), "cold": g.cold_start_count()}


def test_warm_repair_default_and_roundtrip():
    g = Graph(2, np.array([0]), np.array([1]))
    assert g.warm_repair() == "dual"
    for strategy in WARM_REPAIRS:
        g.set_warm_repair(strategy)
        assert g.warm_repair() == strategy
    with pytest.raises(ValueError, match="dual_ratio"):
        g.set_warm_repair("magic")


@pytest.mark.parametrize("strategy", WARM_REPAIRS)
def test_warm_repair_strategy(strategy):
    rng = np.random.default_rng(5000)
    g = chain_graph(rng, 60, 240, 25, lambda g: g.set_warm_repair(strategy))
    c = counters(g)
    assert sum(c.values()) == 25
    # Each strategy reaches only its own repair counter.
    if strategy == "repair_only":
        assert c["dual"] == 0 and c["primal"] == 0
    elif strategy == "primal":
        assert c["dual"] == 0 and c["primal"] > 0
    else:
        assert c["primal"] == 0
    if strategy in ("dual_ratio", "dual_greedy"):
        assert c["dual"] > 0


@pytest.mark.skipif("PYLMCF_WARM_REPAIR_BUDGET" in os.environ,
                    reason="PYLMCF_WARM_REPAIR_BUDGET overrides the setter")
def test_warm_repair_budget_roundtrip():
    g = Graph(2, np.array([0]), np.array([1]))
    assert g.warm_repair_budget() == 64.0
    g.set_warm_repair_budget(2.5)
    assert g.warm_repair_budget() == 2.5
    g.set_warm_repair_budget(0)
    assert g.warm_repair_budget() == 0


@pytest.mark.skipif("PYLMCF_WARM_REPAIR_BUDGET" in os.environ,
                    reason="PYLMCF_WARM_REPAIR_BUDGET overrides the setter")
def test_warm_repair_budget_forces_cold_fallback():
    # dual_ratio repairs every step of this chain under the default budget;
    # a vanishing budget makes repairs bail to the cold fallback, and the
    # results stay correct (chain_graph certifies every step).
    def configure(budget):
        def f(g):
            g.set_warm_repair("dual_ratio")
            g.set_warm_repair_budget(budget)
        return f
    roomy = counters(chain_graph(np.random.default_rng(5100), 60, 240, 25, configure(0)))
    tight = counters(chain_graph(np.random.default_rng(5100), 60, 240, 25, configure(1e-9)))
    assert roomy["cold"] == 0
    assert tight["cold"] > 0


# --- infeasibility certificate --------------------------------------------

def check_barrier(inst, cut, supply_type):
    """The cut proves infeasibility: B must ship more than its boundary allows."""
    s_in, e_in = cut[inst["starts"]], cut[inst["ends"]]
    lo = inst["minimums"] if inst["minimums"] is not None else np.zeros_like(inst["caps"])
    leaving, entering = s_in & ~e_in, e_in & ~s_in
    if supply_type == "geq":
        assert inst["caps"][leaving].sum() - lo[entering].sum() < inst["supply"][cut].sum()
    else:
        assert inst["caps"][entering].sum() - lo[leaving].sum() < -inst["supply"][cut].sum()


def test_infeasibility_cut_simple():
    g = Graph(3, np.array([0, 0, 1]), np.array([1, 2, 2]))
    g.set_edge_costs(np.array([1, 3, 5]))
    g.set_edge_capacities(np.array([3, 3, 5]))
    g.set_node_supply(np.array([5, 0, -5]))
    assert g.infeasibility_cut() is None
    # Node 0 must ship 9 but its out-edges carry at most 6.
    g.set_node_supply(np.array([9, 0, -9]))
    cut = g.infeasibility_cut()
    assert cut.dtype == np.bool_
    assert list(cut) == [True, False, False]
    with pytest.raises(RuntimeError, match="infeasibility_cut"):
        g.solve()


def test_infeasibility_cut_leaves_solution_intact():
    g = Graph(2, np.array([0]), np.array([1]))
    g.set_node_supply(np.array([1, -1]))
    g.set_edge_capacities(np.array([1]))
    g.set_edge_costs(np.array([3]))
    g.solve()
    assert g.infeasibility_cut() is None
    assert g.total_cost() == 3
    assert list(g.result()) == [1]


@pytest.mark.parametrize("with_minimums", [False, True])
@pytest.mark.parametrize("supply_type", ["geq", "leq"])
@pytest.mark.parametrize("seed", range(30))
def test_infeasibility_cut_random(seed, supply_type, with_minimums):
    # Squeeze capacities of a feasible instance until it may break, sometimes
    # also imbalancing it the wrong way; the cut must exist exactly when
    # solve() fails, and must be a valid barrier.
    rng = np.random.default_rng(6000 + seed)
    n = int(rng.integers(2, 20))
    m = int(rng.integers(1, 3 * n))
    inst = random_instance(rng, n, m, with_minimums)
    if seed % 5 == 4:
        # Imbalance the supply type cannot absorb (excess supply under geq,
        # excess demand under leq): infeasible whatever the capacities.
        inst = unbalance(rng, inst, "leq" if supply_type == "geq" else "geq")
    elif seed % 3:
        inst = unbalance(rng, inst, supply_type)
    lo = inst["minimums"] if with_minimums else 0
    inst["caps"] = np.maximum(lo, inst["caps"] - rng.integers(0, 8, m))
    g = build(inst, supply_type)
    cut = g.infeasibility_cut()
    try:
        g.solve()
        solved = True
    except RuntimeError as e:
        assert "INFEASIBLE" in str(e)
        solved = False
    assert solved == (cut is None)
    if cut is not None:
        assert cut.shape == (n,)
        check_barrier(inst, cut, supply_type)
