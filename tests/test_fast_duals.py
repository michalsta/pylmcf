"""Raw dual certificates, without canonicalization or residual searches."""
import numpy as np
import pytest

from pylmcf import Graph, NetworkSimplexLCT, NetworkSimplexLCTDyn
from mcf_certify import random_instance

SOLVERS = [Graph, NetworkSimplexLCT, NetworkSimplexLCTDyn]


def build(cls, inst):
    args = [inst[k] for k in ("supply", "starts", "ends", "caps", "costs")]
    if cls is Graph:
        order = np.lexsort((args[2], args[1]))
        for k in ("starts", "ends", "caps", "costs"):
            inst[k] = np.ascontiguousarray(inst[k][order])
        g = Graph(inst["n"], inst["starts"], inst["ends"])
        g.set_node_supply(inst["supply"])
        g.set_edge_capacities(inst["caps"])
        g.set_edge_costs(inst["costs"])
        if inst["minimums"] is not None:
            inst["minimums"] = np.ascontiguousarray(inst["minimums"][order])
            g.set_edge_minimums(inst["minimums"])
        return g
    return cls(*args)


def certify(g, inst):
    data = g.dual_values()
    pi, rc, lower, upper = [data[k] for k in (
        "potentials", "reduced_costs", "lower_bound_multipliers", "upper_bound_multipliers")]
    assert all(a.dtype == np.int64 for a in data.values())
    expected = [int(c) + int(pi[u]) - int(pi[v]) for c, u, v in
                zip(inst["costs"], inst["starts"], inst["ends"])]
    assert rc.tolist() == expected
    assert np.array_equal(lower, np.maximum(rc, 0))
    assert np.array_equal(upper, np.minimum(rc, 0))
    assert np.array_equal(pi, g.raw_potentials())
    assert np.array_equal(rc, g.reduced_costs())
    assert np.array_equal(lower, g.lower_bound_multipliers())
    assert np.array_equal(upper, g.upper_bound_multipliers())
    flows = g.result()
    lo = inst["minimums"]
    if lo is None:
        lo = np.zeros_like(flows)
    assert np.all(flows[rc > 0] == lo[rc > 0])
    assert np.all(flows[rc < 0] == inst["caps"][rc < 0])
    # Python integers keep the certificate valid even with artificial offsets.
    dual = (-sum(int(b) * int(p) for b, p in zip(inst["supply"], pi))
            + sum(int(u) * int(p) for u, p in zip(inst["caps"], upper))
            + sum(int(l) * int(p) for l, p in zip(lo, lower)))
    assert dual == g.total_cost()
    buffers = [np.empty_like(a) for a in (pi, rc, lower, upper)]
    g.dual_values_into(*buffers)
    assert all(np.array_equal(a, b) for a, b in zip(buffers, (pi, rc, lower, upper)))
    return data


@pytest.mark.parametrize("cls", SOLVERS)
@pytest.mark.parametrize("seed", range(40))
def test_random_certificates_and_supply_changes(cls, seed):
    rng = np.random.default_rng(seed)
    inst = random_instance(rng, 8, 25)
    g = build(cls, inst)
    for _ in range(3):
        g.solve()
        certify(g, inst)
        witness = rng.integers(0, inst["caps"] + 1)
        supply = np.zeros(inst["n"], dtype=np.int64)
        np.add.at(supply, inst["starts"], witness)
        np.add.at(supply, inst["ends"], -witness)
        inst["supply"] = supply
        g.set_node_supply(supply)
        with pytest.raises(RuntimeError, match="solve"):
            g.dual_values()


@pytest.mark.parametrize("seed", range(15))
def test_lower_bounds(seed):
    inst = random_instance(np.random.default_rng(seed), 8, 25, with_minimums=True)
    g = build(Graph, inst)
    g.solve()
    certify(g, inst)


@pytest.mark.parametrize("cls", SOLVERS)
def test_degenerate_disconnected_and_buffer_validation(cls):
    inst = dict(n=5, supply=np.zeros(5, dtype=np.int64),
                starts=np.array([0, 2]), ends=np.array([1, 3]),
                caps=np.array([0, 0]), costs=np.array([3, 0]), minimums=None)
    g = build(cls, inst)
    with pytest.raises(RuntimeError, match="solve"):
        g.raw_potentials()
    g.solve()
    snapshot = certify(g, inst)
    buffers = [np.zeros(n, dtype=np.int64) for n in (5, 2, 2, 2)]
    with pytest.raises(ValueError, match="length"):
        g.dual_values_into(buffers[0][:1], *buffers[1:])
    with pytest.raises(ValueError, match="overlap"):
        g.dual_values_into(buffers[0], buffers[1], buffers[1], buffers[3])
    bad = np.empty(4, dtype=np.int64)[::2]
    with pytest.raises(ValueError, match="contiguous"):
        g.dual_values_into(buffers[0], bad, buffers[2], buffers[3])
    readonly = np.zeros(2, dtype=np.int64)
    readonly.flags.writeable = False
    with pytest.raises(TypeError):
        g.dual_values_into(buffers[0], readonly, buffers[2], buffers[3])
    g.set_node_supply(np.zeros(5, dtype=np.int64))
    with pytest.raises(RuntimeError, match="solve"):
        g.reduced_costs()
    assert all(np.array_equal(a, b) for a, b in zip(snapshot.values(), certify_after(g, inst).values()))


def certify_after(g, inst):
    g.solve()
    return certify(g, inst)


@pytest.mark.parametrize("cls", SOLVERS)
def test_empty_and_failed_solutions(cls):
    empty = np.zeros(0, dtype=np.int64)
    inst = dict(n=0, supply=empty, starts=empty, ends=empty, caps=empty,
                costs=empty, minimums=None)
    g = build(cls, inst)
    g.solve()
    assert all(a.size == 0 for a in g.dual_values().values())
    g.dual_values_into(*[empty.copy() for _ in range(4)])
    inst = dict(n=2, supply=np.array([1, -1]), starts=np.array([0]),
                ends=np.array([1]), caps=np.array([1]), costs=np.array([1]), minimums=None)
    g = build(cls, inst)
    g.solve()
    saved = g.dual_values()
    g.set_edge_capacities(np.array([0], dtype=np.int64))
    with pytest.raises(RuntimeError):
        g.solve()
    with pytest.raises(RuntimeError, match="solve"):
        g.dual_values()
    assert saved["reduced_costs"].shape == (1,)


@pytest.mark.parametrize("function", ["lmcf_lct", "lmcf_lct_dyn"])
def test_functional_duals_preserve_callers_edge_order(function):
    import pylmcf
    fn = getattr(pylmcf, function)
    supply, starts, ends, caps, costs = [np.array(a, dtype=np.int64) for a in
        ([5, 0, -5], [1, 0, 0], [2, 2, 1], [5, 3, 3], [5, 3, 1])]
    flows, data = fn(supply, starts, ends, caps, costs, return_duals=True)
    assert flows.tolist() == [2, 3, 2]
    pi = data["potentials"]
    assert data["reduced_costs"].tolist() == [
        int(c) + int(pi[u]) - int(pi[v]) for c, u, v in zip(costs, starts, ends)]
    assert np.array_equal(fn(supply, starts, ends, caps, costs), flows)


@pytest.mark.parametrize("supply_type,supply", [("geq", [4, 0, -5]), ("leq", [5, 0, -4])])
def test_inequality_supply_duals(supply_type, supply):
    inst = dict(n=3, supply=np.array(supply), starts=np.array([0, 0, 1]),
                ends=np.array([1, 2, 2]), caps=np.array([10, 1, 10]),
                costs=np.array([1, 0, 2]), minimums=None)
    g = build(Graph, inst)
    g.set_supply_type(supply_type)
    g.solve()
    data = certify(g, inst)
    pi = data["potentials"]
    net = np.zeros(3, dtype=np.int64)
    np.add.at(net, inst["starts"], g.result())
    np.add.at(net, inst["ends"], -g.result())
    assert np.all(pi <= 0) if supply_type == "geq" else np.all(pi >= 0)
    assert np.all(pi[net != inst["supply"]] == 0)
