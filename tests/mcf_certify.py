# Shared optimality certificate for min-cost-flow results with potentials.
#
# Potentials are not unique, so tests never compare them against golden
# values.  A (flows, potentials) pair is instead certified from first
# principles, with rc = cost + pi[start] - pi[end]:
#   - primal feasibility: bounds on every edge, and per node
#       eq:  out - in == supply
#       geq: out - in >= supply
#       leq: out - in <= supply;
#   - complementary slackness on edges: rc > 0 => flow == minimum,
#     rc < 0 => flow == capacity;
#   - dual feasibility and complementary slackness on nodes: under geq
#     every pi <= 0, under leq every pi >= 0, and pi == 0 wherever the node
#     constraint has slack;
#   - strong duality: the dual objective computed from the potentials alone
#     equals the reported total cost.
# Together these prove both the flows and the potentials optimal.
#
# pylmcf further promises *canonical* potentials: among those optimal for the
# returned flows, the pointwise-largest with pi <= 0 (eq/geq) or the
# pointwise-smallest with pi >= 0 (leq).  That is checked against a plain
# Bellman-Ford over the residual graph, sharing nothing with the C++ path.

import numpy as np


def certify(n, starts, ends, supply, caps, costs, flows, pi, total_cost,
            minimums=None, supply_type="eq"):
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    supply = np.asarray(supply, dtype=np.int64)
    caps = np.asarray(caps, dtype=np.int64)
    costs = np.asarray(costs, dtype=np.int64)
    flows = np.asarray(flows, dtype=np.int64)
    pi = np.asarray(pi)
    assert pi.dtype == np.int64
    assert pi.shape == (n,)
    lo = (np.zeros_like(flows) if minimums is None
          else np.asarray(minimums, dtype=np.int64))

    assert np.all(flows >= lo) and np.all(flows <= caps)
    net = np.zeros(n, dtype=np.int64)
    np.add.at(net, starts, flows)
    np.add.at(net, ends, -flows)
    slack = net - supply
    if supply_type == "eq":
        assert np.all(slack == 0)
    elif supply_type == "geq":
        assert np.all(slack >= 0)
        assert np.all(pi <= 0)
    elif supply_type == "leq":
        assert np.all(slack <= 0)
        assert np.all(pi >= 0)
    else:
        raise ValueError(supply_type)
    assert np.all(pi[slack != 0] == 0), "potential nonzero on a slack node"

    rc = costs + pi[starts] - pi[ends]
    assert np.all(flows[rc > 0] == lo[rc > 0]), "rc > 0 on an edge above its minimum"
    assert np.all(flows[rc < 0] == caps[rc < 0]), "rc < 0 on an edge below capacity"

    assert np.dot(costs, flows) == total_cost
    dual = (-np.dot(supply, pi)
            + np.dot(np.minimum(rc, 0), caps)
            + np.dot(np.maximum(rc, 0), lo))
    assert dual == total_cost

    expected = bellman_ford_canonical(n, starts, ends, costs, caps, lo, flows,
                                      leq=(supply_type == "leq"))
    assert np.array_equal(pi, expected), "potentials are optimal but not canonical"


def bellman_ford_canonical(n, starts, ends, costs, caps, lo, flows, leq):
    """Extremal optimal potentials for `flows`, by Bellman-Ford.

    geq/eq: shortest residual-path distance from a virtual source with a
    0-cost arc into every node.  leq: minus the distance from each node to a
    virtual sink, i.e. the same on the reversed residual graph, negated.
    """
    fwd = flows < caps
    bwd = flows > lo
    tail = np.concatenate([starts[fwd], ends[bwd]])
    head = np.concatenate([ends[fwd], starts[bwd]])
    w = np.concatenate([costs[fwd], -costs[bwd]])
    if leq:
        tail, head = head, tail
    dist = np.zeros(n, dtype=np.int64)
    for _ in range(n + 1):
        cand = dist.copy()
        np.minimum.at(cand, head, dist[tail] + w)
        if np.array_equal(cand, dist):
            break
        dist = cand
    else:
        raise AssertionError("negative residual cycle: flows are not optimal")
    return -dist if leq else dist


def random_instance(rng, n, m, with_minimums=False):
    """Random feasible (balanced) instance; feasibility by witness construction."""
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
