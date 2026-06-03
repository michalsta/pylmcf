"""Smoke tests for the three non-default solver variants.

All four lmcf_impl instantiations (NetworkSimplex, CycleCanceling,
CostScaling, CapacityScaling) are parametrized over the same problems.
They must agree on total cost and, where the problem has a unique optimal
flow, on the flow vector itself.
"""

import numpy as np
import pytest
from pylmcf import pylmcf_cpp


SOLVERS = {
    "network_simplex":   pylmcf_cpp.lmcf,
    "cycle_canceling":   pylmcf_cpp.lmcf_cycle_canceling,
    "cost_scaling":      pylmcf_cpp.lmcf_cost_scaling,
    "capacity_scaling":  pylmcf_cpp.lmcf_capacity_scaling,
}


def solve(solver_name, supply, starts, ends, caps, costs, mins=None):
    """Call the named solver; return (flows, total_cost)."""
    fn = SOLVERS[solver_name]
    a = lambda x: np.asarray(x, dtype=np.int64)
    if mins is not None:
        flows = fn(a(supply), a(starts), a(ends), a(caps), a(mins), a(costs))
    else:
        flows = fn(a(supply), a(starts), a(ends), a(caps), a(costs))
    cost = int(np.dot(flows, costs))
    return flows, cost


@pytest.mark.parametrize("solver", list(SOLVERS))
class TestSimpleGraph:
    """3-node, 3-edge graph with a unique optimal flow."""

    supply = [5, 0, -5]
    starts = [0, 0, 1]
    ends   = [1, 2, 2]
    caps   = [3, 3, 5]
    costs  = [1, 3, 5]
    # Unique optimum: send 3 units via 0→1→2 (cost 1+5=6 each) and
    # 2 units via 0→2 (cost 3 each) — wait, let's just check what NS gives.
    # NS gives flows=[2,3,2], cost=21.
    expected_flows = [2, 3, 2]
    expected_cost  = 21

    def test_flows_and_cost(self, solver):
        flows, cost = solve(solver, self.supply, self.starts, self.ends,
                            self.caps, self.costs)
        assert cost == self.expected_cost
        assert list(flows) == self.expected_flows


@pytest.mark.parametrize("solver", list(SOLVERS))
class TestLowerBounds:
    """Same graph with a lower-bound forcing min flow on edge 0→1."""

    supply = [5, 0, -5]
    starts = [0, 0, 1]
    ends   = [1, 2, 2]
    caps   = [3, 3, 5]
    costs  = [1, 3, 5]
    mins   = [3, 0, 0]
    expected_flows = [3, 2, 3]
    expected_cost  = 24

    def test_flows_and_cost(self, solver):
        flows, cost = solve(solver, self.supply, self.starts, self.ends,
                            self.caps, self.costs, self.mins)
        assert cost == self.expected_cost
        assert list(flows) == self.expected_flows


@pytest.mark.parametrize("solver", list(SOLVERS))
class TestLargerGraph:
    """5-node graph; check only that all solvers agree on cost."""

    supply = [10, 0, 0, 0, -10]
    starts = [0, 0, 1, 1, 2, 3]
    ends   = [1, 2, 2, 3, 4, 4]
    caps   = [6, 8, 5, 7, 9, 6]
    costs  = [2, 4, 1, 3, 2, 5]

    def test_cost_agrees_with_network_simplex(self, solver):
        _, ref_cost = solve("network_simplex", self.supply, self.starts,
                            self.ends, self.caps, self.costs)
        _, cost = solve(solver, self.supply, self.starts, self.ends,
                        self.caps, self.costs)
        assert cost == ref_cost


@pytest.mark.parametrize("solver", list(SOLVERS))
class TestInfeasibleRaises:
    """A graph with no feasible flow must raise."""

    # Node 0 supplies 5, node 2 demands 5, but the only path has capacity 3.
    supply = [5, 0, -5]
    starts = [0, 1]
    ends   = [1, 2]
    caps   = [3, 3]
    costs  = [1, 1]

    def test_raises_on_infeasible(self, solver):
        with pytest.raises(RuntimeError, match="INFEASIBLE"):
            solve(solver, self.supply, self.starts, self.ends,
                  self.caps, self.costs)


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
def test_narrow_dtype_cost_overflow(dtype):
    """Costs are accumulated in a wide type, so narrow flow dtypes don't
    corrupt the optimum even when path-cost sums exceed the dtype's range.

    Every individual value fits in int8 (<=127), but the cheap path
    0->1->2->3->4 sums to 160 (> int8 max 127). The direct edge 0->4 costs
    120, so the optimum routes everything through it. Before costs were
    widened, int8 overflowed the path potential and spuriously failed.
    """
    a = lambda x: np.asarray(x, dtype=dtype)
    fn = pylmcf_cpp.lmcf
    flows = fn(
        a([10, 0, 0, 0, -10]),          # supply
        a([0, 0, 1, 2, 3]),             # starts: 0->1, 0->4, 1->2, 2->3, 3->4
        a([1, 4, 2, 3, 4]),             # ends
        a([10, 10, 10, 10, 10]),        # capacities
        a([40, 120, 40, 40, 40]),       # costs: path sum 160 vs direct 120
    )
    assert list(flows) == [0, 10, 0, 0, 0]
