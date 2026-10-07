from pylmcf.graph import Graph
import numpy as np
import pytest


def test_graph_simple():
    G = Graph(3, np.array([0, 0, 1]), np.array([1, 2, 2]))
    G.set_edge_costs(np.array([1, 3, 5]))
    G.set_edge_capacities(np.array([3, 3, 5]))
    G.set_node_supply(np.array([5, 0, -5]))
    # G.show()
    G.solve()
    assert all(G.result() == np.array([2, 3, 2]))
    assert G.total_cost() == 21


if __name__ == "__main__":
    test_graph_simple()


def test_default_capacities_are_zero_for_the_solver_too():
    # LEMON's own default upper bound is infinite; Graph's is zero, and the
    # solver, the getters and infeasibility_cut() must all agree on it.
    G = Graph(2, np.array([0]), np.array([1]))
    G.set_node_supply(np.array([1, -1]))
    G.set_edge_costs(np.array([1]))
    assert np.array_equal(G.get_edge_capacities(), [0])
    assert np.array_equal(G.infeasibility_cut(), [True, False])
    with pytest.raises(RuntimeError, match="INFEASIBLE"):
        G.solve()
    G.set_edge_capacities(np.array([1]))
    G.solve()
    assert np.array_equal(G.result(), [1])
    assert G.infeasibility_cut() is None
