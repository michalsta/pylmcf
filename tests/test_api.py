"""
API coverage tests for graph.py and __version__.py.
Targets: as_nx(), FromNX() edge cases, include(), lower-bound path.
"""
import numpy as np
import pytest

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

from pylmcf import include
from pylmcf.graph import Graph


def _simple_graph(solved=False):
    G = Graph(3, np.array([0, 0, 1]), np.array([1, 2, 2]))
    G.set_edge_costs(np.array([1, 3, 5]))
    G.set_edge_capacities(np.array([3, 3, 5]))
    G.set_node_supply(np.array([5, 0, -5]))
    if solved:
        G.solve()
    return G


# ---------------------------------------------------------------------------
# __version__.include()
# ---------------------------------------------------------------------------

def test_include_returns_existing_path():
    p = include()
    assert p.exists()
    assert p.is_dir()


# ---------------------------------------------------------------------------
# Graph.as_nx() — unsolved
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_NX, reason="networkx not installed")
def test_as_nx_unsolved_structure():
    G = _simple_graph(solved=False)
    nx_g = G.as_nx()
    assert nx_g.number_of_nodes() == 3
    assert nx_g.number_of_edges() == 3
    for _, _, data in nx_g.edges(data=True):
        assert "capacity" in data
        assert "cost" in data
        assert "flow" not in data


@pytest.mark.skipif(not HAS_NX, reason="networkx not installed")
def test_as_nx_solved_includes_flow():
    G = _simple_graph(solved=True)
    nx_g = G.as_nx()
    assert nx_g.number_of_nodes() == 3
    for _, _, data in nx_g.edges(data=True):
        assert "flow" in data
        assert "label" in data


# ---------------------------------------------------------------------------
# Graph.FromNX() — error path and lower-bound path
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_NX, reason="networkx not installed")
def test_from_nx_rejects_non_contiguous_nodes():
    G_nx = nx.DiGraph()
    G_nx.add_node(0)
    G_nx.add_node(2)   # gap: node 1 missing
    G_nx.add_edge(0, 2, weight=1, capacity=1)
    with pytest.raises(ValueError, match="contiguous"):
        Graph.FromNX(G_nx)


@pytest.mark.skipif(not HAS_NX, reason="networkx not installed")
def test_from_nx_with_lower_bounds():
    G_nx = nx.DiGraph()
    G_nx.add_node(0, demand=-5)
    G_nx.add_node(1, demand=5)
    G_nx.add_edge(0, 1, weight=2, capacity=10, lower_bound=2)
    G = Graph.FromNX(G_nx, demand="demand", capacity="capacity",
                     lower_bound="lower_bound", weight="weight")
    G.solve()
    flow = G.result()
    assert flow[0] >= 2   # lower bound respected
    assert G.total_cost() == flow[0] * 2


@pytest.mark.skipif(not HAS_NX, reason="networkx not installed")
def test_from_nx_roundtrip():
    G_nx = nx.DiGraph()
    G_nx.add_node(0, demand=-3)
    G_nx.add_node(1, demand=3)
    G_nx.add_edge(0, 1, weight=7, capacity=5)
    G = Graph.FromNX(G_nx)
    G.solve()
    assert G.total_cost() == 3 * 7
