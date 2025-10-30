import networkx as nx
from pylmcf.graph import Graph


def test_networkx_simple():
    import networkx as nx
    import numpy as np

    G_nx = nx.DiGraph()
    G_nx.add_edge(0, 1, cost=1, capacity=1)
    G_nx.add_edge(0, 2, cost=3, capacity=2)
    G_nx.add_edge(1, 2, cost=5, capacity=3)
    G_nx.nodes[0]["demand"] = -5
    G_nx.nodes[1]["demand"] = 0
    G_nx.nodes[2]["demand"] = 5

    G = Graph.FromNX(G_nx, demand="demand", capacity="capacity", weight="cost")
    G.solve()
    assert all(G.result() == np.array([1, 2, 1]))
    assert G.total_cost() == 12


def check_large_graph(seed, no_nodes, no_edges):
    import networkx as nx
    import numpy as np

    np.random.seed(seed)
    G_nx = nx.gnm_random_graph(no_nodes, no_edges, directed=True, seed=seed)

    for u, v in G_nx.edges():
        G_nx[u][v]["cost"] = np.random.randint(1, 10)
        G_nx[u][v]["capacity"] = np.random.randint(1, 20)

    for i in range(no_nodes):
        G_nx.nodes[i]["demand"] = np.random.randint(-10, 10)

    # Add one special node to make sure total demand is zero and saturation is possible
    G_nx.add_node(
        no_nodes, demand=-sum(G_nx.nodes[i]["demand"] for i in range(no_nodes))
    )

    for i in range(no_nodes):
        G_nx.add_edge(no_nodes, i, cost=100_000, capacity=no_nodes * no_edges * 100)
        G_nx.add_edge(i, no_nodes, cost=100_000, capacity=no_nodes * no_edges * 100)

    G = Graph.FromNX(G_nx, demand="demand", capacity="capacity", weight="cost")
    G.solve()
    flows = G.result()
    total_flow = sum(flows)
    assert total_flow >= 0  # Basic sanity check
    nx_cost = nx.min_cost_flow_cost(
        G_nx, demand="demand", capacity="capacity", weight="cost"
    )
    lmcf_cost = G.total_cost()
    G.show()
    assert (
        nx_cost == lmcf_cost
    ), f"Costs do not match: NetworkX={nx_cost}, PyLMCF={lmcf_cost}"


try:
    import pytest

    @pytest.mark.parametrize("seed", range(1, 10))
    @pytest.mark.parametrize(
        "no_nodes,no_edges",
        [
            (2, 2),
            (5, 10),
            (10, 20),
            (20, 50),
            (50, 100),
            (100, 200),
            (200, 500),
            (500, 1000),
            (1000, 2000),
            (2000, 5000),
        ],
    )
    def test_large_graph(seed, no_nodes, no_edges):
        print(
            f"Testing large graph with seed {seed}, nodes={no_nodes}, edges={no_edges}..."
        )
        check_large_graph(seed, no_nodes, no_edges)

except ImportError:
    raise
    pass

if __name__ == "__main__":
    test_networkx_simple()
    for seed in range(1, 1000):
        print(f"Testing large graph with seed {seed}...")
        check_large_graph(seed, no_nodes=4, no_edges=6)
