## pylmcf: Python bindings for Min Cost Flow algorithm from LEMON graph library

### Overview

`pylmcf` provides Python bindings for the Min Cost Flow algorithm implemented in the [LEMON graph library](https://lemon.cs.elte.hu/trac/lemon). It enables efficient network flow optimization in Python applications. It is used by [wnet](https://github.com/michalsta/wnet) (a Python package enabling the efficient computation of Wasserstein and Truncated Wassersten distance between multidimensional distributions) and [wnetalign](https://github.com/michalsta/wnetalign) (A Python package enabling efficient alignment of MS or NMR spectra)


### Features

- Fast min cost flow computation using C++ backend
- Easy-to-use Python API
- Supports directed graphs, capacities, costs, and supplies/demands

### Installation

```bash
pip install pylmcf
```

### Usage

```python
import numpy as np
import pylmcf

# Create a graph with 3 nodes, labelled 0, 1, 2, and edges 0->1, 0->2, 1->2
no_nodes = 3
node_supply = np.array([5, 0, -5])

edge_starts     = np.array([0, 0, 1])
edge_ends       = np.array([1, 2, 2])
edge_costs      = np.array([1, 3, 5])
edge_capacities = np.array([1, 2, 3])

G = pylmcf.Graph(3, edge_starts, edge_ends)
G.set_node_supply(node_supply)
G.set_edge_costs(edge_costs)
G.set_edge_capacities(edge_capacities)

G.show()

# Run the Min Cost Flow algorithm
G.solve()

# Retrieve the flow values
G.result()
# returns: np.array([1, 2, 1]))

# Retrieve the total cost of the flow
G.total_cost()
# == 12

```

### Requirements

- Python 3.7+

### Licence

pylmcf is published under Boost licence.
LEMON (which resides in src/pylmcf/cpp/lemon subdirectory) is also covered by Boost licence.

### References

- [LEMON Graph Library](https://lemon.cs.elte.hu/trac/lemon)
- [wnet package](https://github.com/michalsta/wnet)
- [wnetalign package](https://github.com/michalsta/wnetalign)
- [Min Cost Flow Problem](https://en.wikipedia.org/wiki/Minimum-cost_flow_problem)