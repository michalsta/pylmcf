## pylmcf: Python bindings for Min Cost Flow solvers from the LEMON graph library

### Overview

`pylmcf` provides Python bindings for the min-cost flow solvers implemented in the [LEMON graph library](https://lemon.cs.elte.hu/trac/lemon). It enables efficient network flow optimization in Python applications. It is used by [wnet](https://github.com/michalsta/wnet) (a Python package enabling the efficient computation of Wasserstein and Truncated Wasserstein distance between multidimensional distributions) and [wnetalign](https://github.com/michalsta/wnetalign) (a Python package enabling efficient alignment of MS or NMR spectra).


### Features

- Fast min-cost flow computation using C++ backend (LEMON's Network Simplex by default)
- Multiple solver variants: Network Simplex, Cycle Canceling, Cost Scaling, Capacity Scaling
- Supports capacities, costs, supplies/demands, and per-edge lower bounds (minimum flow)
- NetworkX integration: construct from `nx.DiGraph` or convert results back for visualization
- C++ headers exposed for downstream packages that want to call the solver directly

### Installation

```bash
pip install pylmcf
```

Optional extras for NetworkX support and visualization:

```bash
pip install pylmcf[extras]
```

### Usage

#### Basic usage

```python
import numpy as np
import pylmcf

# 3-node graph with edges 0→1, 0→2, 1→2
# Edges must be sorted by (start, end)
G = pylmcf.Graph(3,
    edge_starts=np.array([0, 0, 1]),
    edge_ends=np.array([1, 2, 2]))

G.set_node_supply(np.array([5, 0, -5]))   # node 0 supplies 5, node 2 demands 5
G.set_edge_costs(np.array([1, 3, 5]))
G.set_edge_capacities(np.array([3, 3, 5]))

G.solve()

G.result()      # np.array([2, 3, 2])  — flow on each edge
G.total_cost()  # 21
```

#### Per-edge lower bounds (minimum flow)

```python
G.set_edge_minimums(np.array([3, 0, 0]))  # edge 0→1 must carry at least 3 units
G.solve()
G.result()      # np.array([3, 2, 3])
G.total_cost()  # 24
```

#### Constructing from a NetworkX graph

> **Note:** `Graph.FromNX()` iterates over graph elements in Python and is significantly slower than constructing `Graph` directly from numpy arrays. Prefer the direct API for performance-sensitive code.

```python
import networkx as nx

G_nx = nx.DiGraph()
G_nx.add_edge(0, 1, weight=1, capacity=3)
G_nx.add_edge(0, 2, weight=3, capacity=3)
G_nx.add_edge(1, 2, weight=5, capacity=5)
G_nx.nodes[0]["demand"] = -5
G_nx.nodes[2]["demand"] = 5

G = pylmcf.Graph.FromNX(G_nx)
G.solve()
G.result()      # np.array([2, 3, 2])
```

#### Alternative solvers

The default solver is LEMON's Network Simplex. Three alternatives are available via the low-level functional API:

```python
from pylmcf import pylmcf_cpp
import numpy as np

a = lambda x: np.array(x, dtype=np.int64)
supply = a([5, 0, -5])
starts = a([0, 0, 1])
ends   = a([1, 2, 2])
caps   = a([3, 3, 5])
costs  = a([1, 3, 5])

flows = pylmcf_cpp.lmcf(supply, starts, ends, caps, costs)               # Network Simplex (default)
flows = pylmcf_cpp.lmcf_cycle_canceling(supply, starts, ends, caps, costs)
flows = pylmcf_cpp.lmcf_cost_scaling(supply, starts, ends, caps, costs)   # int32/int64 only
flows = pylmcf_cpp.lmcf_capacity_scaling(supply, starts, ends, caps, costs)  # int32/int64 only
```

#### Visualization

```python
G.show()               # display with matplotlib
G.show("graph.png")    # save to file
```

#### C++ include path

If you want to call the LEMON solver from your own C++ extension:

```bash
python -m pylmcf --include
```

### Requirements

- Python 3.9+

### Licence

pylmcf is published under the Boost Software Licence.
LEMON (which resides in `src/pylmcf/cpp/lemon`) is also covered by the Boost Software Licence.

### Citation

If you use this software, please cite:

Król J, Bochenek M, Jopa S, Kazimierczuk K, Gambin A, Startek MP (2026).
WNetAlign: fast and accurate spectra alignment using truncated Wasserstein distance and network simplex.
*Briefings in Bioinformatics*, 27(3), bbag247.
https://doi.org/10.1093/bib/bbag247

```bibtex
@article{krol2026wnetalign,
  title   = {WNetAlign: fast and accurate spectra alignment using truncated Wasserstein distance and network simplex},
  author  = {Kr{\'o}l, Justyna and Bochenek, Maria and Jopa, Sylwia and Kazimierczuk, Krzysztof and Gambin, Anna and Startek, Micha{\l} Piotr},
  journal = {Briefings in Bioinformatics},
  volume  = {27},
  number  = {3},
  pages   = {bbag247},
  year    = {2026},
  doi     = {10.1093/bib/bbag247}
}
```

### References

- [LEMON Graph Library](https://lemon.cs.elte.hu/trac/lemon)
- [wnet package](https://github.com/michalsta/wnet)
- [wnetalign package](https://github.com/michalsta/wnetalign)
- [Min Cost Flow Problem](https://en.wikipedia.org/wiki/Minimum-cost_flow_problem)
