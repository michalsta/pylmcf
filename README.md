## pylmcf: min-cost-flow solvers for Python and C++

### Overview

`pylmcf` ships **two independent deliverables from one source tree**:

1. **A Python extension** — fast min-cost flow via [LEMON](https://lemon.cs.elte.hu/trac/lemon)'s
   solvers, with a stateful object API that **warm-restarts** on re-solve.
2. **A header-only C++ include tree** — LEMON extended with warm-restart
   machinery, plus pylmcf's own solvers: a link-cut tree, two link-cut-tree
   network simplex variants, and a specialised 1D-chain solver. Downstream C++
   code compiles against it directly.

It is used by [wnet](https://github.com/michalsta/wnet) (efficient Wasserstein and
Truncated Wasserstein distance between multidimensional distributions) and
[wnetalign](https://github.com/michalsta/wnetalign) (alignment of MS or NMR
spectra). Both re-solve the same graph hundreds of times with one thing moved,
which is what most of the machinery here exists to make cheap.

> Much of deliverable (2) is **not exposed to Python** — the link-cut-tree simplex
> variants and the 1D chain solver are header-only and reachable only from C++.

### Features

- Fast min-cost flow computation using a C++ backend (LEMON's Network Simplex by default)
- **Warm restarts**: re-solving after changing supplies, capacities or costs
  reuses the retained spanning-tree basis instead of rebuilding one
  — [docs](https://github.com/michalsta/pylmcf/blob/main/docs/warm-restart.md)
- Multiple solver variants: Network Simplex, Cycle Canceling, Cost Scaling, Capacity Scaling
- Supports capacities, costs, supplies/demands, and per-edge lower bounds (minimum flow)
- NetworkX integration: construct from `nx.DiGraph` or convert results back for visualization
- Free-threaded CPython support (a separate `cp315-abi3t` wheel)
- A [C++ header tree](https://github.com/michalsta/pylmcf/blob/main/docs/cpp-headers.md)
  for downstream packages, including solvers with no Python binding

### Documentation

| | |
|---|---|
| [Warm restarts](https://github.com/michalsta/pylmcf/blob/main/docs/warm-restart.md) | How re-solving reuses the basis, the repair strategies, the counters and the two policy knobs. Start here — it is on by default and affects every re-solve. |
| [The C++ header tree](https://github.com/michalsta/pylmcf/blob/main/docs/cpp-headers.md) | Consuming `src/pylmcf/cpp/` from your own C++, what is in it, and why the vendored LEMON must not be replaced wholesale. |
| [Link-cut trees](https://github.com/michalsta/pylmcf/blob/main/docs/link-cut-tree.md) | `LinkCutTree<Val>` — a standalone Sleator–Tarjan link-cut tree with path-sum, path-min-with-argmin and lazy path-add. |
| [LCT network simplex](https://github.com/michalsta/pylmcf/blob/main/docs/lct-network-simplex.md) | `NetworkSimplexLCT` and the experimental `NetworkSimplexLCTDyn`, plus the adapter that A/B-tests them against real LEMON. |
| [The 1D chain solver](https://github.com/michalsta/pylmcf/blob/main/docs/chain-solver-1d.md) | `ChainSolver1D` — successive shortest paths specialised to the 1D-chain LP. |
| [Build modes and threading](https://github.com/michalsta/pylmcf/blob/main/docs/build-modes.md) | nanobind split vs linked, free-threading, what is and is not thread-safe, wheels. |
| [Testing and diagnostics](https://github.com/michalsta/pylmcf/blob/main/docs/testing.md) | The two test suites (one of which CI does not run), and the diagnostic build flags. |

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

All integer arrays in the OO API are **int64**, and the dtype is not converted for
you — a mismatched array raises `TypeError`. Costs and minimums must be
non-negative, and the graph must be feasible (total supply == total demand,
sufficient capacity) or `solve()` raises `RuntimeError: INFEASIBLE`.

#### Per-edge lower bounds (minimum flow)

```python
G.set_edge_minimums(np.array([3, 0, 0]))  # edge 0→1 must carry at least 3 units
G.solve()
G.result()      # np.array([3, 2, 3])
G.total_cost()  # 24
```

#### Re-solving: warm restarts

Changing supplies, capacities or costs and calling `solve()` again reuses the
basis from the previous solve. Nothing to switch on:

```python
H = pylmcf.Graph(3,
    edge_starts=np.array([0, 0, 1]),
    edge_ends=np.array([1, 2, 2]))
H.set_edge_costs(np.array([1, 3, 5]))
H.set_edge_capacities(np.array([3, 3, 5]))

for d in (5, 4, 6, 2):
    H.set_node_supply(np.array([d, 0, -d]))
    H.solve()
    print(d, H.result(), H.total_cost())

H.warm_start_count()    # 2
H.cold_start_count()    # 0
H.dual_repair_count()   # 1
```

```
5 [2 3 2] 21
4 [1 3 1] 15
6 [3 3 3] 27
2 [0 2 0] 6
```

The first `solve()` is always cold and is not counted. The counters are how you
tell whether warm restarts are actually firing — a regression that silently routes
every re-solve through a cold rebuild still gives correct answers. Nonzero lower
bounds force a cold solve.

Full details, the repair strategies, the `set_warm_violation_limit()` policy and
the C++ API: **[Warm restarts](https://github.com/michalsta/pylmcf/blob/main/docs/warm-restart.md)**.

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

The functional API is stateless and has no warm restart. It is duck-typed over
int8/16/32/64 (cost and capacity scaling excepted, which need the wider range),
and each function has a with- and without-minimums overload.

#### Visualization

```python
G.show()               # display with matplotlib
G.show("graph.png")    # save to file
```

#### Threading

Independent `Graph` objects may be solved concurrently, including on
free-threaded CPython — `tests/test_free_threading.py` hammers 8 threads against
a serial oracle. **A single `Graph` must not be driven from two threads at once**:
`solve()` mutates the instance, takes no lock, and no binding releases the GIL.
See [Build modes and threading](https://github.com/michalsta/pylmcf/blob/main/docs/build-modes.md#free-threaded-python).

### Using the C++ solvers

```bash
python -m pylmcf --include
```

prints the include root; there is nothing to link.

```bash
g++ -I$(python -m pylmcf --include) -std=c++20 -O2 mycode.cpp -o mycode
```

Beyond LEMON, the tree carries pylmcf's own header-only solvers — `LinkCutTree`,
`NetworkSimplexLCT`, the experimental `NetworkSimplexLCTDyn`, a
`lemon::NetworkSimplex`-compatible adapter, and `ChainSolver1D`. **None of these
have a Python binding.** See
**[The C++ header tree](https://github.com/michalsta/pylmcf/blob/main/docs/cpp-headers.md)**.

### Requirements

- Python 3.10+
- For a build from source (sdist, or a local checkout): a **C++20** compiler

### Licence

pylmcf is published under the Boost Software Licence.

`src/pylmcf/cpp/lemon` is a **modified** copy of the LEMON graph library, also
covered by the Boost Software Licence. `lemon/network_simplex.h` in particular
carries substantial pylmcf-specific work that upstream does not have — the entire
warm-restart machinery — so it **must not be replaced with a stock LEMON
release**. Diff, do not overwrite; see
[the header-tree docs](https://github.com/michalsta/pylmcf/blob/main/docs/cpp-headers.md#the-vendored-lemon-is-modified).

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
- Sleator, D. D. and Tarjan, R. E. (1983). *A data structure for dynamic trees.*
  Journal of Computer and System Sciences 26(3), 362–391.
- Cunningham, W. H. (1976). *A network simplex method.* Mathematical Programming
  11(1), 105–116. — the strongly-feasible-basis anti-cycling rule both LCT
  solvers reproduce.
