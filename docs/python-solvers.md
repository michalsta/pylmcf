# Python APIs for the LCT and chain solvers

All names below are available from both `pylmcf` and `pylmcf.pylmcf_cpp`.
All input arrays must be one-dimensional, contiguous CPU arrays with exact
`np.int64` dtype. Implicit dtype conversion is disabled. The wrappers own
copies of stateful inputs, and returned arrays own their data.

## LCT network simplex

`lmcf_lct(node_supply, edge_starts, edge_ends, capacities, costs)` and
`lmcf_lct_dyn(...)` return an int64 array of optimal edge flows in the input
edge order. Edges need not be sorted; parallel edges and self-loops are allowed.
The latter uses the **experimental** dynamic-tree solver.

For repeated solves, construct `NetworkSimplexLCT(...)` or
`NetworkSimplexLCTDyn(...)` with the same five arrays:

```python
import numpy as np
from pylmcf import NetworkSimplexLCT

solver = NetworkSimplexLCT(
    node_supply=np.array([5, 0, -5], dtype=np.int64),
    edge_starts=np.array([0, 0, 1], dtype=np.int64),
    edge_ends=np.array([1, 2, 2], dtype=np.int64),
    capacities=np.array([3, 3, 5], dtype=np.int64),
    costs=np.array([1, 3, 5], dtype=np.int64),
)
solver.solve()
assert solver.total_cost() == 21
solver.set_node_supply(np.array([4, 0, -4], dtype=np.int64))
solver.solve()
flows = solver.result()
```

- `solve(warm=True)` reuses an optimal basis when possible, falling back to a
  cold solve when repair fails. `solve(warm=False)` forces a cold solve.
- `set_node_supply(supply)` and `set_edge_capacities(capacities)` replace an
  entire array. Updates are validated before mutation. Successful updates
  invalidate the previous result but preserve the basis for repair.
- `result()` and `total_cost()` require a successful solve after the last update.
  An infeasible solve raises `RuntimeError` and invalidates the basis and result;
  a subsequent solve starts cold.
- `warm_start_count()` and `cold_start_count()` count solve attempts, including
  the initial solve and failed attempts. Their sum equals the number of completed
  attempts; these wrapper counters intentionally have the same semantics for
  both LCT implementations.

Supplies must sum to zero. Capacities and costs must be non-negative; lower
bounds are always zero. Costs and topology are immutable on an instance:
construct a new instance to change them. The dynamic variant falls back to cold
on capacity changes; supply changes can use its incremental warm path.

Malformed inputs raise `ValueError`, incompatible dtypes/shapes raise `TypeError`,
and inputs exceeding conservative int64 arithmetic bounds raise `OverflowError`.
The bounds cover internal artificial costs and path potentials as well as the
objective. The sum of capacities is limited to `INT64_MAX / 4`. Individual
capacities must be less than `INT64_MAX / 4`; aggregate positive and
negative supplies are each limited to that value. Passing `INT64_MAX` as an
infinite-capacity sentinel is unsupported. Use finite problem-specific bounds.

As with `Graph`, concurrent mutation/solving of a shared instance requires
caller synchronization. Separate instances own separate solver state.

## 1D chain

`solve_chain_1d(positions, empirical, theoretical, kappa)` solves the
[SimpleTrash chain LP](chain-solver-1d.md). Positions must be sorted
(non-decreasing, with duplicates allowed); negative positions are allowed.
Masses and the integer trash cost `kappa` must be non-negative. All three arrays
must have the same length. Empty inputs produce zero cost and empty flow arrays.

The result is a dictionary:

| Key | Meaning |
|---|---|
| `total_cost` | Python integer objective |
| `emp_in` | int64 flow from source to each position |
| `theo_out` | int64 flow from each position to sink |
| `gap` | int64 signed rightward flow across each consecutive gap; length `max(n-1, 0)` |
| `trash` | Python integer flow directly from source to sink |

The required flow is `max(sum(empirical), sum(theoretical))`. A matched unit can
travel along the chain or bypass it at cost `kappa`; unmatched mass also uses
that bypass. Thus this is the SimpleTrash LP, not an arbitrary lower-bound
network or a normalized Wasserstein distance. Tied optima can have different
per-arc flows from LEMON while having identical total cost.

Input validation rejects unsorted positions, negative masses/cost, mismatched
lengths, and values outside conservative arithmetic bounds before invoking C++.
