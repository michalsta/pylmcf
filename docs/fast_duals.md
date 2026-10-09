# Fast dual certificates

Version 1.3.0 exposes raw simplex dual values without canonicalization or a
residual shortest-path search. The existing `Graph.potentials()` continues to
return canonical potentials and retains its original contract.

```python
g.solve()
certificate = g.dual_values()
pi = certificate["potentials"]
rc = certificate["reduced_costs"]
lower = certificate["lower_bound_multipliers"]
upper = certificate["upper_bound_multipliers"]
```

The same methods exist on `NetworkSimplexLCT` and `NetworkSimplexLCTDyn`.
Individual snapshots are available as `raw_potentials()`, `reduced_costs()`,
`lower_bound_multipliers()` and `upper_bound_multipliers()`. All are fresh int64
arrays in the caller's node/edge order. The LCT functional APIs additionally
accept `return_duals=True`, returning `(flows, certificate)`.

For repeated extraction, preallocate arrays and call
`g.dual_values_into(pi, rc, lower, upper)`. Buffers must have respectively N, M,
M, M elements, be writable, aligned, contiguous int64 arrays, and not overlap.
The call allocates no result arrays. An integer overflow raises rather than
wrapping. Queries before a successful solve or after accepted input updates
raise; previously returned snapshots remain independent of the solver.

## Signs and certificate

For arc u→v with cost c, lower bound l and capacity U:

```
rc = c + pi[u] - pi[v]
lower = max(rc, 0)
upper = min(rc, 0)
dual_objective = -sum(supply * pi) + sum(l * lower) + sum(U * upper)
```

The upper multiplier uses a **nonpositive** sign convention. Positive reduced
cost requires flow at its lower bound; negative reduced cost requires flow at
capacity. For an optimal solution the dual objective equals the primal cost.
With inequality supplies, the usual node-potential sign and slack conditions
also apply. Raw potentials can contain artificial-root offsets near 2^62;
their differences and the complete certificate matter, not individual values.
Use Python integers or checked/wide arithmetic for products and sums. A raw
certificate need not equal the canonical certificate numerically.

## C++ consumers

LEMON `NetworkSimplex`, the LCT solver, the dynamic LCT solver and the LCT
adapter expose `potential`, `reducedCost`, `lowerBoundMultiplier`,
`upperBoundMultiplier`, and `dualValues(pi, rc, lower, upper)` with writable
`std::span<Cost>` buffers. Mapped accessors take LEMON nodes/arcs; the standalone
LCT solvers take integer IDs. Bulk output is indexed by graph IDs, which must
be dense; sparse-ID LEMON graphs should use mapped accessors. Calls are valid
after an OPTIMAL solve and until inputs change. Low-level accessors rely on
these preconditions and valid IDs; Python wrappers enforce them.

The ordinary LCT solver already materializes potentials after solving. The
dynamic variant materializes them lazily on the first dual query, then serves
O(1) reads until the next solve. Flow-only dynamic solves incur no export work.
Artificial costs use the largest absolute real cost (including negative
shifted matching costs) and checked multiplication. This prevents false
infeasibility from an underpriced artificial cycle and signed overflow.
Capacity multipliers are computed inline, not maintained during pivots. This
avoids additional work in solves that never request duals.

`experiments/benchmark_dual_api.py` compares raw and canonical NumPy
reconstruction, allocating snapshots, reusable-buffer extraction and solves.
It records repeated timing distributions and checks strong duality.
