# Changelog

## 1.3.0 (unreleased)

Changes since 1.2.1.

### Added

- Python bindings for the existing C++ LCT and chain solvers:
  `NetworkSimplexLCT`, `NetworkSimplexLCTDyn`, `lmcf_lct`, `lmcf_lct_dyn`, and
  `solve_chain_1d`. The dynamic-tree variant remains experimental.
- Canonical dual potentials through `Graph.potentials()` and
  `return_potentials=True` on all four functional LEMON solver APIs.
- Python controls for supply type, pivot rule, warm-repair strategy, repair time
  budget, and violation limit, plus warm/cold repair counters and
  `Graph.infeasibility_cut()` for diagnosing infeasible problems.
- Read-only input array support across the Python solver APIs, including NumPy
  memory maps. Inputs still require the correct dtype, contiguity, and alignment;
  returned flow and potential arrays remain writable.
- Documentation for the Python solver APIs, C++ headers, warm restarts, LCT and
  chain solvers, build modes, and testing.

### Fixed

- Rejected `Graph` setters leave the problem and retained solution unchanged.
  Previously, an invalid update could partially overwrite a map and cause a
  later warm solve to return a stale, suboptimal flow.
- Default zero capacities are now also applied to the underlying solver, so
  solving agrees with capacity getters and infeasibility cuts before capacities
  are explicitly set.
- Minimum flows above capacities are rejected instead of allowing out-of-bounds
  solver results. Negative node counts are also rejected.
- Empty problems solve at cost zero. Requesting an infeasibility cut on a
  zero-node graph returns `None` without accessing an empty LEMON node list.
- Misaligned input arrays are rejected before C++ dereferences them. Use
  `arr.copy()` or `np.require(arr, requirements="CA")` to realign them.
- Supplies equal to the input dtype's minimum are rejected because LEMON negates
  them. Network-simplex edge costs above `2**62 - 1` are rejected instead of
  incorrectly reporting feasible problems as infeasible.
- NetworkX conversion preserves parallel arcs, supports `MultiDiGraph` input,
  rejects undirected graphs, and rejects fractional attribute values instead of
  silently truncating them.

### Changed

- **NetworkX migration:** `Graph.as_nx()` now exports edge costs as `weight` and
  minimum flows as `lower_bound`, matching `Graph.FromNX()` defaults. Update
  consumers of the old `cost` and `minimum` keys. Node `demand` remains the
  negative of supply. Export returns a `MultiDiGraph` when parallel arcs exist.
- Removed a redundant cost-map copy on each `Graph.solve()` call.
- C++ `ChainSolver1D::solve()` skips allocating and extracting unused output
  flows; `solveFull()` still returns all per-arc flows.
- Expanded C++ oracle coverage and added five sanitizer/hardening CI lanes.

### Numerical range contract

Supplies, capacities, minimums, and flows retain the caller's integer dtype.
Total supply and intermediate flow arithmetic must fit that dtype with room to
spare; path costs, potentials, and the total objective must also fit their
arithmetic types. These aggregate ranges remain unchecked. A capacity equal to
the dtype's maximum means unbounded in LEMON. Use a wider dtype when necessary.
