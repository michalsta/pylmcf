# Changelog

## Unreleased

### Added

- A mass-based warm/cold policy: `setWarmViolationMassFraction(f)` /
  `Graph.set_warm_violation_mass_fraction(f)` skips simplex repair (cold
  start, counted in `policyColdCount()`) when the violated basic-arc mass left
  by the tree patch exceeds `f` times the total positive supply. Negative
  (default) disables it; it combines with the count limit, either of which
  can force the cold start. Measured on 2-D NMR deconvolution, repair beats a
  cold start only while that mass stays below roughly 0.5% of supply.
- Observability for warm/cold policies: `workUnits()` (deterministic work
  accumulated over all runs) and `lastViolationCount()`,
  `lastViolationMass()`, `lastPositiveSupply()` (the measurements of the last
  warm solve whose policy looked at them), with `Graph` counterparts.

### Changed

- NetworkSimplex's pivot loop now shifts the potentials of whichever side of
  the spanning tree is smaller after each pivot, instead of always the moved
  subtree. Both differ by a global constant that no reduced cost sees, so the
  pivots, flows and (after a final renormalization) potentials are exactly
  those of stock LEMON. On wnet's 2-D NMR networks, where a hub node makes
  moved subtrees large, it cuts cutting-plane oracle time by 8%.
  `setSmallerSidePotentials(false)` / `Graph.set_smaller_side_potentials(False)`
  or `PYLMCF_SMALLER_SIDE_POTENTIALS=0` restores the stock update;
  `complementPotentialUpdates()` counts the pivots that used the complement.
- The warm-repair budget (`setWarmRepairBudget()` / `Graph.set_warm_repair_budget()`)
  is now measured in deterministic work units instead of wall time, so a given
  input always takes the same repair/cold path. The wall-time budget could trip
  on an interrupt during the microsecond-scale solves of small subgraphs, which
  changed the optimal basis returned and made outer optimizers converge to
  different answers from run to run. The multiplier now scales the last cold
  solve's work; the default is 25.0 (roughly the old 64x cold time).
  In C++, `lastColdMs()` / `warmRepairLimitMs()` are replaced by
  `lastColdWork()` / `warmRepairWorkLimit()`.

## 1.3.0 (2026-10-09)

Changes since 1.2.1.

### Added

- Python bindings for the existing C++ LCT and chain solvers:
  `NetworkSimplexLCT`, `NetworkSimplexLCTDyn`, `lmcf_lct`, `lmcf_lct_dyn`, and
  `solve_chain_1d`. The dynamic-tree variant remains experimental.
- Canonical dual potentials through `Graph.potentials()` and
  `return_potentials=True` on all four functional LEMON solver APIs.
- Checked raw dual certificates on `Graph`, `NetworkSimplexLCT`, and
  `NetworkSimplexLCTDyn`, including reusable writable output buffers through
  `dual_values_into()`. The LCT functional APIs support `return_duals=True`;
  canonical potentials retain their existing contract.
- Python controls for supply type, pivot rule, warm-repair strategy, repair time
  budget, and violation limit, plus warm/cold repair counters and
  `Graph.infeasibility_cut()` for diagnosing infeasible problems.
- Read-only input array support across the Python solver APIs, including NumPy
  memory maps. Inputs still require the correct dtype, contiguity, and alignment;
  returned flow and potential arrays remain writable.
- Documentation for the Python solver APIs, C++ headers, warm restarts, LCT and
  chain solvers, build modes, and testing.

### Fixed

- CostScaling price refinement falls back to the regular scaling phase when an
  accumulated path rank exceeds its bucket range, avoiding memory corruption
  on valid problems with signed edge costs.
- Linked builds now produce wheels tagged for the building interpreter, including
  musl, 32-bit Windows, PyPy, older free-threaded Python, and forced linked builds.
  Previously, some linked wheels advertised `cp310-abi3` compatibility despite
  containing a Python-specific extension. Build mode, wheel tags, and runtime
  dependencies now share one selector, and conflicting ABI requests are rejected.

- LCT artificial costs account for the largest absolute real cost and use
  checked multiplication, avoiding false infeasibility and signed overflow.
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
