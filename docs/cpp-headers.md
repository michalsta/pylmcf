# The C++ header tree

pylmcf ships two independent deliverables from one source tree:

1. **`pylmcf_cpp`** — the nanobind Python extension, which is what
   `pip install pylmcf` gives you.
2. **A header-only C++ include tree** at `src/pylmcf/cpp/`, which downstream C++
   code compiles against directly.

This page is about (2). It is how `wnet` consumes the solvers, and it is where
most of the recent development has happened.

**Most of this tree is not reachable from Python.** The link-cut-tree simplex
variants and the 1D chain solver are header-only and C++-only. Do not assume a
header listed here has a Python binding — check `src/pylmcf/cpp/pylmcf/pylmcf.cpp`
for the actual binding surface.

---

## Consuming it

```bash
python -m pylmcf --include
```

prints the include root. Everything is header-only from a consumer's point of
view — there is nothing to link.

```bash
g++ -I$(python -m pylmcf --include) -std=c++20 -O2 mycode.cpp -o mycode
```

In CMake:

```cmake
execute_process(
  COMMAND ${Python_EXECUTABLE} -m pylmcf --include
  OUTPUT_VARIABLE PYLMCF_INCLUDE
  OUTPUT_STRIP_TRAILING_WHITESPACE)
target_include_directories(mytarget PRIVATE ${PYLMCF_INCLUDE})
target_compile_features(mytarget PRIVATE cxx_std_20)
```

From Python, the same path is available as `pylmcf.include()`, which returns a
`pathlib.Path` (not a string):

```python
>>> import pylmcf
>>> pylmcf.include()
PosixPath('.../site-packages/pylmcf/cpp')
```

**C++20 is required**, not merely recommended — the headers use it.

---

## What is in it

```
src/pylmcf/cpp/
├── lemon/        the vendored (MODIFIED) LEMON graph library
└── pylmcf/       pylmcf's own headers
```

### Shipped to Python

These back the Python extension. You can include them, but the Python API is
usually the easier route.

| Header | Contents |
|---|---|
| `pylmcf/basics.hpp` | `LEMON_INT` (`int64_t`, the value type), `LEMON_INDEX` (`int`, node/arc ids), `assert_fits_lemon_index()`, `sorted_copy()` |
| `pylmcf/graph.hpp` | `Graph<T>` — `lemon::StaticDigraph` + `lemon::NetworkSimplex`, all solver state, warm restart wired into `solve()` |
| `pylmcf/lmcf.hpp` | the functional API (`lmcf_impl<Solver>`): raw spans in, flows written back |
| `pylmcf/pylmcf.cpp` | the nanobind entry point (a TU, not a header) |
| `pylmcf/py_support.hpp` | nanobind ndarray ↔ `std::span` conversion |

`Graph<T>`'s constructor requires edges **sorted by `(start_node, end_node)`** and
rejects negative or out-of-range node ids, throwing `std::invalid_argument`.

### Header-only, C++ consumers only

This is the novel work. None of it has a Python binding.

| Header | Contents | Docs |
|---|---|---|
| `pylmcf/link_cut_tree.h` | `LinkCutTree<Val>` — a self-contained Sleator–Tarjan link-cut tree. No LEMON dependency. | [Link-cut trees](link-cut-tree.md) |
| `pylmcf/network_simplex_lct.h` | `NetworkSimplexLCT<Value, Cost>` — primal network simplex with the basis in an LCT instead of LEMON's thread/succ_num arrays | [LCT network simplex](lct-network-simplex.md) |
| `pylmcf/network_simplex_lct_dyn.h` | `NetworkSimplexLCTDyn` — **experimental** true dynamic-trees simplex; flow lives *in* the LCT | [LCT network simplex](lct-network-simplex.md#the-dynamic-variant) |
| `pylmcf/network_simplex_lct_adapter.h` | `NetworkSimplexLCTAdapter<GR,V,C>` — mirrors `lemon::NetworkSimplex`'s API so the LCT solver can be A/B-tested drop-in | [LCT network simplex](lct-network-simplex.md#drop-in-ab-testing-against-lemon) |
| `pylmcf/chain_solver_1d.h` | `ChainSolver1D<Value, Cost>` — specialised successive-shortest-path solver for the 1D-chain LP | [1D chain solver](chain-solver-1d.md) |

---

## The vendored LEMON is MODIFIED

`src/pylmcf/cpp/lemon/` began as a copy of [LEMON](https://lemon.cs.elte.hu/trac/lemon),
but **`lemon/network_simplex.h` carries substantial pylmcf-specific work that
upstream does not have.** Overwriting it with a stock LEMON release silently
destroys the warm-restart machinery that `wnet` depends on.

If you are updating the vendored copy, diff rather than replace. The additions
are:

- **`warmRun(PivotRule, WarmRepair, costs_changed)`** and the five `WarmRepair`
  strategies — see [Warm restarts](warm-restart.md).
- Supporting internals: `repairTreeFlows()`, `dualSimplexRepair()`,
  `primalSimplexRepair()`, `dualRatioRepair()`, `dualGreedyRepair()`,
  `syncCapsFromUpper()`, `finalizeOptimal()`, reusable scratch buffers
  (`_repair_*`), a lazily built CSR node→incident-arc index, and exposed
  `internalState()` / `sumSupplyMutable()` / `STATE_*_VAL`.
- Counters: `warmStartCount()`, `coldStartCount()`, `dualRepairCount()`,
  `primalRepairCount()`, `policyColdCount()`.
- Policy: `setWarmViolationLimit()` / `warmViolationLimit()`,
  `setWarmRepairBudget()` / `warmRepairBudget()` / `warmRepairLimitMs()` /
  `lastColdMs()`.
- Stall detection in `dualSimplexRepair` (16 non-decreasing violation counts →
  cold fallback).
- Two opt-in diagnostic compile flags — see
  [Testing and diagnostics](testing.md#diagnostic-build-flags).

`lemon/bits/windows.cc` is the one non-header LEMON file CMake compiles.

LEMON is, like pylmcf, under the Boost Software Licence.

---

## Scope of the header-only solvers

`NetworkSimplexLCT`, `NetworkSimplexLCTDyn` and their adapter all target one
regime — the one the `wnet` workload produces:

- **EQ supply**: supplies sum to zero.
- **Zero lower bounds.** There is no `lowerMap` equivalent.
- **Finite capacities on real arcs.** Big-M artificial arcs supply the initial
  feasible star basis.

If you need lower bounds, GEQ/LEQ supply, or the other three solver families
(cycle canceling, cost scaling, capacity scaling), use `lemon::NetworkSimplex`
or the [functional Python API](../README.md#alternative-solvers) instead.

---

## Anti-cycling, in both LCT solvers

LEMON's exact leaving rule — strict `<` on the first cycle path, `<=` on the
second — is what keeps the spanning tree *strongly feasible* (Cunningham) given
the strongly-feasible artificial-star start. Both LCT solvers reproduce it
exactly.

A smallest-arc-id (Bland) tie-break does **not** preserve that invariant and was
observed to cycle on degenerate (delta == 0) pivots. Do not substitute it.

---

## See also

- [Warm restarts](warm-restart.md)
- [Testing and diagnostics](testing.md) — the `tests_cpp/` oracle suites, which
  are the real correctness net for everything here and are **not** run by CI
- [Build modes and threading](build-modes.md)
