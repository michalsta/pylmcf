# pylmcf documentation

Start at the [project README](../README.md) for installation and the Python API.

## Guides

| | |
|---|---|
| [Warm restarts](warm-restart.md) | How re-solving reuses the spanning-tree basis, the five repair strategies, the counters, and the two policy knobs. Relevant from **Python and C++** — it is on by default and affects every re-solve. |
| [The C++ header tree](cpp-headers.md) | Consuming `src/pylmcf/cpp/` from your own C++, an index of what is in it, and why the vendored LEMON must not be replaced wholesale. |
| [Link-cut trees](link-cut-tree.md) | `LinkCutTree<Val>` — a standalone Sleator–Tarjan link-cut tree with path-sum, path-min-with-argmin and lazy path-add. C++ only. |
| [LCT network simplex](lct-network-simplex.md) | `NetworkSimplexLCT` and the experimental `NetworkSimplexLCTDyn`, plus the adapter that A/B-tests them against real `lemon::NetworkSimplex`. C++ only. |
| [The 1D chain solver](chain-solver-1d.md) | `ChainSolver1D` — successive shortest paths specialised to the 1D-chain LP. C++ only. |
| [Build modes and threading](build-modes.md) | nanobind split vs linked, free-threading, what is and is not thread-safe, how the wheels are built. |
| [Testing and diagnostics](testing.md) | The two test suites — one of which **CI does not run** — and the diagnostic build flags. |

## Which of these applies to me?

- **Using pylmcf from Python?** The [project README](../README.md), then
  [Warm restarts](warm-restart.md) if you re-solve the same graph more than once.
- **Compiling C++ against pylmcf?** [The C++ header tree](cpp-headers.md) first;
  it indexes the rest.
- **Hacking on the solvers?** [Testing and diagnostics](testing.md) before you
  change anything — the oracle suites that actually cover the solver headers are
  not run by CI, so a green CI run means less than it looks.
- **Debugging an import error mentioning nanobind, `TypeError` or
  `std::bad_cast`?** [Build modes and threading](build-modes.md).

Every code example in these pages is compiled and run as written; the pasted
output is real.
