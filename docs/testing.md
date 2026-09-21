# Testing and diagnostics

pylmcf has two test suites with very different status, and it is worth knowing
which is which before trusting a green run.

---

## `tests/` — Python, pytest, run by CI

```bash
cd tests && python -m pytest .
```

or a single test:

```bash
python -m pytest tests/test_graph.py::test_graph_simple
```

| File | Covers |
|---|---|
| `test_graph.py` | the OO API |
| `test_graph_lb.py` | lower bounds |
| `test_networkx.py` | NetworkX interop |
| `test_solver_variants.py` | the four functional solvers |
| `test_api.py` | `as_nx`, `FromNX` edge cases, `include()` |
| `test_free_threading.py` | 8 threads × 25 concurrent solves vs a serial oracle |
| `test_warm_resolve.py` | warm re-solve chains vs a fresh-cold oracle |

Two notes:

- **`test_networkx.py` imports `networkx` at module scope.** Omitting it from a CI
  job is a collection *error*, not a skip. The `pytest` extra
  (`pip install pylmcf[pytest]`) pulls it in.
- **`test_free_threading.py` self-skips** unless the GIL is still off after
  importing the extension. On a normal interpreter that is correct behaviour; on
  a free-threaded one, a skip is a failure, which is what
  `test_wheel_freethreaded` in CI checks.

`test_warm_resolve.py` includes a **counter guard** that fails if warm restarts
silently stop firing. This matters more than it sounds: a regression that routes
every re-solve through a cold `init()` still produces correct answers, so nothing
else in the suite would notice.

### Lint

```bash
ruff check src/ tests/
```

### Release version check

```bash
python .github/scripts/check_version.py
```

The git tag must match `pyproject.toml`. This runs only on tag pushes — on an
untagged branch `git describe --tags --abbrev=0` returns the most recent
*reachable* tag and so fails on any branch whose version has been bumped.

---

## `tests_cpp/` — C++ oracle suites, hand-compiled, NOT in CI

These are the real correctness net for the solver work. **They are not run by
CMake, not collected by pytest, and not run by CI.** Nothing but running them by
hand will catch a regression in the headers they cover.

Each is a standalone `main()` that exits non-zero on failure, with its build line
in the header comment. They all validate against an **independent oracle** —
usually real `lemon::NetworkSimplex` — rather than golden values.

```bash
g++ -I$(python -m pylmcf --include) -std=c++20 -O2 \
    tests_cpp/test_network_simplex_lct.cpp -o /tmp/t && /tmp/t
```

| Suite | Oracle |
|---|---|
| `test_link_cut_tree.cpp` | a brute-force O(n) adjacency-list reference, over randomized op sequences |
| `test_dual_repair.cpp` | LEMON `warmRun`/`dualSimplexRepair` warm chains vs a fresh cold solve after each mutation |
| `test_network_simplex_lct.cpp` | LEMON's array solver (cold) |
| `test_network_simplex_lct_warm.cpp` | LEMON, an independent cold solve per warm step |
| `test_network_simplex_lct_dyn.cpp` | LEMON (the dynamic variant, cold) |
| `test_network_simplex_lct_dyn_warm.cpp` | LEMON (the dynamic variant, warm) |
| `test_lct_adapter.cpp` | real `lemon::NetworkSimplex` on `wnet`'s exact call pattern |
| `test_chain_solver_1d.cpp` | LEMON on the chain LP |

Run the lot:

```bash
INC=$(python -m pylmcf --include)
for f in tests_cpp/*.cpp; do
    t=$(basename "$f" .cpp)
    g++ -I"$INC" -std=c++20 -O2 "$f" -o "/tmp/$t" && "/tmp/$t" \
        && echo "ok   $t" || echo "FAIL $t"
done
```

### `test_dual_repair.cpp` specifically

The most thorough of them. It checks, after every mutation in a warm chain:
status, exact cost, primal feasibility, **and** the dual optimality certificate
via `potential()`. It covers cap/supply mutations and `costs_changed=true` cost
repricing, the `setWarmViolationLimit` policy (0 must suppress repair and be
recorded in `policyColdCount()`; -1 must restore it), and the lower-bounds cold
guard.

It also **fails if the dual-repair or `costs_changed` paths are never
exercised**, so a regression that quietly sends everything through cold `init()`
cannot pass it.

It refuses to run under `PYLMCF_WARM_VIOLATION_LIMIT` or
`PYLMCF_WARM_REPAIR_BUDGET`, because its job is to assert the in-code defaults.

**When you touch any solver header, run the corresponding suite.**

---

## Diagnostic build flags

Two opt-in compile-time flags. Both are **off by default and not set by CMake or
CI** — pass `-D` by hand when experimenting.

### `PYLMCF_PIVOT_STATS`

Declares thread-local counters for A/B-ing pivot rules. Not even declared without
the flag.

```cpp
namespace pylmcf_stats {
  inline thread_local unsigned long long pivot_calls;
  inline thread_local unsigned long long pivot_arcs;
  // diagnostics of the last warmRun() whose repairTreeFlows() failed:
  inline thread_local unsigned long long warm_violations;
  inline thread_local unsigned long long warm_violation_mass;
}
```

```bash
g++ -I$(python -m pylmcf --include) -std=c++20 -O2 -DPYLMCF_PIVOT_STATS ...
```

### `PYLMCF_BLOCK_LOOP`

`=1` or `=2` select alternate `BlockSearchPivotRule::findEnteringArc()`
implementations that hoist the block boundary out of the inner loop. **Same rule,
same arcs, same tie-break** as stock; variant 2 keeps the wrap off the hot path.

Motivated by a profile putting ~20% of solve time in the stock counter/loop
overhead.

### Debug tracing

`PYLMCF_NSLCT_DEBUG` and `PYLMCF_DYN_DEBUG` enable conservation-check tracing in
`network_simplex_lct.h` and `network_simplex_lct_dyn.h` respectively.

### Runtime overrides

Neither of these needs a rebuild; both are read **once per process**. See
[Warm restarts](warm-restart.md#environment-overrides).

| Variable | Overrides |
|---|---|
| `PYLMCF_WARM_VIOLATION_LIMIT` | `setWarmViolationLimit()` |
| `PYLMCF_WARM_REPAIR_BUDGET` | `setWarmRepairBudget()` |

---

## CI

`.github/workflows/run_tests.yml` has no matrix literal. Which combinations run is
decided by `.github/scripts/ci_matrix.py`, which emits an `{"include": [...]}`
object handed to `strategy: matrix`. It is a **covering array, not a
cross-product**: every level of every factor runs, but pairs are only covered
where the pair actually interacts.

Three nested tiers:

| Tier | When | Size |
|---|---|---|
| A | push to a work branch | 4 lanes, all self-hosted `linux-amd64`, ~20 min wall |
| B | `main`, the nightly cron, leaf-package tags | 14 lanes: all 6 platforms, all 6 Pythons, all 4 toolchains, both arches |
| C | `v*` tags on pylmcf and wnet | 35 lanes, the wide matrix |

`workflow_dispatch` carries a `tier` input, so any tier can be run by hand without
tagging anything. `ci_matrix.py` ends in a coverage audit that fails the `select`
job if a deleted lane breaks 1-coverage of any platform, Python, toolchain or
architecture, or if the A ⊆ B ⊆ C nesting stops holding.

**`linux-arm64` is qemu**, on a self-hosted x86 runner, and measures 65–85 minutes
against 8–16 for every other platform. Do not add arm64 lanes without a reason
that names a specific architecture-dependent failure.

---

## Dead scripts — not live examples

`tests/measure_performance.py`, `workshop/workshop.py` and `experiments/*.py`
import `Distribution`, `WassersteinSolver`, `DeconvolutionSolver` and
`DecompositableFlowGraph` — API that no longer lives in pylmcf; it moved to
`wnet`. **They cannot run.** They are excluded from the sdist, and
`measure_performance.py` is not collected by pytest. Do not copy from them.

---

## See also

- [Warm restarts](warm-restart.md)
- [The C++ header tree](cpp-headers.md)
- [Build modes and threading](build-modes.md)
