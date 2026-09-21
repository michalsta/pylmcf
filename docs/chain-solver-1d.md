# The 1D chain solver

`pylmcf/chain_solver_1d.h` — `ChainSolver1D<Value, Cost>`, a specialised min-cost
flow solver for one particular LP: `wnet`'s 1D-chain SimpleTrash problem.

**C++-only.** There is no Python binding.

---

## The LP it solves

Positions lie on a line. Each position carries some *empirical* mass and some
*theoretical* mass, and mass may be transported along the line at cost equal to
distance moved, or discarded at a flat per-unit cost `κ`.

As a flow problem:

```
Source(+F), Sink(-F),  F = max(E, T),  E = Σ emp,  T = Σ theo

Source -> Emp_k     cost 0      cap e_k
Theo_k -> Sink      cost 0      cap t_k
Pos_k <-> Pos_{k+1} cost |Δpos| cap ∞     (both directions)
Source -> Sink      cost κ      cap ∞     (the "trash" bypass)
```

That is the shape `wnet`'s `decompositable_graph.hpp` builds for a 1D
distribution, and it is the shape a general network simplex handles badly: the
positions form a long chain, so every pivot's cycle is potentially the whole
graph.

---

## Method and complexity

Successive shortest paths, specialised to this graph: push the bottleneck along
the cheapest residual source→sink path, maintaining node potentials so reduced
costs stay non-negative, until `F` units are routed.

Every non-trash augmentation saturates a spur, so there are **O(K)
augmentations**, and Dijkstra-with-potentials gives **O(K² log K)** overall.

SSP was chosen over slope-trick deliberately. Its correctness is *mechanical* —
it does not rely on a fragile global invariant the way a primal
slope-trick/lazy-rematch formulation does — which is what makes it possible to
validate bit-exact against LEMON. That validation is the whole point; a faster
algorithm that cannot be checked is not usable here.

---

## API

```cpp
template <typename Value = long long, typename Cost = long long>
struct ChainSolver1D {
  struct Point { Cost pos; Value emp; Value theo; };

  struct Flows {
    Cost               total    = 0;
    std::vector<Value> emp_in;        // Source->Emp_k  flow, size n
    std::vector<Value> theo_out;      // Theo_k->Sink   flow, size n
    std::vector<Value> gap;           // signed rightward chain flow, size n-1
    Value              trash    = 0;  // Source->Sink   flow
  };

  static Cost  solve    (const std::vector<Point>& pts, Cost kappa);
  static Flows solveFull(const std::vector<Point>& pts, Cost kappa);
};
```

Both are `static` — there is no solver object and no warm restart. `solve()`
returns just the optimal cost; `solveFull()` also returns the per-arc flows,
which is what `wnet`'s gradient reads.

`gap[k]` is the **signed** flow between positions `k` and `k+1`, positive meaning
rightward. An empty `pts` gives cost 0 and an empty `Flows`.

Points are expected in increasing `pos` order, as `wnet` produces them.

---

## Example

```cpp
#include <pylmcf/chain_solver_1d.h>
#include <cstdio>
#include <vector>

int main() {
    using Solver = pylmcf::ChainSolver1D<long long, long long>;

    // Positions along a line, each carrying some empirical and theoretical mass.
    std::vector<Solver::Point> pts = {
        {/*pos=*/0,  /*emp=*/5, /*theo=*/0},
        {/*pos=*/10, /*emp=*/0, /*theo=*/3},
        {/*pos=*/25, /*emp=*/2, /*theo=*/4},
    };
    const long long kappa = 18;           // cost of discarding one unit

    printf("total cost = %lld\n", Solver::solve(pts, kappa));

    // solveFull() also returns the per-arc flows wnet's gradient needs.
    Solver::Flows f = Solver::solveFull(pts, kappa);
    printf("trash flow = %lld\n", f.trash);
    for (size_t k = 0; k < pts.size(); ++k)
        printf("  point %zu: emp_in %lld, theo_out %lld\n",
               k, f.emp_in[k], f.theo_out[k]);
    for (size_t k = 0; k + 1 < pts.size(); ++k)
        printf("  gap %zu->%zu: %lld (signed, rightward positive)\n",
               k, k + 1, f.gap[k]);
    return 0;
}
```

```
$ g++ -I$(python -m pylmcf --include) -std=c++20 -O2 chain.cpp -o chain && ./chain
total cost = 66
trash flow = 2
  point 0: emp_in 3, theo_out 0
  point 1: emp_in 0, theo_out 3
  point 2: emp_in 2, theo_out 2
  gap 0->1: 3 (signed, rightward positive)
  gap 1->2: 0 (signed, rightward positive)
```

Reading that off by hand: `E = T = 7`, so `F = 7`.

- 3 units move from position 0 to position 10 — distance 10, cost **30**.
- At position 25, 2 units of empirical mass meet 2 of theoretical locally, cost
  **0**.
- That leaves 2 units of supply at position 0 and 2 units of demand at position
  25. Moving them costs `2 × 25 = 50`; discarding them costs `2 × 18 = 36`. The
  trash bypass wins, cost **36**, and `f.trash == 2`.

Total **66**, and the flows say exactly which of the two options the solver took.

---

## Correctness

`tests_cpp/test_chain_solver_1d.cpp` validates against `lemon::NetworkSimplex`
on the same LP, bit-exact:

```bash
g++ -I$(python -m pylmcf --include) -std=c++20 -O2 \
    tests_cpp/test_chain_solver_1d.cpp -o /tmp/t && /tmp/t
```

Not in CI. See [Testing and diagnostics](testing.md).

---

## See also

- [The C++ header tree](cpp-headers.md)
- [LCT network simplex](lct-network-simplex.md) — the other attack on the same
  long-chain problem
