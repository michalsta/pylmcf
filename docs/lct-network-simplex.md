# LCT network simplex

Two experimental primal network simplex implementations that keep the spanning-tree
basis in a [link-cut tree](link-cut-tree.md) instead of LEMON's
`thread`/`succ_num` arrays, plus an adapter that lets either be dropped in where
`lemon::NetworkSimplex` is expected.

Both have [Python bindings](python-solvers.md): `NetworkSimplexLCT`,
`NetworkSimplexLCTDyn`, `lmcf_lct`, and `lmcf_lct_dyn`. The Python `Graph` class
continues to use LEMON's array solver. The examples and counters below describe
the direct C++ API; Python counters include the initial cold solve.

---

## Why

In LEMON's array implementation, three steps per pivot are O(subtree) or
O(cycle):

| Step | Array cost | LCT cost |
|---|---|---|
| node potentials (`updatePotential`) | O(subtree) shift | `sumToRoot` — O(log n), implicit |
| join node (lowest common ancestor) | walk | `lca` — O(log n) |
| structural pivot (`updateTreeStructure`) | O(subtree) thread re-splice | `cutParent` + `link` — O(log n) |

On a long chain — which is exactly the shape `wnet`'s 1D workload produces — those
subtrees are the whole graph.

---

## `NetworkSimplexLCT`

`pylmcf/network_simplex_lct.h`. The conservative variant: the three steps above
move to the LCT, and the unavoidable O(cycle) work (ratio test, flow change, stem
reversal of the `par`/`pred` arrays) stays in plain arrays, exactly as the array
solver does it.

```cpp
template <typename Value = long long, typename Cost = long long>
class NetworkSimplexLCT {
  enum Status { OPTIMAL, INFEASIBLE };

  explicit NetworkSimplexLCT(int n);          // nodes 0..n-1

  int  addArc(int u, int v, Cost cost, Value cap);   // returns arc id; lower bound 0
  void setSupply(int node, Value s);
  void setCap(int arc_id, Value cap);         // allowed between warm restarts

  Status run();                               // cold solve
  Status warmRun();                           // repair-or-cold warm restart

  Cost  totalCost() const;
  Value flow(int arc_id) const;
  Cost  potential(int node) const;
  int   warmCount() const;
  int   coldCount() const;
};
```

`addArc` must be called before the first `run()`/`warmRun()` — **topology and
costs are fixed for the lifetime of the solver.** Only supplies (`setSupply`) and
capacities (`setCap`) may change between warm restarts.

`potential()` is only meaningful in **differences**. `pi[u]` satisfies the same
reduced-cost relation as LEMON's `_pi` (`rc = cost + pi[src] - pi[tgt]`, zero on
tree arcs), so `pi[u] - pi[v]` matches LEMON exactly; the two vectors differ by a
global offset, which the artificial root makes harmless. It is cached post-solve,
so reading it is O(1).

### Example

```cpp
#include <pylmcf/network_simplex_lct.h>
#include <cstdio>

int main() {
    using Solver = pylmcf::NetworkSimplexLCT<long long, long long>;

    Solver s(3);                          // nodes 0, 1, 2
    const int e01 = s.addArc(0, 1, /*cost=*/1, /*cap=*/3);
    const int e02 = s.addArc(0, 2, 3, 3);
    const int e12 = s.addArc(1, 2, 5, 5);

    s.setSupply(0,  5);
    s.setSupply(2, -5);                   // supplies must sum to zero

    if (s.run() != Solver::OPTIMAL) { printf("infeasible\n"); return 1; }
    printf("cold : flows %lld %lld %lld  cost %lld\n",
           s.flow(e01), s.flow(e02), s.flow(e12), s.totalCost());

    // Warm restart: change supplies (and/or capacities via setCap), re-solve.
    // Costs and topology must stay fixed.
    for (long long d : {4, 6, 2}) {
        s.setSupply(0,  d);
        s.setSupply(2, -d);
        if (s.warmRun() != Solver::OPTIMAL) { printf("infeasible\n"); return 1; }
        printf("warm : supply %lld -> flows %lld %lld %lld  cost %lld\n",
               d, s.flow(e01), s.flow(e02), s.flow(e12), s.totalCost());
    }
    printf("warmCount = %d, coldCount = %d\n", s.warmCount(), s.coldCount());

    // Potentials are only meaningful as differences (global offset is free).
    printf("pi[2] - pi[0] = %lld\n", s.potential(2) - s.potential(0));
    return 0;
}
```

```
$ g++ -I$(python -m pylmcf --include) -std=c++20 -O2 nslct.cpp -o nslct && ./nslct
cold : flows 2 3 2  cost 21
warm : supply 4 -> flows 1 3 1  cost 15
warm : supply 6 -> flows 3 3 3  cost 27
warm : supply 2 -> flows 0 2 0  cost 6
warmCount = 2, coldCount = 1
pi[2] - pi[0] = 3
```

Those costs are bit-identical to what `lemon::NetworkSimplex` and the Python
`Graph` produce on the same instance. Note `coldCount = 1` across three
`warmRun()` calls: one of them fell back to a cold solve, which is normal and
always correct.

### Warm restart strategy

`warmRun()` implements the *Simple* repair-or-cold strategy, mirroring LEMON's
`WarmMode.Simple`:

1. Recompute tree-arc flows for the new capacities and supplies.
2. If they all stay in bounds, the retained basis is still primal feasible —
   and since costs are unchanged it is still dual feasible — hence optimal. The
   pivot loop is run anyway as cheap insurance and does ~0 work.
3. Otherwise, cold solve.

There are no dual/primal repair strategies here, unlike
[LEMON's `warmRun`](warm-restart.md#repair-strategies).

### Counting convention

`run()` does **not** increment either counter; only `warmRun()` does. This is the
opposite of `NetworkSimplexLCTDyn` — see below.

---

## The dynamic variant

`pylmcf/network_simplex_lct_dyn.h` — `NetworkSimplexLCTDyn`. **Experimental.**

This is the "real" dynamic-trees simplex: it pushes flow *into* the LCT, so the
two steps `NetworkSimplexLCT` left in arrays also become logarithmic.

| Step | `NetworkSimplexLCT` | `NetworkSimplexLCTDyn` |
|---|---|---|
| `findLeavingArc` (ratio test) | O(cycle) `_par` walk | two O(log K) path-min queries |
| `changeFlow` | O(cycle) | two O(log K) lazy path range-adds |

On a chain that is the difference between O(K) and O(log K) per pivot, and it is
the only lever that can flip the long-chain verdict.

### How, briefly

The trick is the **rootward-flow (r-frame)** encoding. Per non-root node `u`,

```
r[u] = predDir[u] * flow[predArc[u]]        (positive means flow u -> parent)
```

Augmenting the cycle for entering arc `e = (i, j)` with `join = lca(i, j)` by
`val` becomes

```
r -= val   on path i -> join  (excluding join)
r += val   on path j -> join  (excluding join)
```

The per-edge `predDir` cancels (`predDir² = 1`), so both are **uniform** path
adds. The ratio-test residual is affine in `r` with fixed ±1 slope plus a
per-node constant, so a range add on `r` is a range add on the residuals and the
ratio test is a path-min. No eversion hazard arises: the main tree is never
everted (the artificial root is fixed and `predDir` is an explicit local sign),
so `r` and the residuals are local edge properties and only stem nodes recompute
on a pivot.

Anti-cycling uses the same LEMON strongly-feasible leaving rule, reproduced via
per-side tie-broken path-min argmin.

### API

Identical in shape to `NetworkSimplexLCT`: `addArc`, `setSupply`, `setCap`,
`run`, `warmRun`, `totalCost`, `flow`, `warmCount`, `coldCount`. There is **no
`potential()`**.

### Warm restart

A supply change `Δs[v]` re-routes that imbalance along the tree path `v → R`,
which in the r-frame is **one O(log K) lazy path add per changed node** — not an
O(N) flow recompute. Feasibility is then checked for free on just the perturbed
segments: the prior solve left every tree arc feasible, so only nodes on a
perturbation path can violate, and the segment aggregate the range-add already
maintains is exactly that check.

**Any capacity change falls back to a cold solve** in this first cut.

### Example

```cpp
#include <pylmcf/network_simplex_lct_dyn.h>
#include <cstdio>

int main() {
    using Solver = pylmcf::NetworkSimplexLCTDyn<long long, long long>;

    Solver s(3);
    const int e01 = s.addArc(0, 1, 1, 3);
    const int e02 = s.addArc(0, 2, 3, 3);
    const int e12 = s.addArc(1, 2, 5, 5);
    s.setSupply(0, 5);
    s.setSupply(2, -5);

    if (s.run() != Solver::OPTIMAL) return 1;
    printf("cold: %lld %lld %lld  cost %lld\n",
           s.flow(e01), s.flow(e02), s.flow(e12), s.totalCost());

    // Supply-only change: applied as one O(log K) lazy path add per changed
    // node.  Any capacity change falls back to a cold solve.
    s.setSupply(0, 4);
    s.setSupply(2, -4);
    if (s.warmRun() != Solver::OPTIMAL) return 1;
    printf("warm: %lld %lld %lld  cost %lld (warm=%d cold=%d)\n",
           s.flow(e01), s.flow(e02), s.flow(e12), s.totalCost(),
           s.warmCount(), s.coldCount());
    return 0;
}
```

```
$ g++ -I$(python -m pylmcf --include) -std=c++20 -O2 dyn.cpp -o dyn && ./dyn
cold: 2 3 2  cost 21
warm: 1 3 1  cost 15 (warm=1 cold=1)
```

> **Counter gotcha:** `NetworkSimplexLCTDyn::run()` **does** increment
> `coldCount()`, while `NetworkSimplexLCT::run()` does not. Hence `cold=1` above
> after a single cold `run()` and a successful warm restart. Do not compare the
> two solvers' counters without accounting for this.

---

## Drop-in A/B testing against LEMON

`pylmcf/network_simplex_lct_adapter.h` — `NetworkSimplexLCTAdapter<GR, V, C>`
mirrors exactly the slice of `lemon::NetworkSimplex`'s API that `wnet`'s
`decompositable_graph.hpp` uses, backed by `NetworkSimplexLCT`. That lets the LCT
solver be compared against real LEMON on the exact production call pattern
without touching production `wnet`.

Mirrored surface:

```cpp
explicit NetworkSimplexLCTAdapter(const GR& g);

NetworkSimplexLCTAdapter& upperMap(const M&);     // chainable
NetworkSimplexLCTAdapter& costMap(const M&);
NetworkSimplexLCTAdapter& supplyMap(const M&);

ProblemType run(PivotRule = BLOCK_SEARCH);
ProblemType warmRun(PivotRule = BLOCK_SEARCH, WarmRepair = WarmRepair::Dual);

V totalCost() const;
V flow(const Arc&) const;
C potential(const Node&) const;

int warmStartCount() const;
int coldStartCount() const;
int dualRepairCount() const;      // always 0: Simple strategy has no repairs
int primalRepairCount() const;    // always 0

enum ProblemType { INFEASIBLE, OPTIMAL, UNBOUNDED };
enum PivotRule { FIRST_ELIGIBLE, BEST_ELIGIBLE, BLOCK_SEARCH,
                 CANDIDATE_LIST, ALTERING_LIST };
enum class WarmRepair { RepairOnly, Dual, Primal, DualRatio, DualGreedy };
```

Caveats:

- **`PivotRule` and `WarmRepair` are accepted and ignored.** The LCT solver uses
  its fixed strongly-feasible pivot rule and the Simple warm strategy.
- **No `lowerMap`.** Lower bounds are 0. (`wnet` never calls it.)
- `dualRepairCount()` and `primalRepairCount()` return 0 unconditionally.
- The adapter re-pushes every map on each call, so it is not a performance
  substitute for calling `NetworkSimplexLCT` directly — it is for A/B comparison.

### Example: one body of code, two solvers

```cpp
#include <lemon/list_graph.h>
#include <lemon/network_simplex.h>
#include <pylmcf/network_simplex_lct_adapter.h>

#include <cstdio>

typedef lemon::ListDigraph GR;

// One body of calling code, two solvers behind the same API.
template <typename Solver>
long long drive(const GR& g, const GR::ArcMap<long long>& cap,
                const GR::ArcMap<long long>& cost, GR::Node src, GR::Node dst,
                const char* label) {
    GR::NodeMap<long long> sup(g, 0);
    sup[src] = 5;
    sup[dst] = -5;

    Solver s(g);
    s.upperMap(cap).costMap(cost).supplyMap(sup);
    if (s.run() != Solver::OPTIMAL) { printf("%s infeasible\n", label); return -1; }
    printf("%s cold cost %lld\n", label, (long long)s.totalCost());

    // Re-solve warm after a supply change (costs and topology unchanged).
    sup[src] = 4;
    sup[dst] = -4;
    s.supplyMap(sup);
    if (s.warmRun() != Solver::OPTIMAL) { printf("%s infeasible\n", label); return -1; }
    printf("%s warm cost %lld (warm=%d cold=%d)\n", label, (long long)s.totalCost(),
           s.warmStartCount(), s.coldStartCount());
    return s.totalCost();
}

int main() {
    GR g;
    GR::Node n0 = g.addNode(), n1 = g.addNode(), n2 = g.addNode();
    GR::Arc a01 = g.addArc(n0, n1), a02 = g.addArc(n0, n2), a12 = g.addArc(n1, n2);

    GR::ArcMap<long long> cap(g), cost(g);
    cap[a01] = 3; cap[a02] = 3; cap[a12] = 5;
    cost[a01] = 1; cost[a02] = 3; cost[a12] = 5;

    long long a = drive<lemon::NetworkSimplex<GR, long long, long long>>(
        g, cap, cost, n0, n2, "lemon:");
    long long b = drive<pylmcf::NetworkSimplexLCTAdapter<GR, long long, long long>>(
        g, cap, cost, n0, n2, "lct:  ");
    printf("agree: %d\n", (int)(a == b));
    return a == b ? 0 : 1;
}
```

```
$ g++ -I$(python -m pylmcf --include) -std=c++20 -O2 adapter.cpp -o adapter && ./adapter
lemon: cold cost 21
lemon: warm cost 15 (warm=1 cold=0)
lct:   cold cost 21
lct:   warm cost 15 (warm=1 cold=0)
agree: 1
```

---

## Scope, restated

Both LCT solvers target:

- **EQ supply** — supplies sum to zero
- **zero lower bounds**
- **finite capacities on real arcs**

Big-M artificial arcs give the initial feasible star basis, the same structure
LEMON uses internally. Anything outside that regime wants
`lemon::NetworkSimplex`.

---

## Correctness

Four oracle suites, all validating against real LEMON rather than golden values:

```bash
INC=$(python -m pylmcf --include)
for t in test_network_simplex_lct test_network_simplex_lct_warm \
         test_network_simplex_lct_dyn test_network_simplex_lct_dyn_warm \
         test_lct_adapter; do
    g++ -I$INC -std=c++20 -O2 tests_cpp/$t.cpp -o /tmp/$t && /tmp/$t || echo "FAILED: $t"
done
```

None of them are run by CMake, pytest or CI. **When you touch any of these
headers, run them — nothing else will catch a regression.** See
[Testing and diagnostics](testing.md).

---

## See also

- [Link-cut trees](link-cut-tree.md) — the underlying data structure
- [Warm restarts](warm-restart.md) — LEMON's own, richer warm machinery
- [The C++ header tree](cpp-headers.md)
