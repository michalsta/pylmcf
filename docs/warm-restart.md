# Warm restarts

pylmcf's vendored LEMON carries a warm-restart extension that upstream LEMON does
not have: after an optimal solve, the spanning-tree basis is retained, and a
subsequent solve with changed capacities, supplies or costs starts from that
basis instead of rebuilding one from scratch.

This matters for the workload pylmcf exists to serve. `wnet` and `wnetalign`
solve the same graph hundreds of times with one thing moved, and a cold
`init()` on every re-solve dominates the run.

> **Where the code is:** `src/pylmcf/cpp/lemon/network_simplex.h`. It is a
> *modified* LEMON — see [Vendored LEMON](cpp-headers.md#the-vendored-lemon-is-modified).

---

## From Python

Nothing to switch on. `Graph.solve()` warm-restarts automatically on every
re-solve:

```python
import numpy as np
import pylmcf

G = pylmcf.Graph(3,
    edge_starts=np.array([0, 0, 1]),
    edge_ends=np.array([1, 2, 2]))
G.set_edge_costs(np.array([1, 3, 5]))
G.set_edge_capacities(np.array([3, 3, 5]))

for d in (5, 4, 6, 2):
    G.set_node_supply(np.array([d, 0, -d]))
    G.solve()
    print(d, G.result(), G.total_cost())

print("warm   =", G.warm_start_count())     # 2
print("cold   =", G.cold_start_count())     # 0
print("dual   =", G.dual_repair_count())    # 1
print("primal =", G.primal_repair_count())  # 0
print("policy =", G.policy_cold_count())    # 0
```

```
5 [2 3 2] 21
4 [1 3 1] 15
6 [3 3 3] 27
2 [0 2 0] 6
```

The first `solve()` is always a plain cold `run()` and is **not** counted. Over
a chain of successful re-solves,

```
warm + cold + dual_repair + primal_repair == number of re-solves
```

so the counters are how you tell whether warm restarts are actually firing.
`tests/test_warm_resolve.py` contains exactly such a guard — a regression that
quietly routes every re-solve through a cold `init()` would still produce
correct answers, and only the counters catch it.

### What changes between solves, and what it costs

| You call | Effect on the next `solve()` |
|---|---|
| `set_node_supply` | warm; a supply change is what warm restart is best at |
| `set_edge_capacities` | warm |
| `set_edge_costs` | warm, but potentials are recomputed first (see below) |
| `set_edge_minimums` with any nonzero entry | **always cold** |
| a previous solve returned non-OPTIMAL | next solve is cold |

Nonzero lower bounds force a cold solve because LEMON's `init()` folds `_lower`
into `_supply` and solves in a transformed space, and `finalizeOptimal()`
transforms the flows back. The warm path applies neither transformation, so
reusing a basis there would mix original-space retained flows with
transformed-space capacities — silently wrong results, not a slow path. The cold
fallback is taken inside `warmRun()` itself, so you do not have to think about
it.

### The policy knob

```python
G.set_warm_violation_limit(0)    # never attempt the simplex repair
G.set_warm_violation_limit(-1)   # always attempt it (the default behaviour)
```

When the cheap basis patch (`repairTreeFlows()`) fails, the solver can either
run a simplex repair or give up and go cold. `set_warm_violation_limit(v)` with
`v >= 0` skips the repair whenever the patch failed with more than `v` violated
basic arcs; those skips are counted in `policy_cold_count()`, separately from
`cold_start_count()`.

**In practice only `0` and `-1` are useful settings.** Intermediate thresholds
can lose to *both* extremes, because the choice interacts with the basis
trajectory: a forced cold start yields a basis from which subsequent repairs are
systematically more expensive, while a successful repair keeps later solves on
the cheap fast path. Do not treat this as a dial to tune.

---

## From C++

```cpp
#include <lemon/list_graph.h>
#include <lemon/network_simplex.h>

#include <cstdio>

int main() {
    typedef lemon::ListDigraph GR;
    typedef lemon::NetworkSimplex<GR, long long, long long> NS;

    GR g;
    GR::Node n0 = g.addNode(), n1 = g.addNode(), n2 = g.addNode();
    GR::Arc a01 = g.addArc(n0, n1), a02 = g.addArc(n0, n2), a12 = g.addArc(n1, n2);

    GR::ArcMap<long long> cap(g), cost(g);
    cap[a01] = 3; cap[a02] = 3; cap[a12] = 5;
    cost[a01] = 1; cost[a02] = 3; cost[a12] = 5;
    GR::NodeMap<long long> sup(g, 0);
    sup[n0] = 5; sup[n2] = -5;

    NS ns(g);
    ns.upperMap(cap).costMap(cost).supplyMap(sup);

    // First solve: cold, the ordinary LEMON call.
    if (ns.run() != NS::OPTIMAL) return 1;
    printf("cold cost %lld\n", ns.totalCost());

    // Supplies changed, costs did not -> warm restart, costs_changed = false.
    for (long long d : {4, 6, 2}) {
        sup[n0] = d; sup[n2] = -d;
        ns.supplyMap(sup);
        if (ns.warmRun(NS::BLOCK_SEARCH, NS::WarmRepair::Dual, false) != NS::OPTIMAL)
            return 1;
        printf("warm  supply %lld -> cost %lld\n", d, ns.totalCost());
    }

    // Costs changed -> costs_changed MUST be true, or the result is silently
    // the old flows priced at the new costs.
    cost[a02] = 10;
    ns.costMap(cost);
    if (ns.warmRun(NS::BLOCK_SEARCH, NS::WarmRepair::Dual, /*costs_changed=*/true)
        != NS::OPTIMAL) return 1;
    printf("warm  repriced   -> cost %lld\n", ns.totalCost());

    printf("warm=%d cold=%d dual_repair=%d primal_repair=%d policy_cold=%d\n",
           ns.warmStartCount(), ns.coldStartCount(), ns.dualRepairCount(),
           ns.primalRepairCount(), ns.policyColdCount());
    printf("last cold %lld work units, repair limit %lld (budget x%.1f)\n",
           ns.lastColdWork(), ns.warmRepairWorkLimit(), ns.warmRepairBudget());
    return 0;
}
```

```
$ g++ -I$(python -m pylmcf --include) -std=c++20 -O2 warm.cpp -o warm && ./warm
cold cost 21
warm  supply 4 -> cost 15
warm  supply 6 -> cost 27
warm  supply 2 -> cost 6
warm  repriced   -> cost 12
warm=3 cold=0 dual_repair=1 primal_repair=0 policy_cold=0
last cold 48 work units, repair limit 1200 (budget x25.0)
```

### `costs_changed` is load-bearing

```cpp
ProblemType warmRun(PivotRule pivot_rule  = BLOCK_SEARCH,
                    WarmRepair strategy   = WarmRepair::Dual,
                    bool costs_changed    = false);
```

A warm restart with `costs_changed = false` relies on edge **costs** having
stayed fixed: that is what keeps the retained basis dual-feasible and licenses
the "basis patch succeeded ⇒ already optimal" fast path. If you re-pushed
`costMap()` with different costs and do not say so, the stored potentials price
the *old* costs, the fast path fires anyway, and you get the previous solve's
flows evaluated against the new costs — **measurably suboptimal, and no error**.

With `costs_changed = true` the tree potentials are recomputed for the new costs
and `start()` reoptimizes from the reused basis. This is still warm: the old
basis is typically a handful of pivots from the new optimum.

`Graph::solve()` handles this for you — `set_edge_costs()` sets a `_costs_dirty`
flag that is forwarded as `costs_changed`.

### Repair strategies

`WarmRepair` selects what happens when `repairTreeFlows()` fails:

| Strategy | Behaviour |
|---|---|
| `RepairOnly` | No repair. Fall straight back to a cold `init()`. |
| `Dual` | **Default.** Dual-simplex repair: preserves dual feasibility, restores primal. |
| `Primal` | Primal-pivot repair. Does not preserve dual feasibility, so `start()` reoptimizes. |
| `DualRatio` | `Dual` with a bound-flipping (long-step) ratio test: cheap bound flips of cut-crossing arcs cut the number of basis pivots. |
| `DualGreedy` | Like `DualRatio` but the entering arc is chosen by maximum capacity, aiming to cover the violation in one pivot. Suits widely varying capacities. |

From Python, `G.set_warm_repair("repair_only" | "dual" | "primal" |
"dual_ratio" | "dual_greedy")` selects the strategy for later re-solves, and
`G.warm_repair()` reads it back.

`DualRatio` and `DualGreedy` are **not bit-identical** to `Dual` — they can land
on a different basis at a degenerate optimum. Same cost, possibly different
flow vector. They are opt-in for that reason.

`dualSimplexRepair` also carries stall detection: 16 consecutive
non-decreasing violation counts abandons the repair for the cold fallback.

### The work budget is a tripwire, not a dial

```cpp
ns.setWarmRepairBudget(25.0);   // the default
ns.setWarmRepairBudget(0.0);    // disable
```

From Python: `G.set_warm_repair_budget(mult)` / `G.warm_repair_budget()`.

A repair attempt bails to the (always correct) cold fallback once it has done
`mult` times the work of the last cold solve on that solver (`lastColdWork()`).
`<= 0` disables it, as does having no cold reference yet.

Work is counted in deterministic units: tree-walk steps in the pivot primitives,
an O(√m) estimate per entering-arc search, O(n+m) per `init()`, and the repair
loops' own scans. The budget therefore depends only on the problem and the
basis, never on timing, so a given input always takes the same path. (It used
to be wall time; on networks split into thousands of tiny subgraphs, where a
cold solve takes microseconds, an interrupt during a repair could trip it and
change which optimal basis — and so which answer — an outer optimizer saw.)

The default 25.0 is deliberately enormous. Work units skew about 2.5× between
cold pivots and repair tree walks, so 25× cold work is roughly 64× cold time.
Budget bail-outs cascade through the same trajectory effect as the violation
limit — a measured 16× time budget came out **1.5× slower overall** than no
budget at all — so the threshold has to sit far above the repair/cold ratios of
workloads where repair pays. The worst observed *profitable* repair ran ≈31×
the cold solve's time. Treat the budget as "something has gone catastrophically
wrong", not as a knob.

### Environment overrides

Both policies can be overridden per process without rebuilding callers, which is
what they exist for — A/B timing.

| Variable | Overrides |
|---|---|
| `PYLMCF_WARM_VIOLATION_LIMIT` | `setWarmViolationLimit()` |
| `PYLMCF_WARM_REPAIR_BUDGET` | `setWarmRepairBudget()` |

Each is read **once per process** (a function-local `static`), so setting it from
inside a running program after the first solve does nothing.
`tests_cpp/test_dual_repair.cpp` refuses to run under either, because its whole
job is to assert the in-code defaults.

---

## Crash starts

A cold solve normally starts from LEMON's artificial basis: one artificial arc
per node, all supply routed over them, and the simplex pivots its way out. A
*crash start* builds the starting basis from a flow you supply instead.

### From Python

`G.solve_from_flow(flow, prefer=None)` solves once from `flow` (one int64 per
edge). `G.set_cold_start_flow(flow, prefer=None)` installs a flow that every
later cold solve of `solve()` starts from: the first solve, and any re-solve
whose warm restart falls back to cold. `G.clear_cold_start_flow()` removes it,
and `G.has_cold_start_flow()` says whether one is installed.

```python
import numpy as np
import pylmcf

H = pylmcf.Graph(3,
    edge_starts=np.array([0, 0, 1]),
    edge_ends=np.array([1, 2, 2]))
H.set_edge_costs(np.array([1, 3, 5]))
H.set_edge_capacities(np.array([3, 3, 5]))
H.set_warm_repair("repair_only")         # every failed patch goes cold
for d in (3, 2, 6):
    H.set_node_supply(np.array([d, 0, -d]))
    H.set_cold_start_flow(np.array([0, min(d, 3), max(d - 3, 0)]), prefer=np.array([0]))
    H.solve()
    print(d, H.result(), H.total_cost())
print("crash =", H.crash_start_count(), " cold =", H.cold_start_count(), " warm =", H.warm_start_count())
```

```
3 [0 3 0] 9
2 [0 2 0] 6
6 [3 3 3] 27
crash = 3  cold = 2  warm = 0
```

The flow is copied and not refreshed, which is why the loop installs a new one
for each supply. A stale or infeasible flow still gives the exact optimum,
only more slowly.

### What the crash does

1. Arcs with `0 < flow < cap` become tree arcs, in arc order, unless one would
   close a cycle; then it stays nonbasic at zero. Arcs with `flow >= cap` start
   at their upper bound.
2. The `prefer` arcs, in the order given, join the components that are still
   separate, as zero-flow tree arcs.
3. Each component left after that is attached to the root by one artificial
   arc.
4. Tree flows are recomputed from the supplies (exactly the given flow when
   it was already basic) and potentials from the tree. If the recomputed flows
   break a bound, the primal repair pass runs. Then the ordinary simplex
   finishes.

The flow therefore does not have to be feasible or basic. It only has to be a
good place to begin.

**Choose `prefer` with care.** The preferred arcs fix the starting potentials,
and on transport-like problems that matters more than the flow itself. In
`wnet`, the start flow sends all mass to the trash arcs and the preferred arcs
are the `Source->Emp` and `Theo->Sink` anchors. That prices every matching arc
at "distance minus the trash cost it saves" from the first pivot. Without the
anchors the same flow gained about ±10%. With them, wnet's synthetic
benchmarks ran in 0.08–1.0× the time of the artificial start, with the biggest
gains on the largest components.

### Counters

`crash_start_count()` counts the solves that started from a crash basis, the
first solve included. A crash start inside a warm restart's cold fallback is
also counted in `cold_start_count()`. If the crash basis needed the primal
repair pass, that pass is counted in `primal_repair_count()` as well. So with
a cold-start flow installed, the identity
`warm + cold + dual_repair + primal_repair == re-solves` can overcount.

### Scope

Only EQ supplies without lower bounds use the flow. GEQ/LEQ problems and
nonzero edge minimums run a plain cold solve, counted in `cold_start_count()`
and not in `crash_start_count()`. A flow of the wrong length raises
`ValueError` in Python, and so do `prefer` indices out of range; the C++ call
skips such indices.

### From C++

```cpp
ProblemType crashRun(const std::vector<Value>& flow,        // indexed by graph.id(arc)
                     PivotRule pivot_rule = BLOCK_SEARCH,
                     const std::vector<int>* prefer = nullptr);  // LEMON arc ids
void setColdStartFlow(const std::vector<Value>* flow,       // nullptr: plain cold init()
                      const std::vector<int>* prefer = nullptr);
int crashStartCount() const;
```

`crashRun()` replaces `run()` for a cold solve. `setColdStartFlow()` makes
`warmRun()`'s cold fallback call `crashRun()`. It stores the pointers, not
copies: the vectors must outlive the solves, and the caller refreshes them when
supplies or capacities change. After a crash start, the dual and primal
repairs of later warm restarts never choose an artificial arc to enter the
basis. On a crash basis that could put flow on an artificial arc and end in a
false `INFEASIBLE`. The restriction lasts until the next `init()`.

---

## Scope

The warm path requires:

- **EQ supply** — supplies summing to zero (`_sum_supply == 0`). GEQ/LEQ
  problems go cold.
- **Zero lower bounds** — any nonzero `lowerMap` forces cold.
- Fixed topology. Arcs may not be added or removed between warm solves.

Everything outside that falls back to a cold solve inside `warmRun()`. The
fallback is always correct; it is only slower.

---

## See also

- [The C++ header tree](cpp-headers.md)
- [Testing and diagnostics](testing.md) — `tests_cpp/test_dual_repair.cpp` is the
  oracle suite for everything on this page
