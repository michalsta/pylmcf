# Link-cut trees

`pylmcf/link_cut_tree.h` is a self-contained Sleator–Tarjan link-cut tree in the
splay-tree representation. It has **no LEMON dependency** and is a single class
template — useful on its own, not merely as network-simplex plumbing.

```cpp
#include <pylmcf/link_cut_tree.h>

pylmcf::LinkCutTree<long long> t(n);   // n nodes, ids 0..n-1, no edges
```

`Val` must be a signed arithmetic type; `long long` and `double` both work.

Every operation below is O(log n) amortized.

---

## What it maintains

A forest of unrooted trees, each with a current root, plus a scalar value per
node. On top of that:

- **path sum** — the sum of node values along a path
- **path min with argmin** — the minimum value along a path *and* a node id
  achieving it
- **lazy path add** — add a constant to every node value along a path, in
  O(log n), without touching each node

Aggregates are order-independent, so the reversal lazy used by `makeRoot` only
has to swap children; sum, min and argmin are unaffected by path direction.

---

## Two families of path operations

This is the part worth reading twice.

| Family | Signature shape | Re-roots? |
|---|---|---|
| `*Path` | `pathSum(u, v)`, `pathMin(u, v)`, `pathAdd(u, v, d)`, `pathLen(u, v)` | **Yes** — calls `makeRoot(u)` internally |
| `*ToRoot` | `sumToRoot(u)`, `minToRoot(u)`, `addToRoot(u, d)`, `cutParent(u)` | **No** |

The `*Path` family operates on the unique `u..v` path but moves the tree root to
`u` as a side effect. The `*ToRoot` family operates on the `root..u` path and
leaves the root where it is; it is cheaper and orientation-preserving.

**A fixed-root network simplex needs the second family exclusively**: the
artificial root must never move, and edge orientations relative to it must stay
stable. If you are building anything with a distinguished root, reach for
`*ToRoot` and `cutParent` and treat `makeRoot`/`*Path` as a hazard.

Likewise `lca(u, v)` is relative to the **current** tree root and does not
re-root — so it is only meaningful if the root has not been moved since you
established the one you meant.

---

## API

### Construction and node values

```cpp
explicit LinkCutTree(int n);      // n nodes, ids 0..n-1, no edges
void setVal(int x, Val v);        // the value participating in sum/min
Val  getVal(int x);
```

### Structural

```cpp
bool connected(int x, int y);     // same tree?
int  findRoot(int x);             // root of x's tree, current orientation
void makeRoot(int x);             // evert: make x the root
int  parentOf(int x);             // parent relative to current root, or -1
void link(int x, int y);          // add edge x—y; x and y must be in different trees
void cut(int x, int y);           // remove edge x—y; it must exist
void cutParent(int x);            // detach x from its parent WITHOUT re-rooting
int  lca(int u, int v);           // LCA relative to the current root
```

`cutParent(x)` requires `x` not to be the current root. The component keeping the
old root stays rooted there; the detached component becomes a new tree rooted at
`x`. This is the fixed-root-preserving cut a network simplex wants when the
leaving tree arc is removed.

### Path queries and updates

```cpp
Val                  pathSum(int u, int v);
std::pair<Val,int>   pathMin(int u, int v);    // {min value, a node achieving it}
void                 pathAdd(int u, int v, Val d);
int                  pathLen(int u, int v);    // number of EDGES on the path

Val                  sumToRoot(int u);
std::pair<Val,int>   minToRoot(int u);
void                 addToRoot(int u, Val d);
```

---

## A worked example

```cpp
#include <pylmcf/link_cut_tree.h>
#include <cstdio>

int main() {
    // Six nodes, no edges yet.
    pylmcf::LinkCutTree<long long> t(6);

    // Node values participate in path-sum and path-min.
    for (int v = 0; v < 6; ++v) t.setVal(v, 10 * v);

    // Build   0 - 1 - 2 - 3   and   1 - 4 - 5
    t.link(1, 0);
    t.link(2, 1);
    t.link(3, 2);
    t.link(4, 1);
    t.link(5, 4);

    // Fix node 0 as the root, then never move it again.
    t.makeRoot(0);

    printf("root of 5            = %d\n", t.findRoot(5));
    printf("parentOf(5)          = %d\n", t.parentOf(5));
    printf("connected(3, 5)      = %d\n", (int)t.connected(3, 5));
    printf("lca(3, 5)            = %d\n", t.lca(3, 5));

    // root..u queries: no re-rooting, so node 0 stays the root.
    printf("sumToRoot(3)         = %lld\n", t.sumToRoot(3));   // 0+10+20+30
    auto m = t.minToRoot(3);
    printf("minToRoot(3)         = %lld at node %d\n", m.first, m.second);

    // Lazy add along root..3, then re-read.
    t.addToRoot(3, 5);
    printf("after addToRoot(3,5) = %lld\n", t.sumToRoot(3));   // +5 on 4 nodes

    // u..v queries re-root the tree -- note findRoot afterwards.
    printf("pathSum(3, 5)        = %lld\n", t.pathSum(3, 5));
    printf("pathLen(3, 5)        = %d edges\n", t.pathLen(3, 5));
    printf("root is now          = %d\n", t.findRoot(0));

    // cutParent detaches a node from its parent without moving the root.
    t.makeRoot(0);
    t.cutParent(4);                       // splits off {4, 5}
    printf("after cutParent(4): connected(0,5) = %d, findRoot(5) = %d\n",
           (int)t.connected(0, 5), t.findRoot(5));
    return 0;
}
```

```
$ g++ -I$(python -m pylmcf --include) -std=c++20 -O2 lct.cpp -o lct && ./lct
root of 5            = 0
parentOf(5)          = 4
connected(3, 5)      = 1
lca(3, 5)            = 1
sumToRoot(3)         = 60
minToRoot(3)         = 0 at node 0
after addToRoot(3,5) = 80
pathSum(3, 5)        = 165
pathLen(3, 5)        = 4 edges
root is now          = 3
after cutParent(4): connected(0,5) = 0, findRoot(5) = 4
```

Two things in that output are worth pointing at:

- `sumToRoot(3)` is 60 — it includes the root's own value and node 3's, i.e. all
  four nodes on the `0..3` path, not just the interior. `addToRoot(3, 5)` then
  adds 5 to each of those four, giving 80.
- After `pathSum(3, 5)` the root has moved to **3**. That is `pathSum` calling
  `makeRoot(3)`, exactly as documented, and it is why the example re-establishes
  `makeRoot(0)` before `cutParent(4)`.

---

## Correctness

`tests_cpp/test_link_cut_tree.cpp` brute-forces every operation against an O(n)
adjacency-list reference over many randomized link/cut/update/query sequences. It
is not in CI — see [Testing and diagnostics](testing.md).

```bash
g++ -I$(python -m pylmcf --include) -std=c++20 -O2 \
    tests_cpp/test_link_cut_tree.cpp -o /tmp/t && /tmp/t
```

---

## See also

- [LCT network simplex](lct-network-simplex.md) — what this was built for
- [The C++ header tree](cpp-headers.md)
