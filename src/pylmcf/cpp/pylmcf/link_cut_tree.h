// link_cut_tree.h
// -------------------------------------------------------------------------
// Sleator-Tarjan link-cut tree (splay-tree representation) supporting, in
// O(log n) amortized per operation:
//
//   structural : link, cut, findRoot, connected, makeRoot (evert),
//                 parentOf, lca (relative to the current tree root)
//   path aggr. : per-node scalar value with path-sum and path-min
//                 (min carries an argmin node id), plus a lazy
//                 add-constant-along-a-path update; reversal lazy makes
//                 evert O(log n).
//
// Two families of path operations are exposed:
//
//   *Path(u, v, ...)   — operate on the unique u..v path.  These call
//                        makeRoot(u) internally, so they re-root the tree.
//   *ToRoot(u, ...)    — operate on the root..u path WITHOUT re-rooting.
//                        Cheaper and orientation-preserving; this is what a
//                        fixed-root network-simplex backend wants (the
//                        artificial root must never move and edge
//                        orientations relative to it must stay stable).
//
// Aggregates (sum, min) are order-independent, so the reversal lazy only has
// to swap children — sum/min/argmin are unaffected by path direction.
//
// Self-contained: no LEMON dependency.  Header-only, single class template
// parameterized on the value type (must be a signed arithmetic type; an
// integer or double both work).
//
// Correctness is pinned by tests_cpp/test_link_cut_tree.cpp, which brute-
// forces every operation against an O(n) adjacency-list reference over many
// randomized link/cut/update/query sequences.
// -------------------------------------------------------------------------
#ifndef PYLMCF_LINK_CUT_TREE_H
#define PYLMCF_LINK_CUT_TREE_H

#include <limits>
#include <utility>
#include <vector>

namespace pylmcf {

template <typename Val>
class LinkCutTree {
 public:
  // n real nodes, ids 0..n-1.  Index n is an internal NIL sentinel.
  explicit LinkCutTree(int n) : _n(n) {
    const int NIL = _n;
    _ch0.assign(n + 1, NIL);
    _ch1.assign(n + 1, NIL);
    _fa.assign(n + 1, NIL);
    _rev.assign(n + 1, 0);
    _val.assign(n + 1, Val(0));
    _add.assign(n + 1, Val(0));
    _sz.assign(n + 1, 1);
    _sum.assign(n + 1, Val(0));
    _mn.assign(n + 1, Val(0));
    _mnNode.assign(n + 1, -1);
    // NIL is the aggregate identity: empty path.
    _sz[NIL] = 0;
    _sum[NIL] = Val(0);
    _mn[NIL] = _INF();
    _mnNode[NIL] = -1;
    for (int i = 0; i < n; ++i) _mnNode[i] = i;
  }

  // ---- node value access -------------------------------------------------

  // Set node x's scalar value (the value that participates in path-sum and
  // path-min).  O(log n).
  void setVal(int x, Val v) {
    access(x);
    _val[x] = v;
    pushUp(x);
  }

  // Current scalar value of node x.  O(log n).
  Val getVal(int x) {
    access(x);
    return _val[x];
  }

  // ---- structural --------------------------------------------------------

  // True iff x and y are in the same tree.  O(log n).
  bool connected(int x, int y) { return findRoot(x) == findRoot(y); }

  // Root of x's tree under the current orientation.  O(log n).
  int findRoot(int x) {
    access(x);
    while (_ch0[x] != _n) {
      pushDown(x);
      x = _ch0[x];
    }
    splay(x);
    return x;
  }

  // Make x the root of its tree (evert).  O(log n).
  void makeRoot(int x) {
    access(x);
    applyRev(x);
  }

  // Parent of x relative to the current tree root, or -1 if x is the root.
  // O(log n).
  int parentOf(int x) {
    access(x);
    int p = _ch0[x];
    if (p == _n) return -1;
    pushDown(p);
    while (_ch1[p] != _n) {
      p = _ch1[p];
      pushDown(p);
    }
    splay(p);
    return p;
  }

  // Add edge x—y.  Precondition: x and y are in different trees.  O(log n).
  void link(int x, int y) {
    makeRoot(x);
    _fa[x] = y;
  }

  // Remove edge x—y.  Precondition: edge x—y exists.  O(log n).
  void cut(int x, int y) {
    makeRoot(x);
    access(y);
    splay(y);
    // path x..y is now the splay rooted at y; for an existing direct edge
    // x—y, x is y's left child and x has no right child.
    _ch0[y] = _n;
    _fa[x] = _n;
    pushUp(y);
  }

  // Detach x from its parent (the edge x—parentOf(x)) WITHOUT re-rooting the
  // tree.  Precondition: x is not the current tree root (it has a parent).
  // The component that keeps the old root stays rooted at that old root; the
  // detached component becomes a new tree rooted at x.  O(log n).
  //
  // This is the fixed-root-preserving cut a network-simplex backend needs:
  // the artificial root must not move when the leaving tree arc is removed.
  void cutParent(int x) {
    access(x);
    // After access(x): splay rooted at x, its left subtree is exactly the
    // root..parent(x) path.  Severing it removes the x—parent edge and leaves
    // that path's tree rooted at the original root.
    int p = _ch0[x];
    _ch0[x] = _n;
    _fa[p] = _n;
    pushUp(x);
  }

  // Lowest common ancestor of u and v relative to the CURRENT tree root
  // (does not re-root; valid only if the tree root has not been moved since
  // the intended root was established).  O(log n).
  int lca(int u, int v) {
    access(u);
    return access(v);
  }

  // ---- u..v path operations (re-root: call makeRoot(u)) ------------------

  Val pathSum(int u, int v) {
    makeRoot(u);
    access(v);
    return _sum[v];
  }

  // Returns {min value on path, a node id achieving it}.
  std::pair<Val, int> pathMin(int u, int v) {
    makeRoot(u);
    access(v);
    return {_mn[v], _mnNode[v]};
  }

  void pathAdd(int u, int v, Val d) {
    makeRoot(u);
    access(v);
    applyAdd(v, d);
  }

  // Number of edges on the u..v path.  O(log n).
  int pathLen(int u, int v) {
    makeRoot(u);
    access(v);
    return _sz[v] - 1;
  }

  // ---- root..u path operations (no re-root; fixed-root friendly) ---------

  Val sumToRoot(int u) {
    access(u);
    return _sum[u];
  }

  std::pair<Val, int> minToRoot(int u) {
    access(u);
    return {_mn[u], _mnNode[u]};
  }

  void addToRoot(int u, Val d) {
    access(u);
    applyAdd(u, d);
  }

 private:
  Val _INF() const { return std::numeric_limits<Val>::max() / 2; }

  void applyAdd(int x, Val d) {
    if (x == _n) return;
    _val[x] += d;
    _sum[x] += d * Val(_sz[x]);
    _mn[x] += d;
    _add[x] += d;
  }

  void applyRev(int x) {
    if (x == _n) return;
    std::swap(_ch0[x], _ch1[x]);
    _rev[x] ^= 1;
  }

  void pushDown(int x) {
    if (_rev[x]) {
      applyRev(_ch0[x]);
      applyRev(_ch1[x]);
      _rev[x] = 0;
    }
    if (_add[x] != Val(0)) {
      applyAdd(_ch0[x], _add[x]);
      applyAdd(_ch1[x], _add[x]);
      _add[x] = Val(0);
    }
  }

  void pushUp(int x) {
    const int l = _ch0[x], r = _ch1[x];
    _sz[x] = _sz[l] + 1 + _sz[r];
    _sum[x] = _sum[l] + _val[x] + _sum[r];
    Val m = _val[x];
    int mn = x;
    if (_mn[l] < m) {
      m = _mn[l];
      mn = _mnNode[l];
    }
    if (_mn[r] < m) {
      m = _mn[r];
      mn = _mnNode[r];
    }
    _mn[x] = m;
    _mnNode[x] = mn;
  }

  bool isRoot(int x) {
    int p = _fa[x];
    return p == _n || (_ch0[p] != x && _ch1[p] != x);
  }

  void rotate(int x) {
    int y = _fa[x], z = _fa[y];
    bool xr = (_ch1[y] == x);
    int b = xr ? _ch0[x] : _ch1[x];
    if (!isRoot(y)) {
      if (_ch0[z] == y)
        _ch0[z] = x;
      else
        _ch1[z] = x;
    }
    _fa[x] = z;
    if (xr) {
      _ch1[y] = b;
      _ch0[x] = y;
    } else {
      _ch0[y] = b;
      _ch1[x] = y;
    }
    if (b != _n) _fa[b] = y;
    _fa[y] = x;
    pushUp(y);
    pushUp(x);
  }

  void splay(int x) {
    // Push down lazies along the splay path root..x first.
    static std::vector<int> stk;
    stk.clear();
    int t = x;
    stk.push_back(t);
    while (!isRoot(t)) {
      t = _fa[t];
      stk.push_back(t);
    }
    for (int i = (int)stk.size() - 1; i >= 0; --i) pushDown(stk[i]);
    while (!isRoot(x)) {
      int y = _fa[x], z = _fa[y];
      if (!isRoot(y)) {
        if ((_ch0[z] == y) == (_ch0[y] == x))
          rotate(y);
        else
          rotate(x);
      }
      rotate(x);
    }
  }

  // Expose the preferred path from the tree root down to x; return the last
  // path-parent node encountered (the rooted-LCA witness).
  int access(int x) {
    int last = _n;
    for (int y = x; y != _n; y = _fa[y]) {
      splay(y);
      _ch1[y] = last;
      pushUp(y);
      last = y;
    }
    splay(x);
    return last;
  }

  int _n;
  std::vector<int> _ch0, _ch1, _fa;
  std::vector<char> _rev;
  std::vector<Val> _val, _add, _sum, _mn;
  std::vector<int> _sz, _mnNode;
};

}  // namespace pylmcf

#endif  // PYLMCF_LINK_CUT_TREE_H
