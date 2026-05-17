// test_link_cut_tree.cpp
// -------------------------------------------------------------------------
// Brute-force correctness suite for pylmcf::LinkCutTree.
//
// Maintains a forest twice: once in the link-cut tree under test, once in a
// trivial O(n) adjacency-list + per-node-value reference.  Over many random
// sequences of link / cut / setVal / pathAdd it cross-checks every query:
//
//   connected, findRoot, pathSum, pathMin (value + argmin membership),
//   pathLen, and the fixed-root family (makeRoot then lca / sumToRoot /
//   minToRoot / addToRoot with no intervening structural op).
//
// Build (same -I as test_dual_repair.cpp):
//   g++ -I$(python -m pylmcf --include) -std=c++20 -O2 \
//       tests_cpp/test_link_cut_tree.cpp -o /tmp/tlct && /tmp/tlct
// -------------------------------------------------------------------------
#include <pylmcf/link_cut_tree.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <queue>
#include <random>
#include <vector>

using Val = int64_t;
using LCT = pylmcf::LinkCutTree<Val>;

static int g_fail = 0;
static long g_checks = 0;

#define CHECK(cond, msg)                                                   \
  do {                                                                     \
    ++g_checks;                                                            \
    if (!(cond)) {                                                         \
      std::printf("  FAIL [%s:%d] %s\n", __func__, __LINE__, msg);         \
      ++g_fail;                                                            \
    }                                                                      \
  } while (0)

// ---- O(n) reference forest ---------------------------------------------
struct Ref {
  int n;
  std::vector<std::vector<int>> adj;
  std::vector<Val> val;
  explicit Ref(int n_) : n(n_), adj(n_), val(n_, 0) {}

  void link(int u, int v) {
    adj[u].push_back(v);
    adj[v].push_back(u);
  }
  void cut(int u, int v) {
    adj[u].erase(std::find(adj[u].begin(), adj[u].end(), v));
    adj[v].erase(std::find(adj[v].begin(), adj[v].end(), u));
  }
  bool edge(int u, int v) const {
    return std::find(adj[u].begin(), adj[u].end(), v) != adj[u].end();
  }
  // Unique u..v path (forest), or empty if disconnected.
  std::vector<int> path(int u, int v) const {
    std::vector<int> par(n, -2);
    std::queue<int> q;
    q.push(u);
    par[u] = -1;
    while (!q.empty()) {
      int x = q.front();
      q.pop();
      if (x == v) break;
      for (int y : adj[x])
        if (par[y] == -2) {
          par[y] = x;
          q.push(y);
        }
    }
    if (par[v] == -2) return {};
    std::vector<int> p;
    for (int x = v; x != -1; x = par[x]) p.push_back(x);
    std::reverse(p.begin(), p.end());
    return p;
  }
  bool connected(int u, int v) const { return u == v || !path(u, v).empty(); }
  int root(int from, int x) const {  // root of x's tree if 'from' is root
    auto p = path(from, x);
    return p.empty() ? (x == from ? from : -1) : from;
  }
};

static void campaign(uint64_t seed, int n, int ops) {
  std::mt19937_64 rng(seed);
  auto uni = [&](int lo, int hi) {
    return std::uniform_int_distribution<int>(lo, hi)(rng);
  };
  LCT t(n);
  Ref r(n);

  for (int it = 0; it < ops; ++it) {
    int kind = uni(0, 9);
    if (kind <= 2) {  // link
      int u = uni(0, n - 1), v = uni(0, n - 1);
      if (u != v && !r.connected(u, v)) {
        r.link(u, v);
        t.link(u, v);
      }
    } else if (kind == 3) {  // cut a real edge
      int u = uni(0, n - 1);
      if (!r.adj[u].empty()) {
        int v = r.adj[u][uni(0, (int)r.adj[u].size() - 1)];
        r.cut(u, v);
        t.cut(u, v);
      }
    } else if (kind == 4) {  // setVal
      int u = uni(0, n - 1);
      Val v = uni(-50, 50);
      r.val[u] = v;
      t.setVal(u, v);
    } else if (kind == 5) {  // pathAdd
      int u = uni(0, n - 1), v = uni(0, n - 1);
      if (r.connected(u, v)) {
        Val d = uni(-20, 20);
        for (int x : r.path(u, v)) r.val[x] += d;
        t.pathAdd(u, v, d);
      }
    } else {  // queries
      int u = uni(0, n - 1), v = uni(0, n - 1);
      CHECK(t.connected(u, v) == r.connected(u, v), "connected");
      if (r.connected(u, v)) {
        auto p = r.path(u, v);
        Val s = 0, mn = p.empty() ? 0 : r.val[p[0]];
        for (int x : p) {
          s += r.val[x];
          mn = std::min(mn, r.val[x]);
        }
        CHECK(t.pathSum(u, v) == s, "pathSum");
        CHECK(t.pathLen(u, v) == (int)p.size() - 1, "pathLen");
        auto pm = t.pathMin(u, v);
        CHECK(pm.first == mn, "pathMin value");
        bool on = std::find(p.begin(), p.end(), pm.second) != p.end();
        CHECK(on && r.val[pm.second] == mn, "pathMin argmin");
      }
    }

    // Fixed-root block: pick a root R, evert to it, then run rooted queries
    // with NO intervening structural / makeRoot operation.
    if (kind == 9) {
      int R = uni(0, n - 1);
      t.makeRoot(R);
      for (int q = 0; q < 4; ++q) {
        int u = uni(0, n - 1);
        if (!r.connected(R, u)) continue;
        auto pr = r.path(R, u);
        Val s = 0, mn = pr.empty() ? 0 : r.val[pr[0]];
        for (int x : pr) {
          s += r.val[x];
          mn = std::min(mn, r.val[x]);
        }
        CHECK(t.sumToRoot(u) == s, "sumToRoot");
        auto m = t.minToRoot(u);
        CHECK(m.first == mn, "minToRoot value");
        // lca(R-rooted) of u and v vs reference (deepest common node on the
        // two root-paths).
        int v = uni(0, n - 1);
        if (r.connected(R, v)) {
          auto pv = r.path(R, v);
          int common = R;
          for (int i = 0; i < (int)std::min(pr.size(), pv.size()); ++i) {
            if (pr[i] == pv[i])
              common = pr[i];
            else
              break;
          }
          CHECK(t.lca(u, v) == common, "lca rooted");
        }
        // addToRoot then read back via sumToRoot.
        Val d = uni(-10, 10);
        for (int x : pr) r.val[x] += d;
        t.addToRoot(u, d);
        Val s2 = 0;
        for (int x : r.path(R, u)) s2 += r.val[x];
        CHECK(t.sumToRoot(u) == s2, "addToRoot+sumToRoot");
      }
    }
  }
}

// Dedicated cutParent campaign: it must sever x from its parent WITHOUT
// re-rooting, so the old-root side stays rooted at the fixed root R (the
// invariant the network-simplex backend relies on).  No makeRoot / pathAdd
// between fixing R and the rooted checks.
static void cutparent_campaign(uint64_t seed, int n, int ops) {
  std::mt19937_64 rng(seed);
  auto uni = [&](int lo, int hi) {
    return std::uniform_int_distribution<int>(lo, hi)(rng);
  };
  LCT t(n);
  Ref r(n);
  for (int it = 0; it < ops; ++it) {
    int kind = uni(0, 6);
    if (kind <= 3) {
      int u = uni(0, n - 1), v = uni(0, n - 1);
      if (u != v && !r.connected(u, v)) {
        r.link(u, v);
        t.link(u, v);
      }
    } else if (kind == 4) {
      int u = uni(0, n - 1);
      r.val[u] = uni(-40, 40);
      t.setVal(u, r.val[u]);
    } else {
      int R = uni(0, n - 1);
      t.makeRoot(R);  // fix the root; no further evert until checks done
      int x = uni(0, n - 1);
      if (x == R || !r.connected(R, x)) continue;
      // reference parent of x toward R
      auto px = r.path(R, x);
      int p = px[(int)px.size() - 2];
      r.cut(x, p);
      t.cutParent(x);
      // old-root side stays rooted at R; x side becomes its own tree.
      CHECK(t.findRoot(x) == x, "cutParent: x is its own root");
      // NOTE: findRoot(x) above does an access but not an evert; R side
      // unaffected.  Verify a few rooted path-sums on the R side.
      for (int q = 0; q < 4; ++q) {
        int y = uni(0, n - 1);
        CHECK(t.connected(R, y) == r.connected(R, y), "cutParent: connectivity");
        if (r.connected(R, y)) {
          Val s = 0;
          for (int z : r.path(R, y)) s += r.val[z];
          CHECK(t.sumToRoot(y) == s, "cutParent: sumToRoot on R side");
        }
      }
      r.link(x, p);  // restore; link everts only x's small component
      t.link(x, p);
    }
  }
}

int main() {
  std::printf("link-cut tree brute-force suite\n");
  struct {
    int n, ops, reps;
  } cfgs[] = {{6, 400, 40}, {12, 1200, 30}, {30, 4000, 12}, {80, 9000, 5}};
  uint64_t seed = 0x1234ABCDULL;
  for (auto c : cfgs)
    for (int r = 0; r < c.reps; ++r) campaign(seed++, c.n, c.ops);

  cutparent_campaign(seed++, 12, 3000);
  cutparent_campaign(seed++, 40, 6000);

  std::printf("checks=%ld  fails=%d\n", g_checks, g_fail);
  if (g_fail) {
    std::printf("RESULT: FAILED (%d)\n", g_fail);
    return 1;
  }
  std::printf("RESULT: PASSED\n");
  return 0;
}
