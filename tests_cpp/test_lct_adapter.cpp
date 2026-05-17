// test_lct_adapter.cpp
// -------------------------------------------------------------------------
// Phase D step-1 oracle: drive pylmcf::NetworkSimplexLCTAdapter through the
// EXACT call pattern wnet's decompositable_graph.hpp uses on a
// lemon::NetworkSimplex (StaticDigraph + ArcMap/NodeMap; emplace -> upperMap
// -> costMap -> supplyMap -> run; then upperMap -> supplyMap -> warmRun
// chain), and validate against an independent cold lemon::NetworkSimplex.
//
// Oracle = exact integer totalCost equality + primal feasibility/conservation
// of the adapter's flow (a feasible flow whose cost equals the independent
// optimum is optimal).  Potentials are NOT compared to LEMON's: at degenerate
// optima the LCT basis differs, so pi differs — documented, accepted (same
// reconciliation as the array-side Dual/DualRatio modes).  We DO check the
// adapter's potential() is self-consistent: every basic (tree) real arc has
// reduced cost 0, and potential() is finite/usable for the gradient path.
//
// Build:
//   g++ -I$(python -m pylmcf --include) -std=c++20 -O2 \
//       tests_cpp/test_lct_adapter.cpp -o /tmp/tla && /tmp/tla
// -------------------------------------------------------------------------
#define LEMON_ONLY_TEMPLATES
#include <lemon/static_graph.h>
#include <lemon/network_simplex.h>
#include <pylmcf/network_simplex_lct_adapter.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using Value = int64_t;
using Graph = lemon::StaticDigraph;
using LemonNS = lemon::NetworkSimplex<Graph, Value, Value>;
using Adapter = pylmcf::NetworkSimplexLCTAdapter<Graph, Value, Value>;

static int g_fail = 0;
static long g_checks = 0, g_steps = 0;

#define CHECK(c, m)                                                  \
  do {                                                               \
    ++g_checks;                                                      \
    if (!(c)) {                                                      \
      std::printf("  FAIL [%s:%d] %s\n", __func__, __LINE__, m);     \
      ++g_fail;                                                      \
    }                                                                \
  } while (0)

struct Instance {
  int n;
  std::vector<std::pair<int, int>> arcs;
  std::vector<Value> cost, cap, supply;
};

struct Gen {
  std::mt19937_64 rng;
  explicit Gen(uint64_t s) : rng(s) {}
  int uni(int lo, int hi) {
    return std::uniform_int_distribution<int>(lo, hi)(rng);
  }
  Instance make(int n, int m) {
    Instance in;
    in.n = n;
    for (int e = 0; e < m; ++e) {
      int u = uni(0, n - 1), v = uni(0, n - 1);
      if (u == v) v = (v + 1) % n;
      in.arcs.push_back({u, v});
    }
    in.cost.resize(m);
    for (int i = 0; i < m; ++i) in.cost[i] = uni(0, 50);
    std::vector<int> p(m);
    for (int i = 0; i < m; ++i) p[i] = i;
    std::stable_sort(p.begin(), p.end(),
                     [&](int x, int y) { return in.arcs[x] < in.arcs[y]; });
    auto perm = [&](auto& vv) {
      std::decay_t<decltype(vv)> t(vv.size());
      for (int i = 0; i < (int)vv.size(); ++i) t[i] = vv[p[i]];
      vv.swap(t);
    };
    perm(in.arcs);
    perm(in.cost);
    in.cap.assign(m, 0);
    in.supply.assign(n, 0);
    mutate(in);
    return in;
  }
  void mutate(Instance& in) {  // caps + supply, costs fixed (warm contract)
    const int m = (int)in.arcs.size();
    std::vector<Value> w(m), sup(in.n, 0);
    for (int i = 0; i < m; ++i) {
      w[i] = uni(0, 12);
      sup[in.arcs[i].first] += w[i];
      sup[in.arcs[i].second] -= w[i];
    }
    in.supply = sup;
    for (int i = 0; i < m; ++i)
      in.cap[i] = (uni(0, 2) == 0) ? w[i] : w[i] + uni(0, 18);
  }
};

static Value lemon_cold(const Instance& in, bool& opt) {
  Graph g;
  g.build(in.n, in.arcs.begin(), in.arcs.end());
  Graph::ArcMap<Value> cm(g), um(g);
  Graph::NodeMap<Value> sm(g);
  for (int i = 0; i < (int)in.arcs.size(); ++i) {
    cm[g.arcFromId(i)] = in.cost[i];
    um[g.arcFromId(i)] = in.cap[i];
  }
  for (int v = 0; v < in.n; ++v) sm[g.nodeFromId(v)] = in.supply[v];
  LemonNS ns(g);
  ns.upperMap(um).costMap(cm).supplyMap(sm);
  auto st = ns.run();
  opt = (st == LemonNS::OPTIMAL);
  return opt ? ns.totalCost() : Value(0);
}

// Validate the adapter's current solution vs an independent cold LEMON solve.
static void verify(const char* tag, const Graph& g, Adapter& s,
                   const Instance& in, bool sopt) {
  bool lopt;
  Value lc = lemon_cold(in, lopt);
  CHECK(sopt == lopt, tag);
  if (!sopt || !lopt) return;
  CHECK(s.totalCost() == lc, "adapter totalCost != independent cold LEMON");

  std::vector<Value> bal(in.n, 0);
  Value rec = 0;
  for (int i = 0; i < (int)in.arcs.size(); ++i) {
    Value f = s.flow(g.arcFromId(i));
    CHECK(f >= 0 && f <= in.cap[i], "flow out of [0,cap]");
    rec += f * in.cost[i];
    bal[in.arcs[i].first] -= f;
    bal[in.arcs[i].second] += f;
  }
  CHECK(rec == s.totalCost(), "Sum(c*f) != totalCost");
  bool cons_ok = true;
  for (int v = 0; v < in.n; ++v)
    if (bal[v] != -in.supply[v]) cons_ok = false;
  if (!cons_ok && g_fail < 2) {
    std::printf("REPRO[%s] n=%d m=%d cost==lemon? %d\n", tag, in.n,
                (int)in.arcs.size(), (int)(s.totalCost() == lc));
    std::printf("arcs=");
    for (auto& e : in.arcs) std::printf("(%d,%d)", e.first, e.second);
    std::printf("\ncost=");
    for (auto c : in.cost) std::printf("%lld,", (long long)c);
    std::printf("\ncap=");
    for (auto c : in.cap) std::printf("%lld,", (long long)c);
    std::printf("\nsup=");
    for (auto c : in.supply) std::printf("%lld,", (long long)c);
    std::printf("\nflow=");
    for (int i = 0; i < (int)in.arcs.size(); ++i)
      std::printf("%lld,", (long long)s.flow(g.arcFromId(i)));
    std::printf("\n");
  }
  for (int v = 0; v < in.n; ++v)
    CHECK(bal[v] == -in.supply[v], "conservation violated");

  // potential() self-consistency / gradient-path usability: a basic real arc
  // (0 < flow < cap) must have reduced cost cost+pi[u]-pi[v] == 0.  This is
  // the property the residual-Dijkstra / pi-distance gradient relies on.
  for (int i = 0; i < (int)in.arcs.size(); ++i) {
    Value f = s.flow(g.arcFromId(i));
    if (f > 0 && f < in.cap[i]) {
      Value rc = in.cost[i] +
                 s.potential(g.nodeFromId(in.arcs[i].first)) -
                 s.potential(g.nodeFromId(in.arcs[i].second));
      CHECK(rc == 0, "basic-arc reduced cost != 0 (potential inconsistent)");
    }
  }
}

int main() {
  std::printf("LCT adapter vs LEMON (wnet call pattern)\n");

  // (1) Adversarial cold + warm chain: re-randomize caps/supply each step
  //     (this stresses warmRun's cold-fallback correctness).
  Gen gen(0xC0FFEEu);
  for (int t = 0; t < 1200; ++t) {
    int n = gen.uni(3, 14), m = gen.uni(n, n * 4);
    Instance in = gen.make(n, m);

    Graph g;
    g.build(in.n, in.arcs.begin(), in.arcs.end());
    Graph::ArcMap<Value> cm(g), um(g);
    Graph::NodeMap<Value> sm(g);
    auto push = [&] {
      for (int i = 0; i < m; ++i) {
        cm[g.arcFromId(i)] = in.cost[i];
        um[g.arcFromId(i)] = in.cap[i];
      }
      for (int v = 0; v < in.n; ++v) sm[g.nodeFromId(v)] = in.supply[v];
    };
    push();

    Adapter s(g);                                  // wnet: emplace(graph)
    s.upperMap(um).costMap(cm).supplyMap(sm);      // wnet cold path
    auto st = s.run(Adapter::BLOCK_SEARCH);
    verify("cold-prime", g, s, in, st == Adapter::OPTIMAL);

    int steps = gen.uni(3, 9);
    for (int k = 0; k < steps; ++k) {
      gen.mutate(in);
      push();
      s.upperMap(um).supplyMap(sm);                // wnet warm path
      auto wst = s.warmRun(Adapter::BLOCK_SEARCH, Adapter::WarmRepair::Dual);
      ++g_steps;
      verify("warm-step", g, s, in, wst == Adapter::OPTIMAL);
    }
  }

  // (2) Gentle chain: caps fixed slack, tiny supply nudges -> exercises &
  //     validates the warm fast path through the adapter.
  {
    std::mt19937_64 rng(0x5EED77u);
    auto uni = [&](int lo, int hi) {
      return std::uniform_int_distribution<int>(lo, hi)(rng);
    };
    const int n = 9, m = 24;
    Instance in;
    in.n = n;
    for (int e = 0; e < m; ++e) {
      int u = uni(0, n - 1), v = uni(0, n - 1);
      if (u == v) v = (v + 1) % n;
      in.arcs.push_back({u, v});
    }
    in.cost.resize(m);
    for (int i = 0; i < m; ++i) in.cost[i] = uni(0, 50);
    {
      std::vector<int> p(m);
      for (int i = 0; i < m; ++i) p[i] = i;
      std::stable_sort(p.begin(), p.end(),
                       [&](int x, int y) { return in.arcs[x] < in.arcs[y]; });
      std::vector<std::pair<int, int>> a2(m);
      std::vector<Value> c2(m);
      for (int i = 0; i < m; ++i) {
        a2[i] = in.arcs[p[i]];
        c2[i] = in.cost[p[i]];
      }
      in.arcs.swap(a2);
      in.cost.swap(c2);
    }
    std::vector<Value> w0(m);
    for (int i = 0; i < m; ++i) w0[i] = uni(2, 10);
    in.cap.assign(m, 0);
    for (int i = 0; i < m; ++i) in.cap[i] = w0[i] + 40;
    auto setSup = [&](const std::vector<Value>& w) {
      in.supply.assign(n, 0);
      for (int i = 0; i < m; ++i) {
        in.supply[in.arcs[i].first] += w[i];
        in.supply[in.arcs[i].second] -= w[i];
      }
    };
    setSup(w0);

    Graph g;
    g.build(in.n, in.arcs.begin(), in.arcs.end());
    Graph::ArcMap<Value> cm(g), um(g);
    Graph::NodeMap<Value> sm(g);
    for (int i = 0; i < m; ++i) {
      cm[g.arcFromId(i)] = in.cost[i];
      um[g.arcFromId(i)] = in.cap[i];
    }
    for (int v = 0; v < n; ++v) sm[g.nodeFromId(v)] = in.supply[v];

    Adapter s(g);
    s.upperMap(um).costMap(cm).supplyMap(sm);
    s.run();
    verify("cov-prime", g, s, in, true);

    int warm_hits = 0;
    for (int k = 0; k < 200; ++k) {
      std::vector<Value> w = w0;
      if (k % 5 != 0)
        for (int i = 0; i < m; ++i)
          w[i] = std::max<Value>(0, w0[i] + uni(-2, 2));
      setSup(w);
      for (int v = 0; v < n; ++v) sm[g.nodeFromId(v)] = in.supply[v];
      s.upperMap(um).supplyMap(sm);                // wnet warm path
      int wc0 = s.warmStartCount();
      auto wst = s.warmRun();
      if (s.warmStartCount() > wc0) ++warm_hits;
      verify("cov", g, s, in, wst == Adapter::OPTIMAL);
    }
    std::printf("gentle chain via adapter: warm_hits=%d / 200\n", warm_hits);
    if (warm_hits < 20) {
      std::printf("INEFFECTIVE: warm fast path fired %d (<20)\n", warm_hits);
      ++g_fail;
    }
  }

  std::printf("checks=%ld steps=%ld fails=%d\n", g_checks, g_steps, g_fail);
  if (g_fail) {
    std::printf("RESULT: FAILED (%d)\n", g_fail);
    return 1;
  }
  std::printf("RESULT: PASSED\n");
  return 0;
}
