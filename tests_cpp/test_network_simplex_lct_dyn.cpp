// test_network_simplex_lct.cpp
// -------------------------------------------------------------------------
// Phase B oracle: NetworkSimplexLCT (link-cut-tree cold solver) vs LEMON's
// array NetworkSimplex on the same random feasible instances.
//
// Instances are generated exactly like the dual-repair suite: EQ supply
// (sum == 0) by construction from a witness flow, costs in [0,50], finite
// caps >= witness (so every instance is feasible).  For each instance we
// assert both solvers report OPTIMAL and identical integer total cost, and
// independently verify the LCT solution is primal-feasible (bounds +
// conservation) and that Sum(cost*flow) == reported cost.
//
// Build:
//   g++ -I$(python -m pylmcf --include) -std=c++20 -O2 \
//       tests_cpp/test_network_simplex_lct.cpp -o /tmp/tnsl && /tmp/tnsl
// -------------------------------------------------------------------------
#define LEMON_ONLY_TEMPLATES
#include <lemon/static_graph.h>
#include <lemon/network_simplex.h>
#include <pylmcf/network_simplex_lct_dyn.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using Value = int64_t;
using Graph = lemon::StaticDigraph;
using LemonNS = lemon::NetworkSimplex<Graph, Value, Value>;
using DYNNS = pylmcf::NetworkSimplexLCTDyn<Value, Value>;

static int g_fail = 0;
static long g_checks = 0, g_opt = 0;

#define CHECK(cond, msg)                                              \
  do {                                                                \
    ++g_checks;                                                       \
    if (!(cond)) {                                                    \
      std::printf("  FAIL [%s:%d] %s\n", __func__, __LINE__, msg);    \
      ++g_fail;                                                       \
    }                                                                 \
  } while (0)

struct Instance {
  int n;
  std::vector<std::pair<int, int>> arcs;
  std::vector<Value> cost, cap, supply, wit;
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
    auto perm = [&](auto& v) {
      std::decay_t<decltype(v)> t(v.size());
      for (int i = 0; i < (int)v.size(); ++i) t[i] = v[p[i]];
      v.swap(t);
    };
    perm(in.arcs);
    perm(in.cost);
    in.cap.assign(m, 0);
    in.supply.assign(n, 0);
    in.wit.assign(m, 0);
    mutate(in);
    return in;
  }
  void mutate(Instance& in) {
    const int m = (int)in.arcs.size();
    std::vector<Value> w(m), sup(in.n, 0);
    for (int i = 0; i < m; ++i) {
      w[i] = uni(0, 12);
      sup[in.arcs[i].first] += w[i];
      sup[in.arcs[i].second] -= w[i];
    }
    in.wit = w;
    in.supply = sup;
    for (int i = 0; i < m; ++i)
      in.cap[i] = (uni(0, 2) == 0) ? w[i] : w[i] + uni(0, 18);
  }
};

static Value lemon_solve(const Instance& in, bool& opt) {
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

static void check_instance(const char* tag, const Instance& in) {
  bool lopt;
  Value lcost = lemon_solve(in, lopt);

  DYNNS s(in.n);
  std::vector<int> ids;
  for (int i = 0; i < (int)in.arcs.size(); ++i)
    ids.push_back(
        s.addArc(in.arcs[i].first, in.arcs[i].second, in.cost[i], in.cap[i]));
  for (int v = 0; v < in.n; ++v) s.setSupply(v, in.supply[v]);
  auto st = s.run();
  bool sopt = (st == DYNNS::OPTIMAL);

  bool bad = (sopt != lopt) || (sopt && lopt && s.totalCost() != lcost);
  if (bad && g_fail < 3) {
    std::printf("REPRO n=%d m=%d  lemon{opt=%d cost=%lld} lct{opt=%d cost=%lld}\n",
                in.n, (int)in.arcs.size(), (int)lopt, (long long)lcost,
                (int)sopt, (long long)s.totalCost());
    std::printf("arcs=");
    for (auto& e : in.arcs) std::printf("(%d,%d)", e.first, e.second);
    std::printf("\ncost=");
    for (auto c : in.cost) std::printf("%lld,", (long long)c);
    std::printf("\ncap=");
    for (auto c : in.cap) std::printf("%lld,", (long long)c);
    std::printf("\nsup=");
    for (auto c : in.supply) std::printf("%lld,", (long long)c);
    std::printf("\n");
  }

  CHECK(sopt == lopt, tag);
  if (!sopt || !lopt) return;
  ++g_opt;
  CHECK(s.totalCost() == lcost, "totalCost != LEMON oracle");

  // Independent primal-feasibility + objective recompute.
  std::vector<Value> bal(in.n, 0);
  Value recomputed = 0;
  for (int i = 0; i < (int)in.arcs.size(); ++i) {
    Value f = s.flow(ids[i]);
    CHECK(f >= 0 && f <= in.cap[i], "flow out of [0,cap]");
    recomputed += f * in.cost[i];
    bal[in.arcs[i].first] -= f;
    bal[in.arcs[i].second] += f;
  }
  CHECK(recomputed == s.totalCost(), "Sum(c*f) != totalCost");
  for (int v = 0; v < in.n; ++v)
    CHECK(bal[v] == -in.supply[v], "conservation violated");
}

int main() {
  std::printf("NetworkSimplexLCTDyn vs LEMON oracle\n");
  Gen gen(0xC0FFEEu);
  for (int t = 0; t < 6000; ++t) {
    int n = gen.uni(3, 14);
    int m = gen.uni(n, n * 4);
    check_instance("lct-vs-lemon", gen.make(n, m));
  }
  // A few larger ones.
  for (int t = 0; t < 400; ++t) {
    int n = gen.uni(20, 45);
    int m = gen.uni(n, n * 5);
    check_instance("lct-vs-lemon-big", gen.make(n, m));
  }
  std::printf("checks=%ld  optimal_instances=%ld  fails=%d\n", g_checks, g_opt,
              g_fail);
  if (g_fail) {
    std::printf("RESULT: FAILED (%d)\n", g_fail);
    return 1;
  }
  std::printf("RESULT: PASSED\n");
  return 0;
}
