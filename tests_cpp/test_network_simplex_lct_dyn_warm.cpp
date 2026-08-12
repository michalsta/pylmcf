// test_network_simplex_lct_dyn_warm.cpp
// -------------------------------------------------------------------------
// Lever-1 oracle: NetworkSimplexLCTDyn::warmRun() (incremental supply-delta
// Simple fast path) vs an independent cold LEMON NetworkSimplex at every step.
//
//  - SPARSE chain: caps fixed (generous slack), only a few node supplies
//    nudged each step (Lever-1's target regime) -> exercises & validates the
//    incremental warm fast path (must fire often: coverage guard).
//  - ADVERSARIAL: re-randomize all supplies each step -> the retained basis
//    usually breaks -> warmRun cold-falls-back; validates that path's
//    correctness too.
//
// Oracle = exact integer totalCost equality vs independent cold LEMON +
// primal feasibility/conservation of the Dyn flow.
//
// Build:
//   g++ -I$(python -m pylmcf --include) -std=c++20 -O2 \
//       tests_cpp/test_network_simplex_lct_dyn_warm.cpp -o /tmp/tdw && /tmp/tdw
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
using DYN = pylmcf::NetworkSimplexLCTDyn<Value, Value>;

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

// A chain-ish feasible instance: path 0-1-..-(n-1) (bidirectional, cost=gap,
// big cap) + a few random extra arcs; balanced supplies from a witness flow.
struct Gen {
  std::mt19937_64 rng;
  explicit Gen(uint64_t s) : rng(s) {}
  int uni(int lo, int hi) {
    return std::uniform_int_distribution<int>(lo, hi)(rng);
  }
  Instance make(int n) {
    Instance in;
    in.n = n;
    for (int i = 0; i + 1 < n; ++i) {
      int g = 1 + uni(0, 8);
      in.arcs.push_back({i, i + 1}); in.cost.push_back(g);
      in.arcs.push_back({i + 1, i}); in.cost.push_back(g);
    }
    int extra = uni(0, n);
    for (int k = 0; k < extra; ++k) {
      int u = uni(0, n - 1), v = uni(0, n - 1);
      if (u == v) v = (v + 1) % n;
      in.arcs.push_back({u, v}); in.cost.push_back(uni(1, 40));
    }
    // sort arcs by (src,tgt) for LEMON StaticDigraph::build, permute cost
    const int m = (int)in.arcs.size();
    std::vector<int> p(m);
    for (int i = 0; i < m; ++i) p[i] = i;
    std::stable_sort(p.begin(), p.end(),
                     [&](int a, int b) { return in.arcs[a] < in.arcs[b]; });
    std::vector<std::pair<int, int>> a2(m);
    std::vector<Value> c2(m);
    for (int i = 0; i < m; ++i) { a2[i] = in.arcs[p[i]]; c2[i] = in.cost[p[i]]; }
    in.arcs.swap(a2); in.cost.swap(c2);
    in.cap.assign(m, (Value)5e8);              // FIXED big caps (no cap churn)
    in.supply.assign(n, 0);
    return in;
  }
  // Balanced supply from a random witness flow on the arc set.
  void witnessSupply(Instance& in, int amp) {
    std::fill(in.supply.begin(), in.supply.end(), 0);
    for (size_t i = 0; i < in.arcs.size(); ++i) {
      Value w = uni(0, amp);
      in.supply[in.arcs[i].first] += w;
      in.supply[in.arcs[i].second] -= w;
    }
  }
  // Sparse nudge: change a few node supplies, keep balanced (move mass
  // between two nodes).
  void sparseNudge(Instance& in, int moves) {
    for (int k = 0; k < moves; ++k) {
      int a = uni(0, in.n - 1), b = uni(0, in.n - 1);
      Value d = uni(1, 4);
      in.supply[a] += d; in.supply[b] -= d;
    }
  }
};

static Value lemon_cost(const Instance& in, bool& opt) {
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

static void verify(const char* tag, DYN& s, const Instance& in,
                   const std::vector<int>& ids, bool sopt) {
  bool lopt;
  Value lc = lemon_cost(in, lopt);
  CHECK(sopt == lopt, tag);
  if (!sopt || !lopt) return;
  CHECK(s.totalCost() == lc, "warm totalCost != cold LEMON");
  std::vector<Value> bal(in.n, 0);
  Value rec = 0;
  for (int i = 0; i < (int)in.arcs.size(); ++i) {
    Value f = s.flow(ids[i]);
    CHECK(f >= 0 && f <= in.cap[i], "flow out of [0,cap]");
    rec += f * in.cost[i];
    bal[in.arcs[i].first] -= f;
    bal[in.arcs[i].second] += f;
  }
  CHECK(rec == s.totalCost(), "Sum(c*f) != totalCost");
  for (int v = 0; v < in.n; ++v)
    CHECK(bal[v] == -in.supply[v], "conservation violated");
}

int main() {
  std::printf("NetworkSimplexLCTDyn Lever-1 warm vs cold LEMON\n");

  // (1) Adversarial: re-randomize all supplies each step (cold-fallback path).
  Gen g1(0xABCDEFu);
  for (int t = 0; t < 800; ++t) {
    int n = g1.uni(4, 30);
    Instance in = g1.make(n);
    g1.witnessSupply(in, 6);
    DYN s(in.n);
    std::vector<int> ids;
    for (int i = 0; i < (int)in.arcs.size(); ++i)
      ids.push_back(s.addArc(in.arcs[i].first, in.arcs[i].second,
                             in.cost[i], in.cap[i]));
    for (int v = 0; v < in.n; ++v) s.setSupply(v, in.supply[v]);
    verify("cold-prime", s, in, ids, s.run() == DYN::OPTIMAL);
    int steps = g1.uni(3, 8);
    for (int k = 0; k < steps; ++k) {
      g1.witnessSupply(in, 6);
      for (int v = 0; v < in.n; ++v) s.setSupply(v, in.supply[v]);
      ++g_steps;
      verify("adv-warm", s, in, ids, s.warmRun() == DYN::OPTIMAL);
    }
  }

  // (2) Sparse chain: fixed caps, tiny supply nudges -> incremental fast path.
  long warm_hits = 0, total_warm_steps = 0;
  Gen g2(0x5EED99u);
  for (int t = 0; t < 250; ++t) {
    int n = g2.uni(20, 120);
    Instance in = g2.make(n);
    g2.witnessSupply(in, 5);
    DYN s(in.n);
    std::vector<int> ids;
    for (int i = 0; i < (int)in.arcs.size(); ++i)
      ids.push_back(s.addArc(in.arcs[i].first, in.arcs[i].second,
                             in.cost[i], in.cap[i]));
    for (int v = 0; v < in.n; ++v) s.setSupply(v, in.supply[v]);
    verify("cov-prime", s, in, ids, s.run() == DYN::OPTIMAL);
    int steps = g2.uni(6, 14);
    for (int k = 0; k < steps; ++k) {
      g2.sparseNudge(in, g2.uni(1, 3));
      for (int v = 0; v < in.n; ++v) s.setSupply(v, in.supply[v]);
      int wc0 = s.warmCount();
      auto st = s.warmRun();
      ++total_warm_steps;
      if (s.warmCount() > wc0) ++warm_hits;
      ++g_steps;
      verify("cov-warm", s, in, ids, st == DYN::OPTIMAL);
    }
  }
  std::printf("sparse chain: warm fast-path hits=%ld / %ld\n",
              warm_hits, total_warm_steps);
  if (warm_hits < 50) {
    std::printf("INEFFECTIVE: warm fast path fired only %ld (<50)\n", warm_hits);
    ++g_fail;
  }

  std::printf("checks=%ld steps=%ld fails=%d\n", g_checks, g_steps, g_fail);
  if (g_fail) { std::printf("RESULT: FAILED (%d)\n", g_fail); return 1; }
  std::printf("RESULT: PASSED\n");
  return 0;
}
