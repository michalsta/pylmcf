// test_network_simplex_lct_warm.cpp
// -------------------------------------------------------------------------
// Phase C oracle: NetworkSimplexLCT::warmRun() warm-restart chains vs an
// independent cold LEMON NetworkSimplex at every step.
//
// Per instance: build the LCT solver once, cold run(), then a sequence of
// cap/supply mutations (costs fixed — the warm-restart contract).  After each
// mutation we call warmRun() and assert it agrees with a brand-new cold LEMON
// solve of the mutated data (OPTIMAL status + exact integer total cost), and
// independently verify the LCT flow is primal-feasible and conserves supply.
// A coverage guard asserts the warm (non-cold-fallback) path actually fired.
//
// Build:
//   g++ -I$(python -m pylmcf --include) -std=c++20 -O2 \
//       tests_cpp/test_network_simplex_lct_warm.cpp -o /tmp/tnslw && /tmp/tnslw
// -------------------------------------------------------------------------
#define LEMON_ONLY_TEMPLATES
#include <lemon/static_graph.h>
#include <lemon/network_simplex.h>
#include <pylmcf/network_simplex_lct.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using Value = int64_t;
using Graph = lemon::StaticDigraph;
using LemonNS = lemon::NetworkSimplex<Graph, Value, Value>;
using LCTNS = pylmcf::NetworkSimplexLCT<Value, Value>;

static int g_fail = 0;
static long g_checks = 0, g_steps = 0, g_warm = 0, g_cold = 0;

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
  // Re-randomize a feasible (witness-backed) supply + caps; costs untouched.
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

static void verify(const char* tag, LCTNS& s, const Instance& in,
                   const std::vector<int>& ids, bool sopt) {
  bool lopt;
  Value lcost = lemon_solve(in, lopt);
  CHECK(sopt == lopt, tag);
  if (!sopt || !lopt) return;
  CHECK(s.totalCost() == lcost, "warm totalCost != cold LEMON oracle");
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
  std::printf("NetworkSimplexLCT warm-restart vs cold LEMON oracle\n");
  Gen gen(0xC0FFEEu);
  const int instances = 1500;
  for (int t = 0; t < instances; ++t) {
    int n = gen.uni(3, 14);
    int m = gen.uni(n, n * 4);
    Instance in = gen.make(n, m);

    LCTNS s(in.n);
    std::vector<int> ids;
    for (int i = 0; i < m; ++i)
      ids.push_back(
          s.addArc(in.arcs[i].first, in.arcs[i].second, in.cost[i], in.cap[i]));
    for (int v = 0; v < in.n; ++v) s.setSupply(v, in.supply[v]);
    auto st = s.run();                              // cold prime
    verify("cold-prime", s, in, ids, st == LCTNS::OPTIMAL);

    const int steps = gen.uni(4, 12);
    for (int k = 0; k < steps; ++k) {
      gen.mutate(in);
      for (int i = 0; i < m; ++i) s.setCap(ids[i], in.cap[i]);
      for (int v = 0; v < in.n; ++v) s.setSupply(v, in.supply[v]);
      auto wst = s.warmRun();
      ++g_steps;
      verify("warm-step", s, in, ids, wst == LCTNS::OPTIMAL);
    }
    g_warm = s.warmCount();
    g_cold = s.coldCount();
  }

  std::printf("checks=%ld steps=%ld  (last solver warm=%ld cold=%ld) fails=%d\n",
              g_checks, g_steps, g_warm, g_cold, g_fail);
  // Coverage: the warm (non-cold) fast path must actually be exercised.  The
  // adversarial mutator above re-randomizes everything every step, so the
  // retained basis is always broken and `Simple` correctly cold-falls-back
  // (the array-side Simple mode is identical here — repairs are the job of
  // Dual/Primal, not Simple).  To exercise + validate the warm fast path we
  // use (a) identical re-solves and (b) a GENTLE mutator: caps fixed with
  // generous slack, supply nudged by tiny witness deltas, so the retained
  // optimal basis usually stays feasible.
  {
    std::mt19937_64 rng(0x5EED1234u);
    auto uni = [&](int lo, int hi) {
      return std::uniform_int_distribution<int>(lo, hi)(rng);
    };
    const int n = 10, m = 28;
    Instance in;
    in.n = n;
    for (int e = 0; e < m; ++e) {
      int u = uni(0, n - 1), v = uni(0, n - 1);
      if (u == v) v = (v + 1) % n;
      in.arcs.push_back({u, v});
    }
    in.cost.resize(m);
    for (int i = 0; i < m; ++i) in.cost[i] = uni(0, 50);
    {  // LEMON StaticDigraph::build needs arcs sorted by (src,tgt).
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
    // Base witness; caps = witness + generous slack, FIXED for the chain.
    std::vector<Value> w0(m);
    for (int i = 0; i < m; ++i) w0[i] = uni(2, 10);
    in.cap.assign(m, 0);
    for (int i = 0; i < m; ++i) in.cap[i] = w0[i] + 40;  // never tightened
    auto setSupplyFromWitness = [&](const std::vector<Value>& w) {
      in.supply.assign(n, 0);
      for (int i = 0; i < m; ++i) {
        in.supply[in.arcs[i].first] += w[i];
        in.supply[in.arcs[i].second] -= w[i];
      }
    };
    setSupplyFromWitness(w0);

    LCTNS s(in.n);
    std::vector<int> ids;
    for (int i = 0; i < m; ++i)
      ids.push_back(
          s.addArc(in.arcs[i].first, in.arcs[i].second, in.cost[i], in.cap[i]));
    for (int v = 0; v < n; ++v) s.setSupply(v, in.supply[v]);
    s.run();
    verify("cov-prime", s, in, ids, true);

    int warm_hits = 0, trials = 0;
    for (int k = 0; k < 250; ++k) {
      std::vector<Value> w = w0;
      if (k % 5 != 0) {  // every 5th step is an identical re-solve
        for (int i = 0; i < m; ++i) w[i] = std::max<Value>(0, w0[i] + uni(-2, 2));
      }
      setSupplyFromWitness(w);
      for (int v = 0; v < n; ++v) s.setSupply(v, in.supply[v]);
      int wc0 = s.warmCount();
      auto wst = s.warmRun();
      ++trials;
      if (s.warmCount() > wc0) ++warm_hits;
      verify("cov", s, in, ids, wst == LCTNS::OPTIMAL);
    }
    std::printf("gentle chain: warm_hits=%d / %d (warm=%d cold=%d)\n",
                warm_hits, trials, s.warmCount(), s.coldCount());
    if (warm_hits < 20) {
      std::printf("INEFFECTIVE: warm fast path fired only %d times (<20)\n",
                  warm_hits);
      ++g_fail;
    }
  }

  if (g_fail) {
    std::printf("RESULT: FAILED (%d)\n", g_fail);
    return 1;
  }
  std::printf("RESULT: PASSED\n");
  return 0;
}
