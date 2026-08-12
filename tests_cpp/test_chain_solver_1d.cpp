// test_chain_solver_1d.cpp
// -------------------------------------------------------------------------
// Oracle for ChainSolver1D: build the EXACT wnet SimpleTrash chain LP in
// LEMON, solve both, assert identical optimal total cost.
//
//   nodes : Source(+F) Sink(-F)  F=max(E,T)  + one node per sorted position
//   arcs  : Source->Emp_i  cost0 cap e_i
//           Theo_j->Sink   cost0 cap t_j
//           pos g <-> pos g+1   cost=|Δ| cap INF   (both directions)
//           Source->Sink (trash) cost κ cap INF
//
// Build:
//   g++ -I$(python -m pylmcf --include) -std=c++20 -O2 \
//       tests_cpp/test_chain_solver_1d.cpp -o /tmp/tc && /tmp/tc
// -------------------------------------------------------------------------
#define LEMON_ONLY_TEMPLATES
#include <lemon/static_graph.h>
#include <lemon/network_simplex.h>
#include <pylmcf/chain_solver_1d.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using V = int64_t;
using Graph = lemon::StaticDigraph;
using NS = lemon::NetworkSimplex<Graph, V, V>;
using CS = pylmcf::ChainSolver1D<V, V>;

static int g_fail = 0;
static long g_checks = 0;

#define CHECK(c, m)                                                  \
  do {                                                               \
    ++g_checks;                                                      \
    if (!(c)) { std::printf("  FAIL %s\n", m); ++g_fail; }           \
  } while (0)

// LEMON optimum for the exact chain LP built from `pts` (sorted) + κ.
static bool lemon_solve(const std::vector<CS::Point>& pts, V kappa, V& out) {
  const int K = (int)pts.size();
  // node 0=Source 1=Sink, 2..2+K-1 = position nodes
  const int N = 2 + K;
  std::vector<std::pair<int, int>> arcs;
  std::vector<V> cost, cap;
  V E = 0, T = 0;
  for (int k = 0; k < K; ++k) {
    int nd = 2 + k;
    if (pts[k].emp > 0) {                       // Source -> Emp_k  cap e
      arcs.push_back({0, nd}); cost.push_back(0); cap.push_back(pts[k].emp);
      E += pts[k].emp;
    }
    if (pts[k].theo > 0) {                       // Theo_k -> Sink   cap t
      arcs.push_back({nd, 1}); cost.push_back(0); cap.push_back(pts[k].theo);
      T += pts[k].theo;
    }
  }
  const V INF = (V)4e18;
  for (int k = 0; k + 1 < K; ++k) {              // bidirectional chain
    V g = pts[k + 1].pos - pts[k].pos;
    arcs.push_back({2 + k, 2 + k + 1}); cost.push_back(g); cap.push_back(INF);
    arcs.push_back({2 + k + 1, 2 + k}); cost.push_back(g); cap.push_back(INF);
  }
  arcs.push_back({0, 1}); cost.push_back(kappa); cap.push_back(INF);  // trash
  const V F = std::max(E, T);

  // StaticDigraph::build needs arcs sorted by (src,tgt); permute cost/cap.
  const int M = (int)arcs.size();
  std::vector<int> p(M);
  for (int i = 0; i < M; ++i) p[i] = i;
  std::stable_sort(p.begin(), p.end(),
                   [&](int a, int b) { return arcs[a] < arcs[b]; });
  std::vector<std::pair<int, int>> A(M);
  std::vector<V> C(M), U(M);
  for (int i = 0; i < M; ++i) { A[i]=arcs[p[i]]; C[i]=cost[p[i]]; U[i]=cap[p[i]]; }

  Graph g;
  g.build(N, A.begin(), A.end());
  Graph::ArcMap<V> cm(g), um(g);
  Graph::NodeMap<V> sm(g);
  for (int i = 0; i < M; ++i) { cm[g.arcFromId(i)] = C[i]; um[g.arcFromId(i)] = U[i]; }
  for (int v = 0; v < N; ++v) sm[g.nodeFromId(v)] = 0;
  sm[g.nodeFromId(0)] = F;
  sm[g.nodeFromId(1)] = -F;
  NS ns(g);
  ns.upperMap(um).costMap(cm).supplyMap(sm);
  auto st = ns.run();
  if (st != NS::OPTIMAL) return false;
  out = ns.totalCost();
  return true;
}

int main() {
  std::printf("ChainSolver1D vs LEMON oracle (SimpleTrash chain LP)\n");
  std::mt19937_64 rng(0xC0FFEEu);
  auto uni = [&](int lo, int hi) {
    return std::uniform_int_distribution<int>(lo, hi)(rng);
  };
  for (int t = 0; t < 20000; ++t) {
    int K = uni(2, 30);
    std::vector<CS::Point> pts(K);
    V pos = 0;
    for (int k = 0; k < K; ++k) {
      pos += uni(1, 10);                          // strictly increasing
      pts[k].pos = pos;
      pts[k].emp = 0; pts[k].theo = 0;
      if (uni(0, 1)) pts[k].emp = uni(0, 12);
      else           pts[k].theo = uni(0, 12);
    }
    V kappa = uni(1, 40);
    V lo;
    bool lopt = lemon_solve(pts, kappa, lo);
    CHECK(lopt, "LEMON not OPTIMAL on feasible chain LP");
    if (!lopt) continue;
    V co = CS::solve(pts, kappa);
    if (co != lo && g_fail < 4) {
      std::printf("REPRO K=%d kappa=%lld  cs=%lld lemon=%lld\n", K,
                  (long long)kappa, (long long)co, (long long)lo);
      std::printf(" pts:");
      for (auto& q : pts)
        std::printf(" (p=%lld e=%lld t=%lld)", (long long)q.pos,
                    (long long)q.emp, (long long)q.theo);
      std::printf("\n");
    }
    CHECK(co == lo, "ChainSolver1D cost != LEMON optimum");

    // Validate per-arc flows: feasible + conserving + recompute the cost.
    auto fl = CS::solveFull(pts, kappa);
    CHECK(fl.total == co, "solveFull.total != solve()");
    V Msum = 0, sumE = 0, sumT = 0;
    bool feas = fl.trash >= 0;
    for (int k = 0; k < K; ++k) {
      if (fl.emp_in[k] < 0 || fl.emp_in[k] > pts[k].emp) feas = false;
      if (fl.theo_out[k] < 0 || fl.theo_out[k] > pts[k].theo) feas = false;
      sumE += fl.emp_in[k]; sumT += fl.theo_out[k];
    }
    CHECK(feas, "chain flow out of [0,cap]");
    CHECK(sumE == sumT, "matched emp != matched theo");
    Msum = sumE;
    // Conservation: Φ_k = Σ_{j≤k}(emp_in−theo_out); gap[k]==Φ_k; Φ_{K-1}=0.
    V phi = 0; bool cons = true;
    for (int k = 0; k < K; ++k) {
      phi += fl.emp_in[k] - fl.theo_out[k];
      if (k + 1 < K) { if (fl.gap[k] != phi) cons = false; }
      else           { if (phi != 0) cons = false; }
    }
    CHECK(cons, "chain conservation violated");
    V Etot = 0, Ttot = 0;
    for (int k = 0; k < K; ++k) { Etot += pts[k].emp; Ttot += pts[k].theo; }
    const V Ftot = std::max(Etot, Ttot);
    CHECK(sumE + fl.trash == Ftot, "source conservation (ΣmE + trash != F)");
    // Recompute cost from flows.
    V rc = (V)fl.trash * kappa;
    for (int k = 0; k + 1 < K; ++k) {
      V g = pts[k + 1].pos - pts[k].pos;
      V f = fl.gap[k] < 0 ? -fl.gap[k] : fl.gap[k];
      rc += f * g;
    }
    CHECK(rc == co, "Σ|gap|·dist + κ·trash != cost");
  }
  std::printf("checks=%ld fails=%d\n", g_checks, g_fail);
  if (g_fail) { std::printf("RESULT: FAILED (%d)\n", g_fail); return 1; }
  std::printf("RESULT: PASSED\n");
  return 0;
}
