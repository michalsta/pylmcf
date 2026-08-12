// chain_solver_1d.h
// -------------------------------------------------------------------------
// Specialised solver for the wnet 1D-chain min-cost flow, SimpleTrash mode.
// Verified LP (decompositable_graph.hpp):
//   Source(+F) Sink(-F), F=max(E,T), E=Σemp, T=Σtheo
//   Source->Emp_k cost0 cap e_k ; Theo_k->Sink cost0 cap t_k
//   ChainEdge consecutive positions, BOTH directions, cost=|Δpos|, cap ∞
//   Source->Sink (trash) cost κ, cap ∞
//
// Method: successive-shortest-path min-cost flow specialised to this graph
// (positions form a path; spurs to S/K; one κ bypass).  SSP is the textbook
// MCF algorithm — correctness is *mechanical* (push the bottleneck along the
// cheapest residual S→K path, maintaining node potentials so reduced costs
// stay ≥ 0, until F units are routed); it does NOT rely on a fragile global
// invariant the way the primal slope-trick / lazy-rematch did, so it can be
// validated bit-exact against LEMON.  Every non-trash augmentation saturates
// a spur ⇒ O(K) augmentations; Dijkstra+potentials ⇒ O(K² log K) (fine, and
// far better than network-simplex's pathological p95 on long chains;
// constant-factor tuning is a later step once the oracle is green).
//
// Correctness pinned by tests_cpp/test_chain_solver_1d.cpp vs LEMON.
// -------------------------------------------------------------------------
#ifndef PYLMCF_CHAIN_SOLVER_1D_H
#define PYLMCF_CHAIN_SOLVER_1D_H

#include <limits>
#include <queue>
#include <vector>

namespace pylmcf {

template <typename Value = long long, typename Cost = long long>
struct ChainSolver1D {
  struct Point { Cost pos; Value emp; Value theo; };

  // Per-arc optimal flows (for wnet's gradient, which reads flow per arc).
  struct Flows {
    Cost total = 0;
    std::vector<Value> emp_in;     // Source->Emp_k   flow, size n
    std::vector<Value> theo_out;   // Theo_k->Sink    flow, size n
    std::vector<Value> gap;        // signed rightward chain flow, size n-1
    Value trash = 0;               // Source->Sink    flow
  };

  static Cost solve(const std::vector<Point>& pts, Cost kappa) {
    Flows f;
    return run(pts, kappa, &f);
  }
  static Flows solveFull(const std::vector<Point>& pts, Cost kappa) {
    Flows f;
    run(pts, kappa, &f);
    return f;
  }

 private:
  static Cost run(const std::vector<Point>& pts, Cost kappa, Flows* out) {
    const int n = (int)pts.size();
    if (n == 0) { if (out) *out = Flows{}; return 0; }
    const int N = n + 2;                 // 0=Source, 1=Sink, 2+k = position k
    const int S = 0, K = 1;
    const Value VINF = std::numeric_limits<Value>::max() / 4;
    const Cost CINF = std::numeric_limits<Cost>::max() / 4;

    // Residual graph (paired edges e, e^1).
    std::vector<int> to, nxt, head(N, -1);
    std::vector<Value> cap;
    std::vector<Cost> cst;
    auto add = [&](int u, int v, Value c, Cost w) -> int {
      const int e = (int)to.size();
      to.push_back(v); cap.push_back(c); cst.push_back(w);
      nxt.push_back(head[u]); head[u] = e;
      to.push_back(u); cap.push_back(0); cst.push_back(-w);
      nxt.push_back(head[v]); head[v] = e + 1;
      return e;                                     // forward edge index
    };

    std::vector<int> empE(n, -1), theoE(n, -1), gapR(n, -1), gapL(n, -1);
    int trashE = -1;
    Value E = 0, T = 0;
    for (int k = 0; k < n; ++k) {
      if (pts[k].emp > 0)  { empE[k]  = add(S, 2 + k, pts[k].emp, 0);  E += pts[k].emp; }
      if (pts[k].theo > 0) { theoE[k] = add(2 + k, K, pts[k].theo, 0); T += pts[k].theo; }
    }
    for (int k = 0; k + 1 < n; ++k) {
      const Cost g = pts[k + 1].pos - pts[k].pos;   // = |Δpos|, sorted
      gapR[k] = add(2 + k, 2 + k + 1, VINF, g);      // rightward k→k+1
      gapL[k] = add(2 + k + 1, 2 + k, VINF, g);      // leftward  k+1→k
    }
    trashE = add(S, K, VINF, kappa);                 // trash
    const Value F = (E > T) ? E : T;

    // SSP with Johnson potentials.  All original costs ≥ 0 and backward
    // residuals start empty ⇒ π = 0 is valid for the first Dijkstra; the
    // standard π += dist update keeps reduced costs ≥ 0 thereafter.
    std::vector<Cost> pot(N, 0), dist(N);
    std::vector<int> pe(N);                          // incoming edge in SP tree
    Value pushed = 0;
    Cost total = 0;
    while (pushed < F) {
      std::fill(dist.begin(), dist.end(), CINF);
      dist[S] = 0;
      using QI = std::pair<Cost, int>;
      std::priority_queue<QI, std::vector<QI>, std::greater<QI>> pq;
      pq.push({0, S});
      while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d != dist[u]) continue;
        for (int e = head[u]; e != -1; e = nxt[e]) {
          if (cap[e] <= 0) continue;
          const int v = to[e];
          const Cost rc = cst[e] + pot[u] - pot[v];   // ≥ 0 by invariant
          if (dist[u] + rc < dist[v]) {
            dist[v] = dist[u] + rc;
            pe[v] = e;
            pq.push({dist[v], v});
          }
        }
      }
      if (dist[K] >= CINF) break;                    // unreachable (shouldn't)
      for (int v = 0; v < N; ++v)
        if (dist[v] < CINF) pot[v] += dist[v];

      // Bottleneck along the S→K shortest path, capped by remaining demand.
      Value aug = F - pushed;
      for (int v = K; v != S; v = to[pe[v] ^ 1])
        if (cap[pe[v]] < aug) aug = cap[pe[v]];
      for (int v = K; v != S; v = to[pe[v] ^ 1]) {
        total += (Cost)aug * cst[pe[v]];             // original-cost sum
        cap[pe[v]]     -= aug;
        cap[pe[v] ^ 1] += aug;
      }
      pushed += aug;
    }

    if (out) {
      // Flow on a forward edge e = its reverse residual cap[e^1].
      out->total = total;
      out->emp_in.assign(n, 0);
      out->theo_out.assign(n, 0);
      out->gap.assign(n > 0 ? n - 1 : 0, 0);
      for (int k = 0; k < n; ++k) {
        if (empE[k]  >= 0) out->emp_in[k]   = cap[empE[k]  ^ 1];
        if (theoE[k] >= 0) out->theo_out[k] = cap[theoE[k] ^ 1];
      }
      for (int k = 0; k + 1 < n; ++k)               // net signed rightward
        out->gap[k] = cap[gapR[k] ^ 1] - cap[gapL[k] ^ 1];
      out->trash = cap[trashE ^ 1];
    }
    return total;
  }
};

}  // namespace pylmcf

#endif  // PYLMCF_CHAIN_SOLVER_1D_H
