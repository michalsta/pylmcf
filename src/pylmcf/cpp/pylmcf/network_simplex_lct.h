// network_simplex_lct.h
// -------------------------------------------------------------------------
// Phase B + C: a primal network simplex whose spanning-tree basis is
// maintained by a link-cut tree (pylmcf::LinkCutTree) instead of LEMON's
// O(subtree) thread/succ_num arrays.
//
// What the LCT replaces vs. the array implementation:
//   * node potentials  — derived as a root..u path-sum (sumToRoot), so the
//     per-pivot O(subtree) `updatePotential` shift is implicit/free.
//   * lowest common ancestor (join node) — O(log n) lca().
//   * the structural pivot — cutParent + link, O(log n), replacing the
//     O(subtree) thread re-splice (`updateTreeStructure`).
// The unavoidable O(cycle) work (ratio test + flow change + stem reversal of
// par/pred arrays) is kept as plain arrays, exactly as the array solver does.
//
// Scope: EQ supply (sum == 0), zero lower bounds, finite real-arc caps — the
// regime the exhaustive instance generator (and the wnet workload) produces.
// Big-M artificial arcs give the initial feasible star basis (same structure
// LEMON uses internally).
//
//   Phase B  cold solve            — run()
//   Phase C  warm restart          — warmRun()  (Simple = repair-or-cold:
//            keep the retained basis, recompute tree-arc flows for the new
//            caps/supplies; if still in bounds the basis is optimal — costs
//            are unchanged so it is still dual-feasible — otherwise fall back
//            to a cold solve.  Mirrors LEMON `WarmMode.Simple`.)
//
// Anti-cycling: LEMON's exact leaving rule (strict `<` on the first cycle
// path, `<=` on the second) keeps the spanning tree STRONGLY FEASIBLE
// (Cunningham) given the strongly-feasible artificial-star start.  A
// smallest-arc-id (Bland) tie-break does NOT preserve that invariant and was
// observed to cycle on degenerate (delta == 0) pivots — do not substitute it.
//
// Correctness is pinned by tests_cpp/test_network_simplex_lct.cpp (cold, vs
// LEMON array oracle) and test_network_simplex_lct_warm.cpp (warm chains,
// each step vs an independent cold LEMON solve).
// -------------------------------------------------------------------------
#ifndef PYLMCF_NETWORK_SIMPLEX_LCT_H
#define PYLMCF_NETWORK_SIMPLEX_LCT_H

#include <pylmcf/link_cut_tree.h>

#include <limits>
#include <memory>
#include <vector>

#ifdef PYLMCF_NSLCT_DEBUG
#include <cstdio>
#endif

namespace pylmcf {

template <typename Value = long long, typename Cost = long long>
class NetworkSimplexLCT {
 public:
  enum Status { OPTIMAL, INFEASIBLE };

  explicit NetworkSimplexLCT(int n) : _n(n), _supply(n, Value(0)) {}

  // Add a real arc u->v with the given cost and capacity (lower bound 0).
  // Returns its arc id (0-based, in insertion order).  Call before the first
  // run()/warmRun() (topology and costs are fixed across warm restarts).
  int addArc(int u, int v, Cost cost, Value cap) {
    _src.push_back(u);
    _tgt.push_back(v);
    _cost.push_back(cost);
    _cap.push_back(cap);
    return (int)_src.size() - 1;
  }

  void setSupply(int node, Value s) { _supply[node] = s; }

  // Change a real arc's capacity (allowed between warm restarts; costs and
  // topology must not change).
  void setCap(int arc_id, Value cap) { _cap[arc_id] = cap; }

  // Cold solve from the artificial-star basis (Phase B).
  Status run() {
    ensureBuilt();
    coldStart();
    pivotLoop();
    _haveBasis = true;
    return finishObjective();
  }

  // Warm restart (Phase C, Simple strategy): reuse the retained basis if the
  // recomputed tree-arc flows stay in bounds, else cold-fall-back.
  Status warmRun() {
    ensureBuilt();
    if (!_haveBasis) {
      coldStart();
      pivotLoop();
      _haveBasis = true;
      ++_cold_count;
      return finishObjective();
    }
    if (repairTreeFlows()) {
#ifdef PYLMCF_NSLCT_DEBUG
      dbgConservation("after repairTreeFlows");
#endif
      // Costs unchanged ⇒ retained basis still dual-feasible; repair restored
      // primal feasibility ⇒ optimal.  Run the pivot loop as cheap insurance
      // (it does ~0 work from an already-optimal feasible basis).
      pivotLoop();
#ifdef PYLMCF_NSLCT_DEBUG
      dbgConservation("after warm pivotLoop");
#endif
      ++_warm_count;
      return finishObjective();
    }
    coldStart();
    pivotLoop();
    ++_cold_count;
    return finishObjective();
  }

  Cost totalCost() const { return _total; }
  Value flow(int arc_id) const { return _flow[arc_id]; }
  int warmCount() const { return _warm_count; }
  int coldCount() const { return _cold_count; }

  // Node potential after the last solve.  pi[u] satisfies the SAME reduced-
  // cost relation as LEMON's `_pi` (rc = cost + pi[src] - pi[tgt], == 0 on
  // tree arcs), so DIFFERENCES pi[u]-pi[v] equal LEMON's exactly (the two
  // vectors differ only by a global offset on a connected graph — which the
  // artificial root guarantees).  Cached post-solve so this is O(1)/const.
  Cost potential(int node) const { return _piCache[node]; }

 private:
  enum { ST_UPPER = -1, ST_TREE = 0, ST_LOWER = 1 };

  // Append the per-node artificial arcs and size the basis arrays.  Done once;
  // topology/costs are fixed across warm restarts.
  void ensureBuilt() {
    if (_built) return;
    _m = (int)_src.size();
    _R = _n;
    _N = _n + 1;
    _INF = std::numeric_limits<Value>::max() / 4;
    Cost mx = 1;
    for (int e = 0; e < _m; ++e)
      if (_cost[e] > mx) mx = _cost[e];
    _BIGM = mx * Cost(_n + 2) * Cost(_m + 2) + 1;

    _src.resize(_m + _n);
    _tgt.resize(_m + _n);
    _cost.resize(_m + _n);
    _cap.resize(_m + _n);
    _flow.assign(_m + _n, Value(0));
    _state.assign(_m + _n, ST_LOWER);
    _par.assign(_N, -1);
    _predArc.assign(_N, -1);
    _predDir.assign(_N, 0);
    for (int u = 0; u < _n; ++u) {     // artificial arc ids/costs are stable
      _cost[_m + u] = _BIGM;
      _cap[_m + u] = _INF;
    }
    _built = true;
  }

  // (Re)establish the strongly-feasible artificial-star basis for the current
  // supply, with a fresh link-cut tree.
  void coldStart() {
    const int m = _m, n = _n, R = _R, N = _N;
    for (int e = 0; e < m; ++e) {
      _flow[e] = 0;
      _state[e] = ST_LOWER;
    }
    _lct = std::make_unique<LinkCutTree<Cost>>(N);
    for (int u = 0; u < n; ++u) {
      const int a = m + u;
      _cost[a] = _BIGM;
      _cap[a] = _INF;
      _state[a] = ST_TREE;
      if (_supply[u] >= 0) {            // u -> R, carries supply out
        _src[a] = u;
        _tgt[a] = R;
        _flow[a] = _supply[u];
        _predDir[u] = +1;              // arc u->R == child->parent
      } else {                         // R -> u, carries demand in
        _src[a] = R;
        _tgt[a] = u;
        _flow[a] = -_supply[u];
        _predDir[u] = -1;              // arc R->u == parent->child
      }
      _par[u] = R;
      _predArc[u] = a;
      _lct->link(u, R);                // u is a fresh singleton -> R stays root
      _lct->setVal(u, edgeVal(u));
    }
    _par[R] = -1;
    _predArc[R] = -1;
    _lct->setVal(R, Cost(0));
  }

  // LEMON's repairTreeFlows analog: pin non-tree real arcs to their bound,
  // recompute every tree-arc flow from the current supply via a post-order
  // over the retained tree, and report whether all tree arcs stay in bounds.
  bool repairTreeFlows() {
    const int m = _m, n = _n, R = _R, N = _N;
    for (int e = 0; e < m; ++e) {
      if (_state[e] == ST_LOWER) _flow[e] = 0;
      else if (_state[e] == ST_UPPER) _flow[e] = _cap[e];
    }
    if ((int)_run.size() < N) _run.resize(N);
    for (int u = 0; u < n; ++u) _run[u] = _supply[u];
    _run[R] = Value(0);
    for (int e = 0; e < m; ++e) {
      if (_state[e] != ST_TREE) {
        _run[_src[e]] -= _flow[e];
        _run[_tgt[e]] += _flow[e];
      }
    }
    // Post-order over the retained tree (children before parent).  Children
    // are inverted from _par each call (O(N); a warm restart already costs
    // O(N) to push new caps/supplies).
    _head.assign(N, -1);
    if ((int)_nxt.size() < N) _nxt.resize(N);
    for (int u = 0; u < n; ++u) {       // R == n has no parent
      const int p = _par[u];
      _nxt[u] = _head[p];
      _head[p] = u;
    }
    _order.clear();
    _stk.clear();
    _stk.push_back(R);
    while (!_stk.empty()) {
      int u = _stk.back();
      _stk.pop_back();
      _order.push_back(u);
      for (int c = _head[u]; c != -1; c = _nxt[c]) _stk.push_back(c);
    }
    for (int idx = (int)_order.size() - 1; idx >= 0; --idx) {
      const int u = _order[idx];
      if (u == R) continue;
      const int e = _predArc[u];
      _flow[e] = Value(_predDir[u]) * _run[u];
      _run[_par[u]] += _run[u];
    }
    for (int e = 0; e < m + n; ++e) {
      if (_state[e] != ST_TREE) continue;
      if (e < m) {
        if (_flow[e] < 0 || _flow[e] > _cap[e]) return false;
      } else if (_flow[e] < 0) {       // artificial: cap == INF
        return false;
      }
    }
    return true;
  }

  void pivotLoop() {
    const int m = _m, n = _n;
    LinkCutTree<Cost>& lct = *_lct;
    auto pi = [&](int x) -> Cost { return lct.sumToRoot(x); };
    auto reduced = [&](int a) -> Cost {
      return _cost[a] + pi(_src[a]) - pi(_tgt[a]);
    };

    const long long iter_cap = 1LL * (m + n + 4) * (m + n + 4) + 1000;
    for (long long iter = 0; iter < iter_cap; ++iter) {
      int a = -1;                       // entering: smallest-id eligible arc
      for (int e = 0; e < m + n; ++e) {
        if (_state[e] == ST_TREE) continue;
        const Cost rc = reduced(e);
        if ((_state[e] == ST_LOWER && rc < 0) ||
            (_state[e] == ST_UPPER && rc > 0)) {
          a = e;
          break;
        }
      }
      if (a < 0) break;                 // optimal

      const int i = _src[a], j = _tgt[a];
      int first, second;
      if (_state[a] == ST_LOWER) {
        first = i;
        second = j;
      } else {
        first = j;
        second = i;
      }
      const int join = lct.lca(i, j);

      // Ratio test — LEMON's strongly-feasible leaving rule (`<` first path,
      // `<=` second).  See header note; do not replace with Bland.
      Value delta = _cap[a];
      int leave_arc = -1, u_out = -1, out_path = 0;
      auto consider = [&](int node, int path, bool strict) {
        const int e = _predArc[node];
        Value d;
        if ((path == 1 && _predDir[node] == -1) ||
            (path == 2 && _predDir[node] == +1)) {
          d = _cap[e] - _flow[e];
        } else {
          d = _flow[e];
        }
        if (strict ? (d < delta) : (d <= delta)) {
          delta = d;
          leave_arc = e;
          u_out = node;
          out_path = path;
        }
      };
      for (int u = first; u != join; u = _par[u]) consider(u, 1, true);
      for (int u = second; u != join; u = _par[u]) consider(u, 2, false);

      const Value sign = (_state[a] == ST_LOWER) ? +1 : -1;
      const Value val = sign * delta;
      _flow[a] += val;
      for (int u = i; u != join; u = _par[u])
        _flow[_predArc[u]] -= Value(_predDir[u]) * val;
      for (int u = j; u != join; u = _par[u])
        _flow[_predArc[u]] += Value(_predDir[u]) * val;

      if (out_path == 0) {              // entering arc bound flip, no pivot
        _state[a] = (_state[a] == ST_LOWER) ? ST_UPPER : ST_LOWER;
        continue;
      }

      const int u_in = (out_path == 1) ? first : second;
      const int v_in = (out_path == 1) ? second : first;
      _state[a] = ST_TREE;
      _state[leave_arc] = (_flow[leave_arc] == 0) ? ST_LOWER : ST_UPPER;

      lct.cutParent(u_out);             // sever leaving edge, root stays fixed

      _stem.clear();
      for (int s = u_in;; s = _par[s]) {
        _stem.push_back(s);
        if (s == u_out) break;
      }
      const int k = (int)_stem.size() - 1;
      _oldArc.assign(k, 0);
      _oldDir.assign(k, 0);
      for (int t = 0; t < k; ++t) {
        _oldArc[t] = _predArc[_stem[t]];
        _oldDir[t] = _predDir[_stem[t]];
      }
      for (int t = 0; t < k; ++t) {
        const int child = _stem[t + 1];
        _par[child] = _stem[t];
        _predArc[child] = _oldArc[t];
        _predDir[child] = -_oldDir[t];
      }
      _par[u_in] = v_in;
      _predArc[u_in] = a;
      _predDir[u_in] = (u_in == _src[a]) ? +1 : -1;

      for (int s : _stem) lct.setVal(s, edgeVal(s));
      lct.link(u_in, v_in);
    }
  }

  Status finishObjective() {
    if ((int)_piCache.size() < _n) _piCache.resize(_n);
    for (int u = 0; u < _n; ++u) _piCache[u] = _lct->sumToRoot(u);
    for (int u = 0; u < _n; ++u)
      if (_flow[_m + u] != 0) return INFEASIBLE;
    _total = 0;
    for (int e = 0; e < _m; ++e) _total += Cost(_flow[e]) * _cost[e];
    return OPTIMAL;
  }

#ifdef PYLMCF_NSLCT_DEBUG
  void dbgConservation(const char* where) {
    std::vector<Value> bal(_n + 1, 0);
    for (int e = 0; e < _m + _n; ++e) {
      bal[_src[e]] -= _flow[e];
      bal[_tgt[e]] += _flow[e];
    }
    for (int u = 0; u < _n; ++u) {
      if (bal[u] != -_supply[u]) {
        std::fprintf(stderr, "  !! conservation broken %s node=%d bal=%lld supply=%lld\n",
                     where, u, (long long)bal[u], (long long)_supply[u]);
      }
    }
    for (int u = 0; u < _n; ++u) {
      const int e = _predArc[u];
      if (_state[e] != ST_TREE)
        std::fprintf(stderr, "  !! predArc not TREE %s node=%d e=%d state=%d\n",
                     where, u, e, (int)_state[e]);
    }
  }
#endif

  // LCT value of node u so that pi[u] = pi[par] - predDir[u]*cost(pred); the
  // root..u path-sum then equals the LEMON potential (up to a global offset
  // that does not affect reduced costs or the optimal flow).
  Cost edgeVal(int u) const {
    if (_predArc[u] < 0) return Cost(0);
    return Cost(-_predDir[u]) * _cost[_predArc[u]];
  }

  int _n, _m = 0, _R = 0, _N = 0;
  bool _built = false, _haveBasis = false;
  Value _INF = 0;
  Cost _BIGM = 0;
  std::vector<int> _src, _tgt;
  std::vector<Cost> _cost;
  std::vector<Value> _cap, _flow, _supply;
  std::vector<signed char> _state;
  std::vector<int> _par, _predArc, _predDir;
  std::vector<int> _stem, _oldArc, _oldDir;
  std::vector<Value> _run;
  std::vector<int> _head, _nxt, _order, _stk;
  std::unique_ptr<LinkCutTree<Cost>> _lct;
  std::vector<Cost> _piCache;
  Cost _total = 0;
  int _warm_count = 0, _cold_count = 0;
};

}  // namespace pylmcf

#endif  // PYLMCF_NETWORK_SIMPLEX_LCT_H
