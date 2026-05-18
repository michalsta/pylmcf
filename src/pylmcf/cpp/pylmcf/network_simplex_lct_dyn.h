// network_simplex_lct_dyn.h
// -------------------------------------------------------------------------
// EXPERIMENTAL variant: the "real" dynamic-trees network simplex.  Unlike
// NetworkSimplexLCT (Phase B/C — explicit O(cycle) `_par` walks for the ratio
// test and flow augmentation), this pushes flow INTO the link-cut tree so:
//
//   findLeavingArc  -> two O(log K) path-min queries
//   changeFlow      -> two O(log K) lazy path range-adds
//
// reducing per-pivot O(K) (on a chain) to O(log K) for those two steps — the
// only lever that can flip the chain verdict (see project_wnet_lct memory).
//
// Key design — the ROOTWARD-FLOW (r-frame).  Define, per non-root node u,
//   r[u] = predDir[u] * flow[predArc[u]]      (flow positive u->parent)
// Then augmenting the cycle by `val` (entering arc e=(i,j), join=lca):
//   r -= val  on path i->join (excl join);  r += val  on path j->join (excl).
// The per-edge predDir cancels (predDir^2 = 1) -> two UNIFORM path adds.
// The ratio-test residual is affine in r with fixed ±1 slope + per-node const:
//   i-side residual  vi[u] = +r[u] + off_i[u],  off_i = predDir==+1 ? 0 : cap
//   j-side residual  vj[u] = -r[u] + off_j[u],  off_j = predDir==+1 ? cap : 0
// so a `r += c` path add is `vi += c, vj -= c` and the ratio test is a path-
// min of vi (i-side) / vj (j-side).  No evert/reversal hazard: the main tree
// is never everted (fixed artificial root; predDir is an explicit local
// sign), so r/vi/vj are local edge properties — only stem nodes recompute on
// a pivot, exactly like the cost channel.  Anti-cycling: same LEMON
// strongly-feasible leaving rule (strict `<` first side, `<=` second),
// reproduced via per-side tie-broken path-min argmin.
//
// Correctness pinned by tests_cpp/test_network_simplex_lct_dyn.cpp vs LEMON.
// -------------------------------------------------------------------------
#ifndef PYLMCF_NETWORK_SIMPLEX_LCT_DYN_H
#define PYLMCF_NETWORK_SIMPLEX_LCT_DYN_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#ifdef PYLMCF_DYN_DEBUG
#include <cstdio>
#include <cstdlib>
#endif

namespace pylmcf {

template <typename Value = long long, typename Cost = long long>
class NetworkSimplexLCTDyn {
 public:
  enum Status { OPTIMAL, INFEASIBLE };

  explicit NetworkSimplexLCTDyn(int n) : _n(n), _supply(n, Value(0)) {}

  int addArc(int u, int v, Cost cost, Value cap) {
    _src.push_back(u);
    _tgt.push_back(v);
    _cost.push_back(cost);
    _cap.push_back(cap);
    return (int)_src.size() - 1;
  }
  void setSupply(int node, Value s) { _supply[node] = s; }
  void setCap(int arc_id, Value cap) { _cap[arc_id] = cap; }

  Status run() {
    ensureBuilt();
    coldStart();
    pivotLoop();
    Status st = finish();
    _haveBasis = (st == OPTIMAL);
    snapshot();
    ++_coldCnt;
    return st;
  }

  // Lever 1: incremental warm restart (Simple fast path only).  Costs are
  // fixed across warm solves.  A supply change Δs[v] re-routes that imbalance
  // along the tree path v→R: in the r-frame that is a single O(log K) lazy
  // path range-add per changed node (NOT an O(N) repairTreeFlows recompute).
  // Feasibility is checked for FREE on just the perturbed segments: the prior
  // solve left every tree arc feasible (vi,vj ≥ 0), so only nodes on a
  // perturbation path can violate, and the segment's mi/mj aggregate (which
  // the range-add already maintains) is exactly that check — no O(N) scan.
  // If any perturbed segment goes negative, or any capacity changed (handled
  // conservatively in this first cut), fall back to a cold solve.
  Status warmRun() {
    ensureBuilt();
    if (!_haveBasis) {
      coldStart();
      pivotLoop();
      Status st = finish();
      _haveBasis = (st == OPTIMAL);
      snapshot();
      ++_coldCnt;
      return st;
    }
    bool cap_changed = false;
    for (int e = 0; e < _m && !cap_changed; ++e)
      if (_cap[e] != _prevCap[e]) cap_changed = true;
    if (cap_changed) {                              // conservative: cold
      coldStart();
      pivotLoop();
      Status st = finish();
      _haveBasis = (st == OPTIMAL);
      snapshot();
      ++_coldCnt;
      return st;
    }
    // Supply-only: apply each Δs[v] as one O(log K) lazy path add v→R.
    _changed.clear();
    for (int v = 0; v < _n; ++v) {
      const Value d = _supply[v] - _prevSupply[v];
      if (d == 0) continue;
      _changed.push_back(v);
      const int seg = _lct.segBelow(v, _R);         // path R-excl .. v
      _lct.applyAddI(seg, +d);                       // r += d ⇒ vi += d
      _lct.applyAddJ(seg, -d);                       //          vj -= d
      _lct.segPull();
    }
    bool feasible = true;                            // check perturbed segs
    for (int v : _changed) {                         // (cumulative deltas now)
      const int seg = _lct.segBelow(v, _R);
      if (seg == _lct.NIL) continue;
      if (_lct.mi[seg] < 0 || _lct.mj[seg] < 0) { feasible = false; break; }
    }
    if (feasible) {                                  // optimal: costs fixed ⇒
      ++_warmCnt;                                    // still dual-feasible
      Status st = finish();
      _haveBasis = (st == OPTIMAL);
      snapshot();
      return st;
    }
    coldStart();                                     // basis broke ⇒ cold
    pivotLoop();
    Status st = finish();
    _haveBasis = (st == OPTIMAL);
    snapshot();
    ++_coldCnt;
    return st;
  }

  Cost totalCost() const { return _total; }
  Value flow(int arc_id) const { return _flowOut[arc_id]; }
  int warmCount() const { return _warmCnt; }
  int coldCount() const { return _coldCnt; }

 private:
  enum { ST_UPPER = -1, ST_TREE = 0, ST_LOWER = 1 };
  static constexpr Value VINF = std::numeric_limits<Value>::max() / 4;

  // ---- embedded splay link-cut tree, 3 channels --------------------------
  //   cs : subtree-SUM of cv  (cv = edgeVal; sumToRoot(u) = pi[u])
  //   mi : subtree-MIN of vi  (+ argmin, rightmost-on-tie)
  //   mj : subtree-MIN of vj  (+ argmin, leftmost-on-tie)
  //   lazy: ai (add to vi/mi), aj (add to vj/mj), rv (reverse)
  // NIL sentinel = _N (index n+1).
  struct LCT {
    int N, NIL;
    std::vector<int> ch0, ch1, fa;
    std::vector<char> rv;
    std::vector<Cost> cv, cs;                 // cost channel (sum)
    std::vector<Value> vi, mi, ai;            // i-residual channel (min)
    std::vector<Value> vj, mj, aj;            // j-residual channel (min)
    std::vector<int> air, ajl;                // argmin: vi rightmost, vj leftmost

    void init(int n) {
      N = n; NIL = n;                         // node ids 0..n-1, NIL = n
      const int S = n + 1;
      ch0.assign(S, NIL); ch1.assign(S, NIL); fa.assign(S, NIL);
      rv.assign(S, 0);
      cv.assign(S, 0); cs.assign(S, 0);
      vi.assign(S, 0); mi.assign(S, 0); ai.assign(S, 0); air.assign(S, -1);
      vj.assign(S, 0); mj.assign(S, 0); aj.assign(S, 0); ajl.assign(S, -1);
      // NIL identities
      cs[NIL] = 0;
      mi[NIL] = VINF; air[NIL] = -1;
      mj[NIL] = VINF; ajl[NIL] = -1;
      for (int i = 0; i < n; ++i) { air[i] = i; ajl[i] = i; }
    }
    void pull(int x) {
      if (x == NIL) return;
      const int l = ch0[x], r = ch1[x];
      cs[x] = cs[l] + cv[x] + cs[r];
      // vi min, rightmost argmin on tie (prefer larger in-order position)
      {
        Value m = vi[x]; int a = x;
        if (mi[l] < m) { m = mi[l]; a = air[l]; }
        if (mi[r] <= m) { m = mi[r]; a = air[r]; }   // r is to the right
        mi[x] = m; air[x] = a;
      }
      // vj min, leftmost argmin on tie (prefer smaller in-order position)
      {
        Value m = vj[x]; int a = x;
        if (mj[l] <= m) { m = mj[l]; a = ajl[l]; }    // l is to the left
        if (mj[r] <  m) { m = mj[r]; a = ajl[r]; }
        mj[x] = m; ajl[x] = a;
      }
    }
    void applyAddI(int x, Value c) {
      if (x == NIL) return;
      vi[x] += c; mi[x] += c; ai[x] += c;
    }
    void applyAddJ(int x, Value c) {
      if (x == NIL) return;
      vj[x] += c; mj[x] += c; aj[x] += c;
    }
    void applyRev(int x) {
      if (x == NIL) return;
      std::swap(ch0[x], ch1[x]);
      // Do NOT touch air/ajl here: argmin identity is recomputed by pull()
      // from children on the next splay/access (which pushes this rev down).
      // The tie-direction subtlety under reversal is a correctness (not
      // termination) concern and only ever transient — reversal happens
      // solely inside link() on the freshly-cut detached component, whose
      // stem nodes are setAll-recomputed and the rest re-derived on access.
      rv[x] ^= 1;
    }
    void push(int x) {
      if (rv[x]) {
        if (ch0[x] != NIL) applyRev(ch0[x]);
        if (ch1[x] != NIL) applyRev(ch1[x]);
        rv[x] = 0;
      }
      if (ai[x] != 0) {
        if (ch0[x] != NIL) applyAddI(ch0[x], ai[x]);
        if (ch1[x] != NIL) applyAddI(ch1[x], ai[x]);
        ai[x] = 0;
      }
      if (aj[x] != 0) {
        if (ch0[x] != NIL) applyAddJ(ch0[x], aj[x]);
        if (ch1[x] != NIL) applyAddJ(ch1[x], aj[x]);
        aj[x] = 0;
      }
    }
    bool isRoot(int x) {
      int p = fa[x];
      return p == NIL || (ch0[p] != x && ch1[p] != x);
    }
    void rotate(int x) {
      int y = fa[x], z = fa[y];
      bool xr = (ch1[y] == x);
      int b = xr ? ch0[x] : ch1[x];
      if (!isRoot(y)) { if (ch0[z] == y) ch0[z] = x; else ch1[z] = x; }
      fa[x] = z;
      if (xr) { ch1[y] = b; ch0[x] = y; } else { ch0[y] = b; ch1[x] = y; }
      if (b != NIL) fa[b] = y;
      fa[y] = x;
      pull(y); pull(x);
    }
    std::vector<int> stk;
    void splay(int x) {
      stk.clear();
      int t = x; stk.push_back(t);
      while (!isRoot(t)) { t = fa[t]; stk.push_back(t); }
      for (int i = (int)stk.size() - 1; i >= 0; --i) push(stk[i]);
      while (!isRoot(x)) {
        int y = fa[x], z = fa[y];
        if (!isRoot(y)) ((ch0[z] == y) == (ch0[y] == x)) ? rotate(y) : rotate(x);
        rotate(x);
      }
    }
    int access(int x) {
      int last = NIL;
      for (int y = x; y != NIL; y = fa[y]) {
        splay(y); ch1[y] = last; pull(y); last = y;
      }
      splay(x);
      return last;
    }
    void makeRoot(int x) { access(x); applyRev(x); }
    void link(int x, int y) { makeRoot(x); fa[x] = y; }
    void cutParent(int x) {                    // detach x from its parent
      access(x);
      int p = ch0[x];
      ch0[x] = NIL; fa[p] = NIL; pull(x);
    }
    int lca(int u, int v) { access(u); return access(v); }
    Cost sumToRoot(int u) { access(u); return cs[u]; }
    void setAll(int u, Cost c, Value Vi, Value Vj) {
      access(u); cv[u] = c; vi[u] = Vi; vj[u] = Vj; pull(u);
    }
    // Expose the path (anc-exclusive .. u-inclusive) as a single splay node,
    // returned; anc must be a strict ancestor of u (rooted, no evert).
    // After mutating the returned seg's lazy, call segPull(anc).
    int _segAnc = -1;
    int segBelow(int u, int anc) {
      access(anc); access(u); splay(anc);
      _segAnc = anc;
      return ch1[anc];                         // right subtree of anc = the seg
    }
    void segPull() { if (_segAnc >= 0) pull(_segAnc); }
  };

  void ensureBuilt() {
    if (_built) return;
    _m = (int)_src.size();
    _R = _n; _N = _n + 1;
    Cost mx = 1;
    for (int e = 0; e < _m; ++e) if (_cost[e] > mx) mx = _cost[e];
    _BIGM = mx * Cost(_n + 2) * Cost(_m + 2) + 1;
    _src.resize(_m + _n); _tgt.resize(_m + _n);
    _cost.resize(_m + _n); _cap.resize(_m + _n);
    _state.assign(_m + _n, ST_LOWER);
    _flowOut.assign(_m + _n, 0);
    _par.assign(_N, -1); _predArc.assign(_N, -1); _predDir.assign(_N, 0);
    for (int u = 0; u < _n; ++u) { _cost[_m + u] = _BIGM; _cap[_m + u] = VINF; }
    _built = true;
  }

  Cost edgeVal(int u) const {
    if (_predArc[u] < 0) return Cost(0);
    return Cost(-_predDir[u]) * _cost[_predArc[u]];
  }
  // r[u] = predDir*flow ; vi = r + off_i ; vj = -r + off_j
  Value offI(int u) const {
    return _predDir[u] == 1 ? Value(0) : _cap[_predArc[u]];
  }
  Value offJ(int u) const {
    return _predDir[u] == 1 ? _cap[_predArc[u]] : Value(0);
  }
  void refreshNode(int u) {                    // recompute all 3 channels
    const Value r = Value(_predDir[u]) * _treeFlow(u);
    _lct.setAll(u, edgeVal(u), r + offI(u), -r + offJ(u));
  }
  // current flow on the tree edge predArc[u], read back from the LCT vi.
  Value _treeFlow(int u) {
    // vi[u] = r + off_i ; r = predDir*flow ; flow = predDir*r = predDir*(vi-off_i)
    _lct.access(u);
    const Value r = _lct.vi[u] - offI(u);
    return Value(_predDir[u]) * r;
  }

  void coldStart() {
    const int m = _m, n = _n, R = _R;
    _lct.init(_N);
    for (int e = 0; e < m; ++e) { _flowOut[e] = 0; _state[e] = ST_LOWER; }
    for (int u = 0; u < n; ++u) {
      const int a = m + u;
      _cost[a] = _BIGM; _cap[a] = VINF; _state[a] = ST_TREE;
      Value f;
      if (_supply[u] >= 0) { _src[a]=u; _tgt[a]=R; f=_supply[u];  _predDir[u]=+1; }
      else                 { _src[a]=R; _tgt[a]=u; f=-_supply[u]; _predDir[u]=-1; }
      _par[u] = R; _predArc[u] = a; _flowOut[a] = f;
      _lct.link(u, R);
      const Value r = Value(_predDir[u]) * f;
      _lct.setAll(u, edgeVal(u), r + offI(u), -r + offJ(u));
    }
    _par[R] = -1; _predArc[R] = -1;
    _lct.setAll(R, Cost(0), VINF, VINF);
    _priceNext = 0; ++_piGen; _bad = false;
#ifdef PYLMCF_DYN_DEBUG
    _fdbg.assign(m + n, 0);
    for (int e = 0; e < m + n; ++e) _fdbg[e] = _flowOut[e];
    _dbgIter = 0;
#endif
  }

  void pivotLoop() {
    const int m = _m, n = _n, A = m + n;
    if ((int)_piMemo.size() < _N) { _piMemo.assign(_N, 0); _piStamp.assign(_N, 0); }
    ++_piGen;
    auto pi = [&](int x) -> Cost {
      if (_piStamp[x] == _piGen) return _piMemo[x];
      const Cost v = _lct.sumToRoot(x);
      _piMemo[x] = v; _piStamp[x] = _piGen; return v;
    };
    auto reduced = [&](int a) -> Cost {
      return _cost[a] + pi(_src[a]) - pi(_tgt[a]);
    };
    auto violation = [&](int e) -> Cost {
      const Cost rc = reduced(e);
      if (_state[e] == ST_LOWER) return rc < 0 ? -rc : Cost(0);
      return rc > 0 ? rc : Cost(0);
    };
    int block = (int)std::sqrt((double)A) + 1;
    if (block < 8) block = 8;
    if (block > A) block = A;

    const long long iter_cap = 1LL * (A + 4) * (A + 4) + 1000;
    for (long long iter = 0; iter < iter_cap; ++iter) {
      int a = -1; Cost best = 0;
      int scanned = 0, since_block = 0;
      while (scanned < A) {
        const int e = _priceNext;
        _priceNext = (_priceNext + 1 == A) ? 0 : _priceNext + 1;
        ++scanned; ++since_block;
        if (_state[e] != ST_TREE) {
          const Cost v = violation(e);
          if (v > best) { best = v; a = e; }
        }
        if (since_block >= block) { if (a >= 0) break; since_block = 0; }
      }
      if (a < 0) break;                         // optimal

      const int i = _src[a], j = _tgt[a];
      int first, second;
      if (_state[a] == ST_LOWER) { first = i; second = j; }
      else                       { first = j; second = i; }
      const int join = _lct.lca(i, j);

      // ---- findLeavingArc: path-min over the two cycle segments ----------
      // vi = first-side residual (vF), vj = second-side residual (vS), where
      // first = state==LOWER?i:j (canonical augment direction).  Query by
      // first/second (NOT fixed i/j): first side strict `<` (rightmost-tie =
      // closest to `first`), second side `<=` (leftmost-tie = closest to
      // join) — exactly LEMON's strongly-feasible Cunningham rule.
      Value delta = _cap[a];
      int leave_node = -1, out_path = 0;
      if (first != join) {
        const int seg = _lct.segBelow(first, join);
        if (seg != _lct.NIL && _lct.mi[seg] < delta) {
          delta = _lct.mi[seg]; leave_node = _lct.air[seg]; out_path = 1;
        }
      }
      if (second != join) {
        const int seg = _lct.segBelow(second, join);
        if (seg != _lct.NIL && _lct.mj[seg] <= delta) {
          delta = _lct.mj[seg]; leave_node = _lct.ajl[seg]; out_path = 2;
        }
      }

#ifdef PYLMCF_DYN_DEBUG
      {                                           // reference ratio test
        Value dd = _cap[a]; int lo = -1, op = 0;
        for (int u = first; u != join; u = _par[u]) {
          const int e = _predArc[u];
          const Value d = (_predDir[u] == -1) ? _cap[e] - _fdbg[e] : _fdbg[e];
          if (d < dd) { dd = d; lo = u; op = 1; }
        }
        for (int u = second; u != join; u = _par[u]) {
          const int e = _predArc[u];
          const Value d = (_predDir[u] == 1) ? _cap[e] - _fdbg[e] : _fdbg[e];
          if (d <= dd) { dd = d; lo = u; op = 2; }
        }
        if (dd != delta) {
          std::fprintf(stderr,
            "DELTA MISMATCH it=%d a=%d st=%d i=%d j=%d join=%d "
            "first=%d second=%d  lct.delta=%lld ref.delta=%lld  "
            "lct.out=%d lct.leave=%d ref.out=%d ref.leave=%d\n",
            _dbgIter, a, (int)_state[a], i, j, join, first, second,
            (long long)delta, (long long)dd, out_path, leave_node, op, lo);
          std::exit(7);
        }
      }
#endif
      // ---- changeFlow: two lazy path range-adds in the r-frame ----------
      const Value sign = (_state[a] == ST_LOWER) ? +1 : -1;
      const Value val = sign * delta;
      _flowOut[a] += val;
#ifdef PYLMCF_DYN_DEBUG
      _fdbg[a] += val;
      for (int u = i; u != join; u = _par[u])
        _fdbg[_predArc[u]] -= Value(_predDir[u]) * val;
      for (int u = j; u != join; u = _par[u])
        _fdbg[_predArc[u]] += Value(_predDir[u]) * val;
#endif
      if (i != join && val != 0) {                // r -= val on i..join excl
        const int seg = _lct.segBelow(i, join);
        _lct.applyAddI(seg, -val); _lct.applyAddJ(seg, +val);
        _lct.segPull();
      }
      if (j != join && val != 0) {                // r += val on j..join excl
        const int seg = _lct.segBelow(j, join);
        _lct.applyAddI(seg, +val); _lct.applyAddJ(seg, -val);
        _lct.segPull();
      }

      if (out_path == 0) {                        // bound flip, no pivot
        _state[a] = (_state[a] == ST_LOWER) ? ST_UPPER : ST_LOWER;
        continue;
      }

      const int u_in = (out_path == 1) ? first : second;
      const int v_in = (out_path == 1) ? second : first;
      const int u_out = leave_node;
      const int leave_arc = _predArc[u_out];
      _state[a] = ST_TREE;
      const Value lf = _treeFlow(u_out);
      _state[leave_arc] = (lf == 0) ? ST_LOWER : ST_UPPER;
      _flowOut[leave_arc] = lf;                   // materialize leaving flow

      _lct.cutParent(u_out);

      _stem.clear();
      bool stem_ok = false;
      for (int s = u_in; s >= 0 && (int)_stem.size() <= _N; s = _par[s]) {
        _stem.push_back(s);
        if (s == u_out) { stem_ok = true; break; }
      }
      if (!stem_ok) {                             // wrong argmin ⇒ u_out not
        _bad = true; return;                      // on cycle: bail (oracle
      }                                           // will flag; no infinite loop)
      const int k = (int)_stem.size() - 1;
      _oldArc.assign(k, 0); _oldDir.assign(k, 0);
      _oldFlow.assign(k, 0);
      for (int t = 0; t < k; ++t) {
        _oldArc[t]  = _predArc[_stem[t]];
        _oldDir[t]  = _predDir[_stem[t]];
        _oldFlow[t] = _treeFlow(_stem[t]);        // capture before relinking
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

      // recompute the 3 channels for every stem node from its new pred edge
      // (flow on the physical arc is unchanged; predDir/predArc changed).
      for (int t = 0; t < k; ++t) {
        const int child = _stem[t + 1];
        // _oldFlow[t] is the DIGRAPH flow on physical arc oldArc[t] (now
        // child's pred edge); predDir[child] was set to -oldDir[t] above, so
        // r[child] = predDir[child] * flow_on_arc directly.
        const Value r = Value(_predDir[child]) * _oldFlow[t];
        _lct.setAll(child, edgeVal(child), r + offI(child), -r + offJ(child));
      }
      {
        const Value f = _flowOut[a];              // entering arc flow
        const Value r = Value(_predDir[u_in]) * f;
        _lct.setAll(u_in, edgeVal(u_in), r + offI(u_in), -r + offJ(u_in));
      }
      _lct.link(u_in, v_in);
      ++_piGen;
#ifdef PYLMCF_DYN_DEBUG
      ++_dbgIter;
      for (int u = 0; u < n; ++u) {
        const int e = _predArc[u];
        if (e < 0 || _state[e] != ST_TREE) continue;
        const Value lf = _treeFlow(u);
        if (lf != _fdbg[e]) {
          std::fprintf(stderr,
            "FLOW MISMATCH it=%d node=%d arc=%d predDir=%d  lct=%lld ref=%lld "
            "(pivot a=%d leave_arc=%d u_in=%d v_in=%d k=%d)\n",
            _dbgIter, u, e, _predDir[u], (long long)lf, (long long)_fdbg[e],
            a, leave_arc, u_in, v_in, k);
          std::exit(8);
        }
      }
      {                                            // pi consistency: tree rc==0
        for (int u = 0; u < n; ++u) {
          const int e = _predArc[u];
          if (e < 0 || _state[e] != ST_TREE) continue;
          const Cost rc = _cost[e] + _lct.sumToRoot(_src[e]) -
                          _lct.sumToRoot(_tgt[e]);
          if (rc != 0) {
            std::fprintf(stderr,
              "PI MISMATCH it=%d node=%d arc=%d rc=%lld\n",
              _dbgIter, u, e, (long long)rc);
            std::exit(9);
          }
        }
      }
#endif
    }
  }

  Status finish() {
    if (_bad) return INFEASIBLE;                   // structural bail-out
    for (int u = 0; u < _n; ++u) {                // materialize all tree flows
      if (_predArc[u] >= 0 && _state[_predArc[u]] == ST_TREE)
        _flowOut[_predArc[u]] = _treeFlow(u);
    }
    for (int u = 0; u < _n; ++u)
      if (_flowOut[_m + u] != 0) return INFEASIBLE;
    _total = 0;
    for (int e = 0; e < _m; ++e) _total += Cost(_flowOut[e]) * _cost[e];
    return OPTIMAL;
  }

  void snapshot() {
    if ((int)_prevSupply.size() < _n) _prevSupply.assign(_n, 0);
    if ((int)_prevCap.size() < _m) _prevCap.assign(_m, 0);
    for (int v = 0; v < _n; ++v) _prevSupply[v] = _supply[v];
    for (int e = 0; e < _m; ++e) _prevCap[e] = _cap[e];
  }

  int _n, _m = 0, _R = 0, _N = 0;
  bool _built = false;
  Cost _BIGM = 0;
  std::vector<int> _src, _tgt;
  std::vector<Cost> _cost;
  std::vector<Value> _cap, _flowOut, _supply;
  std::vector<signed char> _state;
  std::vector<int> _par, _predArc, _predDir;
  std::vector<int> _stem, _oldArc, _oldDir, _changed;
  std::vector<Value> _oldFlow, _prevSupply, _prevCap;
  bool _haveBasis = false;
  int _warmCnt = 0, _coldCnt = 0;
  std::vector<Cost> _piMemo;
  std::vector<int> _piStamp;
  int _piGen = 0, _priceNext = 0;
  bool _bad = false;
  LCT _lct;
  Cost _total = 0;
#ifdef PYLMCF_DYN_DEBUG
  std::vector<Value> _fdbg;
  int _dbgIter = 0;
#endif
};

}  // namespace pylmcf

#endif  // PYLMCF_NETWORK_SIMPLEX_LCT_DYN_H
