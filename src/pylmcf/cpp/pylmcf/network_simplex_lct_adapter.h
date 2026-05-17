// network_simplex_lct_adapter.h
// -------------------------------------------------------------------------
// Phase D (step 1): a thin adapter exposing the subset of LEMON's
// NetworkSimplex API that wnet's decompositable_graph.hpp actually uses,
// backed by pylmcf::NetworkSimplexLCT.  This lets the LCT solver be
// drop-in-tested against a real lemon::NetworkSimplex on the exact call
// pattern wnet uses (maps -> run/warmRun -> totalCost/flow/potential)
// WITHOUT touching production wnet.
//
// API surface mirrored (from a grep of decompositable_graph.hpp):
//   ctor(const GR&), upperMap, costMap, supplyMap (chainable),
//   run(PivotRule), warmRun(PivotRule, WarmRepair),
//   totalCost(), flow(Arc), potential(Node),
//   warmStartCount/coldStartCount/dualRepairCount/primalRepairCount,
//   enums ProblemType / PivotRule / WarmRepair.
// Lower bounds are 0 (wnet never calls lowerMap — verified).
//
// potential(): wnet uses it only via DIFFERENCES (residual reduced costs and
// pi[i]-pi[src]); NetworkSimplexLCT's pi obeys the same reduced-cost relation
// as LEMON's, so differences match exactly (global offset cancels).
//
// PivotRule / WarmRepair arguments are accepted for API compatibility but
// ignored: the LCT solver uses its fixed strongly-feasible pivot rule and the
// Simple (repair-or-cold) warm strategy.
// -------------------------------------------------------------------------
#ifndef PYLMCF_NETWORK_SIMPLEX_LCT_ADAPTER_H
#define PYLMCF_NETWORK_SIMPLEX_LCT_ADAPTER_H

#include <pylmcf/network_simplex_lct.h>

#include <lemon/core.h>

#include <memory>
#include <vector>

namespace pylmcf {

template <typename GR, typename V, typename C>
class NetworkSimplexLCTAdapter {
 public:
  typedef typename GR::Node Node;
  typedef typename GR::Arc Arc;

  enum ProblemType { INFEASIBLE, OPTIMAL, UNBOUNDED };
  enum PivotRule {
    FIRST_ELIGIBLE,
    BEST_ELIGIBLE,
    BLOCK_SEARCH,
    CANDIDATE_LIST,
    ALTERING_LIST
  };
  enum class WarmRepair { RepairOnly, Dual, Primal, DualRatio, DualGreedy };

  explicit NetworkSimplexLCTAdapter(const GR& g) : _g(g) {
    _n = 0;
    for (typename GR::NodeIt v(_g); v != lemon::INVALID; ++v) ++_n;
    _m = 0;
    for (typename GR::ArcIt a(_g); a != lemon::INVALID; ++a) ++_m;
    _cap.assign(_m, V(0));
    _cost.assign(_m, C(0));
    _sup.assign(_n, V(0));
    _src.assign(_m, 0);
    _tgt.assign(_m, 0);
    for (typename GR::ArcIt a(_g); a != lemon::INVALID; ++a) {
      const int id = _g.id(a);
      _src[id] = _g.id(_g.source(a));
      _tgt[id] = _g.id(_g.target(a));
    }
  }

  template <typename M>
  NetworkSimplexLCTAdapter& upperMap(const M& m) {
    for (typename GR::ArcIt a(_g); a != lemon::INVALID; ++a)
      _cap[_g.id(a)] = m[a];
    return *this;
  }
  template <typename M>
  NetworkSimplexLCTAdapter& costMap(const M& m) {
    for (typename GR::ArcIt a(_g); a != lemon::INVALID; ++a) {
      const int id = _g.id(a);
      if (_built && _cost[id] != C(m[a])) _costs_dirty = true;
      _cost[id] = m[a];
    }
    return *this;
  }
  template <typename M>
  NetworkSimplexLCTAdapter& supplyMap(const M& m) {
    for (typename GR::NodeIt v(_g); v != lemon::INVALID; ++v)
      _sup[_g.id(v)] = m[v];
    return *this;
  }

  ProblemType run(PivotRule = BLOCK_SEARCH) {
    rebuild();
    return _toPT(_s->run());
  }

  ProblemType warmRun(PivotRule = BLOCK_SEARCH,
                      WarmRepair = WarmRepair::Dual) {
    if (!_built || _costs_dirty) {
      rebuild();
      return _toPT(_s->run());
    }
    for (int e = 0; e < _m; ++e) _s->setCap(e, _cap[e]);
    for (int v = 0; v < _n; ++v) _s->setSupply(v, _sup[v]);
    return _toPT(_s->warmRun());
  }

  V totalCost() const { return _s->totalCost(); }
  V flow(const Arc& a) const { return _s->flow(_g.id(a)); }
  C potential(const Node& v) const { return _s->potential(_g.id(v)); }

  int warmStartCount() const { return _s ? _s->warmCount() : 0; }
  int coldStartCount() const { return _s ? _s->coldCount() : 0; }
  int dualRepairCount() const { return 0; }    // Simple strategy: no repairs
  int primalRepairCount() const { return 0; }

 private:
  ProblemType _toPT(typename NetworkSimplexLCT<V, C>::Status st) const {
    return st == NetworkSimplexLCT<V, C>::OPTIMAL ? OPTIMAL : INFEASIBLE;
  }

  void rebuild() {
    _s = std::make_unique<NetworkSimplexLCT<V, C>>(_n);
    for (int e = 0; e < _m; ++e)
      _s->addArc(_src[e], _tgt[e], _cost[e], _cap[e]);
    for (int v = 0; v < _n; ++v) _s->setSupply(v, _sup[v]);
    _built = true;
    _costs_dirty = false;
  }

  const GR& _g;
  int _n = 0, _m = 0;
  bool _built = false, _costs_dirty = false;
  std::vector<V> _cap, _sup;
  std::vector<C> _cost;
  std::vector<int> _src, _tgt;
  std::unique_ptr<NetworkSimplexLCT<V, C>> _s;
};

}  // namespace pylmcf

#endif  // PYLMCF_NETWORK_SIMPLEX_LCT_ADAPTER_H
