// c++ -std=c++20 -Isrc/pylmcf/cpp tests_cpp/test_fast_duals.cpp -o /tmp/duals
#include <cassert>
#include <iostream>
#include <lemon/static_graph.h>
#include <lemon/network_simplex.h>
#include <pylmcf/network_simplex_lct.h>
#include <pylmcf/network_simplex_lct_dyn.h>
#include <pylmcf/network_simplex_lct_adapter.h>

using I = int64_t;

template<class S> void test_lct() {
    S s(3);
    s.addArc(0, 1, 1, 3); s.addArc(0, 2, 3, 3); s.addArc(1, 2, 5, 5);
    s.setSupply(0, 5); s.setSupply(2, -5);
    assert(s.run() == S::OPTIMAL);
    for (int iter = 0; iter < 4; ++iter) {
        std::vector<I> pi(3), rc(3), lo(3), up(3);
        s.dualValues(pi, rc, lo, up);
        I dual = -5 * pi[0] + 5 * pi[2];
        I caps[] = {3, 3, 5};
        for (int e = 0; e < 3; ++e) {
            assert(rc[e] == s.reducedCost(e));
            assert(lo[e] == s.lowerBoundMultiplier(e));
            assert(up[e] == s.upperBoundMultiplier(e));
            dual += caps[e] * up[e];
        }
        assert(dual == 21 && dual == s.totalCost());
        assert(s.warmRun() == S::OPTIMAL);
    }
}

template<class S> void test_negative_cost_artificial_budget() {
    // The negative dead-end edge cannot carry real flow. A Big-M based only
    // on positive costs instead makes an artificial cycle profitable and
    // falsely reports INFEASIBLE for this feasible, zero-cost problem.
    S s(3);
    s.addArc(0, 1, 0, 5);
    s.addArc(1, 2, -I(10000000000000LL), 4);
    s.setSupply(0, 5); s.setSupply(1, -5);
    assert(s.run() == S::OPTIMAL);
    assert(s.flow(0) == 5 && s.flow(1) == 0 && s.totalCost() == 0);
    S huge(2);
    huge.addArc(0, 1, std::numeric_limits<I>::max() / 4, 1);
    bool overflow = false;
    try { huge.run(); }
    catch (const std::overflow_error&) { overflow = true; }
    assert(overflow);
}

int main() {
    test_lct<pylmcf::NetworkSimplexLCT<I, I>>();
    test_lct<pylmcf::NetworkSimplexLCTDyn<I, I>>();
    test_negative_cost_artificial_budget<pylmcf::NetworkSimplexLCT<I, I>>();
    test_negative_cost_artificial_budget<pylmcf::NetworkSimplexLCTDyn<I, I>>();
    lemon::StaticDigraph g;
    std::vector<std::pair<int,int>> arcs = {{0,1}, {0,2}, {1,2}};
    g.build(3, arcs.begin(), arcs.end());
    lemon::StaticDigraph::ArcMap<I> cost(g), cap(g), lower(g);
    lemon::StaticDigraph::NodeMap<I> supply(g);
    I costs[] = {1,3,5}, caps[] = {3,3,5};
    for (int e=0; e<3; ++e) {
        auto a=g.arcFromId(e); cost[a]=costs[e]; cap[a]=caps[e]; lower[a]=1;
    }
    supply[g.nodeFromId(0)]=5; supply[g.nodeFromId(2)]=-5;
    lemon::NetworkSimplex<lemon::StaticDigraph,I,I> s(g);
    s.costMap(cost).upperMap(cap).lowerMap(lower).supplyMap(supply);
    assert(s.run()==decltype(s)::OPTIMAL);
    std::vector<I> pi(3), rc(3), lo(3), up(3);
    s.dualValues(pi,rc,lo,up);
    I dual=-5*pi[0]+5*pi[2];
    for (int e=0;e<3;++e) dual+=caps[e]*up[e]+lo[e];
    assert(dual==s.totalCost());
    pylmcf::NetworkSimplexLCTAdapter<lemon::StaticDigraph,I,I> adapter(g);
    adapter.costMap(cost).upperMap(cap).supplyMap(supply);
    assert(adapter.run()==decltype(adapter)::OPTIMAL);
    adapter.dualValues(pi,rc,lo,up);
    assert(adapter.reducedCost(g.arcFromId(1))==rc[1]);
    assert(pylmcf::reducedCost<I>(7, -(I(1)<<62), -(I(1)<<62))==7);
    bool overflow=false;
    try { pylmcf::reducedCost<I>(1, std::numeric_limits<I>::max(), -1); }
    catch (const std::overflow_error&) { overflow=true; }
    assert(overflow);
    std::cout << "Fast dual certificates passed for all simplex backends\n";
}
