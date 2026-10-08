// test_degenerate.cpp
// -------------------------------------------------------------------------
// Degenerate inputs through every header-level entry point: 0-3 nodes; no
// edges, self-loops, parallel and antiparallel arcs, chains, isolated nodes;
// zero, balanced and unbalanced supplies; zero and positive capacities; with
// and without minimums and potentials.  The bugs this guards against are
// out-of-bounds reads and other undefined behaviour that release builds
// commit silently and only checked builds see -- e.g. infeasibility_cut() on
// a zero-node graph, where LEMON's Elevator read node 0 of an empty list.
// There is deliberately no oracle here (the other suites check optima): a
// case passes if it returns or throws a std::exception (a rejection is
// fine), and fails if it dies.  Every case runs in its own fork()ed child,
// so one abort does not hide the next; the suite is only meaningful under
// the sanitizer / hardened-library lanes that make such reads abort.
//
// Covers: Graph<T> (GEQ/LEQ, minimums, warm re-solve, cut, potentials,
// getters); lmcf_impl with all four LEMON solvers (minimums, potentials);
// canonical_potentials; NetworkSimplexLCT / NetworkSimplexLCTDyn and the
// LEMON-API adapter (in scope only: balanced supply, as their bindings
// enforce); ChainSolver1D; LinkCutTree.
//
// Build:
//   g++ -I$(python -m pylmcf --include) -std=c++20 -O1 -g -D_GLIBCXX_DEBUG \
//       -fsanitize=address,undefined tests_cpp/test_degenerate.cpp -o /tmp/td && /tmp/td
// -------------------------------------------------------------------------
#include "mcf_oracle.h"

#include <pylmcf/graph.hpp>
#include <pylmcf/lmcf.hpp>
#include <pylmcf/canonical_potentials.hpp>
#include <pylmcf/network_simplex_lct.h>
#include <pylmcf/network_simplex_lct_dyn.h>
#include <pylmcf/network_simplex_lct_adapter.h>
#include <pylmcf/chain_solver_1d.h>
#include <pylmcf/link_cut_tree.h>
#include <lemon/cycle_canceling.h>
#include <lemon/cost_scaling.h>
#include <lemon/capacity_scaling.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using namespace mcf_test;
using V = std::vector<int64_t>;
using I = std::vector<int>;

// Run body in a child; CHECK that it returned or threw, rather than died.
template <typename F>
static void in_child(const std::string& label, F body) {
    std::fflush(stdout);
    const pid_t pid = fork();
    if (pid == 0) {
        try { body(); } catch (const std::exception&) { /* a rejection is fine */ }
        std::fflush(stdout);
        _exit(0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0, "%s: child died (%s %d)", label.c_str(),
          WIFSIGNALED(status) ? "signal" : "exit", WIFSIGNALED(status) ? WTERMSIG(status) : WEXITSTATUS(status));
}

struct Shape { const char* name; int n; I s, e; };

static std::vector<Shape> shapes() {
    return {
        {"n0", 0, {}, {}},
        {"n1", 1, {}, {}},
        {"n1-selfloop", 1, {0}, {0}},
        {"n2", 2, {}, {}},
        {"n2-edge", 2, {0}, {1}},
        {"n2-parallel", 2, {0, 0}, {1, 1}},
        {"n2-both-ways", 2, {0, 1}, {1, 0}},
        {"n3-chain", 3, {0, 1}, {1, 2}},
        {"n3-isolated", 3, {0}, {1}},
    };
}

struct SupplyCase { const char* name; V v; };

static std::vector<SupplyCase> supplies(int n) {
    std::vector<SupplyCase> out{{"zero", V(n, 0)}};
    if (n >= 1) {
        V p(n, 0), q(n, 0);
        p[0] = 2; q[0] = -2;
        out.push_back({"pos", p});
        out.push_back({"neg", q});
    }
    if (n >= 2) {
        V b(n, 0);
        b[0] = 2; b[n - 1] = -2;
        out.push_back({"bal", b});
    }
    return out;
}

static void graph_cases(const Shape& sh, const SupplyCase& sup, int64_t cap) {
    const size_t m = sh.s.size();
    for (int leq = 0; leq < 2; leq++)
    for (int withmin = 0; withmin < 2; withmin++)
        in_child(lbl("Graph %s/%s/cap%lld/%s%s", sh.name, sup.name, (long long)cap,
                     leq ? "leq" : "geq", withmin ? "/min" : ""), [&] {
            I s = sh.s, e = sh.e;
            V sp = sup.v, caps(m, cap), costs(m, 1), mins(m, std::min<int64_t>(1, cap));
            Graph<int64_t> g(sh.n, s, e);
            if (leq) g.set_supply_type(Graph<int64_t>::Solver::LEQ);
            g.set_node_supply(sp);
            g.set_edge_capacities(caps);
            g.set_edge_costs(costs);
            if (withmin) g.set_edge_minimums(mins);
            (void)g.infeasibility_cut();
            if (!under_msan) (void)g.to_string();  // std::to_string inside
            try {
                g.solve();
                (void)g.total_cost();
                free(g.get_edge_flows().data());
                free(g.get_node_potentials().data());
            } catch (const std::runtime_error&) {}
            g.set_node_supply(sp);                 // re-solve: the warm path
            try { g.solve(); } catch (const std::runtime_error&) {}
            (void)g.infeasibility_cut();
            free(g.get_node_supply().data());
            free(g.get_edge_capacities().data());
            free(g.get_edge_minimums().data());
            free(g.get_edge_costs().data());
        });
}

template <typename Call>
static void functional_cases(const char* solver, Call call, const Shape& sh, const SupplyCase& sup, int64_t cap) {
    const size_t m = sh.s.size();
    for (int withmin = 0; withmin < 2; withmin++)
    for (int withpot = 0; withpot < 2; withpot++)
        in_child(lbl("lmcf_%s %s/%s/cap%lld%s%s", solver, sh.name, sup.name, (long long)cap,
                     withmin ? "/min" : "", withpot ? "/pot" : ""), [&] {
            V sp = sup.v, s(sh.s.begin(), sh.s.end()), e(sh.e.begin(), sh.e.end());
            V caps(m, cap), costs(m, 1), out(m);
            V mins = withmin ? V(m, std::min<int64_t>(1, cap)) : V{};
            std::vector<LmcfCost> pot(withpot ? sh.n : 0);
            call(sp, s, e, caps, mins, costs, out, pot);
        });
}

template <typename S>
static void lct_case(const char* name, const Shape& sh, const SupplyCase& sup, int64_t cap) {
    const size_t m = sh.s.size();
    in_child(lbl("%s %s/%s/cap%lld", name, sh.name, sup.name, (long long)cap), [&] {
        S ns(sh.n);
        for (size_t a = 0; a < m; a++) ns.addArc(sh.s[a], sh.e[a], 1, cap);
        for (int u = 0; u < sh.n; u++) ns.setSupply(u, sup.v[u]);
        if (ns.run() == S::OPTIMAL) {
            (void)ns.totalCost();
            for (size_t a = 0; a < m; a++) (void)ns.flow(int(a));
            if constexpr (requires { ns.potential(0); })
                for (int u = 0; u < sh.n; u++) (void)ns.potential(u);
        }
        for (size_t a = 0; a < m; a++) ns.setCap(int(a), cap + 1);
        (void)ns.warmRun();
        (void)ns.warmRun();
    });
}

static void adapter_case(const Shape& sh, const SupplyCase& sup, int64_t cap) {
    const size_t m = sh.s.size();
    in_child(lbl("Adapter %s/%s/cap%lld", sh.name, sup.name, (long long)cap), [&] {
        lemon::StaticDigraph g;
        std::vector<std::pair<int, int>> arcs;
        for (size_t a = 0; a < m; a++) arcs.push_back({sh.s[a], sh.e[a]});
        g.build(sh.n, arcs.begin(), arcs.end());
        lemon::StaticDigraph::ArcMap<int64_t> costs(g, 1), caps(g, cap);
        lemon::StaticDigraph::NodeMap<int64_t> supply(g);
        for (int u = 0; u < sh.n; u++) supply[g.nodeFromId(u)] = sup.v[u];
        pylmcf::NetworkSimplexLCTAdapter<lemon::StaticDigraph, int64_t, int64_t> ad(g);
        ad.upperMap(caps).costMap(costs).supplyMap(supply);
        if (ad.run() == decltype(ad)::OPTIMAL) {
            (void)ad.totalCost();
            for (lemon::StaticDigraph::ArcIt a(g); a != lemon::INVALID; ++a) (void)ad.flow(a);
            for (lemon::StaticDigraph::NodeIt u(g); u != lemon::INVALID; ++u) (void)ad.potential(u);
        }
        (void)ad.run();
    });
}

static void test_shapes() {
    for (const auto& sh : shapes())
    for (const auto& sup : supplies(sh.n))
    for (int64_t cap : {0, 3}) {
        graph_cases(sh, sup, cap);
        functional_cases("ns", [](auto&... a) { lmcf_impl<lemon::NetworkSimplex, int64_t, true>(a...); }, sh, sup, cap);
        functional_cases("cc", [](auto&... a) { lmcf_impl<lemon::CycleCanceling, int64_t>(a...); }, sh, sup, cap);
        functional_cases("cs", [](auto&... a) { lmcf_impl<lemon::CostScaling, int64_t>(a...); }, sh, sup, cap);
        functional_cases("cap", [](auto&... a) { lmcf_impl<lemon::CapacityScaling, int64_t>(a...); }, sh, sup, cap);
        int64_t total = 0;
        for (int64_t x : sup.v) total += x;
        if (total == 0) {  // the LCT solvers' scope: balanced supply
            lct_case<pylmcf::NetworkSimplexLCT<int64_t, int64_t>>("LCT", sh, sup, cap);
            lct_case<pylmcf::NetworkSimplexLCTDyn<int64_t, int64_t>>("LCTDyn", sh, sup, cap);
            adapter_case(sh, sup, cap);
        }
    }
}

static void test_others() {
    for (int n = 0; n <= 2; n++)
    for (int leq = 0; leq < 2; leq++)
        in_child(lbl("canonical_potentials n%d%s", n, leq ? "/leq" : ""), [&] {
            std::vector<LEMON_INDEX> s, e;
            V costs, caps, mins, flows, pi(n, 0), out(n);
            canonical_potentials<int64_t, int64_t>(n, s, e, costs, caps, mins, flows, pi, leq, out);
        });

    using CS = pylmcf::ChainSolver1D<int64_t, int64_t>;
    for (int k = 0; k <= 3; k++)
    for (int64_t kappa : {0, 5})
        in_child(lbl("ChainSolver1D k%d/kappa%lld", k, (long long)kappa), [&] {
            std::vector<CS::Point> pts;
            for (int i = 0; i < k; i++) pts.push_back({int64_t(i), int64_t(i % 2), int64_t(1 - i % 2)});
            (void)CS::solve(pts, kappa);
            (void)CS::solveFull(pts, kappa);
        });

    for (int n = 0; n <= 2; n++)
        in_child(lbl("LinkCutTree n%d", n), [&] {
            pylmcf::LinkCutTree<int64_t> t(n);
            for (int u = 0; u < n; u++) {
                t.setVal(u, u);
                (void)t.getVal(u); (void)t.findRoot(u); (void)t.sumToRoot(u); (void)t.minToRoot(u);
            }
            if (n == 2) {
                t.link(0, 1);
                (void)t.lca(0, 1); (void)t.pathSum(0, 1); (void)t.pathMin(0, 1);
                t.pathAdd(0, 1, 3); (void)t.pathLen(0, 1);
                t.cutParent(0);
            }
        });
}

int main() {
    section("degenerate shapes", [] { test_shapes(); });
    section("canonical potentials, chain solver, link-cut tree", [] { test_others(); });
    std::printf("degenerate cases: %ld\n", checks);
    return finish("test_degenerate");
}
