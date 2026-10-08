// test_graph.cpp
// -------------------------------------------------------------------------
// Suite for Graph<T> (graph.hpp), the stateful C++ API that the Python
// CGraph wraps and downstream C++ code uses directly.  Every optimum is
// checked against LEMON's CapacityScaling (a different algorithm) and every
// (flows, potentials) pair against a full certificate plus a Bellman-Ford
// canonical-potentials oracle — see mcf_oracle.h.  Covers: construction and
// setter validation, getters, solve-state rules, cold solves under EQ/GEQ/LEQ
// with and without lower bounds, warm chains for every pivot rule x repair
// strategy, supply-type switches, recovery after INFEASIBLE, the
// infeasibility certificate, the warm-restart policy knobs, degenerate
// graphs, and a Graph<int32_t> instantiation.
//
// Build:
//   g++ -I$(python -m pylmcf --include) -std=c++20 -O2 tests_cpp/test_graph.cpp -o /tmp/tg && /tmp/tg
// -------------------------------------------------------------------------
#include "mcf_oracle.h"

#include <pylmcf/graph.hpp>

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mcf_test;
using G64 = Graph<int64_t>;
using NS = G64::Solver;

template <typename T>
static std::vector<T> take(std::span<T> s) {  // copy a malloc'd span and free it
    std::vector<T> v(s.begin(), s.end());
    free(s.data());
    return v;
}

template <typename T>
static std::vector<T> narrow(const std::vector<int64_t>& v) {
    return std::vector<T>(v.begin(), v.end());
}

template <typename T>
static std::unique_ptr<Graph<T>> build(Instance& in) {
    auto g = std::make_unique<Graph<T>>(in.n, std::span<int>(in.starts), std::span<int>(in.ends));
    if (in.type == Supply::LEQ) g->set_supply_type(Graph<T>::Solver::LEQ);
    return g;
}

template <typename T>
static void push(Graph<T>& g, const Instance& in) {
    auto sup = narrow<T>(in.supply), caps = narrow<T>(in.caps), costs = narrow<T>(in.costs);
    g.set_node_supply(sup);
    g.set_edge_capacities(caps);
    g.set_edge_costs(costs);
    if (in.has_mins) {
        auto mins = narrow<T>(in.mins);
        g.set_edge_minimums(mins);
    }
}

// Solve and check against the oracle; returns false on any failure.
template <typename T>
static bool solve_and_check(Graph<T>& g, const Instance& in, const std::string& ctx) {
    const int64_t want = oracle_cost(in);
    try {
        g.solve();
    } catch (const std::exception& ex) {
        CHECK(false, "%s: solve threw '%s' (oracle cost %lld)", ctx.c_str(), ex.what(), (long long)want);
        return false;
    }
    CHECK(want >= 0, "%s: solved an instance the oracle calls infeasible", ctx.c_str());
    CHECK(static_cast<int64_t>(g.total_cost()) == want, "%s: cost %lld, oracle %lld", ctx.c_str(),
          (long long)g.total_cost(), (long long)want);
    try {
        const auto f = take(g.get_edge_flows());
        const auto p = take(g.get_node_potentials());
        return certify(in, std::vector<int64_t>(f.begin(), f.end()), std::vector<int64_t>(p.begin(), p.end()),
                       static_cast<int64_t>(g.total_cost()), ctx);
    } catch (const std::exception& ex) {
        // e.g. canonical_potentials rejecting non-optimal flows
        CHECK(false, "%s: reading the result threw '%s'", ctx.c_str(), ex.what());
        return false;
    }
}

static void test_construction() {
    std::vector<int> s{0, 0, 1}, e{1, 2, 2};
    G64 g(3, s, e);
    CHECK(g.no_nodes() == 3 && g.no_edges() == 3, "sizes");
    CHECK(g.edge_starts() == s && g.edge_ends() == e, "edge arrays kept");

    auto bad = [](std::vector<int> s, std::vector<int> e, int n) {
        G64 g(n, s, e);
    };
    CHECK_THROWS(std::invalid_argument, bad({1, 0}, {2, 1}, 3), "unsorted edges");
    CHECK_THROWS(std::invalid_argument, bad({0, 0}, {2, 1}, 3), "unsorted ends within a start");
    CHECK_THROWS(std::invalid_argument, bad({-1}, {1}, 3), "negative start");
    CHECK_THROWS(std::invalid_argument, bad({0}, {-1}, 3), "negative end");
    CHECK_THROWS(std::invalid_argument, bad({0}, {3}, 3), "end out of range");
    CHECK_THROWS(std::invalid_argument, bad({3}, {0}, 3), "start out of range");
    CHECK_THROWS(std::invalid_argument, bad({0, 1}, {1}, 3), "length mismatch");
    CHECK_THROWS(std::invalid_argument, bad({}, {}, -1), "negative node count");
    // Self-loops and parallel edges are legal.
    bad({0, 0, 1}, {0, 1, 1}, 2);
    CHECK(true, "self-loop/parallel accepted");
}

static void test_setters_and_state() {
    std::vector<int> s{0, 0, 1}, e{1, 2, 2};
    G64 g(3, s, e);
    std::vector<int64_t> sup{5, 0, -5}, caps{3, 3, 5}, costs{1, 3, 5}, mins{0, 1, 0};
    std::vector<int64_t> short2{1, 2}, neg{1, -1, 1};

    CHECK_THROWS(std::invalid_argument, g.set_node_supply(short2), "supply length");
    CHECK_THROWS(std::invalid_argument, g.set_edge_capacities(short2), "caps length");
    CHECK_THROWS(std::invalid_argument, g.set_edge_costs(short2), "costs length");
    CHECK_THROWS(std::invalid_argument, g.set_edge_minimums(short2), "mins length");
    CHECK_THROWS(std::invalid_argument, g.set_edge_capacities(neg), "negative cap");
    CHECK_THROWS(std::invalid_argument, g.set_edge_costs(neg), "negative cost");
    CHECK_THROWS(std::invalid_argument, g.set_edge_minimums(neg), "negative minimum");

    g.set_node_supply(sup);
    g.set_edge_capacities(caps);
    g.set_edge_costs(costs);
    g.set_edge_minimums(mins);
    CHECK(take(g.get_node_supply()) == sup, "supply roundtrip");
    CHECK(take(g.get_edge_capacities()) == caps, "caps roundtrip");
    CHECK(take(g.get_edge_costs()) == costs, "costs roundtrip");
    CHECK(take(g.get_edge_minimums()) == mins, "mins roundtrip");

    CHECK_THROWS(std::runtime_error, g.total_cost(), "total_cost before solve");
    CHECK_THROWS(std::runtime_error, g.get_edge_flows(), "flows before solve");
    CHECK_THROWS(std::runtime_error, g.get_node_potentials(), "potentials before solve");

    g.solve();
    // Direct 0->2 saturated (3 units x 3, meets its minimum of 1), the other
    // 2 units over 0->1->2 at 1 + 5.
    CHECK(g.total_cost() == 3 * 3 + 2 * (1 + 5), "README-style optimum: %lld", (long long)g.total_cost());
    // Every setter, and the supply type, invalidates the result.
    auto invalidated = [&](auto&& mutate, const char* what) {
        g.solve();
        mutate();
        CHECK_THROWS(std::runtime_error, g.total_cost(), "%s did not invalidate", what);
    };
    invalidated([&] { g.set_node_supply(sup); }, "set_node_supply");
    invalidated([&] { g.set_edge_capacities(caps); }, "set_edge_capacities");
    invalidated([&] { g.set_edge_costs(costs); }, "set_edge_costs");
    invalidated([&] { g.set_edge_minimums(mins); }, "set_edge_minimums");
    invalidated([&] { g.set_supply_type(NS::GEQ); }, "set_supply_type");
    // Settings that do not change the problem keep the result.
    g.solve();
    g.set_pivot_rule(NS::FIRST_ELIGIBLE);
    g.set_warm_repair(NS::WarmRepair::Primal);
    g.set_warm_violation_limit(-1);
    g.set_warm_repair_budget(64.0);
    CHECK(g.total_cost() > 0, "settings must not invalidate");
    CHECK(g.pivot_rule() == NS::FIRST_ELIGIBLE && g.warm_repair() == NS::WarmRepair::Primal,
          "settings roundtrip");

    // minimum > capacity: caught at solve/cut time, whichever order set.
    std::vector<int64_t> big_min{4, 0, 0};
    g.set_edge_minimums(big_min);
    CHECK_THROWS(std::invalid_argument, g.solve(), "min > cap at solve");
    CHECK_THROWS(std::invalid_argument, g.infeasibility_cut(), "min > cap at cut");
    std::vector<int64_t> ok_min{3, 0, 0};
    g.set_edge_minimums(ok_min);
    g.solve();
    CHECK(take(g.get_edge_flows())[0] == 3, "minimum honoured");

    // A rejected update changes nothing: the valid entries before the
    // negative one must not have been stored, the result stays readable, and
    // the next (warm) solve returns the same optimum as a fresh cold graph.
    {
        std::vector<int64_t> sup1{1, 0, -1}, caps1{1, 1, 1}, costs1{1, 3, 1};
        std::vector<int64_t> bad_costs{10, -1, 0}, bad_caps{0, -1, 2}, bad_mins{1, 0, -1};
        G64 h(3, s, e);
        h.set_node_supply(sup1);
        h.set_edge_capacities(caps1);
        h.set_edge_costs(costs1);
        h.solve();
        const auto flows = take(h.get_edge_flows());
        CHECK_THROWS(std::invalid_argument, h.set_edge_costs(bad_costs), "rejected costs");
        CHECK_THROWS(std::invalid_argument, h.set_edge_capacities(bad_caps), "rejected caps");
        CHECK_THROWS(std::invalid_argument, h.set_edge_minimums(bad_mins), "rejected mins");
        CHECK(take(h.get_edge_costs()) == costs1, "rejected costs left the map unchanged");
        CHECK(take(h.get_edge_capacities()) == caps1, "rejected caps left the map unchanged");
        CHECK(take(h.get_edge_minimums()) == std::vector<int64_t>(3, 0), "rejected mins left the map unchanged");
        CHECK(h.total_cost() == 2 && take(h.get_edge_flows()) == flows, "result survives a rejected update");
        h.solve();
        CHECK(h.total_cost() == 2 && take(h.get_edge_flows()) == flows,
              "re-solve after rejected updates: %lld", (long long)h.total_cost());
    }

    // The same with the negative entry at every position of a graph long
    // enough to reach the setters' blocked main loop (4 lanes per step on
    // some targets) as well as their scalar tail: 11 edges = 2 blocks + 3.
    {
        const int m = 11;
        std::vector<int> s11(m), e11(m);
        for (int i = 0; i < m; i++) { s11[i] = i; e11[i] = i + 1; }
        G64 h(m + 1, s11, e11);
        std::vector<int64_t> caps(m), costs(m), mins(m, 0);
        for (int i = 0; i < m; i++) { caps[i] = 10 + i; costs[i] = 1 + i; }
        h.set_edge_capacities(caps);
        h.set_edge_costs(costs);
        for (int pos = 0; pos < m; pos++) {
            std::vector<int64_t> bad(m, 7);
            bad[pos] = -3;
            CHECK_THROWS(std::invalid_argument, h.set_edge_costs(bad), "rejected costs at %d", pos);
            CHECK_THROWS(std::invalid_argument, h.set_edge_capacities(bad), "rejected caps at %d", pos);
            CHECK_THROWS(std::invalid_argument, h.set_edge_minimums(bad), "rejected mins at %d", pos);
            CHECK(take(h.get_edge_costs()) == costs, "costs unchanged after rejection at %d", pos);
            CHECK(take(h.get_edge_capacities()) == caps, "caps unchanged after rejection at %d", pos);
            CHECK(take(h.get_edge_minimums()) == mins, "mins unchanged after rejection at %d", pos);
        }
        std::vector<int64_t> ok(m, 7);
        h.set_edge_costs(ok);
        CHECK(take(h.get_edge_costs()) == ok, "accepted update after rejections");
    }

    // Supply min() is rejected before anything is written (LEMON negates
    // supplies; the LEQ cut did too, and returned "feasible" for it).
    {
        std::vector<int> s0, e0;
        G64 h(1, s0, e0);
        h.set_supply_type(NS::LEQ);
        std::vector<int64_t> lo{std::numeric_limits<int64_t>::min()}, lo1{std::numeric_limits<int64_t>::min() + 1};
        CHECK_THROWS(std::invalid_argument, h.set_node_supply(lo), "supply min rejected");
        CHECK(take(h.get_node_supply()) == std::vector<int64_t>{0}, "rejected supply left the map unchanged");
        h.set_node_supply(lo1);
        const auto cut = h.infeasibility_cut();
        CHECK(cut.size() == 1 && cut[0], "LEQ min + 1 supply is infeasible, cut names the node");
        CHECK_THROWS(std::runtime_error, h.solve(), "LEQ min + 1 supply solve");
    }
    // Costs above MAX_COST (2^62 - 1) are rejected and change nothing; the
    // bound itself solves (2^62 used to come back INFEASIBLE).
    {
        std::vector<int> s1{0}, e1{1};
        G64 h(2, s1, e1);
        std::vector<int64_t> sup{1, -1}, cap{1}, ok{G64::MAX_COST}, big{G64::MAX_COST + 1}, neg_big{-1};
        h.set_node_supply(sup);
        h.set_edge_capacities(cap);
        h.set_edge_costs(ok);
        CHECK_THROWS(std::invalid_argument, h.set_edge_costs(big), "cost 2^62 rejected");
        CHECK(take(h.get_edge_costs()) == ok, "rejected cost left the map unchanged");
        h.solve();
        CHECK(h.total_cost() == G64::MAX_COST, "cost 2^62 - 1 solves: %lld", (long long)h.total_cost());
        static_assert(G64::MAX_COST == (int64_t(1) << 62) - 1);
        static_assert(Graph<int32_t>::MAX_COST == (int32_t(1) << 30) - 1);
    }

    const std::string str = g.to_string();
    CHECK(str.find("3 nodes and 3 edges") != std::string::npos && str.find("0 -> 1") != std::string::npos,
          "to_string: %s", str.c_str());
}

static void test_degenerate_graphs() {
    {   // no edges, zero supply
        std::vector<int> s, e;
        G64 g(4, s, e);
        std::vector<int64_t> sup(4, 0), none;
        g.set_node_supply(sup);
        g.set_edge_capacities(none);
        g.set_edge_costs(none);
        g.solve();
        CHECK(g.total_cost() == 0, "edgeless cost");
        CHECK(take(g.get_node_potentials()) == std::vector<int64_t>(4, 0), "edgeless potentials");
        CHECK(g.infeasibility_cut().empty(), "edgeless feasible");
    }
    {   // no edges, nonzero supply: infeasible, the supply node is the cut
        std::vector<int> s, e;
        G64 g(2, s, e);
        std::vector<int64_t> sup{1, -1}, none;
        g.set_node_supply(sup);
        g.set_edge_capacities(none);
        g.set_edge_costs(none);
        CHECK_THROWS(std::runtime_error, g.solve(), "edgeless infeasible");
        const auto cut = g.infeasibility_cut();
        CHECK(cut.size() == 2 && cut[0] && !cut[1], "edgeless cut");
    }
    {   // capacities never set: they are zero, for the solver as for the getters
        // and the certificate (LEMON's own default upper bound is infinite)
        std::vector<int> s{0}, e{1};
        G64 g(2, s, e);
        std::vector<int64_t> sup{1, -1}, costs{1};
        g.set_node_supply(sup);
        g.set_edge_costs(costs);
        CHECK(take(g.get_edge_capacities()) == std::vector<int64_t>{0}, "default capacities");
        CHECK_THROWS(std::runtime_error, g.solve(), "default capacities infeasible");
        const auto cut = g.infeasibility_cut();
        CHECK(cut.size() == 2 && cut[0] && !cut[1], "default capacities cut");
        std::vector<int64_t> caps{1};
        g.set_edge_capacities(caps);
        g.solve();
        CHECK(g.total_cost() == 1, "capacities set after construction");
    }
    {   // zero nodes
        std::vector<int> s, e;
        G64 g(0, s, e);
        std::vector<int64_t> none;
        g.set_node_supply(none);
        g.set_edge_capacities(none);
        g.set_edge_costs(none);
        g.solve();
        CHECK(g.total_cost() == 0, "empty graph");
    }
    {   // self-loop with negative-free cost never carries flow
        std::vector<int> s{0, 0}, e{0, 1};
        G64 g(2, s, e);
        std::vector<int64_t> sup{2, -2}, caps{5, 5}, costs{0, 1};
        g.set_node_supply(sup);
        g.set_edge_capacities(caps);
        g.set_edge_costs(costs);
        g.solve();
        CHECK(g.total_cost() == 2, "self-loop graph cost");
    }
}

static void test_cold_random(Rng& rng) {
    for (int iter = 0; iter < 600; iter++) {
        const Supply st = static_cast<Supply>(iter % 3);
        const bool mins = (iter / 3) % 2;
        const int n = static_cast<int>(rint(rng, 2, 30));
        auto in = random_instance(rng, n, static_cast<int>(rint(rng, 1, 4 * n)), mins, st);
        auto g = build<int64_t>(in);
        push(*g, in);
        CHECK(g->infeasibility_cut().empty(), "cold %d: feasible instance got a cut", iter);
        solve_and_check(*g, in, lbl("cold %d %s", iter, supply_name(st)));
    }
}

static void test_warm_chains(Rng& rng) {
    const NS::PivotRule rules[] = {NS::FIRST_ELIGIBLE, NS::BEST_ELIGIBLE, NS::BLOCK_SEARCH,
                                   NS::CANDIDATE_LIST, NS::ALTERING_LIST};
    const NS::WarmRepair repairs[] = {NS::WarmRepair::RepairOnly, NS::WarmRepair::Dual,
                                      NS::WarmRepair::Primal, NS::WarmRepair::DualRatio,
                                      NS::WarmRepair::DualGreedy};
    long dual_total = 0, primal_total = 0, warm_total = 0;
    for (auto rule : rules) {
        for (auto repair : repairs) {
            for (int chain = 0; chain < 4; chain++) {
                const int n = static_cast<int>(rint(rng, 4, 40));
                auto in = random_instance(rng, n, static_cast<int>(rint(rng, n, 4 * n)), false);
                auto g = build<int64_t>(in);
                g->set_pivot_rule(rule);
                g->set_warm_repair(repair);
                push(*g, in);
                char label[64];  // snprintf, not string operator+: see num()
                std::snprintf(label, sizeof label, "rule %d repair %d chain %d", static_cast<int>(rule),
                              static_cast<int>(repair), chain);
                const std::string base(label);
                solve_and_check(*g, in, lbl("%s first", label));
                const int steps = 10;
                for (int step = 0; step < steps; step++) {
                    redraw(rng, in);
                    if (step % 4 == 3)
                        for (auto& c : in.costs) c = rint(rng, 0, 50);
                    push(*g, in);
                    solve_and_check(*g, in, lbl("%s step %d", label, step));
                }
                const int resolves = g->warm_start_count() + g->cold_start_count() +
                                     g->dual_repair_count() + g->primal_repair_count();
                CHECK(resolves == steps, "%s: counters sum %d != %d", base.c_str(), resolves, steps);
                if (repair == NS::WarmRepair::RepairOnly)
                    CHECK(g->dual_repair_count() == 0 && g->primal_repair_count() == 0,
                          "%s: repair_only ran a repair", base.c_str());
                if (repair == NS::WarmRepair::Primal)
                    CHECK(g->dual_repair_count() == 0, "%s: primal ran dual", base.c_str());
                if (repair != NS::WarmRepair::Primal)
                    CHECK(g->primal_repair_count() == 0, "%s: non-primal ran primal", base.c_str());
                dual_total += g->dual_repair_count();
                primal_total += g->primal_repair_count();
                warm_total += g->warm_start_count();
            }
        }
    }
    // A regression that quietly routes everything through cold init must fail.
    CHECK(dual_total > 0 && primal_total > 0 && warm_total > 0,
          "warm paths never exercised: warm=%ld dual=%ld primal=%ld", warm_total, dual_total, primal_total);
}

static void test_minimums_and_supply_switches(Rng& rng) {
    // Lower bounds force cold re-solves; supply-type switches drop the basis.
    // Correctness must survive both, interleaved with balanced warm steps.
    for (int chain = 0; chain < 30; chain++) {
        const int n = static_cast<int>(rint(rng, 3, 25));
        const int m = static_cast<int>(rint(rng, n, 3 * n));
        auto in = random_instance(rng, n, m, chain % 2 == 1);
        auto g = build<int64_t>(in);
        push(*g, in);
        solve_and_check(*g, in, "switch first");
        for (int step = 0; step < 9; step++) {
            redraw(rng, in);
            const Supply st = static_cast<Supply>(step % 3);
            in.type = st;
            if (st != Supply::EQ) {
                for (int u = 0; u < n; u++) {
                    const int64_t x = rint(rng, 0, 2) == 0 ? rint(rng, 0, 4) : 0;
                    in.supply[u] += st == Supply::LEQ ? x : -x;
                }
            }
            // EQ instances are solved under whichever type is current.
            if (st != Supply::EQ) g->set_supply_type(st == Supply::LEQ ? NS::LEQ : NS::GEQ);
            Instance check = in;
            if (st == Supply::EQ) check.type = g->supply_type() == NS::LEQ ? Supply::LEQ : Supply::GEQ;
            push(*g, in);
            solve_and_check(*g, check, lbl("switch chain %d step %d", chain, step));
        }
    }
}

// Barrier inequality of LEMON's Circulation, independently re-derived.
static bool valid_barrier(const Instance& in, const std::vector<char>& cut) {
    int64_t out_cap = 0, in_min = 0, in_cap = 0, out_min = 0, sup = 0;
    for (size_t e = 0; e < in.starts.size(); e++) {
        const bool s = cut[in.starts[e]], t = cut[in.ends[e]];
        if (s && !t) { out_cap += in.caps[e]; out_min += in.mins[e]; }
        if (t && !s) { in_cap += in.caps[e]; in_min += in.mins[e]; }
    }
    for (int u = 0; u < in.n; u++) if (cut[u]) sup += in.supply[u];
    return in.type == Supply::LEQ ? (in_cap - out_min < -sup) : (out_cap - in_min < sup);
}

static void test_infeasibility(Rng& rng) {
    int infeasible = 0;
    for (int iter = 0; iter < 600; iter++) {
        const Supply st = iter % 2 ? Supply::LEQ : Supply::GEQ;
        const int n = static_cast<int>(rint(rng, 2, 20));
        auto in = random_instance(rng, n, static_cast<int>(rint(rng, 1, 3 * n)), iter % 4 >= 2, st);
        for (size_t e = 0; e < in.caps.size(); e++)
            in.caps[e] = std::max(in.mins[e], in.caps[e] - rint(rng, 0, 8));
        if (iter % 5 == 4)  // imbalance the type cannot absorb
            in.supply[rint(rng, 0, n - 1)] += (st == Supply::GEQ ? 1 : -1) * rint(rng, 1, 5);
        auto g = build<int64_t>(in);
        push(*g, in);
        const auto cut = g->infeasibility_cut();
        const bool oracle_feasible = oracle_cost(in) >= 0;
        bool solved = true;
        try { g->solve(); } catch (const std::runtime_error&) { solved = false; }
        CHECK(solved == oracle_feasible, "infeas %d: solve %d vs oracle %d", iter, solved, oracle_feasible);
        CHECK(cut.empty() == solved, "infeas %d: cut presence %d vs solved %d", iter, !cut.empty(), solved);
        if (!cut.empty()) {
            infeasible++;
            CHECK(cut.size() == static_cast<size_t>(n) && valid_barrier(in, cut),
                  "infeas %d: not a valid barrier", iter);
        }
        if (solved) solve_and_check(*g, in, lbl("infeas-feasible %d", iter));
    }
    CHECK(infeasible > 100, "too few infeasible instances (%d) to mean anything", infeasible);

    // Recovery: an INFEASIBLE solve invalidates the basis; the next feasible
    // solve must be right (it goes through a plain run()).
    for (int chain = 0; chain < 40; chain++) {
        const int n = static_cast<int>(rint(rng, 3, 20));
        auto in = random_instance(rng, n, static_cast<int>(rint(rng, n, 3 * n)), false);
        auto g = build<int64_t>(in);
        push(*g, in);
        solve_and_check(*g, in, "recover first");
        auto broken = in;
        broken.supply[0] += 1000;
        broken.supply[n - 1] -= 1000;
        push(*g, broken);
        CHECK_THROWS(std::runtime_error, g->solve(), "recover %d: should be infeasible", chain);
        CHECK_THROWS(std::runtime_error, g->total_cost(), "recover %d: result after failure", chain);
        redraw(rng, in);
        push(*g, in);
        solve_and_check(*g, in, lbl("recover %d", chain));
    }
}

static void test_policy_knobs(Rng& rng) {
    // violation limit 0: every failed basis patch goes straight to cold and is
    // recorded as a policy decision; results stay correct.
    long policy = 0;
    for (int chain = 0; chain < 10; chain++) {
        auto in = random_instance(rng, 40, 160, false);
        auto g = build<int64_t>(in);
        g->set_warm_violation_limit(0);
        push(*g, in);
        solve_and_check(*g, in, "policy first");
        for (int step = 0; step < 8; step++) {
            redraw(rng, in);
            push(*g, in);
            solve_and_check(*g, in, "policy step");
        }
        CHECK(g->dual_repair_count() == 0 && g->primal_repair_count() == 0, "limit 0 still repaired");
        policy += g->policy_cold_count();
    }
    CHECK(policy > 0, "violation limit 0 never recorded a policy cold start");

    if (!std::getenv("PYLMCF_WARM_REPAIR_BUDGET")) {
        std::vector<int> s{0}, e{1};
        G64 g(2, s, e);
        CHECK(g.warm_repair_budget() == 64.0, "default budget");
        g.set_warm_repair_budget(2.5);
        CHECK(g.warm_repair_budget() == 2.5, "budget roundtrip");
    }
}

static void test_int32_graph(Rng& rng) {
    // Graph<T> is a template; T = int32_t must work end to end too.
    for (int iter = 0; iter < 150; iter++) {
        const Supply st = static_cast<Supply>(iter % 3);
        const int n = static_cast<int>(rint(rng, 2, 20));
        auto in = random_instance(rng, n, static_cast<int>(rint(rng, 1, 3 * n)), iter % 2, st);
        auto g = build<int32_t>(in);
        push(*g, in);
        solve_and_check(*g, in, lbl("int32 %d", iter));
        for (int step = 0; step < 3 && st == Supply::EQ && !in.has_mins; step++) {
            redraw(rng, in);
            push(*g, in);
            solve_and_check(*g, in, lbl("int32 warm %d", iter));
        }
    }
}

int main() {
    if (std::getenv("PYLMCF_WARM_VIOLATION_LIMIT")) {
        std::printf("refusing to run under PYLMCF_WARM_VIOLATION_LIMIT (it overrides the policy under test)\n");
        return 2;
    }
    Rng rng(20261005);
    section("construction", [] { test_construction(); }, true);
    section("setters and state", [] { test_setters_and_state(); }, true);
    section("degenerate graphs", [] { test_degenerate_graphs(); }, true);
    section("cold random", [&] { test_cold_random(rng); });
    section("warm chains", [&] { test_warm_chains(rng); });
    section("minimums and supply switches", [&] { test_minimums_and_supply_switches(rng); });
    section("infeasibility", [&] { test_infeasibility(rng); });
    section("policy knobs", [&] { test_policy_knobs(rng); });
    section("int32 graph", [&] { test_int32_graph(rng); });
    return finish("test_graph");
}
