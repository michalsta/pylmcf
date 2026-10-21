// test_smaller_side_potentials.cpp
// -------------------------------------------------------------------------
// setSmallerSidePotentials(): the main pivot loop shifts the potentials of
// whichever side of the spanning tree is smaller.  The two updates differ by
// a global constant, which no reduced cost sees, and the loop renormalizes
// _pi[_root] to 0 at its end, so the solver must be indistinguishable from
// stock LEMON:
//
//   identical:  on every step of random warm chains (supplies, capacities
//               and costs redrawn; EQ/GEQ/LEQ; lower bounds; cold, mass-gated
//               and free warm repair) the on and off twins report the same
//               cost, flows, raw potentials, work units and warm/cold/repair
//               counters, and match the CapacityScaling oracle (certified);
//   exercised:  hub-shaped instances (a source feeding every left node
//               through a capacitated arc, as wnet builds them) take the
//               complement branch on many pivots;
//   drift:      instances with costs near the accepted maximum still agree
//               (the complement branch steps aside for large shifts).
//
// Build:
//   g++ -I$(python -m pylmcf --include) -std=c++20 -O2 tests_cpp/test_smaller_side_potentials.cpp -o /tmp/ts && /tmp/ts
// -------------------------------------------------------------------------
#include "mcf_oracle.h"

#include <pylmcf/graph.hpp>

using namespace mcf_test;

template <typename T>
static std::vector<T> take(std::span<T> s) {
    std::vector<T> v(s.begin(), s.end());
    free(s.data());
    return v;
}

static void push(Graph<int64_t>& g, const Instance& in) {
    g.set_node_supply(in.supply);
    g.set_edge_capacities(in.caps);
    g.set_edge_costs(in.costs);
    if (in.has_mins) g.set_edge_minimums(in.mins);
}

static void recost(Rng& rng, Instance& in, int64_t max_cost) {
    for (auto& c : in.costs)
        if (rint(rng, 0, 3) == 0) c = rint(rng, 0, max_cost);
}

// wnet-like hub: node 0 feeds left nodes 2..L+1 (capacity = their mass),
// right nodes drain to node 1 (capacity = their mass), dense-ish left->right
// matching, left->1 and 0->right trash arcs.  Supply F at 0, -F at 1.
static Instance hub_instance(Rng& rng, int L, int R, int deg, int64_t max_cost) {
    Instance in;
    in.n = 2 + L + R;
    std::vector<std::tuple<int, int, int64_t, int64_t>> arcs;  // s, t, cap, cost
    const int64_t BIG = int64_t(1) << 40;
    int64_t a_sum = 0, b_sum = 0;
    for (int i = 0; i < L; i++) {
        const int64_t a = rint(rng, 1, 30);
        a_sum += a;
        arcs.emplace_back(0, 2 + i, a, 0);
        arcs.emplace_back(2 + i, 1, BIG, max_cost);
        for (int d = 0; d < deg; d++)
            arcs.emplace_back(2 + i, 2 + L + int(rint(rng, 0, R - 1)), BIG, rint(rng, 0, max_cost));
    }
    for (int j = 0; j < R; j++) {
        const int64_t b = rint(rng, 1, 30);
        b_sum += b;
        arcs.emplace_back(2 + L + j, 1, b, 0);
        arcs.emplace_back(0, 2 + L + j, BIG, max_cost);
    }
    std::sort(arcs.begin(), arcs.end());
    arcs.erase(std::unique(arcs.begin(), arcs.end(), [](const auto& x, const auto& y) {
                   return std::get<0>(x) == std::get<0>(y) && std::get<1>(x) == std::get<1>(y);
               }), arcs.end());
    for (const auto& [s, t, c, w] : arcs) {
        in.starts.push_back(s);
        in.ends.push_back(t);
        in.caps.push_back(c);
        in.costs.push_back(w);
        in.mins.push_back(0);
    }
    in.supply.assign(in.n, 0);
    const int64_t F = std::max(a_sum, b_sum);
    in.supply[0] = F;
    in.supply[1] = -F;
    return in;
}

// Redraw the hub's left/right masses (capacities of the 0->left and
// right->1 arcs) and the supply F.
static void hub_redraw(Rng& rng, Instance& in) {
    int64_t a_sum = 0, b_sum = 0;
    for (size_t e = 0; e < in.starts.size(); e++) {
        if (in.starts[e] == 0 && in.costs[e] == 0) { in.caps[e] = rint(rng, 1, 30); a_sum += in.caps[e]; }
        if (in.ends[e] == 1 && in.costs[e] == 0) { in.caps[e] = rint(rng, 1, 30); b_sum += in.caps[e]; }
    }
    const int64_t F = std::max(a_sum, b_sum);
    in.supply[0] = F;
    in.supply[1] = -F;
}

static long long g_complement = 0;

template <typename Setup, typename Redraw>
static void twins(uint64_t seed, Instance in, int steps, Setup setup, Redraw redraw_fn, const char* what) {
    Rng rng(seed ^ 0x5157);
    Graph<int64_t> on(in.n, std::span<int>(in.starts), std::span<int>(in.ends));
    Graph<int64_t> off(in.n, std::span<int>(in.starts), std::span<int>(in.ends));
    for (auto* g : {&on, &off}) {
        if (in.type == Supply::LEQ) g->set_supply_type(Graph<int64_t>::Solver::LEQ);
        setup(*g);
    }
    on.set_smaller_side_potentials(true);
    off.set_smaller_side_potentials(false);
    CHECK(on.smaller_side_potentials() && !off.smaller_side_potentials(), "toggle getter");
    for (int step = 0; step < steps; step++) {
        push(on, in);
        push(off, in);
        on.solve();
        off.solve();
        const auto ctx = lbl("%s seed %llu step %d", what, (unsigned long long)seed, step);
        CHECK(on.total_cost() == off.total_cost(), "%s: cost %lld vs %lld", ctx.c_str(),
              (long long)on.total_cost(), (long long)off.total_cost());
        CHECK(on.total_cost() == oracle_cost(in), "%s: cost vs oracle", ctx.c_str());
        const auto f_on = take(on.get_edge_flows()), f_off = take(off.get_edge_flows());
        CHECK(f_on == f_off, "%s: flows differ", ctx.c_str());
        bool same_pi = true;
        for (int v = 0; v < in.n; v++) same_pi &= on.raw_potential(v) == off.raw_potential(v);
        CHECK(same_pi, "%s: raw potentials differ", ctx.c_str());
        CHECK(on.work_units() == off.work_units(), "%s: work units %lld vs %lld (different pivots)",
              ctx.c_str(), on.work_units(), off.work_units());
        CHECK(on.warm_start_count() == off.warm_start_count() && on.cold_start_count() == off.cold_start_count() &&
              on.dual_repair_count() == off.dual_repair_count() && on.policy_cold_count() == off.policy_cold_count(),
              "%s: warm/cold/repair counters differ", ctx.c_str());
        CHECK(off.complement_potential_updates() == 0, "%s: stock twin took the complement branch", ctx.c_str());
        certify(in, f_on, take(on.get_node_potentials()), on.total_cost(), ctx);
        redraw_fn(rng, in);
    }
    g_complement += on.complement_potential_updates();
}

static void random_chains() {
    const auto none = [](Graph<int64_t>&) {};
    for (Supply type : {Supply::EQ, Supply::GEQ, Supply::LEQ})
        for (bool mins : {false, true})
            for (int k = 0; k < 30; k++) {
                Rng rng(0x7E57 + 977 * k + 13 * int(type) + (mins ? 5 : 0));
                twins(0x7E57 + 977 * k, random_instance(rng, 20 + 3 * k, 90 + 17 * k, mins, type), 8, none,
                      [](Rng& r, Instance& in) { redraw(r, in); recost(r, in, 50); }, "random");
            }
}

static void repair_policies() {
    for (int policy = 0; policy < 3; policy++)
        for (int k = 0; k < 30; k++) {
            Rng rng(0xBEE5 + 31 * k + policy);
            twins(0xBEE5 + 31 * k, random_instance(rng, 40, 260, false, k % 3 ? Supply::EQ : Supply::GEQ), 10,
                  [&](Graph<int64_t>& g) {
                      if (policy == 0) g.set_warm_violation_limit(0);
                      if (policy == 1) { g.set_warm_violation_limit(-1); g.set_warm_violation_mass_fraction(0.05); }
                      if (policy == 2) g.set_warm_violation_limit(-1);
                  },
                  [](Rng& r, Instance& in) { redraw(r, in); }, "policy");
        }
}

static void hubs() {
    const long long before = g_complement;
    for (int k = 0; k < 40; k++) {
        Rng rng(0x40B + 7 * k);
        twins(0x40B + 7 * k, hub_instance(rng, 30 + 4 * k, 25 + 3 * k, 6, 1000), 8,
              [&](Graph<int64_t>& g) { if (k % 2) g.set_warm_violation_limit(-1); },
              [](Rng& r, Instance& in) { hub_redraw(r, in); }, "hub");
    }
    CHECK(g_complement > before, "hub instances never took the complement branch");
}

static void large_costs() {
    // Costs near Graph::MAX_COST / n: potentials are large and some shifts
    // exceed the complement branch's limit.
    for (int k = 0; k < 20; k++) {
        Rng rng(0xB16 + k);
        const int64_t big = Graph<int64_t>::MAX_COST / 4096;
        twins(0xB16 + k, hub_instance(rng, 40, 35, 5, big), 6, [](Graph<int64_t>&) {},
              [](Rng& r, Instance& in) { hub_redraw(r, in); }, "large");
    }
}

int main() {
    section("random warm chains: identical to stock", random_chains);
    section("warm repair policies: identical to stock", repair_policies);
    section("hub instances take the complement branch", hubs);
    section("large costs", large_costs);
    printf("complement updates taken: %lld\n", g_complement);
    return finish("test_smaller_side_potentials");
}
