// test_warm_mass_gate.cpp
// -------------------------------------------------------------------------
// The mass-based warm/cold policy of the vendored NetworkSimplex
// (setWarmViolationMassFraction): after a failed tree patch, simplex repair
// is skipped (cold start, counted as a policy cold start) exactly when the
// violated basic-arc mass exceeds fraction x total positive supply.
//
//   decision:   on every warm re-solve the policy outcome matches the rule
//               evaluated on the reported violation mass and supply, for a
//               range of fractions;
//   extremes:   fraction 0 never repairs a violated basis, a huge fraction
//               repairs exactly like the ungated solver (same counters);
//   combined:   a count limit of 0 still forces cold whatever the fraction;
//   disabled:   a negative fraction restores the ungated behaviour;
//   measured:   the reported violation count/mass agree with a recount from
//               the flows the patch left (via a twin solver at limit 0).
// Every solve is checked against the CapacityScaling oracle and certified.
//
// Build:
//   g++ -I$(python -m pylmcf --include) -std=c++20 -O2 tests_cpp/test_warm_mass_gate.cpp -o /tmp/tm && /tmp/tm
// -------------------------------------------------------------------------
#include "mcf_oracle.h"

#include <pylmcf/graph.hpp>

#include <cmath>

using namespace mcf_test;

template <typename T>
static std::vector<T> take(std::span<T> s) {
    std::vector<T> v(s.begin(), s.end());
    free(s.data());
    return v;
}

struct Counters {
    int warm, cold, dual, primal, policy;
};

static Counters counters(const Graph<int64_t>& g) {
    return {g.warm_start_count(), g.cold_start_count(), g.dual_repair_count(),
            g.primal_repair_count(), g.policy_cold_count()};
}

// Drive one random chain of re-solves; `before` configures the graph and
// `after` sees each warm re-solve's counter deltas and the graph.
template <typename Setup, typename Each>
static void chain(uint64_t seed, int n, int m, int steps, Setup before, Each after) {
    Rng rng(seed);
    auto in = random_instance(rng, n, m, false);
    Graph<int64_t> g(in.n, std::span<int>(in.starts), std::span<int>(in.ends));
    before(g);
    auto push = [&] {
        g.set_node_supply(in.supply);
        g.set_edge_capacities(in.caps);
        g.set_edge_costs(in.costs);
    };
    push();
    for (int step = 0; step < steps; step++) {
        const Counters c0 = counters(g);
        g.solve();
        const Counters c1 = counters(g);
        const auto f = take(g.get_edge_flows());
        const auto p = take(g.get_node_potentials());
        CHECK(g.total_cost() == oracle_cost(in), "seed %llu step %d: cost vs oracle",
              (unsigned long long)seed, step);
        certify(in, f, p, g.total_cost(), "mass gate chain");
        if (step > 0) after(g, c0, c1, step);
        redraw(rng, in);
        push();
    }
}

static void decision_matches_rule() {
    long gated = 0, repaired = 0;
    for (double fraction : {0.0, 1e-4, 1e-3, 1e-2, 0.05, 0.2, 1.0}) {
        for (int k = 0; k < 30; k++) {
            chain(0xA110 + 97 * k, 40 + k, 160 + 5 * k, 10,
                  [&](Graph<int64_t>& g) {
                      g.set_warm_violation_limit(-1);
                      g.set_warm_violation_mass_fraction(fraction);
                  },
                  [&](Graph<int64_t>& g, Counters c0, Counters c1, int step) {
                      const int policy = c1.policy - c0.policy;
                      const int repairs = (c1.dual - c0.dual) + (c1.primal - c0.primal);
                      const bool patched = c1.warm - c0.warm == 1;
                      if (patched) {
                          CHECK(policy == 0 && repairs == 0, "fast path took a policy branch");
                          return;
                      }
                      // The patch failed: the policy looked at the violation.
                      const double mass = g.last_violation_mass();
                      const double supply = g.last_positive_supply();
                      CHECK(g.last_violation_count() > 0, "failed patch reports no violation (step %d)", step);
                      CHECK(mass > 0 && supply > 0, "violation mass %g supply %g", mass, supply);
                      const bool skip = mass > fraction * supply;
                      CHECK(policy == (skip ? 1 : 0),
                            "fraction %g: mass %g supply %g -> policy %d", fraction, mass, supply, policy);
                      if (skip) {
                          CHECK(repairs == 0, "repair ran although the gate skipped it");
                          ++gated;
                      } else {
                          // Repair attempted: it either succeeded or fell back cold.
                          CHECK(repairs + (c1.cold - c0.cold) == 1, "no repair and no cold start");
                          ++repaired;
                      }
                  });
        }
    }
    CHECK(gated > 0, "the gate never skipped a repair");
    CHECK(repaired > 0, "the gate never allowed a repair");
}

static void extremes_and_disabled() {
    for (int k = 0; k < 40; k++) {
        const uint64_t seed = 0xB0B0 + 31 * k;
        Counters zero{}, huge{}, none{}, negative{};
        auto run = [&](double fraction, bool set, Counters& out) {
            chain(seed, 50, 220, 12,
                  [&](Graph<int64_t>& g) {
                      g.set_warm_violation_limit(-1);
                      if (set) g.set_warm_violation_mass_fraction(fraction);
                  },
                  [&](Graph<int64_t>& g, Counters, Counters, int) { out = counters(g); });
        };
        run(0.0, true, zero);
        run(1e300, true, huge);
        run(0.0, false, none);
        run(-1.0, true, negative);
        CHECK(zero.dual == 0 && zero.primal == 0, "fraction 0 repaired (seed %llu)", (unsigned long long)seed);
        CHECK(huge.dual == none.dual && huge.cold == none.cold && huge.warm == none.warm && huge.policy == 0,
              "huge fraction differs from ungated (seed %llu)", (unsigned long long)seed);
        CHECK(negative.dual == none.dual && negative.cold == none.cold && negative.policy == none.policy,
              "negative fraction differs from ungated (seed %llu)", (unsigned long long)seed);
        CHECK(zero.policy + zero.warm >= none.dual, "fraction 0: fewer gated solves than ungated repairs");
    }
}

static void count_limit_still_wins() {
    long policy = 0;
    for (int k = 0; k < 30; k++) {
        chain(0xC0DE + 13 * k, 45, 200, 10,
              [&](Graph<int64_t>& g) {
                  g.set_warm_violation_limit(0);
                  g.set_warm_violation_mass_fraction(1e300);
              },
              [&](Graph<int64_t>& g, Counters c0, Counters c1, int) {
                  CHECK(c1.dual == c0.dual && c1.primal == c0.primal, "limit 0 with a huge fraction repaired");
                  policy += c1.policy - c0.policy;
              });
    }
    CHECK(policy > 0, "limit 0 never forced a policy cold start");
}

static void measurement_is_the_violation() {
    // A solver at limit 0 measures the violation and goes cold; the same
    // instance chain with a huge fraction measures it and repairs.  Both see
    // the identical patched basis (same history up to the measurement), so
    // the reported numbers must agree on every step where the patch failed.
    for (int k = 0; k < 30; k++) {
        const uint64_t seed = 0xD00D + 7 * k;
        std::vector<std::pair<int, double>> a, b;
        auto record = [&](std::vector<std::pair<int, double>>& out, long limit, double fraction) {
            chain(seed, 40, 170, 2,
                  [&](Graph<int64_t>& g) {
                      g.set_warm_violation_limit(limit);
                      g.set_warm_violation_mass_fraction(fraction);
                  },
                  [&](Graph<int64_t>& g, Counters c0, Counters c1, int) {
                      if (c1.warm == c0.warm)
                          out.emplace_back(g.last_violation_count(), g.last_violation_mass());
                  });
        };
        record(a, 0, -1.0);
        record(b, -1, 1e300);
        CHECK(a == b, "seed %llu: measurements differ between policies", (unsigned long long)seed);
    }
}

int main() {
    section("decision matches the mass rule", decision_matches_rule);
    section("extremes and disabled gate", extremes_and_disabled);
    section("count limit still forces cold", count_limit_still_wins);
    section("reported violation is policy independent", measurement_is_the_violation);
    return finish("test_warm_mass_gate");
}
