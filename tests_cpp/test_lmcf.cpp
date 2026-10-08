// test_lmcf.cpp
// -------------------------------------------------------------------------
// Suite for the stateless C++ API in lmcf.hpp (lmcf, lmcf_cycle_canceling,
// lmcf_cost_scaling, lmcf_capacity_scaling, lmcf_impl) and for basics.hpp.
// Optima are checked against CapacityScaling on a ListDigraph, flows and
// potentials against the full certificate of mcf_oracle.h.  Covers: every
// solver at every value width the Python bindings expose, arbitrary edge
// order (the internal sort and the mapping of flows back to the caller's
// order), the wide cost type on narrow inputs whose optimum overflows them,
// lower bounds, GEQ supply, potentials through lmcf_impl for every solver,
// the whole input-validation contract, and the empty problem.
//
// Build:
//   g++ -I$(python -m pylmcf --include) -std=c++20 -O2 tests_cpp/test_lmcf.cpp -o /tmp/tl && /tmp/tl
// -------------------------------------------------------------------------
#include "mcf_oracle.h"

#include <pylmcf/basics.hpp>
#include <pylmcf/lmcf.hpp>

#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace mcf_test;

enum class Solver { NS, CC, CS, CAP };
static const char* solver_name(Solver s) {
    return s == Solver::NS ? "lmcf" : s == Solver::CC ? "cycle_canceling"
         : s == Solver::CS ? "cost_scaling" : "capacity_scaling";
}

template <typename T, typename U>
static std::vector<T> narrow(const std::vector<U>& v) {
    return std::vector<T>(v.begin(), v.end());
}

// The public entry points, by solver, with or without minimums.  The solver
// is a template parameter: LEMON's CapacityScaling does not even compile for
// int8/int16 (std::min on promoted operands), so narrow widths must only
// instantiate the solvers that support them.
template <Solver S, typename T>
static LmcfCost call_public(std::vector<T>& sup, std::vector<T>& st, std::vector<T>& en,
                            std::vector<T>& cap, std::vector<T>* mins, std::vector<T>& cost,
                            std::vector<T>& out) {
    if (mins) {
        if constexpr (S == Solver::NS) return lmcf<T>(sup, st, en, cap, *mins, cost, out);
        if constexpr (S == Solver::CC) return lmcf_cycle_canceling<T>(sup, st, en, cap, *mins, cost, out);
        if constexpr (S == Solver::CS) return lmcf_cost_scaling<T>(sup, st, en, cap, *mins, cost, out);
        if constexpr (S == Solver::CAP) return lmcf_capacity_scaling<T>(sup, st, en, cap, *mins, cost, out);
    }
    if constexpr (S == Solver::NS) return lmcf<T>(sup, st, en, cap, cost, out);
    if constexpr (S == Solver::CC) return lmcf_cycle_canceling<T>(sup, st, en, cap, cost, out);
    if constexpr (S == Solver::CS) return lmcf_cost_scaling<T>(sup, st, en, cap, cost, out);
    if constexpr (S == Solver::CAP) return lmcf_capacity_scaling<T>(sup, st, en, cap, cost, out);
}

// lmcf_impl directly, which is the only way to get potentials from C++.
template <Solver S, typename T>
static LmcfCost call_impl(std::vector<T>& sup, std::vector<T>& st, std::vector<T>& en,
                          std::vector<T>& cap, std::vector<T>& mins, std::vector<T>& cost,
                          std::vector<T>& out, std::vector<LmcfCost>& pot) {
    if constexpr (S == Solver::NS) return lmcf_impl<lemon::NetworkSimplex, T, true>(sup, st, en, cap, mins, cost, out, pot);
    if constexpr (S == Solver::CC) return lmcf_impl<lemon::CycleCanceling, T>(sup, st, en, cap, mins, cost, out, pot);
    if constexpr (S == Solver::CS) return lmcf_impl<lemon::CostScaling, T>(sup, st, en, cap, mins, cost, out, pot);
    if constexpr (S == Solver::CAP) return lmcf_impl<lemon::CapacityScaling, T>(sup, st, en, cap, mins, cost, out, pot);
}

// Shuffle edge order: the functional API accepts any order and must map flows
// back to it.  Returns the shuffled instance (unsorted; the oracle and the
// certificate do not need sorted edges).
static Instance shuffled(Rng& rng, const Instance& in) {
    std::vector<size_t> perm(in.starts.size());
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    Instance out = in;
    for (size_t j = 0; j < perm.size(); j++) {
        out.starts[j] = in.starts[perm[j]];
        out.ends[j] = in.ends[perm[j]];
        out.caps[j] = in.caps[perm[j]];
        out.costs[j] = in.costs[perm[j]];
        out.mins[j] = in.mins[perm[j]];
    }
    return out;
}

template <Solver s, typename T>
static void run_random(Rng& rng, int iters, int max_n, int64_t max_cost, const char* tname) {
    for (int iter = 0; iter < iters; iter++) {
        const bool mins = iter % 2;
        const Supply st = iter % 3 == 2 ? Supply::GEQ : Supply::EQ;
        const int n = static_cast<int>(rint(rng, 2, max_n));
        const int m = static_cast<int>(rint(rng, 1, 2 * n));
        Instance in = random_instance(rng, n, m, mins, st, max_cost);
        if (iter % 4) in = shuffled(rng, in);
        const std::string ctx = std::string(solver_name(s)) + "<" + tname + "> " + num(iter);
        const int64_t want = oracle_cost(in);

        auto sup = narrow<T>(in.supply), sts = narrow<T>(in.starts), ens = narrow<T>(in.ends);
        auto cap = narrow<T>(in.caps), cost = narrow<T>(in.costs), mn = narrow<T>(in.mins);
        std::vector<T> out(m);
        try {
            const LmcfCost got = call_public<s, T>(sup, sts, ens, cap, mins ? &mn : nullptr, cost, out);
            CHECK(got == want, "%s: cost %lld, oracle %lld", ctx.c_str(), (long long)got, (long long)want);
            int64_t sum = 0;
            for (int e = 0; e < m; e++) {
                CHECK(out[e] >= in.mins[e] && out[e] <= in.caps[e], "%s: edge %d out of bounds", ctx.c_str(), e);
                sum += in.costs[e] * static_cast<int64_t>(out[e]);
            }
            CHECK(sum == got, "%s: flows (in caller order) cost %lld != reported %lld", ctx.c_str(),
                  (long long)sum, (long long)got);

            // Potentials through lmcf_impl: same optimum, full certificate.
            std::vector<T> out2(m), mn_or_empty = mins ? mn : std::vector<T>{};
            std::vector<LmcfCost> pot(n);
            const LmcfCost got2 = call_impl<s, T>(sup, sts, ens, cap, mn_or_empty, cost, out2, pot);
            CHECK(got2 == want, "%s: impl cost %lld, oracle %lld", ctx.c_str(), (long long)got2, (long long)want);
            certify(in, std::vector<int64_t>(out2.begin(), out2.end()), std::vector<int64_t>(pot.begin(), pot.end()),
                    got2, ctx + " impl");
        } catch (const std::exception& ex) {
            CHECK(false, "%s: threw '%s' (oracle %lld)", ctx.c_str(), ex.what(), (long long)want);
        }
    }
}

template <Solver S>
static void random_wide(Rng& rng) {
    run_random<S, int64_t>(rng, 150, 25, 50, "int64");
    run_random<S, int32_t>(rng, 150, 25, 50, "int32");
}

// Narrow widths (bound in Python for these two solvers only): small instances
// whose values fit, but whose optimum, accumulated in the wide LmcfCost,
// routinely overflows the narrow type.
template <Solver S>
static void random_narrow(Rng& rng) {
    run_random<S, int16_t>(rng, 150, 8, 100, "int16");
    run_random<S, int8_t>(rng, 150, 4, 100, "int8");  // |supply| <= 8 edges x 15 + 6 < 128
}

static void test_random(Rng& rng) {
    random_wide<Solver::NS>(rng);
    random_wide<Solver::CC>(rng);
    random_wide<Solver::CS>(rng);
    random_wide<Solver::CAP>(rng);
    random_narrow<Solver::NS>(rng);
    random_narrow<Solver::CC>(rng);
}

static void test_narrow_cost_overflow() {
    // 100 units over an edge of cost 100: optimum 10000, far outside int8.
    auto one = [](auto solver_tag) {
        constexpr Solver s = decltype(solver_tag)::value;
        std::vector<int8_t> sup{100, -100}, st{0}, en{1}, cap{100}, cost{100}, out(1);
        const LmcfCost got = call_public<s, int8_t>(sup, st, en, cap, nullptr, cost, out);
        CHECK(got == 10000 && out[0] == 100, "%s: int8 optimum %lld", solver_name(s), (long long)got);
    };
    one(std::integral_constant<Solver, Solver::NS>{});
    one(std::integral_constant<Solver, Solver::CC>{});
}

template <Solver s>
static void validation_for() {
    using V = std::vector<int64_t>;
    {
        const char* nm = solver_name(s);
        auto run = [](V sup, V st, V en, V cap, V mins, V cost, size_t out_n, size_t pot_n = 0) {
            V out(out_n);
            std::vector<LmcfCost> pot(pot_n);
            return call_impl<s, int64_t>(sup, st, en, cap, mins, cost, out, pot);
        };
        const V sup{5, 0, -5}, st{0, 0, 1}, en{1, 2, 2}, cap{3, 3, 5}, cost{1, 3, 5};
        CHECK(run(sup, st, en, cap, {}, cost, 3) == 21, "%s: baseline", nm);
        CHECK_THROWS(std::invalid_argument, run(sup, {0, 0}, en, cap, {}, cost, 3), "%s: starts len", nm);
        CHECK_THROWS(std::invalid_argument, run(sup, st, {1, 2}, cap, {}, cost, 3), "%s: ends len", nm);
        CHECK_THROWS(std::invalid_argument, run(sup, st, en, {3, 3}, {}, cost, 3), "%s: caps len", nm);
        CHECK_THROWS(std::invalid_argument, run(sup, st, en, cap, {0, 0}, cost, 3), "%s: mins len", nm);
        CHECK_THROWS(std::invalid_argument, run(sup, st, en, cap, {}, {1, 3}, 3), "%s: costs len", nm);
        CHECK_THROWS(std::invalid_argument, run(sup, st, en, cap, {}, cost, 2), "%s: result len", nm);
        CHECK_THROWS(std::invalid_argument, run(sup, st, en, cap, {}, cost, 3, 2), "%s: potentials len", nm);
        CHECK_THROWS(std::invalid_argument, run(sup, {-1, 0, 1}, en, cap, {}, cost, 3), "%s: negative id", nm);
        CHECK_THROWS(std::invalid_argument, run(sup, st, {1, 2, 3}, cap, {}, cost, 3), "%s: id range", nm);
        CHECK_THROWS(std::invalid_argument, run(sup, st, en, {3, -3, 5}, {}, cost, 3), "%s: negative cap", nm);
        CHECK_THROWS(std::invalid_argument, run(sup, st, en, cap, {0, -1, 0}, cost, 3), "%s: negative min", nm);
        CHECK_THROWS(std::invalid_argument, run(sup, st, en, cap, {4, 0, 0}, cost, 3), "%s: min > cap", nm);
        CHECK_THROWS(std::runtime_error, run({9, 0, -9}, st, en, cap, {}, cost, 3), "%s: capacity infeasible", nm);
        // GEQ semantics: excess demand is fine, excess supply is not.
        CHECK(run({3, 0, -9}, st, en, cap, {}, cost, 3) == 9, "%s: geq excess demand", nm);
        CHECK_THROWS(std::runtime_error, run({9, 0, -3}, st, en, {9, 9, 9}, {}, cost, 3), "%s: geq excess supply", nm);
        // The empty problem is trivially optimal.
        CHECK(run({}, {}, {}, {}, {}, {}, 0, 0) == 0, "%s: empty problem", nm);
        // Nodes without edges, zero supply.
        CHECK(run({0, 0}, {}, {}, {}, {}, {}, 0, 2) == 0, "%s: edgeless", nm);
        // Every LEMON solver negates supplies: min() is rejected.  (min() + 1
        // is accepted but not yet safe everywhere -- CycleCanceling overflows
        // on it internally; see the supply-sum range checks still to come.)
        const int64_t lo = std::numeric_limits<int64_t>::min();
        CHECK_THROWS(std::invalid_argument, run({0, 0, lo}, st, en, cap, {}, cost, 3), "%s: min supply", nm);
    }
}

// The three solvers that accept negative costs: a negative-cost cycle at
// LEMON's "infinite" capacity (the type's max) is unbounded and must say so;
// at a finite capacity it is an ordinary optimum.
template <Solver s>
static void unbounded_for() {
    using V = std::vector<int64_t>;
    const int64_t inf = std::numeric_limits<int64_t>::max();
    V sup{0, 0}, st{0, 1}, en{1, 0}, cost{-1, 0}, none, out(2);
    V cap{inf, inf};
    std::vector<LmcfCost> no_pot;
    bool said_unbounded = false;
    try {
        call_impl<s, int64_t>(sup, st, en, cap, none, cost, out, no_pot);
    } catch (const std::runtime_error& ex) {
        said_unbounded = std::string(ex.what()).find("UNBOUNDED") != std::string::npos;
    }
    CHECK(said_unbounded, "%s: negative cycle at infinite capacity not reported UNBOUNDED", solver_name(s));
    V cap5{5, 5};
    std::vector<LmcfCost> pot;
    const LmcfCost got = call_impl<s, int64_t>(sup, st, en, cap5, none, cost, out, pot);
    CHECK(got == -5 && out[0] == 5 && out[1] == 5, "%s: finite negative cycle optimum %lld",
          solver_name(s), (long long)got);
}

static void test_validation() {
    using V = std::vector<int64_t>;
    validation_for<Solver::NS>();
    validation_for<Solver::CC>();
    validation_for<Solver::CS>();
    validation_for<Solver::CAP>();
    // Negative costs: rejected by network simplex only.
    V sup{5, 0, -5}, st{0, 0, 1}, en{1, 2, 2}, cap{3, 3, 5}, neg{1, -3, 5}, out(3);
    CHECK_THROWS(std::invalid_argument, lmcf<int64_t>(sup, st, en, cap, neg, out), "lmcf negative cost");
    CHECK(lmcf_cycle_canceling<int64_t>(sup, st, en, cap, neg, out) == 3 * -3 + 2 * 6,
          "cycle canceling accepts negative costs");
    unbounded_for<Solver::CC>();
    unbounded_for<Solver::CS>();
    unbounded_for<Solver::CAP>();
    // Network simplex's artificial arcs cost 2^62: a real cost of 2^62 used
    // to be reported INFEASIBLE on a feasible problem; it is now rejected,
    // and 2^62 - 1, the largest accepted cost, solves.
    {
        const int64_t big = std::numeric_limits<int64_t>::max() / 2;
        V s1{1, -1}, a{0}, b{1}, c1{1}, o1(1);
        V too_big{big + 1}, largest{big};
        CHECK_THROWS(std::invalid_argument, lmcf<int64_t>(s1, a, b, c1, too_big, o1), "lmcf cost 2^62");
        CHECK(lmcf<int64_t>(s1, a, b, c1, largest, o1) == big && o1[0] == 1, "lmcf cost 2^62 - 1");
    }
    // min() + 1 under network simplex (an unmet GEQ demand: optimal at 0).
    {
        V s3{0, 0, std::numeric_limits<int64_t>::min() + 1}, o3(3);
        V st3{0, 0, 1}, en3{1, 2, 2}, cap3{3, 3, 5}, cost3{1, 3, 5};
        CHECK(lmcf<int64_t>(s3, st3, en3, cap3, cost3, o3) == 0, "lmcf min + 1 supply");
    }
    // The supply minimum at a narrow width (only NS and CC compile for int8).
    {
        std::vector<int8_t> s8{0, -128}, a8{0}, b8{1}, c8{1}, m8, k8{1}, o8(1);
        std::vector<LmcfCost> p8;
        CHECK_THROWS(std::invalid_argument,
                     (call_impl<Solver::NS, int8_t>(s8, a8, b8, c8, m8, k8, o8, p8)), "int8 NS -128 supply");
        CHECK_THROWS(std::invalid_argument,
                     (call_impl<Solver::CC, int8_t>(s8, a8, b8, c8, m8, k8, o8, p8)), "int8 CC -128 supply");
    }
}

static void test_basics() {
    const size_t max = static_cast<size_t>(std::numeric_limits<LEMON_INDEX>::max());
    assert_fits_lemon_index(0, "Node");
    assert_fits_lemon_index(max, "Node");
    CHECK_THROWS(std::overflow_error, assert_fits_lemon_index(max + 1, "Edge"), "INT_MAX + 1 must throw");
    try {
        assert_fits_lemon_index(max + 1, "Edge");
    } catch (const std::overflow_error& ex) {
        CHECK(std::string(ex.what()).find("Edge count") == 0, "message names the count: %s", ex.what());
    }

    // sorted_copy: for types without copy assignment (std::sort cannot
    // shuffle them in place).
    struct NoAssign {
        const int key;
        const int tag;
    };
    std::vector<NoAssign> v;
    Rng rng(7);
    for (int i = 0; i < 200; i++) v.push_back({static_cast<int>(rint(rng, 0, 50)), i});
    const auto s = sorted_copy(v, [](const NoAssign& a, const NoAssign& b) { return a.key < b.key; });
    CHECK(s.size() == v.size(), "sorted_copy size");
    std::vector<int> tags;
    for (size_t i = 0; i < s.size(); i++) {
        if (i) CHECK(s[i - 1].key <= s[i].key, "sorted_copy order at %zu", i);
        CHECK(v[s[i].tag].key == s[i].key, "sorted_copy element integrity");
        tags.push_back(s[i].tag);
    }
    std::sort(tags.begin(), tags.end());
    std::vector<int> all(v.size());
    std::iota(all.begin(), all.end(), 0);
    CHECK(tags == all, "sorted_copy is a permutation");
    CHECK(sorted_copy(std::vector<NoAssign>{}, [](const NoAssign&, const NoAssign&) { return false; }).empty(),
          "sorted_copy empty");
}

int main() {
    Rng rng(20261006);
    section("basics", [] { test_basics(); }, true);
    section("validation", [] { test_validation(); }, true);
    section("narrow cost overflow", [] { test_narrow_cost_overflow(); });
    section("random", [&] { test_random(rng); });
    return finish("test_lmcf");
}
