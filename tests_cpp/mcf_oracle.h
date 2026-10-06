// mcf_oracle.h — shared scaffolding for the tests_cpp suites of the public
// C++ API (test_graph.cpp, test_lmcf.cpp).  Not a test itself: run_all.sh
// builds only test_*.cpp.
//
// Everything here is independent of the code under test:
//   - random_instance(): feasible by construction (a witness flow defines the
//     supplies; capacities sit on or above it), optionally unbalanced in the
//     direction a supply type can absorb;
//   - oracle_cost(): the optimum by LEMON's CapacityScaling — a different
//     algorithm from the network simplex inside Graph/lmcf — on its own graph,
//     with LEQ handled by an equality-form reformulation through a hub node;
//   - certify(): primal feasibility, complementary slackness, the supply-type
//     sign rule and strong duality, for a (flows, potentials) pair;
//   - canonical_oracle(): the extremal optimal potentials by textbook
//     Bellman-Ford over the residual graph.
#ifndef PYLMCF_TESTS_MCF_ORACLE_H
#define PYLMCF_TESTS_MCF_ORACLE_H

#define LEMON_ONLY_TEMPLATES
#include <lemon/capacity_scaling.h>
#include <lemon/static_graph.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace mcf_test {

inline long checks = 0, fails = 0;

#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        ++mcf_test::checks;                                                \
        if (!(cond)) {                                                     \
            ++mcf_test::fails;                                             \
            if (mcf_test::fails <= 30) {                                   \
                std::printf("FAIL %s:%d: ", __FILE__, __LINE__);           \
                std::printf(__VA_ARGS__);                                  \
                std::printf("\n");                                         \
                std::fflush(stdout); /* survive a later hang/abort */      \
            }                                                              \
        }                                                                  \
    } while (0)

// Expect `expr` to throw exactly `Exc` (not a sibling or a base).
#define CHECK_THROWS(Exc, expr, ...)                                       \
    do {                                                                   \
        bool threw_ = false, other_ = false;                               \
        try { expr; } catch (const Exc&) { threw_ = true; }                \
        catch (...) { other_ = true; }                                     \
        CHECK(threw_ && !other_, __VA_ARGS__);                             \
    } while (0)

#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
#define MCF_TEST_UNDER_MSAN 1
#endif
#endif

// Sections that exercise error paths build exception messages with
// std::to_string and string concatenation, which run inside the
// uninstrumented libstdc++.so; MSan then reports its own blind spot as an
// uninitialised read.  Under MSan those sections are skipped (and say so);
// the ASan, debug-mode and UBSan lanes still run them.
#ifdef MCF_TEST_UNDER_MSAN
inline constexpr bool under_msan = true;
#else
inline constexpr bool under_msan = false;
#endif

// Run one test section; an exception escaping it is a reported failure, and
// the remaining sections still run.
template <typename F>
inline void section(const char* name, F&& body, bool builds_error_messages = false) {
    if (builds_error_messages && under_msan) {
        std::printf("section %s: skipped under MSan (error-message paths)\n", name);
        return;
    }
    try {
        body();
    } catch (const std::exception& ex) {
        CHECK(false, "section %s: uncaught exception '%s'", name, ex.what());
    } catch (...) {
        CHECK(false, "section %s: uncaught non-std exception", name);
    }
}

// Number to string without std::to_string, whose libstdc++ internals MSan
// cannot see (it then flags every label built from them).
inline std::string num(long long v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%lld", v);
    return std::string(buf);
}

// printf-style label, for the same reason as num(): std::string operator+
// runs partly in the uninstrumented libstdc++ (extern template basic_string).
template <typename... A>
inline std::string lbl(const char* fmt, A... args) {
    char buf[160];
    std::snprintf(buf, sizeof buf, fmt, args...);
    return std::string(buf);
}

inline int finish(const char* name) {
    std::printf("%s: checks=%ld fails=%ld\n", name, checks, fails);
    std::printf("RESULT: %s\n", fails ? "FAILED" : "PASSED");
    return fails ? 1 : 0;
}

enum class Supply { EQ, GEQ, LEQ };

inline const char* supply_name(Supply s) {
    return s == Supply::EQ ? "eq" : s == Supply::GEQ ? "geq" : "leq";
}

struct Instance {
    int n = 0;
    std::vector<int> starts, ends;          // sorted by (start, end)
    std::vector<int64_t> supply, caps, costs, mins;  // mins all-zero if unused
    bool has_mins = false;
    Supply type = Supply::EQ;
};

using Rng = std::mt19937_64;

inline int64_t rint(Rng& rng, int64_t lo, int64_t hi) {
    return std::uniform_int_distribution<int64_t>(lo, hi)(rng);
}

// Feasible instance with n nodes, m edges (no self-loops, parallel edges
// allowed), sorted by (start, end).  Unbalanced types shift supply in the
// direction they can absorb: GEQ adds demand, LEQ adds supply.
inline Instance random_instance(Rng& rng, int n, int m, bool with_mins,
                                Supply type = Supply::EQ, int64_t max_cost = 50) {
    Instance in;
    in.n = n;
    in.type = type;
    in.has_mins = with_mins;
    std::vector<std::pair<int, int>> arcs(m);
    for (auto& a : arcs) {
        a.first = static_cast<int>(rint(rng, 0, n - 1));
        a.second = static_cast<int>(rint(rng, 0, n - 2));
        if (a.second >= a.first) a.second++;
    }
    std::sort(arcs.begin(), arcs.end());
    in.supply.assign(n, 0);
    for (const auto& a : arcs) {
        const int64_t lo = with_mins ? rint(rng, 0, 3) : 0;
        const int64_t wit = lo + rint(rng, 0, 12);
        in.starts.push_back(a.first);
        in.ends.push_back(a.second);
        in.mins.push_back(lo);
        in.caps.push_back(wit + (rint(rng, 0, 2) ? rint(rng, 0, 18) : 0));
        in.costs.push_back(rint(rng, 0, max_cost));
        in.supply[a.first] += wit;
        in.supply[a.second] -= wit;
    }
    if (type != Supply::EQ) {
        for (int u = 0; u < n; u++) {
            const int64_t x = rint(rng, 0, 2) == 0 ? rint(rng, 0, 5) : 0;
            in.supply[u] += type == Supply::LEQ ? x : -x;
        }
        in.supply[rint(rng, 0, n - 1)] += type == Supply::LEQ ? 1 : -1;
    }
    return in;
}

// New witness on the same topology: new supplies and capacities, costs and
// minimums kept.  For warm-restart chains.
inline void redraw(Rng& rng, Instance& in) {
    std::fill(in.supply.begin(), in.supply.end(), 0);
    for (size_t e = 0; e < in.starts.size(); e++) {
        const int64_t wit = in.mins[e] + rint(rng, 0, 12);
        in.caps[e] = wit + (rint(rng, 0, 2) ? rint(rng, 0, 18) : 0);
        in.supply[in.starts[e]] += wit;
        in.supply[in.ends[e]] -= wit;
    }
}

// Optimal cost by CapacityScaling, or -1 if infeasible.  CapacityScaling has
// GEQ semantics, which covers EQ and GEQ; LEQ is put in equality form with a
// hub node draining unused supply through free, uncapacitated edges.  Built
// on a StaticDigraph (arcs sorted here; order does not matter for the cost):
// LEMON's ListDigraph wraps an unsigned counter on purpose, which the strict
// UBSan lane rejects, and its notifier trips MSan.
inline int64_t oracle_cost(const Instance& in) {
    struct A { int s, t; int64_t lo, up, cost; };
    std::vector<A> arcs;
    int64_t big = 1, total = 0;
    for (size_t e = 0; e < in.starts.size(); e++) {
        arcs.push_back({in.starts[e], in.ends[e], in.mins[e], in.caps[e], in.costs[e]});
        big += in.caps[e];
    }
    std::vector<int64_t> sup(in.supply);
    for (int u = 0; u < in.n; u++) {
        total += in.supply[u];
        big += in.supply[u] < 0 ? -in.supply[u] : in.supply[u];
    }
    int n = in.n;
    if (in.type == Supply::LEQ) {
        if (total < 0) return -1;
        sup.push_back(-total);
        for (int u = 0; u < in.n; u++) arcs.push_back({u, n, 0, big, 0});
        n++;
    } else if (in.type == Supply::EQ && total != 0) {
        return -1;
    }
    std::sort(arcs.begin(), arcs.end(), [](const A& x, const A& y) {
        return x.s != y.s ? x.s < y.s : x.t < y.t;
    });
    std::vector<std::pair<int, int>> st;
    for (const auto& a : arcs) st.emplace_back(a.s, a.t);
    lemon::StaticDigraph g;
    g.build(n, st.begin(), st.end());
    lemon::StaticDigraph::ArcMap<int64_t> lo(g), up(g), cost(g);
    lemon::StaticDigraph::NodeMap<int64_t> supm(g);
    for (size_t j = 0; j < arcs.size(); j++) {
        const auto a = g.arcFromId(static_cast<int>(j));
        lo[a] = arcs[j].lo;
        up[a] = arcs[j].up;
        cost[a] = arcs[j].cost;
    }
    for (int u = 0; u < n; u++) supm[g.nodeFromId(u)] = sup[u];
    lemon::CapacityScaling<lemon::StaticDigraph, int64_t, int64_t> cs(g);
    cs.lowerMap(lo).upperMap(up).costMap(cost).supplyMap(supm);
    if (cs.run() != decltype(cs)::OPTIMAL) return -1;
    return cs.totalCost();
}

// Extremal optimal potentials for `flows`: largest pi <= 0 (EQ/GEQ) or
// smallest pi >= 0 (LEQ), by Bellman-Ford over the residual graph.
inline std::vector<int64_t> canonical_oracle(const Instance& in, const std::vector<int64_t>& flows) {
    struct A { int u, v; int64_t w; };
    std::vector<A> arcs;
    for (size_t e = 0; e < in.starts.size(); e++) {
        if (flows[e] < in.caps[e]) arcs.push_back({in.starts[e], in.ends[e], in.costs[e]});
        if (flows[e] > in.mins[e]) arcs.push_back({in.ends[e], in.starts[e], -in.costs[e]});
    }
    const bool leq = in.type == Supply::LEQ;
    if (leq) for (auto& a : arcs) std::swap(a.u, a.v);
    std::vector<int64_t> d(in.n, 0);
    for (int it = 0; it <= in.n; it++) {
        bool ch = false;
        for (const auto& a : arcs)
            if (d[a.u] + a.w < d[a.v]) { d[a.v] = d[a.u] + a.w; ch = true; }
        if (!ch) break;
    }
    if (leq) for (auto& x : d) x = -x;
    return d;
}

// Full certificate.  Returns true iff every check passed.
inline bool certify(const Instance& in, const std::vector<int64_t>& flows,
                    const std::vector<int64_t>& pi, int64_t total_cost, const std::string& ctx) {
    const long before = fails;
    const size_t m = in.starts.size();
    CHECK(flows.size() == m && pi.size() == static_cast<size_t>(in.n), "%s: sizes", ctx.c_str());
    if (flows.size() != m || pi.size() != static_cast<size_t>(in.n)) return false;
    std::vector<int64_t> net(in.n, 0);
    int64_t cost = 0, dual = 0;
    for (size_t e = 0; e < m; e++) {
        CHECK(flows[e] >= in.mins[e] && flows[e] <= in.caps[e], "%s: edge %zu out of bounds", ctx.c_str(), e);
        net[in.starts[e]] += flows[e];
        net[in.ends[e]] -= flows[e];
        cost += in.costs[e] * flows[e];
        const int64_t rc = in.costs[e] + pi[in.starts[e]] - pi[in.ends[e]];
        CHECK(!(rc > 0 && flows[e] != in.mins[e]), "%s: edge %zu rc>0 above minimum", ctx.c_str(), e);
        CHECK(!(rc < 0 && flows[e] != in.caps[e]), "%s: edge %zu rc<0 below capacity", ctx.c_str(), e);
        dual += rc < 0 ? rc * in.caps[e] : rc * in.mins[e];
    }
    for (int u = 0; u < in.n; u++) {
        const int64_t slack = net[u] - in.supply[u];
        if (in.type == Supply::EQ) CHECK(slack == 0, "%s: node %d unbalanced", ctx.c_str(), u);
        if (in.type == Supply::GEQ) CHECK(slack >= 0 && pi[u] <= 0, "%s: node %d geq", ctx.c_str(), u);
        if (in.type == Supply::LEQ) CHECK(slack <= 0 && pi[u] >= 0, "%s: node %d leq", ctx.c_str(), u);
        CHECK(slack == 0 || pi[u] == 0, "%s: node %d slack with pi != 0", ctx.c_str(), u);
        dual -= in.supply[u] * pi[u];
    }
    CHECK(cost == total_cost, "%s: reported cost %lld != sum cost*flow %lld", ctx.c_str(),
          (long long)total_cost, (long long)cost);
    CHECK(dual == total_cost, "%s: dual %lld != primal %lld", ctx.c_str(), (long long)dual,
          (long long)total_cost);
    CHECK(pi == canonical_oracle(in, flows), "%s: potentials not canonical", ctx.c_str());
    return fails == before;
}

}  // namespace mcf_test

#endif  // PYLMCF_TESTS_MCF_ORACLE_H
