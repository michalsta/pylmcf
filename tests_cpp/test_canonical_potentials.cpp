// test_canonical_potentials.cpp
// -------------------------------------------------------------------------
// Oracle for canonical_potentials(): random feasible instances (EQ, GEQ and
// LEQ supply, with and without lower bounds) solved by LEMON's NetworkSimplex.
// For each solve the canonical potentials are computed twice:
//   (a) with LEMON's raw potentials as Johnson heights (the production path);
//   (b) with garbage heights, forcing the Bellman-Ford fallback;
// and both must equal (c) an independent textbook Bellman-Ford over the
// residual graph written out here.  They must also certify the solution
// (complementary slackness + supply-type sign rule).  Finally, a deliberately
// non-optimal flow must be rejected with std::logic_error.
//
// The Python suite cannot reach the fallback path (LEMON's network simplex
// always hands over exact heights), which is why this file exists.
//
// Build:
//   g++ -I$(python -m pylmcf --include) -std=c++20 -O2 tests_cpp/test_canonical_potentials.cpp -o /tmp/tcp && /tmp/tcp
// -------------------------------------------------------------------------
#define LEMON_ONLY_TEMPLATES
#include <lemon/static_graph.h>
#include <lemon/network_simplex.h>
#include <pylmcf/canonical_potentials.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

using V = int64_t;
using Graph = lemon::StaticDigraph;
using NS = lemon::NetworkSimplex<Graph, V, V>;

static long checks = 0, fails = 0;
#define CHECK(cond, ...)                                    \
    do {                                                    \
        ++checks;                                           \
        if (!(cond)) {                                      \
            ++fails;                                        \
            if (fails <= 20) {                              \
                std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
                std::printf(__VA_ARGS__);                   \
                std::printf("\n");                          \
            }                                               \
        }                                                   \
    } while (0)

// Textbook Bellman-Ford: extremal potentials, deliberately sharing no code
// with canonical_potentials.hpp.
static std::vector<V> oracle(int n, const std::vector<LEMON_INDEX>& s,
                             const std::vector<LEMON_INDEX>& t,
                             const std::vector<V>& c, const std::vector<V>& cap,
                             const std::vector<V>& lo, const std::vector<V>& f,
                             bool leq) {
    struct A { int u, v; V w; };
    std::vector<A> arcs;
    for (size_t e = 0; e < s.size(); e++) {
        if (f[e] < cap[e]) arcs.push_back({s[e], t[e], c[e]});
        if (f[e] > lo[e]) arcs.push_back({t[e], s[e], -c[e]});
    }
    if (leq) for (auto& a : arcs) std::swap(a.u, a.v);
    std::vector<V> d(n, 0);
    for (int it = 0; it <= n; it++) {
        bool ch = false;
        for (const auto& a : arcs)
            if (d[a.u] + a.w < d[a.v]) { d[a.v] = d[a.u] + a.w; ch = true; }
        if (!ch) break;
    }
    if (leq) for (auto& x : d) x = -x;
    return d;
}

int main() {
    std::mt19937_64 rng(20261005);
    auto rint = [&](V lo, V hi) { return std::uniform_int_distribution<V>(lo, hi)(rng); };
    int instances = 0, nonopt_tests = 0;

    for (int iter = 0; iter < 3000; iter++) {
        const int n = static_cast<int>(rint(2, 30));
        const int m = static_cast<int>(rint(1, 4 * n));
        const bool with_lo = iter % 2;
        const int stype = iter % 3;   // 0 EQ, 1 GEQ (sum<0), 2 LEQ (sum>0)

        std::vector<std::pair<LEMON_INDEX, LEMON_INDEX>> arcs(m);
        for (auto& a : arcs) {
            a.first = static_cast<LEMON_INDEX>(rint(0, n - 1));
            a.second = static_cast<LEMON_INDEX>(rint(0, n - 2));
            if (a.second >= a.first) a.second++;
        }
        std::sort(arcs.begin(), arcs.end());
        std::vector<LEMON_INDEX> s(m), t(m);
        std::vector<V> c(m), cap(m), lo(m), wit(m), sup(n, 0);
        for (int e = 0; e < m; e++) {
            s[e] = arcs[e].first;
            t[e] = arcs[e].second;
            c[e] = rint(0, 50);
            lo[e] = with_lo ? rint(0, 3) : 0;
            wit[e] = lo[e] + rint(0, 12);
            cap[e] = wit[e] + (rint(0, 2) ? rint(0, 18) : 0);
            sup[s[e]] += wit[e];
            sup[t[e]] -= wit[e];
        }
        if (stype) {
            for (int u = 0; u < n; u++) {
                const V x = rint(0, 2) == 0 ? rint(0, 5) : 0;
                sup[u] += stype == 2 ? x : -x;
            }
            sup[rint(0, n - 1)] += stype == 2 ? 1 : -1;
        }

        Graph g;
        g.build(n, arcs.begin(), arcs.end());
        Graph::ArcMap<V> cm(g), um(g), lm(g);
        Graph::NodeMap<V> sm(g);
        for (int e = 0; e < m; e++) {
            cm[g.arcFromId(e)] = c[e];
            um[g.arcFromId(e)] = cap[e];
            lm[g.arcFromId(e)] = lo[e];
        }
        for (int u = 0; u < n; u++) sm[g.nodeFromId(u)] = sup[u];
        NS ns(g);
        ns.costMap(cm).upperMap(um).supplyMap(sm);
        if (with_lo) ns.lowerMap(lm);
        const bool leq = stype == 2;
        if (leq) ns.supplyType(NS::LEQ);
        if (ns.run() != NS::OPTIMAL) { CHECK(false, "iter %d not OPTIMAL", iter); continue; }
        instances++;

        std::vector<V> f(m), raw(n), garbage(n), a(n), b(n);
        for (int e = 0; e < m; e++) f[e] = ns.flow(g.arcFromId(e));
        for (int u = 0; u < n; u++) {
            raw[u] = ns.potential(g.nodeFromId(u));
            garbage[u] = rint(-1000, 1000);
        }
        canonical_potentials<V, V>(n, s, t, c, cap, lo, f, raw, leq, a);
        canonical_potentials<V, V>(n, s, t, c, cap, lo, f, garbage, leq, b);
        const auto want = oracle(n, s, t, c, cap, lo, f, leq);
        CHECK(a == want, "iter %d: Johnson path differs from oracle", iter);
        CHECK(b == want, "iter %d: Bellman-Ford fallback differs from oracle", iter);

        // Certificate: CS on edges, sign rule and slack on nodes.
        std::vector<V> net(n, 0);
        for (int e = 0; e < m; e++) {
            net[s[e]] += f[e];
            net[t[e]] -= f[e];
            const V rc = c[e] + a[s[e]] - a[t[e]];
            CHECK(!(rc > 0 && f[e] != lo[e]), "iter %d edge %d: rc>0 above minimum", iter, e);
            CHECK(!(rc < 0 && f[e] != cap[e]), "iter %d edge %d: rc<0 below cap", iter, e);
        }
        for (int u = 0; u < n; u++) {
            CHECK(leq ? a[u] >= 0 : a[u] <= 0, "iter %d node %d: sign", iter, u);
            CHECK(net[u] == sup[u] || a[u] == 0, "iter %d node %d: slack with pi != 0", iter, u);
        }

        // A strictly worse feasible flow: one more unit around a 2-cycle
        // u->v->u of positive cost, if the instance has one with room.
        for (int e = 0; e < m; e++) {
            if (f[e] < cap[e] && c[e] > 0) {
                // Find an antiparallel edge with room to absorb the unit.
                for (int r = 0; r < m; r++) {
                    if (s[r] == t[e] && t[r] == s[e] && f[r] < cap[r]) {
                        auto worse = f;
                        worse[e] += 1;
                        worse[r] += 1;
                        nonopt_tests++;
                        bool threw = false;
                        try {
                            canonical_potentials<V, V>(n, s, t, c, cap, lo, worse, raw, leq, b);
                        } catch (const std::logic_error&) {
                            threw = true;
                        }
                        CHECK(threw, "iter %d: non-optimal flow accepted", iter);
                        e = m;
                        break;
                    }
                }
            }
        }
    }

    CHECK(nonopt_tests > 100, "non-optimal rejection exercised only %d times", nonopt_tests);
    std::printf("checks=%ld instances=%d nonopt=%d fails=%ld\n", checks, instances, nonopt_tests, fails);
    std::printf("RESULT: %s\n", fails ? "FAILED" : "PASSED");
    return fails ? 1 : 0;
}
