#ifndef PYLMCF_CANONICAL_POTENTIALS_HPP
#define PYLMCF_CANONICAL_POTENTIALS_HPP

#include <algorithm>
#include <functional>
#include <queue>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "basics.hpp"

// Canonical node potentials for an optimal flow.
//
// LEMON's network simplex reports the potentials of its final spanning tree.
// Those are dual-optimal but not canonical: a node whose tree path to the
// artificial root still runs through a zero-flow artificial arc inherits
// that arc's ART_COST (2^62 for int64 costs).  This happens routinely under
// GEQ supply constraints with nonzero total supply (a tight supply node whose
// out-edges are all saturated), and yields potentials around -2^62 — valid,
// but useless to a caller and an overflow hazard in any arithmetic on them.
//
// For a fixed optimal flow, the dual-optimal potentials are exactly those
// with nonnegative reduced cost rc = cost + pi[u] - pi[v] on every residual
// arc, plus the supply-type sign condition (GEQ: pi <= 0, LEQ: pi >= 0, zero
// on nodes whose constraint has slack — implied at optimality by the sign
// condition, see below).  This returns the extremal member of that set:
//   GEQ/EQ: the pointwise-largest pi with pi <= 0, i.e. pi[v] = shortest
//           residual-path distance to v from a virtual source with a 0-cost
//           arc into every node;
//   LEQ:    the pointwise-smallest pi with pi >= 0, i.e. pi[v] = minus the
//           shortest residual-path distance from v to a virtual sink.
// Its magnitudes are bounded by residual path costs.  Slack nodes come out
// as 0 automatically: a slack node with nonzero canonical potential would
// close a negative cycle through the virtual node, contradicting optimality.
//
// The input potentials (from the solver) serve only as Johnson reweighting:
// optimal potentials make every residual reduced cost nonnegative, so a
// single Dijkstra suffices.  Should they not (a solver whose potentials are
// only approximately dual-feasible), heights are recomputed by Bellman-Ford
// instead; only a negative residual cycle — the flows themselves are not
// optimal — is reported, as std::logic_error.
template <typename V, typename C>
void canonical_potentials(
    LEMON_INDEX no_nodes,
    std::span<const LEMON_INDEX> starts,
    std::span<const LEMON_INDEX> ends,
    std::span<const C> costs,
    std::span<const V> caps,
    std::span<const V> minimums,   // empty => all zero
    std::span<const V> flows,
    std::span<const C> solver_pi,
    bool leq,
    std::span<C> out)
{
    const size_t m = starts.size();
    const size_t n = static_cast<size_t>(no_nodes);
    // Residual arcs as (tail, head, weight).  Under LEQ the graph is
    // reversed and the potentials negated, which turns "distance to sink"
    // into the same single-source problem.
    std::vector<size_t> deg(n + 1, 0);
    auto for_each_residual = [&](auto&& emit) {
        for (size_t e = 0; e < m; e++) {
            const V lo = minimums.empty() ? V(0) : minimums[e];
            if (flows[e] < caps[e]) emit(starts[e], ends[e], costs[e]);
            if (flows[e] > lo) emit(ends[e], starts[e], -costs[e]);
        }
    };
    for_each_residual([&](LEMON_INDEX u, LEMON_INDEX v, C) {
        deg[leq ? v : u]++;
    });
    std::vector<size_t> first(n + 1, 0);
    for (size_t u = 0; u < n; u++) first[u + 1] = first[u] + deg[u];
    std::vector<std::pair<LEMON_INDEX, C>> adj(first[n]);
    std::vector<size_t> fill(first.begin(), first.end() - 1);
    for_each_residual([&](LEMON_INDEX u, LEMON_INDEX v, C w) {
        if (leq) adj[fill[v]++] = {u, w};
        else     adj[fill[u]++] = {v, w};
    });

    // Johnson heights: h = pi (GEQ) or -pi (LEQ); every arc in adj then has
    // reduced weight w + h[tail] - h[head] >= 0.
    std::vector<C> h(n);
    for (size_t u = 0; u < n; u++) h[u] = leq ? -solver_pi[u] : solver_pi[u];
    bool heights_ok = true;
    for (size_t u = 0; u < n && heights_ok; u++)
        for (size_t k = first[u]; k < first[u + 1]; k++)
            if (adj[k].second + h[u] - h[adj[k].first] < 0) { heights_ok = false; break; }
    if (!heights_ok) {
        // Bellman-Ford from the virtual source (0-cost arc into every node).
        std::fill(h.begin(), h.end(), C(0));
        bool changed = true;
        for (size_t round = 0; changed; round++) {
            if (round > n)
                throw std::logic_error("canonical_potentials: negative residual cycle, flows are not optimal");
            changed = false;
            for (size_t u = 0; u < n; u++)
                for (size_t k = first[u]; k < first[u + 1]; k++) {
                    const auto [v, w] = adj[k];
                    if (h[u] + w < h[v]) { h[v] = h[u] + w; changed = true; }
                }
        }
    }
    C h_src = 0;
    for (size_t u = 0; u < n; u++) if (u == 0 || h[u] > h_src) h_src = h[u];

    // Multi-source Dijkstra on reduced weights; the virtual source reaches
    // every node at reduced distance h_src - h[u] >= 0.
    std::vector<C> dist(n);
    std::vector<char> done(n, 0);
    using Item = std::pair<C, LEMON_INDEX>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;
    for (size_t u = 0; u < n; u++) {
        dist[u] = h_src - h[u];
        pq.emplace(dist[u], static_cast<LEMON_INDEX>(u));
    }
    while (!pq.empty()) {
        const auto [d, u] = pq.top();
        pq.pop();
        if (done[u] || d != dist[u]) continue;
        done[u] = 1;
        for (size_t k = first[u]; k < first[u + 1]; k++) {
            const auto [v, w] = adj[k];
            const C rw = w + h[u] - h[v];
            if (rw < 0)   // unreachable: heights were validated above
                throw std::logic_error("canonical_potentials: inconsistent heights");
            if (d + rw < dist[v]) {
                dist[v] = d + rw;
                pq.emplace(dist[v], v);
            }
        }
    }

    // Undo the reweighting: true distance = reduced distance - h_src + h[u].
    for (size_t u = 0; u < n; u++) {
        const C d = dist[u] - h_src + h[u];
        out[u] = leq ? -d : d;
    }
}

#endif // PYLMCF_CANONICAL_POTENTIALS_HPP
