#ifndef PYLMCF_LMCF_HPP
#define PYLMCF_LMCF_HPP

#include <optional>
#include <span>
#include <vector>
#include <utility>
#include <algorithm>
#include <iostream>
#include <lemon/static_graph.h>
#include <lemon/network_simplex.h>
#include <lemon/cycle_canceling.h>
#include <lemon/cost_scaling.h>
#include <lemon/capacity_scaling.h>
#include <type_traits>

#include "basics.hpp"


template <typename T>
void print_span(std::span<T> span) {
    for (auto& elem : span) {
        std::cerr << elem << " ";
    }
    std::cerr << std::endl;
}

// The flow/capacity/supply value type matches the caller's array dtype T, but
// costs are accumulated in a wide type: individual costs fit in T, yet path
// potentials and the cost*flow products that LEMON computes internally overflow
// a narrow T (e.g. int8/int16), silently corrupting the optimum. int64 cost
// arithmetic is what the OO API (Graph<int64_t>) already uses.
using LmcfCost = std::int64_t;

// Core implementation — selects the LEMON solver via template template parameter.
// minimums may be an empty span {} to indicate no lower bounds (zero by default).
// validate_costs: if true, rejects negative arc costs (required by NetworkSimplex).
// Returns the optimal total cost (in the wide cost type, never the narrow T).
template <template <typename...> class Solver, typename T, bool validate_costs = false>
LmcfCost lmcf_impl(
    std::span<T> node_supply,
    std::span<T> edges_starts,
    std::span<T> edges_ends,
    std::span<T> capacities,
    std::span<T> minimums,
    std::span<T> costs,
    std::span<T> result
    )
    requires std::is_signed<T>::value && std::is_integral<T>::value
    {
    if (edges_starts.size() != edges_ends.size() || edges_starts.size() != capacities.size() ||
        (!minimums.empty() && edges_starts.size() != minimums.size()) ||
        edges_starts.size() != costs.size() || edges_starts.size() != result.size()) {
        throw std::invalid_argument("All edge arrays and result must be the same size");
    }

    const size_t no_edges = edges_starts.size();
    const size_t no_nodes = node_supply.size();
    assert_fits_lemon_index(no_nodes, "Node");
    assert_fits_lemon_index(no_edges, "Edge");

    for (size_t i = 0; i < no_edges; i++) {
        if (static_cast<size_t>(edges_starts[i]) >= no_nodes || static_cast<size_t>(edges_ends[i]) >= no_nodes) {
            throw std::invalid_argument("Edge start or end index out of bounds: start=" + std::to_string(edges_starts[i]) + ", end=" + std::to_string(edges_ends[i]));
        }
        if (capacities[i] < 0) {
            throw std::invalid_argument("Capacities must be non-negative");
        }
        if (!minimums.empty() && minimums[i] < 0) {
            throw std::invalid_argument("Minimums must be non-negative");
        }
        if constexpr (validate_costs) {
            if (costs[i] < 0) {
                throw std::invalid_argument("Costs must be non-negative");
            }
        }
    }

    // StaticDigraph (flat arrays, cache-friendly for the residual traversals
    // the LEMON MCF algorithms run) requires arcs sorted by (source, target).
    // The functional API accepts arbitrary edge order, so we sort via a
    // permutation `perm` (perm[j] = caller's index of the j-th sorted arc) and
    // map flows back to the caller's order on the way out. A pre-sorted input
    // (the common case) skips the sort entirely.
    std::vector<LEMON_INDEX> perm(no_edges);
    for (size_t i = 0; i < no_edges; i++) perm[i] = static_cast<LEMON_INDEX>(i);
    auto less_edge = [&](LEMON_INDEX a, LEMON_INDEX b) {
        if (edges_starts[a] != edges_starts[b]) return edges_starts[a] < edges_starts[b];
        return edges_ends[a] < edges_ends[b];
    };
    bool already_sorted = true;
    for (size_t j = 1; j < no_edges; j++)
        if (less_edge(perm[j], perm[j - 1])) { already_sorted = false; break; }
    if (!already_sorted)
        std::sort(perm.begin(), perm.end(), less_edge);

    std::vector<std::pair<LEMON_INDEX, LEMON_INDEX>> arcs;
    arcs.reserve(no_edges);
    for (size_t j = 0; j < no_edges; j++)
        arcs.emplace_back(static_cast<LEMON_INDEX>(edges_starts[perm[j]]),
                          static_cast<LEMON_INDEX>(edges_ends[perm[j]]));

    lemon::StaticDigraph graph;
    graph.build(static_cast<LEMON_INDEX>(no_nodes), arcs.begin(), arcs.end());

    // Arc with id j (== arcFromId(j)) is the j-th arc passed to build(), i.e.
    // the caller's edge perm[j].
    lemon::StaticDigraph::ArcMap<T> capacities_map(graph);
    lemon::StaticDigraph::ArcMap<LmcfCost> costs_map(graph);
    for (size_t j = 0; j < no_edges; j++) {
        const auto a = graph.arcFromId(static_cast<LEMON_INDEX>(j));
        capacities_map[a] = capacities[perm[j]];
        costs_map[a] = costs[perm[j]];
    }

    lemon::StaticDigraph::NodeMap<T> node_supply_map(graph);
    for (size_t i = 0; i < no_nodes; i++)
        node_supply_map[graph.nodeFromId(static_cast<LEMON_INDEX>(i))] = node_supply[i];

    using SolverType = Solver<lemon::StaticDigraph, T, LmcfCost>;
    SolverType solver(graph);

    std::optional<lemon::StaticDigraph::ArcMap<T>> minimums_map;
    if (!minimums.empty()) {
        minimums_map.emplace(graph);
        for (size_t j = 0; j < no_edges; j++)
            (*minimums_map)[graph.arcFromId(static_cast<LEMON_INDEX>(j))] = minimums[perm[j]];
        solver.lowerMap(*minimums_map);
    }

    solver.upperMap(capacities_map);
    solver.costMap(costs_map);
    solver.supplyMap(node_supply_map);
    auto status = solver.run();
    if (status != SolverType::OPTIMAL) {
        if (status == SolverType::INFEASIBLE)
            throw std::runtime_error("Solver failed: problem is INFEASIBLE");
        else if (status == SolverType::UNBOUNDED)
            throw std::runtime_error("Solver failed: problem is UNBOUNDED");
        else
            throw std::runtime_error("Solver failed with unknown status");
    }

    for (size_t j = 0; j < no_edges; j++)
        result[perm[j]] = solver.flow(graph.arcFromId(static_cast<LEMON_INDEX>(j)));

    return solver.totalCost();
}

// NetworkSimplex — requires non-negative costs
template <typename T>
LmcfCost lmcf(
    std::span<T> node_supply,
    std::span<T> edges_starts,
    std::span<T> edges_ends,
    std::span<T> capacities,
    std::span<T> minimums,
    std::span<T> costs,
    std::span<T> result
    )
    requires std::is_signed<T>::value && std::is_integral<T>::value
{
    return lmcf_impl<lemon::NetworkSimplex, T, true>(
        node_supply, edges_starts, edges_ends, capacities, minimums, costs, result);
}

// Backward-compatible overload — no lower bounds
template <typename T>
LmcfCost lmcf(
    std::span<T> node_supply,
    std::span<T> edges_starts,
    std::span<T> edges_ends,
    std::span<T> capacities,
    std::span<T> costs,
    std::span<T> result
    )
    requires std::is_signed<T>::value && std::is_integral<T>::value
{
    return lmcf<T>(node_supply, edges_starts, edges_ends, capacities, std::span<T>{}, costs, result);
}

// CycleCanceling variant
template <typename T>
LmcfCost lmcf_cycle_canceling(
    std::span<T> node_supply,
    std::span<T> edges_starts,
    std::span<T> edges_ends,
    std::span<T> capacities,
    std::span<T> minimums,
    std::span<T> costs,
    std::span<T> result
    )
    requires std::is_signed<T>::value && std::is_integral<T>::value
{
    return lmcf_impl<lemon::CycleCanceling, T>(
        node_supply, edges_starts, edges_ends, capacities, minimums, costs, result);
}

// CycleCanceling overload — no lower bounds
template <typename T>
LmcfCost lmcf_cycle_canceling(
    std::span<T> node_supply,
    std::span<T> edges_starts,
    std::span<T> edges_ends,
    std::span<T> capacities,
    std::span<T> costs,
    std::span<T> result
    )
    requires std::is_signed<T>::value && std::is_integral<T>::value
{
    return lmcf_cycle_canceling<T>(node_supply, edges_starts, edges_ends, capacities, std::span<T>{}, costs, result);
}

// CostScaling variant
template <typename T>
LmcfCost lmcf_cost_scaling(
    std::span<T> node_supply,
    std::span<T> edges_starts,
    std::span<T> edges_ends,
    std::span<T> capacities,
    std::span<T> minimums,
    std::span<T> costs,
    std::span<T> result
    )
    requires std::is_signed<T>::value && std::is_integral<T>::value
{
    return lmcf_impl<lemon::CostScaling, T>(
        node_supply, edges_starts, edges_ends, capacities, minimums, costs, result);
}

// CostScaling overload — no lower bounds
template <typename T>
LmcfCost lmcf_cost_scaling(
    std::span<T> node_supply,
    std::span<T> edges_starts,
    std::span<T> edges_ends,
    std::span<T> capacities,
    std::span<T> costs,
    std::span<T> result
    )
    requires std::is_signed<T>::value && std::is_integral<T>::value
{
    return lmcf_cost_scaling<T>(node_supply, edges_starts, edges_ends, capacities, std::span<T>{}, costs, result);
}

// CapacityScaling variant
template <typename T>
LmcfCost lmcf_capacity_scaling(
    std::span<T> node_supply,
    std::span<T> edges_starts,
    std::span<T> edges_ends,
    std::span<T> capacities,
    std::span<T> minimums,
    std::span<T> costs,
    std::span<T> result
    )
    requires std::is_signed<T>::value && std::is_integral<T>::value
{
    return lmcf_impl<lemon::CapacityScaling, T>(
        node_supply, edges_starts, edges_ends, capacities, minimums, costs, result);
}

// CapacityScaling overload — no lower bounds
template <typename T>
LmcfCost lmcf_capacity_scaling(
    std::span<T> node_supply,
    std::span<T> edges_starts,
    std::span<T> edges_ends,
    std::span<T> capacities,
    std::span<T> costs,
    std::span<T> result
    )
    requires std::is_signed<T>::value && std::is_integral<T>::value
{
    return lmcf_capacity_scaling<T>(node_supply, edges_starts, edges_ends, capacities, std::span<T>{}, costs, result);
}

#endif // PYLMCF_LMCF_HPP
