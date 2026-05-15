#ifndef PYLMCF_LMCF_HPP
#define PYLMCF_LMCF_HPP

#include <optional>
#include <span>
#include <iostream>
#include <lemon/list_graph.h>
#include <lemon/network_simplex.h>
#include <lemon/cycle_canceling.h>
#include <lemon/cost_scaling.h>
#include <lemon/capacity_scaling.h>
#include <type_traits>


template <typename T>
void print_span(std::span<T> span) {
    for (auto& elem : span) {
        std::cerr << elem << " ";
    }
    std::cerr << std::endl;
}

// Core implementation — selects the LEMON solver via template template parameter.
// minimums may be an empty span {} to indicate no lower bounds (zero by default).
// validate_costs: if true, rejects negative arc costs (required by NetworkSimplex).
template <template <typename...> class Solver, typename T, bool validate_costs = false>
T lmcf_impl(
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

    lemon::ListDigraph graph;

    std::vector<lemon::ListDigraph::Node> nodes;
    nodes.reserve(no_nodes);
    for (size_t i = 0; i < no_nodes; i++)
        nodes.push_back(graph.addNode());

    std::vector<lemon::ListDigraph::Arc> arcs;
    arcs.reserve(no_edges);
    for (size_t i = 0; i < no_edges; i++)
        arcs.push_back(graph.addArc(nodes[edges_starts[i]], nodes[edges_ends[i]]));

    lemon::ListDigraph::ArcMap<T> capacities_map(graph);
    lemon::ListDigraph::ArcMap<T> costs_map(graph);
    for (size_t i = 0; i < no_edges; i++) {
        capacities_map[arcs[i]] = capacities[i];
        costs_map[arcs[i]] = costs[i];
    }

    lemon::ListDigraph::NodeMap<typename std::make_signed<T>::type> node_supply_map(graph);
    for (size_t i = 0; i < no_nodes; i++)
        node_supply_map[nodes[i]] = node_supply[i];

    using SolverType = Solver<lemon::ListDigraph, T, T>;
    SolverType solver(graph);

    std::optional<lemon::ListDigraph::ArcMap<T>> minimums_map;
    if (!minimums.empty()) {
        minimums_map.emplace(graph);
        for (size_t i = 0; i < no_edges; i++)
            (*minimums_map)[arcs[i]] = minimums[i];
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

    for (size_t i = 0; i < no_edges; i++)
        result[i] = solver.flow(arcs[i]);

    return solver.totalCost();
}

// NetworkSimplex — requires non-negative costs
template <typename T>
T lmcf(
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
T lmcf(
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
T lmcf_cycle_canceling(
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
T lmcf_cycle_canceling(
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
T lmcf_cost_scaling(
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
T lmcf_cost_scaling(
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
T lmcf_capacity_scaling(
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
T lmcf_capacity_scaling(
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
