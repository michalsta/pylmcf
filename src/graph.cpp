#include <stdexcept>
#include <span>
#include <vector>
#include <lemon/static_graph.h>
#include <lemon/network_simplex.h>

#ifdef PYBIND11_VERSION_MAJOR
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "py_support.h"
#endif

lemon::StaticDigraph make_lemon_graph(size_t no_nodes, const std::span<int64_t> &edge_starts,
    const std::span<int64_t> &edge_ends) {
    const size_t no_edges = edge_starts.size();

    // Make sure all edge arrays and result are the same size
    if (edge_starts.size() != edge_ends.size()) {
        throw std::invalid_argument("All edge arrays must be the same size");
    }

    // Make sure all arcs are valid
    for (size_t ii = 0; ii < no_edges; ii++) {
        if (static_cast<size_t>(edge_starts[ii]) >= no_nodes || static_cast<size_t>(edge_ends[ii]) >= no_nodes) {
            throw std::invalid_argument("Edge start or end index out of bounds: start=" + std::to_string(edge_starts[ii]) + ", end=" + std::to_string(edge_ends[ii]));
        }
    }

    lemon::StaticDigraph lemon_graph;

    std::vector<std::pair<int, int>> arcs;
    arcs.reserve(no_edges);
    for (size_t ii = 0; ii < no_edges; ii++)
        arcs.emplace_back(edge_starts[ii], edge_ends[ii]);

    if(!std::is_sorted(arcs.begin(), arcs.end()))
        throw std::invalid_argument("Edges must be sorted by start node, then by end node");

    lemon_graph.build(no_nodes, arcs.begin(), arcs.end());

    return lemon_graph;
}


template <typename T> class Graph {
private:
    const size_t _no_nodes;
    const std::vector<int64_t> edges_starts;
    const std::vector<int64_t> edges_ends;
    const std::vector<int64_t> costs;

    const lemon::StaticDigraph lemon_graph;
    lemon::StaticDigraph::NodeMap<T> node_supply_map;
    lemon::StaticDigraph::ArcMap<T> capacities_map;
    lemon::StaticDigraph::ArcMap<T> costs_map;

    lemon::NetworkSimplex<lemon::StaticDigraph, T, T> solver;

public:
    Graph(size_t no_nodes, const std::span<int64_t> &edge_starts,
        const std::span<int64_t> &edge_ends, const std::span<T> &costs):

        _no_nodes(no_nodes),
        edges_starts(edge_starts.begin(), edge_starts.end()),
        edges_ends(edge_ends.begin(), edge_ends.end()),
        costs(costs.begin(), costs.end()),
        lemon_graph(make_lemon_graph(no_nodes, edge_starts, edge_ends)),
        node_supply_map(lemon_graph),
        capacities_map(lemon_graph),
        costs_map(lemon_graph),
        solver(lemon_graph)
        {
            const size_t no_edges = edges_starts.size();

            // Make sure all edge arrays and result are the same size
            if (edges_starts.size() != edges_ends.size() || edges_starts.size() != costs.size()) {
                throw std::invalid_argument("All edge arrays and result must be the same size");
            }

            // Make sure all arcs are valid, capacities are positive, and costs are non-negative
            for (size_t ii = 0; ii < no_edges; ii++) {
                if (static_cast<size_t>(edges_starts[ii]) >= no_nodes || static_cast<size_t>(edges_ends[ii]) >= no_nodes) {
                    throw std::invalid_argument("Edge start or end index out of bounds: start=" + std::to_string(edges_starts[ii]) + ", end=" + std::to_string(edges_ends[ii]));
                }
                if (costs[ii] < 0) {
                    throw std::invalid_argument("Costs must be non-negative");
                }
            }

            // Add costs to the arcs
            for (size_t ii = 0; ii < no_edges; ii++) {
                costs_map[lemon_graph.arcFromId(ii)] = costs[ii];
            }
        };


    Graph() = delete;
    Graph(Graph&&) = delete;
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;


    inline size_t no_nodes() const {
        return _no_nodes;
    }

    inline size_t no_edges() const {
        return edges_starts.size();
    }

    void set_node_supply(const std::span<T> &node_supply) {
        if (node_supply.size() != no_nodes())
            throw std::invalid_argument("Node supply must have the same size as the number of nodes");

        for (size_t ii = 0; ii < no_nodes(); ii++)
            node_supply_map[lemon_graph.nodeFromId(ii)] = node_supply[ii];

    }

    void set_edge_capacities(const std::span<T> &capacities) {
        if (capacities.size() != no_edges())
            throw std::invalid_argument("Capacities must have the same size as the number of edges");

        for (size_t ii = 0; ii < no_edges(); ii++)
        {
            // std::cerr << "Setting capacity " << capacities[ii] << " for edge " << ii << std::endl;
            capacities_map[lemon_graph.arcFromId(ii)] = capacities[ii];
        }

        solver.upperMap(capacities_map);
    }

    void solve(){
        solver.supplyMap(node_supply_map);
        solver.costMap(costs_map);
        solver.run();
    }

    T total_cost() const {
        return solver.totalCost();
    }

    std::span<T> extract_result() const {
        T* data = static_cast<T*>(malloc(sizeof(T) * no_edges()));
        for (size_t ii = 0; ii < no_edges(); ii++)
        {
            data[ii] = solver.flow(lemon_graph.arcFromId(ii));
            // std::cerr << "Flow for edge " << ii << " is " << data[ii] << std::endl;
        }
        return std::span<T>(data, no_edges());
    }

    std::string to_string() const {
        std::string out = "Graph with " + std::to_string(no_nodes()) + " nodes and " + std::to_string(no_edges()) + " edges\n";
        out += "Edges:\n";
        for (size_t ii = 0; ii < no_edges(); ii++) {
            out += "  " + std::to_string(lemon_graph.id(lemon_graph.source(lemon_graph.arcFromId(ii)))) + " -> " + std::to_string(lemon_graph.id(lemon_graph.target(lemon_graph.arcFromId(ii)))) + " with cost " + std::to_string(costs_map[lemon_graph.arcFromId(ii)]) + " and capacity " + std::to_string(capacities_map[lemon_graph.arcFromId(ii)]) + "\n";
        }
        // for (size_t ii = 0; ii < no_edges(); ii++) {
        //     out += "  " + std::to_string(edges_starts[ii]) + " -> " + std::to_string(edges_ends[ii]) + " with cost " + std::to_string(costs[ii]) + "\n";
        // }
        return out;
    }

    #ifdef PYBIND11_VERSION_MAJOR
    Graph(size_t no_nodes, const py::array_t<int64_t> &edge_starts,
        const py::array_t<int64_t> &edge_ends, const py::array_t<T> &costs):
        Graph(no_nodes, numpy_to_span(edge_starts), numpy_to_span(edge_ends), numpy_to_span(costs)) {};

    void set_node_supply_py(const py::array_t<T> &node_supply) {
        set_node_supply(numpy_to_span(node_supply));
    }

    void set_edge_capacities_py(const py::array_t<T> &capacities) {
        set_edge_capacities(numpy_to_span(capacities));
    }

    py::array_t<T> extract_result_py() const {
        return mallocd_span_to_owning_numpy(extract_result());
    }
    #endif

};
