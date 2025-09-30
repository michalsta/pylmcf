#ifndef PYLMCF_GRAPH_HPP
#define PYLMCF_GRAPH_HPP

#include <stdexcept>
#include <span>
#include <vector>
#include <lemon/static_graph.h>
#include <lemon/network_simplex.h>

#include "basics.hpp"

#ifdef INCLUDE_NANOBIND_STUFF
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "py_support.hpp"
#endif

inline lemon::StaticDigraph make_lemon_graph(LEMON_INDEX no_nodes, const std::span<LEMON_INDEX> &edge_starts,
    const std::span<LEMON_INDEX> &edge_ends) {
    const size_t no_edges = edge_starts.size();

    // Make sure all edge arrays and result are the same size
    if (edge_starts.size() != edge_ends.size()) {
        throw std::invalid_argument("All edge arrays must be the same size");
    }

    // Make sure all arcs are valid
    for (LEMON_INDEX ii = 0; ii < no_edges; ii++) {
        if (edge_starts[ii] >= no_nodes || edge_ends[ii] >= no_nodes) {
            throw std::invalid_argument("Edge start or end index out of bounds: start=" + std::to_string(edge_starts[ii]) + ", end=" + std::to_string(edge_ends[ii]));
        }
    }

    lemon::StaticDigraph lemon_graph;

    std::vector<std::pair<LEMON_INDEX, LEMON_INDEX>> arcs;
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
    const LEMON_INDEX _no_nodes;
    const std::vector<LEMON_INDEX> _edge_starts;
    const std::vector<LEMON_INDEX> _edge_ends;

    const lemon::StaticDigraph lemon_graph;
    lemon::StaticDigraph::NodeMap<T> node_supply_map;
    lemon::StaticDigraph::ArcMap<T> capacities_map;
    lemon::StaticDigraph::ArcMap<T> costs_map;

    lemon::NetworkSimplex<lemon::StaticDigraph, T, T> solver;

public:
    Graph(LEMON_INDEX no_nodes, const std::span<LEMON_INDEX> &edge_starts,
        const std::span<LEMON_INDEX> &edge_ends):

        _no_nodes(no_nodes),
        _edge_starts(edge_starts.begin(), edge_starts.end()),
        _edge_ends(edge_ends.begin(), edge_ends.end()),
        lemon_graph(make_lemon_graph(no_nodes, edge_starts, edge_ends)),
        node_supply_map(lemon_graph),
        capacities_map(lemon_graph),
        costs_map(lemon_graph),
        solver(lemon_graph)
        {};


    Graph() = delete;
    Graph(Graph&&) = delete;
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;


    inline LEMON_INDEX no_nodes() const {
        return _no_nodes;
    }

    inline LEMON_INDEX no_edges() const {
        return _edge_starts.size();
    }

    inline const std::vector<LEMON_INDEX>& edge_starts() const {
        return _edge_starts;
    }

    inline const std::vector<LEMON_INDEX>& edge_ends() const {
        return _edge_ends;
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

    std::span<T> get_edge_capacities() const {
        T* data = static_cast<T*>(malloc(sizeof(T) * no_edges()));
        for (size_t ii = 0; ii < no_edges(); ii++)
        {
            data[ii] = capacities_map[lemon_graph.arcFromId(ii)];
        }
        return std::span<T>(data, no_edges());
    }

    void set_edge_costs(const std::span<T> &costs) {
        if (costs.size() != no_edges())
            throw std::invalid_argument("Costs must have the same size as the number of edges");

        for (size_t ii = 0; ii < no_edges(); ii++)
        {
            if (costs[ii] < 0)
                throw std::invalid_argument("Costs must be non-negative");
            costs_map[lemon_graph.arcFromId(ii)] = costs[ii];
        }

        solver.costMap(costs_map);
    }

    std::span<T> get_edge_costs() const {
        T* data = static_cast<T*>(malloc(sizeof(T) * no_edges()));
        for (size_t ii = 0; ii < no_edges(); ii++)
        {
            data[ii] = costs_map[lemon_graph.arcFromId(ii)];
        }
        return std::span<T>(data, no_edges());
    }

    void solve(){
        solver.supplyMap(node_supply_map);
        solver.costMap(costs_map);
        solver.run();
    }

    T total_cost() const {
        return solver.totalCost();
    }

    std::span<T> get_edge_flows() const {
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

#ifdef INCLUDE_NANOBIND_STUFF
    Graph(LEMON_INDEX no_nodes, const nb::ndarray<LEMON_INDEX, nb::shape<-1>> &edge_starts,
        const nb::ndarray<LEMON_INDEX, nb::shape<-1>> &edge_ends):
        Graph(no_nodes, numpy_to_span(edge_starts), numpy_to_span<LEMON_INDEX>(edge_ends)) {};

    void set_node_supply_py(const nb::ndarray<T, nb::shape<-1>> &node_supply) {
        set_node_supply(numpy_to_span(node_supply));
    }

    void set_edge_capacities_py(const nb::ndarray<T, nb::shape<-1>> &capacities) {
        set_edge_capacities(numpy_to_span(capacities));
    }

    void set_edge_costs_py(const nb::ndarray<T, nb::shape<-1>> &costs) {
        set_edge_costs(numpy_to_span(costs));
    }

    nb::ndarray<T, nb::numpy, nb::shape<-1>> get_edge_capacities_py() const {
        return steal_mallocd_span_to_np_array(get_edge_capacities());
    }

    nb::ndarray<T, nb::numpy, nb::shape<-1>> get_edge_costs_py() const {
        return steal_mallocd_span_to_np_array(get_edge_costs());
    }

    nb::ndarray<T, nb::numpy, nb::shape<-1>> extract_result_py() const {
        return steal_mallocd_span_to_np_array(get_edge_flows());
    }
#endif

};

#endif // PYLMCF_GRAPH_HPP