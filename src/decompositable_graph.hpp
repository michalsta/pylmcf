#include <vector>
#include <span>
#include <algorithm>
#include <unordered_map>

#include <lemon/static_graph.h>
#include <lemon/network_simplex.h>

#include "py_support.h"
#include "graph_elements.hpp"
#include "spectrum.hpp"

#include <iostream>



class FlowSubgraph {
    std::vector<FlowNode> nodes;
    std::vector<FlowEdge> edges;
    lemon::StaticDigraph lemon_graph;
    lemon::StaticDigraph::NodeMap<LEMON_INT> node_supply_map;
    lemon::StaticDigraph::ArcMap<LEMON_INT> capacities_map;
    lemon::StaticDigraph::ArcMap<LEMON_INT> costs_map;
    std::optional<lemon::NetworkSimplex<lemon::StaticDigraph>> solver;
    size_t simple_trash_idx;
    LEMON_INT empirical_intensity;
    LEMON_INT theoretical_intensity;

public:
    FlowSubgraph(
        const std::vector<size_t>& subgraph_node_ids,
        const std::vector<FlowNode>& all_nodes,
        const std::vector<FlowEdge>& all_edges
    ) :
        lemon_graph(),
        node_supply_map(lemon_graph),
        capacities_map(lemon_graph),
        costs_map(lemon_graph),
        solver(),
        simple_trash_idx(std::numeric_limits<size_t>::max()),
        empirical_intensity(0),
        theoretical_intensity(0)
    {
        nodes.reserve(subgraph_node_ids.size()+2);
        nodes.push_back(FlowNode(0, SourceNode()));
        nodes.push_back(FlowNode(1, SinkNode()));
        auto& source_node = nodes[0];
        auto& sink_node = nodes[1];

        std::unordered_map<size_t, size_t> node_id_map;

        for (const auto& node_id : subgraph_node_ids)
        {
            node_id_map[node_id] = nodes.size();
            const FlowNodeType& node_type = all_nodes[node_id].get_type();
            nodes.push_back(FlowNode(nodes.size(), node_type));
            auto& new_node = nodes.back();
            if(std::holds_alternative<EmpiricalNode>(node_type))
            {
                edges.emplace_back(
                    edges.size(),
                    source_node,
                    new_node,
                    SrcToEmpiricalEdge()
                );
            }
            else if(std::holds_alternative<TheoreticalNode>(node_type))
            {
                edges.emplace_back(
                    edges.size(),
                    new_node,
                    sink_node,
                    TheoreticalToSinkEdge()
                );
            }
            else throw std::runtime_error("Invalid FlowNode type. This shouldn't happen.");
        }

        for (const FlowEdge& edge : all_edges)
        {
            const FlowNode& start_node = edge.get_start_node();
            const auto start_node_it = node_id_map.find(start_node.get_id());
            if (start_node_it == node_id_map.end()) continue;
            const FlowNode& end_node = edge.get_end_node();
            const auto end_node_it = node_id_map.find(end_node.get_id());
            if (end_node_it == node_id_map.end()) continue;
            edges.emplace_back(
                    edges.size(),
                    nodes[start_node_it->second],
                    nodes[end_node_it->second],
                    edge.get_type()
            );
        }
    }

    void add_simple_trash(LEMON_INT cost) {
        edges.emplace_back(
            edges.size(),
            nodes[0],
            nodes[1],
            SimpleTrashEdge(cost)
        );
    }

    void build() {
        edges = sorted_copy(edges, [](const FlowEdge& a, const FlowEdge& b) {
            if(a.get_start_node_id() != b.get_start_node_id())
                return a.get_start_node_id() < b.get_start_node_id();
            return a.get_end_node_id() < b.get_end_node_id();
        });
        std::vector<std::pair<int, int>> arcs;
        arcs.reserve(edges.size());
        for (const FlowEdge& edge : edges)
            arcs.emplace_back(edge.get_start_node_id(), edge.get_end_node_id());
        lemon_graph.build(nodes.size(), arcs.begin(), arcs.end());

        for (size_t ii = 0; ii < nodes.size(); ++ii)
            node_supply_map[lemon_graph.nodeFromId(ii)] = 0;

        for (size_t ii = 0; ii < edges.size(); ++ii)
            costs_map[lemon_graph.arcFromId(ii)] = std::visit([&](const auto& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, MatchingEdge>) return arg.get_cost();
                    else if constexpr (std::is_same_v<T, SrcToEmpiricalEdge>) return (LEMON_INT) 0;
                    else if constexpr (std::is_same_v<T, TheoreticalToSinkEdge>) return (LEMON_INT) 0;
                    else if constexpr (std::is_same_v<T, SimpleTrashEdge>) { simple_trash_idx = ii; return arg.get_cost(); }
                    else { throw std::runtime_error("Invalid FlowEdgeType"); };
                }, edges[ii].get_type());

        for (size_t ii = 0; ii < edges.size(); ++ii)
        {
            capacities_map[lemon_graph.arcFromId(ii)] = std::visit([&](const auto& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, MatchingEdge>) return (LEMON_INT) 0;
                    else if constexpr (std::is_same_v<T, SrcToEmpiricalEdge>) {
                        LEMON_INT intensity = (LEMON_INT) std::get<EmpiricalNode>(edges[ii].get_end_node().get_type()).get_intensity();
                        empirical_intensity += intensity;
                        return intensity;
                    }
                    else if constexpr (std::is_same_v<T, TheoreticalToSinkEdge>) return (LEMON_INT) 0;
                    else if constexpr (std::is_same_v<T, SimpleTrashEdge>) return (LEMON_INT) 0;
                    else { throw std::runtime_error("Invalid FlowEdgeType"); };
                }, edges[ii].get_type());
        }
        //solver.emplace(lemon_graph);//lemon::NetworkSimplex<lemon::StaticDigraph>(lemon_graph);
        //solver->upperMap(capacities_map);
    }

    void set_point(const std::vector<INTENSITY_TYPE>& point) {
        theoretical_intensity = 0;
        for (size_t ii = 0; ii < edges.size(); ++ii)
        {
            const FlowEdge& edge = edges[ii];
            std::visit([&](const auto& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, MatchingEdge>) {
                    const auto& theoretical_node_type = std::get<TheoreticalNode>(edge.get_end_node().get_type());
                    capacities_map[lemon_graph.arcFromId(ii)] = (LEMON_INT) std::min<double>(
                        theoretical_node_type.get_intensity() * point[theoretical_node_type.get_spectrum_id()],
                        std::get<EmpiricalNode>(edge.get_start_node().get_type()).get_intensity());
                    }
                else if constexpr (std::is_same_v<T, TheoreticalToSinkEdge>) {
                    const auto& theoretical_node_type = std::get<TheoreticalNode>(edge.get_start_node().get_type());
                    LEMON_INT intensity = (LEMON_INT) (theoretical_node_type.get_intensity() * point[theoretical_node_type.get_spectrum_id()]);
                    lemon_graph.arcFromId(ii);
                    capacities_map[lemon_graph.arcFromId(ii)] = intensity;
                    theoretical_intensity += intensity;
                }
                else if constexpr (std::is_same_v<T, SrcToEmpiricalEdge>) {}
                else if constexpr (std::is_same_v<T, SimpleTrashEdge>) {}
                else { throw std::runtime_error("Invalid FlowEdgeType"); };
            }, edge.get_type());
        }
        const LEMON_INT total_flow = std::max<LEMON_INT>(empirical_intensity, theoretical_intensity);
        if(simple_trash_idx != std::numeric_limits<size_t>::max())
        {
            capacities_map[lemon_graph.arcFromId(simple_trash_idx)] = total_flow;
            costs_map[lemon_graph.arcFromId(simple_trash_idx)] = std::get<SimpleTrashEdge>(edges[simple_trash_idx].get_type()).get_cost();
        }
        node_supply_map[lemon_graph.nodeFromId(0)] = total_flow;
        node_supply_map[lemon_graph.nodeFromId(1)] = -total_flow;
        solver.emplace(lemon_graph);
        solver->upperMap(capacities_map);
        solver->costMap(costs_map);
        solver->supplyMap(node_supply_map);
        solver->run();
    }

    LEMON_INT total_cost() const {
        return solver->totalCost();
    };

    std::string to_string() const {
        std::string result;
        result += "FlowSubgraph:\n";
        result += "Nodes:\n";
        for (const auto& node : nodes) {
            result += node.to_string() + "\n";
        }
        result += "Edges:\n";
        for (int ii = 0; ii < lemon_graph.arcNum(); ++ii) {
            result += "Edge " + std::to_string(lemon_graph.id(lemon_graph.arcFromId(ii))) + ": " +
                      std::to_string(lemon_graph.id(lemon_graph.source(lemon_graph.arcFromId(ii)))) + " -> " +
                      std::to_string(lemon_graph.id(lemon_graph.target(lemon_graph.arcFromId(ii)))) + " cost: " +
                      std::to_string(costs_map[lemon_graph.arcFromId(ii)]) + " capacity: " +
                      std::to_string(capacities_map[lemon_graph.arcFromId(ii)]) + " flow: " +
                      std::to_string(solver->flow(lemon_graph.arcFromId(ii))) + "\n";
        }
        return result;
    };

    std::string lemon_to_string() const {
        std::string result;
        result += "Lemon graph:\n";
        result += "Nodes:\n";
        for (int ii = 0; ii < lemon_graph.nodeNum(); ++ii) {
            result += "Node " + std::to_string(lemon_graph.id(lemon_graph.nodeFromId(ii))) + " supply: " +
                      std::to_string(node_supply_map[lemon_graph.nodeFromId(ii)]) + "\n";
        }
        result += "Edges:\n";
        for (int ii = 0; ii < lemon_graph.arcNum(); ++ii) {
            result += "Edge " + std::to_string(lemon_graph.id(lemon_graph.arcFromId(ii))) + ": " +
                      std::to_string(lemon_graph.id(lemon_graph.source(lemon_graph.arcFromId(ii)))) + " -> " +
                      std::to_string(lemon_graph.id(lemon_graph.target(lemon_graph.arcFromId(ii)))) + " cost: " +
                      std::to_string(costs_map[lemon_graph.arcFromId(ii)]) + " capacity: " +
                      std::to_string(capacities_map[lemon_graph.arcFromId(ii)]) + " flow: " +
                      std::to_string(solver->flow(lemon_graph.arcFromId(ii))) + "\n";
        }
        return result;
    };

    size_t no_nodes() const {
        return nodes.size();
    };

    size_t no_edges() const {
        return edges.size();
    };

    const std::vector<FlowNode>& get_nodes() const {
        return nodes;
    };

    const std::vector<FlowEdge>& get_edges() const {
        return edges;
    };

    void flows_for_spectrum(size_t spectrum_id,
                            std::vector<size_t>& empirical_peak_indices,
                            std::vector<size_t>& theoretical_peak_indices,
                            std::vector<LEMON_INT>& flows) const
    {
        for (size_t ii = 0; ii < edges.size(); ++ii)
        {
            const FlowEdge& edge = edges[ii];
            const LEMON_INT flow = solver->flow(lemon_graph.arcFromId(ii));
            if (flow == 0) continue;
            std::visit([&](const auto& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, MatchingEdge>) {
                    const auto& theoretical_node_type = std::get<TheoreticalNode>(edge.get_end_node().get_type());
                    if(theoretical_node_type.get_spectrum_id() == spectrum_id)
                    {
                        empirical_peak_indices.push_back(std::get<EmpiricalNode>(edge.get_start_node().get_type()).get_peak_index());
                        theoretical_peak_indices.push_back(theoretical_node_type.get_peak_index());
                        flows.push_back(flow);
                    }
                }
                else if constexpr (std::is_same_v<T, TheoreticalToSinkEdge>) {}
                else if constexpr (std::is_same_v<T, SrcToEmpiricalEdge>) {}
                else if constexpr (std::is_same_v<T, SimpleTrashEdge>) {}
                else { throw std::runtime_error("Invalid FlowEdgeType"); };
            }, edge.get_type());
        }
    };

    template<typename T>
    size_t count_nodes_of_type() const {
        size_t result = 0;
        for (const auto& node : nodes)
            if(std::holds_alternative<T>(node.get_type()))
                result++;
        return result;
    }

    template<typename T>
    size_t count_edges_of_type() const {
        size_t result = 0;
        for (const auto& edge : edges)
            if(std::holds_alternative<T>(edge.get_type()))
                result++;
        return result;
    }

    double matching_density() const {
        const double nominator = count_edges_of_type<MatchingEdge>();
        const double denominator = count_nodes_of_type<EmpiricalNode>() * count_nodes_of_type<TheoreticalNode>();
        return nominator / denominator;
    }

};

class DecompositableFlowGraph {
    std::vector<FlowNode> nodes;
    std::vector<FlowEdge> edges;

    const size_t _no_theoretical_spectra;

    std::vector<size_t> dead_end_node_ids;
    std::vector<std::unique_ptr<FlowSubgraph>> flow_subgraphs;

public:
    DecompositableFlowGraph(
    const Spectrum* empirical_spectrum,
    const std::vector<Spectrum*>& theoretical_spectra,
    const py::function* dist_fun,
    LEMON_INT max_dist
    ) :
    _no_theoretical_spectra(theoretical_spectra.size())
    {
        {
            size_t no_nodes = 2 + empirical_spectrum->size();
            for (auto& ts : theoretical_spectra)
                no_nodes += ts->size();
            nodes.reserve(no_nodes);
        }

        // Create placeholder source and sink nodes
        nodes.emplace_back(FlowNode(0, SourceNode()));
        nodes.emplace_back(FlowNode(1, SinkNode()));

        for (size_t empirical_idx = 0; empirical_idx < empirical_spectrum->size(); ++empirical_idx) {
            nodes.emplace_back(FlowNode(
                                    nodes.size(),
                                    EmpiricalNode(
                                        empirical_idx,
                                        empirical_spectrum->intensities[empirical_idx])));
        }

        for (size_t theoretical_spectrum_idx = 0; theoretical_spectrum_idx < theoretical_spectra.size(); ++theoretical_spectrum_idx)
        {
            #ifdef DO_TONS_OF_PRINTS
            size_t no_processed = 0;
            size_t no_included = 0;
            #endif

            const auto& theoretical_spectrum = theoretical_spectra[theoretical_spectrum_idx];

            for (size_t theoretical_peak_idx = 0; theoretical_peak_idx < theoretical_spectrum->size(); ++theoretical_peak_idx) {
                nodes.emplace_back(FlowNode(
                                        nodes.size(),
                                            TheoreticalNode(
                                                theoretical_spectrum_idx,
                                                theoretical_peak_idx,
                                                theoretical_spectrum->intensities[theoretical_peak_idx])));
                const auto& theoretical_node = nodes.back();

                // Calculate the distance between the empirical and theoretical peaks
                auto [indices, distances] = empirical_spectrum->closer_than(
                    theoretical_spectrum->get_point(theoretical_peak_idx),
                    dist_fun,
                    max_dist
                );
                #ifdef DO_TONS_OF_PRINTS
                no_processed += theoretical_spectrum->size();
                no_included += indices.size();
                std::cout << no_included << " / " << no_processed << " = " << static_cast<float>(no_included) / static_cast<float>(no_processed) << std::endl;
                #endif

                for (size_t ii = 0; ii < indices.size(); ++ii)
                    edges.emplace_back(FlowEdge(
                        edges.size(),
                        nodes[indices[ii] + 2], // +2 to skip the source and sink nodes
                        theoretical_node,
                        MatchingEdge(distances[ii])
                    ));
            }
        }
        build_subgraphs();
    };

    size_t no_nodes() const {
        return nodes.size();
    };
    size_t no_edges() const {
        return edges.size();
    };
    size_t no_theoretical_spectra() const {
        return _no_theoretical_spectra;
    };

    const std::vector<FlowNode>& get_nodes() const {
        return nodes;
    };
    const std::vector<FlowEdge>& get_edges() const {
        return edges;
    };

    std::vector<std::vector<size_t>> neighbourhood_lists() const {
        std::vector<std::vector<size_t>> neighbourhood_lists;
        neighbourhood_lists.resize(nodes.size());
        for (const auto& edge : edges) {
            const size_t start_node_id = edge.get_start_node_id();
            const size_t end_node_id = edge.get_end_node_id();
            neighbourhood_lists[start_node_id].push_back(end_node_id);
            neighbourhood_lists[end_node_id].push_back(start_node_id);
        }
        return neighbourhood_lists;
    };

    std::pair<std::vector<std::vector<size_t>>, std::vector<size_t>> split_into_subgraphs() const {
        std::vector<std::vector<size_t>> subgraphs;
        std::vector<size_t> dead_end_nodes;

        std::vector<bool> visited(nodes.size(), false);
        visited[0] = true; // Mark the source node as visited
        visited[1] = true; // Mark the sink node as visited
        std::vector<size_t> stack;
        std::vector<std::vector<size_t>> neighbourhood_lists = this->neighbourhood_lists();

        for (size_t node_id = 0; node_id < nodes.size(); ++node_id) {
            if (!visited[node_id]) {
                std::vector<size_t>& neighbours = neighbourhood_lists[node_id];
                if(neighbours.size() == 0) {
                    dead_end_nodes.push_back(node_id);
                } else {
                    std::vector<size_t> subgraph;
                    stack.push_back(node_id);
                    while (!stack.empty()) {
                        size_t current_node = stack.back();
                        stack.pop_back();
                        if (!visited[current_node]) {
                            visited[current_node] = true;
                            subgraph.push_back(current_node);
                            for (size_t neighbour : neighbourhood_lists[current_node]) {
                                if (!visited[neighbour]) {
                                    stack.push_back(neighbour);
                                }
                            }
                        }
                    }
                    // TODO: potentially remove this
                    std::sort(subgraph.begin(), subgraph.end());
                    subgraphs.push_back(subgraph);
                }
            }
        }
        return {subgraphs, dead_end_nodes};
    }

    void build_subgraphs() {
        auto [_subgraphs, _dead_end_nodes] = this->split_into_subgraphs();

        dead_end_node_ids = std::move(_dead_end_nodes);

        // TODO: optimize, right now this is needlessly O(subgraphs.size() * edges.size()),
        // can be O(subgraphs.size() + edges.size())
        flow_subgraphs.reserve(_subgraphs.size());
        for (const auto& subgraph_node_ids : _subgraphs)
        {
            #ifdef DO_TONS_OF_PRINTS
            std::cout << "Subgraph" << std::endl;
            #endif
            flow_subgraphs.emplace_back(std::make_unique<FlowSubgraph>(
                    subgraph_node_ids,
                    nodes,
                    edges
            ));
        }
    }

    void add_simple_trash(LEMON_INT cost) {
        for (auto& flow_subgraph : flow_subgraphs)
            flow_subgraph->add_simple_trash(cost);
    };

    void build() {
        for (auto& flow_subgraph : flow_subgraphs)
            flow_subgraph->build();
    };

    void set_point(const std::vector<INTENSITY_TYPE>& point) {
        for (auto& flow_subgraph : flow_subgraphs)
            flow_subgraph->set_point(point);
    };

    LEMON_INT total_cost() const {
        LEMON_INT total_cost = 0;
        for (const auto& flow_subgraph : flow_subgraphs)
            total_cost += flow_subgraph->total_cost();
        return total_cost;
    };

    size_t no_subgraphs() const {
        return flow_subgraphs.size();
    };

    const FlowSubgraph& get_subgraph(size_t idx) const {
        if (idx >= flow_subgraphs.size())
            throw std::out_of_range("Subgraph index out of range");
        return *flow_subgraphs[idx];
    };

    std::string to_string() const {
        std::string result;
        for (const auto& flow_subgraph : flow_subgraphs)
            result += flow_subgraph->to_string();
        return result;
    };

    std::string lemon_to_string() const {
        std::string result;
        for (const auto& flow_subgraph : flow_subgraphs)
            result += flow_subgraph->lemon_to_string();
        return result;
    };

    std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<LEMON_INT>> flows_for_spectrum(size_t spectrum_id) const {
        std::vector<size_t> empirical_peak_indices;
        std::vector<size_t> theoretical_peak_indices;
        std::vector<LEMON_INT> flows;
        for (const auto& flow_subgraph : flow_subgraphs)
            flow_subgraph->flows_for_spectrum(spectrum_id, empirical_peak_indices, theoretical_peak_indices, flows);
        return {empirical_peak_indices, theoretical_peak_indices, flows};
    };

    size_t count_matching_edges() const {
        size_t result = 0;
        for (const auto& edge : edges)
            std::visit([&](const auto& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, MatchingEdge>) result++;
            },
            edge.get_type());
        return result;
    }

    template<typename T>
    size_t count_nodes_of_type() const {
        size_t result = 0;
        for (const auto& node : nodes)
            if(std::holds_alternative<T>(node.get_type()))
                result++;
        return result;
    }

    template<typename T>
    size_t count_edges_of_type() const {
        size_t result = 0;
        for (const auto& edge : edges)
            if(std::holds_alternative<T>(edge.get_type()))
                result++;
        return result;
    }

    double matching_density() const {
        const double nominator = count_edges_of_type<MatchingEdge>();
        const double denominator = 0;
        for (const auto& flow_subgraph : flow_subgraphs)
            denominator += flow_subgraph->count_nodes_of_type<EmpiricalNode>() * flow_subgraph->count_nodes_of_type<TheoreticalNode>();
        return nominator / denominator;
    }
};

