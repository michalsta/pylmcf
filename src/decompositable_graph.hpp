#include <vector>
#include <span>
#include <algorithm>
#include <unordered_map>

#include "py_support.h"
#include "graph_elements.hpp"
#include "spectrum.hpp"



class FlowSubgraph {
    std::vector<FlowNode> nodes;
    std::vector<FlowEdge> edges;

public:
    FlowSubgraph(
        const std::vector<size_t>& subgraph_node_ids,
        const std::vector<FlowNode>& all_nodes,
        const std::vector<FlowEdge>& all_edges
    ){
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


};

class DecompositableFlowGraph {
    std::vector<FlowNode> nodes;
    std::vector<FlowEdge> edges;

    const size_t _no_theoretical_spectra;

    std::vector<size_t> dead_end_node_ids;
    std::vector<FlowSubgraph> flow_subgraphs;

public:

    // DecompositableFlowGraph(
    //     const Spectrum* empirical_spectrum,
    //     const std::vector<const Spectrum*>& theoretical_spectra,
    //     const std::vector<const py::function*>& dist_funs,
    //     const std::vector<const LEMON_INT>& max_dists
    // ) : DecompositableFlowGraph(
    //     empirical_spectrum,
    //     std::span<const Spectrum*>(theoretical_spectra.data(), theoretical_spectra.size()),
    //     std::span<const py::function*>(dist_funs.data(), dist_funs.size()),
    //     std::span<const LEMON_INT>(max_dists.data(), max_dists.size())
    // ) {};


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
        //nodes.emplace_back(SinkNode(1));

        for (size_t empirical_idx = 0; empirical_idx < empirical_spectrum->size(); ++empirical_idx) {
            nodes.emplace_back(FlowNode(
                                    nodes.size(),
                                    EmpiricalNode(
                                        empirical_idx,
                                        empirical_spectrum->intensities[empirical_idx])));
        }

        for (size_t theoretical_spectrum_idx = 0; theoretical_spectrum_idx < theoretical_spectra.size(); ++theoretical_spectrum_idx)
        {
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
            flow_subgraphs.emplace_back(
                FlowSubgraph(
                    subgraph_node_ids,
                    nodes,
                    edges)
            );




    }

};

