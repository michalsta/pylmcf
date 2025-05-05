#include "py_support.h"
#include "graph_elements.hpp"
#include "spectrum.hpp"


class DecompositableFlowGraph {
    std::vector<FlowNodeVariant> nodes;
    std::vector<FlowEdgeVariant> edges;

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
        const std::vector<py::function*>& dist_funs,
        const std::vector<LEMON_INT>& max_dists
    ) {
        {
            size_t no_nodes = 2 + empirical_spectrum->size();
            for (auto& ts : theoretical_spectra)
                no_nodes += ts->size();
            nodes.reserve(no_nodes);
        }

        // Create placeholder source and sink nodes
        nodes.emplace_back(SourceNode(0));
        nodes.emplace_back(SinkNode(1));

        for (size_t empirical_idx = 0; empirical_idx < empirical_spectrum->size(); ++empirical_idx) {
            nodes.emplace_back(EmpiricalNode(
                                    nodes.size(),
                                    empirical_idx,
                                    empirical_spectrum->intensities[empirical_idx]));
            const auto& empirical_node = std::get<EmpiricalNode>(nodes.back());

            for (size_t theoretical_spectrum_idx = 0; theoretical_spectrum_idx < theoretical_spectra.size(); ++theoretical_spectrum_idx)
            {
                const auto& theoretical_spectrum = theoretical_spectra[theoretical_spectrum_idx];

                for (size_t theoretical_peak_idx = 0; theoretical_peak_idx < theoretical_spectrum->size(); ++theoretical_peak_idx) {
                    nodes.emplace_back(TheoreticalNode(
                                            nodes.size(),
                                            theoretical_spectrum_idx,
                                            theoretical_peak_idx,
                                            theoretical_spectrum->intensities[theoretical_peak_idx]));
                    const auto& theoretical_node = std::get<TheoreticalNode>(nodes.back());
                    const auto& dist_fun = dist_funs[theoretical_spectrum_idx];
                    auto& max_dist = max_dists[theoretical_spectrum_idx];

                    // Calculate the distance between the empirical and theoretical peaks
                    auto [indices, distances] = empirical_spectrum->closer_than(
                        theoretical_spectrum->get_point(theoretical_peak_idx),
                        max_dist,
                        dist_fun
                    );

                    for (size_t ii = 0; ii < indices.size(); ++ii)
                        edges.emplace_back(MatchingEdge(
                            edges.size(),
                            empirical_node,
                            theoretical_node,
                            distances[ii]
                        ));
                }
            }
        }
    };
};