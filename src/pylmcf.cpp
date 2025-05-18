#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <span>
#include <iostream>
#include <fstream>
#include <type_traits>

#include "py_support.h"

#include "lmcf.cpp" // Yes, ugly. But it works.
#include "graph.cpp"
#include "spectrum.hpp"
#include "decompositable_graph.hpp"
#include "graph_elements.hpp"


template <typename T>
py::array_t<T> py_lmcf(
    py::array_t<T> node_supply,
    py::array_t<T> edges_starts,
    py::array_t<T> edges_ends,
    py::array_t<T> capacities,
    py::array_t<T> costs
    ) {
    auto node_supply_span = numpy_to_span(node_supply);
    auto edges_starts_span = numpy_to_span(edges_starts);
    auto edges_ends_span = numpy_to_span(edges_ends);
    auto capacities_span = numpy_to_span(capacities);
    auto costs_span = numpy_to_span(costs);

    py::array_t<T> result(edges_starts_span.size());
    lmcf(node_supply_span, edges_starts_span, edges_ends_span, capacities_span, costs_span, numpy_to_span(result));

    return result;
}


PYBIND11_MODULE(pylmcf_cpp, m) {
    m.doc() = "Python binding for the lmcf algorithm implemented in C++";
/*    m.def("lmcf", &py_lmcf<int8_t>, "Compute the lmcf for a given graph");
    m.def("lmcf", &py_lmcf<int16_t>, "Compute the lmcf for a given graph");
    m.def("lmcf", &py_lmcf<int32_t>, "Compute the lmcf for a given graph");
    m.def("lmcf", &py_lmcf<int64_t>, "Compute the lmcf for a given graph");
 */   m.def("lmcf", &py_lmcf<int8_t>, "Compute the lmcf for a given graph");
    m.def("lmcf", &py_lmcf<int16_t>, "Compute the lmcf for a given graph");
    m.def("lmcf", &py_lmcf<int32_t>, "Compute the lmcf for a given graph");
    m.def("lmcf", &py_lmcf<int64_t>, "Compute the lmcf for a given graph");
 /*   m.def("lmcf", &py_lmcf<float>, "Compute the lmcf for a given graph");
    m.def("lmcf", &py_lmcf<double>, "Compute the lmcf for a given graph");
*/
    py::class_<Graph<int64_t>>(m, "LemonGraph")
        .def(py::init<size_t, const py::array_t<int64_t> &, const py::array_t<int64_t> &, const py::array_t<uint64_t> &>())
        .def("no_nodes", &Graph<int64_t>::no_nodes)
        .def("no_edges", &Graph<int64_t>::no_edges)
        .def("set_node_supply", &Graph<int64_t>::set_node_supply_py)
        .def("set_edge_capacities", &Graph<int64_t>::set_edge_capacities_py)
        .def("solve", &Graph<int64_t>::solve)
        .def("total_cost", &Graph<int64_t>::total_cost)
        .def("result", &Graph<int64_t>::extract_result_py)
        .def("__str__", &Graph<int64_t>::to_string);
;

    py::class_<Spectrum>(m, "CSpectrum")
        .def(py::init<py::array, py::array_t<LEMON_INT>>())
        .def("size", &Spectrum::size)
        .def("intensities", &Spectrum::py_get_intensities)
        .def("positions", &Spectrum::py_get_positions)
        .def("get_point", &Spectrum::get_point)
        .def("closer_than", &Spectrum::closer_than);

    py::class_<FlowNode>(m, "CFlowNode")
        .def("id", &FlowNode::get_id)
        .def("layer", &FlowNode::layer);

    py::class_<FlowEdge>(m, "CFlowEdge")
        .def("start_node_id", &FlowEdge::get_start_node_id)
        .def("end_node_id", &FlowEdge::get_end_node_id);


    py::class_<DecompositableFlowGraph>(m, "CDecompositableFlowGraph")
        .def(py::init<const Spectrum*, const std::vector<Spectrum*>&, const py::function*, LEMON_INT>())
        .def("no_nodes", &DecompositableFlowGraph::no_nodes)
        .def("no_edges", &DecompositableFlowGraph::no_edges)
        .def("no_theoretical_spectra", &DecompositableFlowGraph::no_theoretical_spectra)
        .def("nodes", &DecompositableFlowGraph::get_nodes)
        .def("edges", &DecompositableFlowGraph::get_edges)
        .def("split_into_subgraphs", &DecompositableFlowGraph::split_into_subgraphs)
        .def("build", &DecompositableFlowGraph::build)
        .def("set_point", &DecompositableFlowGraph::set_point)
        .def("total_cost", &DecompositableFlowGraph::total_cost)
        .def("add_simple_trash", &DecompositableFlowGraph::add_simple_trash)
        .def("neighbourhood_lists", &DecompositableFlowGraph::neighbourhood_lists)
        .def("no_subgraphs", &DecompositableFlowGraph::no_subgraphs)
        .def("get_subgraph", &DecompositableFlowGraph::get_subgraph, py::return_value_policy::reference)
        .def("__str__", &DecompositableFlowGraph::to_string)
        .def("lemon_to_string", &DecompositableFlowGraph::lemon_to_string)
        .def("flows_for_spectrum", [](DecompositableFlowGraph& self, size_t spectrum_id) {
            auto [empirical_peak_indices, theoretical_peak_indices, flows] = self.flows_for_spectrum(spectrum_id);
            return std::make_tuple(vector_to_numpy_copy(empirical_peak_indices),
                                   vector_to_numpy_copy(theoretical_peak_indices),
                                   vector_to_numpy_copy(flows));
        }, py::return_value_policy::move)
        .def("count_empirical_nodes", &DecompositableFlowGraph::count_nodes_of_type<EmpiricalNode>)
        .def("count_theoretical_nodes", &DecompositableFlowGraph::count_nodes_of_type<TheoreticalNode>)
        .def("count_matching_edges", &DecompositableFlowGraph::count_edges_of_type<MatchingEdge>)
        .def("count_theoretical_to_sink_edges", &DecompositableFlowGraph::count_edges_of_type<TheoreticalToSinkEdge>)
        .def("count_src_to_empirical_edges", &DecompositableFlowGraph::count_edges_of_type<SrcToEmpiricalEdge>)
        .def("count_simple_trash_edges", &DecompositableFlowGraph::count_edges_of_type<SimpleTrashEdge>)
        .def("matching_density", &DecompositableFlowGraph::matching_density);

    py::class_<FlowSubgraph>(m, "CFlowSubgraph")
        .def("no_nodes", &FlowSubgraph::no_nodes)
        .def("no_edges", &FlowSubgraph::no_edges)
        .def("nodes", &FlowSubgraph::get_nodes)
        .def("edges", &FlowSubgraph::get_edges)
        .def("build", &FlowSubgraph::build)
        .def("set_point", &FlowSubgraph::set_point)
        .def("total_cost", &FlowSubgraph::total_cost)
        .def("add_simple_trash", &FlowSubgraph::add_simple_trash)
        .def("lemon_to_string", &FlowSubgraph::lemon_to_string)
        .def("to_string", &FlowSubgraph::to_string)
        .def("count_empirical_nodes", &FlowSubgraph::count_nodes_of_type<EmpiricalNode>)
        .def("count_theoretical_nodes", &FlowSubgraph::count_nodes_of_type<TheoreticalNode>)
        .def("count_matching_edges", &FlowSubgraph::count_edges_of_type<MatchingEdge>)
        .def("count_theoretical_to_sink_edges", &FlowSubgraph::count_edges_of_type<TheoreticalToSinkEdge>)
        .def("count_src_to_empirical_edges", &FlowSubgraph::count_edges_of_type<SrcToEmpiricalEdge>)
        .def("count_simple_trash_edges", &FlowSubgraph::count_edges_of_type<SimpleTrashEdge>)
        .def("matching_density", &FlowSubgraph::matching_density);

    py::class_<lemon::StaticDigraph>(m, "LemonStaticGraph")
        .def("no_nodes", &lemon::StaticDigraph::nodeNum)
        .def("no_edges", &lemon::StaticDigraph::arcNum);


    /* py::class_<lemon::NetworkSimplex<lemon::StaticDigraph, int64_t, int64_t>>(m, "LemonNetworkSimplex")
        .def(py::init<lemon::StaticDigraph&>())
        .def("supply_map", &lemon::NetworkSimplex<lemon::StaticDigraph, int64_t, int64_t>::supplyMap)
        .def("cost_map", &lemon::NetworkSimplex<lemon::StaticDigraph, int64_t, int64_t>::costMap)
        .def("upper_map", &lemon::NetworkSimplex<lemon::StaticDigraph, int64_t, int64_t>::upperMap)
        .def("run", &lemon::NetworkSimplex<lemon::StaticDigraph, int64_t, int64_t>::run)
        .def("total_cost", &lemon::NetworkSimplex<lemon::StaticDigraph, int64_t, int64_t>::totalCost)
        .def("flow", &lemon::NetworkSimplex<lemon::StaticDigraph, int64_t, int64_t>::flow);
    */
    m.def("check_spectrum", [](const Spectrum& spectrum) {});
    m.def("check_vspectrum", [](const std::vector<Spectrum*>& spectra) {});

}
