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
        .def("subgraphs", &DecompositableFlowGraph::split_into_subgraphs)
        .def("build", &DecompositableFlowGraph::build)
        .def("set_point", &DecompositableFlowGraph::set_point)
        .def("total_cost", &DecompositableFlowGraph::total_cost);



    m.def("check_spectrum", [](const Spectrum& spectrum) {});
    m.def("check_vspectrum", [](const std::vector<Spectrum*>& spectra) {});

}
