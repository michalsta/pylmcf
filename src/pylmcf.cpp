#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <span>
#include <iostream>
#include <fstream>
#include <type_traits>

#include "lmcf.cpp" // Yes, ugly. But it works.
#include "graph.cpp"


namespace py = pybind11;

template <typename T>
std::span<T> numpy_to_span(py::array_t<T> array) {
    py::buffer_info info = array.request();
    if (info.ndim != 1) {
        throw std::invalid_argument("Only 1D arrays are supported");
    }
    return std::span<T>(static_cast<T*>(info.ptr), info.size);
}

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
    py::class_<Graph<uint64_t>>(m, "Graph")
        .def(py::init<size_t, const std::span<size_t>&, const std::span<size_t>&, const std::span<uint64_t>&>())
        .def("no_nodes", &Graph<uint64_t>::no_nodes)
        .def("no_edges", &Graph<uint64_t>::no_edges)
        .def("set_node_supply", &Graph<uint64_t>::set_node_supply)
        .def("set_edge_capacities", &Graph<uint64_t>::set_edge_capacities)
        .def("solve", &Graph<uint64_t>::solve)
        .def("total_cost", &Graph<uint64_t>::total_cost)
        .def("result", &Graph<uint64_t>::extract_result);
;
}
