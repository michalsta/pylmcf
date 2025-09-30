#include <span>
#include <iostream>
#include <fstream>
#include <type_traits>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "py_support.hpp"

#include "lmcf.hpp"
#include "graph.hpp"


namespace nb = nanobind;


template <typename T>
nb::ndarray<T, nb::numpy, nb::shape<-1>> py_lmcf(
    nb::ndarray<T, nb::shape<-1>> node_supply,
    nb::ndarray<T, nb::shape<-1>> edges_starts,
    nb::ndarray<T, nb::shape<-1>> edges_ends,
    nb::ndarray<T, nb::shape<-1>> capacities,
    nb::ndarray<T, nb::shape<-1>> costs
    ) {
    auto node_supply_span = numpy_to_span<T>(node_supply);
    auto edges_starts_span = numpy_to_span<T>(edges_starts);
    auto edges_ends_span = numpy_to_span<T>(edges_ends);
    auto capacities_span = numpy_to_span<T>(capacities);
    auto costs_span = numpy_to_span<T>(costs);

    nb::ndarray<T, nb::numpy, nb::shape<-1>> result = create_empty_numpy_array<T>(edges_starts_span.size());
    std::span<T> result_span(static_cast<T*>(result.data()), result.shape(0));
    lmcf(node_supply_span, edges_starts_span, edges_ends_span, capacities_span, costs_span, result_span);

    return result;
}

NB_MODULE(pylmcf_cpp, m) {
    m.doc() = "Python binding for the LEMON min cost flow solver";
    m.def("lmcf", &py_lmcf<int8_t>, "Compute the lmcf for a given graph");
    m.def("lmcf", &py_lmcf<int16_t>, "Compute the lmcf for a given graph");
    m.def("lmcf", &py_lmcf<int32_t>, "Compute the lmcf for a given graph");
    m.def("lmcf", &py_lmcf<int64_t>, "Compute the lmcf for a given graph");


    nb::class_<Graph<int64_t>>(m, "CGraph")
        .def(nb::init<LEMON_INDEX, const nb::ndarray<LEMON_INDEX, nb::shape<-1>> &, const nb::ndarray<LEMON_INDEX, nb::shape<-1>> &>())
        .def("no_nodes", &Graph<int64_t>::no_nodes)
        .def("no_edges", &Graph<int64_t>::no_edges)
        .def("edge_starts", &Graph<int64_t>::edge_starts)
        .def("edge_ends", &Graph<int64_t>::edge_ends)
        .def("set_node_supply", &Graph<int64_t>::set_node_supply_py)
        .def("set_edge_capacities", &Graph<int64_t>::set_edge_capacities_py)
        .def("get_edge_capacities", &Graph<int64_t>::get_edge_capacities_py)
        .def("set_edge_costs", &Graph<int64_t>::set_edge_costs_py)
        .def("get_edge_costs", &Graph<int64_t>::get_edge_costs_py)
        .def("solve", &Graph<int64_t>::solve)
        .def("total_cost", &Graph<int64_t>::total_cost)
        .def("result", &Graph<int64_t>::extract_result_py)
        .def("__str__", &Graph<int64_t>::to_string);
}
