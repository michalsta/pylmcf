#include <span>
#include <iostream>
#include <fstream>
#include <type_traits>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/bind_vector.h>

#include "py_support.hpp"

#include "lmcf.hpp"
#include "graph.hpp"


namespace nb = nanobind;


template <typename T>
nb::ndarray<T, nb::numpy, nb::shape<-1>> py_lmcf(
    ndarray_1d<T> node_supply,
    ndarray_1d<T> edges_starts,
    ndarray_1d<T> edges_ends,
    ndarray_1d<T> capacities,
    ndarray_1d<T> minimums,
    ndarray_1d<T> costs
    ) {
    auto node_supply_span = numpy_to_span<T>(node_supply);
    auto edges_starts_span = numpy_to_span<T>(edges_starts);
    auto edges_ends_span = numpy_to_span<T>(edges_ends);
    auto capacities_span = numpy_to_span<T>(capacities);
    auto minimums_span = numpy_to_span<T>(minimums);
    auto costs_span = numpy_to_span<T>(costs);

    nb::ndarray<T, nb::numpy, nb::shape<-1>> result = create_empty_numpy_array<T>(edges_starts_span.size());
    std::span<T> result_span(static_cast<T*>(result.data()), result.shape(0));
    lmcf(node_supply_span, edges_starts_span, edges_ends_span, capacities_span, minimums_span, costs_span, result_span);

    return result;
}

template <typename T>
nb::ndarray<T, nb::numpy, nb::shape<-1>> py_lmcf_no_minimums(
    ndarray_1d<T> node_supply,
    ndarray_1d<T> edges_starts,
    ndarray_1d<T> edges_ends,
    ndarray_1d<T> capacities,
    ndarray_1d<T> costs
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

template <typename T>
nb::ndarray<T, nb::numpy, nb::shape<-1>> py_lmcf_cycle_canceling(
    ndarray_1d<T> node_supply,
    ndarray_1d<T> edges_starts,
    ndarray_1d<T> edges_ends,
    ndarray_1d<T> capacities,
    ndarray_1d<T> minimums,
    ndarray_1d<T> costs
    ) {
    auto node_supply_span = numpy_to_span<T>(node_supply);
    auto edges_starts_span = numpy_to_span<T>(edges_starts);
    auto edges_ends_span = numpy_to_span<T>(edges_ends);
    auto capacities_span = numpy_to_span<T>(capacities);
    auto minimums_span = numpy_to_span<T>(minimums);
    auto costs_span = numpy_to_span<T>(costs);

    nb::ndarray<T, nb::numpy, nb::shape<-1>> result = create_empty_numpy_array<T>(edges_starts_span.size());
    std::span<T> result_span(static_cast<T*>(result.data()), result.shape(0));
    lmcf_cycle_canceling(node_supply_span, edges_starts_span, edges_ends_span, capacities_span, minimums_span, costs_span, result_span);

    return result;
}

template <typename T>
nb::ndarray<T, nb::numpy, nb::shape<-1>> py_lmcf_cycle_canceling_no_minimums(
    ndarray_1d<T> node_supply,
    ndarray_1d<T> edges_starts,
    ndarray_1d<T> edges_ends,
    ndarray_1d<T> capacities,
    ndarray_1d<T> costs
    ) {
    auto node_supply_span = numpy_to_span<T>(node_supply);
    auto edges_starts_span = numpy_to_span<T>(edges_starts);
    auto edges_ends_span = numpy_to_span<T>(edges_ends);
    auto capacities_span = numpy_to_span<T>(capacities);
    auto costs_span = numpy_to_span<T>(costs);

    nb::ndarray<T, nb::numpy, nb::shape<-1>> result = create_empty_numpy_array<T>(edges_starts_span.size());
    std::span<T> result_span(static_cast<T*>(result.data()), result.shape(0));
    lmcf_cycle_canceling(node_supply_span, edges_starts_span, edges_ends_span, capacities_span, costs_span, result_span);

    return result;
}

template <typename T>
nb::ndarray<T, nb::numpy, nb::shape<-1>> py_lmcf_cost_scaling(
    ndarray_1d<T> node_supply,
    ndarray_1d<T> edges_starts,
    ndarray_1d<T> edges_ends,
    ndarray_1d<T> capacities,
    ndarray_1d<T> minimums,
    ndarray_1d<T> costs
    ) {
    auto node_supply_span = numpy_to_span<T>(node_supply);
    auto edges_starts_span = numpy_to_span<T>(edges_starts);
    auto edges_ends_span = numpy_to_span<T>(edges_ends);
    auto capacities_span = numpy_to_span<T>(capacities);
    auto minimums_span = numpy_to_span<T>(minimums);
    auto costs_span = numpy_to_span<T>(costs);

    nb::ndarray<T, nb::numpy, nb::shape<-1>> result = create_empty_numpy_array<T>(edges_starts_span.size());
    std::span<T> result_span(static_cast<T*>(result.data()), result.shape(0));
    lmcf_cost_scaling(node_supply_span, edges_starts_span, edges_ends_span, capacities_span, minimums_span, costs_span, result_span);

    return result;
}

template <typename T>
nb::ndarray<T, nb::numpy, nb::shape<-1>> py_lmcf_cost_scaling_no_minimums(
    ndarray_1d<T> node_supply,
    ndarray_1d<T> edges_starts,
    ndarray_1d<T> edges_ends,
    ndarray_1d<T> capacities,
    ndarray_1d<T> costs
    ) {
    auto node_supply_span = numpy_to_span<T>(node_supply);
    auto edges_starts_span = numpy_to_span<T>(edges_starts);
    auto edges_ends_span = numpy_to_span<T>(edges_ends);
    auto capacities_span = numpy_to_span<T>(capacities);
    auto costs_span = numpy_to_span<T>(costs);

    nb::ndarray<T, nb::numpy, nb::shape<-1>> result = create_empty_numpy_array<T>(edges_starts_span.size());
    std::span<T> result_span(static_cast<T*>(result.data()), result.shape(0));
    lmcf_cost_scaling(node_supply_span, edges_starts_span, edges_ends_span, capacities_span, costs_span, result_span);

    return result;
}

template <typename T>
nb::ndarray<T, nb::numpy, nb::shape<-1>> py_lmcf_capacity_scaling(
    ndarray_1d<T> node_supply,
    ndarray_1d<T> edges_starts,
    ndarray_1d<T> edges_ends,
    ndarray_1d<T> capacities,
    ndarray_1d<T> minimums,
    ndarray_1d<T> costs
    ) {
    auto node_supply_span = numpy_to_span<T>(node_supply);
    auto edges_starts_span = numpy_to_span<T>(edges_starts);
    auto edges_ends_span = numpy_to_span<T>(edges_ends);
    auto capacities_span = numpy_to_span<T>(capacities);
    auto minimums_span = numpy_to_span<T>(minimums);
    auto costs_span = numpy_to_span<T>(costs);

    nb::ndarray<T, nb::numpy, nb::shape<-1>> result = create_empty_numpy_array<T>(edges_starts_span.size());
    std::span<T> result_span(static_cast<T*>(result.data()), result.shape(0));
    lmcf_capacity_scaling(node_supply_span, edges_starts_span, edges_ends_span, capacities_span, minimums_span, costs_span, result_span);

    return result;
}

template <typename T>
nb::ndarray<T, nb::numpy, nb::shape<-1>> py_lmcf_capacity_scaling_no_minimums(
    ndarray_1d<T> node_supply,
    ndarray_1d<T> edges_starts,
    ndarray_1d<T> edges_ends,
    ndarray_1d<T> capacities,
    ndarray_1d<T> costs
    ) {
    auto node_supply_span = numpy_to_span<T>(node_supply);
    auto edges_starts_span = numpy_to_span<T>(edges_starts);
    auto edges_ends_span = numpy_to_span<T>(edges_ends);
    auto capacities_span = numpy_to_span<T>(capacities);
    auto costs_span = numpy_to_span<T>(costs);

    nb::ndarray<T, nb::numpy, nb::shape<-1>> result = create_empty_numpy_array<T>(edges_starts_span.size());
    std::span<T> result_span(static_cast<T*>(result.data()), result.shape(0));
    lmcf_capacity_scaling(node_supply_span, edges_starts_span, edges_ends_span, capacities_span, costs_span, result_span);

    return result;
}

NB_MODULE(pylmcf_cpp, m) {
    m.doc() = "Python binding for the LEMON min cost flow solver";
    // No-minimums overloads registered first so old call sites (5 arrays) continue to work
    m.def("lmcf", &py_lmcf_no_minimums<int8_t>, "Compute the lmcf for a given graph");
    m.def("lmcf", &py_lmcf_no_minimums<int16_t>, "Compute the lmcf for a given graph");
    m.def("lmcf", &py_lmcf_no_minimums<int32_t>, "Compute the lmcf for a given graph");
    m.def("lmcf", &py_lmcf_no_minimums<int64_t>, "Compute the lmcf for a given graph");
    m.def("lmcf", &py_lmcf<int8_t>, "Compute the lmcf for a given graph");
    m.def("lmcf", &py_lmcf<int16_t>, "Compute the lmcf for a given graph");
    m.def("lmcf", &py_lmcf<int32_t>, "Compute the lmcf for a given graph");
    m.def("lmcf", &py_lmcf<int64_t>, "Compute the lmcf for a given graph");
    // Cycle-canceling variants
    m.def("lmcf_cycle_canceling", &py_lmcf_cycle_canceling_no_minimums<int8_t>, "Compute the lmcf using cycle-canceling for a given graph");
    m.def("lmcf_cycle_canceling", &py_lmcf_cycle_canceling_no_minimums<int16_t>, "Compute the lmcf using cycle-canceling for a given graph");
    m.def("lmcf_cycle_canceling", &py_lmcf_cycle_canceling_no_minimums<int32_t>, "Compute the lmcf using cycle-canceling for a given graph");
    m.def("lmcf_cycle_canceling", &py_lmcf_cycle_canceling_no_minimums<int64_t>, "Compute the lmcf using cycle-canceling for a given graph");
    m.def("lmcf_cycle_canceling", &py_lmcf_cycle_canceling<int8_t>, "Compute the lmcf using cycle-canceling for a given graph");
    m.def("lmcf_cycle_canceling", &py_lmcf_cycle_canceling<int16_t>, "Compute the lmcf using cycle-canceling for a given graph");
    m.def("lmcf_cycle_canceling", &py_lmcf_cycle_canceling<int32_t>, "Compute the lmcf using cycle-canceling for a given graph");
    m.def("lmcf_cycle_canceling", &py_lmcf_cycle_canceling<int64_t>, "Compute the lmcf using cycle-canceling for a given graph");
    // Cost-scaling variants (int32/int64 only — small types lack required arithmetic range)
    m.def("lmcf_cost_scaling", &py_lmcf_cost_scaling_no_minimums<int32_t>, "Compute the lmcf using cost-scaling for a given graph");
    m.def("lmcf_cost_scaling", &py_lmcf_cost_scaling_no_minimums<int64_t>, "Compute the lmcf using cost-scaling for a given graph");
    m.def("lmcf_cost_scaling", &py_lmcf_cost_scaling<int32_t>, "Compute the lmcf using cost-scaling for a given graph");
    m.def("lmcf_cost_scaling", &py_lmcf_cost_scaling<int64_t>, "Compute the lmcf using cost-scaling for a given graph");
    // Capacity-scaling variants (int32/int64 only — small types lack required arithmetic range)
    m.def("lmcf_capacity_scaling", &py_lmcf_capacity_scaling_no_minimums<int32_t>, "Compute the lmcf using capacity-scaling for a given graph");
    m.def("lmcf_capacity_scaling", &py_lmcf_capacity_scaling_no_minimums<int64_t>, "Compute the lmcf using capacity-scaling for a given graph");
    m.def("lmcf_capacity_scaling", &py_lmcf_capacity_scaling<int32_t>, "Compute the lmcf using capacity-scaling for a given graph");
    m.def("lmcf_capacity_scaling", &py_lmcf_capacity_scaling<int64_t>, "Compute the lmcf using capacity-scaling for a given graph");


    nb::class_<Graph<int64_t>>(m, "CGraph")
        .def(nb::init<LEMON_INDEX, const ndarray_1d<LEMON_INDEX> &, const ndarray_1d<LEMON_INDEX> &>())
        .def("no_nodes", &Graph<int64_t>::no_nodes)
        .def("no_edges", &Graph<int64_t>::no_edges)
        .def("edge_starts", &Graph<int64_t>::edge_starts_py)
        .def("edge_ends", &Graph<int64_t>::edge_ends_py)
        .def("set_node_supply", &Graph<int64_t>::set_node_supply_py)
        .def("get_node_supply", &Graph<int64_t>::get_node_supply_py)
        .def("set_edge_capacities", &Graph<int64_t>::set_edge_capacities_py)
        .def("get_edge_capacities", &Graph<int64_t>::get_edge_capacities_py)
        .def("set_edge_minimums", &Graph<int64_t>::set_edge_minimums_py)
        .def("get_edge_minimums", &Graph<int64_t>::get_edge_minimums_py)
        .def("set_edge_costs", &Graph<int64_t>::set_edge_costs_py)
        .def("get_edge_costs", &Graph<int64_t>::get_edge_costs_py)
        .def("solve", &Graph<int64_t>::solve)
        .def("total_cost", &Graph<int64_t>::total_cost)
        .def("result", &Graph<int64_t>::extract_result_py)
        .def("__str__", &Graph<int64_t>::to_string);
}
