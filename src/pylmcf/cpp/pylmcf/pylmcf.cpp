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
    // Build mode of *this* extension, read by is_nanobind_split() and by the
    // import-time consistency check. NB_BACKEND_MODULE is defined only when
    // nanobind_add_module() was given BACKEND_MODULE, i.e. only in split mode.
    // Extensions in different modes carry different nanobind internals and
    // silently lose sight of each other's registered types, so the mode has to
    // be observable from Python rather than inferred from a filename.
#if defined(NB_BACKEND_MODULE)
    m.attr("nanobind_split") = true;
#else
    m.attr("nanobind_split") = false;
#endif

    m.doc() = "Python binding for the LEMON min cost flow solver";

    using nb::arg;
    // Implicit dtype conversion is DISABLED on every array parameter
    // (noconvert).  nanobind's second overload-resolution pass would
    // otherwise silently convert ANY numeric array that does not exactly
    // match one registered dtype set — float64, mixed integer widths, ... —
    // to the FIRST registered overload (int8), truncating values with
    // wraparound and returning confidently wrong flows.  All arrays of one
    // call must share a single exact signed-integer dtype.
#define PYLMCF_ARGS_NOMIN                                                    \
    arg("node_supply").noconvert(), arg("edge_starts").noconvert(),          \
    arg("edge_ends").noconvert(), arg("capacities").noconvert(),             \
    arg("costs").noconvert()
#define PYLMCF_ARGS_MIN                                                      \
    arg("node_supply").noconvert(), arg("edge_starts").noconvert(),          \
    arg("edge_ends").noconvert(), arg("capacities").noconvert(),             \
    arg("minimums").noconvert(), arg("costs").noconvert()
    // Friendly TypeError raised when no overload matches (registered last in
    // each family, so it is only reached after every real overload failed).
    auto add_dtype_catchall = [&m](const char* name, const char* dtypes) {
        std::string msg =
            std::string("pylmcf.") + name + "(): no overload matched. All "
            "arrays must be 1-D, C-contiguous, on the CPU, and share ONE "
            "exact signed-integer dtype (" + dtypes + "); pass 5 arrays for "
            "no lower bounds or 6 arrays (minimums before costs) with lower "
            "bounds. Implicit dtype conversion is disabled to prevent silent "
            "truncation — convert explicitly, e.g. arr.astype(np.int64).";
        m.def(name, [msg](nb::args, nb::kwargs) -> nb::object {
            throw nb::type_error(msg.c_str());
        });
    };

    // No-minimums overloads registered first so old call sites (5 arrays) continue to work
    m.def("lmcf", &py_lmcf_no_minimums<int8_t>, "Compute the lmcf for a given graph", PYLMCF_ARGS_NOMIN);
    m.def("lmcf", &py_lmcf_no_minimums<int16_t>, "Compute the lmcf for a given graph", PYLMCF_ARGS_NOMIN);
    m.def("lmcf", &py_lmcf_no_minimums<int32_t>, "Compute the lmcf for a given graph", PYLMCF_ARGS_NOMIN);
    m.def("lmcf", &py_lmcf_no_minimums<int64_t>, "Compute the lmcf for a given graph", PYLMCF_ARGS_NOMIN);
    m.def("lmcf", &py_lmcf<int8_t>, "Compute the lmcf for a given graph", PYLMCF_ARGS_MIN);
    m.def("lmcf", &py_lmcf<int16_t>, "Compute the lmcf for a given graph", PYLMCF_ARGS_MIN);
    m.def("lmcf", &py_lmcf<int32_t>, "Compute the lmcf for a given graph", PYLMCF_ARGS_MIN);
    m.def("lmcf", &py_lmcf<int64_t>, "Compute the lmcf for a given graph", PYLMCF_ARGS_MIN);
    add_dtype_catchall("lmcf", "int8/int16/int32/int64");
    // Cycle-canceling variants
    m.def("lmcf_cycle_canceling", &py_lmcf_cycle_canceling_no_minimums<int8_t>, "Compute the lmcf using cycle-canceling for a given graph", PYLMCF_ARGS_NOMIN);
    m.def("lmcf_cycle_canceling", &py_lmcf_cycle_canceling_no_minimums<int16_t>, "Compute the lmcf using cycle-canceling for a given graph", PYLMCF_ARGS_NOMIN);
    m.def("lmcf_cycle_canceling", &py_lmcf_cycle_canceling_no_minimums<int32_t>, "Compute the lmcf using cycle-canceling for a given graph", PYLMCF_ARGS_NOMIN);
    m.def("lmcf_cycle_canceling", &py_lmcf_cycle_canceling_no_minimums<int64_t>, "Compute the lmcf using cycle-canceling for a given graph", PYLMCF_ARGS_NOMIN);
    m.def("lmcf_cycle_canceling", &py_lmcf_cycle_canceling<int8_t>, "Compute the lmcf using cycle-canceling for a given graph", PYLMCF_ARGS_MIN);
    m.def("lmcf_cycle_canceling", &py_lmcf_cycle_canceling<int16_t>, "Compute the lmcf using cycle-canceling for a given graph", PYLMCF_ARGS_MIN);
    m.def("lmcf_cycle_canceling", &py_lmcf_cycle_canceling<int32_t>, "Compute the lmcf using cycle-canceling for a given graph", PYLMCF_ARGS_MIN);
    m.def("lmcf_cycle_canceling", &py_lmcf_cycle_canceling<int64_t>, "Compute the lmcf using cycle-canceling for a given graph", PYLMCF_ARGS_MIN);
    add_dtype_catchall("lmcf_cycle_canceling", "int8/int16/int32/int64");
    // Cost-scaling variants (int32/int64 only — small types lack required arithmetic range)
    m.def("lmcf_cost_scaling", &py_lmcf_cost_scaling_no_minimums<int32_t>, "Compute the lmcf using cost-scaling for a given graph", PYLMCF_ARGS_NOMIN);
    m.def("lmcf_cost_scaling", &py_lmcf_cost_scaling_no_minimums<int64_t>, "Compute the lmcf using cost-scaling for a given graph", PYLMCF_ARGS_NOMIN);
    m.def("lmcf_cost_scaling", &py_lmcf_cost_scaling<int32_t>, "Compute the lmcf using cost-scaling for a given graph", PYLMCF_ARGS_MIN);
    m.def("lmcf_cost_scaling", &py_lmcf_cost_scaling<int64_t>, "Compute the lmcf using cost-scaling for a given graph", PYLMCF_ARGS_MIN);
    add_dtype_catchall("lmcf_cost_scaling", "int32/int64");
    // Capacity-scaling variants (int32/int64 only — small types lack required arithmetic range)
    m.def("lmcf_capacity_scaling", &py_lmcf_capacity_scaling_no_minimums<int32_t>, "Compute the lmcf using capacity-scaling for a given graph", PYLMCF_ARGS_NOMIN);
    m.def("lmcf_capacity_scaling", &py_lmcf_capacity_scaling_no_minimums<int64_t>, "Compute the lmcf using capacity-scaling for a given graph", PYLMCF_ARGS_NOMIN);
    m.def("lmcf_capacity_scaling", &py_lmcf_capacity_scaling<int32_t>, "Compute the lmcf using capacity-scaling for a given graph", PYLMCF_ARGS_MIN);
    m.def("lmcf_capacity_scaling", &py_lmcf_capacity_scaling<int64_t>, "Compute the lmcf using capacity-scaling for a given graph", PYLMCF_ARGS_MIN);
    add_dtype_catchall("lmcf_capacity_scaling", "int32/int64");
#undef PYLMCF_ARGS_NOMIN
#undef PYLMCF_ARGS_MIN


    nb::class_<Graph<int64_t>>(m, "CGraph")
        // Edge-index arrays: exact int32 (LEMON_INDEX) or exact int64 (via a
        // checked narrowing overload in Graph — numpy's default int dtype).
        // Anything else (float, unsigned, ...) is rejected: noconvert.
        .def(nb::init<LEMON_INDEX, const ndarray_1d<LEMON_INDEX> &, const ndarray_1d<LEMON_INDEX> &>(),
             arg("no_nodes"), arg("edge_starts").noconvert(), arg("edge_ends").noconvert())
        .def(nb::init<LEMON_INDEX, const ndarray_1d<int64_t> &, const ndarray_1d<int64_t> &>(),
             arg("no_nodes"), arg("edge_starts").noconvert(), arg("edge_ends").noconvert())
        .def("no_nodes", &Graph<int64_t>::no_nodes)
        .def("no_edges", &Graph<int64_t>::no_edges)
        .def("edge_starts", &Graph<int64_t>::edge_starts_py)
        .def("edge_ends", &Graph<int64_t>::edge_ends_py)
        .def("set_node_supply", &Graph<int64_t>::set_node_supply_py, arg("supply").noconvert())
        .def("get_node_supply", &Graph<int64_t>::get_node_supply_py)
        .def("set_edge_capacities", &Graph<int64_t>::set_edge_capacities_py, arg("capacities").noconvert())
        .def("get_edge_capacities", &Graph<int64_t>::get_edge_capacities_py)
        .def("set_edge_minimums", &Graph<int64_t>::set_edge_minimums_py, arg("minimums").noconvert())
        .def("get_edge_minimums", &Graph<int64_t>::get_edge_minimums_py)
        .def("set_edge_costs", &Graph<int64_t>::set_edge_costs_py, arg("costs").noconvert())
        .def("get_edge_costs", &Graph<int64_t>::get_edge_costs_py)
        .def("solve", &Graph<int64_t>::solve)
        .def("warm_start_count", &Graph<int64_t>::warm_start_count)
        .def("cold_start_count", &Graph<int64_t>::cold_start_count)
        .def("dual_repair_count", &Graph<int64_t>::dual_repair_count)
        .def("primal_repair_count", &Graph<int64_t>::primal_repair_count)
        .def("policy_cold_count", &Graph<int64_t>::policy_cold_count)
        .def("set_warm_violation_limit", &Graph<int64_t>::set_warm_violation_limit)
        .def("total_cost", &Graph<int64_t>::total_cost)
        .def("result", &Graph<int64_t>::extract_result_py)
        .def("__str__", &Graph<int64_t>::to_string);
}
