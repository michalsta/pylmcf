#include <span>
#include <iostream>
#include <fstream>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/bind_vector.h>

#include "py_support.hpp"

#include "lmcf.hpp"
#include "graph.hpp"
#include "new_solver_bindings.hpp"


namespace nb = nanobind;


// One wrapper per arity, parameterised by the LEMON solver.  Flows come back
// in the caller's dtype; potentials (the dual solution, one per node) are
// always int64 because costs are accumulated in LmcfCost (see lmcf.hpp) and
// path potentials overflow a narrow dtype long before the flows do.
template <typename T>
nb::object wrap_result(nb::ndarray<T, nb::numpy, nb::shape<-1>> flows,
                       std::optional<nb::ndarray<LmcfCost, nb::numpy, nb::shape<-1>>> potentials) {
    if (!potentials)
        return nb::cast(flows);
    return nb::make_tuple(flows, *potentials);
}

template <template <typename...> class Solver, bool validate_costs, typename T>
nb::object py_mcf(
    ndarray_1d<T> node_supply,
    ndarray_1d<T> edges_starts,
    ndarray_1d<T> edges_ends,
    ndarray_1d<T> capacities,
    ndarray_1d<T> minimums,
    ndarray_1d<T> costs,
    bool return_potentials
    ) {
    auto node_supply_span = numpy_to_span<T>(node_supply);
    auto edges_starts_span = numpy_to_span<T>(edges_starts);
    auto minimums_span = numpy_to_span<T>(minimums);

    nb::ndarray<T, nb::numpy, nb::shape<-1>> result = create_empty_numpy_array<T>(edges_starts_span.size());
    std::span<T> result_span(static_cast<T*>(result.data()), result.shape(0));

    std::optional<nb::ndarray<LmcfCost, nb::numpy, nb::shape<-1>>> potentials;
    std::span<LmcfCost> potentials_span;
    if (return_potentials) {
        potentials = create_empty_numpy_array<LmcfCost>(node_supply_span.size());
        potentials_span = std::span<LmcfCost>(static_cast<LmcfCost*>(potentials->data()), potentials->shape(0));
    }

    lmcf_impl<Solver, T, validate_costs>(
        node_supply_span, edges_starts_span, numpy_to_span<T>(edges_ends),
        numpy_to_span<T>(capacities), minimums_span, numpy_to_span<T>(costs),
        result_span, potentials_span);

    return wrap_result<T>(result, potentials);
}

template <template <typename...> class Solver, bool validate_costs, typename T>
nb::object py_mcf_no_minimums(
    ndarray_1d<T> node_supply,
    ndarray_1d<T> edges_starts,
    ndarray_1d<T> edges_ends,
    ndarray_1d<T> capacities,
    ndarray_1d<T> costs,
    bool return_potentials
    ) {
    auto node_supply_span = numpy_to_span<T>(node_supply);
    auto edges_starts_span = numpy_to_span<T>(edges_starts);

    nb::ndarray<T, nb::numpy, nb::shape<-1>> result = create_empty_numpy_array<T>(edges_starts_span.size());
    std::span<T> result_span(static_cast<T*>(result.data()), result.shape(0));

    std::optional<nb::ndarray<LmcfCost, nb::numpy, nb::shape<-1>>> potentials;
    std::span<LmcfCost> potentials_span;
    if (return_potentials) {
        potentials = create_empty_numpy_array<LmcfCost>(node_supply_span.size());
        potentials_span = std::span<LmcfCost>(static_cast<LmcfCost*>(potentials->data()), potentials->shape(0));
    }

    lmcf_impl<Solver, T, validate_costs>(
        node_supply_span, edges_starts_span, numpy_to_span<T>(edges_ends),
        numpy_to_span<T>(capacities), std::span<T>{}, numpy_to_span<T>(costs),
        result_span, potentials_span);

    return wrap_result<T>(result, potentials);
}

using GraphSolver = Graph<int64_t>::Solver;

// Python spells LEMON's enums as lowercase strings.
const std::pair<const char*, GraphSolver::SupplyType> supply_type_names[] = {
    {"geq", GraphSolver::GEQ},
    {"leq", GraphSolver::LEQ},
};

const std::pair<const char*, GraphSolver::PivotRule> pivot_rule_names[] = {
    {"first_eligible", GraphSolver::FIRST_ELIGIBLE},
    {"best_eligible", GraphSolver::BEST_ELIGIBLE},
    {"block_search", GraphSolver::BLOCK_SEARCH},
    {"candidate_list", GraphSolver::CANDIDATE_LIST},
    {"altering_list", GraphSolver::ALTERING_LIST},
};

const std::pair<const char*, GraphSolver::WarmRepair> warm_repair_names[] = {
    {"repair_only", GraphSolver::WarmRepair::RepairOnly},
    {"dual", GraphSolver::WarmRepair::Dual},
    {"primal", GraphSolver::WarmRepair::Primal},
    {"dual_ratio", GraphSolver::WarmRepair::DualRatio},
    {"dual_greedy", GraphSolver::WarmRepair::DualGreedy},
};

template <typename E, size_t N>
E enum_from_name(const std::pair<const char*, E> (&table)[N], const std::string& name, const char* what) {
    std::string valid;
    for (const auto& [n, v] : table) {
        if (name == n) return v;
        valid += std::string(valid.empty() ? "" : ", ") + "'" + n + "'";
    }
    throw std::invalid_argument(std::string(what) + " must be one of " + valid + ", got '" + name + "'");
}

template <typename E, size_t N>
std::string enum_name(const std::pair<const char*, E> (&table)[N], E value) {
    for (const auto& [n, v] : table)
        if (v == value) return n;
    throw std::logic_error("unnamed enum value");
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

    pylmcf_python::bind_new_solvers(m);

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
    arg("costs").noconvert(), arg("return_potentials") = false
#define PYLMCF_ARGS_MIN                                                      \
    arg("node_supply").noconvert(), arg("edge_starts").noconvert(),          \
    arg("edge_ends").noconvert(), arg("capacities").noconvert(),             \
    arg("minimums").noconvert(), arg("costs").noconvert(),                \
    arg("return_potentials") = false
    // Friendly TypeError raised when no overload matches (registered last in
    // each family, so it is only reached after every real overload failed).
    auto add_dtype_catchall = [&m](const char* name, const char* dtypes) {
        std::string msg =
            std::string("pylmcf.") + name + "(): no overload matched. All "
            "arrays must be 1-D, C-contiguous, on the CPU, and share ONE "
            "exact signed-integer dtype (" + dtypes + "); pass 5 arrays for "
            "no lower bounds or 6 arrays (minimums before costs) with lower "
            "bounds, optionally followed by return_potentials=True. Implicit dtype conversion is disabled to prevent silent "
            "truncation — convert explicitly, e.g. arr.astype(np.int64).";
        m.def(name, [msg](nb::args, nb::kwargs) -> nb::object {
            throw nb::type_error(msg.c_str());
        });
    };

    // No-minimums overloads registered first so old call sites (5 arrays) continue to work
    m.def("lmcf", &py_mcf_no_minimums<lemon::NetworkSimplex, true, int8_t>, "Compute the lmcf for a given graph", PYLMCF_ARGS_NOMIN);
    m.def("lmcf", &py_mcf_no_minimums<lemon::NetworkSimplex, true, int16_t>, "Compute the lmcf for a given graph", PYLMCF_ARGS_NOMIN);
    m.def("lmcf", &py_mcf_no_minimums<lemon::NetworkSimplex, true, int32_t>, "Compute the lmcf for a given graph", PYLMCF_ARGS_NOMIN);
    m.def("lmcf", &py_mcf_no_minimums<lemon::NetworkSimplex, true, int64_t>, "Compute the lmcf for a given graph", PYLMCF_ARGS_NOMIN);
    m.def("lmcf", &py_mcf<lemon::NetworkSimplex, true, int8_t>, "Compute the lmcf for a given graph", PYLMCF_ARGS_MIN);
    m.def("lmcf", &py_mcf<lemon::NetworkSimplex, true, int16_t>, "Compute the lmcf for a given graph", PYLMCF_ARGS_MIN);
    m.def("lmcf", &py_mcf<lemon::NetworkSimplex, true, int32_t>, "Compute the lmcf for a given graph", PYLMCF_ARGS_MIN);
    m.def("lmcf", &py_mcf<lemon::NetworkSimplex, true, int64_t>, "Compute the lmcf for a given graph", PYLMCF_ARGS_MIN);
    add_dtype_catchall("lmcf", "int8/int16/int32/int64");
    // Cycle-canceling variants
    m.def("lmcf_cycle_canceling", &py_mcf_no_minimums<lemon::CycleCanceling, false, int8_t>, "Compute the lmcf using cycle-canceling for a given graph", PYLMCF_ARGS_NOMIN);
    m.def("lmcf_cycle_canceling", &py_mcf_no_minimums<lemon::CycleCanceling, false, int16_t>, "Compute the lmcf using cycle-canceling for a given graph", PYLMCF_ARGS_NOMIN);
    m.def("lmcf_cycle_canceling", &py_mcf_no_minimums<lemon::CycleCanceling, false, int32_t>, "Compute the lmcf using cycle-canceling for a given graph", PYLMCF_ARGS_NOMIN);
    m.def("lmcf_cycle_canceling", &py_mcf_no_minimums<lemon::CycleCanceling, false, int64_t>, "Compute the lmcf using cycle-canceling for a given graph", PYLMCF_ARGS_NOMIN);
    m.def("lmcf_cycle_canceling", &py_mcf<lemon::CycleCanceling, false, int8_t>, "Compute the lmcf using cycle-canceling for a given graph", PYLMCF_ARGS_MIN);
    m.def("lmcf_cycle_canceling", &py_mcf<lemon::CycleCanceling, false, int16_t>, "Compute the lmcf using cycle-canceling for a given graph", PYLMCF_ARGS_MIN);
    m.def("lmcf_cycle_canceling", &py_mcf<lemon::CycleCanceling, false, int32_t>, "Compute the lmcf using cycle-canceling for a given graph", PYLMCF_ARGS_MIN);
    m.def("lmcf_cycle_canceling", &py_mcf<lemon::CycleCanceling, false, int64_t>, "Compute the lmcf using cycle-canceling for a given graph", PYLMCF_ARGS_MIN);
    add_dtype_catchall("lmcf_cycle_canceling", "int8/int16/int32/int64");
    // Cost-scaling variants (int32/int64 only — small types lack required arithmetic range)
    m.def("lmcf_cost_scaling", &py_mcf_no_minimums<lemon::CostScaling, false, int32_t>, "Compute the lmcf using cost-scaling for a given graph", PYLMCF_ARGS_NOMIN);
    m.def("lmcf_cost_scaling", &py_mcf_no_minimums<lemon::CostScaling, false, int64_t>, "Compute the lmcf using cost-scaling for a given graph", PYLMCF_ARGS_NOMIN);
    m.def("lmcf_cost_scaling", &py_mcf<lemon::CostScaling, false, int32_t>, "Compute the lmcf using cost-scaling for a given graph", PYLMCF_ARGS_MIN);
    m.def("lmcf_cost_scaling", &py_mcf<lemon::CostScaling, false, int64_t>, "Compute the lmcf using cost-scaling for a given graph", PYLMCF_ARGS_MIN);
    add_dtype_catchall("lmcf_cost_scaling", "int32/int64");
    // Capacity-scaling variants (int32/int64 only — small types lack required arithmetic range)
    m.def("lmcf_capacity_scaling", &py_mcf_no_minimums<lemon::CapacityScaling, false, int32_t>, "Compute the lmcf using capacity-scaling for a given graph", PYLMCF_ARGS_NOMIN);
    m.def("lmcf_capacity_scaling", &py_mcf_no_minimums<lemon::CapacityScaling, false, int64_t>, "Compute the lmcf using capacity-scaling for a given graph", PYLMCF_ARGS_NOMIN);
    m.def("lmcf_capacity_scaling", &py_mcf<lemon::CapacityScaling, false, int32_t>, "Compute the lmcf using capacity-scaling for a given graph", PYLMCF_ARGS_MIN);
    m.def("lmcf_capacity_scaling", &py_mcf<lemon::CapacityScaling, false, int64_t>, "Compute the lmcf using capacity-scaling for a given graph", PYLMCF_ARGS_MIN);
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
        .def("set_warm_repair_budget", &Graph<int64_t>::set_warm_repair_budget, arg("multiplier"))
        .def("warm_repair_budget", &Graph<int64_t>::warm_repair_budget)
        .def("set_supply_type", [](Graph<int64_t>& g, const std::string& name) {
                 g.set_supply_type(enum_from_name(supply_type_names, name, "supply type"));
             }, arg("supply_type"))
        .def("supply_type", [](const Graph<int64_t>& g) {
                 return enum_name(supply_type_names, g.supply_type());
             })
        .def("set_pivot_rule", [](Graph<int64_t>& g, const std::string& name) {
                 g.set_pivot_rule(enum_from_name(pivot_rule_names, name, "pivot rule"));
             }, arg("pivot_rule"))
        .def("pivot_rule", [](const Graph<int64_t>& g) {
                 return enum_name(pivot_rule_names, g.pivot_rule());
             })
        .def("set_warm_repair", [](Graph<int64_t>& g, const std::string& name) {
                 g.set_warm_repair(enum_from_name(warm_repair_names, name, "warm repair strategy"));
             }, arg("strategy"))
        .def("warm_repair", [](const Graph<int64_t>& g) {
                 return enum_name(warm_repair_names, g.warm_repair());
             })
        .def("infeasibility_cut", &Graph<int64_t>::infeasibility_cut_py)
        .def("total_cost", &Graph<int64_t>::total_cost)
        .def("result", &Graph<int64_t>::extract_result_py)
        .def("potentials", &Graph<int64_t>::extract_potentials_py)
        .def("__str__", &Graph<int64_t>::to_string);
}
