#ifndef PYLMCF_NEW_SOLVER_BINDINGS_HPP
#define PYLMCF_NEW_SOLVER_BINDINGS_HPP

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "py_support.hpp"
#include "network_simplex_lct.h"
#include "network_simplex_lct_dyn.h"
#include "chain_solver_1d.h"

namespace pylmcf_python {
using Int = std::int64_t;
using Array = ndarray_1d<Int>;
constexpr Int limit = std::numeric_limits<Int>::max() / 4;

inline Int checked_add(Int a, Int b) {
    if (b > limit - a)
        throw std::overflow_error("Input exceeds the solver's safe int64 arithmetic range");
    return a + b;
}
inline Int checked_mul(Int a, Int b) {
    if (b && a > limit / b)
        throw std::overflow_error("Input exceeds the solver's safe int64 arithmetic range");
    return a * b;
}
inline std::vector<Int> copy_input(Array input) {
    auto s = numpy_to_span(input);
    return {s.begin(), s.end()};
}
inline auto output_array(const std::vector<Int>& values) {
    auto out = create_empty_numpy_array<Int>(values.size());
    std::copy(values.begin(), values.end(), out.data());
    return out;
}
inline void check_size(size_t actual, size_t expected) {
    if (actual != expected) throw std::invalid_argument("Array lengths do not match");
}
inline void check_supply(const std::vector<Int>& supply) {
    Int positive = 0, negative = 0;
    for (Int x : supply) {
        if (x < -limit || x > limit) throw std::overflow_error("Supply exceeds solver range");
        if (x >= 0) positive = checked_add(positive, x);
        else negative = checked_add(negative, -x);
    }
    if (positive != negative) throw std::invalid_argument("Node supplies must sum to zero");
}
inline void check_caps(const std::vector<Int>& caps, const std::vector<Int>& costs) {
    Int bound = 0, capacity_sum = 0;
    for (size_t i = 0; i < caps.size(); ++i) {
        if (caps[i] < 0) throw std::invalid_argument("Capacities must be non-negative");
        if (caps[i] >= limit) throw std::overflow_error("Capacities must be below INT64_MAX / 4");
        capacity_sum = checked_add(capacity_sum, caps[i]);
        bound = checked_add(bound, checked_mul(caps[i], costs[i]));
    }
}

// Own all inputs and validate before touching the unchecked C++ solver API.
// Costs/topology are immutable; only supplies and capacities can be updated.
template <typename Solver>
class LCTSolver {
    std::vector<Int> supply_, starts_, ends_, caps_, costs_;
    std::unique_ptr<Solver> solver_;
    bool solved_ = false, basis_ = false;
    int warm_ = 0, cold_ = 0;

    void require_solved() const {
        if (!solved_) throw std::runtime_error("solve() must succeed before reading results");
    }
public:
    LCTSolver(Array supply, Array starts, Array ends, Array caps, Array costs)
        : supply_(copy_input(supply)), starts_(copy_input(starts)),
          ends_(copy_input(ends)), caps_(copy_input(caps)), costs_(copy_input(costs)) {
        const size_t n = supply_.size(), m = starts_.size();
        // Artificial arcs and the LCT sentinel also use signed int indices.
        if (n > size_t(std::numeric_limits<int>::max() - 3) ||
            m > size_t(std::numeric_limits<int>::max() - 3) - n)
            throw std::overflow_error("Too many nodes or edges");
        check_size(ends_.size(), m); check_size(caps_.size(), m); check_size(costs_.size(), m);
        check_supply(supply_);
        Int max_cost = 1;
        for (size_t e = 0; e < m; ++e) {
            if (starts_[e] < 0 || ends_[e] < 0 ||
                size_t(starts_[e]) >= n || size_t(ends_[e]) >= n)
                throw std::invalid_argument("Edge index out of bounds");
            if (costs_[e] < 0) throw std::invalid_argument("Costs must be non-negative");
            max_cost = std::max(max_cost, costs_[e]);
        }
        // Bound Big-M and path potentials, not just the final objective.
        Int big_m = checked_add(checked_mul(checked_mul(max_cost, n + 2), m + 2), 1);
        checked_mul(big_m, 4 * (Int(n) + 2));
        check_caps(caps_, costs_);
        solver_ = std::make_unique<Solver>(int(n));
        for (size_t e = 0; e < m; ++e)
            solver_->addArc(int(starts_[e]), int(ends_[e]), costs_[e], caps_[e]);
        for (size_t v = 0; v < n; ++v) solver_->setSupply(int(v), supply_[v]);
    }
    void set_node_supply(Array input) {
        auto values = copy_input(input);
        check_size(values.size(), supply_.size()); check_supply(values);
        for (size_t v = 0; v < values.size(); ++v) solver_->setSupply(int(v), values[v]);
        supply_ = std::move(values); solved_ = false;
    }
    void set_edge_capacities(Array input) {
        auto values = copy_input(input);
        check_size(values.size(), caps_.size()); check_caps(values, costs_);
        for (size_t e = 0; e < values.size(); ++e) solver_->setCap(int(e), values[e]);
        caps_ = std::move(values); solved_ = false;
    }
    void solve(bool warm = true) {
        solved_ = false;
        const bool reuse = warm && basis_;
        basis_ = false;
        const int before = solver_->warmCount();
        auto status = reuse ? solver_->warmRun() : solver_->run();
        if (reuse && solver_->warmCount() > before) ++warm_;
        else ++cold_;
        if (status != Solver::OPTIMAL) throw std::runtime_error("Solver failed: problem is INFEASIBLE");
        solved_ = basis_ = true;
    }
    auto result() const {
        require_solved();
        auto out = create_empty_numpy_array<Int>(starts_.size());
        for (size_t e = 0; e < starts_.size(); ++e) out.data()[e] = solver_->flow(int(e));
        return out;
    }
    Int total_cost() const { require_solved(); return solver_->totalCost(); }
    int warm_start_count() const { return warm_; }
    int cold_start_count() const { return cold_; }
};

template <typename Solver>
void bind_lct(nb::module_& m, const char* class_name, const char* function_name, const char* doc) {
    using Wrapper = LCTSolver<Solver>;
    using nb::arg;
    nb::class_<Wrapper>(m, class_name, doc)
        .def(nb::init<Array, Array, Array, Array, Array>(),
             arg("node_supply").noconvert(), arg("edge_starts").noconvert(),
             arg("edge_ends").noconvert(), arg("capacities").noconvert(), arg("costs").noconvert())
        .def("set_node_supply", &Wrapper::set_node_supply, arg("supply").noconvert())
        .def("set_edge_capacities", &Wrapper::set_edge_capacities, arg("capacities").noconvert())
        .def("solve", &Wrapper::solve, arg("warm") = true)
        .def("result", &Wrapper::result)
        .def("total_cost", &Wrapper::total_cost)
        .def("warm_start_count", &Wrapper::warm_start_count)
        .def("cold_start_count", &Wrapper::cold_start_count);
    m.def(function_name, [](Array supply, Array starts, Array ends, Array caps, Array costs) {
        Wrapper solver(supply, starts, ends, caps, costs);
        solver.solve();
        return solver.result();
    }, doc, arg("node_supply").noconvert(), arg("edge_starts").noconvert(),
       arg("edge_ends").noconvert(), arg("capacities").noconvert(), arg("costs").noconvert());
}

inline nb::dict solve_chain_1d(Array positions, Array empirical, Array theoretical, Int kappa) {
    auto pos = numpy_to_span(positions), emp = numpy_to_span(empirical), theo = numpy_to_span(theoretical);
    check_size(emp.size(), pos.size()); check_size(theo.size(), pos.size());
    if (pos.size() > size_t((std::numeric_limits<int>::max() - 2) / 8))
        throw std::overflow_error("Too many chain points");
    if (kappa < 0) throw std::invalid_argument("kappa must be non-negative");
    using Chain = pylmcf::ChainSolver1D<Int, Int>;
    std::vector<Chain::Point> points;
    Int e_sum = 0, t_sum = 0, width = 0;
    for (size_t i = 0; i < pos.size(); ++i) {
        if (emp[i] < 0 || theo[i] < 0) throw std::invalid_argument("Masses must be non-negative");
        e_sum = checked_add(e_sum, emp[i]); t_sum = checked_add(t_sum, theo[i]);
        if (i) {
            if (pos[i] < pos[i - 1]) throw std::invalid_argument("Positions must be sorted");
            const auto gap = std::uint64_t(pos[i]) - std::uint64_t(pos[i - 1]);
            if (gap > std::uint64_t(limit)) throw std::overflow_error("Position gap exceeds solver range");
            width = checked_add(width, Int(gap));
        }
        points.push_back({pos[i], emp[i], theo[i]});
    }
    Int path_bound = checked_mul(checked_add(width, kappa), 4 * (Int(pos.size()) + 2));
    checked_mul(path_bound, std::max<Int>(1, std::max(e_sum, t_sum)));
    auto flows = Chain::solveFull(points, kappa);
    nb::dict out;
    out["total_cost"] = flows.total;
    out["emp_in"] = output_array(flows.emp_in);
    out["theo_out"] = output_array(flows.theo_out);
    out["gap"] = output_array(flows.gap);
    out["trash"] = flows.trash;
    return out;
}

inline void bind_new_solvers(nb::module_& m) {
    bind_lct<pylmcf::NetworkSimplexLCT<Int, Int>>(m, "NetworkSimplexLCT", "lmcf_lct",
        "Link-cut-tree network simplex. Contiguous int64 arrays; balanced supplies, "
        "non-negative costs and finite capacities; zero lower bounds. Edge order is preserved.");
    bind_lct<pylmcf::NetworkSimplexLCTDyn<Int, Int>>(m, "NetworkSimplexLCTDyn", "lmcf_lct_dyn",
        "Experimental dynamic-tree network simplex. Same inputs as lmcf_lct. "
        "Warm restarts support supply changes; capacity changes trigger a cold solve.");
    m.def("solve_chain_1d", &solve_chain_1d,
        "Solve the sorted 1D SimpleTrash chain with int64 positions and masses. "
        "Return total_cost, emp_in, theo_out, signed rightward gap flows, and trash.",
        nb::arg("positions").noconvert(), nb::arg("empirical").noconvert(),
        nb::arg("theoretical").noconvert(), nb::arg("kappa"));
}
} // namespace pylmcf_python
#endif
