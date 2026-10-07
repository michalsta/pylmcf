#ifndef PYLMCF_GRAPH_HPP
#define PYLMCF_GRAPH_HPP

#include <stdexcept>
#include <span>
#include <vector>
#include <lemon/static_graph.h>
#include <lemon/network_simplex.h>
#include <lemon/circulation.h>
#include <lemon/adaptors.h>

#include "basics.hpp"
#include "canonical_potentials.hpp"

#ifdef INCLUDE_NANOBIND_STUFF
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "py_support.hpp"
#endif

inline lemon::StaticDigraph make_lemon_graph(LEMON_INDEX no_nodes, const std::span<LEMON_INDEX> &edge_starts,
    const std::span<LEMON_INDEX> &edge_ends) {
    const size_t no_edges = edge_starts.size();

    if (no_nodes < 0)
        throw std::invalid_argument("Number of nodes must be non-negative");

    // Make sure all edge arrays and result are the same size
    if (edge_starts.size() != edge_ends.size()) {
        throw std::invalid_argument("All edge arrays must be the same size");
    }

    // Make sure all arcs are valid. LEMON_INDEX is signed, so guard against
    // negative indices explicitly — they would otherwise pass the upper-bound
    // check and feed an invalid node id into lemon_graph.build() (UB).
    for (size_t ii = 0; ii < no_edges; ii++) {
        if (edge_starts[ii] < 0 || edge_starts[ii] >= no_nodes ||
            edge_ends[ii] < 0 || edge_ends[ii] >= no_nodes) {
            throw std::invalid_argument("Edge start or end index out of bounds: start=" + std::to_string(edge_starts[ii]) + ", end=" + std::to_string(edge_ends[ii]));
        }
    }

    lemon::StaticDigraph lemon_graph;

    std::vector<std::pair<LEMON_INDEX, LEMON_INDEX>> arcs;
    arcs.reserve(no_edges);
    for (size_t ii = 0; ii < no_edges; ii++)
        arcs.emplace_back(edge_starts[ii], edge_ends[ii]);

    if(!std::is_sorted(arcs.begin(), arcs.end()))
        throw std::invalid_argument("Edges must be sorted by start node, then by end node");

    lemon_graph.build(no_nodes, arcs.begin(), arcs.end());

    return lemon_graph;
}


template <typename T> class Graph {
public:
    using Solver = lemon::NetworkSimplex<lemon::StaticDigraph, T, T>;

private:
    const LEMON_INDEX _no_nodes;
    const std::vector<LEMON_INDEX> _edge_starts;
    const std::vector<LEMON_INDEX> _edge_ends;

    const lemon::StaticDigraph lemon_graph;
    lemon::StaticDigraph::NodeMap<T> node_supply_map;
    lemon::StaticDigraph::ArcMap<T> capacities_map;
    lemon::StaticDigraph::ArcMap<T> minimums_map;
    lemon::StaticDigraph::ArcMap<T> costs_map;

    lemon::NetworkSimplex<lemon::StaticDigraph, T, T> solver;
    bool _solved = false;
    // True when the solver retains an optimal basis from a previous solve()
    // that warmRun() may restart from.  Invalidated by a non-OPTIMAL solve:
    // after a failed run the internal tree state is not a reusable basis.
    bool _basis_valid = false;
    // Costs changed since the last solve.  warmRun() must then recompute the
    // tree potentials and reoptimize instead of taking the
    // "repair succeeded => already optimal" fast path, which is only valid
    // while the retained basis prices the current costs.
    bool _costs_dirty = false;

    typename Solver::SupplyType _supply_type = Solver::GEQ;
    typename Solver::PivotRule _pivot_rule = Solver::BLOCK_SEARCH;
    typename Solver::WarmRepair _warm_repair = Solver::WarmRepair::Dual;

public:
    Graph(LEMON_INDEX no_nodes, const std::span<LEMON_INDEX> &edge_starts,
        const std::span<LEMON_INDEX> &edge_ends):

        _no_nodes(no_nodes),
        _edge_starts(edge_starts.begin(), edge_starts.end()),
        _edge_ends(edge_ends.begin(), edge_ends.end()),
        lemon_graph(make_lemon_graph(no_nodes, edge_starts, edge_ends)),
        node_supply_map(lemon_graph),
        capacities_map(lemon_graph),
        minimums_map(lemon_graph),
        costs_map(lemon_graph),
        solver(lemon_graph)
        {
            // LEMON's own default upper bound is infinite, but capacities_map
            // (what the getters, check_bounds() and infeasibility_cut() read)
            // starts at zero.  Make the solver agree with it until
            // set_edge_capacities() is called.  Same for costs (LEMON's
            // default is 1): the setters roll back from the solver's copies,
            // so those must always equal the maps.
            solver.upperMap(capacities_map);
            solver.costMap(costs_map);
        };


    Graph() = delete;
    Graph(Graph&&) = delete;
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;


    inline LEMON_INDEX no_nodes() const {
        return _no_nodes;
    }

    inline LEMON_INDEX no_edges() const {
        return _edge_starts.size();
    }

    inline const std::vector<LEMON_INDEX>& edge_starts() const {
        return _edge_starts;
    }

    inline const std::vector<LEMON_INDEX>& edge_ends() const {
        return _edge_ends;
    }

    void set_node_supply(const std::span<T> &node_supply) {
        if (node_supply.size() != static_cast<size_t>(no_nodes()))
            throw std::invalid_argument("Node supply must have the same size as the number of nodes");

        for (LEMON_INT ii = 0; ii < no_nodes(); ii++)
            node_supply_map[lemon_graph.nodeFromId(ii)] = node_supply[ii];

        _solved = false;
    }

    // Caller must free() the returned span's data.
    std::span<T> get_node_supply() const {
        T* data = static_cast<T*>(malloc(sizeof(T) * no_nodes()));
        for (LEMON_INT ii = 0; ii < no_nodes(); ii++)
        {
            data[ii] = node_supply_map[lemon_graph.nodeFromId(ii)];
        }
        return std::span<T>(data, no_nodes());
    }

    // Stores values into map in one pass and reports whether they were all
    // non-negative.  The sign bit of an OR over all values, rather than a
    // compare-and-throw per element, keeps the loop branch-free, so it
    // vectorizes (even on baseline x86-64, where SSE2 has no 64-bit compare)
    // and reads the input only once.
    template <typename Map>
    bool store_non_negative(Map& map, const std::span<T>& values) {
#if defined(__GNUC__) && !defined(__clang__) && defined(__aarch64__)
        // GCC does not split this OR reduction into independent accumulators
        // on AArch64, so a single one is a loop-carried dependency bound by
        // the 2-cycle NEON latency (1.15-1.2x slower than the old loop on an
        // M1).  Four explicit accumulators fix it.  Only here: clang splits
        // the reduction itself, and both compilers vectorize this form worse
        // on x86.  LEMON_INDEX (arcFromId's own type) avoids an int64->int
        // conversion of ii + k that blocks vectorization.
        T acc[4] = {0, 0, 0, 0};
        const LEMON_INDEX n = no_edges();
        LEMON_INDEX ii = 0;
        for (; ii + 4 <= n; ii += 4)
            for (int k = 0; k < 4; k++) {
                acc[k] |= values[ii + k];
                map[lemon_graph.arcFromId(ii + k)] = values[ii + k];
            }
        for (; ii < n; ii++) {
            acc[0] |= values[ii];
            map[lemon_graph.arcFromId(ii)] = values[ii];
        }
        return ((acc[0] | acc[1]) | (acc[2] | acc[3])) >= 0;
#else
        T acc = 0;
        for (LEMON_INT ii = 0; ii < no_edges(); ii++) {
            acc |= values[ii];
            map[lemon_graph.arcFromId(ii)] = values[ii];
        }
        return acc >= 0;
#endif
    }

    // Undoes a rejected store_non_negative(): a rejected update must change
    // nothing, and the solver still holds the last accepted values (every
    // successful setter pushes its map into it; solving never rewrites real
    // arcs' costs or bounds).
    template <typename Map, typename Old>
    void restore_from_solver(Map& map, Old old) {
        for (LEMON_INT ii = 0; ii < no_edges(); ii++) {
            const auto a = lemon_graph.arcFromId(ii);
            map[a] = old(solver.internalArcId(a));
        }
    }

    void set_edge_capacities(const std::span<T> &capacities) {
        if (capacities.size() != static_cast<size_t>(no_edges()))
            throw std::invalid_argument("Capacities must have the same size as the number of edges");

        if (!store_non_negative(capacities_map, capacities)) {
            restore_from_solver(capacities_map, [&](int i) { return solver.internalUpper(i); });
            throw std::invalid_argument("Capacities must be non-negative");
        }

        solver.upperMap(capacities_map);
        _solved = false;
    }

    void set_edge_minimums(const std::span<T> &minimums) {
        if (minimums.size() != static_cast<size_t>(no_edges()))
            throw std::invalid_argument("Minimums must have the same size as the number of edges");

        if (!store_non_negative(minimums_map, minimums)) {
            restore_from_solver(minimums_map, [&](int i) { return solver.internalLower(i); });
            throw std::invalid_argument("Minimums must be non-negative");
        }

        solver.lowerMap(minimums_map);
        _solved = false;
    }

    // Caller must free() the returned span's data.
    std::span<T> get_edge_capacities() const {
        T* data = static_cast<T*>(malloc(sizeof(T) * no_edges()));
        for (LEMON_INT ii = 0; ii < no_edges(); ii++)
        {
            data[ii] = capacities_map[lemon_graph.arcFromId(ii)];
        }
        return std::span<T>(data, no_edges());
    }

    // Caller must free() the returned span's data.
    std::span<T> get_edge_minimums() const {
        T* data = static_cast<T*>(malloc(sizeof(T) * no_edges()));
        for (LEMON_INT ii = 0; ii < no_edges(); ii++)
        {
            data[ii] = minimums_map[lemon_graph.arcFromId(ii)];
        }
        return std::span<T>(data, no_edges());
    }

    void set_edge_costs(const std::span<T> &costs) {
        if (costs.size() != static_cast<size_t>(no_edges()))
            throw std::invalid_argument("Costs must have the same size as the number of edges");

        if (!store_non_negative(costs_map, costs)) {
            restore_from_solver(costs_map, [&](int i) { return solver.internalCost(i); });
            throw std::invalid_argument("Costs must be non-negative");
        }

        solver.costMap(costs_map);
        _costs_dirty = true;
        _solved = false;
    }

    // Caller must free() the returned span's data.
    std::span<T> get_edge_costs() const {
        T* data = static_cast<T*>(malloc(sizeof(T) * no_edges()));
        for (LEMON_INT ii = 0; ii < no_edges(); ii++)
        {
            data[ii] = costs_map[lemon_graph.arcFromId(ii)];
        }
        return std::span<T>(data, no_edges());
    }

    // LEMON only asserts lower <= upper in debug builds; violating it makes
    // every solver return flows outside their bounds without complaint.  The
    // setters cannot check it (capacities and minimums are set separately),
    // so it is checked where the pair is consumed.
    void check_bounds() const {
        for (LEMON_INDEX ii = 0; ii < no_edges(); ii++) {
            const auto a = lemon_graph.arcFromId(ii);
            if (minimums_map[a] > capacities_map[a])
                throw std::invalid_argument("Edge " + std::to_string(ii) + " has minimum " +
                    std::to_string(minimums_map[a]) + " above its capacity " +
                    std::to_string(capacities_map[a]));
        }
    }

    void solve(){
        check_bounds();
        // LEMON's init() rejects an empty node set and run() reports that as
        // INFEASIBLE; an empty problem is trivially optimal at cost 0.
        if (_no_nodes == 0) {
            _costs_dirty = false;
            _solved = true;
            return;
        }
        solver.supplyMap(node_supply_map);
        solver.costMap(costs_map);
        // Re-solves warm-restart from the retained basis.  warmRun() itself
        // falls back to a cold init()+start() whenever the basis cannot be
        // reused (non-EQ supply, nonzero lower bounds, failed repair), so
        // every mutation of capacities/supplies/costs/minimums is safe here;
        // _costs_dirty only steers it off the costs-unchanged fast path.
        const auto status = _basis_valid
            ? solver.warmRun(_pivot_rule, _warm_repair, _costs_dirty)
            : solver.run(_pivot_rule);
        _basis_valid = (status == Solver::OPTIMAL);
        if (status != Solver::OPTIMAL) {
            if (status == Solver::INFEASIBLE)
                throw std::runtime_error("Solver failed: problem is INFEASIBLE "
                                         "(infeasibility_cut() names a set of nodes that proves it)");
            else if (status == Solver::UNBOUNDED)
                throw std::runtime_error("Solver failed: problem is UNBOUNDED");
            else
                throw std::runtime_error("Solver failed with unknown status");
        }
        _costs_dirty = false;
        _solved = true;
    }

    // Warm-restart observability, backed by the solver's counters.  Only
    // warmRun() increments them: the first solve() and any solve() after a
    // non-OPTIMAL result go through plain run() and count in neither, so
    // across a chain of successful re-solves
    //   warm + cold + dual_repair + primal_repair == number of re-solves.
    int warm_start_count() const { return solver.warmStartCount(); }
    int cold_start_count() const { return solver.coldStartCount(); }
    int dual_repair_count() const { return solver.dualRepairCount(); }
    int primal_repair_count() const { return solver.primalRepairCount(); }
    int policy_cold_count() const { return solver.policyColdCount(); }
    void set_warm_violation_limit(long v) { solver.setWarmViolationLimit(v); }
    // Repair time budget as a multiple of the last cold solve's wall time
    // (<= 0 disables).  A catastrophe tripwire, not a tuning knob — see
    // NetworkSimplex::setWarmRepairBudget().  PYLMCF_WARM_REPAIR_BUDGET, if
    // set, overrides it.
    void set_warm_repair_budget(double mult) { solver.setWarmRepairBudget(mult); }
    double warm_repair_budget() const { return solver.warmRepairBudget(); }

    // Direction of the supply/demand constraints when total supply != 0
    // (LEMON's SupplyType; with zero total supply both mean equality):
    //   GEQ (default): out - in >= supply; needs sum(supply) <= 0, i.e.
    //                  every supply is shipped, demands may go unmet.
    //   LEQ:           out - in <= supply; needs sum(supply) >= 0, i.e.
    //                  every demand is met, supplies may go unused.
    // Warm restarts only apply to zero total supply; others re-solve cold.
    void set_supply_type(typename Solver::SupplyType st) {
        _supply_type = st;
        solver.supplyType(st);
        _basis_valid = false;
        _solved = false;
    }
    typename Solver::SupplyType supply_type() const { return _supply_type; }

    // Pivot rule for every solve (cold solves and warmRun()'s reoptimization).
    void set_pivot_rule(typename Solver::PivotRule rule) { _pivot_rule = rule; }
    typename Solver::PivotRule pivot_rule() const { return _pivot_rule; }

    // Basis-repair strategy used by warm re-solves.  DualRatio/DualGreedy are
    // not bit-identical to Dual at degenerate optima (same cost, possibly
    // different optimal flows).
    void set_warm_repair(typename Solver::WarmRepair strategy) { _warm_repair = strategy; }
    typename Solver::WarmRepair warm_repair() const { return _warm_repair; }

    // A certificate of infeasibility (LEMON's Circulation barrier): a set B of
    // nodes such that, under GEQ supply constraints,
    //   sum(cap of edges leaving B) - sum(minimum of edges entering B)
    //       < sum(supply of B),
    // i.e. B must push out more than its boundary can carry.  Under LEQ the
    // roles flip: sum(cap entering B) - sum(minimum leaving B) < -sum(supply
    // of B).  Returns an empty vector when the current data is feasible;
    // otherwise one flag per node.  Independent of solve(); reads the current
    // supplies, capacities and minimums.
    std::vector<char> infeasibility_cut() const {
        check_bounds();
        using G = lemon::StaticDigraph;
        using AM = G::ArcMap<T>;
        using NM = G::NodeMap<T>;
        std::vector<char> cut;
        auto collect = [&](auto& circ) {
            if (circ.run()) return;
            cut.resize(no_nodes());
            for (LEMON_INDEX ii = 0; ii < no_nodes(); ii++)
                cut[ii] = circ.barrier(lemon_graph.nodeFromId(ii));
        };
        if (_supply_type == Solver::GEQ) {
            lemon::Circulation<G, AM, AM, NM> circ(lemon_graph, minimums_map, capacities_map, node_supply_map);
            collect(circ);
        } else {
            // LEQ is GEQ on the reversed graph with negated supplies.
            using R = lemon::ReverseDigraph<const G>;
            NM negated(lemon_graph);
            for (LEMON_INDEX ii = 0; ii < no_nodes(); ii++) {
                const auto n = lemon_graph.nodeFromId(ii);
                negated[n] = -node_supply_map[n];
            }
            R rev(lemon_graph);
            lemon::Circulation<R, AM, AM, NM> circ(rev, minimums_map, capacities_map, negated);
            collect(circ);
        }
        return cut;
    }

    T total_cost() const {
        if (!_solved)
            throw std::runtime_error("solve() must be called before total_cost()");
        return solver.totalCost();
    }

    // Caller must free() the returned span's data.
    std::span<T> get_edge_flows() const {
        if (!_solved)
            throw std::runtime_error("solve() must be called before reading results");
        T* data = static_cast<T*>(malloc(sizeof(T) * no_edges()));
        for (LEMON_INT ii = 0; ii < no_edges(); ii++)
            data[ii] = solver.flow(lemon_graph.arcFromId(ii));
        return std::span<T>(data, no_edges());
    }

    // Node potentials (the dual solution) of the last solve, indexed by node
    // id.  They follow LEMON's convention: the reduced cost of edge (u, v) is
    //   rc = cost + pi[u] - pi[v],
    // and complementary slackness holds against the returned flows (rc > 0
    // => flow == minimum, rc < 0 => flow == capacity).  Of all potentials
    // that are optimal for the returned flows, this is the canonical one
    // (see canonical_potentials.hpp): the pointwise-largest with pi <= 0
    // under GEQ, the pointwise-smallest with pi >= 0 under LEQ.  LEMON's raw
    // tree potentials can carry its 2^62 artificial cost instead.
    // Caller must free() the returned span's data.
    std::span<T> get_node_potentials() const {
        if (!_solved)
            throw std::runtime_error("solve() must be called before reading potentials");
        const LEMON_INDEX n = no_nodes();
        const LEMON_INDEX m = no_edges();
        std::vector<T> costs(m), caps(m), mins(m), flows(m), raw(n);
        for (LEMON_INDEX ii = 0; ii < m; ii++) {
            const auto a = lemon_graph.arcFromId(ii);
            costs[ii] = costs_map[a];
            caps[ii] = capacities_map[a];
            mins[ii] = minimums_map[a];
            flows[ii] = solver.flow(a);
        }
        for (LEMON_INDEX ii = 0; ii < n; ii++)
            raw[ii] = solver.potential(lemon_graph.nodeFromId(ii));
        T* data = static_cast<T*>(malloc(sizeof(T) * n));
        std::span<T> out(data, n);
        try {
            canonical_potentials<T, T>(n, _edge_starts, _edge_ends, costs, caps, mins,
                                       flows, raw, _supply_type == Solver::LEQ, out);
        } catch (...) {
            free(data);
            throw;
        }
        return out;
    }

    std::string to_string() const {
        std::string out = "Graph with " + std::to_string(no_nodes()) + " nodes and " + std::to_string(no_edges()) + " edges\n";
        out += "Edges:\n";
        for (LEMON_INT ii = 0; ii < no_edges(); ii++) {
            out += "  " + std::to_string(lemon_graph.id(lemon_graph.source(lemon_graph.arcFromId(ii)))) + " -> " + std::to_string(lemon_graph.id(lemon_graph.target(lemon_graph.arcFromId(ii)))) + " with cost " + std::to_string(costs_map[lemon_graph.arcFromId(ii)]) +
            " and capacity " + std::to_string(capacities_map[lemon_graph.arcFromId(ii)]) +
            " and minimum " + std::to_string(minimums_map[lemon_graph.arcFromId(ii)]) + "\n";
        }
        return out;
    }

#ifdef INCLUDE_NANOBIND_STUFF
    Graph(LEMON_INDEX no_nodes, const ndarray_1d<LEMON_INDEX> &edge_starts,
        const ndarray_1d<LEMON_INDEX> &edge_ends):
        Graph(no_nodes, numpy_to_span(edge_starts), numpy_to_span<LEMON_INDEX>(edge_ends)) {};

    // int64 edge-index arrays (numpy's default integer dtype on Linux) are
    // accepted via an explicit checked narrowing to LEMON_INDEX.  This is a
    // deliberate exact-dtype overload, NOT nanobind implicit conversion —
    // conversions are disabled (noconvert) module-wide to prevent silent
    // dtype truncation.
    static std::vector<LEMON_INDEX> checked_index_vector(const ndarray_1d<std::int64_t> &a) {
        auto s = numpy_to_span<std::int64_t>(a);
        std::vector<LEMON_INDEX> v(s.size());
        for (size_t i = 0; i < s.size(); i++) {
            if (s[i] < std::numeric_limits<LEMON_INDEX>::min() ||
                s[i] > std::numeric_limits<LEMON_INDEX>::max())
                throw std::invalid_argument(
                    "Edge index does not fit LEMON's 32-bit node index: " + std::to_string(s[i]));
            v[i] = static_cast<LEMON_INDEX>(s[i]);
        }
        return v;
    }

    Graph(LEMON_INDEX no_nodes, std::vector<LEMON_INDEX> edge_starts,
        std::vector<LEMON_INDEX> edge_ends):
        Graph(no_nodes,
              std::span<LEMON_INDEX>(edge_starts.data(), edge_starts.size()),
              std::span<LEMON_INDEX>(edge_ends.data(), edge_ends.size())) {};

    Graph(LEMON_INDEX no_nodes, const ndarray_1d<std::int64_t> &edge_starts,
        const ndarray_1d<std::int64_t> &edge_ends):
        Graph(no_nodes, checked_index_vector(edge_starts), checked_index_vector(edge_ends)) {};

    void set_node_supply_py(const ndarray_1d<T> &node_supply) {
        set_node_supply(numpy_to_span(node_supply));
    }

    void set_edge_capacities_py(const ndarray_1d<T> &capacities) {
        set_edge_capacities(numpy_to_span(capacities));
    }

    void set_edge_minimums_py(const ndarray_1d<T> &minimums) {
        set_edge_minimums(numpy_to_span(minimums));
    }

    void set_edge_costs_py(const ndarray_1d<T> &costs) {
        set_edge_costs(numpy_to_span(costs));
    }

    nb::ndarray<T, nb::numpy, nb::shape<-1>> get_node_supply_py() const {
        return steal_mallocd_span_to_np_array(get_node_supply());
    }

    nb::ndarray<T, nb::numpy, nb::shape<-1>> get_edge_capacities_py() const {
        return steal_mallocd_span_to_np_array(get_edge_capacities());
    }

    nb::ndarray<T, nb::numpy, nb::shape<-1>> get_edge_minimums_py() const {
        return steal_mallocd_span_to_np_array(get_edge_minimums());
    }

    nb::ndarray<T, nb::numpy, nb::shape<-1>> get_edge_costs_py() const {
        return steal_mallocd_span_to_np_array(get_edge_costs());
    }

    nb::ndarray<T, nb::numpy, nb::shape<-1>> extract_result_py() const {
        return steal_mallocd_span_to_np_array(get_edge_flows());
    }

    nb::ndarray<T, nb::numpy, nb::shape<-1>> extract_potentials_py() const {
        return steal_mallocd_span_to_np_array(get_node_potentials());
    }

    nb::object infeasibility_cut_py() const {
        const std::vector<char> cut = infeasibility_cut();
        if (cut.empty())
            return nb::none();
        bool* data = static_cast<bool*>(malloc(sizeof(bool) * cut.size()));
        for (size_t ii = 0; ii < cut.size(); ii++)
            data[ii] = cut[ii] != 0;
        return nb::cast(steal_mallocd_span_to_np_array(std::span<bool>(data, cut.size())));
    }

    nb::ndarray<LEMON_INDEX, nb::numpy, nb::shape<-1>, nb::ro> edge_starts_py() const {
        return nb::ndarray<LEMON_INDEX, nb::numpy, nb::shape<-1>, nb::ro>(
            const_cast<LEMON_INDEX*>(_edge_starts.data()), { _edge_starts.size() }, nb::find(this));
    }

    nb::ndarray<LEMON_INDEX, nb::numpy, nb::shape<-1>, nb::ro> edge_ends_py() const {
        return nb::ndarray<LEMON_INDEX, nb::numpy, nb::shape<-1>, nb::ro>(
            const_cast<LEMON_INDEX*>(_edge_ends.data()), { _edge_ends.size() }, nb::find(this));
    }
#endif

};

#endif // PYLMCF_GRAPH_HPP