// test_env_overrides.cpp
// -------------------------------------------------------------------------
// The two environment overrides of the vendored NetworkSimplex:
// PYLMCF_WARM_VIOLATION_LIMIT and PYLMCF_WARM_REPAIR_BUDGET.  Each is read
// ONCE per process (a function-local static), so one process can observe one
// setting only: every scenario runs in its own fork()ed child, which sets the
// environment before any solver exists.  The parent only collects exit codes.
//
//   overridden: limit "0", budget "3.5" -> the setters are ignored, and the
//               limit-0 policy really is in force (no simplex repair ever
//               runs, policy cold starts are recorded), with every result
//               still checked against the CapacityScaling oracle;
//   empty:      both set to "" -> treated as unset, setters honoured;
//   unset:      both absent    -> defaults (-2, 64.0), setters honoured.
//
// The other suites refuse to run under these variables; this one sets and
// clears them itself, so it is independent of the caller's environment.
//
// Build:
//   g++ -I$(python -m pylmcf --include) -std=c++20 -O2 tests_cpp/test_env_overrides.cpp -o /tmp/te && /tmp/te
// -------------------------------------------------------------------------
#include "mcf_oracle.h"

#include <pylmcf/graph.hpp>

#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>

using namespace mcf_test;
using NS = Graph<int64_t>::Solver;

template <typename T>
static std::vector<T> take(std::span<T> s) {
    std::vector<T> v(s.begin(), s.end());
    free(s.data());
    return v;
}

// A NetworkSimplex on a tiny graph, for the getters.
struct Tiny {
    lemon::StaticDigraph g;
    Tiny() {
        std::vector<std::pair<int, int>> arcs{{0, 1}};
        g.build(2, arcs.begin(), arcs.end());
    }
};

static void setters_honoured(const char* scenario) {
    Tiny t;
    NS ns(t.g);
    CHECK(ns.warmViolationLimit() == -2, "%s: default violation limit %ld", scenario, ns.warmViolationLimit());
    CHECK(ns.warmRepairBudget() == 64.0, "%s: default budget %g", scenario, ns.warmRepairBudget());
    ns.setWarmViolationLimit(7);
    ns.setWarmRepairBudget(2.5);
    CHECK(ns.warmViolationLimit() == 7, "%s: setter ignored (limit %ld)", scenario, ns.warmViolationLimit());
    CHECK(ns.warmRepairBudget() == 2.5, "%s: setter ignored (budget %g)", scenario, ns.warmRepairBudget());
}

static void overridden() {
    {
        Tiny t;
        NS ns(t.g);
        ns.setWarmViolationLimit(-1);
        ns.setWarmRepairBudget(64.0);
        CHECK(ns.warmViolationLimit() == 0, "env limit not in force: %ld", ns.warmViolationLimit());
        CHECK(ns.warmRepairBudget() == 3.5, "env budget not in force: %g", ns.warmRepairBudget());
    }
    // Behaviour, not just the getter: through Graph, which sets limit -1
    // explicitly here, the env's 0 must still suppress every simplex repair.
    Rng rng(0xE7E7);
    long policy = 0;
    for (int chain = 0; chain < 20; chain++) {
        auto in = random_instance(rng, 40, 160, false);
        Graph<int64_t> g(in.n, std::span<int>(in.starts), std::span<int>(in.ends));
        g.set_warm_violation_limit(-1);
        CHECK(g.warm_repair_budget() == 3.5, "Graph sees env budget");
        auto push = [&] {
            g.set_node_supply(in.supply);
            g.set_edge_capacities(in.caps);
            g.set_edge_costs(in.costs);
        };
        push();
        for (int step = 0; step < 8; step++) {
            g.solve();
            const auto f = take(g.get_edge_flows());
            const auto p = take(g.get_node_potentials());
            CHECK(g.total_cost() == oracle_cost(in), "chain %d step %d: cost vs oracle", chain, step);
            certify(in, f, p, g.total_cost(), "env chain");
            redraw(rng, in);
            push();
        }
        CHECK(g.dual_repair_count() == 0 && g.primal_repair_count() == 0,
              "chain %d: a repair ran under env limit 0", chain);
        policy += g.policy_cold_count();
    }
    CHECK(policy > 0, "env limit 0 never produced a policy cold start");
}

// Run `body` in a child with the given environment; true iff it passed.
template <typename F>
static bool in_child(const char* scenario, const char* limit, const char* budget, F body) {
    std::fflush(stdout);
    const pid_t pid = fork();
    if (pid == 0) {
        if (limit) setenv("PYLMCF_WARM_VIOLATION_LIMIT", limit, 1);
        else unsetenv("PYLMCF_WARM_VIOLATION_LIMIT");
        if (budget) setenv("PYLMCF_WARM_REPAIR_BUDGET", budget, 1);
        else unsetenv("PYLMCF_WARM_REPAIR_BUDGET");
        section(scenario, body);
        std::printf("  %s: checks=%ld fails=%ld\n", scenario, checks, fails);
        std::fflush(stdout);
        _exit(fails ? 1 : 0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

int main() {
    const bool a = in_child("overridden", "0", "3.5", overridden);
    const bool b = in_child("empty", "", "", [] { setters_honoured("empty"); });
    const bool c = in_child("unset", nullptr, nullptr, [] { setters_honoured("unset"); });
    CHECK(a, "scenario 'overridden' failed");
    CHECK(b, "scenario 'empty' failed");
    CHECK(c, "scenario 'unset' failed");
    return finish("test_env_overrides");
}
