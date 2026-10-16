// Independent solvers on concurrent threads (the free-threading use case),
// for the ThreadSanitizer lanes and as a determinism check everywhere.
//
// Each thread owns its Graph and runs a warm-restart chain under every repair
// strategy; nothing is shared but process-wide state inside the library: the
// function-local statics that read the PYLMCF_WARM_* environment overrides
// on first use, which every thread reaches at the same time here. A thread's
// costs must equal the CapacityScaling oracle and the same chain run alone.
//
// g++ -I$(python -m pylmcf --include) -std=c++20 -O2 -pthread tests_cpp/test_concurrent.cpp -o /tmp/t && /tmp/t
#include "mcf_oracle.h"

#include <pylmcf/graph.hpp>

#include <barrier>
#include <thread>
#include <vector>

using namespace mcf_test;
using G64 = Graph<int64_t>;
using NS = G64::Solver;

namespace {

constexpr int kThreads = 6;
constexpr int kSteps = 12;

const NS::WarmRepair kStrategies[] = {
    NS::WarmRepair::RepairOnly, NS::WarmRepair::Dual, NS::WarmRepair::Primal,
    NS::WarmRepair::DualRatio, NS::WarmRepair::DualGreedy,
};

void push(G64& g, const Instance& in) {
    g.set_node_supply(std::span<const int64_t>(in.supply));
    g.set_edge_capacities(std::span<const int64_t>(in.caps));
    g.set_edge_costs(std::span<const int64_t>(in.costs));
}

// One warm-restart chain; returns the cost after every solve.
std::vector<int64_t> chain(int seed, NS::WarmRepair strategy, std::barrier<>* start) {
    Rng rng(seed);
    Instance in = random_instance(rng, 40, 160, /*with_mins=*/false);
    G64 g(in.n, std::span<int>(in.starts), std::span<int>(in.ends));
    g.set_warm_repair(strategy);
    g.set_warm_repair_budget(1.5);   // tight: repairs and cold fallbacks both occur
    if (start) start->arrive_and_wait();   // every thread hits the env statics at once
    std::vector<int64_t> costs;
    for (int s = 0; s < kSteps; ++s) {
        push(g, in);
        g.solve();
        costs.push_back(g.total_cost());
        redraw(rng, in);
    }
    return costs;
}

std::vector<int64_t> oracle(int seed) {
    Rng rng(seed);
    Instance in = random_instance(rng, 40, 160, false);
    std::vector<int64_t> costs;
    for (int s = 0; s < kSteps; ++s) {
        costs.push_back(oracle_cost(in));
        redraw(rng, in);
    }
    return costs;
}

}  // namespace

int main() {
    for (auto strategy : kStrategies) {
        std::vector<std::vector<int64_t>> seq(kThreads), par(kThreads);
        for (int t = 0; t < kThreads; ++t) seq[t] = chain(100 + t, strategy, nullptr);
        std::barrier<> start(kThreads);
        std::vector<std::thread> threads;
        for (int t = 0; t < kThreads; ++t)
            threads.emplace_back([&, t] { par[t] = chain(100 + t, strategy, &start); });
        for (auto& th : threads) th.join();
        for (int t = 0; t < kThreads; ++t) {
            CHECK(par[t] == seq[t], "thread %d: concurrent chain differs from the same chain run alone", t);
            CHECK(seq[t] == oracle(100 + t), "thread %d: chain cost differs from the CapacityScaling oracle", t);
        }
    }
    return finish("test_concurrent");
}
