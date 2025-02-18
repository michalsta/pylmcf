#include "graph.cpp"
#include "lmcf.cpp"

template class Graph<int64_t>;
template int64_t lmcf(
    std::span<int64_t> node_supply,
    std::span<int64_t> edges_starts,
    std::span<int64_t> edges_ends,
    std::span<int64_t> capacities,
    std::span<int64_t> costs,
    std::span<int64_t> result
    );