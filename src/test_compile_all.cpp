#include <cstddef>
#include <cstdint>
#include <vector>
#include <span>
#include <algorithm>
#include <stdexcept>
#include "graph.cpp"
#include "lmcf.cpp"
#include "spectrum.hpp"
#include "decompositable_graph.hpp"
#include "graph_elements.hpp"

template class Graph<int64_t>;
template int64_t lmcf(
    std::span<int64_t> node_supply,
    std::span<int64_t> edges_starts,
    std::span<int64_t> edges_ends,
    std::span<int64_t> capacities,
    std::span<int64_t> costs,
    std::span<int64_t> result
    );