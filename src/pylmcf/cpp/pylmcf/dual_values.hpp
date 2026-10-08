#ifndef PYLMCF_DUAL_VALUES_HPP
#define PYLMCF_DUAL_VALUES_HPP

#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace pylmcf {
// Avoid signed overflow, including artificial-root offsets. Callers must not
// subtract exported int64 potentials in narrow NumPy arithmetic unchecked.
template <typename C>
inline C checkedDualAdd(C a, C b) {
    if ((b > 0 && a > std::numeric_limits<C>::max() - b) ||
        (b < 0 && a < std::numeric_limits<C>::min() - b))
        throw std::overflow_error("Dual value exceeds the cost type");
    return a + b;
}
template <typename C>
inline C reducedCost(C cost, C source, C target) {
    // Group like-signed potentials first to cancel their common offset.
    if ((source < 0) == (target < 0))
        return checkedDualAdd(cost, C(source - target));
    if (target == std::numeric_limits<C>::min()) {
        return checkedDualAdd(checkedDualAdd(checkedDualAdd(cost, source), C(-(target + 1))), C(1));
    }
    return checkedDualAdd(checkedDualAdd(cost, source), C(-target));
}
// Artificial paths must dominate negative as well as positive real paths.
// A too-small Big-M leaves artificial flow and falsely reports infeasibility.
template <typename C>
inline C simplexArtificialCost(std::span<const C> costs, int n, int m) {
    C magnitude = 1;
    for (C cost : costs) {
        if (cost == std::numeric_limits<C>::min())
            throw std::overflow_error("Artificial cost exceeds the cost type");
        magnitude = std::max(magnitude, cost < 0 ? C(-cost) : cost);
    }
    const C nodes = C(n) + C(2), arcs = C(m) + C(2);
    if (magnitude > (std::numeric_limits<C>::max() - C(1)) / nodes / arcs)
        throw std::overflow_error("Artificial cost exceeds the cost type");
    return magnitude * nodes * arcs + C(1);
}

template <typename C>
inline void checkDualBuffers(size_t n, size_t m, std::span<C> pi,
                             std::span<C> rc, std::span<C> lower,
                             std::span<C> upper) {
    if (pi.size() != n || rc.size() != m || lower.size() != m || upper.size() != m)
        throw std::invalid_argument("Dual buffer lengths do not match nodes and arcs");
    // Output buffers must not alias: otherwise even an apparently successful
    // call could overwrite an already-produced part of the certificate.
    std::span<C> buffers[] = {pi, rc, lower, upper};
    for (int i = 0; i < 4; ++i) for (int j = 0; j < i; ++j) {
        auto a = reinterpret_cast<std::uintptr_t>(buffers[i].data());
        auto b = reinterpret_cast<std::uintptr_t>(buffers[j].data());
        if (!buffers[i].empty() && !buffers[j].empty() &&
            a < b + buffers[j].size_bytes() && b < a + buffers[i].size_bytes())
            throw std::invalid_argument("Dual output buffers must not overlap");
    }
}
} // namespace pylmcf
#endif
