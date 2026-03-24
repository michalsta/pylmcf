#ifndef LEMON_BASICS_HPP
#define LEMON_BASICS_HPP

#include <cstdint>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <span>


#define LEMON_INT int64_t
#define LEMON_INDEX int

#include <limits>
#include <string>

inline void assert_fits_lemon_index(size_t count, const char* what) {
    if (count > static_cast<size_t>(std::numeric_limits<LEMON_INDEX>::max()))
        throw std::overflow_error(
            std::string(what) + " count (" + std::to_string(count) +
            ") exceeds LEMON_INDEX max (" +
            std::to_string(std::numeric_limits<LEMON_INDEX>::max()) + ")");
}

template <typename T>
inline std::vector<T> sorted_copy(const std::vector<T>& vec, auto compare) {
    /* Use when T has deleted copy assignment and you can't just std::sort it */
    std::vector<size_t> indices;
    indices.reserve(vec.size());
    for (size_t ii = 0; ii < vec.size(); ii++)
        indices.push_back(ii);
    std::sort(indices.begin(), indices.end(),
              [&vec, compare](size_t a, size_t b) { return compare(vec[a], vec[b]); });
    std::vector<T> sorted_vec;
    sorted_vec.reserve(vec.size());
    for (size_t ii = 0; ii < vec.size(); ii++)
        sorted_vec.push_back(vec[indices[ii]]);
    return sorted_vec;
};


#endif // LEMON_BASICS_HPP