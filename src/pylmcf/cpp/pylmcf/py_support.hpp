#ifndef PYLMCF_PY_SUPPORT_H
#define PYLMCF_PY_SUPPORT_H


#include <stdexcept>
#include <span>
#include <vector>
#include <cstring>
#include <cstdint>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>


namespace nb = nanobind;

// 1D host array used for every Python->C++ ndarray parameter. We deliberately do
// NOT request nb::c_contig here: that would make nanobind silently copy strided
// inputs, hiding an allocation+copy on this performance-critical path. Instead we
// accept the array as-is and fail loudly in numpy_to_span() if it is not
// stride-1, since the std::span below assumes contiguous memory.
template <typename T>
using ndarray_1d = nb::ndarray<T, nb::shape<-1>, nb::device::cpu>;

// A typed pointer must be aligned to its element: dereferencing a misaligned
// int64_t* is undefined behaviour, and the optimizer may act on it (e.g.
// peeling a vectorized loop to a 16-byte boundary assuming 8-byte alignment,
// then using aligned vector loads).  NumPy readily produces such arrays
// (flags.aligned == False): views at an odd offset into a byte buffer,
// np.frombuffer(..., offset=1), fields of packed structured arrays.
template <typename T>
void require_aligned(const void* data, size_t size) {
    const auto offset = reinterpret_cast<std::uintptr_t>(data) % alignof(T);
    if (size > 0 && offset != 0) {
        throw std::invalid_argument(
            "pylmcf requires arrays aligned to their element size (" +
            std::to_string(alignof(T)) + " bytes) and does not copy inputs "
            "(this is a performance-critical path), but received an array whose "
            "data starts " + std::to_string(offset) + " byte(s) past an aligned "
            "address (numpy flags.aligned is False). Make an aligned copy on the "
            "Python side before passing it in, e.g. `arr = arr.copy()` or "
            "`arr = np.require(arr, requirements=\"CA\")` "
            "(np.ascontiguousarray does not realign a contiguous array).");
    }
}

template <typename T>
std::span<T> numpy_to_span(ndarray_1d<T> array) {
    // stride(0) is in elements; a size<=1 array is trivially contiguous whatever
    // its reported stride.
    if (array.shape(0) > 1 && array.stride(0) != 1) {
        throw std::invalid_argument(
            "pylmcf requires C-contiguous arrays and does not copy inputs "
            "(this is a performance-critical path), but received a non-contiguous "
            "array with element stride " + std::to_string(array.stride(0)) + ". "
            "Make it contiguous on the Python side before passing it in, e.g. "
            "`arr = np.ascontiguousarray(arr)`.");
    }
    require_aligned<T>(array.data(), array.shape(0));
    return std::span<T>(static_cast<T*>(array.data()), array.shape(0));
}

template <typename T>
std::vector<T> numpy_to_vector(nb::ndarray<T, nb::shape<-1>> array) {
    require_aligned<T>(array.data(), array.shape(0));
    return std::vector<T>(static_cast<T*>(array.data()), static_cast<T*>(array.data()) + array.shape(0));
}

template <typename T>
nb::ndarray<T, nb::numpy, nb::shape<-1>> steal_mallocd_span_to_np_array(std::span<T> span) {
    auto capsule = nb::capsule(span.data(), [](void* data) noexcept { free(data); });
    return nb::ndarray<T, nb::numpy, nb::shape<-1>>(span.data(), { span.size() }, capsule);
}

template <typename T>
nb::ndarray<T> copy_vector_to_numpy(const std::vector<T>& vec) {
    nb::ndarray<T> arr(vec.size());

    // Copy the data
    std::memcpy(arr.mutable_data(), vec.data(), vec.size() * sizeof(T));

    return arr;
}

template <typename T>
nb::ndarray<T, nb::numpy, nb::shape<-1>> create_empty_numpy_array(size_t size) {
    T* data = new T[size];
    nb::capsule capsule(data, [](void* data) noexcept { delete[] static_cast<T*>(data); });
    return nb::ndarray<T, nb::numpy, nb::shape<-1>>(data, {size}, capsule);
}

// Snapshot API shared by Graph and the LCT wrappers. Into variants avoid
// allocations; all output arrays are writable, aligned, contiguous int64.
template <typename Owner, typename T>
nb::dict dual_snapshot(const Owner& owner, size_t n, size_t m) {
    auto pi = create_empty_numpy_array<T>(n);
    auto rc = create_empty_numpy_array<T>(m);
    auto lower = create_empty_numpy_array<T>(m);
    auto upper = create_empty_numpy_array<T>(m);
    owner.dual_values(std::span<T>(pi.data(), n), std::span<T>(rc.data(), m),
                      std::span<T>(lower.data(), m), std::span<T>(upper.data(), m));
    nb::dict out;
    out["potentials"] = pi; out["reduced_costs"] = rc;
    out["lower_bound_multipliers"] = lower; out["upper_bound_multipliers"] = upper;
    return out;
}

#endif // PYLMCF_PY_SUPPORT_H