#ifndef PYLMCF_PY_SUPPORT_H
#define PYLMCF_PY_SUPPORT_H


#include <stdexcept>
#include <span>
#include <vector>
#include <cstring>

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
    return std::span<T>(static_cast<T*>(array.data()), array.shape(0));
}

template <typename T>
std::vector<T> numpy_to_vector(nb::ndarray<T, nb::shape<-1>> array) {
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

#endif // PYLMCF_PY_SUPPORT_H