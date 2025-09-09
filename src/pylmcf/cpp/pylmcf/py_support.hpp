#ifndef PYLMCF_PY_SUPPORT_H
#define PYLMCF_PY_SUPPORT_H


#include <stdexcept>
#include <span>
#include <vector>
#include <cstring>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>


namespace nb = nanobind;

template <typename T>
std::span<T> numpy_to_span(nb::ndarray<T, nb::shape<-1>> array) {
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