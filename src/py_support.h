#ifndef PYLMCF_PY_SUPPORT_H
#define PYLMCF_PY_SUPPORT_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <span>
#include <vector>


namespace py = pybind11;

template <typename T>
std::span<T> numpy_to_span(py::array_t<T> array) {
    py::buffer_info info = array.request();
    if (info.ndim != 1) {
        throw std::invalid_argument("Only 1D arrays are supported");
    }
    return std::span<T>(static_cast<T*>(info.ptr), info.size);
}

template <typename T>
std::vector<T> numpy_to_vector(py::array_t<T> array) {
    py::buffer_info info = array.request();
    if (info.ndim != 1) {
        throw std::invalid_argument("Only 1D arrays are supported");
    }
    return std::vector<T>(static_cast<T*>(info.ptr), static_cast<T*>(info.ptr) + info.size);
}

template <typename T>
py::array_t<T> mallocd_span_to_owning_numpy(std::span<T> span) {
    auto capsule = py::capsule(span.data(), [](void* data) { free(data); });
    return py::array_t<T>(span.size(), span.data(), capsule);
}

template <typename T>
py::array_t<T> vector_to_numpy_copy(const std::vector<T>& vec) {
    // Create a new py::array_t with the same size
    py::array_t<T> arr(vec.size());

    // Copy the data
    std::memcpy(arr.mutable_data(), vec.data(), vec.size() * sizeof(T));

    return arr;
}

#endif // PYLMCF_PY_SUPPORT_H