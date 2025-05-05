#ifndef PYLMCF_SPECTRUM_HPP
#define PYLMCF_SPECTRUM_HPP

#include <array>

#include "basics.hpp"
#include "py_support.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


class Spectrum {
    using Point_t = std::pair<const py::array*, size_t>;
    const py::array py_positions;
    const py::array_t<LEMON_INT> py_intensities;
public:
    const std::span<const LEMON_INT> intensities;

    Spectrum(py::array positions, py::array_t<LEMON_INT> intensities)
        : py_positions(positions), py_intensities(intensities), intensities(numpy_to_span(intensities)) {
        if (positions.shape()[1] != intensities.shape()[0]) {
            throw std::invalid_argument("Positions and intensities must have the same size");
        }
    }

    size_t size() const {
        return intensities.size();
    }

    Point_t get_point(size_t idx) const {
        if (idx >= size()) {
            throw std::out_of_range("Index out of range");
        }
        return {&py_positions, idx};
    }

    std::pair<std::vector<size_t>, std::vector<LEMON_INT>> closer_than(
        const Point_t point,
        LEMON_INT max_dist,
        const py::function* wrapped_dist_fun
    ) const
    {
        std::vector<size_t> indices;
        std::vector<LEMON_INT> distances;

        py::object distances_obj = (*wrapped_dist_fun)(point, py_positions);
        py::array_t<LEMON_INT> distances_array = distances_obj.cast<py::array_t<LEMON_INT>>();
        py::buffer_info distances_info = distances_array.request();
        LEMON_INT* distances_ptr = static_cast<LEMON_INT*>(distances_info.ptr);
        if (distances_info.ndim != 1) {
            throw std::invalid_argument("Only 1D arrays are supported");
        }
        for (size_t ii = 0; ii < size(); ++ii) {
            if(distances_ptr[ii] <= max_dist) {
                indices.push_back(ii);
                distances.push_back(distances_ptr[ii]);
            }
        }
        return {indices, distances};
    }

    const py::array& py_get_positions() const {
        return py_positions;
    }

    const py::array_t<LEMON_INT>& py_get_intensities() const {
        return py_intensities;
    }
};

#endif // PYLMCF_SPECTRUM_HPP