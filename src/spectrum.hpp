#ifndef PYLMCF_SPECTRUM_HPP
#define PYLMCF_SPECTRUM_HPP

#include <array>

#include "basics.hpp"
#include "py_support.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


class Spectrum {
    using Point_t = std::pair<const Spectrum*, size_t>;
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
        return {this, idx};
    }

    

    const py::array& py_get_positions() const {
        return py_positions;
    }

    const py::array_t<LEMON_INT>& py_get_intensities() const {
        return py_intensities;
    }
};

#endif // PYLMCF_SPECTRUM_HPP