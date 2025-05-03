#include <array>

#include "basics.hpp"
#include "py_support.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


class Spectrum {
    const py::array positions;
    const py::array_t<LEMON_INT> intensities;

public:
    Spectrum(py::array positions, py::array_t<LEMON_INT> intensities)
        : positions(positions), intensities(intensities) {
        if (positions.shape()[1] != intensities.shape()[0]) {
            throw std::invalid_argument("Positions and intensities must have the same size");
        }}

    size_t size() const {
        return intensities.size();
    }

    std::vector<LEMON_INT> get_intensities() const {
        return std::vector<LEMON_INT>(intensities.data(), intensities.data() + size());
    }

    const py::array& py_get_positions() const {
        return positions;
    }

    const py::array_t<LEMON_INT>& py_get_intensities() const {
        return intensities;
    }
};