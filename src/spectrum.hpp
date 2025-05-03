#include <array>

#include "basics.hpp"
#include "py_support.h"

class Spectrum {
    const py::array py_positions;
    const py::array_t<LEMON_INT> intensities;

public:
    Spectrum(py::array positions, py::array_t<LEMON_INT> intensities)
        : py_positions(positions), intensities(intensities) {
        if (positions.shape()[1] != intensities.shape()[0]) {
            throw std::invalid_argument("Positions and intensities must have the same size");
        }}

    size_t size() const {
        return intensities.size();
    }

    std::vector<LEMON_INT> get_intensities() const {
        return std::vector<LEMON_INT>(intensities.data(), intensities.data() + size());
    }
};