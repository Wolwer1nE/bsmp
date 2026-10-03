#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "elastic_nullspace.h"

namespace {

float dot_product(const std::vector<float>& a, const std::vector<float>& b) {
    float value = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        value += a[i] * b[i];
    }
    return value;
}

}  // namespace

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== Rigid-body mode smoke test ===" << std::endl;

    const std::vector<float> coordinates = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
    };

    const bsmp::NodeLayout layout = bsmp::NodeLayout::fromNodeCoordinates(coordinates);
    const std::vector<std::vector<float>> modes = bsmp::build_rigid_body_modes(layout);

    if (modes.size() != 6) {
        std::cerr << "Expected 6 rigid-body modes, got " << modes.size() << std::endl;
        return 1;
    }

    for (size_t i = 0; i < modes.size(); ++i) {
        const float self_dot = dot_product(modes[i], modes[i]);
        std::cout << "mode[" << i << "] self-dot = " << self_dot << std::endl;
        if (std::fabs(self_dot - 1.0f) > 1e-5f) {
            std::cerr << "Mode " << i << " is not normalized" << std::endl;
            return 1;
        }
        for (size_t j = 0; j < i; ++j) {
            const float cross_dot = dot_product(modes[i], modes[j]);
            if (std::fabs(cross_dot) > 1e-5f) {
                std::cerr << "Modes " << i << " and " << j << " are not orthogonal" << std::endl;
                return 1;
            }
        }
    }

    std::cout << "Smoke test passed: 6 orthonormal rigid-body modes constructed." << std::endl;
    return 0;
}