#include "elastic_nullspace.h"

#include <cmath>
#include <stdexcept>

namespace bsmp {

namespace {

float dot_product(const std::vector<float>& a, const std::vector<float>& b) {
    float value = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        value += a[i] * b[i];
    }
    return value;
}

void normalize(std::vector<float>& v) {
    float norm_sq = dot_product(v, v);
    if (norm_sq <= 0.0f) {
        return;
    }
    const float inv_norm = 1.0f / std::sqrt(norm_sq);
    for (float& value : v) {
        value *= inv_norm;
    }
}

void orthonormalize(std::vector<float>& v, const std::vector<std::vector<float>>& basis) {
    for (int pass = 0; pass < 2; ++pass) {
        for (const auto& b : basis) {
            const float projection = dot_product(v, b);
            for (size_t i = 0; i < v.size(); ++i) {
                v[i] -= projection * b[i];
            }
        }
    }
    normalize(v);
}

}  // namespace

std::vector<std::vector<float>> build_rigid_body_modes(const NodeLayout& layout) {
    if (!layout.isValid()) {
        throw std::invalid_argument("NodeLayout is invalid");
    }

    const int nodes = layout.numNodes();
    std::vector<std::vector<float>> modes;
    modes.reserve(6);

    std::vector<float> tx(nodes * 3, 0.0f);
    std::vector<float> ty(nodes * 3, 0.0f);
    std::vector<float> tz(nodes * 3, 0.0f);
    std::vector<float> rx(nodes * 3, 0.0f);
    std::vector<float> ry(nodes * 3, 0.0f);
    std::vector<float> rz(nodes * 3, 0.0f);

    for (int node = 0; node < nodes; ++node) {
        const float x = layout.coordinates[node * 3 + 0];
        const float y = layout.coordinates[node * 3 + 1];
        const float z = layout.coordinates[node * 3 + 2];

        tx[node * 3 + 0] = 1.0f;
        ty[node * 3 + 1] = 1.0f;
        tz[node * 3 + 2] = 1.0f;

        rx[node * 3 + 0] = 0.0f;
        rx[node * 3 + 1] = -z;
        rx[node * 3 + 2] = y;

        ry[node * 3 + 0] = z;
        ry[node * 3 + 1] = 0.0f;
        ry[node * 3 + 2] = -x;

        rz[node * 3 + 0] = -y;
        rz[node * 3 + 1] = x;
        rz[node * 3 + 2] = 0.0f;
    }

    std::vector<std::vector<float>> raw_modes = {tx, ty, tz, rx, ry, rz};
    for (auto& mode : raw_modes) {
        orthonormalize(mode, modes);
        modes.push_back(mode);
    }

    return modes;
}

}  // namespace bsmp