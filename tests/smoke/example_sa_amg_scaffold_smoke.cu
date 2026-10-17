#include <cuda_runtime.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "piezo_block_system.h"
#include "sa_amg_preconditioner.h"

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== SA-AMG scaffold smoke test ===" << std::endl;

    const int n = 6;
    const std::vector<float> c_u_dense = {
        4.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 7.0f, 0.5f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.5f, 5.0f, 0.2f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.2f, 9.0f, 0.3f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.3f, 6.0f, 0.4f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.4f, 8.0f,
    };

    auto c_u = bsmp::HostBlockMatrix::fromDense(n, n, c_u_dense).toDeviceMatrix();

    bsmp::SAAMGParameters params;
    params.regularization_epsilon = 1e-6f;
    params.setup_amg.max_levels = 3;
    params.setup_amg.min_coarse_block_rows = 1;

    auto preconditioner = bsmp::createSAAMGPreconditioner(*c_u, params);
    if (!preconditioner) {
        std::cerr << "Failed to construct SA-AMG scaffold preconditioner" << std::endl;
        return 1;
    }

    std::cout << "alpha = " << preconditioner->alpha() << std::endl;
    if (!(preconditioner->alpha() > 0.0f)) {
        std::cerr << "Expected positive regularization alpha" << std::endl;
        return 1;
    }

    const std::vector<float> rhs = {1.0f, -2.0f, 0.5f, 3.0f, -1.0f, 2.0f};
    std::vector<float> out(n, 0.0f);

    float* d_rhs = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_rhs, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_rhs, rhs.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    if (!preconditioner->apply(d_rhs, d_out)) {
        std::cerr << "SA-AMG scaffold apply() failed" << std::endl;
        cudaFree(d_rhs);
        cudaFree(d_out);
        return 1;
    }

    cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_rhs);
    cudaFree(d_out);

    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(out[i])) {
            std::cerr << "Non-finite output at index " << i << std::endl;
            return 1;
        }
    }

    std::cout << "apply(rhs) first entries:";
    for (int i = 0; i < n; ++i) {
        std::cout << ' ' << out[i];
    }
    std::cout << std::endl;
    std::cout << "Smoke test passed: separate SA-AMG scaffold builds on regularized C_u + alpha I." << std::endl;
    return 0;
}