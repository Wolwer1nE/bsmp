#include "chebyshev_smoother.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace bsmp {

namespace {

float dot_product(const std::vector<float>& a, const std::vector<float>& b) {
    float value = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        value += a[i] * b[i];
    }
    return value;
}

float vector_norm(const std::vector<float>& v) {
    return std::sqrt(std::max(dot_product(v, v), 0.0f));
}

void normalize(std::vector<float>& v) {
    const float norm = vector_norm(v);
    if (norm <= 1e-12f) {
        return;
    }
    for (float& value : v) {
        value /= norm;
    }
}

std::vector<float> matvec_dense(const std::vector<float>& matrix,
                                int rows,
                                int cols,
                                const std::vector<float>& x) {
    std::vector<float> y(rows, 0.0f);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            y[row] += matrix[row * cols + col] * x[col];
        }
    }
    return y;
}

std::vector<float> apply_diagonal_scaled_operator(const std::vector<float>& a_dense,
                                                  const std::vector<float>& inverse_diagonal,
                                                  int n,
                                                  const std::vector<float>& x) {
    const std::vector<float> ax = matvec_dense(a_dense, n, n, x);
    std::vector<float> scaled(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        scaled[i] = inverse_diagonal[i] * ax[i];
    }
    return scaled;
}

void validate_inputs(const std::vector<float>& a_dense,
                     const std::vector<float>& inverse_diagonal,
                     int n,
                     int rhs_size = -1,
                     int x_size = -1) {
    if (n <= 0) {
        throw std::invalid_argument("Chebyshev smoother expects a positive matrix size");
    }
    if (static_cast<int>(a_dense.size()) != n * n) {
        throw std::invalid_argument("Chebyshev smoother expects a dense n-by-n matrix");
    }
    if (static_cast<int>(inverse_diagonal.size()) != n) {
        throw std::invalid_argument("Chebyshev smoother inverse diagonal size mismatch");
    }
    if (rhs_size >= 0 && rhs_size != n) {
        throw std::invalid_argument("Chebyshev smoother rhs size mismatch");
    }
    if (x_size >= 0 && x_size != n) {
        throw std::invalid_argument("Chebyshev smoother state vector size mismatch");
    }
}

}  // namespace

float estimate_max_eigenvalue(const std::vector<float>& a_dense,
                              const std::vector<float>& inverse_diagonal,
                              int n,
                              const ChebyshevSmootherParameters& parameters) {
    validate_inputs(a_dense, inverse_diagonal, n);
    if (parameters.power_iterations <= 0) {
        throw std::invalid_argument("Chebyshev power_iterations must be positive");
    }
    if (parameters.eigenvalue_safety < 1.0f) {
        throw std::invalid_argument("Chebyshev eigenvalue_safety must be >= 1");
    }

    std::vector<float> x(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        x[i] = 1.0f + 0.125f * static_cast<float>((i % 7) - 3);
    }
    normalize(x);

    float rayleigh = 0.0f;
    for (int iter = 0; iter < parameters.power_iterations; ++iter) {
        std::vector<float> y = apply_diagonal_scaled_operator(a_dense, inverse_diagonal, n, x);
        const float norm_y = vector_norm(y);
        if (norm_y <= 1e-12f) {
            return 0.0f;
        }
        for (float& value : y) {
            value /= norm_y;
        }
        x = std::move(y);
        const std::vector<float> bx = apply_diagonal_scaled_operator(a_dense, inverse_diagonal, n, x);
        rayleigh = dot_product(x, bx);
    }

    if (rayleigh <= 0.0f || !std::isfinite(rayleigh)) {
        const std::vector<float> bx = apply_diagonal_scaled_operator(a_dense, inverse_diagonal, n, x);
        rayleigh = std::max(vector_norm(bx), 1e-6f);
    }
    return parameters.eigenvalue_safety * rayleigh;
}

void chebyshev_smooth(const std::vector<float>& a_dense,
                      const std::vector<float>& inverse_diagonal,
                      const std::vector<float>& rhs,
                      std::vector<float>& x,
                      float lambda_max,
                      const ChebyshevSmootherParameters& parameters) {
    const int n = static_cast<int>(rhs.size());
    validate_inputs(a_dense, inverse_diagonal, n, static_cast<int>(rhs.size()), static_cast<int>(x.size()));
    if (parameters.steps <= 0) {
        throw std::invalid_argument("Chebyshev steps must be positive");
    }
    if (parameters.lower_bound_ratio <= 0.0f || parameters.lower_bound_ratio >= 1.0f) {
        throw std::invalid_argument("Chebyshev lower_bound_ratio must be in (0, 1)");
    }
    if (lambda_max <= 0.0f || !std::isfinite(lambda_max)) {
        throw std::invalid_argument("Chebyshev lambda_max must be positive and finite");
    }

    const float lambda_min = std::max(parameters.lower_bound_ratio * lambda_max, 1e-6f * lambda_max);
    const float center = 0.5f * (lambda_max + lambda_min);
    const float radius = 0.5f * (lambda_max - lambda_min);
    const float pi = 3.14159265358979323846f;

    for (int step = 0; step < parameters.steps; ++step) {
        const std::vector<float> ax = matvec_dense(a_dense, n, n, x);
        const float theta = std::cos(pi * (2.0f * static_cast<float>(step) + 1.0f)
                                     / (2.0f * static_cast<float>(parameters.steps)));
        const float omega = 1.0f / (center - radius * theta);
        for (int i = 0; i < n; ++i) {
            const float residual = rhs[i] - ax[i];
            x[i] += omega * inverse_diagonal[i] * residual;
        }
    }
}

}  // namespace bsmp