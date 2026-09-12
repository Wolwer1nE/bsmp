#ifndef BSMP_CHEBYSHEV_SMOOTHER_H
#define BSMP_CHEBYSHEV_SMOOTHER_H

#include <vector>

namespace bsmp {

struct ChebyshevSmootherParameters {
    int steps = 3;
    int power_iterations = 12;
    float lower_bound_ratio = 0.3f;
    float eigenvalue_safety = 1.1f;
};

float estimate_max_eigenvalue(const std::vector<float>& a_dense,
                              const std::vector<float>& inverse_diagonal,
                              int n,
                              const ChebyshevSmootherParameters& parameters = ChebyshevSmootherParameters{});

void chebyshev_smooth(const std::vector<float>& a_dense,
                      const std::vector<float>& inverse_diagonal,
                      const std::vector<float>& rhs,
                      std::vector<float>& x,
                      float lambda_max,
                      const ChebyshevSmootherParameters& parameters = ChebyshevSmootherParameters{});

}  // namespace bsmp

#endif  // BSMP_CHEBYSHEV_SMOOTHER_H