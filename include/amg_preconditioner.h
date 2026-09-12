#ifndef BSMP_AMG_PRECONDITIONER_H
#define BSMP_AMG_PRECONDITIONER_H

#include <memory>
#include <string>

#include "block_sparse_matrix.h"

enum class PreconditionerType {
    None,
    ScalarJacobi,
    BlockJacobi,
    AMG,
    SAAMG,
};

const char* preconditionerTypeName(PreconditionerType type);
bool parsePreconditionerType(const std::string& value, PreconditionerType& type_out);

struct AMGParameters {
    int max_levels = 4;
    int min_coarse_block_rows = 4;
    int pre_sweeps = 2;
    int post_sweeps = 2;
    int coarse_sweeps = 12;
    float relaxation = 0.8f;
};

struct PreconditionerOptions {
    PreconditionerType type = PreconditionerType::AMG;
    AMGParameters amg;
};

class LinearPreconditioner {
   public:
    virtual ~LinearPreconditioner() = default;

    virtual bool apply(const float* d_rhs, float* d_out) = 0;
    virtual PreconditionerType type() const = 0;
};

std::unique_ptr<LinearPreconditioner> createPreconditioner(BlockSparseMatrix& A,
                                                           const PreconditionerOptions& options);

#endif  // BSMP_AMG_PRECONDITIONER_H