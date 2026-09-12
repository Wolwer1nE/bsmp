#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cusolverDn.h>

#include "deflated_pcg_eigensolver.h"
#include "elastic_nullspace.h"
#include "mass_orthogonalization.h"
#include "node_layout.h"
#include "piezo_block_system.h"
#include "piezo_scaling.h"
#include "sa_amg_preconditioner.h"
#include "schur_operator.h"

namespace {

struct ExampleOptions {
    bool synthetic = true;
    std::string stiffness_path;
    std::string cu_path;
    std::string cuphi_path;
    std::string cphi_path;
    std::string mass_path;
    std::string coords_path;
    std::string ordering = "node-based";
    std::string dielectric_sign = "negated";
    std::string scaling_mode = "diag";
    std::string eigensolve_path = "auto";
    int grounded_dof = 0;
    int modes = 3;
    int max_iters = 400;
    float tol = 1e-3f;

    bsmp::SAAMGParameters sa_amg;
    bsmp::DeflatedPCGEigenSolverParameters eigen;
};

struct ScalarSparseTriplet {
    int row = 0;
    int col = 0;
    float value = 0.0f;
};

struct ScalarSparseMatrix {
    int num_rows = 0;
    int num_cols = 0;
    std::vector<ScalarSparseTriplet> entries;
};

struct FullMatrixIndexPartition {
    int num_nodes = 0;
    int mechanical_dofs = 0;
    int electrical_dofs = 0;
    std::vector<int> mechanical_indices;
    std::vector<int> electrical_indices;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

float eigenvalue_to_frequency_hz(float eigenvalue) {
    constexpr float kTwoPi = 6.28318530717958647692f;
    if (eigenvalue <= 0.0f) {
        return -1.0f;
    }
    return std::sqrt(eigenvalue) / kTwoPi;
}

bsmp::PiezoEquilibrationScaling build_selected_scaling(const bsmp::PiezoBlockSystem& system,
                                                       const ExampleOptions& options) {
    if (options.scaling_mode == "none") {
        return bsmp::build_identity_scaling(system);
    }
    if (options.scaling_mode == "field") {
        return bsmp::build_field_scaling(system);
    }
    if (options.scaling_mode == "diag") {
        return bsmp::build_equilibration_scaling(system);
    }
    fail("Unsupported scaling mode: " + options.scaling_mode + ". Expected none, field, or diag.");
}

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "Synthetic demo mode (default):\n"
        << "  " << program << "\n\n"
        << "File-backed mode (full matrices):\n"
        << "  " << program << " --stiffness <file> --mass <file> --coords <file> [options]\n\n"
        << "Legacy file-backed mode (pre-split blocks):\n"
        << "  " << program << " --cu <file> --cuphi <file> --cphi <file> --mass <file> --coords <file> [options]\n\n"
        << "Options:\n"
        << "  --synthetic                      Force synthetic demo mode\n"
        << "  --stiffness <file>               Full mixed stiffness matrix C\n"
        << "  --cu <file>                      Mechanical stiffness block C_u\n"
        << "  --cuphi <file>                   Coupling block C_uphi\n"
        << "  --cphi <file>                    Dielectric block C_phi\n"
        << "  --mass <file>                    Full mixed mass matrix M (or mechanical M in legacy block mode)\n"
        << "  --coords <file>                  Node coordinates file (x y z per line)\n"
        << "  --ordering <node-based|block-wise>  Ordering of the provided full matrices\n"
        << "  --dielectric-sign <negated|as-is>   How to interpret the Phi-Phi block inside full C (default: negated)\n"
        << "  --scaling <none|field|diag>      System preprocessing before SA-AMG / eigensolve (default: diag)\n"
        << "  --eigensolve-path <auto|sa-amg|unpreconditioned|explicit-schur-cusolver>\n"
        << "                                   Which eigensolver branch to run (default: auto)\n"
        << "  --grounded-dof <index>           Electrical DOF to ground (default: 0)\n"
        << "  --modes <count>                  Number of eigenpairs to compute\n"
        << "  --tol <value>                    Deflated PCG tolerance\n"
        << "  --max-iters <count>              Deflated PCG iteration limit\n"
        << "  --sa-amg-regularization-epsilon <value>\n"
        << "  --sa-amg-pre-sweeps <count>\n"
        << "  --sa-amg-post-sweeps <count>\n"
        << "  --sa-amg-jacobi-damping <value>\n"
        << "  --sa-amg-prolongation-damping <value>\n"
        << "  --sa-amg-use-chebyshev <0|1>\n"
        << "  --verbose                        Print per-iteration eigensolver diagnostics\n"
        << "  --help                           Show this message\n\n"
        << "Notes:\n"
        << "  - In full-matrix mode, the example extracts C_u/C_uphi/C_phi/M_u internally from the provided full C and M.\n"
        << "  - The default assumes the full mixed stiffness stores the dielectric block as -C_phi in the Phi-Phi corner.\n";
}

int parse_int(const std::string& text, const std::string& label) {
    try {
        return std::stoi(text);
    } catch (const std::exception&) {
        fail("Failed to parse integer for " + label + ": " + text);
    }
}

float parse_float(const std::string& text, const std::string& label) {
    try {
        return std::stof(text);
    } catch (const std::exception&) {
        fail("Failed to parse float for " + label + ": " + text);
    }
}

std::string require_value(int argc, char** argv, int& index, const std::string& option) {
    if (index + 1 >= argc) {
        fail("Missing value for option " + option);
    }
    ++index;
    return argv[index];
}

bool has_full_matrix_inputs(const ExampleOptions& options) {
    return !options.stiffness_path.empty();
}

bool has_legacy_block_inputs(const ExampleOptions& options) {
    return !options.cu_path.empty() || !options.cuphi_path.empty() || !options.cphi_path.empty();
}

ExampleOptions parse_options(int argc, char** argv) {
    ExampleOptions options;
    options.sa_amg.regularization_epsilon = 1e-6f;
    options.sa_amg.hierarchy.pre_sweeps = 2;
    options.sa_amg.hierarchy.post_sweeps = 2;
    options.sa_amg.hierarchy.jacobi_damping = 0.8f;
    options.sa_amg.hierarchy.prolongation_damping = 0.6666667f;
    options.eigen.num_eigenpairs = options.modes;
    options.eigen.max_iterations = options.max_iters;
    options.eigen.tolerance = options.tol;
    options.eigen.random_seed = 7u;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--synthetic") {
            options.synthetic = true;
        } else if (arg == "--stiffness") {
            options.synthetic = false;
            options.stiffness_path = require_value(argc, argv, i, arg);
        } else if (arg == "--cu") {
            options.synthetic = false;
            options.cu_path = require_value(argc, argv, i, arg);
        } else if (arg == "--cuphi") {
            options.synthetic = false;
            options.cuphi_path = require_value(argc, argv, i, arg);
        } else if (arg == "--cphi") {
            options.synthetic = false;
            options.cphi_path = require_value(argc, argv, i, arg);
        } else if (arg == "--mass") {
            options.synthetic = false;
            options.mass_path = require_value(argc, argv, i, arg);
        } else if (arg == "--coords") {
            options.synthetic = false;
            options.coords_path = require_value(argc, argv, i, arg);
        } else if (arg == "--ordering") {
            options.ordering = require_value(argc, argv, i, arg);
        } else if (arg == "--dielectric-sign") {
            options.dielectric_sign = require_value(argc, argv, i, arg);
        } else if (arg == "--scaling") {
            options.scaling_mode = require_value(argc, argv, i, arg);
        } else if (arg == "--eigensolve-path") {
            options.eigensolve_path = require_value(argc, argv, i, arg);
        } else if (arg == "--grounded-dof") {
            options.grounded_dof = parse_int(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--modes") {
            options.modes = parse_int(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--tol") {
            options.tol = parse_float(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--max-iters") {
            options.max_iters = parse_int(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--sa-amg-regularization-epsilon") {
            options.sa_amg.regularization_epsilon = parse_float(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--sa-amg-pre-sweeps") {
            options.sa_amg.hierarchy.pre_sweeps = parse_int(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--sa-amg-post-sweeps") {
            options.sa_amg.hierarchy.post_sweeps = parse_int(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--sa-amg-jacobi-damping") {
            options.sa_amg.hierarchy.jacobi_damping = parse_float(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--sa-amg-prolongation-damping") {
            options.sa_amg.hierarchy.prolongation_damping = parse_float(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--sa-amg-use-chebyshev") {
            options.sa_amg.hierarchy.use_chebyshev = parse_int(require_value(argc, argv, i, arg), arg) != 0;
        } else if (arg == "--verbose") {
            options.eigen.verbose = true;
        } else {
            fail("Unknown option: " + arg + ". Use --help to see available options.");
        }
    }

    if (options.ordering != "node-based" && options.ordering != "block-wise") {
        fail("Unsupported ordering label: " + options.ordering + ". Expected node-based or block-wise.");
    }
    if (options.dielectric_sign != "negated" && options.dielectric_sign != "as-is") {
        fail("Unsupported dielectric sign mode: " + options.dielectric_sign + ". Expected negated or as-is.");
    }
    if (options.scaling_mode != "none" && options.scaling_mode != "field" && options.scaling_mode != "diag") {
        fail("Unsupported scaling mode: " + options.scaling_mode + ". Expected none, field, or diag.");
    }
    if (options.eigensolve_path != "auto" &&
        options.eigensolve_path != "sa-amg" &&
        options.eigensolve_path != "unpreconditioned" &&
        options.eigensolve_path != "explicit-schur-cusolver") {
        fail("Unsupported eigensolve path: " + options.eigensolve_path +
             ". Expected auto, sa-amg, unpreconditioned, or explicit-schur-cusolver.");
    }
    if (!options.synthetic) {
        const bool full_inputs = has_full_matrix_inputs(options);
        const bool legacy_inputs = has_legacy_block_inputs(options);
        if (full_inputs && legacy_inputs) {
            fail("Provide either full-matrix inputs (--stiffness/--mass/--coords) or legacy block inputs (--cu/--cuphi/--cphi/--mass/--coords), not both.");
        }
        if (full_inputs) {
            if (options.mass_path.empty() || options.coords_path.empty()) {
                fail("Full-matrix mode requires --stiffness, --mass, and --coords.");
            }
        } else {
            if (options.cu_path.empty() || options.cuphi_path.empty() || options.cphi_path.empty() ||
                options.mass_path.empty() || options.coords_path.empty()) {
                fail("Legacy block mode requires --cu, --cuphi, --cphi, --mass, and --coords.");
            }
        }
    }
    if (options.modes <= 0) {
        fail("--modes must be positive.");
    }
    if (options.max_iters <= 0) {
        fail("--max-iters must be positive.");
    }
    if (options.tol <= 0.0f) {
        fail("--tol must be positive.");
    }

    options.eigen.num_eigenpairs = options.modes;
    options.eigen.max_iterations = options.max_iters;
    options.eigen.tolerance = options.tol;
    return options;
}

std::string first_nonempty_line(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        fail("Failed to open sparse matrix file: " + path);
    }

    std::string line;
    while (std::getline(in, line)) {
        const size_t pos = line.find_first_not_of(" \t\r\n");
        if (pos != std::string::npos) {
            return line.substr(pos);
        }
    }
    return std::string();
}

std::string to_lower(std::string text) {
    for (char& ch : text) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return text;
}

void symmetrize_if_triangular(ScalarSparseMatrix& matrix, const std::string& path) {
    if (matrix.num_rows != matrix.num_cols) {
        return;
    }

    std::map<std::pair<int, int>, double> values;
    double upper_abs_sum = 0.0;
    double lower_abs_sum = 0.0;
    for (const auto& entry : matrix.entries) {
        values[std::make_pair(entry.row, entry.col)] += static_cast<double>(entry.value);
        if (entry.row < entry.col) {
            upper_abs_sum += std::fabs(static_cast<double>(entry.value));
        } else if (entry.row > entry.col) {
            lower_abs_sum += std::fabs(static_cast<double>(entry.value));
        }
    }

    const double triangle_tolerance = 1e-12;
    const bool upper_only = upper_abs_sum > triangle_tolerance &&
                            lower_abs_sum <= triangle_tolerance * std::max(1.0, upper_abs_sum);
    const bool lower_only = lower_abs_sum > triangle_tolerance &&
                            upper_abs_sum <= triangle_tolerance * std::max(1.0, lower_abs_sum);
    if (!upper_only && !lower_only) {
        return;
    }

    std::vector<std::pair<std::pair<int, int>, double>> mirrored_entries;
    mirrored_entries.reserve(values.size());
    for (const auto& item : values) {
        const int row = item.first.first;
        const int col = item.first.second;
        if (row == col) {
            continue;
        }
        if (upper_only && row < col && values.count(std::make_pair(col, row)) == 0) {
            mirrored_entries.push_back({std::make_pair(col, row), item.second});
        } else if (lower_only && row > col && values.count(std::make_pair(col, row)) == 0) {
            mirrored_entries.push_back({std::make_pair(col, row), item.second});
        }
    }

    for (const auto& mirrored : mirrored_entries) {
        values[mirrored.first] += mirrored.second;
    }

    matrix.entries.clear();
    matrix.entries.reserve(values.size());
    for (const auto& item : values) {
        if (std::fabs(item.second) <= triangle_tolerance) {
            continue;
        }
        matrix.entries.push_back({item.first.first, item.first.second, static_cast<float>(item.second)});
    }
}

void complete_missing_symmetric_pairs(ScalarSparseMatrix& matrix) {
    if (matrix.num_rows != matrix.num_cols) {
        return;
    }

    std::map<std::pair<int, int>, double> values;
    for (const auto& entry : matrix.entries) {
        values[std::make_pair(entry.row, entry.col)] += static_cast<double>(entry.value);
    }

    std::vector<std::pair<int, int>> processed_pairs;
    processed_pairs.reserve(values.size());
    for (const auto& item : values) {
        const int row = item.first.first;
        const int col = item.first.second;
        if (row >= col) {
            continue;
        }
        processed_pairs.push_back(item.first);
    }

    for (const auto& key : processed_pairs) {
        const int row = key.first;
        const int col = key.second;
        const auto transpose_key = std::make_pair(col, row);

        const auto it = values.find(key);
        const auto jt = values.find(transpose_key);
        const double upper = (it != values.end()) ? it->second : 0.0;
        const double lower = (jt != values.end()) ? jt->second : 0.0;
        const bool has_upper = (it != values.end());
        const bool has_lower = (jt != values.end());

        if (has_upper && has_lower) {
            const double symmetric_value = 0.5 * (upper + lower);
            values[key] = symmetric_value;
            values[transpose_key] = symmetric_value;
        } else if (has_upper) {
            values[transpose_key] = upper;
        } else if (has_lower) {
            values[key] = lower;
        }
    }

    constexpr double kDropTolerance = 1e-12;
    matrix.entries.clear();
    matrix.entries.reserve(values.size());
    for (const auto& item : values) {
        if (std::fabs(item.second) <= kDropTolerance) {
            continue;
        }
        matrix.entries.push_back({item.first.first, item.first.second, static_cast<float>(item.second)});
    }
}

ScalarSparseMatrix load_plain_triplet_sparse(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        fail("Failed to open sparse matrix file: " + path);
    }

    ScalarSparseMatrix matrix;
    std::string line;
    while (std::getline(in, line)) {
        const size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) {
            continue;
        }
        if (line[start] == '#' || line[start] == '%') {
            continue;
        }

        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        long long row = -1;
        long long col = -1;
        double value = 0.0;
        if (!(iss >> row >> col >> value)) {
            fail("Failed to parse sparse matrix triplet line: " + line);
        }
        if (row < 0 || col < 0) {
            fail("Sparse matrix indices must be non-negative in file: " + path);
        }

        matrix.entries.push_back({static_cast<int>(row), static_cast<int>(col), static_cast<float>(value)});
        matrix.num_rows = std::max(matrix.num_rows, static_cast<int>(row) + 1);
        matrix.num_cols = std::max(matrix.num_cols, static_cast<int>(col) + 1);
    }

    if (matrix.entries.empty()) {
        fail("Sparse matrix file is empty: " + path);
    }
    symmetrize_if_triangular(matrix, path);
    complete_missing_symmetric_pairs(matrix);
    return matrix;
}

ScalarSparseMatrix load_matrix_market_sparse(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        fail("Failed to open Matrix Market file: " + path);
    }

    std::string header;
    if (!std::getline(in, header)) {
        fail("Matrix Market file is empty: " + path);
    }

    std::istringstream hs(header);
    std::string mm, object, format, field, symmetry;
    hs >> mm >> object >> format >> field >> symmetry;
    object = to_lower(object);
    format = to_lower(format);
    field = to_lower(field);
    symmetry = to_lower(symmetry);
    if (mm != "%%MatrixMarket" || object != "matrix" || format != "coordinate") {
        fail("Only Matrix Market coordinate matrices are supported: " + path);
    }
    if (field != "real" && field != "double") {
        fail("Only real Matrix Market matrices are supported: " + path);
    }
    const bool symmetric = (symmetry == "symmetric");

    std::string line;
    do {
        if (!std::getline(in, line)) {
            fail("Unexpected EOF before Matrix Market size line: " + path);
        }
    } while (!line.empty() && line[0] == '%');

    std::istringstream size_stream(line);
    long long rows = 0;
    long long cols = 0;
    long long nnz = 0;
    if (!(size_stream >> rows >> cols >> nnz)) {
        fail("Failed to parse Matrix Market size line in: " + path);
    }

    ScalarSparseMatrix matrix;
    matrix.num_rows = static_cast<int>(rows);
    matrix.num_cols = static_cast<int>(cols);
    matrix.entries.reserve(static_cast<size_t>(symmetric ? (2 * nnz) : nnz));

    for (long long index = 0; index < nnz; ++index) {
        long long row = 0;
        long long col = 0;
        double value = 0.0;
        if (!(in >> row >> col >> value)) {
            fail("Failed to read Matrix Market entry #" + std::to_string(index) + " from: " + path);
        }

        const int zero_row = static_cast<int>(row) - 1;
        const int zero_col = static_cast<int>(col) - 1;
        matrix.entries.push_back({zero_row, zero_col, static_cast<float>(value)});
        if (symmetric && zero_row != zero_col) {
            matrix.entries.push_back({zero_col, zero_row, static_cast<float>(value)});
        }
    }

    if (!symmetric) {
        symmetrize_if_triangular(matrix, path);
        complete_missing_symmetric_pairs(matrix);
    }

    return matrix;
}

ScalarSparseMatrix load_scalar_sparse_matrix_auto(const std::string& path) {
    const std::string first = first_nonempty_line(path);
    if (!first.empty() && first.rfind("%%MatrixMarket", 0) == 0) {
        return load_matrix_market_sparse(path);
    }
    return load_plain_triplet_sparse(path);
}

std::vector<float> load_coordinates_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        fail("Failed to open coordinates file: " + path);
    }

    std::vector<float> coordinates;
    std::string line;
    while (std::getline(in, line)) {
        const size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) {
            continue;
        }
        if (line[start] == '#' || line[start] == '%') {
            continue;
        }

        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        std::vector<float> values;
        float value = 0.0f;
        while (iss >> value) {
            values.push_back(value);
        }
        if (values.size() < 3) {
            fail("Failed to parse coordinate line: " + line);
        }
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        if (values.size() >= 4) {
            x = values[1];
            y = values[2];
            z = values[3];
        } else {
            x = values[0];
            y = values[1];
            z = values[2];
        }
        coordinates.push_back(x);
        coordinates.push_back(y);
        coordinates.push_back(z);
    }

    if (coordinates.empty()) {
        fail("Coordinates file is empty: " + path);
    }
    return coordinates;
}

FullMatrixIndexPartition build_full_matrix_index_partition(const ExampleOptions& options,
                                                          const ScalarSparseMatrix& full_matrix,
                                                          const std::vector<float>& coordinates) {
    if (coordinates.size() % 3 != 0) {
        fail("Coordinate vector must contain triples (x y z).");
    }

    FullMatrixIndexPartition partition;
    partition.num_nodes = static_cast<int>(coordinates.size() / 3);
    partition.mechanical_dofs = partition.num_nodes * 3;
    partition.electrical_dofs = full_matrix.num_rows - partition.mechanical_dofs;

    if (full_matrix.num_rows != full_matrix.num_cols) {
        fail("Full stiffness matrix must be square.");
    }
    if (partition.electrical_dofs <= 0) {
        fail("Full stiffness dimension is too small to contain electrical DOFs inferred from coords.");
    }

    partition.mechanical_indices.reserve(partition.mechanical_dofs);
    partition.electrical_indices.reserve(partition.electrical_dofs);

    if (options.ordering == "block-wise") {
        for (int index = 0; index < partition.mechanical_dofs; ++index) {
            partition.mechanical_indices.push_back(index);
        }
        for (int index = 0; index < partition.electrical_dofs; ++index) {
            partition.electrical_indices.push_back(partition.mechanical_dofs + index);
        }
    } else {
        if (partition.electrical_dofs != partition.num_nodes) {
            fail("node-based full-matrix mode currently requires one electrical DOF per node.");
        }
        for (int node = 0; node < partition.num_nodes; ++node) {
            partition.mechanical_indices.push_back(node * 4 + 0);
            partition.mechanical_indices.push_back(node * 4 + 1);
            partition.mechanical_indices.push_back(node * 4 + 2);
            partition.electrical_indices.push_back(node * 4 + 3);
        }
    }

    return partition;
}

std::unique_ptr<BlockSparseMatrix> build_block_matrix_from_triplets(int num_rows,
                                                                    int num_cols,
                                                                    const std::vector<ScalarSparseTriplet>& triplets) {
    BlockSparseMatrixConfig config{};
    config.num_rows = num_rows;
    config.num_cols = num_cols;
    config.block_size = BSMP_BLOCK_SIZE;

    std::map<std::pair<int, int>, std::vector<float>> blocks;
    for (const auto& triplet : triplets) {
        const int block_row = triplet.row / BSMP_BLOCK_SIZE;
        const int block_col = triplet.col / BSMP_BLOCK_SIZE;
        const int local_row = triplet.row % BSMP_BLOCK_SIZE;
        const int local_col = triplet.col % BSMP_BLOCK_SIZE;
        auto& block = blocks[std::make_pair(block_row, block_col)];
        if (block.empty()) {
            block.assign(BSMP_BLOCK_SIZE * BSMP_BLOCK_SIZE, 0.0f);
        }
        block[local_row * BSMP_BLOCK_SIZE + local_col] += triplet.value;
    }

    config.num_nonzero_blocks = static_cast<int>(blocks.size());
    std::vector<int> block_rows;
    std::vector<int> block_cols;
    std::vector<float> block_data;
    block_rows.reserve(blocks.size());
    block_cols.reserve(blocks.size());
    block_data.reserve(blocks.size() * BSMP_BLOCK_SIZE * BSMP_BLOCK_SIZE);

    for (const auto& item : blocks) {
        block_rows.push_back(item.first.first);
        block_cols.push_back(item.first.second);
        block_data.insert(block_data.end(), item.second.begin(), item.second.end());
    }

    std::unique_ptr<BlockSparseMatrix> matrix(new BlockSparseMatrix(config));
    matrix->initialize(block_rows, block_cols, block_data);
    return matrix;
}

bsmp::PiezoBlockSystem load_system_from_full_matrices(const ExampleOptions& options,
                                                      const std::vector<float>& coordinates) {
    const ScalarSparseMatrix full_stiffness = load_scalar_sparse_matrix_auto(options.stiffness_path);
    const ScalarSparseMatrix full_mass = load_scalar_sparse_matrix_auto(options.mass_path);

    if (full_mass.num_rows != full_mass.num_cols) {
        fail("Full mass matrix must be square.");
    }
    if (full_stiffness.num_rows != full_mass.num_rows || full_stiffness.num_cols != full_mass.num_cols) {
        fail("Full stiffness and mass matrices must have the same dimensions.");
    }

    const FullMatrixIndexPartition partition = build_full_matrix_index_partition(options,
                                                                                full_stiffness,
                                                                                coordinates);

    std::vector<int> mechanical_local(full_stiffness.num_rows, -1);
    std::vector<int> electrical_local(full_stiffness.num_rows, -1);
    for (size_t local = 0; local < partition.mechanical_indices.size(); ++local) {
        mechanical_local[partition.mechanical_indices[local]] = static_cast<int>(local);
    }
    for (size_t local = 0; local < partition.electrical_indices.size(); ++local) {
        electrical_local[partition.electrical_indices[local]] = static_cast<int>(local);
    }

    std::vector<ScalarSparseTriplet> cu_entries;
    std::vector<ScalarSparseTriplet> cuphi_entries;
    std::vector<ScalarSparseTriplet> cphi_entries;
    std::vector<ScalarSparseTriplet> mass_entries;

    for (const auto& triplet : full_stiffness.entries) {
        const int row_u = mechanical_local[triplet.row];
        const int col_u = mechanical_local[triplet.col];
        const int row_phi = electrical_local[triplet.row];
        const int col_phi = electrical_local[triplet.col];

        if (row_u >= 0 && col_u >= 0) {
            cu_entries.push_back({row_u, col_u, triplet.value});
        } else if (row_u >= 0 && col_phi >= 0) {
            cuphi_entries.push_back({row_u, col_phi, triplet.value});
        } else if (row_phi >= 0 && col_phi >= 0) {
            const float dielectric_value = (options.dielectric_sign == "negated") ? -triplet.value : triplet.value;
            cphi_entries.push_back({row_phi, col_phi, dielectric_value});
        }
    }

    for (const auto& triplet : full_mass.entries) {
        const int row_u = mechanical_local[triplet.row];
        const int col_u = mechanical_local[triplet.col];
        if (row_u >= 0 && col_u >= 0) {
            mass_entries.push_back({row_u, col_u, triplet.value});
        }
    }

    return bsmp::PiezoBlockSystem(
        build_block_matrix_from_triplets(partition.mechanical_dofs, partition.mechanical_dofs, cu_entries),
        build_block_matrix_from_triplets(partition.mechanical_dofs, partition.electrical_dofs, cuphi_entries),
        build_block_matrix_from_triplets(partition.electrical_dofs, partition.electrical_dofs, cphi_entries),
        build_block_matrix_from_triplets(partition.mechanical_dofs, partition.mechanical_dofs, mass_entries),
        partition.mechanical_dofs,
        partition.electrical_dofs);
}

bsmp::PiezoBlockSystem load_system_from_files(const ExampleOptions& options) {
    BlockSparseMatrixConfig cu_cfg{}, cuphi_cfg{}, cphi_cfg{}, mass_cfg{};
    std::vector<int> cu_rows, cu_cols, cuphi_rows, cuphi_cols, cphi_rows, cphi_cols, mass_rows, mass_cols;
    std::vector<float> cu_data, cuphi_data, cphi_data, mass_data;

    if (!bsmp::load_block_sparse_matrix_auto(options.cu_path, BSMP_BLOCK_SIZE, cu_cfg, cu_rows, cu_cols, cu_data) ||
        !bsmp::load_block_sparse_matrix_auto(options.cuphi_path, BSMP_BLOCK_SIZE, cuphi_cfg, cuphi_rows, cuphi_cols, cuphi_data) ||
        !bsmp::load_block_sparse_matrix_auto(options.cphi_path, BSMP_BLOCK_SIZE, cphi_cfg, cphi_rows, cphi_cols, cphi_data) ||
        !bsmp::load_block_sparse_matrix_auto(options.mass_path, BSMP_BLOCK_SIZE, mass_cfg, mass_rows, mass_cols, mass_data)) {
        fail("Failed to load one or more block matrices for the article eigen example.");
    }

    if (cu_cfg.num_rows != cu_cfg.num_cols || mass_cfg.num_rows != mass_cfg.num_cols) {
        fail("C_u and M must be square.");
    }
    if (cu_cfg.num_rows != mass_cfg.num_rows || cu_cfg.num_cols != mass_cfg.num_cols) {
        fail("C_u and M must have the same size.");
    }
    if (cuphi_cfg.num_rows != cu_cfg.num_rows || cuphi_cfg.num_cols != cphi_cfg.num_rows) {
        fail("C_uphi dimensions must match C_u rows and C_phi rows.");
    }
    if (cphi_cfg.num_rows != cphi_cfg.num_cols) {
        fail("C_phi must be square.");
    }

    std::unique_ptr<BlockSparseMatrix> c_u(new BlockSparseMatrix(cu_cfg));
    std::unique_ptr<BlockSparseMatrix> c_uphi(new BlockSparseMatrix(cuphi_cfg));
    std::unique_ptr<BlockSparseMatrix> c_phi(new BlockSparseMatrix(cphi_cfg));
    std::unique_ptr<BlockSparseMatrix> mass(new BlockSparseMatrix(mass_cfg));
    c_u->initialize(cu_rows, cu_cols, cu_data);
    c_uphi->initialize(cuphi_rows, cuphi_cols, cuphi_data);
    c_phi->initialize(cphi_rows, cphi_cols, cphi_data);
    mass->initialize(mass_rows, mass_cols, mass_data);

    return bsmp::PiezoBlockSystem(std::move(c_u),
                                  std::move(c_uphi),
                                  std::move(c_phi),
                                  std::move(mass),
                                  cu_cfg.num_rows,
                                  cphi_cfg.num_rows);
}

bsmp::PiezoBlockSystem build_synthetic_system(std::vector<float>& coordinates_out) {
    coordinates_out = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 0.0f,
    };

    const int mechanical_dofs = 12;
    const int electrical_dofs = 4;

    const std::vector<float> c_u = {
        6,0,0,-2,0,0,-2,0,0,0,0,0,
        0,6,0,0,-2,0,0,-2,0,0,0,0,
        0,0,6,0,0,-2,0,0,-2,0,0,0,
        -2,0,0,6,0,0,0,0,0,-2,0,0,
        0,-2,0,0,6,0,0,0,0,0,-2,0,
        0,0,-2,0,0,6,0,0,0,0,0,-2,
        -2,0,0,0,0,0,6,0,0,-2,0,0,
        0,-2,0,0,0,0,0,6,0,0,-2,0,
        0,0,-2,0,0,0,0,0,6,0,0,-2,
        0,0,0,-2,0,0,-2,0,0,6,0,0,
        0,0,0,0,-2,0,0,-2,0,0,6,0,
        0,0,0,0,0,-2,0,0,-2,0,0,6,
    };

    const std::vector<float> c_uphi = {
        0.30f, 0.00f, 0.00f, 0.10f,
        0.10f, 0.05f, 0.00f, 0.00f,
        0.05f, 0.00f, 0.10f, 0.00f,
        0.25f, 0.10f, 0.00f, 0.00f,
        0.00f, 0.20f, 0.10f, 0.00f,
        0.00f, 0.00f, 0.20f, 0.10f,
        0.00f, 0.15f, 0.00f, 0.25f,
        0.05f, 0.10f, 0.15f, 0.00f,
        0.10f, 0.00f, 0.25f, 0.05f,
        0.20f, 0.00f, 0.05f, 0.20f,
        0.00f, 0.25f, 0.00f, 0.15f,
        0.15f, 0.10f, 0.10f, 0.00f,
    };

    const std::vector<float> c_phi = {
        3.0f, -1.0f,  0.0f, -1.0f,
       -1.0f,  3.0f, -1.0f, -0.5f,
        0.0f, -1.0f,  2.5f, -0.5f,
       -1.0f, -0.5f, -0.5f,  3.0f,
    };

    const std::vector<float> mass = {
        2,0,0,0,0,0,0,0,0,0,0,0,
        0,2,0,0,0,0,0,0,0,0,0,0,
        0,0,2,0,0,0,0,0,0,0,0,0,
        0,0,0,2,0,0,0,0,0,0,0,0,
        0,0,0,0,2,0,0,0,0,0,0,0,
        0,0,0,0,0,2,0,0,0,0,0,0,
        0,0,0,0,0,0,3,0,0,0,0,0,
        0,0,0,0,0,0,0,3,0,0,0,0,
        0,0,0,0,0,0,0,0,3,0,0,0,
        0,0,0,0,0,0,0,0,0,3,0,0,
        0,0,0,0,0,0,0,0,0,0,3,0,
        0,0,0,0,0,0,0,0,0,0,0,3,
    };

    return bsmp::PiezoBlockSystem::fromDense(c_u, c_uphi, c_phi, mass, mechanical_dofs, electrical_dofs);
}

float max_relative_residual(const bsmp::SchurOperator& schur,
                           BlockSparseMatrix& mass,
                           const bsmp::DeflatedPCGEigenResult& result) {
    float worst = 0.0f;
    for (const auto& eigenpair : result.eigenpairs) {
        if (eigenpair.eigenvector.empty()) {
            continue;
        }
        std::vector<float> sx;
        bsmp::make_schur_operator(const_cast<bsmp::SchurOperator&>(schur))(eigenpair.eigenvector, sx);
        const std::vector<float> mx = bsmp::apply_mass_matrix(mass, eigenpair.eigenvector);
        float diff_sq = 0.0f;
        float sx_sq = 0.0f;
        for (size_t i = 0; i < sx.size(); ++i) {
            const float diff = sx[i] - eigenpair.eigenvalue * mx[i];
            diff_sq += diff * diff;
            sx_sq += sx[i] * sx[i];
        }
        const float relative = std::sqrt(diff_sq) / (std::sqrt(sx_sq) > 0.0f ? std::sqrt(sx_sq) : 1.0f);
        worst = std::max(worst, relative);
    }
    return worst;
}

struct SolveAttempt {
    bsmp::DeflatedPCGEigenResult result;
    long long solve_ms = 0;
    float worst_relative_residual = std::numeric_limits<float>::infinity();
    std::string label;
};

void check_cusolver_status(cusolverStatus_t status, const std::string& where) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        fail("cuSOLVER failure at " + where + ": status=" + std::to_string(static_cast<int>(status)));
    }
}

std::vector<double> to_double_vector(const std::vector<float>& values) {
    return std::vector<double>(values.begin(), values.end());
}

SolveAttempt run_eigensolve_attempt(const std::string& label,
                                    bsmp::SchurOperator& schur,
                                    BlockSparseMatrix& mass,
                                    const bsmp::DeflatedPCGEigenSolverParameters& parameters,
                                    LinearPreconditioner* preconditioner) {
    const auto solve_start = std::chrono::high_resolution_clock::now();
    const bsmp::DeflatedPCGEigenResult result = bsmp::solve_deflated_pcg_eigenproblem(
        schur,
        mass,
        parameters,
        preconditioner);
    const auto solve_end = std::chrono::high_resolution_clock::now();

    SolveAttempt attempt;
    attempt.result = result;
    attempt.solve_ms = std::chrono::duration_cast<std::chrono::milliseconds>(solve_end - solve_start).count();
    attempt.worst_relative_residual = max_relative_residual(schur, mass, attempt.result);
    attempt.label = label;
    return attempt;
}

SolveAttempt run_explicit_schur_fallback_attempt(bsmp::SchurOperator& schur,
                                                 BlockSparseMatrix& mass,
                                                 const ExampleOptions& options) {
    const std::vector<double> dense_schur = to_double_vector(schur.explicitDenseSchur());
    const std::vector<double> dense_mass = to_double_vector(bsmp::HostBlockMatrix::fromMatrix(mass).toDense());
    const int n = mass.getConfig().num_rows;

    cusolverDnHandle_t handle = nullptr;
    double* d_a = nullptr;
    double* d_b = nullptr;
    double* d_w = nullptr;
    double* d_work = nullptr;
    int* d_info = nullptr;

    auto cleanup = [&]() {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_w);
        cudaFree(d_work);
        cudaFree(d_info);
        if (handle != nullptr) {
            cusolverDnDestroy(handle);
        }
    };

    const auto solve_start = std::chrono::high_resolution_clock::now();

    check_cusolver_status(cusolverDnCreate(&handle), "cusolverDnCreate");
    if (cudaMalloc(&d_a, n * n * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_b, n * n * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_w, n * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_info, sizeof(int)) != cudaSuccess) {
        cleanup();
        fail("Failed to allocate device buffers for explicit Schur cuSOLVER fallback");
    }

    if (cudaMemcpy(d_a, dense_schur.data(), n * n * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_b, dense_mass.data(), n * n * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
        cleanup();
        fail("Failed to upload dense Schur fallback matrices to the GPU");
    }

    int workspace_size = 0;
    check_cusolver_status(
        cusolverDnDsygvd_bufferSize(handle,
                                    CUSOLVER_EIG_TYPE_1,
                                    CUSOLVER_EIG_MODE_VECTOR,
                                    CUBLAS_FILL_MODE_UPPER,
                                    n,
                                    d_a,
                                    n,
                                    d_b,
                                    n,
                                    d_w,
                                    &workspace_size),
        "cusolverDnSsygvd_bufferSize");

    if (cudaMalloc(&d_work, workspace_size * sizeof(double)) != cudaSuccess) {
        cleanup();
        fail("Failed to allocate cuSOLVER workspace for explicit Schur fallback");
    }

    check_cusolver_status(
        cusolverDnDsygvd(handle,
                         CUSOLVER_EIG_TYPE_1,
                         CUSOLVER_EIG_MODE_VECTOR,
                         CUBLAS_FILL_MODE_UPPER,
                         n,
                         d_a,
                         n,
                         d_b,
                         n,
                         d_w,
                         d_work,
                         workspace_size,
                         d_info),
        "cusolverDnSsygvd");

    int info = 0;
    if (cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        cleanup();
        fail("Failed to download cuSOLVER info flag for explicit Schur fallback");
    }

    std::vector<double> eigenvalues(n, 0.0);
    std::vector<double> eigenvectors_column_major(n * n, 0.0);
    if (cudaMemcpy(eigenvalues.data(), d_w, n * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(eigenvectors_column_major.data(), d_a, n * n * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
        cleanup();
        fail("Failed to download explicit Schur cuSOLVER results");
    }

    const auto solve_end = std::chrono::high_resolution_clock::now();
    cleanup();

    SolveAttempt attempt;
    attempt.label = "explicit-schur-cusolver";
    attempt.solve_ms = std::chrono::duration_cast<std::chrono::milliseconds>(solve_end - solve_start).count();
    attempt.result.converged = (info == 0);
    attempt.result.total_iterations = 0;
    const int eigenpairs_to_keep = std::min(options.modes, n);
    attempt.result.eigenpairs.reserve(eigenpairs_to_keep);

    for (int i = 0; i < eigenpairs_to_keep; ++i) {
        auto evaluate_candidate = [&](const std::vector<float>& candidate_vector) {
            bsmp::DeflatedPCGEigenpair candidate;
            candidate.eigenvalue = static_cast<float>(eigenvalues[i]);
            candidate.eigenvector = candidate_vector;
            candidate.iterations = 0;

            std::vector<double> sx(n, 0.0);
            std::vector<double> mx(n, 0.0);
            for (int row = 0; row < n; ++row) {
                const double x_row = static_cast<double>(candidate.eigenvector[row]);
                for (int col = 0; col < n; ++col) {
                    sx[row] += dense_schur[row * n + col] * static_cast<double>(candidate.eigenvector[col]);
                    mx[row] += dense_mass[row * n + col] * static_cast<double>(candidate.eigenvector[col]);
                }
            }

            double diff_sq = 0.0;
            double sx_sq = 0.0;
            for (int j = 0; j < n; ++j) {
                const double diff = sx[j] - static_cast<double>(candidate.eigenvalue) * mx[j];
                diff_sq += diff * diff;
                sx_sq += sx[j] * sx[j];
            }
            candidate.residual_norm = static_cast<float>(std::sqrt(diff_sq));
            const float relative_residual = candidate.residual_norm / std::max(static_cast<float>(std::sqrt(sx_sq)), 1.0f);
            candidate.converged = relative_residual <= std::max(options.tol, 1e-4f);
            return std::make_pair(candidate, relative_residual);
        };

        std::vector<float> column_major_vector(n, 0.0f);
        std::vector<float> row_major_vector(n, 0.0f);
        for (int row = 0; row < n; ++row) {
            column_major_vector[row] = static_cast<float>(eigenvectors_column_major[row + i * n]);
            row_major_vector[row] = static_cast<float>(eigenvectors_column_major[i + row * n]);
        }

        auto best = evaluate_candidate(column_major_vector);
        auto alternate = evaluate_candidate(row_major_vector);
        if (alternate.second < best.second) {
            best = std::move(alternate);
        }

        bsmp::DeflatedPCGEigenpair eigenpair;
        eigenpair = std::move(best.first);

        attempt.result.eigenpairs.push_back(std::move(eigenpair));
    }

    attempt.result.converged = attempt.result.converged && !attempt.result.eigenpairs.empty();
    attempt.worst_relative_residual = 0.0f;
    for (const auto& eigenpair : attempt.result.eigenpairs) {
        attempt.result.converged = attempt.result.converged && eigenpair.converged;
        const float eigen_scale = std::max(std::fabs(eigenpair.eigenvalue), 1.0f);
        const float dense_relative_residual = eigenpair.residual_norm / eigen_scale;
        attempt.worst_relative_residual = std::max(attempt.worst_relative_residual, dense_relative_residual);
    }
    return attempt;
}

bool has_positive_eigenvalue_estimate(const bsmp::DeflatedPCGEigenResult& result) {
    for (const auto& eigenpair : result.eigenpairs) {
        if (eigenpair.eigenvalue > 0.0f) {
            return true;
        }
    }
    return false;
}

bool is_iterative_attempt_acceptable(const SolveAttempt& attempt,
                                     int requested_modes,
                                     float tolerance) {
    if (attempt.result.eigenpairs.size() != static_cast<size_t>(requested_modes)) {
        return false;
    }
    if (!has_positive_eigenvalue_estimate(attempt.result)) {
        return false;
    }
    if (attempt.result.converged) {
        return true;
    }

    const float explicit_residual_slack = std::max(2.0f * tolerance, 2e-3f);
    return std::isfinite(attempt.worst_relative_residual) &&
           attempt.worst_relative_residual <= explicit_residual_slack;
}

SolveAttempt run_iterative_attempt_with_retries(const std::string& label,
                                                bsmp::SchurOperator& schur,
                                                BlockSparseMatrix& mass,
                                                const bsmp::DeflatedPCGEigenSolverParameters& parameters,
                                                LinearPreconditioner* preconditioner,
                                                int requested_modes,
                                                float tolerance,
                                                int max_attempts) {
    SolveAttempt best_attempt;
    bool has_best_attempt = false;

    for (int attempt_index = 0; attempt_index < std::max(1, max_attempts); ++attempt_index) {
        bsmp::DeflatedPCGEigenSolverParameters seeded_parameters = parameters;
        seeded_parameters.random_seed = parameters.random_seed + 7919u * static_cast<unsigned int>(attempt_index);

        SolveAttempt current_attempt = run_eigensolve_attempt(label,
                                                              schur,
                                                              mass,
                                                              seeded_parameters,
                                                              preconditioner);

        const bool current_ok = is_iterative_attempt_acceptable(current_attempt,
                                                                requested_modes,
                                                                tolerance);
        const bool best_ok = has_best_attempt && is_iterative_attempt_acceptable(best_attempt,
                                                                                 requested_modes,
                                                                                 tolerance);

        if (!has_best_attempt ||
            (current_ok && !best_ok) ||
            (!current_ok && !best_ok &&
             (current_attempt.result.eigenpairs.size() > best_attempt.result.eigenpairs.size() ||
              (current_attempt.result.eigenpairs.size() == best_attempt.result.eigenpairs.size() &&
               current_attempt.worst_relative_residual < best_attempt.worst_relative_residual)))) {
            best_attempt = std::move(current_attempt);
            has_best_attempt = true;
        }

        if (current_ok) {
            return has_best_attempt ? best_attempt : current_attempt;
        }

        if (attempt_index + 1 < std::max(1, max_attempts)) {
            std::cout << "Iterative eigensolve attempt '" << label
                      << "' with seed " << seeded_parameters.random_seed
                      << " did not converge cleanly; retrying with a different initial guess."
                      << std::endl;
        }
    }

    return best_attempt;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::cout << std::fixed << std::setprecision(6);
        const ExampleOptions options = parse_options(argc, argv);

        std::vector<float> coordinates;
        if (!options.synthetic) {
            coordinates = load_coordinates_file(options.coords_path);
        }
        bsmp::PiezoBlockSystem system = options.synthetic
            ? build_synthetic_system(coordinates)
            : (has_full_matrix_inputs(options)
                   ? load_system_from_full_matrices(options, coordinates)
                   : load_system_from_files(options));

        std::cout << "=== SA-AMG + deflated PCG eigen example ===" << std::endl;
        std::cout << "mode: " << (options.synthetic ? "synthetic" : "file-backed") << std::endl;
        std::cout << "ordering label: " << options.ordering
                  << (has_full_matrix_inputs(options) ? " (used for full-matrix block extraction)" : " (legacy block input mode)")
                  << std::endl;
        std::cout << "mechanical dofs: " << system.mechanicalDofs() << ", electrical dofs: " << system.electricalDofs() << std::endl;
        if (has_full_matrix_inputs(options)) {
            std::cout << "dielectric sign mode: " << options.dielectric_sign << std::endl;
        }
        std::cout << "scaling mode: " << options.scaling_mode << std::endl;
        std::cout << "eigensolve path request: " << options.eigensolve_path << std::endl;

        const int expected_nodes = system.mechanicalDofs() / 3;
        if (system.mechanicalDofs() % 3 != 0) {
            fail("Mechanical DOFs must be divisible by 3 for rigid-body mode construction.");
        }
        if (static_cast<int>(coordinates.size()) != expected_nodes * 3) {
            fail("Coordinate count does not match mechanical DOFs / 3.");
        }
        if (options.grounded_dof < 0 || options.grounded_dof >= system.electricalDofs()) {
            fail("Grounded electrical DOF is out of range.");
        }

        if (!bsmp::apply_grounding_constraint(system.dielectric(), options.grounded_dof)) {
            fail("Failed to apply grounding constraint to C_phi.");
        }

        const auto scaling = build_selected_scaling(system, options);
        bsmp::PiezoBlockSystem scaled = bsmp::scale_piezo_system(system, scaling);
        bsmp::SchurOperator schur(scaled);

        const bsmp::NodeLayout layout = bsmp::NodeLayout::fromNodeCoordinates(
            coordinates,
            system.electricalDofs() == expected_nodes);
        if (!layout.isValid()) {
            fail("Constructed node layout is invalid.");
        }
        const auto rbm = bsmp::build_rigid_body_modes(layout);

        auto preconditioner = bsmp::createSAAMGPreconditioner(scaled.mechanicalStiffness(),
                                                              layout,
                                                              rbm,
                                                              options.sa_amg);
        if (!preconditioner) {
            fail("Failed to construct SA-AMG preconditioner for the article eigen example.");
        }

        SolveAttempt attempt;
        if (options.eigensolve_path == "sa-amg") {
            attempt = run_iterative_attempt_with_retries("sa-amg",
                                                         schur,
                                                         scaled.mass(),
                                                         options.eigen,
                                                         preconditioner.get(),
                                                         options.modes,
                                                         options.tol,
                                                         4);
        } else if (options.eigensolve_path == "unpreconditioned") {
            attempt = run_eigensolve_attempt("unpreconditioned", schur, scaled.mass(), options.eigen, nullptr);
        } else if (options.eigensolve_path == "explicit-schur-cusolver") {
            attempt = run_explicit_schur_fallback_attempt(schur, scaled.mass(), options);
        } else {
            attempt = run_iterative_attempt_with_retries("sa-amg",
                                                         schur,
                                                         scaled.mass(),
                                                         options.eigen,
                                                         preconditioner.get(),
                                                         options.modes,
                                                         options.tol,
                                                         4);
            if (!is_iterative_attempt_acceptable(attempt, options.modes, options.tol)) {
                std::cout << "Primary eigensolve attempt (SA-AMG preconditioned) did not converge cleanly; retrying without a preconditioner." << std::endl;
                SolveAttempt fallback_attempt = run_eigensolve_attempt("unpreconditioned", schur, scaled.mass(), options.eigen, nullptr);
                if (is_iterative_attempt_acceptable(fallback_attempt, options.modes, options.tol) ||
                    fallback_attempt.worst_relative_residual < attempt.worst_relative_residual) {
                    attempt = std::move(fallback_attempt);
                }
            }
            if (!is_iterative_attempt_acceptable(attempt, options.modes, options.tol)) {
                std::cout << "Iterative Schur eigensolve still looks unhealthy; retrying with an explicit Schur fallback." << std::endl;
                SolveAttempt explicit_fallback = run_explicit_schur_fallback_attempt(schur, scaled.mass(), options);
                if ((!has_positive_eigenvalue_estimate(attempt.result) && has_positive_eigenvalue_estimate(explicit_fallback.result)) ||
                    explicit_fallback.result.converged ||
                    explicit_fallback.worst_relative_residual < attempt.worst_relative_residual) {
                    attempt = std::move(explicit_fallback);
                }
            }
        }
        const bsmp::DeflatedPCGEigenResult& result = attempt.result;

        std::cout << "SA-AMG regularization alpha = " << preconditioner->alpha() << std::endl;
        std::cout << "eigensolve path: " << attempt.label << std::endl;
        std::cout << "requested modes: " << options.modes << std::endl;
        std::cout << "converged: " << (result.converged ? "yes" : "no") << std::endl;
        std::cout << "total iterations: " << result.total_iterations << std::endl;
        std::cout << "solve time [ms]: " << attempt.solve_ms << std::endl;

        for (size_t i = 0; i < result.eigenpairs.size(); ++i) {
            const auto& eigenpair = result.eigenpairs[i];
            const float frequency_hz = eigenvalue_to_frequency_hz(eigenpair.eigenvalue);
            std::cout << "mode " << i
                      << ": lambda=" << eigenpair.eigenvalue
                      << ", frequency_hz=";
            if (frequency_hz > 0.0f) {
                std::cout << frequency_hz;
            } else {
                std::cout << "n/a";
            }
            std::cout
                      << ", iterations=" << eigenpair.iterations
                      << ", residual=" << eigenpair.residual_norm
                      << ", converged=" << (eigenpair.converged ? "yes" : "no")
                      << std::endl;
        }

        const float worst_relative_residual = attempt.worst_relative_residual;
        std::cout << "worst explicit relative eigen residual = " << worst_relative_residual << std::endl;

        if (!result.converged) {
            std::cerr << "Example finished, but not all requested modes converged to the requested tolerance." << std::endl;
            return 2;
        }

        std::cout << "Example completed successfully." << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "example_sa_amg_pcg_eigen failed: " << error.what() << std::endl;
        return 1;
    }
}