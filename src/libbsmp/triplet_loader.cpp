#include "libbsmp/triplet_loader.h"

#include <cctype>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <unordered_map>

struct PairHash {
    size_t operator()(const std::pair<int, int>& p) const noexcept {
        return (static_cast<size_t>(p.first) << 32) ^ static_cast<size_t>(p.second);
    }
};

bool load_triplet_file_as_block_sparse(const std::string& path,
                                       int block_size,
                                       BlockSparseMatrixConfig& config_out,
                                       std::vector<int>& block_rows_out,
                                       std::vector<int>& block_cols_out,
                                       std::vector<float>& block_data_out) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Failed to open triplet file: " << path << std::endl;
        return false;
    }

    int max_row = -1, max_col = -1;

    std::map<std::pair<int, int>, std::vector<float>> blocks;

    std::string line;
    while (std::getline(in, line)) {
        // trim leading spaces
        size_t p = line.find_first_not_of(" \t\r\n");
        if (p == std::string::npos)
            continue;
        if (line[p] == '#')
            continue;

        // Whitespace or comma separated values
        long long i, j;
        double v;
        char delim1 = 0, delim2 = 0;
        std::istringstream ss(line);
        if ((ss >> i >> delim1 >> j >> delim2 >> v) &&
            ((delim1 == ',' && delim2 == ',') || (delim1 == ',' && delim2 == ' ') || (delim1 == ' ' && delim2 == ' ') || (delim1 == ' ' && delim2 == ','))) {
            // ok
        } else {
            // fallback: try space-separated
            ss.clear();
            ss.str(line);
            if (!(ss >> i >> j >> v)) {
                std::cerr << "Warning: could not parse line: '" << line << "'" << std::endl;
                continue;
            }
        }

        if (i < 0 || j < 0) {
            std::cerr << "Warning: negative indices are ignored: " << i << "," << j << std::endl;
            continue;
        }

        max_row = std::max<int>(max_row, static_cast<int>(i));
        max_col = std::max<int>(max_col, static_cast<int>(j));

        int brow = static_cast<int>(i) / block_size;
        int bcol = static_cast<int>(j) / block_size;
        int local_i = static_cast<int>(i) % block_size;
        int local_j = static_cast<int>(j) % block_size;

        auto key = std::make_pair(brow, bcol);
        auto it = blocks.find(key);
        if (it == blocks.end()) {
            std::vector<float> data(block_size * block_size, 0.0f);
            data[local_i * block_size + local_j] = static_cast<float>(v);
            blocks.emplace(key, std::move(data));
        } else {
            it->second[local_i * block_size + local_j] += static_cast<float>(v);
        }
    }

    if (max_row < 0 || max_col < 0) {
        config_out.num_rows = 0;
        config_out.num_cols = 0;
        config_out.block_size = block_size;
        config_out.num_nonzero_blocks = 0;
        block_rows_out.clear();
        block_cols_out.clear();
        block_data_out.clear();
        return true;
    }

    // fill outputs
    config_out.num_rows = (max_row + 1);
    config_out.num_cols = (max_col + 1);
    config_out.block_size = bsmp::kBlockSize;
    config_out.num_nonzero_blocks = static_cast<int>(blocks.size());

    block_rows_out.reserve(blocks.size());
    block_cols_out.reserve(blocks.size());
    block_data_out.reserve(blocks.size() * block_size * block_size);

    for (const auto& kv : blocks) {
        block_rows_out.push_back(kv.first.first);
        block_cols_out.push_back(kv.first.second);
        const std::vector<float>& data = kv.second;
        block_data_out.insert(block_data_out.end(), data.begin(), data.end());
    }

    return true;
}

BlockSparseMatrix* load_triplet_file_to_matrix(const std::string& path,
                                               int block_size) {
    BlockSparseMatrixConfig cfg;
    std::vector<int> brow, bcol;
    std::vector<float> bdata;
    if (!load_triplet_file_as_block_sparse(path, block_size, cfg, brow, bcol, bdata)) {
        return nullptr;
    }

    // override to compile-time constant to avoid mismatch
    cfg.block_size = bsmp::kBlockSize;
    BlockSparseMatrix* m = new BlockSparseMatrix(cfg);
    m->initialize(brow, bcol, bdata);
    return m;
}

static inline std::string to_lower(std::string s) {
    for (char& c : s)
        c = (char)std::tolower((unsigned char)c);
    return s;
}

bool load_matrix_market_as_block_sparse(const std::string& path,
                                        int block_size,
                                        BlockSparseMatrixConfig& config_out,
                                        std::vector<int>& block_rows_out,
                                        std::vector<int>& block_cols_out,
                                        std::vector<float>& block_data_out) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Failed to open Matrix Market file: " << path << std::endl;
        return false;
    }

    std::string header;
    if (!std::getline(in, header)) {
        std::cerr << "Empty Matrix Market file: " << path << std::endl;
        return false;
    }
    if (header.rfind("%%MatrixMarket", 0) != 0) {
        std::cerr << "Invalid Matrix Market header" << std::endl;
        return false;
    }

    // tokenize header
    std::istringstream hs(header);
    std::string mm, obj, fmt, field, symm;
    hs >> mm >> obj >> fmt >> field >> symm;
    obj = to_lower(obj);
    fmt = to_lower(fmt);
    field = to_lower(field);
    symm = to_lower(symm);
    if (!(obj == "matrix" && fmt == "coordinate" && (field == "real" || field == "double"))) {
        std::cerr << "Unsupported Matrix Market (need coordinate real)" << std::endl;
        return false;
    }
    bool symmetric = (symm == "symmetric");

    // skip comments
    std::string line;
    do {
        if (!std::getline(in, line)) {
            std::cerr << "Unexpected EOF before size line" << std::endl;
            return false;
        }
    } while (!line.empty() && line[0] == '%');

    // size line: m n nnz
    std::istringstream ssz(line);
    long long m, n, nnz;
    ssz >> m >> n >> nnz;
    if (!ssz) {
        std::cerr << "Failed to parse size line" << std::endl;
        return false;
    }

    std::map<std::pair<int, int>, std::vector<float>> blocks;
    for (long long k = 0; k < nnz; ++k) {
        long long ii, jj;
        double vv;
        if (!(in >> ii >> jj >> vv)) {
            std::cerr << "Failed to read entry #" << k << std::endl;
            return false;
        }
        // Matrix Market is 1-based
        int i = (int)ii - 1;
        int j = (int)jj - 1;
        auto put = [&](int r, int c, double val) {
            int brow = r / block_size, bcol = c / block_size;
            int li = r % block_size, lj = c % block_size;
            auto key = std::make_pair(brow, bcol);
            auto it = blocks.find(key);
            if (it == blocks.end()) {
                std::vector<float> data(block_size * block_size, 0.0f);
                data[li * block_size + lj] = (float)val;
                blocks.emplace(key, std::move(data));
            } else {
                it->second[li * block_size + lj] += (float)val;
            }
        };
        put(i, j, vv);
        if (symmetric && i != j)
            put(j, i, vv);
    }

    config_out.num_rows = (int)m;
    config_out.num_cols = (int)n;
    config_out.block_size = bsmp::kBlockSize;
    config_out.num_nonzero_blocks = (int)blocks.size();

    block_rows_out.reserve(blocks.size());
    block_cols_out.reserve(blocks.size());
    block_data_out.reserve(blocks.size() * block_size * block_size);
    for (const auto& kv : blocks) {
        block_rows_out.push_back(kv.first.first);
        block_cols_out.push_back(kv.first.second);
        const auto& data = kv.second;
        block_data_out.insert(block_data_out.end(), data.begin(), data.end());
    }

    return true;
}

BlockSparseMatrix* load_matrix_market_to_matrix(const std::string& path, int block_size) {
    BlockSparseMatrixConfig cfg;
    std::vector<int> br, bc;
    std::vector<float> bd;
    if (!load_matrix_market_as_block_sparse(path, block_size, cfg, br, bc, bd))
        return nullptr;
    cfg.block_size = bsmp::kBlockSize;
    BlockSparseMatrix* m = new BlockSparseMatrix(cfg);
    m->initialize(br, bc, bd);
    return m;
}
