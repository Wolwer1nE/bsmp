#include <cstdint>
#include <string_view>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace bsmp {

namespace experiment {

namespace fs = std::filesystem;
using json = nlohmann::json;

class Run {
   public:
    void solver(std::string_view name);
    void preconditioner(std::string_view name);
    void max_iters(uint32_t value);
    void tolerance(float value);
    void time(float value);
    void residual(float value);

    json get() const;

    void write(fs::path) const;

   private:
    json data;
};

class Record {
   public:
    void schema_version(std::string_view version);
    void hardware(std::string_view gpu_model, std::string_view cuda_version, std::string_view compute_capability);
    void type(std::string_view type);
    void matrix(std::string_view name,
                uint32_t rows,
                uint32_t columns,
                uint32_t nnz);
    void rhs(uint32_t size,
             bool from_task);
    void run(bsmp::experiment::Run);

    void write(fs::path path) const;
    json get() const;

   private:
    json data;
};

}  // namespace experiment
}  // namespace bsmp
