#include "record/record_builder.h"
#include <fstream>

namespace bsmp {

namespace experiment {

void Record::schema_version(std::string_view version) {
    data["schema_version"] = version;
}
void Record::hardware(std::string_view gpu_model, std::string_view cuda_version, std::string_view compute_capability) {
    data["hardware"]["gpu_model"] = gpu_model;
    data["hardware"]["cuda_version"] = cuda_version;
    data["hardware"]["compute_capability"] = compute_capability;
}

void Record::type(std::string_view type) {
    data["problem"]["type"] = type;
}

void Record::matrix(std::string_view name, uint32_t rows, uint32_t columns, uint32_t nnz) {
    data["problem"]["matrix"]["name"] = name;
    data["problem"]["matrix"]["rows"] = rows;
    data["problem"]["matrix"]["columns"] = columns;
    data["problem"]["matrix"]["nnz"] = nnz;
}

void Record::rhs(uint32_t size, bool from_task) {
    data["problem"]["rhs"]["size"] = size;
    data["problem"]["rhs"]["from_task"] = from_task;
}

void Record::run(bsmp::experiment::Run r) {
    data["runs"].push_back(r.get());
}

json Record::get() const {
    return data;
}

void Record::write(fs::path path) const {
    std::ofstream ofs(path);
    ofs << data;
}

void Run::solver(std::string_view name) {
    data["parameters"]["solver"] = name;
}

void Run::preconditioner(std::string_view name) {
    data["parameters"]["preconditioner"] = name;
}

void Run::max_iters(uint32_t value) {
    data["parameters"]["max_iters"] = value;
}

void Run::tolerance(float value) {
    data["parameters"]["tolerance"] = value;
}

void Run::time(float value) {
    data["metrics"]["time"] = value;
}

void Run::residual(float value) {
    data["metrics"]["residual"] = value;
}

void Run::write(fs::path path) const {
    std::ofstream ofs(path);
    ofs << data;
}

json Run::get() const {
    return data;
}


}  // namespace experiment
}  // namespace bsmp
