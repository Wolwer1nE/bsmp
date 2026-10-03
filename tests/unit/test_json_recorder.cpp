#include <filesystem>
#include <fstream>
#include <nlohmann/json-schema.hpp>

#include "gtest/gtest.h"
#include "record/record_builder.h"

using nlohmann::json;
using nlohmann::json_schema::json_validator;
namespace fs = std::filesystem;

class JsonRecorderTest : public ::testing::Test {
   protected:
    static inline nlohmann::json schema_json;
    static inline fs::path tmp;

    static void SetUpTestSuite() {
        tmp = fs::current_path() / "tmp";
        fs::create_directories(tmp);
        std::ifstream ifs("data/schemas/v1/record.schema.json");
        ASSERT_TRUE(ifs) << "could not open schema file";
        schema_json = nlohmann::json::parse(ifs);
    }

    static void TearDownTestSuite() {
        fs::remove_all(tmp);
    }

    static void validate(const nlohmann::json& doc) {
        nlohmann::json_schema::json_validator validator;
        validator.set_root_schema(schema_json);
        EXPECT_NO_THROW(validator.validate(doc));
    }

    static bsmp::experiment::Record make_record() {
        using namespace bsmp::experiment;
        Record record;
        record.schema_version("1.0");
        record.hardware("NVIDIA GeForce RTX 3090", "11.1", "8.6");
        record.type("spmv");
        record.matrix("matrix1", 1000, 1000, 5000);
        record.rhs(1000, true);
        bsmp::experiment::Run run;
        run.max_iters(100);
        run.preconditioner("jacobi");
        run.solver("cg");
        run.residual(1e-6);
        run.time(0.123);
        record.run(run);
        return record;
    }
};

TEST_F(JsonRecorderTest, ShouldEncodeCorrectExperimentData) {
    auto record = make_record();
    validate(record.get());
}

TEST_F(JsonRecorderTest, ShouldSupportMultipleRuns) {
    auto record = make_record();
    // Let's add another run
    bsmp::experiment::Run second_run;
    second_run.max_iters(200);
    second_run.preconditioner("ilu");
    second_run.solver("gmres");
    second_run.residual(1e-8);
    second_run.time(0.456);

    record.run(second_run);
    // And another run
    record.run(second_run);
    // And another run
    record.run(second_run);

    auto j = record.get();
    validate(j);
    EXPECT_EQ(j["runs"].size(), 4);
}

TEST_F(JsonRecorderTest, ShouldWriteJsonFile) {
    auto record = make_record();
    fs::path path = tmp / "test_record.json";
    record.write(path);
    std::ifstream ifs(path);
    auto read = nlohmann::json::parse(ifs);
    EXPECT_EQ(read, record.get());
}
