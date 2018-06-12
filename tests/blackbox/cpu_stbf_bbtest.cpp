
#include <algorithm>

#pragma warning(push, 0)
#include <gtest/gtest.h>
#pragma warning(pop)

#include "queries/weighted.hpp"
#include "export_import.hpp"
#include "common/time.hpp"

struct InputParameters {
    size_t num_points_p_;
    size_t num_points_q_;
    size_t top_k_;
public:
    InputParameters() {}

    InputParameters(size_t num_points_p, size_t num_points_q, size_t top_k) :
        num_points_p_(num_points_p), num_points_q_(num_points_q), top_k_(top_k) {
    }
};

using namespace sl::queries;

class InputInitializer : public ::testing::TestWithParam<InputParameters> {
public:
    InputInitializer() {
    }

    virtual void SetUp() {
        input_parameters_ = GetParam();

        data::UniformRealRandomGenerator rrg_x(0., 1.);
        data::UniformRealRandomGenerator rrg_y(0., 1.);
        data::UniformIntRandomGenerator irg(1, 10);

        wq.SetTopK(input_parameters_.top_k_);
        wq.InitRandom(input_parameters_.num_points_p_, input_parameters_.num_points_q_, rrg_x, rrg_y, irg);
        //wq.ToFile("test.json");
        //wq.FromFile("test.json");
    }

    virtual void TearDown() {

    }

protected:
    InputParameters input_parameters_;
    sl::queries::WeightedQuery wq;
};

class InputInitializerSmall : public InputInitializer { };
class InputInitializerBig : public InputInitializer { };

bool CheckOuput(sl::queries::NonConstData<sl::queries::data::WeightedPoint> &a, sl::queries::NonConstData<sl::queries::data::WeightedPoint> &b, int line) { // we pass a copy
    auto sorting_function = [](const sl::queries::data::WeightedPoint &a, const sl::queries::data::WeightedPoint &b) -> bool {
        if (a.point_.x_ == b.point_.x_) return a.point_.y_ < b.point_.y_;
        else return a.point_.x_ < b.point_.x_;
    };

    std::sort(a.Points().begin(), a.Points().end(), sorting_function);
    std::sort(b.Points().begin(), b.Points().end(), sorting_function);

    EXPECT_EQ(a.Points().size(), b.Points().size()) << " Line: " << line;
    if (a.Points().size() != b.Points().size()) {
        return false;
    }
    bool equal = std::equal(a.Points().begin(), a.Points().end(), b.Points().begin());
    EXPECT_TRUE(equal) << " Line: " << line;

    return equal;
}

long long RunAlgorithm(
    sl::queries::WeightedQuery &wq,
    sl::queries::WeightedQuery::AlgorithmType alg_type,
    sl::queries::NonConstData<sl::queries::data::WeightedPoint> *output,
    sl::queries::algorithms::DistanceType distance_type) {

    long long time_taken = sl::time::measure<>::execution([&wq, &alg_type, &distance_type]() {
        wq.RunAlgorithm(alg_type, distance_type);
    });

    *output = wq.GetOuputCopy(); // we do a copy
    wq.ClearOutput();

    std::cout << "\nTime(ms): " << time_taken << ". Input size: " << wq.GetInputP().GetPoints().size() << ". Ouput size: " << output->GetPoints().size();
    return time_taken;
}

void RunAlgorithmAndCompareWithPrevious(
    sl::queries::WeightedQuery &wq,
    sl::queries::WeightedQuery::AlgorithmType alg_type,
    sl::queries::NonConstData<sl::queries::data::WeightedPoint> *output,
    sl::queries::NonConstData<sl::queries::data::WeightedPoint> &previous_ouput,
    sl::queries::algorithms::DistanceType distance_type,
    int line = __LINE__) {

    long long time_taken = RunAlgorithm(wq, alg_type, output, distance_type);

    bool equal = CheckOuput(previous_ouput, *output, line);

    std::cout << (equal ? " EQUAL" : " DIFFERENT\n");
}

void RunAll(sl::queries::WeightedQuery &wq, sl::queries::algorithms::DistanceType distance_type) {
    sl::queries::NonConstData<sl::queries::data::WeightedPoint> stbf_output;
    RunAlgorithm(wq, sl::queries::WeightedQuery::AlgorithmType::SINGLE_THREAD_BRUTE_FORCE, &stbf_output, distance_type);

    sl::queries::NonConstData<sl::queries::data::WeightedPoint> stbfd_output;
    RunAlgorithmAndCompareWithPrevious(wq, sl::queries::WeightedQuery::AlgorithmType::SINGLE_THREAD_BRUTE_FORCE_DISCARDING, &stbfd_output, stbf_output, distance_type);

    sl::queries::NonConstData<sl::queries::data::WeightedPoint> mtbf_output;
    RunAlgorithmAndCompareWithPrevious(wq, sl::queries::WeightedQuery::AlgorithmType::MULTI_THREAD_BRUTE_FORCE, &mtbf_output, stbf_output, distance_type);

    sl::queries::NonConstData<sl::queries::data::WeightedPoint> mtbfd_output;
    RunAlgorithmAndCompareWithPrevious(wq, sl::queries::WeightedQuery::AlgorithmType::MULTI_THREAD_BRUTE_FORCE_DISCARDING, &mtbfd_output, stbf_output, distance_type);

    sl::queries::NonConstData<sl::queries::data::WeightedPoint> sts_output;
    RunAlgorithmAndCompareWithPrevious(wq, sl::queries::WeightedQuery::AlgorithmType::SINGLE_THREAD_SORTING, &sts_output, stbf_output, distance_type);

    sl::queries::NonConstData<sl::queries::data::WeightedPoint> mts_output;
    RunAlgorithmAndCompareWithPrevious(wq, sl::queries::WeightedQuery::AlgorithmType::MULTI_THREAD_SORTING, &mts_output, stbf_output, distance_type);

    sl::queries::NonConstData<sl::queries::data::WeightedPoint> gpubf_output;
    RunAlgorithmAndCompareWithPrevious(wq, sl::queries::WeightedQuery::AlgorithmType::GPU_BRUTE_FORCE, &gpubf_output, stbf_output, distance_type);

    sl::queries::NonConstData<sl::queries::data::WeightedPoint> gpubfd_output;
    RunAlgorithmAndCompareWithPrevious(wq, sl::queries::WeightedQuery::AlgorithmType::GPU_BRUTE_FORCE_DISCARTING, &gpubfd_output, stbf_output, distance_type);

    std::cout << '\n';
}

TEST_P(InputInitializerSmall, TestOutputCorrectness) {
    RunAll(wq, sl::queries::algorithms::DistanceType::Nearest);
    //RunAll(wq, sl::queries::algorithms::DistanceType::Furthest);
}

INSTANTIATE_TEST_CASE_P(InstantiationName, InputInitializerSmall, ::testing::Values(
    //InputParameters(0, 0, 0),
    //InputParameters(0, 1, 0),
    //InputParameters(1, 0, 1),

    //InputParameters(1, 1, 1),
    //InputParameters(10, 10, 10),
    //InputParameters(32, 10, 32),
    //InputParameters(33, 10, 33),
    //InputParameters(64, 10, 64),
    //InputParameters(65, 10, 65),
    //InputParameters(100, 10, 100),
    //InputParameters(128, 10, 128),
    //InputParameters(1000, 10, 1000),
    //InputParameters(1024, 10, 1024),
    //InputParameters(1025, 10, 1025),
    //InputParameters(2048, 10, 2048),
    //InputParameters(2049, 10, 2049),

    //InputParameters(1, 10, 1),
    //InputParameters(10, 100, 10),
    //InputParameters(32, 100, 32),
    //InputParameters(33, 100, 33),
    //InputParameters(64, 100, 64),
    //InputParameters(65, 100, 65),
    //InputParameters(100, 1000, 100),
    InputParameters(128, 1000, 128)
    //InputParameters(1000, 1000, 1000),
    //InputParameters(1024, 2000, 1024),
    //InputParameters(1025, 3000, 1025),
    //InputParameters(2048, 4000, 2048),
    //InputParameters(2049, 8192, 2049),

    //InputParameters(21, 5, 4),
    //InputParameters(10, 100, 5),
    //InputParameters(32, 10, 5),
    //InputParameters(33, 100, 10),
    //InputParameters(64, 100, 10),
    //InputParameters(65, 100, 10),
    //InputParameters(100, 1000, 20),
    //InputParameters(128, 1000, 20),
    //InputParameters(1000, 1000, 30),
    //InputParameters(1024, 2000, 30),
    //InputParameters(1025, 3000, 30),
    //InputParameters(2048, 4000, 40),
    //InputParameters(2049, 8192, 50)
));

//INSTANTIATE_TEST_CASE_P(InstantiationName, InputInitializerBig, ::testing::Values(
//    InputParameters(10000, 10),
//    InputParameters(20000, 10),
//    InputParameters(100000, 10)
//));


//TEST_P(InputInitializer, TestOutputCorrectness) {
//    sl::queries::NonConstData<sl::queries::data::WeightedPoint> output;
//    RunAlgorithm(wq, sl::queries::WeightedQuery::AlgorithmType::GPU_BRUTE_FORCE, &output);
//}
//
//INSTANTIATE_TEST_CASE_P(InstantiationName2, InputInitializer, ::testing::Values(
//    InputParameters(131072, 10)
//));