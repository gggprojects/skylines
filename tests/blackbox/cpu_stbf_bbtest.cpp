
#include <algorithm>

#pragma warning(push, 0)
#include <gtest/gtest.h>
#pragma warning(pop)

#include "queries/weighted.hpp"
#include "export_import.hpp"
#include "time_utils.hpp"

struct InputParameters {
    int num_points_p_;
    int num_points_q_;
public:
    InputParameters() {}

    InputParameters(int num_points_p, int num_points_q) :
        num_points_p_(num_points_p), num_points_q_(num_points_q) {
    }
};

class InputInitializer : public ::testing::TestWithParam<InputParameters> {
public:
    InputInitializer() {
    }

    virtual void SetUp() {
        input_parameters_ = GetParam();
        wq.InitRandom(input_parameters_.num_points_p_, input_parameters_.num_points_q_);
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
        if (a.point_.x_ < b.point_.x_) return true;
        else {
            if(a.point_.x_ == b.point_.x_) return a.point_.y_ < b.point_.y_;
            else return false;
        }
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

    *output = wq.GetOuput(); // we do a copy
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
    RunAlgorithmAndCompareWithPrevious(wq, sl::queries::WeightedQuery::AlgorithmType::SINGLE_THREAD_BRUTE_FORCE_DISCARTING, &stbfd_output, stbf_output, distance_type);

    sl::queries::NonConstData<sl::queries::data::WeightedPoint> sts_output;
    RunAlgorithmAndCompareWithPrevious(wq, sl::queries::WeightedQuery::AlgorithmType::SINGLE_THREAD_SORTING, &sts_output, stbfd_output, distance_type);

    sl::queries::NonConstData<sl::queries::data::WeightedPoint> mtbf_output;
    RunAlgorithmAndCompareWithPrevious(wq, sl::queries::WeightedQuery::AlgorithmType::MULTI_THREAD_BRUTE_FORCE, &mtbf_output, sts_output, distance_type);

    sl::queries::NonConstData<sl::queries::data::WeightedPoint> mts_output;
    RunAlgorithmAndCompareWithPrevious(wq, sl::queries::WeightedQuery::AlgorithmType::MULTI_THREAD_SORTING, &mts_output, mtbf_output, distance_type);

    sl::queries::NonConstData<sl::queries::data::WeightedPoint> gpubf_output;
    RunAlgorithmAndCompareWithPrevious(wq, sl::queries::WeightedQuery::AlgorithmType::GPU_BRUTE_FORCE, &gpubf_output, mts_output, distance_type);
    std::cout << '\n';
}

TEST_P(InputInitializerSmall, TestOutputCorrectness) {
    RunAll(wq, sl::queries::algorithms::DistanceType::Neartest);
    RunAll(wq, sl::queries::algorithms::DistanceType::Furthest);
}

INSTANTIATE_TEST_CASE_P(InstantiationName, InputInitializerSmall, ::testing::Values(
    InputParameters(0, 0),
    InputParameters(0, 1),
    InputParameters(1, 0),
    InputParameters(1, 1),
    InputParameters(10, 10),
    InputParameters(32, 10),
    InputParameters(33, 10),
    InputParameters(64, 10),
    InputParameters(65, 10),
    InputParameters(100, 10),
    InputParameters(128, 10),
    InputParameters(1000, 10),
    InputParameters(1024, 10),
    InputParameters(1025, 10),
    InputParameters(2048, 10),
    InputParameters(2049, 10),

    InputParameters(0, 10),
    InputParameters(1, 10),
    InputParameters(10, 100),
    InputParameters(32, 100),
    InputParameters(33, 100),
    InputParameters(64, 100),
    InputParameters(65, 100),
    InputParameters(100, 1000),
    InputParameters(128, 1000),
    InputParameters(1000, 1000),
    InputParameters(1024, 2000),
    InputParameters(1025, 3000),
    InputParameters(2048, 4000),
    InputParameters(2049, 8192)
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
//INSTANTIATE_TEST_CASE_P(InstantiationName, InputInitializer, ::testing::Values(
//    InputParameters(131072, 10)
//));