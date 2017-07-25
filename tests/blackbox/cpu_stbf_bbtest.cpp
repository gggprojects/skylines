
#include <algorithm>

#include <gtest/gtest.h>

#include "queries/weighted.hpp"
#include "export_import.hpp"

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
    InputInitializer() : wq(sl::error::ThreadsErrors::Instanciate()) {
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

void CheckOuput(sl::queries::NonConstData<sl::queries::data::WeightedPoint> &a, sl::queries::NonConstData<sl::queries::data::WeightedPoint> &b, int line) {
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
    if (a.Points().size() == b.Points().size()) {
        EXPECT_TRUE(std::equal(a.Points().begin(), a.Points().end(), b.Points().begin())) << " Line: " << line;
    }
}

TEST_P(InputInitializer, TestOutputCorrectness) {

    wq.RunSingleThreadBruteForce();
    sl::queries::NonConstData<sl::queries::data::WeightedPoint> stbf_output;
    stbf_output = wq.GetOuput();

    wq.RunSingleThreadBruteForceDiscarting();
    sl::queries::NonConstData<sl::queries::data::WeightedPoint> stbfd_output;
    stbfd_output = wq.GetOuput();

    CheckOuput(stbf_output, stbfd_output, __LINE__);

    wq.RunSingleThreadSorting();
    sl::queries::NonConstData<sl::queries::data::WeightedPoint> sts_output;
    sts_output = wq.GetOuput();

    CheckOuput(stbfd_output, sts_output, __LINE__);

    wq.RunMultiThreadBruteForce();
    sl::queries::NonConstData<sl::queries::data::WeightedPoint> mtbf_output;
    mtbf_output = wq.GetOuput();

    CheckOuput(sts_output, mtbf_output, __LINE__);

    wq.RunMultiThreadSorting();
    sl::queries::NonConstData<sl::queries::data::WeightedPoint> mts_output;
    mts_output = wq.GetOuput();

    CheckOuput(stbf_output, mts_output, __LINE__);
}

INSTANTIATE_TEST_CASE_P(InstantiationName, InputInitializer, ::testing::Values(
    InputParameters(0, 0),
    InputParameters(0, 1),
    InputParameters(1, 0),
    InputParameters(1, 1),
    InputParameters(10, 10),
    InputParameters(1000, 10),
    InputParameters(10, 1000)
));
