#include "queries/algorithms/algorithm.hpp"

namespace sl { namespace queries { namespace algorithms {
    Algorithm::Algorithm(
        const std::string &logger,
        error::ThreadErrors_ptr error_ptr,
        const Data<data::WeightedPoint> &input_p, const Data<data::Point> &input_q) :
        common::SkylineElement("", "info", error_ptr),
        input_p_(input_p), input_q_(input_q) {
    }

    bool Algorithm::Init(NonConstData<data::WeightedPoint> *output) {
        if (IsEmpty()) return false;
        ClearOutput(output);
        return true;
    }

    bool Algorithm::IsEmpty() {
        return input_p_.GetPoints().empty() || input_q_.GetPoints().empty();
    }

    void Algorithm::ClearOutput(NonConstData<data::WeightedPoint> *output) {
        output->Clear();
    }
}}}

