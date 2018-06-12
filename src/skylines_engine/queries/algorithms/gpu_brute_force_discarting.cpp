#include <iostream>

#include "queries/algorithms/gpu_brute_force_discarting.hpp"
#include "gpu/gpu_memory.hpp"
#include "queries/data.hpp"

extern "C" void ComputeGPUSkylineDiscarting(
    const std::vector<sl::queries::data::WeightedPoint> &input_p,
    const std::vector<sl::queries::data::Point> &input_q,
    std::vector<sl::queries::data::WeightedPoint> *output,
    sl::queries::algorithms::DistanceType distance_type,
    size_t top_k,
    sl::queries::data::Statistics *stadistics_results);

namespace sl { namespace queries { namespace algorithms {
    data::Statistics GPUBruteForceDiscarting::Run(NonConstData<data::WeightedPoint> *output, DistanceType distance_type) {
        if (!Init(output)) return data::Statistics();
        //if (!CheckInputCorrectness(input_p_.GetPoints(), input_q_.GetPoints())) {
        //    SL_LOG_ERROR("Invalid imput");
        //}
        //todo: check constant memory max
        return Compute(output, distance_type);
    }

    data::Statistics GPUBruteForceDiscarting::Compute(NonConstData<data::WeightedPoint> *output, DistanceType distance_type) {
        data::Statistics results;
        ComputeGPUSkylineDiscarting(input_p_.GetPoints(), input_q_.GetPoints(), &output->Points(), distance_type, top_k_, &results);
        return results;
    }
}}}

