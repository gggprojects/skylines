#include "queries/algorithms/gpu_brute_force.hpp"
#include "gpu/gpu_memory.hpp"
#include "queries/data.hpp"

extern "C" void ComputeGPUSkyline(
    const std::vector<sl::queries::data::WeightedPoint> &input_p,
    const std::vector<sl::queries::data::Point> &input_q,
    std::vector<sl::queries::data::WeightedPoint> *output);

namespace sl { namespace queries { namespace algorithms {
    void GPUBruteForce::Run(NonConstData<data::WeightedPoint> *output) {
        if (!Init(output)) return;
        //todo: check constant memory max
        Compute(output);
    }

    void GPUBruteForce::Compute(NonConstData<data::WeightedPoint> *output) {
        ComputeGPUSkyline(input_p_.GetPoints(), input_q_.GetPoints(), &output->Points());
    }
}}}

