#include "queries/algorithms/gpu_brute_force.hpp"

namespace sl { namespace queries { namespace algorithms {
    void GPUBruteForce::Run(NonConstData<data::WeightedPoint> *output) {
        if (!Init(output)) return;
        Compute(output);
    }

    void GPUBruteForce::Compute(NonConstData<data::WeightedPoint> *output) {
    }
}}}

