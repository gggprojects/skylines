#ifndef SKYLINES_QUERIES_ALGORITHMS_GPU_BRUTE_FORCE_HPP
#define SKYLINES_QUERIES_ALGORITHMS_GPU_BRUTE_FORCE_HPP

#include "queries/algorithms/algorithm.hpp"
#include "gpu/gpu_devices.hpp"

namespace sl { namespace queries { namespace algorithms {
    class GPUBruteForce : public Algorithm {
    public:
        GPUBruteForce(
            const Data<data::WeightedPoint> &input_p, const Data<data::Point> &input_q) :
            Algorithm("GPUBruteForce", input_p, input_q) {
        }

    protected:
        data::Statistics Run(NonConstData<data::WeightedPoint> *output, DistanceType distance_type) final;
        data::Statistics Compute(NonConstData<data::WeightedPoint> *output, DistanceType distance_type);

        gpu::GPUDevices gpu_devices_;
    };
}}}

#endif
