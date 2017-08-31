#ifndef SKYLINES_QUERIES_ALGORITHMS_GPU_BRUTE_FORCE_HPP
#define SKYLINES_QUERIES_ALGORITHMS_GPU_BRUTE_FORCE_HPP

#include "queries/algorithms/algorithm.hpp"
#include "gpu/gpu_devices.hpp"

namespace sl { namespace queries { namespace algorithms {
    class GPUBruteForce : public Algorithm {
    public:
        GPUBruteForce(
            error::ThreadErrors_ptr error_ptr, 
            const Data<data::WeightedPoint> &input_p, const Data<data::Point> &input_q) :
            Algorithm("GPUBruteForce", error_ptr, input_p, input_q),
            gpu_devices_(error_ptr) {
        }

    protected:
        void Run(NonConstData<data::WeightedPoint> *output) final;
        void Compute(NonConstData<data::WeightedPoint> *output);

        gpu::GPUDevices gpu_devices_;
    };
}}}

#endif
