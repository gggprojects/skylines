#ifndef SKYLINES_QUERIES_ALGORITHMS_GPU_BRUTE_FORCE_HPP
#define SKYLINES_QUERIES_ALGORITHMS_GPU_BRUTE_FORCE_HPP

#include "queries/algorithms/algorithm.hpp"

namespace sl { namespace queries { namespace algorithms {
    class GPUBruteForce : public Algorithm {
    public:
        GPUBruteForce(const Data<data::WeightedPoint> &input_p, const Data<data::Point> &input_q) :
            Algorithm(input_p, input_q) {
        }

    protected:
        void Run(NonConstData<data::WeightedPoint> *output) final;
        void Compute(NonConstData<data::WeightedPoint> *output);
    };
}}}

#endif
