#ifndef SKYLINES_QUERIES_ALGORITHMS_MULTI_THREAD_BRUTE_FORCE_HPP
#define SKYLINES_QUERIES_ALGORITHMS_MULTI_THREAD_BRUTE_FORCE_HPP

#include "queries/algorithms/algorithm.hpp"

namespace sl { namespace queries { namespace algorithms {
    class MultiThreadBruteForce : public Algorithm {
    public:
        MultiThreadBruteForce(
            error::ThreadErrors_ptr error_ptr, 
            const Data<data::WeightedPoint> &input_p, const Data<data::Point> &input_q) :
            Algorithm("MultiThreadBruteForce", error_ptr, input_p, input_q) {
        }

    protected:
        void ComputeSingleThreadBruteForce(
            std::vector<data::WeightedPoint>::const_iterator first_skyline_candidate,
            std::vector<data::WeightedPoint>::const_iterator last_skyline_candidate,
            NonConstData<data::WeightedPoint> *output);
        void Run(NonConstData<data::WeightedPoint> *output) final;
        void Compute(NonConstData<data::WeightedPoint> *output);
    };
}}}

#endif