#ifndef SKYLINES_QUERIES_ALGORITHMS_SINGLE_THREAD_BRUTE_FORCE_HPP
#define SKYLINES_QUERIES_ALGORITHMS_SINGLE_THREAD_BRUTE_FORCE_HPP

#include "queries/algorithms/algorithm.hpp"

namespace sl { namespace queries { namespace algorithms {
    class SingleThreadBruteForce : public Algorithm {
    public:
        SingleThreadBruteForce(
            const Data<data::WeightedPoint> &input_p, const Data<data::Point> &input_q) :
            Algorithm("SingleThreadBruteForce", input_p, input_q) {
        }

    protected:
        data::Statistics Run(NonConstData<data::WeightedPoint> *output, DistanceType distance_type) final;
        data::Statistics Compute(NonConstData<data::WeightedPoint> *output, DistanceType distance_type);

        template<class Comparator>
        data::Statistics SingleThreadBruteForce::_Compute(Comparator comparator_function, NonConstData<data::WeightedPoint> *output);
    };
}}}

#endif
