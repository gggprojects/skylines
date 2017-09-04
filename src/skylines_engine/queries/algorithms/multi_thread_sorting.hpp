#ifndef SKYLINES_QUERIES_ALGORITHMS_MULTI_THREAD_SORTING_HPP
#define SKYLINES_QUERIES_ALGORITHMS_MULTI_THREAD_SORTING_HPP

#include "queries/algorithms/algorithm.hpp"

namespace sl { namespace queries { namespace algorithms {
    class MultiThreadSorting : public Algorithm {
    public:
        MultiThreadSorting(
            const Data<data::WeightedPoint> &input_p, const Data<data::Point> &input_q) :
            Algorithm("MultiThreadSorting", input_p, input_q) {
        }

    protected:
        void Run(NonConstData<data::WeightedPoint> *output) final;
        void Compute(NonConstData<data::WeightedPoint> *output);
    };
}}}

#endif
