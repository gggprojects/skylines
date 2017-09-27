#ifndef SKYLINES_QUERIES_ALGORITHMS_SINGLE_THREAD_SORTING_HPP
#define SKYLINES_QUERIES_ALGORITHMS_SINGLE_THREAD_SORTING_HPP

#include "queries/algorithms/algorithm.hpp"

namespace sl { namespace queries { namespace algorithms {
    class SingleThreadSorting : public Algorithm {
    public:
        SingleThreadSorting(
            const Data<data::WeightedPoint> &input_p, const Data<data::Point> &input_q) :
            Algorithm("SingleThreadSorting", input_p, input_q) {
        }

    protected:
        void Run(NonConstData<data::WeightedPoint> *output, DistanceType distance_type) final;
        void Compute(NonConstData<data::WeightedPoint> *output, DistanceType distance_type);

        template<class Comparator, class Sorter>
        void SingleThreadSorting::_Compute(
            Comparator comparator_function,
            Sorter sorter_funtion,
            NonConstData<data::WeightedPoint> *output);
    };
}}}

#endif
