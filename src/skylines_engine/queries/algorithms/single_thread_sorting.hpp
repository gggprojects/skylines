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
        data::Statistics Run(NonConstData<data::WeightedPoint> *output, DistanceType distance_type) final;
        data::Statistics Compute(NonConstData<data::WeightedPoint> *output, DistanceType distance_type);

        template<class Comparator, class Sorter>
        data::Statistics ComputeSkylines(Comparator comparator_function, Sorter sorter_function, std::vector<data::WeightedPoint> *skylines);

        template<class Comparator, class Sorter>
        data::Statistics _Compute(
            Comparator comparator_function,
            Sorter sorter_funtion,
            NonConstData<data::WeightedPoint> *output);
    };
}}}

#endif
