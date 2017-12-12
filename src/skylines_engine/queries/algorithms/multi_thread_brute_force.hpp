#ifndef SKYLINES_QUERIES_ALGORITHMS_MULTI_THREAD_BRUTE_FORCE_HPP
#define SKYLINES_QUERIES_ALGORITHMS_MULTI_THREAD_BRUTE_FORCE_HPP

#include <mutex>

#include "queries/algorithms/algorithm.hpp"

namespace sl { namespace queries { namespace algorithms {
    class MultiThreadBruteForce : public Algorithm {
    public:
        MultiThreadBruteForce(
            const Data<data::WeightedPoint> &input_p, const Data<data::Point> &input_q) :
            Algorithm("MultiThreadBruteForce", input_p, input_q) {
        }

    protected:

        data::Statistics Run(NonConstData<data::WeightedPoint> *output, DistanceType distance_type) final;
        data::Statistics Compute(NonConstData<data::WeightedPoint> *output, DistanceType distance_type);

        template<class Comparator>
        data::Statistics _Compute(Comparator comparator_function, NonConstData<data::WeightedPoint> *output);

        template<class Comparator>
        data::Statistics ComputeSkylines(Comparator comparator_function, std::vector<data::WeightedPoint> *skylines);

        template<class Comparator>
        data::Statistics ComputeSingleThreadBruteForce(
            std::vector<data::WeightedPoint>::const_iterator first_skyline_candidate,
            std::vector<data::WeightedPoint>::const_iterator last_skyline_candidate,
            Comparator comparator_function,
            std::vector<data::WeightedPoint> *skylines);
    private:
        std::mutex output_mutex_;
    };
}}}

#endif
