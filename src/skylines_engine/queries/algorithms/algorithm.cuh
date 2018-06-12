
#ifndef SKYLINES_QUERIES_ALGORITHMS_ALGORITHM_CUH
#define SKYLINES_QUERIES_ALGORITHMS_ALGORITHM_CUH

#include <cuda_runtime.h>
#include "queries/data/data_structures.hpp"
#include "queries/data/stadistics.hpp"

namespace sl { namespace queries { namespace algorithms {

    template<class Comparator>
    __host__ __device__ static inline bool IsDominated(
        const sl::queries::data::WeightedPoint &a,
        const sl::queries::data::WeightedPoint &b,
        const sl::queries::data::Point *input_q,
        const int q_size,
        Comparator comparator_function) {
        for (int i = 0; i < q_size; i++) {
            float a_distance = a.SquaredDistance(input_q[i]);
            float b_distance = b.SquaredDistance(input_q[i]);
            if (comparator_function(a_distance, b_distance)) {
                return false;
            }
        }
        return true;
    }

    template<class Comparator>
    __host__ __device__ static inline int Dominator(
        const sl::queries::data::WeightedPoint &a,
        const sl::queries::data::WeightedPoint &b,
        const sl::queries::data::Point *input_q,
        const int q_size, 
        Comparator comparator_function) {
        bool a_is_dominated_by_b = true;
        bool b_is_dominated_by_a = true;
        for (int i = 0; i < q_size; i++) {
            float a_distance = a.SquaredDistance(input_q[i]);
            float b_distance = b.SquaredDistance(input_q[i]);

            if (comparator_function(a_distance, b_distance)) {
                a_is_dominated_by_b = false;
                if (!b_is_dominated_by_a) return -1;
            }
            if (comparator_function(b_distance, a_distance)) {
                b_is_dominated_by_a = false;
                if (!a_is_dominated_by_b) return -1;
            }
        }
        //at this point one domains the other
        if (a_is_dominated_by_b) return 1;
        return 0;
    }

    struct PointStatistics {
        PointStatistics(const data::WeightedPoint &wp, const std::pair<float, float> &s) : wp_(wp), s_(s) {
        }

        friend bool operator<(const PointStatistics &a, const PointStatistics &b) {
            return a.s_.second > b.s_.second;
            // your code here
        }

        data::WeightedPoint wp_;
        std::pair<float, float> s_;
    };
}}}
#endif
