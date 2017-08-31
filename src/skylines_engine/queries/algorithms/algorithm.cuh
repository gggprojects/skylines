
#ifndef SKYLINES_QUERIES_ALGORITHMS_ALGORITHM_CUH
#define SKYLINES_QUERIES_ALGORITHMS_ALGORITHM_CUH

#include <cuda_runtime.h>
#include "queries/data/data_structures.hpp"

__host__ __device__ static inline bool IsDominated_impl(
    const sl::queries::data::WeightedPoint &a,
    const sl::queries::data::WeightedPoint &b,
    const sl::queries::data::Point *input_q,
    const int q_size) {
    for (int i = 0; i < q_size; i++) {
        float a_distance = a.SquaredDistance(input_q[i]);
        float b_distance = b.SquaredDistance(input_q[i]);
        if (a_distance <= b_distance) {
            return false;
        }
    }
    return true;
}

#endif
