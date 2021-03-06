#ifndef SKYLINES_QUERIES_ALGORITHMS_ALGORITHM_HPP
#define SKYLINES_QUERIES_ALGORITHMS_ALGORITHM_HPP

#include <algorithm>
#include <stack>
#include <ppl.h>
#include <future>
#include <functional>

#include "queries/data.hpp"
#include "common/skyline_element.hpp"
#include "queries/algorithms/algorithm.cuh"
#include "queries/algorithms/distance_type.hpp"

namespace sl { namespace queries { namespace algorithms {

    class Algorithm : public common::SkylineElement {
    public:

        Algorithm(
            const std::string &logger,
            const Data<data::WeightedPoint> &input_p,
            const Data<data::Point> &input_q);

        virtual data::Statistics Run(NonConstData<data::WeightedPoint> *output, DistanceType distance_type) = 0;

        void SetTopK(size_t top_k) {
            top_k_ = top_k;
        }

        void ComputeTopK(const std::vector<data::WeightedPoint> &all_skylines, NonConstData<data::WeightedPoint> *output);

    protected:
        std::pair<float, float> ComputeSkylineStatistics(const data::WeightedPoint &skyline);

        virtual bool Init(NonConstData<data::WeightedPoint> *output);
        bool IsEmpty();
        void ClearOutput(NonConstData<data::WeightedPoint> *output);

        const Data<data::WeightedPoint> &input_p_;
        const Data<data::Point> &input_q_;

        size_t top_k_;
    };
}}}

#endif
