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

namespace sl { namespace queries { namespace algorithms {
    class Algorithm : public common::SkylineElement {
    public:
        Algorithm(
            const std::string &logger,
            const Data<data::WeightedPoint> &input_p, const Data<data::Point> &input_q);

        virtual void Run(NonConstData<data::WeightedPoint> *output) = 0;


        inline bool static IsDominated(const data::WeightedPoint &a, const data::WeightedPoint &b, const std::vector<data::Point> &q) {
            return IsDominated_impl(a, b, q.data(), static_cast<int>(q.size()));
        }

        inline int static Dominator(const data::WeightedPoint &a, const data::WeightedPoint &b, const std::vector<data::Point> &q) {
            return Dominator_impl(a, b, q.data(), static_cast<int>(q.size()));
        }

    protected:
        virtual bool Init(NonConstData<data::WeightedPoint> *output);
        bool IsEmpty();
        void ClearOutput(NonConstData<data::WeightedPoint> *output);

        const Data<data::WeightedPoint> &input_p_;
        const Data<data::Point> &input_q_;
    };
}}}

#endif
