#include <set>

#include "queries/algorithms/algorithm.hpp"

namespace sl { namespace queries { namespace algorithms {
    Algorithm::Algorithm(
        const std::string &logger,
        const Data<data::WeightedPoint> &input_p,
        const Data<data::Point> &input_q) :
        common::SkylineElement("", "info"),
        input_p_(input_p), input_q_(input_q),
        top_k_(input_p_.GetPoints().size()) // by default all k
        {
    }

    bool Algorithm::Init(NonConstData<data::WeightedPoint> *output) {
        if (IsEmpty()) return false;
        ClearOutput(output);
        return true;
    }

    bool Algorithm::IsEmpty() {
        return input_p_.GetPoints().empty() || input_q_.GetPoints().empty();
    }

    void Algorithm::ClearOutput(NonConstData<data::WeightedPoint> *output) {
        output->Clear();
    }

    std::pair<float, float> Algorithm::ComputeSkylineStatistics(const data::WeightedPoint &skyline) {
        std::pair<float, float> min_max_statistics;
        min_max_statistics.first = std::numeric_limits<float>::max();
        min_max_statistics.second = std::numeric_limits<float>::min();
        for (const data::Point &q : input_q_.GetPoints()) {
            float distance = skyline.SquaredDistance(q);
            if (distance < min_max_statistics.first) {
                min_max_statistics.first = distance;
            }
            if (distance > min_max_statistics.second) {
                min_max_statistics.second = distance;
            }
        }
        return min_max_statistics;
    }

    void Algorithm::ComputeTopK(const std::vector<data::WeightedPoint> &all_skylines, NonConstData<data::WeightedPoint> *output) {

        std::set<PointStatistics> points;
        float max_distance_in_set = std::numeric_limits<float>::max();
        for (const data::WeightedPoint &skyline : all_skylines) {
            std::pair<float, float> min_max_distance = ComputeSkylineStatistics(skyline);
            if (points.size() < top_k_ || min_max_distance.second < max_distance_in_set) {
                points.insert(PointStatistics(skyline, min_max_distance));
                if(points.size() > top_k_)
                    points.erase(points.begin());
                max_distance_in_set = points.begin()->s_.second;
            }
        }

        for (const PointStatistics &ps : points) {
            output->Add(ps.wp_);
        }
    }
}}}

