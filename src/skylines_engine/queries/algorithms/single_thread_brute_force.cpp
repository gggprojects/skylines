#include <iostream>

#include "queries/algorithms/single_thread_brute_force.hpp"

namespace sl { namespace queries { namespace algorithms {

    data::Statistics SingleThreadBruteForce::Run(NonConstData<data::WeightedPoint> *output, DistanceType distance_type) {
        if (!Init(output)) return data::Statistics();
        return Compute(output, distance_type);
    }

    template<class Comparator>
    data::Statistics SingleThreadBruteForce::ComputeSkylines(Comparator comparator_function, std::vector<data::WeightedPoint> *skylines) {
        data::Statistics stats_results;
        std::vector<data::WeightedPoint>::const_iterator skyline_candidate;

        const sl::queries::data::Point *input_q = input_q_.GetPoints().data();
        const int q_size = static_cast<int>(input_q_.GetPoints().size());

        std::vector<data::WeightedPoint>::const_iterator first_element = input_p_.GetPoints().cbegin();
        std::vector<data::WeightedPoint>::const_iterator last_element = input_p_.GetPoints().cend();

        for (skyline_candidate = first_element;
            skyline_candidate != last_element;
            ++skyline_candidate) {
            std::vector<data::WeightedPoint>::const_iterator dominator_candidate = first_element;
            bool is_skyline = true;
            while (is_skyline && dominator_candidate != last_element) {
                if (skyline_candidate != dominator_candidate) {
                    if (IsDominated(*skyline_candidate, *dominator_candidate, input_q, q_size, comparator_function)) {
                        is_skyline = false;
                    }
                    stats_results.num_comparisions_++;
                }
                ++dominator_candidate;
            }

            if (is_skyline) {
                skylines->emplace_back(*skyline_candidate);
            }
        }

        return stats_results;
    }

    template<class Comparator>
    data::Statistics SingleThreadBruteForce::_Compute(Comparator comparator_function, NonConstData<data::WeightedPoint> *output) {
        std::vector<data::WeightedPoint> skylines;
        data::Statistics stats_results = ComputeSkylines(comparator_function, &skylines);
        ComputeTopK(skylines, output);
        stats_results.output_size_ = output->GetPoints().size();
        return stats_results;
    }

    data::Statistics SingleThreadBruteForce::Compute(NonConstData<data::WeightedPoint> *output, DistanceType distance_type) {
        //std::cout << "Computing STBF\n";
        switch (distance_type) {
            case sl::queries::algorithms::DistanceType::Nearest:
                return _Compute([](const float a, const float b) -> bool { return a <= b; }, output);
                break;
            case sl::queries::algorithms::DistanceType::Furthest:
                return _Compute([](const float a, const float b) -> bool { return a >= b; }, output);
                break;
            default:
                break;
        }
        return data::Statistics();
    }
}}}

