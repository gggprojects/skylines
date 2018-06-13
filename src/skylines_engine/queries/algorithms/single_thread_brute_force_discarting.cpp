#include <iostream>

#include "queries/algorithms/single_thread_brute_force_discarting.hpp"

namespace sl { namespace queries { namespace algorithms {
    data::Statistics SingleThreadBruteForceDiscarting::Run(NonConstData<data::WeightedPoint> *output, DistanceType distance_type) {
        if (!Init(output)) return data::Statistics();
        return Compute(output, distance_type);
    }

    template<class Comparator>
    data::Statistics SingleThreadBruteForceDiscarting::ComputeSkylines(Comparator comparator_function, std::vector<data::WeightedPoint> *skylines) {
        data::Statistics stats_results;

        const sl::queries::data::Point *input_q = input_q_.GetPoints().data();
        const int q_size = static_cast<int>(input_q_.GetPoints().size());

        std::vector<data::WeightedPoint>::const_iterator first_element = input_p_.GetPoints().cbegin();
        std::vector<data::WeightedPoint>::const_iterator last_element = input_p_.GetPoints().cend();

        std::vector<bool> needs_to_be_checked(input_p_.GetPoints().size(), true);
        std::vector<bool>::iterator skyline_candidate_needs_to_be_checked = needs_to_be_checked.begin();
        std::vector<data::WeightedPoint>::const_iterator skyline_candidate;
        for (skyline_candidate = first_element;
            skyline_candidate != last_element;
            ++skyline_candidate, ++skyline_candidate_needs_to_be_checked) {
            if (*skyline_candidate_needs_to_be_checked) {
                std::vector<bool>::iterator dominator_candidate_needs_to_be_checked = skyline_candidate_needs_to_be_checked + 1;
                std::vector<data::WeightedPoint>::const_iterator dominator_candidate = skyline_candidate + 1;

                bool is_skyline = true;
                while (is_skyline && dominator_candidate != last_element) {
                    if (*dominator_candidate_needs_to_be_checked) {
                        int dominator = Dominator(*skyline_candidate, *dominator_candidate, input_q, q_size, &stats_results.num_comparisions_, comparator_function);
                        if (dominator == 1) {
                            is_skyline = false;
                        } else if (dominator == 0) {
                            *dominator_candidate_needs_to_be_checked = false;
                        }
                    } else {
                        if (IsDominated(*skyline_candidate, *dominator_candidate, input_q, q_size, &stats_results.num_comparisions_, comparator_function)) {
                            is_skyline = false;
                        }
                    }
                    ++dominator_candidate;
                    ++dominator_candidate_needs_to_be_checked;
                }

                if (is_skyline) {
                    skylines->emplace_back(*skyline_candidate);
                }
            }
        }

        return stats_results;
    }

    template<class Comparator>
    data::Statistics SingleThreadBruteForceDiscarting::_Compute(Comparator comparator_function, NonConstData<data::WeightedPoint> *output) {
        std::vector<data::WeightedPoint> skylines;
        data::Statistics stats_results = ComputeSkylines(comparator_function, &skylines);
        ComputeTopK(skylines, output);
        stats_results.output_size_ = output->GetPoints().size();
        return stats_results;
    }

    data::Statistics SingleThreadBruteForceDiscarting::Compute(NonConstData<data::WeightedPoint> *output, DistanceType distance_type) {
        //std::cout << "Computing STBFD\n";
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

