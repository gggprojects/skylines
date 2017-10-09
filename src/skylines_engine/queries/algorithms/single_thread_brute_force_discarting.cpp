#include "queries/algorithms/single_thread_brute_force_discarting.hpp"

namespace sl { namespace queries { namespace algorithms {
    data::Statistics SingleThreadBruteForceDiscarting::Run(NonConstData<data::WeightedPoint> *output, DistanceType distance_type) {
        if (!Init(output)) return data::Statistics();
        return Compute(output, distance_type);
    }

    template<class Comparator>
    data::Statistics SingleThreadBruteForceDiscarting::_Compute(Comparator comparator_funciton, NonConstData<data::WeightedPoint> *output) {
        std::vector<bool> needs_to_be_checked(input_p_.GetPoints().size(), true);

        data::Statistics statistics;
        std::vector<bool>::iterator a_needs_to_be_checked = needs_to_be_checked.begin();
        for (std::vector<data::WeightedPoint>::const_iterator a = input_p_.GetPoints().cbegin(); a != input_p_.GetPoints().cend(); ++a, ++a_needs_to_be_checked) {
            if (*a_needs_to_be_checked) {
                std::vector<bool>::iterator b_needs_to_be_checked = needs_to_be_checked.begin();
                std::vector<data::WeightedPoint>::const_iterator b = input_p_.GetPoints().cbegin();

                bool is_skyline = true;
                while (is_skyline && b != input_p_.GetPoints().cend()) {
                    if (a != b && *b_needs_to_be_checked) {
                        int dominator = Dominator(*a, *b, input_q_.GetPoints(), comparator_funciton, &statistics);
                        if (dominator == 1) {
                            is_skyline = false;
                        } else if (dominator == 0) {
                            *b_needs_to_be_checked = false;
                        }
                    }
                    ++b;
                    ++b_needs_to_be_checked;
                }

                if (is_skyline) {
                    output->Add(*a);
                }
            }
        }

        statistics.output_size_ = output->Points().size();
        return statistics;
    }

    data::Statistics SingleThreadBruteForceDiscarting::Compute(NonConstData<data::WeightedPoint> *output, DistanceType distance_type) {
        switch (distance_type) {
            case sl::queries::algorithms::DistanceType::Neartest:
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

