#include "queries/algorithms/single_thread_brute_force.hpp"

namespace sl { namespace queries { namespace algorithms {
    void SingleThreadBruteForce::Run(NonConstData<data::WeightedPoint> *output, DistanceType distance_type) {
        if (!Init(output)) return;
        Compute(output, distance_type);
    }

    template<class Comparator>
    void SingleThreadBruteForce::_Compute(Comparator comparator_function, NonConstData<data::WeightedPoint> *output) {
        for (std::vector<data::WeightedPoint>::const_iterator skyline_candidate = input_p_.GetPoints().cbegin();
            skyline_candidate != input_p_.GetPoints().cend(); ++skyline_candidate) {
            std::vector<data::WeightedPoint>::const_iterator dominator_candidate = input_p_.GetPoints().cbegin();
            bool is_skyline = true;
            while (is_skyline && dominator_candidate != input_p_.GetPoints().cend()) {
                if (skyline_candidate != dominator_candidate) {
                    if (IsDominated(*skyline_candidate, *dominator_candidate, input_q_.GetPoints(), comparator_function)) {
                        is_skyline = false;
                    }
                }
                ++dominator_candidate;
            }

            if (is_skyline) {
                output->Add(*skyline_candidate);
            }
        }
    }

    void SingleThreadBruteForce::Compute(NonConstData<data::WeightedPoint> *output, DistanceType distance_type) {
        switch (distance_type) {
            case sl::queries::algorithms::DistanceType::Neartest:
                _Compute([](const float a, const float b) -> bool { return a <= b; }, output);
                break;
            case sl::queries::algorithms::DistanceType::Furthest:
                _Compute([](const float a, const float b) -> bool { return a >= b; }, output);
                break;
            default:
                break;
        }
    }
}}}

