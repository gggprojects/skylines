#include "queries/algorithms/single_thread_brute_force.hpp"

namespace sl { namespace queries { namespace algorithms {
    void SingleThreadBruteForce::Run(NonConstData<data::WeightedPoint> *output) {
        if (!Init(output)) return;
        Compute(output);
    }

    void SingleThreadBruteForce::Compute(NonConstData<data::WeightedPoint> *output) {
        for (std::vector<data::WeightedPoint>::const_iterator skyline_candidate = input_p_.GetPoints().cbegin();
            skyline_candidate != input_p_.GetPoints().cend(); ++skyline_candidate) {
            std::vector<data::WeightedPoint>::const_iterator dominator_candidate = input_p_.GetPoints().cbegin();
            bool is_skyline = true;
            while (is_skyline && dominator_candidate != input_p_.GetPoints().cend()) {
                if (skyline_candidate != dominator_candidate) {
                    if (IsDominated(*skyline_candidate, *dominator_candidate, input_q_.GetPoints())) {
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
}}}

