#include "queries/algorithms/multi_thread_brute_force.hpp"

namespace sl { namespace queries { namespace algorithms {
    void MultiThreadBruteForce::Run(NonConstData<data::WeightedPoint> *output) {
        if (!Init(output)) return;
        Compute(output);
    }

    void MultiThreadBruteForce::ComputeSingleThreadBruteForce(
        std::vector<data::WeightedPoint>::const_iterator first_skyline_candidate,
        std::vector<data::WeightedPoint>::const_iterator last_skyline_candidate,
        NonConstData<data::WeightedPoint> *output) {

        for (std::vector<data::WeightedPoint>::const_iterator skyline_candidate = first_skyline_candidate; skyline_candidate != last_skyline_candidate; ++skyline_candidate) {
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
                output->SafetyAdd(*skyline_candidate);
            }
        }
    }

    void MultiThreadBruteForce::Compute(NonConstData<data::WeightedPoint> *output) {
        unsigned int concurent_threads_supported = std::thread::hardware_concurrency();
        size_t num_elements_p = input_p_.GetPoints().size();

        if (concurent_threads_supported > num_elements_p) {
            concurent_threads_supported = static_cast<unsigned int>(num_elements_p);
        }

        size_t chunk = num_elements_p / concurent_threads_supported;
        std::vector<std::thread> workers;
        for (size_t i = 0; i < concurent_threads_supported; i++) {
            workers.emplace_back(
                std::thread([this, &chunk, i, output] {
                std::vector<data::WeightedPoint>::const_iterator first_skyline_candidate = input_p_.GetPoints().cbegin() + (i * chunk);
                std::vector<data::WeightedPoint>::const_iterator last_skyline_candidate = input_p_.GetPoints().cbegin() + ((i + 1) * chunk);
                ComputeSingleThreadBruteForce(first_skyline_candidate, last_skyline_candidate, output);
            }));
        }

        if (input_p_.GetPoints().cbegin() + (concurent_threads_supported * chunk) != input_p_.GetPoints().cend()) {
            workers.emplace_back(
                std::thread([this, &chunk, &concurent_threads_supported, output] {
                std::vector<data::WeightedPoint>::const_iterator first_skyline_candidate = input_p_.GetPoints().cbegin() + (concurent_threads_supported * chunk);
                std::vector<data::WeightedPoint>::const_iterator last_skyline_candidate = input_p_.GetPoints().end();
                ComputeSingleThreadBruteForce(first_skyline_candidate, last_skyline_candidate, output);
            }));
        }

        for (std::thread &t : workers) {
            t.join();
        }
    }
}}}

