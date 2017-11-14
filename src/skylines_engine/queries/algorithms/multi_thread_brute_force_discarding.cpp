#include <sstream>
#include <iostream>

#include "queries/algorithms/multi_thread_brute_force_discarding.hpp"

namespace sl { namespace queries { namespace algorithms {
    data::Statistics MultiThreadBruteForceDiscading::Run(NonConstData<data::WeightedPoint> *output, DistanceType distance_type) {
        if (!Init(output)) return data::Statistics();
        return Compute(output, distance_type);
    }

    template<class Comparator>
    void MultiThreadBruteForceDiscading::ComputeSingleThreadBruteForceDiscarding(
        std::vector<data::WeightedPoint>::const_iterator first_skyline_candidate,
        std::vector<data::WeightedPoint>::const_iterator last_skyline_candidate,
        NonConstData<data::WeightedPoint> *output,
        Comparator comparator_function,
        data::Statistics *statistics) {

        std::vector<bool> needs_to_be_checked(input_p_.GetPoints().size(), true);
        std::vector<bool>::iterator a_needs_to_be_checked = needs_to_be_checked.begin() + std::distance(input_p_.GetPoints().cbegin(), first_skyline_candidate);
        size_t i = 0;
        for (std::vector<data::WeightedPoint>::const_iterator skyline_candidate = first_skyline_candidate; skyline_candidate != last_skyline_candidate; ++skyline_candidate, ++a_needs_to_be_checked) {
            if (*a_needs_to_be_checked) {
                std::vector<bool>::iterator b_needs_to_be_checked = needs_to_be_checked.begin();
                std::vector<data::WeightedPoint>::const_iterator dominator_candidate = input_p_.GetPoints().cbegin();

                bool is_skyline = true;
                while (is_skyline && dominator_candidate != input_p_.GetPoints().cend()) {
                    if (skyline_candidate != dominator_candidate && *b_needs_to_be_checked) {
                        int dominator = Dominator(*skyline_candidate, *dominator_candidate, input_q_.GetPoints(), comparator_function, statistics);
                        if (dominator == 1) {
                            is_skyline = false;
                        } else if (dominator == 0) {
                            *b_needs_to_be_checked = false;
                        }
                    }
                    ++dominator_candidate;
                    ++b_needs_to_be_checked;
                }

                if (is_skyline) {
                    output->SafetyAdd(*skyline_candidate);
                }
            }
        }

        statistics->output_size_ = output->Points().size();
    }

    template<class Comparator>
    data::Statistics MultiThreadBruteForceDiscading::_Compute(Comparator comparator_function, NonConstData<data::WeightedPoint> *output) {
        data::Statistics statistics;
        unsigned int concurent_threads_supported = std::thread::hardware_concurrency();
        size_t num_elements_p = input_p_.GetPoints().size();

        if (concurent_threads_supported > num_elements_p) {
            concurent_threads_supported = static_cast<unsigned int>(num_elements_p);
        }

        size_t chunk = num_elements_p / concurent_threads_supported;
        std::vector<std::thread> workers;
        std::vector<data::Statistics> partial_statistics(concurent_threads_supported + 1);
        for (size_t i = 0; i < concurent_threads_supported; i++) {
            data::Statistics *ps = &partial_statistics[i];
            workers.emplace_back(
                std::thread([this, &chunk, i, output, comparator_function, ps] {
                std::vector<data::WeightedPoint>::const_iterator first_skyline_candidate = input_p_.GetPoints().cbegin() + (i * chunk);
                std::vector<data::WeightedPoint>::const_iterator last_skyline_candidate = input_p_.GetPoints().cbegin() + ((i + 1) * chunk);
                ComputeSingleThreadBruteForceDiscarding(first_skyline_candidate, last_skyline_candidate, output, comparator_function, ps);
            }));
        }

        bool last_chunk = false;
        if (input_p_.GetPoints().cbegin() + (concurent_threads_supported * chunk) != input_p_.GetPoints().cend()) {
            last_chunk = true;
            data::Statistics *ps = &partial_statistics[concurent_threads_supported];
            workers.emplace_back(
                std::thread([this, &chunk, &concurent_threads_supported, output, comparator_function, ps] {
                std::vector<data::WeightedPoint>::const_iterator first_skyline_candidate = input_p_.GetPoints().cbegin() + (concurent_threads_supported * chunk);
                std::vector<data::WeightedPoint>::const_iterator last_skyline_candidate = input_p_.GetPoints().end();
                ComputeSingleThreadBruteForceDiscarding(first_skyline_candidate, last_skyline_candidate, output, comparator_function, ps);
            }));
        }

        for (std::thread &t : workers) {
            t.join();
        }

        //accum the statistics
        for (size_t i = 0; i < partial_statistics.size(); i++) {
            if (i == partial_statistics.size() - 1 && last_chunk)
                statistics += partial_statistics[i];
        }

        statistics.output_size_ = output->Points().size();
        return statistics;
    }

    data::Statistics MultiThreadBruteForceDiscading::Compute(NonConstData<data::WeightedPoint> *output, DistanceType distance_type) {
        switch (distance_type) {
            case sl::queries::algorithms::DistanceType::Neartest:
                return _Compute(
                    [](const float a, const float b) -> bool { return a <= b; },
                    output);
                break;
            case sl::queries::algorithms::DistanceType::Furthest:
                return _Compute(
                    [](const float a, const float b) -> bool { return a >= b; },
                    output);
                break;
            default:
                break;
        }
        return data::Statistics();
    }
}}}

