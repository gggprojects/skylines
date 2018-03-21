#include <sstream>
#include <iostream>
#include <atomic>

#include "queries/algorithms/multi_thread_brute_force_discarding.hpp"

namespace sl { namespace queries { namespace algorithms {
    data::Statistics MultiThreadBruteForceDiscarding::Run(NonConstData<data::WeightedPoint> *output, DistanceType distance_type) {
        if (!Init(output)) return data::Statistics();
        return Compute(output, distance_type);
    }

    template<class Comparator>
    data::Statistics MultiThreadBruteForceDiscarding::ComputeSingleThreadBruteForceDiscarding(
        std::vector<data::WeightedPoint>::const_iterator first_skyline_candidate,
        std::vector<data::WeightedPoint>::const_iterator last_skyline_candidate,
        Comparator comparator_function,
        std::mutex &are_skylines_mutex,
        std::vector<bool>::iterator is_skyline) {

        data::Statistics stats_results;

        const sl::queries::data::Point *input_q = input_q_.GetPoints().data();
        const int q_size = static_cast<int>(input_q_.GetPoints().size());

        std::vector<data::WeightedPoint>::const_iterator last_p_element = input_p_.GetPoints().cend();
        size_t i = 0;
        for (std::vector<data::WeightedPoint>::const_iterator skyline_candidate = first_skyline_candidate; skyline_candidate != last_skyline_candidate; ++skyline_candidate, ++is_skyline) {
            if (*is_skyline) {
                std::vector<data::WeightedPoint>::const_iterator dominator_candidate = skyline_candidate + 1;
                std::vector<bool>::iterator dominator_candidate_is_skyline = is_skyline + 1;
                while (*is_skyline && dominator_candidate != last_p_element) {
                    if (*dominator_candidate_is_skyline) {
                        int dominator = Dominator(*skyline_candidate, *dominator_candidate, input_q, q_size, comparator_function);
                        if (dominator == 1) {
                            std::lock_guard<std::mutex> lock_(are_skylines_mutex);
                            *is_skyline = false;
                        } else if (dominator == 0) {
                            std::lock_guard<std::mutex> lock_(are_skylines_mutex);
                            *dominator_candidate_is_skyline = false;
                        }
                    } else {
                        if (IsDominated(*skyline_candidate, *dominator_candidate, input_q, q_size, comparator_function)) {
                            std::lock_guard<std::mutex> lock_(are_skylines_mutex);
                            *is_skyline = false;
                        }
                    }
                    stats_results.num_comparisions_++;
                    ++dominator_candidate;
                    ++dominator_candidate_is_skyline;
                }
            }
        }

        return stats_results;
    }

    template<class Comparator>
    data::Statistics MultiThreadBruteForceDiscarding::ComputeSkylines(Comparator comparator_function, std::vector<data::WeightedPoint> *skylines) {

        unsigned int concurent_threads_supported = std::thread::hardware_concurrency();
        size_t num_elements_p = input_p_.GetPoints().size();

        if (concurent_threads_supported > num_elements_p) {
            concurent_threads_supported = static_cast<unsigned int>(num_elements_p);
        }

        std::mutex stats_mutex;
        data::Statistics stats_results;

        std::mutex are_skylines_mutex;
        std::vector<bool> are_skylines(num_elements_p, true);

        size_t chunk = num_elements_p / concurent_threads_supported;
        std::vector<std::thread> workers;
        for (size_t i = 0; i < concurent_threads_supported; i++) {
            workers.emplace_back(
                std::thread([this, &chunk, i, comparator_function, &stats_mutex, &stats_results, &are_skylines_mutex, &are_skylines] {
                std::vector<data::WeightedPoint>::const_iterator first_skyline_candidate = input_p_.GetPoints().cbegin() + (i * chunk);
                std::vector<data::WeightedPoint>::const_iterator last_skyline_candidate = input_p_.GetPoints().cbegin() + ((i + 1) * chunk);
                data::Statistics partial_stats = ComputeSingleThreadBruteForceDiscarding(first_skyline_candidate, last_skyline_candidate, comparator_function, are_skylines_mutex, are_skylines.begin() + (i * chunk));

                std::lock_guard<std::mutex> lock(stats_mutex);
                stats_results.num_comparisions_ += partial_stats.num_comparisions_;
            }));
        }

        bool last_chunk = false;
        if (input_p_.GetPoints().cbegin() + (concurent_threads_supported * chunk) != input_p_.GetPoints().cend()) {
            last_chunk = true;
            workers.emplace_back(
                std::thread([this, &chunk, &concurent_threads_supported, comparator_function, &stats_mutex, &stats_results, &are_skylines_mutex, &are_skylines] {
                std::vector<data::WeightedPoint>::const_iterator first_skyline_candidate = input_p_.GetPoints().cbegin() + (concurent_threads_supported * chunk);
                std::vector<data::WeightedPoint>::const_iterator last_skyline_candidate = input_p_.GetPoints().end();
                data::Statistics partial_stats = ComputeSingleThreadBruteForceDiscarding(first_skyline_candidate, last_skyline_candidate, comparator_function, are_skylines_mutex, are_skylines.begin() + (concurent_threads_supported * chunk));

                std::lock_guard<std::mutex> lock(stats_mutex);
                stats_results.num_comparisions_ += partial_stats.num_comparisions_;
            }));
        }

        for (std::thread &t : workers) {
            t.join();
        }

        std::vector<data::WeightedPoint>::const_iterator it = input_p_.GetPoints().cbegin();
        for (bool is_skyline : are_skylines) {
            if (is_skyline) {
                skylines->emplace_back(*it);
            }
            ++it;
        }

        return stats_results;
    }

    template<class Comparator>
    data::Statistics MultiThreadBruteForceDiscarding::_Compute(Comparator comparator_function, NonConstData<data::WeightedPoint> *output) {
        std::vector<data::WeightedPoint> skylines;
        data::Statistics stats_results = ComputeSkylines(comparator_function, &skylines);
        ComputeTopK(skylines, output);
        stats_results.output_size_ = output->GetPoints().size();
        return stats_results;
    }

    data::Statistics MultiThreadBruteForceDiscarding::Compute(NonConstData<data::WeightedPoint> *output, DistanceType distance_type) {
        //std::cout << "Computing MTBFD\n";
        switch (distance_type) {
            case sl::queries::algorithms::DistanceType::Nearest:
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

