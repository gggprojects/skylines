#include <algorithm>
#include <ppl.h>

#include "queries/algorithms/multi_thread_sorting.hpp"

namespace sl { namespace queries { namespace algorithms {
    data::Statistics MultiThreadSorting::Run(NonConstData<data::WeightedPoint> *output, DistanceType distance_type) {
        if (!Init(output)) return data::Statistics();
        return Compute(output, distance_type);
    }

    template<class Comparator>
    data::Statistics ComputeSingleThreadSorting(
        std::vector<data::WeightedPoint> skyline_elements,
        std::vector<data::WeightedPoint> *skyline_candidates,
        const Data<data::Point> &input_q,
        Comparator comparator_function,
        const data::WeightedPoint &first_skyline) {

        data::Statistics stats_results;
        const sl::queries::data::Point *input_q_array = input_q.GetPoints().data();
        const int q_size = static_cast<int>(input_q.GetPoints().size());
        std::vector<data::WeightedPoint>::const_reverse_iterator skyline_candidate = skyline_candidates->crbegin();
        while (skyline_candidate != skyline_candidates->crend()) {

            if (IsDominated(*skyline_candidate, first_skyline, input_q_array, q_size, comparator_function)) {
                ++skyline_candidate;
                skyline_candidate = std::vector<data::WeightedPoint>::reverse_iterator(skyline_candidates->erase(skyline_candidate.base()));
                stats_results.num_comparisions_++;
            } else {
                std::vector<data::WeightedPoint>::const_reverse_iterator dominator_candidate = skyline_elements.crbegin();
                bool is_skyline = true;
                while (is_skyline && dominator_candidate != skyline_elements.crend()) {
                    if (IsDominated(*skyline_candidate, *dominator_candidate, input_q_array, q_size, comparator_function)) {
                        is_skyline = false;
                    }
                    dominator_candidate++;
                    stats_results.num_comparisions_++;
                }
                ++skyline_candidate;

                if (!is_skyline) {
                    skyline_candidate = std::vector<data::WeightedPoint>::reverse_iterator(skyline_candidates->erase(skyline_candidate.base()));
                }
            }
        }

        return stats_results;
    }

    template<class Comparator, class Sorter>
    data::Statistics MultiThreadSorting::_Compute(
        Comparator comparator_function,
        Sorter sorter_function,
        NonConstData<data::WeightedPoint> *output) {

        data::Statistics stats_results;

        //copy P
        NonConstData<data::WeightedPoint> sorted_input;
        sorted_input = input_p_;

        // sort by the first point in Q
        const data::Point &first_q = input_q_.GetPoints()[0];
        concurrency::parallel_sort(sorted_input.Points().begin(), sorted_input.Points().end(), sorter_function);

        unsigned int concurent_threads_supported = std::thread::hardware_concurrency();
        size_t num_elements_p = input_p_.GetPoints().size();

        if (concurent_threads_supported > num_elements_p) {
            concurent_threads_supported = static_cast<unsigned int>(num_elements_p);
        }

        size_t chunk = num_elements_p / concurent_threads_supported;

        //we split the vector
        std::vector<NonConstData<data::WeightedPoint>> sorted_splitted(concurent_threads_supported);
        for (size_t i = 0; i < concurent_threads_supported; i++) {
            sorted_splitted[i].Points().reserve(chunk);
            std::vector<data::WeightedPoint>::const_iterator first = sorted_input.GetPoints().cbegin() + (i * chunk);
            std::vector<data::WeightedPoint>::const_iterator last =
                i < concurent_threads_supported - 1 ?
                sorted_input.GetPoints().cbegin() + ((i + 1) * chunk) :
                sorted_input.GetPoints().cend();
            sorted_splitted[i].Points().assign(first, last);
        }

        data::WeightedPoint first_skyline = *sorted_input.GetPoints().cbegin();
        sorted_input.Clear(); //to release memory

        std::mutex stats_mutex;

        for (unsigned int num_threads = concurent_threads_supported; num_threads > 0; num_threads--) {

            std::vector<std::thread> workers(num_threads);
            for (unsigned int t = 0; t < num_threads; t++) {
                workers[t] = std::thread([this, &sorted_splitted, &concurent_threads_supported, t, num_threads, comparator_function, first_skyline, &stats_mutex, &stats_results] {
                    const NonConstData<data::WeightedPoint> &skyline_elements = sorted_splitted[t];
                    NonConstData<data::WeightedPoint> &skyline_candidates = sorted_splitted[t + concurent_threads_supported - num_threads];
                    data::Statistics partial_stats = ComputeSingleThreadSorting(skyline_elements.GetPoints(), &skyline_candidates.Points(), input_q_, comparator_function, first_skyline);
                    std::lock_guard<std::mutex> lock(stats_mutex);
                    stats_results.num_comparisions_ += partial_stats.num_comparisions_;
                });
            }
            for (std::thread &t : workers) {
                t.join();
            }
        }

        //concat the result
        std::vector<data::WeightedPoint> skylines;
        for (const NonConstData<data::WeightedPoint> &o : sorted_splitted) {
            skylines.insert(skylines.begin(), o.GetPoints().cbegin(), o.GetPoints().cend());
        }

        ComputeTopK(skylines, output);
        stats_results.output_size_ = output->GetPoints().size();

        return stats_results;
    }

    data::Statistics MultiThreadSorting::Compute(NonConstData<data::WeightedPoint> *output, DistanceType distance_type) {
        const data::Point &first_q = input_q_.GetPoints()[0];
        switch (distance_type) {
            case sl::queries::algorithms::DistanceType::Nearest:
                return _Compute(
                    [](const float a, const float b) -> bool { return a <= b; },
                    [&first_q](const data::WeightedPoint &a, const data::WeightedPoint &b) -> bool { return a.SquaredDistance(first_q) < b.SquaredDistance(first_q); },
                    output);
                break;
            case sl::queries::algorithms::DistanceType::Furthest:
                return _Compute(
                    [](const float a, const float b) -> bool { return a >= b; },
                    [&first_q](const data::WeightedPoint &a, const data::WeightedPoint &b) -> bool { return a.SquaredDistance(first_q) > b.SquaredDistance(first_q); },
                    output);
                break;
            default:
                return data::Statistics();
                break;
        }
    }
}}}

