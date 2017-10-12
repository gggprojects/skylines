#include "queries/algorithms/multi_thread_sorting.hpp"
#include <iostream>

namespace sl { namespace queries { namespace algorithms {
    data::Statistics MultiThreadSorting::Run(NonConstData<data::WeightedPoint> *output, DistanceType distance_type) {
        if (!Init(output)) return data::Statistics();
        return Compute(output, distance_type);
    }

    template<class Comparator>
    void ComputeSingleThreadSorting(
        const std::vector<data::WeightedPoint> &skyline_elements,
        std::vector<data::WeightedPoint> *skyline_candiates,
        const Data<data::Point> &input_q,
        Comparator comparator_function,
        data::Statistics *statistics) {

        std::vector<data::WeightedPoint>::const_reverse_iterator skyline_candidate = skyline_candiates->crbegin();
        while (skyline_candidate != skyline_candiates->crend()) {
            std::vector<data::WeightedPoint>::const_reverse_iterator skyline_element = skyline_elements.crbegin();
            bool is_skyline = true;
            while (is_skyline && skyline_element != skyline_elements.crend()) {
                if (skyline_candidate->IsDominated(*skyline_element, input_q.GetPoints(), comparator_function, statistics)) {
                    is_skyline = false;
                }
                skyline_element++;
            }
            ++skyline_candidate;

            if (!is_skyline) {
                skyline_candidate = std::vector<data::WeightedPoint>::reverse_iterator(skyline_candiates->erase(skyline_candidate.base()));
            }
        }
    }

    template<class Comparator, class Sorter>
    data::Statistics MultiThreadSorting::_Compute(
        Comparator comparator_function,
        Sorter sorter_function,
        NonConstData<data::WeightedPoint> *output) {
        //copy P
        NonConstData<data::WeightedPoint> sorted_input(input_p_);

        // sort by the first point in Q
        std::sort(sorted_input.Points().begin(), sorted_input.Points().end(), sorter_function);

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

        sorted_input.Clear(); //to release memory

        data::Statistics statistics;
        std::vector<NonConstData<data::WeightedPoint>> sorted_splitted_output(concurent_threads_supported);
        for (unsigned int num_threads = concurent_threads_supported; num_threads > 0; num_threads--) {

            std::vector<std::thread> workers(num_threads);
            std::vector<data::Statistics> partial_statistics(num_threads);

            for (unsigned int t = 0; t < num_threads; t++) {
                data::Statistics *partial_statistic = &partial_statistics[t];

                workers[t] = std::thread([this, &sorted_splitted, &sorted_splitted_output, concurent_threads_supported, t, num_threads, comparator_function, partial_statistic] {
                    size_t sorted_skyline_elemenents_index = t;
                    size_t sorted_skyline_candidates_index = t + concurent_threads_supported - num_threads;

                    sorted_splitted_output[sorted_skyline_candidates_index].Points().assign(sorted_splitted[sorted_skyline_candidates_index].GetPoints().begin(), sorted_splitted[sorted_skyline_candidates_index].GetPoints().end());

                    const NonConstData<data::WeightedPoint> &skyline_elements = sorted_splitted[sorted_skyline_elemenents_index];
                    NonConstData<data::WeightedPoint> &skyline_candidates = sorted_splitted_output[sorted_skyline_candidates_index];
                    ComputeSingleThreadSorting(skyline_elements.GetPoints(), &skyline_candidates.Points(), input_q_, comparator_function, partial_statistic);
                });
            }
            for (std::thread &t : workers) {
                if(t.joinable())
                    t.join();
            }
            //reassign the results
            for (int t = concurent_threads_supported - num_threads; t < concurent_threads_supported; t++) {
                sorted_splitted[t].Move(std::move(sorted_splitted_output[t]));
            }
            //accum the statistics
            for (const data::Statistics & ds : partial_statistics) {
                statistics += ds;
            }
        }

        //concat the result
        for (const NonConstData<data::WeightedPoint> &o : sorted_splitted) {
            output->Points().insert(output->Points().begin(), o.GetPoints().cbegin(), o.GetPoints().cend());
        }

        statistics.output_size_ = output->Points().size();
        return statistics;
    }

    data::Statistics MultiThreadSorting::Compute(NonConstData<data::WeightedPoint> *output, DistanceType distance_type) {
        const data::Point &first_q = input_q_.GetPoints()[0];
        switch (distance_type) {
            case sl::queries::algorithms::DistanceType::Neartest:
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
                break;
        }
        return data::Statistics();
    }
}}}

