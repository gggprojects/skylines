#include "queries/algorithms/multi_thread_sorting.hpp"

namespace sl { namespace queries { namespace algorithms {
    void MultiThreadSorting::Run(NonConstData<data::WeightedPoint> *output) {
        if (!Init(output)) return;
        Compute(output);
    }

    void ComputeSingleThreadSorting(
        const std::vector<data::WeightedPoint> &skyline_elements,
        std::vector<data::WeightedPoint> *skyline_candiates,
        const Data<data::Point> &input_q) {

        std::vector<data::WeightedPoint>::const_reverse_iterator skyline_candidate = skyline_candiates->crbegin();
        while (skyline_candidate != skyline_candiates->crend()) {
            std::vector<data::WeightedPoint>::const_reverse_iterator skyline_element = skyline_elements.crbegin();
            bool is_skyline = true;
            while (is_skyline && skyline_element != skyline_elements.crend()) {
                if (skyline_candidate->IsDominated(*skyline_element, input_q.GetPoints())) {
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

    void MultiThreadSorting::Compute(NonConstData<data::WeightedPoint> *output) {
        //copy P
        NonConstData<data::WeightedPoint> sorted_input;
        sorted_input = input_p_;

        // sort by the first point in Q
        const data::Point &first_q = input_q_.GetPoints()[0];
        concurrency::parallel_buffered_sort(sorted_input.Points().begin(), sorted_input.Points().end(), [&first_q](const data::WeightedPoint &a, const data::WeightedPoint &b) {
            return a.SquaredDistance(first_q) < b.SquaredDistance(first_q);
        });

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

        for (unsigned int num_threads = concurent_threads_supported; num_threads > 0; num_threads--) {

            std::vector<std::thread> workers(num_threads);
            for (unsigned int t = 0; t < num_threads; t++) {
                const NonConstData<data::WeightedPoint> &skyline_elements = sorted_splitted[t];
                NonConstData<data::WeightedPoint> &skyline_candidates = sorted_splitted[t + concurent_threads_supported - num_threads];
                ComputeSingleThreadSorting(skyline_elements.GetPoints(), &skyline_candidates.Points(), input_q_);

                workers[t] = std::thread([this, &sorted_splitted, &concurent_threads_supported, t, num_threads] {
                    const NonConstData<data::WeightedPoint> &skyline_elements = sorted_splitted[t];
                    NonConstData<data::WeightedPoint> &skyline_candidates = sorted_splitted[t + concurent_threads_supported - num_threads];
                    ComputeSingleThreadSorting(skyline_elements.GetPoints(), &skyline_candidates.Points(), input_q_);
                });
            }
            for (std::thread &t : workers) {
                t.join();
            }
        }

        //concat the result
        for (const NonConstData<data::WeightedPoint> &o : sorted_splitted) {
            output->Points().insert(output->Points().begin(), o.GetPoints().cbegin(), o.GetPoints().cend());
        }
    }
}}}

