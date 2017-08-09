#include "queries/algorithms/single_thread_sorting.hpp"

namespace sl { namespace queries { namespace algorithms {
    void SingleThreadSorting::Run(NonConstData<data::WeightedPoint> *output) {
        if (!Init(output)) return;
        Compute(output);
    }

    void SingleThreadSorting::Compute(NonConstData<data::WeightedPoint> *output) {
        //copy P
        NonConstData<data::WeightedPoint> sorted_input;
        sorted_input = input_p_;

        // sort by the first point in Q in parallel
        const data::Point &first_q = input_q_.GetPoints()[0];
        std::sort(sorted_input.Points().begin(), sorted_input.Points().end(), [&first_q](const data::WeightedPoint &a, const data::WeightedPoint &b) {
            return a.SquaredDistance(first_q) < b.SquaredDistance(first_q);
        });

        std::vector<data::WeightedPoint>::const_iterator first_block_element = sorted_input.GetPoints().cbegin();
        std::vector<data::WeightedPoint>::const_iterator last_block_element = sorted_input.GetPoints().cend();
        //the first element is skyline
        output->Add(sorted_input.GetPoints()[0]);

        for (std::vector<data::WeightedPoint>::const_iterator skyline_candidate = first_block_element + 1; skyline_candidate != last_block_element; ++skyline_candidate) {
            std::vector<data::WeightedPoint>::const_reverse_iterator skyline_element = output->GetPoints().crbegin();
            bool is_skyline = true;
            while (is_skyline && skyline_element != output->Points().rend()) {
                if (skyline_candidate->IsDominated(*skyline_element, input_q_.GetPoints())) {
                    is_skyline = false;
                }
                skyline_element++;
            }
            if (is_skyline) {
                output->Add(*skyline_candidate);
            }
        }
    }
}}}

