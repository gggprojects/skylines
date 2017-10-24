#include "queries/algorithms/single_thread_sorting.hpp"

namespace sl { namespace queries { namespace algorithms {
    data::Statistics SingleThreadSorting::Run(NonConstData<data::WeightedPoint> *output, DistanceType distance_type) {
        if (!Init(output)) return data::Statistics();
        return Compute(output, distance_type);
    }

    template<class Comparator, class Sorter>
    data::Statistics SingleThreadSorting::_Compute(
        Comparator comparator_function,
        Sorter sorter_function,
        NonConstData<data::WeightedPoint> *output) {

        //copy P
        NonConstData<data::WeightedPoint> sorted_input;
        sorted_input = input_p_;

        //sorting depending on function
        std::sort(sorted_input.Points().begin(), sorted_input.Points().end(), sorter_function);

        //the first element is skyline
        output->Add(sorted_input.GetPoints()[0]);

        std::vector<data::WeightedPoint>::const_iterator first_block_element = sorted_input.GetPoints().cbegin();
        std::vector<data::WeightedPoint>::const_iterator last_block_element = sorted_input.GetPoints().cend();

        data::Statistics statistics;
        for (std::vector<data::WeightedPoint>::const_iterator skyline_candidate = first_block_element + 1; skyline_candidate != last_block_element; ++skyline_candidate) {
            std::vector<data::WeightedPoint>::const_iterator skyline_element = output->GetPoints().cbegin();
            bool is_skyline = true;
            while (is_skyline && skyline_element != output->Points().end()) {
                if (skyline_candidate->IsDominated(*skyline_element, input_q_.GetPoints(), comparator_function, &statistics)) {
                    is_skyline = false;
                }
                skyline_element++;
            }
            if (is_skyline) {
                output->Add(*skyline_candidate);
            }
        }

        statistics.output_size_ = output->Points().size();
        return statistics;
    }

    data::Statistics SingleThreadSorting::Compute(NonConstData<data::WeightedPoint> *output, DistanceType distance_type) {
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

