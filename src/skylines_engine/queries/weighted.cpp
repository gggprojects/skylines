#include <algorithm>

#include "queries/weighted.hpp"

namespace sl { namespace queries {
    WeightedQuery::WeightedQuery(error::ThreadErrors_ptr error_ptr) :
        SkylineElement("WeightedQuery", "info", error_ptr) {
    }

    void WeightedQuery::InitRandom(size_t num_points_p, size_t num_points_q) {
        input_p_.InitRandom(num_points_p);
        input_q_.InitRandom(num_points_q);
    }

    void WeightedQuery::Render() const {
        glColor3f(1, 0, 0);
        glPointSize(3);
        input_p_.Render();

        glColor3f(0, 1, 0);
        glPointSize(3);
        input_q_.Render();

        glColor3f(0, 0, 1);
        glPointSize(6);
        output_.Render();
    }

    int WeightedQuery::RunSingleThreadSorting() {
        if (input_p_.GetPoints().empty()) return 0;
        if (input_q_.GetPoints().empty()) return 0;

        ComputeSkylineSingleThreadSorting();
        return 0;
    }

    int WeightedQuery::RunSingleFB() {
        return 0;
    }

    void WeightedQuery::ComputeSkylineSingleThreadSorting() {
        //copy P
        NonConstData<data::WeightedPoint> sorted_input;
        sorted_input = input_p_;

        // sort by the first point in Q
        const data::Point &first_q = input_q_.GetPoints()[0];
        std::sort(sorted_input.Points().begin(), sorted_input.Points().end(), [&first_q](const data::WeightedPoint &a, const data::WeightedPoint &b) {
            return a.SquaredDistance(first_q) < b.SquaredDistance(first_q);
        });

        //the first element is skyline
        output_.Add(sorted_input.GetPoints()[0]);

        for(std::vector<data::WeightedPoint>::const_iterator skyline_candidate = sorted_input.GetPoints().cbegin() + 1; skyline_candidate != sorted_input.GetPoints().cend(); ++skyline_candidate) {
            std::vector<data::WeightedPoint>::const_reverse_iterator skyline_element = output_.GetPoints().crbegin();
            bool is_skyline = true;
            while (is_skyline && skyline_element != output_.Points().rend()) {
                if (skyline_candidate->IsDominated(*skyline_element, input_q_.GetPoints())) {
                    is_skyline = false;
                } else {
                    skyline_element++;
                }
            }
            if (is_skyline) {
                output_.Add(*skyline_candidate);
            }
        }
    }
}}
