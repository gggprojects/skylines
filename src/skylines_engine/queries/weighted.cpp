#include <algorithm>

#include "queries/weighted.hpp"

namespace sl { namespace queries {
    WeightedQuery::WeightedQuery(error::ThreadErrors_ptr error_ptr) :
        SkylineElement("WeightedQuery", "info", error_ptr) {
    }

    void WeightedQuery::InitRandom(size_t num_points) {
        input_p_.InitRandom(num_points);
        input_q_.InitRandom(num_points);
    }

    void WeightedQuery::Render() const {
        glColor3f(1, 0, 0);
        glPointSize(3);
        input_p_.Render();
        glColor3f(0, 1, 0);
        glPointSize(5);
        input_q_.Render();
        glColor3f(0, 0, 1);
        glPointSize(10);
        output_.Render();
    }

    int WeightedQuery::Run() {
        //inititally all p are skylines
        output_ = input_p_;

        ComputeSkylineSingleThread();
        return 0;
    }

    void WeightedQuery::ComputeSkylineSingleThread() {
        std::vector<data::WeightedPoint>::iterator it_i = output_.Points().begin();
        if (it_i == output_.Points().end()) return;

        while(it_i + 1 != output_.Points().end()) { //not the last point
            std::vector<data::WeightedPoint>::iterator it_j = it_i + 1; //next point
            bool is_skyline = true;
            while (it_j != output_.Points().end()) {
                if (it_i->IsDominated(*it_j, input_q_.GetPoints())) {
                    /*
                    if it_i is dominated by it_j --> it_i will never be skyline. So we remove it from future comparisions
                    */
                    it_i = output_.Points().erase(it_i);
                    it_j = output_.Points().end(); // force exit
                    is_skyline = false;
                } else {
                    ++it_j;
                }
            }
            if (is_skyline) {
                ++it_i;
            }
        }
    }
}}
