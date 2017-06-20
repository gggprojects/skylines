#include "queries/weighted.hpp"

namespace sl { namespace queries {
    WeightedQuery::WeightedQuery(error::ThreadErrors_ptr error_ptr) :
        SkylineElement("WeightedQuery", "info", error_ptr) {
    }

    void WeightedQuery::InitRandom(size_t num_points) {
        input_p_.InitRandom(num_points);
        input_q_.InitRandom(num_points);
    }

    int WeightedQuery::Run() {
        return 0;
    }

    void WeightedQuery::Render() const {
        glColor3f(1, 0, 0);
        glPointSize(3);
        input_p_.Render();
        glColor3f(0, 1, 0);
        glPointSize(5);
        input_q_.Render();
        glColor3f(0, 0, 1);
        glPointSize(1);
        output_.Render();
    }
}}
