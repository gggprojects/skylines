#include "queries/weighted.hpp"

namespace sl { namespace queries {
    WeightedQuery::WeightedQuery(error::ThreadErrors_ptr error_ptr) :
        SkylineElement("WeightedQuery", "info", error_ptr) {
    }

    int WeightedQuery::Run() {
        SL_LOG_DEBUG("Debug");
        SL_LOG_INFO("Info");
        SL_LOG_WARN("Warn");
        SL_LOG_ERROR("Error");
        SetSeverity("Debug");
        SL_LOG_DEBUG("Debug");
        return 0;
    }

    void WeightedQuery::Render() {
        input_data_.Render();
    }
}}
