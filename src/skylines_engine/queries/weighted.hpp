#ifndef SKYLINES_QUERIES_WEIGHTED_HPP
#define SKYLINES_QUERIES_WEIGHTED_HPP

#include "export_import.hpp"
#include "common/skyline_element.hpp"

namespace sl { namespace queries {
    class skylines_engine_DLL_EXPORTS WeightedQuery : public common::SkylineElement {
    public:
        WeightedQuery(error::ThreadErrors_ptr error_ptr);
        int Run();
    private:

    };
}}
#endif // !SKYLINES_QUERIES_WEIGHTED_HPP
