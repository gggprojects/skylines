#ifndef SKYLINES_QUERIES_WEIGHTED_HPP
#define SKYLINES_QUERIES_WEIGHTED_HPP

#include "export_import.hpp"
#include "common/skyline_element.hpp"
#include "common/irenderable.hpp"
#include "queries/InputData.hpp"
#include "queries/OutputData.hpp"

namespace sl { namespace queries {
    class skylines_engine_DLL_EXPORTS WeightedQuery :
        public common::SkylineElement,
        public common::IRenderable {
    public:
        WeightedQuery(error::ThreadErrors_ptr error_ptr);
        int Run();

        void Render() final;
    private:
        InputData input_data_;
        OutputData output_data_;
    };
}}
#endif // !SKYLINES_QUERIES_WEIGHTED_HPP

