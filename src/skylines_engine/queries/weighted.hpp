#ifndef SKYLINES_QUERIES_WEIGHTED_HPP
#define SKYLINES_QUERIES_WEIGHTED_HPP

#include "export_import.hpp"
#include "common/skyline_element.hpp"
#include "common/irenderable.hpp"
#include "queries/data_capable.hpp"

namespace sl { namespace queries {
    class skylines_engine_DLL_EXPORTS WeightedQuery :
        public common::SkylineElement,
        public common::IRenderable,
        public DataCapable {
    public:
        WeightedQuery(error::ThreadErrors_ptr error_ptr);

        void InitRandom(size_t num_points);
        int Run();
        void Render() const final;
    };
}}
#endif // !SKYLINES_QUERIES_WEIGHTED_HPP

