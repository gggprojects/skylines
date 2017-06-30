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

        void InitRandom(size_t num_points_p, size_t num_points_q);
        void RunSingleThreadBruteForce();
        void RunSingleThreadBruteForceDiscarting();
        void RunSingleThreadSorting();
        void RunMultiThreadBruteForce();
        void RunGPUBruteForce();

        void Render() const final;

    private:
        bool IsEmpty();
        bool Init();
        void ComputeSkylineSingleThreadSorting();
        void ComputeSingleThreadBruteForce();
        void ComputeSingleThreadBruteForceDiscarting();
        void ComputeMultiThreadBruteForce();
        void ComputeSingleThreadBruteForce(
            std::vector<data::WeightedPoint>::const_iterator first_skyline_candidate,
            std::vector<data::WeightedPoint>::const_iterator last_skyline_candidate);
        void ComputeGPUBruteForce();
    };
}}
#endif // !SKYLINES_QUERIES_WEIGHTED_HPP

