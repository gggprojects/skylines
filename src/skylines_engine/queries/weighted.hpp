#ifndef SKYLINES_QUERIES_WEIGHTED_HPP
#define SKYLINES_QUERIES_WEIGHTED_HPP

#include "export_import.hpp"
#include "common/skyline_element.hpp"
#include "common/irenderable.hpp"
#include "queries/data_capable.hpp"
#include "queries/algorithms/single_thread_brute_force.hpp"
#include "queries/algorithms/single_thread_brute_force_discarting.hpp"
#include "queries/algorithms/single_thread_sorting.hpp"
#include "queries/algorithms/multi_thread_brute_force.hpp"
#include "queries/algorithms/multi_thread_sorting.hpp"
#include "queries/algorithms/gpu_brute_force.hpp"

namespace sl { namespace queries {
    class skylines_engine_DLL_EXPORTS WeightedQuery :
        public common::SkylineElement,
        public common::IRenderable,
        public DataCapable {
    public:
        enum AlgorithmType {
            SINGLE_THREAD_BRUTE_FORCE = 0,
            SINGLE_THREAD_BRUTE_FORCE_DISCARTING = 1,
            SINGLE_THREAD_SORTING = 2,
            MULTI_THREAD_BRUTE_FORCE = 3,
            MULTI_THREAD_SORTING = 4,
            GPU_BRUTE_FORCE = 5
        };

        WeightedQuery();

        void RunAlgorithm(AlgorithmType type, algorithms::DistanceType distance_type);

        void InitRandom(size_t num_points_p, size_t num_points_q);
        void Render() const final;

    private:
        std::vector<std::shared_ptr<algorithms::Algorithm>> algorithms_;
    };
}}
#endif // !SKYLINES_QUERIES_WEIGHTED_HPP

