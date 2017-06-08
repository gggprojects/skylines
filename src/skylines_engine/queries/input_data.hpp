#ifndef SLYLINES_QUERIES_INPUT_DATA_HPP
#define SLYLINES_QUERIES_INPUT_DATA_HPP

#include "export_import.hpp"
#include "common/irenderable.hpp"
#include "queries/data/data_structures.hpp"

namespace sl { namespace queries {
    class skylines_engine_DLL_EXPORTS InputData : public common::IRenderable {
    public:
        InputData() {}

        InputData(std::vector<data::Point> &&points) {
            points_ = std::move(points);
        }

        InputData& operator=(InputData &&points) {
            points_ = std::move(points.points_);
            return *this;
        }

        void Render() final;

        void InitRandom(size_t num_points);

        const std::vector<data::Point> & GetPoints() const { return points_; }
        void Clear() { points_.clear(); }

    private:
        std::vector<data::Point> points_;
    };
}}
#endif // !SLYLINES_QUERIES_INPUT_DATA_HPP
