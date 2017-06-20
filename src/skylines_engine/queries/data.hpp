#ifndef SLYLINES_QUERIES_INPUT_DATA_HPP
#define SLYLINES_QUERIES_INPUT_DATA_HPP

#include "export_import.hpp"
#include "common/irenderable.hpp"
#include "queries/data/data_structures.hpp"

namespace sl { namespace queries {
    template<class T>
    class skylines_engine_DLL_EXPORTS Data : public common::IRenderable {
    public:
        Data() {}

        Data(std::vector<T> &&points) {
            points_ = std::move(points);
        }

        Data& operator=(Data &&points) {
            points_ = std::move(points.points_);
            return *this;
        }

        void Render() const final {
            glBegin(GL_POINTS);
            for (const T &p : points_) {
                p.Render();
            }
            glEnd();
        }

        void InitRandom(size_t num_points) {
            data::UniformRealRandomGenerator r;
            points_.clear();
            points_.reserve(num_points);
            for (size_t i = 0; i < num_points; i++) {
                points_.emplace_back(T(r));
            }
        }

        const std::vector<T> & GetPoints() const { return points_; }
        void Clear() { points_.clear(); }
        void Add(T &&v) { points_.emplace_back(std::move(v)); }
    private:
        std::vector<T> points_;
    };
}}
#endif // !SLYLINES_QUERIES_INPUT_DATA_HPP
