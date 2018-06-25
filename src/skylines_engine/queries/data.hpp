#ifndef SLYLINES_QUERIES_INPUT_DATA_HPP
#define SLYLINES_QUERIES_INPUT_DATA_HPP

#include <mutex>

#include "export_import.hpp"
#include "common/irenderable.hpp"
#include "queries/data/data_structures.hpp"

namespace sl { namespace queries {

    class DataCapable;

    template<class T>
    class skylines_engine_DLL_EXPORTS Data {
        friend class DataCapable;
    public:
        Data() {}

        Data(const Data &other) {
            *this = other;
        }

        Data(Data &&other) {
            *this = other;
        }

        Data& operator=(const Data &other) {
            points_ = other.points_;
            return *this;
        }

        Data& operator=(Data &&other) {
            points_ = std::move(other.points_);
            return *this;
        }

        void InitRandom(size_t num_points,
            data::UniformRealRandomGenerator &rrg_x,
            data::UniformRealRandomGenerator &rrg_y,
            data::UniformIntRandomGenerator &irg) {
            points_.clear();
            points_.reserve(num_points);
            for (size_t i = 0; i < num_points; i++) {
                points_.emplace_back(T(rrg_x, rrg_y, irg));
            }
        }

        void InitRandom(size_t num_points,
            data::UniformRealRandomGenerator &rrg_x,
            data::UniformRealRandomGenerator &rrg_y) {
            points_.clear();
            points_.reserve(num_points);
            for (size_t i = 0; i < num_points; i++) {
                points_.emplace_back(T(rrg_x, rrg_y));
            }
        }

        const std::vector<T> & GetPoints() const { return points_; }
        void Clear() { points_.clear(); }
        void Add(T &&v) { points_.emplace_back(std::move(v)); }
        void Add(const T &v) { points_.push_back(v); }
        void SafetyAdd(const T &v) {
            std::lock_guard<std::mutex> lock(output_mutex_);
            points_.push_back(v);
        }

        T* GetDataPointer() { return &points_[0]; }

        void Resize(size_t new_size) { points_.resize(new_size); }
    protected:
        std::mutex output_mutex_;
        std::vector<T> points_;
    };

    template<class T>
    class skylines_engine_DLL_EXPORTS NonConstData : public Data<T> {
    public:
        NonConstData & operator=(const NonConstData &other) {
            points_ = other.points_;
            return *this;
        }
        NonConstData() {}

        NonConstData(const Data<T> &other) : Data<T>(other) {
        }

        void Move(NonConstData<T> &&other)  {
            points_ = std::move(other.points_);
        }

        std::vector<T> & Points() { return points_; }
    };
}}
#endif // !SLYLINES_QUERIES_INPUT_DATA_HPP
