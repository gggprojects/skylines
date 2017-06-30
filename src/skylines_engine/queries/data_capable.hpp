#ifndef SLYLINES_QUERIES_DATA_CAPABLE_HPP
#define SLYLINES_QUERIES_DATA_CAPABLE_HPP

#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/rapidjson.h>

#include "export_import.hpp"
#include "queries/data.hpp"

namespace sl { namespace queries {

    class skylines_engine_DLL_EXPORTS DataCapable {
    public:
        void Clear() {
            input_p_.Clear();
            input_q_.Clear();
            output_.Clear();
        }

        void ClearOutput() {
            output_.Clear();
        }

        bool FromFile(const std::string &filename);
        bool ToFile(const std::string &filename);

        size_t GetClosetsPointPosition(const data::Point &point);

        const Data<data::WeightedPoint> & GetInputP() const { return input_p_; }
        const Data<data::Point> & GetInputQ() const { return input_q_; }
        const NonConstData<data::WeightedPoint> & GetOuput() const { return output_; }

    private:
        bool ReadJsonFile(const std::string &filename);
        bool ReadBinaryFile(const std::string &filename);

        bool WriteJsonFile(const std::string &filename);
        bool WriteBinaryFile(const std::string &filename);

        bool FromJson(const std::string &json_str);
        std::string ToJson();

    protected:
        Data<data::WeightedPoint> input_p_;
        Data<data::Point> input_q_;
        NonConstData<data::WeightedPoint> output_;
    };
}}
#endif