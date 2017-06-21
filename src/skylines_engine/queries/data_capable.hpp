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

        bool FromJson(const std::string &json_str) {
            rapidjson::Document document;
            document.Parse(json_str.c_str());

            if (document.HasParseError()) {
                return false;
            }

            if (!document.IsObject()) return false;
            if (!document.HasMember("P")) return false;
            if (!document.HasMember("Q")) return false;
            if (!document["P"].IsArray()) return false;
            if (!document["Q"].IsArray()) return false;

            const rapidjson::Value& p = document["P"];
            for (rapidjson::SizeType i = 0; i < p.Size(); i++) {
                float x = p[i]["x"].GetFloat();
                float y = p[i]["y"].GetFloat();
                float weight = p[i]["w"].GetFloat();
                input_p_.Add(data::WeightedPoint(data::Point(x, y), weight));
            }

            const rapidjson::Value& q = document["Q"];
            for (rapidjson::SizeType i = 0; i < q.Size(); i++) {
                float x = q[i]["x"].GetFloat();
                float y = q[i]["y"].GetFloat();
                input_q_.Add(data::Point(x, y));

            }
            return true;
        }

        std::string ToJson() {
            rapidjson::Document document;

            // define the document as an object rather than an array
            document.SetObject();

            // must pass an allocator when the object may need to allocate memory
            rapidjson::Document::AllocatorType& allocator = document.GetAllocator();

            //P
            rapidjson::Value wp_array(rapidjson::kArrayType);
            for (const data::WeightedPoint &wp : input_p_.GetPoints()) {
                rapidjson::Value wp_object(rapidjson::kObjectType);
                wp_object.AddMember("x", wp.point_.x_, allocator);
                wp_object.AddMember("y", wp.point_.y_, allocator);
                wp_object.AddMember("w", wp.weight_, allocator);
                wp_array.PushBack(wp_object, allocator);
            }
            document.AddMember("P", wp_array, allocator);

            //Q
            rapidjson::Value p_array(rapidjson::kArrayType);
            for (const data::Point &p : input_q_.GetPoints()) {
                rapidjson::Value p_object(rapidjson::kObjectType);
                p_object.AddMember("x", p.x_, allocator);
                p_object.AddMember("y", p.y_, allocator);
                p_array.PushBack(p_object, allocator);
            }
            document.AddMember("Q", p_array, allocator);

            rapidjson::StringBuffer strbuf;
            rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
            document.Accept(writer);

            return std::move(std::string(strbuf.GetString()));
        }

    protected:
        Data<data::WeightedPoint> input_p_;
        Data<data::Point> input_q_;
        NonConstData<data::WeightedPoint> output_;
    };
}}
#endif