
#include <fstream>

#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/rapidjson.h>

#include "queries/data_capable.hpp"

namespace sl { namespace queries {

    bool HasEnding(std::string const &fullString, std::string const &ending) {
        if (fullString.length() >= ending.length()) {
            return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
        } else {
            return false;
        }
    }

    std::string ReadAllFile(const std::string &filename) {
        std::ifstream t(filename);
        std::string json_str((std::istreambuf_iterator<char>(t)),
            std::istreambuf_iterator<char>());
        return std::move(json_str);
    }

    bool DataCapable::FromFile(const std::string &filename) {
        Clear();
        filename_loaded_ = filename;
        if (HasEnding(filename, "json")) {
            return ReadJsonFile(filename);
        }
        if (HasEnding(filename, "bin")) {
            return ReadBinaryFile(filename);
        }
        return false;
    }

    bool DataCapable::ReadJsonFile(const std::string &filename) {
        std::string json_str = ReadAllFile(filename);
        return FromJson(json_str);
    }

    bool DataCapable::FromJson(const std::string &json_str) {
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
            int weight = p[i]["w"].GetInt();
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

    bool DataCapable::ReadBinaryFile(const std::string &filename) {
        std::ifstream input(filename, std::ios::binary);
        if (!input.is_open()) return false;

        size_t input_p_num_elements = 0;
        input.read(reinterpret_cast<char*>(&input_p_num_elements), sizeof(size_t));
        input_p_.Resize(input_p_num_elements);
        input.read(reinterpret_cast<char*>(input_p_.GetDataPointer()), input_p_num_elements * sizeof(queries::data::WeightedPoint));

        size_t input_q_num_elements = 0;
        input.read(reinterpret_cast<char*>(&input_q_num_elements), sizeof(size_t));
        input_q_.Resize(input_q_num_elements);
        input.read(reinterpret_cast<char*>(input_q_.GetDataPointer()), input_q_num_elements * sizeof(queries::data::Point));

        input.close();
        return true;
    }

    void StringToFile(const std::string &filename, const std::string &json_string) {
        std::ofstream fout(filename, std::ios::out);
        fout.write(json_string.c_str(), json_string.size());
        fout.close();
    }

    bool DataCapable::ToFile(const std::string &filename) {
        if (HasEnding(filename, "json")) {
            return WriteJsonFile(filename);
        }
        if (HasEnding(filename, "bin")) {
            return WriteBinaryFile(filename);
        }
        return true;
    }

    bool DataCapable::WriteJsonFile(const std::string &filename) {
        std::string json_string = ToJson();
        StringToFile(filename, json_string);
        return true;
    }

    std::string DataCapable::ToJson() {
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

    bool DataCapable::WriteBinaryFile(const std::string &filename) {
        std::ofstream fout(filename, std::ios::out | std::ios::binary);

        size_t input_p_num_elements = input_p_.GetPoints().size();
        fout.write(reinterpret_cast<char *>(&input_p_num_elements), sizeof(size_t));
        fout.write(reinterpret_cast<char*>(input_p_.GetDataPointer()), input_p_.GetPoints().size() * sizeof(queries::data::WeightedPoint));

        size_t input_q_num_elements = input_q_.GetPoints().size();
        fout.write(reinterpret_cast<char *>(&input_q_num_elements), sizeof(size_t));
        fout.write(reinterpret_cast<char*>(input_q_.GetDataPointer()), input_q_.GetPoints().size() * sizeof(queries::data::Point));

        fout.close();
        return true;
    }

    size_t DataCapable::GetClosetsPointPosition(const data::Point &point) {
        float min_distance = static_cast<float>(std::pow(2, 2) + std::pow(2, 2));
        size_t closest_one = 0;
        for (size_t i = 0; i < input_p_.GetPoints().size(); i++) {
            float d = input_p_.GetPoints()[i].SquaredDistance(point);
            if (d < min_distance) {
                closest_one = i;
                min_distance = d;
            }
        }
        return closest_one;
    }

}}