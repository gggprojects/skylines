#ifndef SLYLINES_QUERIES_DATA_CAPABLE_HPP
#define SLYLINES_QUERIES_DATA_CAPABLE_HPP

#include "export_import.hpp"
#include "queries/input_data.hpp"
#include "queries/output_data.hpp"

namespace sl { namespace queries {

    class skylines_engine_DLL_EXPORTS DataCapable {
    public:
        const InputData& GetInputData() const { return input_data_; }
        const OutputData& GetOutputData() const { return output_data_; }

        void SetInputData(InputData &&input_data) {
            input_data_ = std::move(input_data);
        }

        void ClearInputData() {
            input_data_.Clear();
        }
    protected:
        InputData input_data_;
        OutputData output_data_;
    };
}}
#endif