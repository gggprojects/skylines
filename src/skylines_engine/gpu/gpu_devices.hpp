
#ifndef SKYLINES_GPU_DEVICES_HPP
#define SKYLINES_GPU_DEVICES_HPP

#include <vector>
#include "common/skyline_element.hpp"

namespace sl { namespace gpu {

    class GPUDevices : common::SkylineElement {
    public:
        GPUDevices(error::ThreadErrors_ptr error_ptr);
        void PrintGPUsInfo();
    private:
        static std::vector<int> devices_ids_;
    };

    std::vector<int> GPUDevices::devices_ids_;
}}

#endif // GPU_DEVICES_HPP
