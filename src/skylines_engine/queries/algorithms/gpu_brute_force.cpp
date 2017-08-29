#include "queries/algorithms/gpu_brute_force.hpp"
#include "gpu/gpu_memory.hpp"
#include "queries/data.hpp"

#include <cuda_runtime.h>
#include "queries/algorithms/test.cu"

extern "C" void ComputeTest();

namespace sl { namespace queries { namespace algorithms {
    void GPUBruteForce::Run(NonConstData<data::WeightedPoint> *output) {
        if (!Init(output)) return;
        Compute(output);
    }

    void GPUBruteForce::Compute(NonConstData<data::WeightedPoint> *output) {

        unsigned int concurent_threads_supported = std::thread::hardware_concurrency();
        size_t num_elements_p = input_p_.GetPoints().size();

        if (concurent_threads_supported > num_elements_p) {
            concurent_threads_supported = static_cast<unsigned int>(num_elements_p);
        }

        std::vector<sl::gpu::GPUStream> streams(concurent_threads_supported);

        //upload to GPU memory
        sl::gpu::GPUStream stream;
        sl::gpu::GPUMemory<sl::queries::data::WeightedPoint> gpu_input_p(input_p_.GetPoints(), stream);

        int numBlocks = 1;
        dim3 threadsPerBlock(1, 1);
        ComputeTest();
    }
}}}

